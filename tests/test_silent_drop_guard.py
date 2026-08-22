"""Vapi accepts speech and never produces audio: detection, recovery, and the guard
that makes double-speaking impossible.

The defect, reproduced in Vapi's own server-side log on call
``01a026d8-ba00-744f-ae52-5de7e833cae6``: ``pipeline.sayQueuePush`` ->
``pipeline.botSpeechStarted`` ran 6.755 s, 5.759 s, DROPPED, 3.066 s, 0.487 s,
9.610 s, DROPPED, with ``assistant.voice.connectionOpened`` reporting a healthy TTS
websocket throughout and no error event anywhere. The adapter saw a 200 on every one
of those and could not tell them apart. The callee heard nothing.

Every test here fails against the previous adapter, most of them because the machinery
did not exist. The ones worth reading first are the two that constrain the FIX rather
than the feature:

- ``test_no_amount_of_waiting_turns_an_unconfirmed_delivery_into_a_drop`` -- because
  holding the model.url stream open for 20 s after a flush DELAYED the render by
  20.3 s rather than losing it (probe call 01a02723). A guard that treated "no audio
  yet" as a drop would have re-spoken on top of a live utterance.
- ``test_the_matcher_boundary_from_probe_call_01a02723`` -- the real margin between a
  spoken utterance and a dropped one, measured on the call that produced both.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

import httpx
import pytest
from starlette.testclient import TestClient

from fake_hermes import FakeScript, build_fake_hermes_transport
from test_ack_control import CONTROL_URL, make_control
from test_turns import _ScriptedHermes, delta, done, make_settings, make_state, tool_start
from vapi_hermes_voice.ack_journal import AckJournal
from vapi_hermes_voice.config import Settings
from vapi_hermes_voice.server import create_app
from vapi_hermes_voice.speech_feedback import (
    SpeechLedger,
    concat_assistant_text,
    content_tokens,
    spoken_coverage,
    user_text,
)
from vapi_hermes_voice.turns import register_delivery, stream_turn
from vapi_hermes_voice.vapi_events import parse_caller_speech_event, parse_server_message

# The default in config.py. Named here so a test that depends on the threshold says so.
THRESHOLD = 0.5


# --------------------------------------------------------------------------- matcher


def test_vapi_re_transcribes_rather_than_echoes_and_the_matcher_survives_it() -> None:
    """Measured on probe call 01a02727: the adapter delivered "ACK ONE PLEASE HOLD."
    and Vapi's own ``messages[]`` returned "a c k one please hold" -- case gone,
    punctuation gone, the acronym spelled out letter by letter. An equality test, or
    any matcher that needed the punctuation, would have called that a drop and
    re-spoken over a live utterance.
    """
    assert spoken_coverage("ACK ONE PLEASE HOLD.", "a c k one please hold") == 0.75
    assert spoken_coverage("ACK ONE PLEASE HOLD.", "a c k one please hold") >= THRESHOLD
    # And the same call's control-delivered answer, which came back verbatim.
    assert (
        spoken_coverage("ANSWER ALPHA IS FIFTY MILLIGRAMS.", "Answer alpha is fifty milligrams.")
        == 1.0
    )


def test_the_matcher_boundary_from_probe_call_01a02723() -> None:
    """The tightest real margin, from the call that produced a spoken utterance and a
    dropped one side by side.

    On 01a02723 the adapter-equivalent probe streamed "PROBE CHARLIE THREE.", Vapi
    accepted it (logged ``Voice input``, fired ``assistant.speechStarted``) and then
    cleared it before any audio played. The record of what WAS spoken on that call
    reads "probe alpha one Probe bravo two.".

    "PROBE CHARLIE THREE." scores 0.333 against it -- one shared content word out of
    three, and that one only because every phrase in the probe deliberately began with
    "PROBE". Real adapter text shares no such prefix, so 0.333 is a FLOOR on the
    margin rather than a typical case. Against a 0.5 threshold it is correctly
    condemned, with 0.167 of headroom, and the utterances that DID play score 1.0.

    This is the number the whole guard's safety rests on, in both directions: raise
    the threshold above ~0.75 and a re-transcribed acknowledgement gets condemned;
    drop it below ~0.34 and this genuine drop reads as a success and the guard
    silently disarms.
    """
    heard = "probe alpha one Probe bravo two."
    assert spoken_coverage("PROBE CHARLIE THREE.", heard) == pytest.approx(1 / 3)
    assert spoken_coverage("PROBE CHARLIE THREE.", heard) < THRESHOLD
    assert spoken_coverage("PROBE ALPHA ONE.", heard) == 1.0
    assert spoken_coverage("PROBE BRAVO TWO.", heard) == 1.0


def test_stopwords_cannot_manufacture_a_match() -> None:
    """Two unrelated sentences share only structure words. If those counted, an
    arbitrary answer would look "spoken" against an arbitrary record and the guard
    would never fire at all.
    """
    delivered = "The appointment is on the tenth of March at the clinic."
    unrelated = "It is a nice day and the weather will be with us."
    assert spoken_coverage(delivered, unrelated) < THRESHOLD
    assert "the" not in content_tokens(delivered)


def test_a_delivery_with_no_matchable_words_is_never_condemned() -> None:
    """Nothing to look for means no evidence, and the safe reading of no evidence is
    never "dropped".

    Only a delivery with no alphanumeric content at all reaches this: stopword removal
    falls back to the unfiltered tokens rather than emptying the set, so an ordinary
    short phrase ("It is the one.") is still matched on its content word and scores 0
    against an empty record -- which is correct, and is what makes the guard able to
    fire on short answers at all.
    """
    assert spoken_coverage("...", "") == 1.0
    assert spoken_coverage("It is the one.", "") == 0.0
    assert content_tokens("It is the.") == ("it", "is", "the")


def test_deliveries_that_vapi_merged_into_one_message_are_still_found() -> None:
    """Probe call 01a02727 returned ONE assistant message reading "a c k one please
    hold Answer alpha is fifty milligrams." for two separate deliveries. Comparing per
    message would have found the first and condemned the second.
    """
    merged = [
        ("user", "Hello."),
        ("assistant", "a c k one please hold Answer alpha is fifty milligrams."),
    ]
    heard = concat_assistant_text(merged)
    assert spoken_coverage("ACK ONE PLEASE HOLD.", heard) >= THRESHOLD
    assert spoken_coverage("ANSWER ALPHA IS FIFTY MILLIGRAMS.", heard) >= THRESHOLD
    assert user_text(merged) == "Hello."


# ---------------------------------------------------------------------------- ledger


def ledger_with(*, kind: str, text: str, age_s: float, **kw: Any) -> tuple[SpeechLedger, Any]:
    import time as _time

    led = SpeechLedger(**kw)
    delivery = led.register(kind=kind, text=text, now=_time.monotonic() - age_s)
    return led, delivery


def reconcile(led: SpeechLedger, heard: str, *, window_s: float = 15.0, advanced: bool = True):
    import time as _time

    return led.reconcile_history(
        heard=heard,
        threshold=THRESHOLD,
        settled_before=_time.monotonic() - window_s,
        history_advanced=advanced,
    )


def test_an_answer_absent_from_vapis_settled_record_is_a_confirmed_drop() -> None:
    """The core detection. Vapi took the answer (200 from Live Call Control), never
    rendered it, and the next turn's history proves it: the callee's words are there,
    ours are not.
    """
    led, delivery = ledger_with(kind="answer", text="Marvin is on five milligrams.", age_s=30.0)
    spoken, dropped = reconcile(led, "Okay let me check.")
    assert spoken == []
    assert dropped == [delivery]
    assert delivery.state == "dropped"
    assert delivery.evidence == "history_absent"


def test_no_amount_of_waiting_turns_an_unconfirmed_delivery_into_a_drop() -> None:
    """ "No audio yet" is evidence of UNKNOWN, not of loss.

    Measured on probe call 01a02723: a flushed chunk on a stream held open for 20 s
    rendered 20.3 s late -- the instant the response ended -- rather than being lost. A
    timer-based verdict would have re-spoken it on top of itself. So nothing in this
    ledger expires: only an inbound record that positively lacks the delivery can
    condemn it, and until one arrives the delivery stays exactly where it is, however
    old.
    """
    led, delivery = ledger_with(kind="answer", text="Marvin is on five milligrams.", age_s=600.0)
    assert delivery.state == "unconfirmed"
    assert led.outstanding() == [delivery]
    # No API exists to expire it, and the only claim path refuses an unconfirmed one.
    assert led.claim_replay(max_age_seconds=10_000.0) is None
    assert delivery.state == "unconfirmed"


def test_a_delivery_younger_than_the_confirm_window_is_never_condemned() -> None:
    """The late-render interlock. An utterance still sitting in
    ``pipeline.sayQueuePush`` (up to 9.610 s measured on call 01a026d8) is absent from
    the record and about to be spoken; condemning it is how a guard against silence
    becomes a cause of double-speaking.
    """
    led, delivery = ledger_with(kind="answer", text="Marvin is on five milligrams.", age_s=2.0)
    spoken, dropped = reconcile(led, "Okay let me check.", window_s=15.0)
    assert (spoken, dropped) == ([], [])
    assert delivery.state == "unconfirmed"


def test_absence_proves_nothing_without_the_liveness_interlock() -> None:
    """A record that is empty, truncated, or shaped differently than assumed would
    otherwise read as "nothing we ever said was spoken" and condemn every delivery on
    the call at once. The proof required is that the record covers the turn the
    delivery was made on.
    """
    led, delivery = ledger_with(kind="answer", text="Marvin is on five milligrams.", age_s=30.0)
    spoken, dropped = reconcile(led, "", advanced=False)
    assert (spoken, dropped) == ([], [])
    assert delivery.state == "unconfirmed"


def test_confirmed_spoken_is_absorbing() -> None:
    """Once anything vouches for a delivery, no later absence may un-vouch for it.
    Otherwise a record that ages out of Vapi's own history would resurrect a
    already-spoken answer and say it a second time.
    """
    led, delivery = ledger_with(kind="answer", text="Marvin is on five milligrams.", age_s=30.0)
    led.confirm_any_started(before=float("inf"), evidence="speech-update")
    assert delivery.state == "spoken"
    spoken, dropped = reconcile(led, "totally unrelated text")
    assert (spoken, dropped) == ([], [])
    assert delivery.state == "spoken"
    assert delivery.evidence == "speech-update"


def test_a_speech_update_cannot_vouch_for_a_later_delivery() -> None:
    """``speech-update`` says "assistant audio started" without saying whose text, so
    it may only credit deliveries that already existed when it fired.
    """
    import time as _time

    led = SpeechLedger()
    now = _time.monotonic()
    earlier = led.register(kind="answer", text="first", now=now - 5)
    later = led.register(kind="answer", text="second", now=now)
    settled = led.confirm_any_started(before=now - 1, evidence="speech-update")
    assert settled == [earlier]
    assert later.state == "unconfirmed"


# -------------------------------------------------------------------------- recovery


def test_an_acknowledgement_is_never_replayed() -> None:
    """An acknowledgement is a promise about the immediate future ("give me a
    second"), so a stale one is not merely redundant, it is FALSE -- re-speaking it
    after the answer has landed would be worse than the silence it was meant to cover.
    An answer is still true whenever it arrives, which is why one is replayable and
    the other is not.
    """
    led, delivery = ledger_with(kind="ack", text="Okay, one moment.", age_s=30.0)
    _, dropped = reconcile(led, "unrelated")
    assert dropped == [delivery]  # detected and journalled
    assert led.claim_replay(max_age_seconds=45.0) is None  # and never re-spoken


def test_a_dropped_answer_is_claimed_for_replay_exactly_once() -> None:
    """The at-most-once guarantee. Two callers, one claim -- and a claim is spent even
    if the replay that follows it fails, because retrying a replay is how one delivery
    becomes three utterances.
    """
    led, delivery = ledger_with(kind="answer", text="Marvin is on five milligrams.", age_s=20.0)
    reconcile(led, "unrelated")
    assert led.claim_replay(max_age_seconds=45.0) is delivery
    assert led.claim_replay(max_age_seconds=45.0) is None
    assert delivery.replay_issued is True


def test_a_replayed_answer_can_never_be_replayed_again() -> None:
    """The replay is itself a delivery and is confirmed like any other -- but the
    ORIGINAL can never return to a replayable state, so a second drop of the same text
    cannot compound.
    """
    led, delivery = ledger_with(kind="answer", text="Marvin is on five milligrams.", age_s=20.0)
    reconcile(led, "unrelated")
    led.claim_replay(max_age_seconds=45.0)
    led.mark_replayed(delivery)
    assert delivery.state == "replayed"
    spoken, dropped = reconcile(led, "unrelated")
    assert (spoken, dropped) == ([], [])
    assert led.claim_replay(max_age_seconds=45.0) is None


def test_the_per_call_replay_cap_bounds_a_systematic_false_positive() -> None:
    """The backstop for the one failure this design cannot rule out by reasoning: if
    Vapi ever stopped committing assistant messages, every delivery would look absent.
    Two utterances of damage, not a monologue.
    """
    import time as _time

    led = SpeechLedger(max_replays=2)
    old = _time.monotonic() - 20
    for i in range(5):
        led.register(kind="answer", text=f"answer number {i} about prednisolone", now=old)
    reconcile(led, "unrelated")
    claimed = [led.claim_replay(max_age_seconds=45.0) for _ in range(5)]
    assert sum(c is not None for c in claimed) == 2


def test_a_stale_dropped_answer_is_recorded_but_not_re_spoken() -> None:
    """Freshness is a limit on the recovery, not a trigger for it: the callee has moved
    on, and a late answer to a superseded question is its own defect.
    """
    led, delivery = ledger_with(kind="answer", text="Marvin is on five milligrams.", age_s=300.0)
    _, dropped = reconcile(led, "unrelated")
    assert dropped == [delivery]
    assert led.claim_replay(max_age_seconds=45.0) is None


def test_the_ledger_counts_what_it_forgets() -> None:
    """Section 6: never report a completeness signal you cannot vouch for."""
    led = SpeechLedger(max_deliveries=2)
    for i in range(5):
        led.register(kind="answer", text=f"answer {i}")
    assert len(led.deliveries) == 2
    assert led.forgotten == 3


# --------------------------------------------------------------------------- journal


def test_a_delivery_is_journalled_unconfirmed_before_any_evidence_exists() -> None:
    """Written at the moment of delivery, not at the moment of confirmation. A
    delivery whose outcome is unknown has to be distinguishable from one that never
    happened (docs/integration-contracts.md section 6).
    """
    settings = make_settings()
    state = make_state(settings)
    journal = AckJournal(max_calls=8, max_entries_per_call=8, ttl_seconds=60.0)
    register_delivery(state, journal, kind="answer", text="Marvin is on five milligrams.")
    snapshot = journal.snapshot(state.call_ref)
    assert snapshot is not None
    (entry,) = snapshot.speech_outcomes
    assert entry.outcome == "unconfirmed"
    assert entry.evidence == ""
    assert entry.settled_after_ms is None
    # No answer text ever enters the record; an acknowledgement's does, because it is
    # an operator-configured pool phrase this journal already publishes.
    assert entry.text is None


def test_the_journal_record_is_refined_in_place_by_the_verdict() -> None:
    settings = make_settings()
    state = make_state(settings)
    journal = AckJournal(max_calls=8, max_entries_per_call=8, ttl_seconds=60.0)
    delivery = register_delivery(state, journal, kind="ack", text="Okay, one moment.")
    delivery.at_monotonic_s -= 30.0
    reconcile(state.speech, "some other thing entirely")
    snapshot = journal.snapshot(state.call_ref)
    assert snapshot is not None
    (entry,) = snapshot.speech_outcomes
    assert entry.outcome == "confirmed_dropped"
    assert entry.evidence == "history_absent"
    assert entry.settled_after_ms is not None and entry.settled_after_ms >= 29_000
    assert entry.text == "Okay, one moment."


def test_a_confirmed_spoken_delivery_says_so_in_the_journal() -> None:
    settings = make_settings()
    state = make_state(settings)
    journal = AckJournal(max_calls=8, max_entries_per_call=8, ttl_seconds=60.0)
    delivery = register_delivery(state, journal, kind="ack", text="Okay, one moment.")
    delivery.at_monotonic_s -= 30.0
    reconcile(state.speech, "okay one moment")
    snapshot = journal.snapshot(state.call_ref)
    assert snapshot is not None
    (entry,) = snapshot.speech_outcomes
    assert entry.outcome == "confirmed_spoken"
    assert entry.evidence == "history"


# ------------------------------------------------------------------- turn plumbing


def test_a_streamed_acknowledgement_is_entered_in_the_ledger_when_it_goes_out() -> None:
    """Vapi echoing the chunk back as ``voice-input`` within ~1 ms is not evidence it
    was spoken, so the question is opened here and settled later.
    """
    settings = make_settings()
    state = make_state(settings)
    control, _ = make_control(lambda _r: httpx.Response(200, json={}))

    async def drive() -> None:
        reaping: set[asyncio.Task[Any]] = set()
        agen = stream_turn(
            settings=settings,
            hermes=_ScriptedHermes([(0.2, tool_start()), (0.05, delta("Five mg.")), (0.0, done())]),
            control=control,
            control_url=CONTROL_URL,
            state=state,
            instructions="i",
            history=[],
            user_input="u",
            reaping=reaping,
        )
        async for _ in agen:
            pass
        for task in list(reaping):
            with contextlib.suppress(Exception):
                await task
        await control.aclose()

    asyncio.run(drive())
    kinds = [d.kind for d in state.speech.deliveries]
    assert "ack" in kinds, "the acknowledgement must be entered in the ledger"
    assert all(d.state == "unconfirmed" for d in state.speech.deliveries)


def test_an_answer_vapi_accepted_is_entered_in_the_ledger() -> None:
    """A 2xx from Live Call Control means "Vapi took it", never "the callee heard it".
    Call 01a026d8 proves the difference: two of seven accepted pushes were DROPPED.
    """
    settings = make_settings()
    state = make_state(settings)
    control, requests = make_control(lambda _r: httpx.Response(200, json={}))

    async def drive() -> None:
        reaping: set[asyncio.Task[Any]] = set()
        agen = stream_turn(
            settings=settings,
            hermes=_ScriptedHermes(
                [(0.2, tool_start()), (0.05, delta("Five milligrams twice daily.")), (0.0, done())]
            ),
            control=control,
            control_url=CONTROL_URL,
            state=state,
            instructions="i",
            history=[],
            user_input="u",
            reaping=reaping,
        )
        async for _ in agen:
            pass
        for task in list(reaping):
            with contextlib.suppress(Exception):
                await task
        await control.aclose()

    asyncio.run(drive())
    assert requests, "the answer must have been delivered through control"
    answers = [d for d in state.speech.deliveries if d.kind == "answer"]
    assert [d.text for d in answers] == ["Five milligrams twice daily."]
    assert answers[0].state == "unconfirmed"


# ----------------------------------------------------------------- server end to end

API_KEY = "adapter-key-0123456789"
AUTH = {"Authorization": f"Bearer {API_KEY}"}
WEBHOOK_SECRET = "vapi-server-secret-0123456789"
CALL_ID = "01a02723-3877-7dd8-a7da-59e56c42a744"


def _state_for(client: TestClient) -> Any:
    return _state_for_id(client, CALL_ID)


def _state_for_id(client: TestClient, call_id: str) -> Any:
    return client.app.state.calls.peek(call_id)  # type: ignore[attr-defined]


def app_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "hermes_base_url": "http://fake-hermes.invalid",
        "hermes_api_key": "k",
        "adapter_api_key": API_KEY,
        "warmup_on_start": False,
        "filler_after_seconds": 5.0,
        "_env_file": None,
    }
    values.update(overrides)
    return Settings(**values)


def body(*, messages: list[dict[str, Any]], control: bool = True) -> dict[str, Any]:
    call: dict[str, Any] = {"id": CALL_ID, "type": "inboundPhoneCall"}
    if control:
        call["monitor"] = {"controlUrl": CONTROL_URL}
    return {
        "model": "hermes",
        "stream": True,
        "messages": messages,
        "call": call,
        "metadata": {},
        "customer": {"number": "+15551230000"},
    }


@contextlib.contextmanager
def app_with(**overrides: Any):
    """A running app whose Vapi control endpoint is captured rather than called."""
    transport, _ = build_fake_hermes_transport(FakeScript(deltas=["Five mg."]))
    says: list[dict[str, Any]] = []

    def control_handler(request: httpx.Request) -> httpx.Response:
        says.append(json.loads(request.content))
        return httpx.Response(200, json={})

    settings = app_settings(**overrides)
    app = create_app(
        settings,
        hermes_transport=transport,
        vapi_control_transport=httpx.MockTransport(control_handler),
    )
    with TestClient(app) as client:
        yield client, settings, says


def seed_dropped_answer(client: TestClient, settings: Settings, text: str) -> Any:
    """Put one confirmed-droppable answer on the call, aged past the confirm window."""
    # Drive one real turn so the registry holds state for CALL_ID.
    client.post(
        "/chat/completions",
        headers=AUTH,
        json=body(messages=[{"role": "user", "content": "What did the vet prescribe?"}]),
    )
    state = _state_for(client)
    state.speech.deliveries.clear()
    delivery = state.speech.register(kind="answer", text=text)
    delivery.at_monotonic_s -= settings.speech_confirm_window_seconds + 10.0
    return state, delivery


def test_a_dropped_answer_is_re_spoken_on_the_next_turn() -> None:
    """End to end, with no assistant configuration at all: the next turn's own request
    body carries Vapi's account of what was spoken, the missing answer is identified
    from it, and it goes back out through Live Call Control.
    """
    with app_with() as (client, settings, says):
        state, delivery = seed_dropped_answer(
            client, settings, "Marvin is on five milligrams of prednisolone."
        )
        says.clear()
        client.post(
            "/chat/completions",
            headers=AUTH,
            json=body(
                messages=[
                    {"role": "user", "content": "What did the vet prescribe?"},
                    {"role": "user", "content": "Hello? Are you there?"},
                ]
            ),
        )
        replays = [s for s in says if "prednisolone" in s.get("content", "")]
        assert replays, f"the dropped answer was never re-spoken (control saw {says})"
        assert delivery.state == "replayed"


def test_an_answer_vapis_record_accounts_for_is_never_re_spoken() -> None:
    """The other half, and the more important one: presence in the record is a veto."""
    with app_with() as (client, settings, says):
        state, delivery = seed_dropped_answer(
            client, settings, "Marvin is on five milligrams of prednisolone."
        )
        says.clear()
        client.post(
            "/chat/completions",
            headers=AUTH,
            json=body(
                messages=[
                    {"role": "user", "content": "What did the vet prescribe?"},
                    # Vapi's re-transcription of the answer it DID speak.
                    {
                        "role": "assistant",
                        "content": "Marvin is on 5 milligrams of prednisolone",
                    },
                    {"role": "user", "content": "Thanks."},
                ]
            ),
        )
        assert not [s for s in says if "prednisolone" in s.get("content", "")]
        assert delivery.state == "spoken"


def test_detection_without_recovery_is_a_supported_configuration() -> None:
    """``speech_drop_replay=False`` leaves the record complete and only declines to
    act -- for anyone who would rather read the journal than have the adapter speak on
    its own initiative.
    """
    with app_with(speech_drop_replay=False) as (client, settings, says):
        state, delivery = seed_dropped_answer(client, settings, "Marvin is on prednisolone.")
        says.clear()
        client.post(
            "/chat/completions",
            headers=AUTH,
            json=body(
                messages=[
                    {"role": "user", "content": "What did the vet prescribe?"},
                    {"role": "user", "content": "Hello?"},
                ]
            ),
        )
        assert delivery.state == "dropped"
        assert not [s for s in says if "prednisolone" in s.get("content", "")]


def test_the_replay_is_declared_to_hermes_so_the_model_cannot_answer_twice() -> None:
    """The one double-speaking risk the ledger cannot prevent on its own: the model
    answering the callee's "are you there?" from a history that -- correctly -- does
    not contain the dropped answer, so the callee hears the answer once from the replay
    and once more in the model's own words. Closed by telling Hermes the replayed text
    was said, which is true by the time it hears the callee's next utterance.
    """
    transport, hermes_state = build_fake_hermes_transport(FakeScript(deltas=["Anything."]))
    settings = app_settings()
    app = create_app(
        settings,
        hermes_transport=transport,
        vapi_control_transport=httpx.MockTransport(lambda _r: httpx.Response(200, json={})),
    )
    with TestClient(app) as client:
        client.post(
            "/chat/completions",
            headers=AUTH,
            json=body(messages=[{"role": "user", "content": "What did the vet prescribe?"}]),
        )
        state = _state_for(client)
        state.speech.deliveries.clear()
        delivery = state.speech.register(kind="answer", text="Marvin is on prednisolone.")
        delivery.at_monotonic_s -= settings.speech_confirm_window_seconds + 10.0
        client.post(
            "/chat/completions",
            headers=AUTH,
            json=body(
                messages=[
                    {"role": "user", "content": "What did the vet prescribe?"},
                    {"role": "user", "content": "Hello? Are you there?"},
                ]
            ),
        )
    sent = json.dumps(hermes_state.runs[-1]["body"])
    assert "Marvin is on prednisolone." in sent, (
        "Hermes must be told the replayed answer was spoken, or it will answer twice"
    )


# --------------------------------------------------------------------------- webhook


def test_the_webhook_does_not_exist_unless_a_secret_is_configured() -> None:
    """FAIL CLOSED. If somebody points the assistant's server URL at this adapter and
    the secret is not configured HERE, every event 404s and none is accepted -- rather
    than the adapter taking unauthenticated live-call events from whoever finds the
    path. Nothing degrades: the load-bearing feedback channel is the conversation
    history Vapi already sends.
    """
    with app_with() as (client, _settings, _says):
        assert client.post("/vapi/server", json={}).status_code == 404


def test_a_wrong_secret_is_404_and_not_401() -> None:
    """The endpoint's existence is not something an unauthenticated caller gets to
    learn, so a bad secret is indistinguishable from no such route."""
    with app_with(vapi_server_secret=WEBHOOK_SECRET) as (client, _s, _says):
        r = client.post("/vapi/server", json={}, headers={"x-vapi-secret": "wrong-but-long-enough"})
        assert r.status_code == 404
        assert client.post("/vapi/server", json={}).status_code == 404


def test_an_assistant_speech_update_confirms_outstanding_deliveries() -> None:
    """The timely channel: ~50-250 ms after audio really begins, versus the next turn.
    Verified live on call 01a0272a to arrive with only ``server: {url, secret}`` set --
    no ``serverMessages`` edit.
    """
    with app_with(vapi_server_secret=WEBHOOK_SECRET) as (client, settings, _says):
        state, delivery = seed_dropped_answer(client, settings, "Marvin is on prednisolone.")
        r = client.post(
            "/vapi/server",
            headers={"x-vapi-secret": WEBHOOK_SECRET},
            json={
                "message": {
                    "type": "speech-update",
                    "status": "started",
                    "role": "assistant",
                    "turn": 1,
                    "call": {"id": CALL_ID},
                }
            },
        )
        assert r.status_code == 200
        assert delivery.state == "spoken"
        assert delivery.evidence == "speech-update"


def test_a_speech_update_about_the_callee_confirms_nothing() -> None:
    """The same event type is sent for the CALLEE's speech on every turn of every call.
    Crediting a delivery because the callee started talking would confirm audio that
    never played -- switching the guard off exactly when the callee is asking "are you
    there?".
    """
    with app_with(vapi_server_secret=WEBHOOK_SECRET) as (client, settings, _says):
        state, delivery = seed_dropped_answer(client, settings, "Marvin is on prednisolone.")
        client.post(
            "/vapi/server",
            headers={"x-vapi-secret": WEBHOOK_SECRET},
            json={
                "message": {
                    "type": "speech-update",
                    "status": "started",
                    "role": "user",
                    "call": {"id": CALL_ID},
                }
            },
        )
        assert delivery.state == "unconfirmed"


def test_the_webhook_never_produces_a_drop_verdict() -> None:
    """Silence on this channel is silence, not evidence. Only Vapi's committed
    conversation history -- a settled account of a completed turn -- can condemn a
    delivery, so no sequence of webhook events can.
    """
    with app_with(vapi_server_secret=WEBHOOK_SECRET) as (client, settings, says):
        state, delivery = seed_dropped_answer(client, settings, "Marvin is on prednisolone.")
        says.clear()
        for kind in ("status-update", "hang", "conversation-update", "user-interrupted"):
            client.post(
                "/vapi/server",
                headers={"x-vapi-secret": WEBHOOK_SECRET},
                json={"message": {"type": kind, "call": {"id": CALL_ID}}},
            )
        assert delivery.state == "unconfirmed"
        assert says == []


def test_a_webhook_for_an_unknown_call_creates_no_state() -> None:
    """A webhook must not be a way to mint call state: an authenticated but misdirected
    event stream could otherwise fill the registry and evict the calls that have turns
    behind them.
    """
    with app_with(vapi_server_secret=WEBHOOK_SECRET) as (client, _s, _says):
        r = client.post(
            "/vapi/server",
            headers={"x-vapi-secret": WEBHOOK_SECRET},
            json={
                "message": {
                    "type": "speech-update",
                    "status": "started",
                    "role": "assistant",
                    "call": {"id": "01a00000-0000-7000-8000-000000000000"},
                }
            },
        )
        assert r.status_code == 200
        assert _state_for_id(client, "01a00000-0000-7000-8000-000000000000") is None


def test_a_malformed_or_irrelevant_body_is_ignored_without_erroring() -> None:
    """A webhook that errors on input it does not care about invites Vapi to retry it,
    and there is nothing to retry."""
    assert parse_server_message(b"not json") is None
    assert parse_server_message(b"[]") is None
    assert parse_server_message(b'{"message": 3}') is None
    assert parse_server_message(b'{"message": {"type": "transcript"}}') is None
    # A plausible event with an implausible call id never reaches a lookup or a log.
    assert (
        parse_server_message(
            json.dumps(
                {
                    "message": {
                        "type": "speech-update",
                        "status": "started",
                        "role": "assistant",
                        "call": {"id": "../../etc/passwd"},
                    }
                }
            ).encode()
        )
        is None
    )


def test_assistant_speech_started_is_honoured_when_present_and_carries_text() -> None:
    """Opt-in, and it did NOT fire for either Live Call Control ``say`` on probe call
    01a02727 -- so it is used as strong per-utterance evidence when it arrives and
    nothing depends on it.
    """
    event = parse_server_message(
        json.dumps(
            {
                "message": {
                    "type": "assistant.speechStarted",
                    "text": "Marvin is on prednisolone.",
                    "turn": 2,
                    "source": "model",
                    "call": {"id": CALL_ID},
                }
            }
        ).encode()
    )
    assert event is not None
    assert event.text == "Marvin is on prednisolone."
    with app_with(vapi_server_secret=WEBHOOK_SECRET) as (client, settings, _says):
        state, delivery = seed_dropped_answer(client, settings, "Marvin is on prednisolone.")
        other = state.speech.register(kind="answer", text="Something else entirely.")
        client.post(
            "/vapi/server",
            headers={"x-vapi-secret": WEBHOOK_SECRET},
            json={
                "message": {
                    "type": "assistant.speechStarted",
                    "text": "Marvin is on prednisolone.",
                    "call": {"id": CALL_ID},
                }
            },
        )
        assert delivery.state == "spoken"
        assert delivery.evidence == "assistant.speechStarted"
        assert other.state == "unconfirmed", "text-bearing evidence is per-utterance"


# --------------------------------------------------------- caller speech -> hold
#
# The MIRROR of the guard above: ``speech-update`` about the CALLEE (``role: "user"``)
# proves nothing about a delivery (see ``test_a_speech_update_about_the_callee_confirms_nothing``
# above and ``parse_server_message``'s docstring) but is real, structural evidence --
# from Vapi's own transcriber/VAD, not a guess -- that the callee is talking right
# now. Live call 01a028f1 interrupted Mike because an answer queued through Live Call
# Control has no such evidence available to it at all: it landed 46.527s into the
# call while the transcriber's own turn was still open (closed only at 47.754s).


def test_parse_caller_speech_event_recognises_started_and_stopped() -> None:
    started = parse_caller_speech_event(
        json.dumps(
            {
                "message": {
                    "type": "speech-update",
                    "status": "started",
                    "role": "user",
                    "call": {"id": CALL_ID},
                }
            }
        ).encode()
    )
    assert started is not None
    assert started.call_id == CALL_ID
    assert started.started is True

    stopped = parse_caller_speech_event(
        json.dumps(
            {
                "message": {
                    "type": "speech-update",
                    "status": "stopped",
                    "role": "user",
                    "call": {"id": CALL_ID},
                }
            }
        ).encode()
    )
    assert stopped is not None
    assert stopped.started is False


def test_parse_caller_speech_event_ignores_the_assistants_own_speech() -> None:
    """The mirror image of ``parse_server_message``'s ``role == "assistant"`` filter:
    this parser is the CALLEE half, so an assistant-role event is not its job."""
    assert (
        parse_caller_speech_event(
            json.dumps(
                {
                    "message": {
                        "type": "speech-update",
                        "status": "started",
                        "role": "assistant",
                        "call": {"id": CALL_ID},
                    }
                }
            ).encode()
        )
        is None
    )


def test_parse_caller_speech_event_ignores_malformed_or_irrelevant_input() -> None:
    assert parse_caller_speech_event(b"not json") is None
    assert parse_caller_speech_event(b"[]") is None
    assert parse_caller_speech_event(b'{"message": {"type": "transcript"}}') is None
    assert (
        parse_caller_speech_event(
            json.dumps(
                {"message": {"type": "speech-update", "status": "started", "role": "user"}}
            ).encode()
        )
        is None
    ), "no call id at all"
    assert (
        parse_caller_speech_event(
            json.dumps(
                {
                    "message": {
                        "type": "speech-update",
                        "status": "started",
                        "role": "user",
                        "call": {"id": "../../etc/passwd"},
                    }
                }
            ).encode()
        )
        is None
    ), "an implausible call id must never reach a lookup"


def test_a_caller_speech_update_holds_a_pending_answer_without_cancelling_it() -> None:
    """The whole point of the distinction from ``supersede_pending_answer``: the
    background delivery is still there, still able to speak, once the hold clears.
    """
    with app_with(vapi_server_secret=WEBHOOK_SECRET) as (client, settings, _says):
        client.post(
            "/chat/completions",
            headers=AUTH,
            json=body(messages=[{"role": "user", "content": "What did the vet prescribe?"}]),
        )
        state = _state_for(client)
        assert state.caller_speaking is False
        client.post(
            "/vapi/server",
            headers={"x-vapi-secret": WEBHOOK_SECRET},
            json={
                "message": {
                    "type": "speech-update",
                    "status": "started",
                    "role": "user",
                    "call": {"id": CALL_ID},
                }
            },
        )
        assert state.caller_speaking is True
        client.post(
            "/vapi/server",
            headers={"x-vapi-secret": WEBHOOK_SECRET},
            json={
                "message": {
                    "type": "speech-update",
                    "status": "stopped",
                    "role": "user",
                    "call": {"id": CALL_ID},
                }
            },
        )
        assert state.caller_speaking is False


def test_a_caller_speech_update_for_an_unknown_call_creates_no_state() -> None:
    with app_with(vapi_server_secret=WEBHOOK_SECRET) as (client, _s, _says):
        r = client.post(
            "/vapi/server",
            headers={"x-vapi-secret": WEBHOOK_SECRET},
            json={
                "message": {
                    "type": "speech-update",
                    "status": "started",
                    "role": "user",
                    "call": {"id": "01a00000-0000-7000-8000-000000000000"},
                }
            },
        )
        assert r.status_code == 200
        assert _state_for_id(client, "01a00000-0000-7000-8000-000000000000") is None
