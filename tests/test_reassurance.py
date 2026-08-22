"""Reassurance while an already-acknowledged wait runs long (turns.py).

The defect: ``filler_min_gap_seconds`` is a FLOOR on how close two holding lines may
be, and nothing was ever spending what it allowed. The acknowledgement ends the
model.url response and the rest of the turn runs on a background continuation that
spoke exactly once, at the end -- so an acknowledged turn whose answer took a long time
was dead air for the whole of it, which is the complaint this adapter exists to fix.

Measured on live call 01a028f1, from Vapi's own server-side call log (the platform's
record of what it spoke, not the adapter's of what it sent):

    70.19  model first token   -> the acknowledgement
    70.47  botSpeechStarted       "Okay. Let me check."
    71.45  botSpeechStopped
    94.67  sayQueuePush        -> the answer
    95.00  botSpeechStarted
                                  23.55 s of unbroken silence in between, with the
                                  callee saying nothing at all in it

A second window on the same call ran 12.45 s (118.76 -> 131.21). Across 34 turns of
adapter journal the acknowledgement-to-answer wait had p50 9.0 s, p75 14.6 s, max
33.2 s, so this is the ordinary shape of a calendar turn and not one bad call.

Those 34 waits also decide the trigger, and the shape matters for reading the tests:
eleven of them sit in a 3.1-second band at 13.2-16.3 s (one Hermes tool round trip,
answered), then there is an empty stretch, then the painful tail is three turns at
19.1, 24.6 and 33.2 s. A trigger inside the cluster would fire on 14 of 34 turns and
land within 5 seconds of the answer on 8 of them -- a line inserted just before the
payload it was covering for. So the shipped ``reassure_after_seconds`` is 18.0, past
the cluster: it fires on the tail, and the 14.0 s window of the reviewed call
correctly hears nothing. See ``config.reassure_after_seconds`` for the full
arithmetic.

Every test here scales the shipped ratios down by 100x so the suite stays fast, and
the ratio is what is being asserted. At the shipped defaults (``reassure_after_seconds``
18.0, ``filler_min_gap_seconds`` 10.0, ``reassure_backoff`` 2.0) reassurances fall due
18 s, 54 s and 126 s into a silence, so a 25 s wait hears exactly one and a 60 s wait
two.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import Any

import httpx

from test_turns import _parse_chunk, _ScriptedHermes, delta, done, make_settings, make_state
from vapi_hermes_voice.ack_journal import AckJournal
from vapi_hermes_voice.call_state import CallState
from vapi_hermes_voice.config import Settings
from vapi_hermes_voice.turns import stream_turn
from vapi_hermes_voice.vapi_control import VapiControlClient

CONTROL_URL = "https://phone-call-websocket.vapi.ai/call-1/control"


class _Said:
    """Everything handed to Live Call Control on a turn, in order, with timestamps.

    Timestamps because most assertions here are about WHEN a line was spoken relative
    to the acknowledgement that opened the silence; order because requirement 4 --
    nothing spoken after the answer -- is an ordering property, not a timing one.
    """

    def __init__(self) -> None:
        self.at: list[float] = []
        self.content: list[str] = []
        self.zero = time.monotonic()

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.at.append(time.monotonic() - self.zero)
        self.content.append(json.loads(request.read())["content"])
        return httpx.Response(200, json={"status": "ok"})


def reassure_settings(**overrides: Any) -> Settings:
    """The shipped shape at 1/100 scale: 10.0 -> 0.10 cooldown, 18.0 -> 0.18 trigger."""
    values: dict[str, Any] = {
        "filler_after_seconds": 0.02,
        "filler_min_gap_seconds": 0.10,
        "reassure_after_seconds": 0.18,
        "reassure_backoff": 2.0,
        "filler_phrases": ["Okay, let me check."],
        "reassure_phrases": ["Still working.", "Almost there.", "Still on it."],
    }
    values.update(overrides)
    return make_settings(**values)


async def drive(
    events: list[tuple[float, Any]],
    settings: Settings,
    *,
    state: CallState | None = None,
    journal: AckJournal | None = None,
    handler: Any = None,
) -> tuple[_Said, list[str]]:
    """One turn through ``stream_turn`` with a live control channel.

    Returns what reached control and the SSE content chunks (i.e. the acknowledgement).
    The background continuation is drained through ``reaping`` before returning, so
    every assertion below reads a finished turn rather than a racing one.
    """
    said = _Said()

    def record_then_answer(request: httpx.Request) -> httpx.Response:
        said.handler(request)
        return (handler or (lambda _r: httpx.Response(200, json={"status": "ok"})))(request)

    control = VapiControlClient(transport=httpx.MockTransport(record_then_answer))
    reaping: set[asyncio.Task[Any]] = set()
    chunks: list[str] = []
    agen = stream_turn(
        settings=settings,
        hermes=_ScriptedHermes(events),
        control=control,
        control_url=CONTROL_URL,
        state=state if state is not None else make_state(settings),
        instructions="instructions",
        history=[],
        user_input="hello",
        reaping=reaping,
        journal=journal,
    )
    async for chunk in agen:
        for item in _parse_chunk(chunk):
            if isinstance(item, tuple) and item[0] == "content":
                chunks.append(item[1])
    while reaping:
        for task in list(reaping):
            with contextlib.suppress(Exception):
                await task
    await control.aclose()
    return said, chunks


def reassurances(said: _Said, settings: Settings) -> list[str]:
    return [c for c in said.content if c in settings.reassure_phrases]


# --- 1. the reported defect: a long acknowledged wait is no longer silent ------


async def test_a_25s_shaped_wait_hears_exactly_one_reassurance() -> None:
    """One, not none and not two: the acceptance shape of the live 23.55 s window.

    Scaled 25 s against an 18 s trigger -- the silence runs 0.28 s, the first
    reassurance falls due at 0.18 s, and the second (0.36 s after it, backoff 2.0)
    falls outside. Before this existed the continuation spoke only once, at the end, so
    this asserts exactly the line the callee did not hear on call 01a028f1.
    """
    settings = reassure_settings()
    events = [(0.30, delta("September seventh at nine works.")), (0.01, done())]

    said, acks = await drive(events, settings)

    spoken = reassurances(said, settings)
    assert len(spoken) == 1, f"expected exactly one reassurance, got {said.content}"
    assert acks[0].startswith("Okay, let me check."), acks
    # Due 0.18 s after the acknowledgement, which itself lands at ~filler_after (0.02).
    assert 0.18 <= said.at[0] <= 0.28, said.at


async def test_a_60s_shaped_wait_hears_two_reassurances_and_the_gaps_double() -> None:
    """Two, and the second twice as far out as the first, so the count grows with the
    logarithm of the wait rather than with the wait.

    Scaled 60 s against an 18 s trigger. A third would fall due 0.72 s after the
    second, far past the end of this silence, which is the whole point of the backoff:
    a caller waiting a minute is told twice, not six times.
    """
    settings = reassure_settings()
    events = [(0.70, delta("Both slots are open.")), (0.01, done())]

    said, _acks = await drive(events, settings)

    spoken = reassurances(said, settings)
    assert len(spoken) == 2, f"expected two reassurances, got {said.content}"
    first, second = said.at[0], said.at[1]
    assert 0.18 <= first <= 0.28, said.at
    # The second gap is reassure_after * backoff = 0.36, measured from the first.
    assert 0.36 <= second - first <= 0.48, said.at


# --- 2. the cooldown is a floor, and it is call-global ------------------------


async def test_no_two_holding_lines_are_closer_than_the_cooldown() -> None:
    """Every audible gap, acknowledgement included, is at least the cooldown.

    The acknowledgement goes out on the SSE stream and the reassurances through
    control, so this is the one assertion that has to span both channels: the callee
    cannot tell them apart, and the requirement is about what the callee hears.
    """
    settings = reassure_settings(filler_min_gap_seconds=0.10)
    events = [(0.70, delta("Both slots are open.")), (0.01, done())]

    said, _acks = await drive(events, settings)

    spoken = reassurances(said, settings)
    assert len(spoken) == 2, said.content
    # The acknowledgement lands at ~filler_after_seconds; treat it as the first entry.
    heard = [settings.filler_after_seconds, *said.at[: len(spoken)]]
    gaps = [b - a for a, b in zip(heard, heard[1:], strict=False)]
    assert all(gap >= settings.filler_min_gap_seconds for gap in gaps), gaps


async def test_a_reassurance_spends_the_slot_the_next_turns_acknowledgement_needs() -> None:
    """The cooldown is global to the CALL: a reassurance on turn 1 gates turn 2's
    acknowledgement, exactly as turn 1's own acknowledgement would.

    Discriminating by construction, and that is the point of the odd-looking numbers.
    Turn 1's acknowledgement lands at ~0.02 s and its reassurance at ~0.56 s; turn 2
    starts at ~0.64 s. That is ~0.08 s after the reassurance -- inside the 0.30 s
    cooldown, so refused -- but ~0.62 s after the acknowledgement, which would have
    allowed it. So this can only pass if the reassurance moved the call-global anchor,
    and it fails the moment reassurance is given a budget of its own.

    Same 10:18 ratio as the shipped defaults, scaled up 3x from the module fixture so
    the two turns are far enough apart to time reliably.
    """
    settings = reassure_settings(filler_min_gap_seconds=0.30, reassure_after_seconds=0.54)
    state = make_state(settings)

    said_one, chunks_one = await drive(
        [(0.62, delta("Both slots are open.")), (0.01, done())], settings, state=state
    )
    assert len(reassurances(said_one, settings)) == 1, said_one.content
    assert any(c.startswith("Okay, let me check.") for c in chunks_one), (
        f"turn 1 must still be acknowledged: {chunks_one}"
    )

    said_two, chunks_two = await drive(
        [(0.05, delta("Nine works.")), (0.01, done())], settings, state=state
    )

    # Turn 2 was refused its acknowledgement, so it never hands off: it stays on the
    # SSE stream and its answer arrives as ordinary content there. Nothing on EITHER
    # channel may be a holding phrase from EITHER pool.
    heard = [*chunks_two, *said_two.content]
    for line in heard:
        assert not any(phrase in line for phrase in settings.holding_phrases), (
            f"turn 2 spoke a holding phrase inside the cooldown: {line!r}"
        )
    assert any("Nine works." in line for line in heard), heard


# --- 3. nothing is ever spoken on top of, or after, the answer ----------------


async def test_the_answer_is_always_the_last_thing_said() -> None:
    """Requirement 4, as an ordering assertion rather than a timing one.

    A holding phrase after the answer is worse than silence: it re-opens a wait the
    callee has already been let out of. Both POSTs come from one coroutine and the
    reassurance is fully awaited before the loop can break, so this holds structurally
    -- asserted here so a later refactor that moves either onto its own task fails
    loudly instead of quietly.
    """
    settings = reassure_settings()
    answer = "Both slots are open."
    events = [(0.70, delta(answer)), (0.01, done())]

    said, _acks = await drive(events, settings)

    assert said.content.count(answer) == 1, said.content
    assert said.content[-1] == answer, said.content


async def test_a_wait_that_ends_on_the_deadline_still_puts_the_answer_last() -> None:
    """The deliberately nasty case: Hermes concludes right around the moment a
    reassurance falls due, repeatedly, so the ordering is exercised on both sides of
    the boundary rather than only on the comfortable side.

    Whichever way a trial lands -- reassurance then answer, or answer alone -- the
    answer is last. What is NOT asserted is which of the two happens, because that is a
    genuine coin flip on the scheduler and pinning it would be asserting noise.
    """
    settings = reassure_settings()
    answer = "Nine or nine thirty."
    for _ in range(12):
        said, _acks = await drive([(0.20, delta(answer)), (0.001, done())], settings)
        assert said.content[-1] == answer, said.content
        assert said.content.count(answer) == 1, said.content


async def test_the_answer_waits_for_a_reassurance_still_in_flight_and_is_never_lost() -> None:
    """The case ordering alone does not settle: the answer becoming ready while the
    reassurance POST is still open.

    What Vapi does with a ``say`` that arrives mid-utterance was measured, not assumed
    -- ``tests/e2e/say_queue_probe.py`` on real call 01a0291d-7528, POSTing a second
    ``say`` 513 ms into a 5.5 s first one. Vapi's own log, ms from the first push:

        +371   utterance 1 botSpeechStarted
        +884   push 2                        <- 513 ms INTO utterance 1
       +5884   utterance 1 botSpeechStopped  <- 5000 ms AFTER push 2
       +6235   utterance 2 botSpeechStarted  <- 351 ms after 1 stopped
       +7137   utterance 2 botSpeechStopped

    Strictly queued: not truncated, not overlapped, and above all NOT DROPPED. So the
    worst this costs is the answer waiting out one holding phrase (~1.0-1.5 s), and the
    thing that would have been unacceptable -- a reassurance in flight swallowing the
    answer -- cannot happen.

    What this test can hold up in CI is the adapter's half of that: a reassurance POST
    that is still open when Hermes concludes must not lose, reorder, or duplicate the
    answer. The control channel is made deliberately slow so the overlap is certain
    rather than incidental.
    """
    settings = reassure_settings()
    answer = "Both slots are open."
    slow_started: list[float] = []

    def slow_reassurance(request: httpx.Request) -> httpx.Response:
        content = json.loads(request.read())["content"]
        if content in settings.reassure_phrases:
            slow_started.append(time.monotonic())
            # Still in flight when the 0.22 s Hermes conclusion below lands.
            time.sleep(0.15)
        return httpx.Response(200, json={"status": "ok"})

    said, _acks = await drive(
        [(0.22, delta(answer)), (0.001, done())], settings, handler=slow_reassurance
    )

    assert slow_started, f"the slow reassurance never went out: {said.content}"
    assert said.content.count(answer) == 1, said.content
    assert said.content[-1] == answer, said.content
    assert reassurances(said, settings) == said.content[:-1], said.content


async def test_a_fast_turn_hears_no_reassurance_at_all() -> None:
    """The answer arriving before the first deadline must leave the callee alone.

    Reassurance firing on a turn with no long silence in it is nagging, and the live
    journal says that is the overwhelming majority: 31 of 34 turns waited under 18 s.
    """
    settings = reassure_settings()
    events = [(0.05, delta("Nine works.")), (0.01, done())]

    said, _acks = await drive(events, settings)

    assert said.content == ["Nine works."], said.content


# --- 4. the phrases themselves ------------------------------------------------


async def test_a_reassurance_never_repeats_the_line_before_it() -> None:
    settings = reassure_settings()
    events = [(0.70, delta("Both slots are open.")), (0.01, done())]

    said, acks = await drive(events, settings)

    spoken = reassurances(said, settings)
    assert len(spoken) == 2, said.content
    assert spoken[0] != spoken[1], spoken
    for phrase in spoken:
        assert phrase not in settings.filler_phrases, (
            f"{phrase!r} is an acknowledgement line: eleven seconds into a silence the"
            " callee said nothing eleven seconds ago, so a line that acknowledges them"
            " acknowledges something that did not happen"
        )
        assert not any(phrase in ack for ack in acks), (phrase, acks)


def test_the_shipped_pools_are_disjoint_and_the_reassurances_acknowledge_nothing() -> None:
    """Read the shipped defaults, not a test fixture: the wording IS the requirement.

    A reassurance opening with an acknowledgement token ("Got it", "Understood",
    "Okay") refers to something the callee did not just say. On a call whose reported
    defect was the assistant answering half a sentence, that is precisely the wrong
    impression to leave, and it is the reason these are a second pool rather than a
    second draw from the first.
    """
    acks = Settings.model_fields["filler_phrases"].default
    reassure = Settings.model_fields["reassure_phrases"].default

    assert set(acks).isdisjoint(reassure)
    for phrase in reassure:
        first_word = phrase.split()[0].strip(",").lower()
        assert first_word not in {"okay", "got", "understood", "right", "alright", "sure"}, phrase


def test_the_shipped_delay_clears_the_scored_cooldown_with_render_lag_to_spare() -> None:
    """``reassure_after_seconds`` has to clear the SCORED floor, not just the configured
    one, and the two are measured on different clocks.

    The cooldown is enforced on the adapter's claim; the E2E harness scores it
    speaker-to-speaker (``deadlines.Budgets.ack_cooldown_s``), and the two lines reach
    the speaker down different paths. Measured on live call 01a028f1: a line streamed
    as model output rendered 0.28 s after its first token, a ``sayQueuePush`` 0.33 s
    and 0.38 s after its POST. So the observed gap is the configured gap plus a
    lag DIFFERENCE that can go either way, and a timer sitting exactly on the floor
    scores as a FAIL whenever it goes the wrong way -- verified directly against
    `evaluate`: a 9.95 s observed gap fails `r2_ack_cooldown`, 10.00 s passes.

    One second of headroom is what makes that a non-question, and this is the assertion
    that fails if someone later "tidies" the default down onto the floor.
    """
    delay = Settings.model_fields["reassure_after_seconds"].default
    floor = Settings.model_fields["filler_min_gap_seconds"].default
    # Largest render-lag asymmetry seen on 01a028f1, doubled: the sign is not knowable
    # in advance, so the headroom has to cover it in both directions.
    worst_observed_lag_skew_s = 2 * (0.38 - 0.28)

    assert delay >= floor + worst_observed_lag_skew_s, (
        f"reassure_after_seconds={delay} leaves only {delay - floor}s over the"
        f" {floor}s cooldown, which render-lag asymmetry alone can eat"
    )


# --- 5. switched off, and mis-configured -------------------------------------


async def test_reassurance_disabled_restores_the_previous_behaviour_exactly() -> None:
    """``reassure_after_seconds <= 0``: one control POST, the answer, however long the
    silence -- exactly what this file's defect describes.
    """
    settings = reassure_settings(reassure_after_seconds=0.0)
    events = [(0.70, delta("Both slots are open.")), (0.01, done())]

    said, acks = await drive(events, settings)

    assert said.content == ["Both slots are open."], said.content
    assert acks and acks[0].startswith("Okay, let me check."), acks


async def test_a_timer_below_the_cooldown_backs_off_instead_of_spinning() -> None:
    """A first reassurance due sooner than the cooldown allows must not busy-loop.

    The claim is refused, the deadline is re-armed from the anchor that refused it, and
    the gap doubles -- so attempts are geometric (a handful across this 0.7 s silence,
    ~7 across a real minute) and the loop spends the rest of its life waiting on
    Hermes. An implementation that re-armed from "now" instead would spin at timeout
    0.0 for the whole silence: a hot loop on the event loop that carries the call.
    """
    settings = reassure_settings(filler_min_gap_seconds=5.0, reassure_after_seconds=0.02)
    attempts = 0
    real_claim = CallState.claim_reassurance

    def counting_claim(self: CallState, **kwargs: Any) -> str | None:
        nonlocal attempts
        attempts += 1
        return real_claim(self, **kwargs)

    CallState.claim_reassurance = counting_claim  # type: ignore[method-assign]
    try:
        said, _acks = await drive([(0.70, delta("Both slots are open.")), (0.01, done())], settings)
    finally:
        CallState.claim_reassurance = real_claim  # type: ignore[method-assign]

    assert said.content == ["Both slots are open."], said.content
    assert 1 <= attempts <= 12, f"{attempts} claim attempts: the deadline is not backing off"


# --- 6. off-box attribution ---------------------------------------------------


async def test_a_reassurance_is_journalled_as_the_adapters_own() -> None:
    """Recorded, with its text and channel, so ``GET /debug/acks`` can still say who
    wrote every holding phrase the callee heard.

    Without this the harness sees a pool phrase spoken with no adapter emission behind
    it, which is the signature of a MODEL-authored holding line -- the exact confusion
    the journal exists to prevent, manufactured by the fix.
    """
    settings = reassure_settings()
    journal = AckJournal(max_calls=8, max_entries_per_call=16, ttl_seconds=900.0)
    events = [(0.30, delta("Nine works.")), (0.01, done())]

    said, _acks = await drive(events, settings, journal=journal)

    spoken = reassurances(said, settings)
    assert len(spoken) == 1, said.content
    snapshot = journal.snapshot("ref-1")
    assert snapshot is not None
    texts = [(record.text, record.channel) for record in snapshot.acks]
    assert (spoken[0], "control") in texts, texts
    # The acknowledgement that opened the silence is still recorded, on its own channel.
    assert any(channel == "stream" for _text, channel in texts), texts


async def test_an_undelivered_reassurance_is_never_recorded_as_spoken() -> None:
    """A control POST Vapi refused said nothing, so the journal must not claim it did.

    Same order as the answer-delivery fallback line and the opposite of the SSE
    acknowledgement, and the difference is what can be observed: this path sees its own
    outcome, so recording before the POST would publish a line the callee never heard.
    """
    settings = reassure_settings()
    journal = AckJournal(max_calls=8, max_entries_per_call=16, ttl_seconds=900.0)

    def refuse_reassurance(request: httpx.Request) -> httpx.Response:
        content = json.loads(request.read())["content"]
        if content in settings.reassure_phrases:
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json={"status": "ok"})

    said, _acks = await drive(
        [(0.30, delta("Nine works.")), (0.01, done())],
        settings,
        journal=journal,
        handler=refuse_reassurance,
    )

    assert reassurances(said, settings), said.content
    snapshot = journal.snapshot("ref-1")
    assert snapshot is not None
    for record in snapshot.acks:
        assert record.text not in settings.reassure_phrases, (
            f"{record.text!r} was journalled as spoken after Vapi refused it"
        )
    # And the answer still got through: a failed reassurance never becomes a failed turn.
    assert said.content[-1] == "Nine works.", said.content
