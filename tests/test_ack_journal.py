"""The adapter's own record of the acknowledgements it emitted, and the endpoint
that serves it.

Why the record exists at all: an acknowledgement's text cannot say who wrote it.
"Okay, one moment." and "Sure, give me a second." are verbatim members of
``config._DEFAULT_FILLER_PHRASES``, and ``speech.VOICE_SYSTEM_PROMPT`` forbids the
MODEL from producing lines of that shape precisely because a model-authored one is
indistinguishable to the callee and so defeats the call-global cooldown. Any observer
downstream -- Vapi's transcript, the websocket transport, the E2E harness -- therefore
cannot attribute a spoken holding phrase by reading it. ``GET /debug/acks/{call_ref}``
is the evidence that turns that inference into a fact.

Two layers here: the bounded store on its own (``AckJournal``), then the endpoint and
its wiring over the real ASGI stack. Both matter -- a store nothing writes to and an
endpoint that records nothing are the same bug approached from two directions.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import httpx
from starlette.testclient import TestClient

from fake_hermes import FakeScript, build_fake_hermes_transport
from test_server_http import API_KEY, AUTH, make_settings, running_app, vapi_body
from vapi_hermes_voice.ack_journal import MAX_TEXT_CHARS, AckJournal, JournalSnapshot
from vapi_hermes_voice.call_state import call_ref
from vapi_hermes_voice.server import create_app

CONTROL_URL = "https://phone-call-websocket.vapi.ai/call-1/control"

# A turn that produces its answer slowly enough for the dead-air acknowledgement to
# fire, paired with `filler_after_seconds` well below it.
SLOW_ANSWER = {"deltas": ["Paris."], "delta_interval_s": 0.4}
ACK_SETTINGS: dict[str, Any] = {"filler_after_seconds": 0.05, "filler_phrases": ["One moment."]}


def make_journal(**overrides: Any) -> AckJournal:
    values: dict[str, Any] = {"max_calls": 4, "max_entries_per_call": 3, "ttl_seconds": 60.0}
    values.update(overrides)
    return AckJournal(**values)


def drive_turn(client: TestClient, call_id: str, *, control_url: str | None = None) -> str:
    """POST one streaming turn and return everything the callee would have heard."""
    body = vapi_body(call_id=call_id)
    if control_url is not None:
        body["call"]["monitor"] = {"controlUrl": control_url}
    with client.stream("POST", "/chat/completions", json=body, headers=AUTH) as response:
        return "".join(response.iter_text())


def acks_of(client: TestClient, call_id: str) -> dict[str, Any]:
    response = client.get(f"/debug/acks/{call_ref(call_id)}", headers=AUTH)
    assert response.status_code == 200, response.text
    return dict(response.json())


_FILLER_LOG_RE = re.compile(r"turn filler call=(\S+) elapsed_ms=(\d+) channel=(\S+)")


def assert_log_agrees(log_text: str, ack: dict[str, Any], *, channel: str) -> None:
    """The journal entry and the ``turn filler`` log line must be the same event.

    Both are written from one call site off one ``elapsed_ms`` (``turns._record_ack``)
    for exactly this reason: a record that can disagree with the log is worse than no
    record, because the next person to read them has to work out which one lied.
    """
    matches = _FILLER_LOG_RE.findall(log_text)
    assert len(matches) == 1, f"expected exactly one filler log line, got {matches}"
    _ref, elapsed_ms, logged_channel = matches[0]
    assert logged_channel == channel == ack["channel"]
    assert int(elapsed_ms) == ack["elapsed_ms"]


# --- the bounded store ------------------------------------------------------------


def test_records_text_channel_and_both_times() -> None:
    journal = make_journal()
    before = time.time()
    journal.record("abcdef012345", text="Okay, one moment.", channel="control", elapsed_ms=1130)
    after = time.time()

    snapshot = journal.snapshot("abcdef012345")
    assert snapshot is not None
    entries, dropped = snapshot.acks, snapshot.dropped
    assert dropped == 0
    assert [e.text for e in entries] == ["Okay, one moment."]
    assert entries[0].channel == "control"
    assert entries[0].elapsed_ms == 1130
    assert before <= entries[0].at_epoch_s <= after


def test_wire_form_exports_wall_clock_and_not_monotonic() -> None:
    """The exported timestamp must be one an off-box reader can align to its own clock.
    ``time.monotonic()`` has an arbitrary per-boot origin, so it is not a point in time
    anywhere outside this process: publishing it would invite exactly the cross-host
    arithmetic it cannot support.
    """
    journal = make_journal()
    journal.record("abcdef012345", text="Right, one second.", channel="stream", elapsed_ms=310)
    snapshot = journal.snapshot("abcdef012345")
    assert snapshot is not None
    assert set(snapshot.acks[0].as_dict()) == {"text", "channel", "at_epoch_s", "elapsed_ms"}
    # ...but it is kept internally, because TTL eviction must measure AGE, and a wall
    # clock stepped backwards by NTP would make an entry immortal.
    assert snapshot.acks[0].at_monotonic_s > 0


def test_unknown_call_ref_is_absent_not_empty() -> None:
    """None and [] are different answers: "no record, fall back to unknown" versus
    "this call genuinely emitted nothing". Collapsing them would let a reader accuse
    the model over a call the adapter never handled.
    """
    assert make_journal().snapshot("abcdef012345") is None


def test_an_opened_call_that_said_nothing_is_an_empty_record_not_an_absent_one() -> None:
    """The distinction the whole feature turns on. A call the adapter handled and
    correctly stayed silent on is what licenses a reader to call a holding phrase the
    callee heard MODEL-AUTHORED; a call it never saw must make the same reader say
    unknown. Both used to be "no bucket".
    """
    journal = make_journal()
    journal.open("abcdef012345")
    assert journal.snapshot("abcdef012345") == JournalSnapshot([], 0, [], 0)
    assert journal.snapshot("fedcba543210") is None


def test_an_empty_record_expires_on_its_own_age() -> None:
    """An empty bucket has no entry to expire, so it needs a timestamp of its own or
    it would live until the LRU cap pushed it out.
    """
    journal = make_journal(ttl_seconds=0.05)
    journal.open("abcdef012345")
    assert journal.snapshot("abcdef012345") == JournalSnapshot([], 0, [], 0)
    time.sleep(0.07)
    assert journal.snapshot("abcdef012345") is None


def test_opening_the_same_call_twice_does_not_duplicate_or_reset_it() -> None:
    journal = make_journal()
    journal.open("abcdef012345")
    journal.record("abcdef012345", text="Okay, one moment.", channel="control", elapsed_ms=5)
    journal.open("abcdef012345")  # a second turn on the same call
    assert len(journal) == 1
    snapshot = journal.snapshot("abcdef012345")
    assert snapshot is not None
    assert [e.text for e in snapshot.acks] == ["Okay, one moment."]


def test_entries_per_call_are_capped_and_the_loss_is_counted() -> None:
    journal = make_journal(max_entries_per_call=3)
    for i in range(10):
        journal.record("abcdef012345", text=f"phrase {i}", channel="stream", elapsed_ms=i)

    snapshot = journal.snapshot("abcdef012345")
    assert snapshot is not None
    entries, dropped = snapshot.acks, snapshot.dropped
    assert [e.text for e in entries] == ["phrase 7", "phrase 8", "phrase 9"]  # oldest evicted
    # Counted, never silently discarded: a consumer that cannot see the record is
    # incomplete would read a missing entry as "the model wrote that line".
    assert dropped == 7


def test_calls_retained_are_capped_evicting_the_oldest() -> None:
    journal = make_journal(max_calls=4)
    refs = [f"{i:012x}" for i in range(10)]
    for ref in refs:
        journal.record(ref, text="Okay, one moment.", channel="control", elapsed_ms=1)

    assert len(journal) == 4
    assert [journal.snapshot(ref) for ref in refs[:6]] == [None] * 6
    assert all(journal.snapshot(ref) is not None for ref in refs[6:])


def test_a_flood_cannot_exceed_the_stated_worst_case() -> None:
    """The whole memory argument in one assertion: whatever arrives, the store holds at
    most ``max_calls * max_entries_per_call`` records of at most ``MAX_TEXT_CHARS``.
    """
    journal = make_journal(max_calls=4, max_entries_per_call=3)
    for call in range(200):
        for _ in range(20):
            journal.record(f"{call:012x}", text="x" * 5_000, channel="stream", elapsed_ms=0)

    held = [journal.snapshot(f"{call:012x}") for call in range(200)]
    entries = [entry for snapshot in held if snapshot is not None for entry in snapshot.acks]
    assert len(journal) == 4
    assert len(entries) == 4 * 3
    assert all(len(entry.text) == MAX_TEXT_CHARS for entry in entries)


def test_entries_expire_by_age_and_the_call_disappears_with_them() -> None:
    journal = make_journal(ttl_seconds=0.05)
    journal.record("abcdef012345", text="Okay, one moment.", channel="control", elapsed_ms=1)
    assert journal.snapshot("abcdef012345") is not None

    time.sleep(0.07)
    # Not "0 entries, 1 lost": with nothing left to attribute with, the honest answer
    # is that there is no record, which is what makes a reader fall back to unknown.
    assert journal.snapshot("abcdef012345") is None
    assert len(journal) == 0


def test_ttl_expiry_sweeps_calls_that_are_still_being_written_to() -> None:
    """A bucket written to a moment ago can still hold an entry from the start of a
    long call, so expiry cannot only look at the least-recently-used end.
    """
    journal = make_journal(ttl_seconds=0.05, max_entries_per_call=8)
    journal.record("abcdef012345", text="old", channel="stream", elapsed_ms=1)
    time.sleep(0.07)
    journal.record("abcdef012345", text="new", channel="stream", elapsed_ms=2)

    snapshot = journal.snapshot("abcdef012345")
    assert snapshot is not None
    entries, dropped = snapshot.acks, snapshot.dropped
    assert [e.text for e in entries] == ["new"]
    assert dropped == 1


def test_reading_never_creates_an_entry() -> None:
    """A lookup is the one thing whose key a caller supplies, so it must not be a way
    to make the map grow.
    """
    journal = make_journal()
    for i in range(1000):
        assert journal.snapshot(f"{i:012x}") is None
    assert len(journal) == 0


# --- the endpoint: authentication, exposure, and what it refuses to say ------------


def test_endpoint_requires_the_adapter_api_key() -> None:
    with running_app(FakeScript(deltas=[])) as (client, _, _state):
        ref = call_ref("call-1")
        assert client.get(f"/debug/acks/{ref}").status_code == 401
        wrong = {"Authorization": "Bearer adapter-key-9999999999"}
        assert client.get(f"/debug/acks/{ref}", headers=wrong).status_code == 401
        # Both accepted header shapes, exactly as on /chat/completions. 404 here is
        # "authenticated, no record for this call", which is the next test.
        assert client.get(f"/debug/acks/{ref}", headers=AUTH).status_code == 404
        bare = {"Authorization": API_KEY}
        assert client.get(f"/debug/acks/{ref}", headers=bare).status_code == 404


def test_unknown_call_ref_is_404_with_no_body_detail() -> None:
    with running_app(FakeScript(deltas=[])) as (client, _, _state):
        response = client.get(f"/debug/acks/{call_ref('never-happened')}", headers=AUTH)
        assert response.status_code == 404
        assert response.json() == {"error": {"message": "not found"}}


def test_malformed_call_ref_is_404_and_is_never_echoed(caplog: Any) -> None:
    """The path segment is caller-supplied: it must not become a lookup key, and it
    must not reach a log line either.
    """
    junk = "not-a-call-ref-etcpasswd"
    with running_app(FakeScript(deltas=[])) as (client, _, _state):
        assert client.get(f"/debug/acks/{junk}", headers=AUTH).status_code == 404
    assert "etcpasswd" not in caplog.text


def test_disabled_journal_unregisters_the_route_entirely() -> None:
    """Off must be indistinguishable from never-built: 404, not 403."""
    with running_app(FakeScript(deltas=[]), debug_ack_journal=False) as (client, _, _state):
        assert client.get(f"/debug/acks/{call_ref('call-1')}", headers=AUTH).status_code == 404


def test_disabled_journal_records_nothing_at_all() -> None:
    with running_app(FakeScript(**SLOW_ANSWER), debug_ack_journal=False, **ACK_SETTINGS) as (
        client,
        _,
        _state,
    ):
        assert "One moment." in drive_turn(client, "call-off")  # the ack still happens
        assert client.get(f"/debug/acks/{call_ref('call-off')}", headers=AUTH).status_code == 404


def test_route_secret_also_covers_the_debug_surface() -> None:
    """The debug endpoint must never be reachable on a path the chat endpoint 404s."""
    secret = "route-secret-0123456789"
    with running_app(FakeScript(**SLOW_ANSWER), route_secret=secret, **ACK_SETTINGS) as (
        client,
        _,
        _state,
    ):
        body = vapi_body(call_id="call-rs")
        with client.stream(
            "POST", f"/v/{secret}/chat/completions", json=body, headers=AUTH
        ) as response:
            assert "One moment." in "".join(response.iter_text())

        ref = call_ref("call-rs")
        assert client.get(f"/debug/acks/{ref}", headers=AUTH).status_code == 404
        wrong = f"/v/route-secret-9999999999/debug/acks/{ref}"
        assert client.get(wrong, headers=AUTH).status_code == 404
        behind_secret = client.get(f"/v/{secret}/debug/acks/{ref}", headers=AUTH)
        assert behind_secret.status_code == 200
        assert len(behind_secret.json()["acks"]) == 1


# --- end to end: both delivery channels are recorded, and agree with the log ------


def test_stream_channel_acknowledgement_is_recorded(caplog: Any) -> None:
    """No control URL on the request, so the acknowledgement is embedded in the SSE
    stream -- and the record says so.
    """
    caplog.set_level("INFO")
    with running_app(FakeScript(**SLOW_ANSWER), **ACK_SETTINGS) as (client, _, _state):
        assert "One moment." in drive_turn(client, "call-s")
        record = acks_of(client, "call-s")

    assert record["call_ref"] == call_ref("call-s")
    assert record["dropped"] == 0
    assert [(a["text"], a["channel"]) for a in record["acks"]] == [("One moment.", "stream")]
    # The BARE phrase, not the SSE framing: `<flush />` is inaudible transport detail,
    # and matching against it off-box would fail for no reason.
    assert "<flush" not in record["acks"][0]["text"]
    assert_log_agrees(caplog.text, record["acks"][0], channel="stream")


def test_control_channel_acknowledgement_is_recorded(caplog: Any) -> None:
    """Delivered via Vapi Live Call Control, which leaves NOTHING on model.url. This
    is the channel the E2E harness structurally cannot see, and the whole reason the
    endpoint has to exist.
    """
    caplog.set_level("INFO")
    said: list[str] = []

    def control(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            said.append(str(json.loads(request.content)["content"]))
        return httpx.Response(200, json={"status": "ok"})

    hermes_transport, _state = build_fake_hermes_transport(FakeScript(**SLOW_ANSWER))
    app = create_app(
        make_settings(**ACK_SETTINGS),
        hermes_transport=hermes_transport,
        vapi_control_transport=httpx.MockTransport(control),
    )
    with TestClient(app) as client:
        heard_on_stream = drive_turn(client, "call-c", control_url=CONTROL_URL)
        record = acks_of(client, "call-c")

    assert said[:1] == ["One moment."]
    assert "One moment." not in heard_on_stream  # never both: the callee hears it once
    assert [(a["text"], a["channel"]) for a in record["acks"]] == [("One moment.", "control")]
    assert_log_agrees(caplog.text, record["acks"][0], channel="control")


def test_two_calls_are_kept_apart() -> None:
    with running_app(FakeScript(**SLOW_ANSWER), **ACK_SETTINGS) as (client, _, _state):
        for call_id in ("call-a", "call-b"):
            drive_turn(client, call_id)
        assert len(acks_of(client, "call-a")["acks"]) == 1
        assert len(acks_of(client, "call-b")["acks"]) == 1
        assert acks_of(client, "call-a")["call_ref"] != acks_of(client, "call-b")["call_ref"]


def test_a_fast_turn_records_an_empty_record_not_an_absent_one() -> None:
    """A fast turn correctly speaks no acknowledgement -- and that is the single most
    important thing this endpoint can say. "I drove this turn and emitted nothing" is
    what lets a reader call a holding phrase the callee heard MODEL-AUTHORED; a 404
    would collapse it into "I have no idea", which is the answer the harness already
    had before this endpoint existed.
    """
    fast = FakeScript(deltas=["Paris."], delta_interval_s=0.0)
    with running_app(fast, filler_after_seconds=5.0) as (client, _, _state):
        assert "Paris." in drive_turn(client, "call-fast")
        record = acks_of(client, "call-fast")
    assert record["acks"] == []
    assert record["dropped"] == 0


def test_limits_are_reported_alongside_the_record() -> None:
    with running_app(
        FakeScript(**SLOW_ANSWER),
        debug_ack_journal_max_calls=7,
        debug_ack_journal_max_entries_per_call=5,
        debug_ack_journal_ttl_seconds=123.0,
        **ACK_SETTINGS,
    ) as (client, _, _state):
        drive_turn(client, "call-l")
        limits = acks_of(client, "call-l")["limits"]

    assert limits["max_calls"] == 7
    assert limits["max_entries_per_call"] == 5
    assert limits["ttl_seconds"] == 123.0
    assert limits["max_text_chars"] == MAX_TEXT_CHARS


# --- the acknowledgement is recorded even if the stream dies at that exact yield ---


async def test_a_stream_ack_is_recorded_even_when_vapi_drops_the_connection() -> None:
    """Live failure (call 01a02681, turn 1): Vapi dropped the model.url connection at
    exactly the yield that hands the acknowledgement to the response. The bytes were
    already on the wire and WERE spoken -- Vapi's own message list has bot@2.642
    "Alright, let me see." -- but the generator was closed at that yield, so a
    ``_record_ack`` placed after it never ran. The record then omitted an
    acknowledgement the callee had heard while still reporting ``dropped == 0``, i.e.
    claiming to be complete, and an off-box reader attributed the phrase to the model:
    a false MODEL-AUTHORED verdict, the exact failure this journal exists to prevent.

    ``aclose()`` at the acknowledgement chunk raises GeneratorExit at that yield, which
    is the same cancellation the live connection drop produced.
    """
    from test_turns import _ScriptedHermes, delta, done, make_state
    from test_turns import make_settings as turn_settings
    from vapi_hermes_voice.turns import stream_turn

    journal = make_journal()
    settings = turn_settings(filler_after_seconds=0.05, filler_phrases=["One moment."])
    state = make_state(settings)
    reaping: set[Any] = set()
    agen = stream_turn(
        settings=settings,
        hermes=_ScriptedHermes([(0.4, delta("Paris.")), (0.0, done())]),
        state=state,
        instructions="instructions",
        history=[],
        user_input="hello",
        reaping=reaping,
        journal=journal,
    )
    heard = ""
    async for chunk in agen:
        heard += chunk
        if "One moment." in chunk:
            # The callee has the bytes. Now the connection dies, right here.
            await agen.aclose()
            break

    assert "One moment." in heard  # it really was on the wire
    snapshot = journal.snapshot(state.call_ref)
    assert snapshot is not None, "the acknowledgement the callee heard left no record"
    entries, dropped = snapshot.acks, snapshot.dropped
    assert [(e.text, e.channel) for e in entries] == [("One moment.", "stream")]
    # And the record does not quietly claim to be complete while missing an entry.
    assert dropped == 0
