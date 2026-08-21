"""Regression tests for vapi_hermes_voice.turns: the filler/content race.

Drives ``stream_turn`` directly against a scripted fake Hermes event source with
precise ``asyncio`` timing control, rather than going through the full FastAPI/SSE
stack (test_server_http.py) -- these tests are about exact event ordering and race
windows, which is far more deterministic to express and assert against a single
generator than against a real socket.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import time
from collections.abc import AsyncIterator
from typing import Any

from vapi_hermes_voice.call_state import CallState
from vapi_hermes_voice.config import Settings
from vapi_hermes_voice.hermes_client import HermesTurnEvent
from vapi_hermes_voice.speech import FillerPicker, sanitize_spoken
from vapi_hermes_voice.turns import stream_turn


class _ScriptedHermes:
    """A fake HermesClient exposing only ``run_turn``, replaying timed events."""

    def __init__(self, events: list[tuple[float, HermesTurnEvent]]) -> None:
        self._events = events

    def run_turn(self, **_kwargs: Any) -> AsyncIterator[HermesTurnEvent]:
        return self._replay()

    async def _replay(self) -> AsyncIterator[HermesTurnEvent]:
        for delay, event in self._events:
            if delay:
                await asyncio.sleep(delay)
            yield event


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "hermes_base_url": "http://fake-hermes.invalid",
        "hermes_api_key": "unit-test-key",
        "adapter_api_key": "unit-test-secret-0123456789",
        "filler_after_seconds": 0.06,
        "filler_min_gap_seconds": 0.02,  # tests override this explicitly when it matters
        "filler_use_flush": True,
        "filler_phrases": ["One moment.", "Let me check.", "Just a second.", "Hold on."],
        "filler_max_per_turn": 1,
        "_env_file": None,
    }
    values.update(overrides)
    return Settings(**values)


def make_state(settings: Settings) -> CallState:
    return CallState(
        session_id="s-1",
        session_key="k-1",
        call_ref="ref-1",
        filler=FillerPicker(settings.filler_phrases),
    )


def delta(text: str) -> HermesTurnEvent:
    return HermesTurnEvent(kind="delta", text=text)


def tool_start() -> HermesTurnEvent:
    return HermesTurnEvent(kind="tool_start")


def done(text: str = "") -> HermesTurnEvent:
    return HermesTurnEvent(kind="done", text=text)


def error(text: str) -> HermesTurnEvent:
    return HermesTurnEvent(kind="error", text=text)


def _parse_chunk(chunk: str) -> list[tuple[str, str] | str]:
    out: list[tuple[str, str] | str] = []
    for line in chunk.split("\n"):
        line = line.strip()
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload == "[DONE]":
            out.append("[DONE]")
            continue
        obj = json.loads(payload)
        d = obj["choices"][0]["delta"]
        if "content" in d:
            out.append(("content", d["content"]))
        elif "role" in d:
            out.append(("role", d["role"]))
    return out


async def run(
    events: list[tuple[float, HermesTurnEvent]],
    settings: Settings,
    *,
    on_content: Any = None,
    state: CallState | None = None,
) -> list[tuple[str, str] | str]:
    """Drive stream_turn to completion; return parsed (kind, text)/"[DONE]" items.

    ``on_content`` (if given) is awaited with the just-yielded content text right
    after each content chunk is received by this "consumer" -- used to simulate the
    consumer being busy (e.g. still writing a chunk to the socket) while a Hermes
    event completes concurrently in the background.

    ``state`` (if given) is reused as-is instead of building a fresh one -- used to
    model two sequential turns of the *same* call, exactly as CallStateRegistry
    hands the same CallState back to server.py on every turn of a call.
    """
    reaping: set[asyncio.Task[None]] = set()
    parsed: list[tuple[str, str] | str] = []
    agen = stream_turn(
        settings=settings,
        hermes=_ScriptedHermes(events),
        state=state if state is not None else make_state(settings),
        instructions="instructions",
        history=[],
        user_input="hello",
        reaping=reaping,
    )
    async for chunk in agen:
        items = _parse_chunk(chunk)
        parsed.extend(items)
        if on_content is not None:
            for item in items:
                if isinstance(item, tuple) and item[0] == "content":
                    await on_content(item[1])
    for task in list(reaping):
        with contextlib.suppress(Exception):
            await task
    return parsed


def content_chunks(parsed: list[tuple[str, str] | str]) -> list[str]:
    return [item[1] for item in parsed if isinstance(item, tuple) and item[0] == "content"]


def is_filler_chunk(text: str, phrases: list[str]) -> bool:
    stripped = text.strip()
    return any(stripped == phrase or stripped.startswith(phrase + " ") for phrase in phrases)


def filler_chunks(parsed: list[tuple[str, str] | str], phrases: list[str]) -> list[str]:
    return [c for c in content_chunks(parsed) if is_filler_chunk(c, phrases)]


def real_chunks(parsed: list[tuple[str, str] | str], phrases: list[str]) -> list[str]:
    return [c for c in content_chunks(parsed) if not is_filler_chunk(c, phrases)]


# --- 1. filler fires, then content arrives cleanly ---------------------------


async def test_filler_before_content_no_interleave() -> None:
    settings = make_settings(filler_after_seconds=0.05)
    events = [
        (0.25, delta("Paris is the capital of France.")),
        (0.02, done()),
    ]
    parsed = await run(events, settings)

    fillers = filler_chunks(parsed, settings.filler_phrases)
    reals = real_chunks(parsed, settings.filler_phrases)
    assert len(fillers) == 1
    assert "".join(reals) == sanitize_spoken("Paris is the capital of France.")
    content_order = [item for item in parsed if isinstance(item, tuple) and item[0] == "content"]
    first_real_index = next(
        i
        for i, (_, text) in enumerate(content_order)
        if not is_filler_chunk(text, settings.filler_phrases)
    )
    filler_index = next(
        i
        for i, (_, text) in enumerate(content_order)
        if is_filler_chunk(text, settings.filler_phrases)
    )
    assert filler_index < first_real_index


# --- 2. content arrives just before the deadline: filler fully suppressed ----


async def test_content_just_before_deadline_suppresses_filler() -> None:
    settings = make_settings(filler_after_seconds=0.08)
    events = [
        (0.05, delta("Quick answer.")),
        (0.02, done()),
    ]
    parsed = await run(events, settings)

    assert filler_chunks(parsed, settings.filler_phrases) == []
    assert "".join(real_chunks(parsed, settings.filler_phrases)) == sanitize_spoken("Quick answer.")


# --- 3. content arrives while the filler chunk is still "in flight" ----------


async def test_content_arrives_during_filler_emission_no_loss() -> None:
    settings = make_settings(filler_after_seconds=0.05)
    # The delta is scheduled well after the filler fires, but the *consumer* stalls
    # (simulating slow network transmission of the filler bytes) long enough that
    # the background Hermes delta task has already completed by the time the
    # consumer asks stream_turn for the next chunk.
    events = [
        (0.20, delta("Right after the filler.")),
        (0.02, done()),
    ]

    async def stall_after_filler(text: str) -> None:
        if is_filler_chunk(text, settings.filler_phrases):
            await asyncio.sleep(0.25)  # crosses the delta's t=0.20 completion

    parsed = await run(events, settings, on_content=stall_after_filler)

    fillers = filler_chunks(parsed, settings.filler_phrases)
    reals = real_chunks(parsed, settings.filler_phrases)
    assert len(fillers) == 1
    assert "".join(reals) == sanitize_spoken("Right after the filler.")
    assert "  " not in "".join(reals)  # no run-together words at the boundary


# --- 4. THE core race: tool_start must not re-arm once content has begun -----


async def test_tool_start_rearm_suppressed_once_content_started() -> None:
    settings = make_settings(filler_after_seconds=0.06)
    events = [
        (0.02, delta("Hello, ")),  # content starts well before the initial deadline
        (0.02, tool_start()),  # a buggy adapter re-arms the filler here
        (0.40, delta("world.")),  # long dead air post-tool_start
        (0.02, done()),
    ]
    parsed = await run(events, settings)

    fillers = filler_chunks(parsed, settings.filler_phrases)
    assert fillers == [], "a filler was spoken after real content had already started for this turn"
    reals = real_chunks(parsed, settings.filler_phrases)
    assert "".join(reals) == sanitize_spoken("Hello, " + "world.")


# --- 5. byte-exact: no character loss under fine-grained chunking ------------


async def test_byte_exact_zero_dropped_characters() -> None:
    settings = make_settings(filler_after_seconds=5.0)  # never fires
    full_text = "Marvin's **emergency visit** went well, the vet said `no issues` were found."
    pieces = [
        "Marvin's ",
        "**emerg",
        "ency vis",
        "it** ",
        "went well, ",
        "the vet said `no ",
        "issues` were found.",
    ]
    assert "".join(pieces) == full_text
    events: list[tuple[float, HermesTurnEvent]] = [(0.001, delta(p)) for p in pieces]
    events.append((0.001, done()))
    parsed = await run(events, settings)

    assert filler_chunks(parsed, settings.filler_phrases) == []
    reals = real_chunks(parsed, settings.filler_phrases)
    assert "".join(reals) == sanitize_spoken(full_text)


# --- 6. hard cap on total fillers per turn ------------------------------------


async def test_max_fillers_per_turn_capped() -> None:
    settings = make_settings(
        filler_after_seconds=0.05, filler_min_gap_seconds=0.01, filler_max_per_turn=1
    )
    # Three separate tool-start/dead-air cycles precede the answer -- each would
    # legitimately re-arm the timer, but the turn must still speak at most one
    # filler in total (matches the live "five fillers in a row" report).
    events = [
        (0.20, tool_start()),
        (0.20, tool_start()),
        (0.20, tool_start()),
        (0.20, delta("Finally, the answer.")),
        (0.02, done()),
    ]
    parsed = await run(events, settings)

    fillers = filler_chunks(parsed, settings.filler_phrases)
    assert len(fillers) == settings.filler_max_per_turn == 1
    reals = real_chunks(parsed, settings.filler_phrases)
    assert "".join(reals) == sanitize_spoken("Finally, the answer.")


async def test_max_fillers_per_turn_configurable_above_default() -> None:
    settings = make_settings(
        filler_after_seconds=0.05,
        filler_min_gap_seconds=0.01,
        filler_max_per_turn=2,
        filler_phrases=["One moment.", "Let me check.", "Just a second."],
    )
    events = [
        (0.20, tool_start()),
        (0.20, tool_start()),
        (0.20, tool_start()),
        (0.20, delta("Finally, the answer.")),
        (0.02, done()),
    ]
    parsed = await run(events, settings)

    assert len(filler_chunks(parsed, settings.filler_phrases)) == 2


# --- 7. minimum spacing between consecutive fillers ---------------------------


async def test_minimum_spacing_between_consecutive_fillers() -> None:
    filler_after = 0.03
    min_gap = 0.15
    settings = make_settings(
        filler_after_seconds=filler_after,
        filler_min_gap_seconds=min_gap,
        filler_max_per_turn=3,
        filler_phrases=["One moment.", "Let me check.", "Just a second.", "Hold on."],
    )
    events = [
        (0.06, tool_start()),  # dead air -> filler #1 at ~0.03
        # A dead-air opportunity too soon after filler #1 (~0.09, only ~0.06s after
        # filler #1) must be suppressed by the min-gap floor, not just re-armed and
        # spoken again.
        (0.30, tool_start()),
        (0.30, delta("Real answer.")),  # dead air -> filler #2, now past the gap
        (0.02, done()),
    ]

    timestamps: list[float] = []

    async def record_filler_time(text: str) -> None:
        if is_filler_chunk(text, settings.filler_phrases):
            timestamps.append(time.monotonic())

    parsed = await run(events, settings, on_content=record_filler_time)

    fillers = filler_chunks(parsed, settings.filler_phrases)
    assert len(fillers) == 2, "min-gap floor did not suppress a too-soon re-armed filler"
    gap = timestamps[1] - timestamps[0]
    assert gap >= min_gap - 0.03, gap


# --- 8. never repeat the same filler phrase twice within one turn -------------


async def test_filler_never_repeats_within_one_turn(monkeypatch: Any) -> None:
    import vapi_hermes_voice.speech as speech_module

    # Deterministic: always take the first surviving candidate. Combined with the
    # ["A.", "B.", "C."] pool this forces pick #3 to repeat pick #1 unless the
    # picker excludes everything already used this turn (not just the immediately
    # previous pick).
    monkeypatch.setattr(speech_module.random, "choice", lambda seq: seq[0])

    settings = make_settings(
        filler_after_seconds=0.03,
        filler_min_gap_seconds=0.01,
        filler_max_per_turn=3,
        filler_phrases=["A.", "B.", "C."],
    )
    events = [
        (0.20, tool_start()),
        (0.20, tool_start()),
        (0.20, delta("Done.")),
        (0.02, done()),
    ]
    parsed = await run(events, settings)

    fillers = filler_chunks(parsed, settings.filler_phrases)
    assert len(fillers) == 3
    picked = {c.strip().split(" <flush")[0].strip() for c in fillers}
    assert picked == {"A.", "B.", "C."}, "the same filler phrase was spoken twice in one turn"


# --- 9. <flush /> is only ever a trailing control token on a filler chunk ----


async def test_flush_token_never_leaks_detached_from_filler() -> None:
    settings = make_settings(
        filler_after_seconds=0.05,
        filler_min_gap_seconds=0.01,
        filler_max_per_turn=2,
        filler_use_flush=True,
        filler_phrases=["One moment.", "Let me check."],
    )
    events = [
        (0.20, tool_start()),
        (0.20, delta("The vet gave Marvin a clean bill of health.")),
        (0.02, done()),
    ]
    parsed = await run(events, settings)

    fillers = filler_chunks(parsed, settings.filler_phrases)
    reals = real_chunks(parsed, settings.filler_phrases)
    assert len(fillers) == 2
    phrase_alternation = "|".join(re.escape(p) for p in settings.filler_phrases)
    for chunk in fillers:
        match = re.fullmatch(rf"(?:{phrase_alternation}) <flush /> ", chunk)
        assert match is not None, f"flush token detached from a full filler phrase: {chunk!r}"
    for chunk in reals:
        assert "<flush" not in chunk, f"flush control token leaked into real content: {chunk!r}"


async def test_flush_token_absent_when_disabled_even_with_multiple_fillers() -> None:
    settings = make_settings(
        filler_after_seconds=0.05,
        filler_min_gap_seconds=0.01,
        filler_max_per_turn=2,
        filler_use_flush=False,
        filler_phrases=["One moment.", "Let me check."],
    )
    events = [
        (0.20, tool_start()),
        (0.20, delta("All set.")),
        (0.02, done()),
    ]
    parsed = await run(events, settings)

    assert len(filler_chunks(parsed, settings.filler_phrases)) == 2
    assert "<flush" not in "".join(content_chunks(parsed))


# --- 10. sanitizer buffer is flushed before an error apology -----------------


async def test_error_path_flushes_pending_sanitizer_buffer() -> None:
    settings = make_settings(filler_after_seconds=5.0)
    events = [
        (0.001, delta("Hello **wor")),  # "wor" is held back: unresolved bold span
        (0.001, error("Sorry, that's taking too long.")),
    ]
    parsed = await run(events, settings)

    joined = "".join(content_chunks(parsed))
    assert "wor" in joined, "buffered text was silently dropped on the error path"
    assert joined == "Hello wor Sorry, that's taking too long."


# --- 11. the acceptance UX contract: fast first filler, then a real gap ------


async def test_first_filler_lands_fast_then_respects_min_gap() -> None:
    """First filler within ~filler_after_seconds; a second only after the full gap.

    Uses production-representative numbers (matches the recommended deployment
    defaults): the caller must hear a holding line within about a second, then
    genuine silence for the configured minimum gap before another one -- even
    though a Hermes turn can legitimately keep re-arming the timer for many
    seconds (an 18 s search, per the live report) before it has an answer.
    """
    filler_after = 1.2
    min_gap = 8.0
    settings = make_settings(
        filler_after_seconds=filler_after,
        filler_min_gap_seconds=min_gap,
        filler_max_per_turn=2,
        filler_phrases=["One moment, let me check.", "Bear with me one second."],
    )
    events = [
        (2.5, tool_start()),  # re-arm opportunity well inside the min-gap window
        (7.0, tool_start()),  # another dead-air window; still (barely) too soon
        (5.5, delta("Real answer.")),  # content finally arrives at t ~= 15s
        (0.05, done()),
    ]

    start = time.monotonic()
    timestamps: list[float] = []

    async def record_filler_time(text: str) -> None:
        if is_filler_chunk(text, settings.filler_phrases):
            timestamps.append(time.monotonic() - start)

    parsed = await run(events, settings, on_content=record_filler_time)

    fillers = filler_chunks(parsed, settings.filler_phrases)
    reals = real_chunks(parsed, settings.filler_phrases)
    assert len(fillers) == 2
    assert len(timestamps) == 2

    first_offset, second_offset = timestamps
    tolerance = 0.4
    assert abs(first_offset - filler_after) <= tolerance, (
        f"first filler landed at {first_offset:.2f}s, expected ~{filler_after}s"
    )
    assert second_offset >= first_offset + min_gap - tolerance, (
        f"second filler at {second_offset:.2f}s violated the "
        f"{min_gap}s minimum gap since the first filler at {first_offset:.2f}s"
    )
    assert "".join(reals) == sanitize_spoken("Real answer.")


# --- 12. min-gap/cap must never leak across turns of the same call -----------


async def test_filler_timing_is_independent_across_sequential_turns() -> None:
    """Two sequential turns of the *same* call must each get their own fast filler.

    Live report: the caller heard ~9s of true silence at the start of a new
    question, consistent with a hypothesis that the min-gap/cap state from the
    previous turn's filler was leaking into the next one. CallStateRegistry hands
    server.py the *same* CallState object on every turn of a call (only the
    session ids and the FillerPicker's phrase-repeat memory are meant to persist
    across turns) -- this drives stream_turn twice against that same CallState,
    back to back, and asserts turn 2's first filler lands at ~filler_after_seconds
    regardless of when turn 1's filler fired, not blocked until filler_min_gap_seconds
    after it.
    """
    filler_after = 0.06
    min_gap = 5.0  # much larger than filler_after: a real leak would be obvious
    settings = make_settings(
        filler_after_seconds=filler_after,
        filler_min_gap_seconds=min_gap,
        filler_max_per_turn=1,
    )
    shared_state = make_state(settings)

    async def timed_turn(label: str) -> tuple[list[tuple[str, str] | str], list[float]]:
        start = time.monotonic()
        timestamps: list[float] = []

        async def record(text: str) -> None:
            if is_filler_chunk(text, settings.filler_phrases):
                timestamps.append(time.monotonic() - start)

        events = [
            (0.30, delta(f"{label} answer.")),
            (0.02, done()),
        ]
        parsed = await run(events, settings, on_content=record, state=shared_state)
        return parsed, timestamps

    parsed1, ts1 = await timed_turn("Turn one")
    parsed2, ts2 = await timed_turn("Turn two")  # starts immediately after turn 1 ends

    assert len(filler_chunks(parsed1, settings.filler_phrases)) == 1
    assert len(filler_chunks(parsed2, settings.filler_phrases)) == 1
    assert len(ts1) == 1 and len(ts2) == 1
    tolerance = 0.35
    assert abs(ts1[0] - filler_after) <= tolerance, ts1
    assert abs(ts2[0] - filler_after) <= tolerance, (
        f"turn 2's filler landed at {ts2[0]:.2f}s (expected ~{filler_after}s); "
        "min-gap/cap state leaked in from turn 1"
    )
    assert "".join(real_chunks(parsed1, settings.filler_phrases)) == sanitize_spoken(
        "Turn one answer."
    )
    assert "".join(real_chunks(parsed2, settings.filler_phrases)) == sanitize_spoken(
        "Turn two answer."
    )


async def test_filler_max_per_turn_resets_across_sequential_turns() -> None:
    """filler_max_per_turn is a per-turn budget, not a per-call one."""
    settings = make_settings(
        filler_after_seconds=0.05,
        filler_min_gap_seconds=0.01,
        filler_max_per_turn=1,
    )
    shared_state = make_state(settings)

    events = [(0.30, delta("First answer.")), (0.02, done())]
    parsed1 = await run(events, settings, state=shared_state)
    assert len(filler_chunks(parsed1, settings.filler_phrases)) == 1

    events2 = [(0.30, delta("Second answer.")), (0.02, done())]
    parsed2 = await run(events2, settings, state=shared_state)
    assert len(filler_chunks(parsed2, settings.filler_phrases)) == 1, (
        "turn 2's own filler budget was suppressed by turn 1 having already used its one filler"
    )


# --- 13. filler text is emitted byte-exact, even as the very first turn write --


async def test_filler_text_is_byte_exact_verbatim_as_first_write_of_turn() -> None:
    """The emitted filler content must equal the configured phrase exactly.

    Live report: a filler phrase lost its leading character when it was the very
    first thing spoken in a turn (e.g. "I have that information..." arrived as
    "have that information..."). This asserts the raw SSE content for the filler
    chunk -- stripped only of the trailing " " and optional " <flush />" suffix
    _filler_text() itself appends -- is character-for-character identical to one
    of the configured phrases, for every phrase in the pool and with flush both
    enabled and disabled, specifically as the first content chunk of the turn.
    """
    phrases = [
        "I have that information right here, give me a second.",
        "Let me pull that up for you.",
        "One moment while I check.",
    ]
    for use_flush in (True, False):
        settings = make_settings(
            filler_after_seconds=0.03,
            filler_min_gap_seconds=0.01,
            filler_max_per_turn=1,
            filler_use_flush=use_flush,
            filler_phrases=phrases,
        )
        events = [(0.3, delta("Answer.")), (0.02, done())]
        parsed = await run(events, settings)

        content_items = [
            item for item in parsed if isinstance(item, tuple) and item[0] == "content"
        ]
        assert content_items, "no content chunk was emitted"
        first_text = content_items[0][1]
        assert is_filler_chunk(first_text, phrases), (
            f"first write of the turn was not a filler: {first_text!r}"
        )

        raw = first_text
        if use_flush:
            assert raw.endswith(" <flush /> ")
            raw = raw[: -len(" <flush /> ")]
        else:
            assert raw.endswith(" ")
            raw = raw[:-1]
        assert raw in phrases, (
            f"filler text {raw!r} does not match any configured phrase verbatim "
            f"(use_flush={use_flush}) -- leading/trailing characters were altered"
        )


# --- 14. a filler phrase + its <flush/> token is one atomic SSE write --------


async def test_filler_and_flush_token_are_one_atomic_sse_frame() -> None:
    """A filler phrase plus its ``<flush />`` token can never be split.

    Live report: Vapi played one filler phrase as two separate spoken segments
    ten seconds apart ("Give me a moment to" / "find that. Hold on. ..."). This
    proves -- at the raw SSE wire level, not just the parsed content -- that
    stream_turn always writes a filler's phrase and its flush token together in
    exactly one ``data: ...`` frame from exactly one ``yield``: it is impossible
    for our own emission to be split across two writes (network-level chunking
    downstream of us is a separate question this test cannot answer).
    """
    settings = make_settings(
        filler_after_seconds=0.04,
        filler_min_gap_seconds=0.01,
        filler_max_per_turn=3,
        filler_use_flush=True,
        filler_phrases=["Give me a moment to find that.", "Hold on.", "Checking that for you."],
    )
    events: list[tuple[float, HermesTurnEvent]] = [
        (0.20, tool_start()),
        (0.20, tool_start()),
        (0.20, delta("Found it.")),
        (0.02, done()),
    ]
    reaping: set[asyncio.Task[None]] = set()
    raw_chunks: list[str] = []
    async for chunk in stream_turn(
        settings=settings,
        hermes=_ScriptedHermes(events),
        state=make_state(settings),
        instructions="instructions",
        history=[],
        user_input="hello",
        reaping=reaping,
    ):
        raw_chunks.append(chunk)
    for task in list(reaping):
        with contextlib.suppress(Exception):
            await task

    filler_frames = [c for c in raw_chunks if any(p in c for p in settings.filler_phrases)]
    assert len(filler_frames) == settings.filler_max_per_turn == 3
    for frame in filler_frames:
        # Exactly one SSE "data:" line -- one yield, one frame -- per filler.
        assert frame.count("data: ") == 1, frame
        # The phrase and its flush token are both present in that same frame.
        matching_phrase = next(p for p in settings.filler_phrases if p in frame)
        assert matching_phrase in frame
        assert "<flush />" in frame
        # Never more than one flush token in a single filler's own frame.
        assert frame.count("<flush") == 1, frame
    # No filler phrase is ever fragmented: none of it appears in any *other*
    # chunk (role/finish/done/real-content) besides its own single frame.
    other_chunks = [c for c in raw_chunks if c not in filler_frames]
    for other in other_chunks:
        for phrase in settings.filler_phrases:
            assert phrase not in other, f"filler text leaked into a non-filler chunk: {other!r}"
