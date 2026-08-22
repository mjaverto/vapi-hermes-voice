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
        # Disjoint from every filler pool used anywhere in these tests
        # (Settings._check_pools_are_disjoint refuses an overlap) and, at the default
        # reassure_after_seconds, unreachable inside a sub-second test turn: tests that
        # want a reassurance set reassure_after_seconds explicitly.
        "reassure_phrases": ["Nearly done.", "Not long now.", "Still on it."],
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
        reassure=FillerPicker(settings.reassure_phrases),
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

    Every turn now gets an acknowledgement opportunity: there is no per-turn opt-out
    left, so suppression is only ever the ``content_started`` gate or the call-global
    cooldown on ``state``.
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


# --- 6. the cooldown, not a count, bounds acknowledgements within one turn -----


async def test_cooldown_bounds_acknowledgements_within_one_turn() -> None:
    settings = make_settings(filler_after_seconds=0.05, filler_min_gap_seconds=5.0)
    # Three separate tool-start/dead-air cycles precede the answer -- each would
    # legitimately re-arm the timer, but the cooldown is far longer than the whole
    # turn, so exactly one line is spoken (the live "five fillers in a row" report).
    events = [
        (0.20, tool_start()),
        (0.20, tool_start()),
        (0.20, tool_start()),
        (0.20, delta("Finally, the answer.")),
        (0.02, done()),
    ]
    parsed = await run(events, settings)

    fillers = filler_chunks(parsed, settings.filler_phrases)
    assert len(fillers) == 1
    reals = real_chunks(parsed, settings.filler_phrases)
    assert "".join(reals) == sanitize_spoken("Finally, the answer.")


async def test_long_turn_outliving_the_cooldown_gets_a_second_acknowledgement() -> None:
    """A duration, not a count: crossing the cooldown mid-turn earns another line.

    The retired ``filler_max_per_turn=1`` forbade this outright, so a genuinely
    long multi-tool turn went silent for the rest of its life however long it ran.
    The requirement is a 10 s cooldown, so a turn that spans two cooldown windows
    must be allowed two acknowledgements.
    """
    settings = make_settings(filler_after_seconds=0.05, filler_min_gap_seconds=0.30)
    events = [
        (0.20, tool_start()),  # dead air -> ack #1 at ~0.05
        (0.20, tool_start()),  # ~0.25s in: inside the cooldown, must stay silent
        (0.40, delta("Finally, the answer.")),  # dead air past 0.30s -> ack #2
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
    """First line within ~filler_after_seconds; a second only after the full gap.

    Uses the shipped defaults verbatim: the callee must hear an acknowledgement
    inside the two-second requirement, then genuine silence for the whole
    call-global cooldown before another one -- even though a Hermes turn can
    legitimately keep re-arming the timer for many seconds (an 18 s search, per the
    live report) before it has an answer.
    """
    filler_after = 0.9
    min_gap = 10.0
    settings = make_settings(
        filler_after_seconds=filler_after,
        filler_min_gap_seconds=min_gap,
        filler_phrases=["Okay, let me check.", "Bear with me one second."],
    )
    events = [
        (2.5, tool_start()),  # re-arm opportunity well inside the cooldown window
        (8.5, tool_start()),  # t=11s: the cooldown has now expired
        (4.0, delta("Real answer.")),  # content finally arrives at t ~= 15s
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


# --- 12. the cooldown is GLOBAL to the call, i.e. it DOES cross turns ---------


async def test_second_turn_inside_the_cooldown_gets_no_acknowledgement() -> None:
    """Turn 2, three seconds after turn 1's line, must be silent.

    The requirement is a ten-second cooldown global to the call: "she must then NOT
    say that or anything like it for at least 10 seconds, and that cooldown is
    global to the call, not per-turn". A turn-local timestamp -- which is what
    ``stream_turn`` used to keep -- is reinitialised by every HTTP POST and so only
    ever spaced out two lines inside a single turn; turn N+1 fired its own line
    seconds after turn N regardless. CallStateRegistry hands server.py the *same*
    CallState on every turn of a call, so this drives stream_turn twice against one
    CallState and asserts the second turn stays quiet.
    """
    settings = make_settings(filler_after_seconds=0.05, filler_min_gap_seconds=10.0)
    shared_state = make_state(settings)

    turn_one = [(0.30, delta("Turn one answer.")), (0.02, done())]
    first = await run(turn_one, settings, state=shared_state)
    assert len(filler_chunks(first, settings.filler_phrases)) == 1
    acked_at = shared_state.last_ack_at
    assert acked_at is not None

    await asyncio.sleep(0.05)  # turn 2 begins well inside the 10 s cooldown
    turn_two = [(0.30, delta("Turn two answer.")), (0.02, done())]
    second = await run(turn_two, settings, state=shared_state)

    assert filler_chunks(second, settings.filler_phrases) == [], (
        "turn 2 spoke an acknowledgement inside the call-global cooldown"
    )
    assert shared_state.last_ack_at == acked_at, "a refused claim moved the cooldown anchor"
    # The answer itself is never affected by the cooldown.
    assert "".join(real_chunks(second, settings.filler_phrases)) == sanitize_spoken(
        "Turn two answer."
    )


async def test_turn_after_the_cooldown_expires_gets_an_acknowledgement() -> None:
    """The other half of the contract: once the gap has passed, speak again."""
    settings = make_settings(filler_after_seconds=0.05, filler_min_gap_seconds=0.40)
    shared_state = make_state(settings)

    turn_one = [(0.30, delta("Turn one answer.")), (0.02, done())]
    first = await run(turn_one, settings, state=shared_state)
    assert len(filler_chunks(first, settings.filler_phrases)) == 1

    # Turn 1's line landed at ~0.05s and the turn ran ~0.32s, so waiting out the
    # remainder of the 0.40s cooldown puts turn 2 legitimately past it.
    await asyncio.sleep(0.25)
    turn_two = [(0.30, delta("Turn two answer.")), (0.02, done())]
    second = await run(turn_two, settings, state=shared_state)

    assert len(filler_chunks(second, settings.filler_phrases)) == 1, (
        "the cooldown had expired but turn 2 still stayed silent"
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
        "Okay, let me check.",
        "Got it, one second.",
        "Understood, one moment.",
    ]
    for use_flush in (True, False):
        settings = make_settings(
            filler_after_seconds=0.03,
            filler_min_gap_seconds=0.01,
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
    assert len(filler_frames) == 3
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


# --- 15. R2: an acknowledgement on EVERY turn, including the first -------------
#
# The opening turn used to be suppressed outright (server.py passed
# allow_fillers=not opening_turn). Live report, in the callee's own words: "When it
# first calls, I pick up, and then it's like 10 seconds before it says anything."
# The first turn is precisely where the dead air hurt most, so the suppression is
# gone and there is no per-turn opt-out left at all.


async def test_first_turn_of_a_call_speaks_an_acknowledgement() -> None:
    settings = make_settings(filler_after_seconds=0.05, filler_min_gap_seconds=10.0)
    state = make_state(settings)
    assert state.last_ack_at is None, "a brand-new call must start with no cooldown"
    events = [
        (0.30, delta("Hi, I'm calling on behalf of Mike.")),
        (0.0, done()),
    ]
    parsed = await run(events, settings, state=state)

    assert len(filler_chunks(parsed, settings.filler_phrases)) == 1, (
        "the first turn of the call was silent through 0.3s of dead air"
    )
    assert state.last_ack_at is not None
    assert "".join(real_chunks(parsed, settings.filler_phrases)).strip() == (
        "Hi, I'm calling on behalf of Mike."
    )


async def test_acknowledgement_fires_within_two_seconds_on_a_turn_with_no_tool_activity() -> None:
    """The R2 headline, asserted with the shipped defaults and zero tool events.

    Two things at once, because both were broken. (a) No tool activity: the callee
    wants an answer to having been spoken to, so gating this on a tool run would
    leave every plain conversational turn -- "okay, thanks, that's all I needed" --
    in silence for the whole Hermes round trip. (b) Inside two seconds: measured
    against the real default rather than a scaled-down test value, because the
    number itself is the fix. The deployed value was 2.5 s, which broke the ceiling
    before Vapi had spent any of it; Vapi's endpointing spends ~0.4-1.6 s before
    this process is invoked at all, so the adapter's share must be under a second.
    """
    default_after = Settings.model_fields["filler_after_seconds"].default
    default_gap = Settings.model_fields["filler_min_gap_seconds"].default
    settings = make_settings(filler_after_seconds=default_after, filler_min_gap_seconds=default_gap)
    assert settings.filler_after_seconds <= 1.0, "default no longer fits the 2 s budget"
    events = [(3.0, delta("Eventually.")), (0.0, done())]
    assert not any(event.kind == "tool_start" for _, event in events)

    start = time.monotonic()
    offsets: list[float] = []

    async def record(text: str) -> None:
        if is_filler_chunk(text, settings.filler_phrases):
            offsets.append(time.monotonic() - start)

    parsed = await run(events, settings, on_content=record)

    assert len(filler_chunks(parsed, settings.filler_phrases)) == 1
    assert offsets[0] < 2.0, f"acknowledgement landed at {offsets[0]:.2f}s, past the 2 s ceiling"


async def test_r2_deadline_ack_matches_only_the_configured_phrase_pool_no_tool_claim() -> None:
    """R2 companion: the acknowledgement is byte-exact from the pool, never a claim.

    ``test_acknowledgement_fires_within_two_seconds_on_a_turn_with_no_tool_activity``
    (above) proves the 2s ceiling; this proves the SPOKEN TEXT on a turn with no
    tool activity is one of the configured, generic phrases and never a
    tool-specific claim ("let me look that up", "I have that information right
    here") -- the live complaint was exactly a filler that claimed a lookup was
    happening on a call where nothing had been looked up yet.
    """
    default_after = Settings.model_fields["filler_after_seconds"].default
    default_gap = Settings.model_fields["filler_min_gap_seconds"].default
    settings = make_settings(filler_after_seconds=default_after, filler_min_gap_seconds=default_gap)
    assert settings.filler_after_seconds <= 2.0, "production default no longer fits the R2 budget"

    events = [(default_after + 0.15, done())]  # silent past the deadline, then nothing to add
    start = time.monotonic()
    offsets: list[float] = []

    async def record(text: str) -> None:
        if is_filler_chunk(text, settings.filler_phrases):
            offsets.append(time.monotonic() - start)

    parsed = await run(events, settings, on_content=record)

    fillers = filler_chunks(parsed, settings.filler_phrases)
    assert len(fillers) == 1
    assert offsets and offsets[0] <= 2.0, (
        f"acknowledgement landed at {offsets}, past the 2s ceiling"
    )

    raw = fillers[0].strip()
    if settings.filler_use_flush:
        assert raw.endswith("<flush />")
        raw = raw[: -len("<flush />")].strip()
    else:
        raw = raw.strip()
    assert raw in settings.filler_phrases, (
        f"acknowledgement {raw!r} does not match a configured phrase verbatim -- it "
        "may be a tool-specific claim rather than a generic acknowledgement"
    )
    for tool_claim in ("look", "found", "search", "information right here", "checking"):
        assert tool_claim not in raw.casefold(), (
            f"acknowledgement falsely claims tool/lookup activity: {raw!r}"
        )


async def test_error_path_still_works_on_the_first_turn() -> None:
    settings = make_settings(filler_after_seconds=0.02, filler_min_gap_seconds=10.0)
    parsed = await run([(0.10, error("Sorry, something went wrong."))], settings)
    assert len(filler_chunks(parsed, settings.filler_phrases)) == 1
    assert "Sorry, something went wrong." in "".join(content_chunks(parsed))
    assert parsed[-1] == "[DONE]"


# --- 16. R2: a barge-in retry storm produces exactly one acknowledgement -------


async def _cancelled_storm(
    settings: Settings,
    state: CallState,
    *,
    attempts: int = 6,
    abandon_after: float = 0.15,
    between: float = 0.02,
) -> list[float]:
    """Replay a barge-in retry storm; return the offset of every line spoken.

    Each attempt is a real ``stream_turn`` against a Hermes run that never answers,
    which Vapi abandons ``abandon_after`` seconds in -- the observed pattern, where
    five of six streams were torn down mid-flight. Every attempt shares ``state``.
    """
    reaping: set[asyncio.Task[None]] = set()
    start = time.monotonic()
    offsets: list[float] = []
    for _ in range(attempts):
        agen = stream_turn(
            settings=settings,
            hermes=_ScriptedHermes([(30.0, delta("never arrives")), (0.0, done())]),
            state=state,
            instructions="instructions",
            history=[],
            user_input="are you there?",
            reaping=reaping,
        )
        try:
            async with asyncio.timeout(abandon_after):  # Vapi hangs up on this stream
                async for chunk in agen:
                    for item in _parse_chunk(chunk):
                        if (
                            isinstance(item, tuple)
                            and item[0] == "content"
                            and is_filler_chunk(item[1], settings.filler_phrases)
                        ):
                            offsets.append(time.monotonic() - start)
        except TimeoutError:
            pass
        finally:
            await agen.aclose()
        await asyncio.sleep(between)  # Vapi's next barge-in retry
    for task in list(reaping):
        with contextlib.suppress(Exception):
            await task
    return offsets


async def test_retry_storm_inside_one_cooldown_speaks_exactly_one_acknowledgement() -> None:
    """Six rapid attempts on one call, five cancelled mid-stream: one line total.

    Live report: Vapi barge-in re-POSTed one turn SIX times inside sixteen seconds,
    cancelling five of the streams mid-flight. Every attempt shares one CallState,
    so the cooldown must survive both the turn boundary and the cancellation --
    including for the attempt whose line went into a stream Vapi then destroyed,
    which still counts as spoken because the callee most likely heard it. Here the
    whole storm fits inside one cooldown window, so the callee hears one line.
    """
    settings = make_settings(filler_after_seconds=0.05, filler_min_gap_seconds=2.0)
    shared_state = make_state(settings)

    offsets = await _cancelled_storm(settings, shared_state)

    assert len(offsets) == 1, f"a retry storm produced {len(offsets)} acknowledgements"
    assert offsets[0] < settings.filler_min_gap_seconds


async def test_retry_storm_outliving_the_cooldown_never_speaks_once_per_attempt() -> None:
    """A storm longer than the cooldown is still paced by it, not by attempt count.

    The cooldown is a duration, so a storm that outlives it legitimately earns
    another line -- but never one per attempt, and never two closer together than
    the gap. This is the assertion that would have caught the live "five holding
    lines in a row" behaviour whatever the retry timing happened to be.
    """
    settings = make_settings(filler_after_seconds=0.05, filler_min_gap_seconds=0.30)
    shared_state = make_state(settings)

    offsets = await _cancelled_storm(settings, shared_state)

    assert len(offsets) < 6, f"the storm spoke one line per attempt: {offsets!r}"
    gaps = [later - earlier for earlier, later in zip(offsets, offsets[1:], strict=False)]
    assert all(gap >= settings.filler_min_gap_seconds - 0.05 for gap in gaps), gaps


# --- 17. R2: content that has already started forbids one, and costs nothing ---


async def test_no_acknowledgement_once_content_started_and_cooldown_untouched() -> None:
    """The R1 fast path case: a sub-500ms local reason line, then a slow Hermes run.

    On the first outbound turn the adapter speaks the reason for calling itself,
    with no Hermes round trip, so content starts long before the acknowledgement
    timer would fire. Two things must hold: nothing is spoken ("okay, let me check"
    in front of "hi, I'm calling about..." is nonsense), and the call-global
    cooldown must NOT be spent by a turn that never spoke -- otherwise the callee's
    *next* utterance, the one that genuinely needs an acknowledgement, is silenced
    by a slot nobody used.
    """
    settings = make_settings(filler_after_seconds=0.05, filler_min_gap_seconds=10.0)
    state = make_state(settings)
    events = [
        (0.01, delta("Hi, I'm calling about a medical follow-up for Mike.")),
        (0.50, tool_start()),  # a slow tool run follows: still no acknowledgement
        (0.50, delta(" One moment while I confirm.")),
        (0.02, done()),
    ]
    parsed = await run(events, settings, state=state)

    assert filler_chunks(parsed, settings.filler_phrases) == []
    assert state.last_ack_at is None, (
        "a turn that spoke no acknowledgement consumed the call-global cooldown anyway"
    )

    # Proof that the unspent slot is still available to the next turn.
    follow_up = await run([(0.40, delta("Confirmed."))], settings, state=state)
    assert len(filler_chunks(follow_up, settings.filler_phrases)) == 1


# --- 18. R2: NO acknowledgement once content has begun streaming, ever ---------


async def test_r2_no_ack_after_content_started_even_with_a_tool_start_and_long_gap_after() -> None:
    """Once real content has begun, no acknowledgement is ever spoken for that turn.

    Two independent, redundant guards make this true, and this test's mutation
    below defeats them together: (1) a ``tool_start`` after content has begun must
    not re-arm ``filler_deadline`` (``turns.py``'s ``elif turn_event.kind ==
    "tool_start": if not content_started: ...``), and (2), even if the deadline
    were re-armed, the dead-air branch's own ``if not content_started:`` check
    (right before calling ``_filler_text``) refuses to speak. Both must hold for
    "no ack after content, ever" to be true, so this scenario -- content, then a
    tool_start, then a long gap -- is the one shape that actually reaches either
    guard: with no tool_start at all, ``filler_deadline`` stays ``None`` forever
    once content starts and the dead-air branch is architecturally unreachable, so
    a plain post-content silence proves nothing.
    """
    settings = make_settings(filler_after_seconds=0.03, filler_min_gap_seconds=0.01)
    events = [
        (0.01, delta("The vet said")),  # content starts
        (0.05, tool_start()),  # a buggy adapter could re-arm the filler here
        (0.30, delta(" no issues were found.")),  # dead air past the tool_start
        (0.02, done()),
    ]
    parsed = await run(events, settings)

    assert filler_chunks(parsed, settings.filler_phrases) == [], (
        "an acknowledgement was spoken after content had already started for the turn"
    )
    assert "".join(real_chunks(parsed, settings.filler_phrases)) == sanitize_spoken(
        "The vet said no issues were found."
    )
