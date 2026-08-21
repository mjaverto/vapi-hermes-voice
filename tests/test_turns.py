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
) -> list[tuple[str, str] | str]:
    """Drive stream_turn to completion; return parsed (kind, text)/"[DONE]" items.

    ``on_content`` (if given) is awaited with the just-yielded content text right
    after each content chunk is received by this "consumer" -- used to simulate the
    consumer being busy (e.g. still writing a chunk to the socket) while a Hermes
    event completes concurrently in the background.
    """
    reaping: set[asyncio.Task[None]] = set()
    parsed: list[tuple[str, str] | str] = []
    agen = stream_turn(
        settings=settings,
        hermes=_ScriptedHermes(events),
        state=make_state(settings),
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
