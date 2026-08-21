"""Per-turn orchestration: one Vapi POST -> one Hermes run -> one OpenAI SSE stream.

Cleanup contract: whatever happens to the response -- completion, timeout, client
disconnect, task cancellation -- the underlying Hermes run is stopped. Abandoning the
Hermes events stream without ``POST /v1/runs/{id}/stop`` leaves the run executing
unboundedly (docs/integration-contracts.md section 2), so finalization never depends
on the (possibly already cancelled) request task: cleanup runs on a fresh background
task registered in ``reaping``, which the app lifespan drains on shutdown.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator
from typing import cast

from .call_state import CallState
from .config import Settings
from .hermes_client import HermesClient, HermesTurnEvent
from .speech import DeltaSanitizer, sanitize_spoken
from .vapi_events import ChunkWriter

logger = logging.getLogger(__name__)

APOLOGY_LINE = "Sorry, I'm having trouble right now. Could you say that again?"


def _reap(reaping: set[asyncio.Task[None]], task: asyncio.Task[None]) -> None:
    """Track a cleanup task until it finishes; log (never raise) on failure."""
    reaping.add(task)

    def _done(finished: asyncio.Task[None]) -> None:
        reaping.discard(finished)
        if finished.cancelled():
            return
        exc = finished.exception()
        if exc is not None:
            logger.warning("turn cleanup failed error=%s", type(exc).__name__)

    task.add_done_callback(_done)


async def _cleanup(
    agen: AsyncGenerator[HermesTurnEvent, None],
    next_task: asyncio.Task[HermesTurnEvent] | None,
) -> None:
    """Cancel a pending event fetch and close the Hermes turn (delivering the stop)."""
    if next_task is not None and not next_task.done():
        next_task.cancel()
    if next_task is not None:
        with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration, Exception):
            await next_task
    with contextlib.suppress(Exception):
        await agen.aclose()


def _filler_text(state: CallState, settings: Settings, used_this_turn: set[str]) -> str:
    """Build one complete filler phrase, with its ``<flush />`` token if enabled.

    Returns a single string: the phrase plus, when enabled, exactly one trailing
    `` <flush />`` token. This is passed to a single ``writer.content(...)`` call
    (one ``yield``, one SSE ``data:`` frame) at its one call site in ``stream_turn``
    -- it is never split across multiple deltas, never fed through DeltaSanitizer
    (which could otherwise hold part of it back across a chunk boundary), and never
    concatenated with anything else before being written. A filler phrase and its
    flush token are therefore always one atomic write from this adapter's side.
    """
    text = state.filler.pick(exclude=used_this_turn)
    used_this_turn.add(text)
    if settings.filler_use_flush:
        # <flush /> forces immediate TTS transmission (contracts section 1.6).
        text += " <flush />"
    return text + " "


async def stream_turn(
    *,
    settings: Settings,
    hermes: HermesClient,
    state: CallState,
    instructions: str,
    history: list[dict[str, str]],
    user_input: str,
    reaping: set[asyncio.Task[None]],
    allow_fillers: bool = True,
) -> AsyncIterator[str]:
    """Drive one voice turn, yielding OpenAI SSE lines (role, content*, finish, DONE).

    Filler phrases race the first Hermes event: after ``filler_after_seconds`` of
    dead air a non-repeating holding line is spoken, and the timer re-arms on tool
    starts (each Hermes tool round trip adds ~2.9 s of silence) so a long multi-tool
    run can still get a fresh holding line before it has anything to say. Three hard
    limits keep that from sounding robotic or, worse, colliding with the answer --
    a filler is only ever spoken when ALL THREE allow it:

    - ``content_started`` is a single-owner flag flipped the instant the turn's
      first ``delta`` event arrives (asyncio is single-threaded and the flag is
      only ever read/written inside this synchronous loop body, never across an
      ``await``, so the check-then-set is atomic by construction -- the same
      reasoning ``CallStateRegistry`` relies on for its own check-then-mutate
      state). Once set, no later ``tool_start`` may re-arm the filler deadline,
      and the dead-air branch re-checks the same flag before speaking: a filler
      can never be spoken once the answer has begun.
    - ``filler_max_per_turn`` caps the *total* holding lines for one turn
      regardless of how many tool-start/dead-air cycles precede the answer, and
      each pick excludes every phrase already used this turn so a caller never
      hears the same line twice in one turn.
    - ``filler_min_gap_seconds`` is a structural floor between the end of one
      filler and the start of the next, re-checked at the moment a filler would
      be spoken (not just when the deadline is armed), so no ordering of
      tool-start re-arms can produce two fillers closer together than this gap.

    Hermes ``error`` events are already safe generic messages by the client's
    contract; they are spoken as ordinary content -- after flushing whatever text
    the sanitizer was still holding back -- so the caller hears an apology instead
    of Vapi surfacing a platform error, and no already-buffered words are lost.

    ``allow_fillers=False`` disables holding lines outright for this turn. The
    synthetic opening turn uses it: nothing is pending there (the callee has not
    spoken), so "give me a second" is nonsense and front-loads the call with noise
    before the greeting. The deadline is never armed and never re-armed, so no
    amount of Hermes latency can produce one.
    """
    writer = ChunkWriter()
    received_at = time.monotonic()
    loop = asyncio.get_running_loop()
    outcome = "server_error"
    ttfb_ms: int | None = None
    agen = cast(
        AsyncGenerator[HermesTurnEvent, None],
        hermes.run_turn(
            session_id=state.session_id,
            session_key=state.session_key,
            instructions=instructions,
            conversation_history=history,
            user_input=user_input,
        ),
    )
    next_task: asyncio.Task[HermesTurnEvent] | None = None
    try:
        yield writer.role()
        sanitizer = DeltaSanitizer()
        filler_after = settings.filler_after_seconds
        filler_min_gap = settings.filler_min_gap_seconds
        filler_deadline: float | None = loop.time() + filler_after if allow_fillers else None
        last_filler_at: float | None = None  # last filler emission only, not content
        emitted_delta = False
        content_started = False  # single-owner: set once, forever forbids new fillers
        filler_count = 0
        filler_used_this_turn: set[str] = set()
        while True:
            if next_task is None:
                next_task = asyncio.ensure_future(anext(agen))
            timeout: float | None = None
            if filler_deadline is not None:
                timeout = max(0.0, filler_deadline - loop.time())
            done, _pending = await asyncio.wait({next_task}, timeout=timeout)
            now = loop.time()
            if not done:
                # Dead air: the filler window elapsed before the next Hermes event.
                filler_deadline = None  # re-armed only by a later tool_start
                if (
                    allow_fillers
                    and not content_started
                    and filler_count < settings.filler_max_per_turn
                    and (last_filler_at is None or now - last_filler_at >= filler_min_gap)
                ):
                    yield writer.content(_filler_text(state, settings, filler_used_this_turn))
                    filler_count += 1
                    logger.info(
                        "turn filler call=%s elapsed_ms=%d count=%d",
                        state.call_ref,
                        int((time.monotonic() - received_at) * 1000),
                        filler_count,
                    )
                    last_filler_at = now
                continue
            try:
                turn_event = next_task.result()
            except StopAsyncIteration:
                turn_event = HermesTurnEvent(kind="done")
            next_task = None
            if turn_event.kind == "delta":
                # Content has begun: cancel any pending filler atomically and, via
                # content_started, permanently forbid future re-arms below.
                filler_deadline = None
                content_started = True
                text = sanitizer.feed(turn_event.text)
                if text:
                    yield writer.content(text)
                    if not emitted_delta:
                        emitted_delta = True
                        ttfb_ms = int((time.monotonic() - received_at) * 1000)
                        logger.info("turn first_delta call=%s ttfb_ms=%d", state.call_ref, ttfb_ms)
            elif turn_event.kind == "tool_start":
                if (
                    allow_fillers
                    and not content_started
                    and (filler_count < settings.filler_max_per_turn)
                ):
                    filler_deadline = now + filler_after
            elif turn_event.kind == "done":
                remainder = sanitizer.flush()
                if not emitted_delta and not remainder and turn_event.text:
                    # Hermes delivered final text without streaming any deltas.
                    remainder = sanitize_spoken(turn_event.text)
                if remainder:
                    yield writer.content(remainder)
                outcome = "ok"
                break
            else:  # error: text is a safe, generic message by contract
                # Flush whatever the sanitizer was still holding back (e.g. an
                # unresolved markdown span) before the apology -- a caller who
                # already heard the start of an answer must never lose its tail.
                remainder = sanitizer.flush()
                if remainder:
                    # DeltaSanitizer._emit() already stripped trailing whitespace off
                    # `remainder`; without this space it would glue onto the apology
                    # ("...worSorry" instead of "...wor Sorry").
                    yield writer.content(remainder + " ")
                yield writer.content(turn_event.text or APOLOGY_LINE)
                outcome = "error"
                break
        yield writer.finish()
        yield writer.done()
    except asyncio.CancelledError:
        outcome = "disconnected"
        raise
    except Exception:
        outcome = "failed"
        logger.exception("turn failed call=%s", state.call_ref)
        raise
    finally:
        # Mandatory: closing the Hermes generator triggers the run stop. Run it on a
        # fresh task so a cancelled request can never abort the stop delivery.
        _reap(reaping, asyncio.get_running_loop().create_task(_cleanup(agen, next_task)))
        total_ms = int((time.monotonic() - received_at) * 1000)
        logger.info(
            "turn end call=%s ttfb_ms=%s total_ms=%d outcome=%s",
            state.call_ref,
            ttfb_ms if ttfb_ms is not None else "-",
            total_ms,
            outcome,
        )


async def complete_turn(
    *,
    settings: Settings,
    hermes: HermesClient,
    state: CallState,
    instructions: str,
    history: list[dict[str, str]],
    user_input: str,
) -> str:
    """Non-streaming variant (``"stream": false``): the full sanitized answer, no fillers."""
    pieces: list[str] = []
    sanitizer = DeltaSanitizer()
    final_text = ""
    agen = cast(
        AsyncGenerator[HermesTurnEvent, None],
        hermes.run_turn(
            session_id=state.session_id,
            session_key=state.session_key,
            instructions=instructions,
            conversation_history=history,
            user_input=user_input,
        ),
    )
    try:
        async for event in agen:
            if event.kind == "delta":
                pieces.append(sanitizer.feed(event.text))
            elif event.kind == "done":
                pieces.append(sanitizer.flush())
                final_text = event.text
            elif event.kind == "error":
                return event.text or APOLOGY_LINE
    finally:
        with contextlib.suppress(Exception):
            await agen.aclose()
    spoken = "".join(pieces)
    if not spoken.strip() and final_text:
        spoken = sanitize_spoken(final_text)
    return spoken
