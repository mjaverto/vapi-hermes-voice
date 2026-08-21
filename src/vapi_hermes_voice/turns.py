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


def _filler_text(state: CallState, settings: Settings, used_this_turn: set[str]) -> str | None:
    """Claim an acknowledgement and render it, or None if the cooldown forbids one.

    On success, returns a single string: the phrase plus, when enabled, exactly one
    trailing `` <flush />`` token. This is passed to a single ``writer.content(...)``
    call (one ``yield``, one SSE ``data:`` frame) at its one call site in
    ``stream_turn`` -- it is never split across multiple deltas, never fed through
    DeltaSanitizer (which could otherwise hold part of it back across a chunk
    boundary), and never concatenated with anything else before being written. A
    phrase and its flush token are therefore always one atomic write from this
    adapter's side.

    Returns None when :meth:`CallState.claim_acknowledgement` refuses -- i.e. this
    call already spoke one inside the cooldown window. Refusal costs nothing: no
    phrase is consumed from the picker and the caller simply stays silent.
    """
    text = state.claim_acknowledgement(
        min_gap_seconds=settings.filler_min_gap_seconds, exclude=used_this_turn
    )
    if text is None:
        return None
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
) -> AsyncIterator[str]:
    """Drive one voice turn, yielding OpenAI SSE lines (role, content*, finish, DONE).

    An acknowledgement ("okay, let me check") races the first Hermes event: if the
    turn has produced no content after ``filler_after_seconds``, one is spoken so the
    callee gets an immediate answer to having been spoken to instead of dead air.
    That is the requirement -- a brief acknowledgement within two seconds of the
    callee finishing ANY turn -- so it is deliberately NOT conditional on a tool
    running, and deliberately not suppressed on the first turn of the call, which is
    where the dead air was worst. The timer also re-arms on tool starts (each Hermes
    tool round trip adds ~2.9 s of silence) so a long multi-tool run can still get a
    fresh line once the cooldown below has expired.

    Two gates, and BOTH must allow it:

    - ``content_started`` is a single-owner flag flipped the instant the turn's
      first ``delta`` event arrives (asyncio is single-threaded and the flag is
      only ever read/written inside this synchronous loop body, never across an
      ``await``, so the check-then-set is atomic by construction -- the same
      reasoning ``CallStateRegistry`` relies on for its own check-then-mutate
      state). Once set, no later ``tool_start`` may re-arm the deadline, and the
      dead-air branch re-checks the same flag before speaking: an acknowledgement
      can never be spoken once the answer has begun. This is also what keeps it
      out of the way of a fast turn -- Hermes answering in 300 ms means the callee
      never hears one, which is correct: there was no dead air to cover.
    - ``CallState.claim_acknowledgement`` enforces ``filler_min_gap_seconds`` as a
      cooldown GLOBAL TO THE CALL, not to the turn. It is consulted at the moment
      the line would be spoken, so no ordering of turn boundaries, tool-start
      re-arms or cancelled-and-retried requests can produce two closer together
      than the gap. A refused claim leaves the cooldown untouched: a turn that
      stayed silent never spends the slot.

    Hermes ``error`` events are already safe generic messages by the client's
    contract; they are spoken as ordinary content -- after flushing whatever text
    the sanitizer was still holding back -- so the caller hears an apology instead
    of Vapi surfacing a platform error, and no already-buffered words are lost.
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
        filler_deadline: float | None = loop.time() + filler_after
        emitted_delta = False
        content_started = False  # single-owner: set once, forever forbids new fillers
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
                # Dead air: the acknowledgement window elapsed before the next Hermes
                # event. The cooldown is checked inside the claim, not here, so the
                # answer is always the freshest one at the moment of speaking.
                filler_deadline = None  # re-armed only by a later tool_start
                if not content_started:
                    ack = _filler_text(state, settings, filler_used_this_turn)
                    if ack is not None:
                        yield writer.content(ack)
                        logger.info(
                            "turn filler call=%s elapsed_ms=%d",
                            state.call_ref,
                            int((time.monotonic() - received_at) * 1000),
                        )
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
                if not content_started:
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
