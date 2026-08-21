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
from typing import Any, TypeVar, cast

from .ack_journal import AckJournal
from .call_state import CallState
from .config import Settings
from .hermes_client import HermesClient, HermesTurnEvent
from .speech import SpokenTurn, sanitize_spoken
from .vapi_control import VapiControlClient
from .vapi_events import ChunkWriter

logger = logging.getLogger(__name__)

APOLOGY_LINE = "Sorry, I'm having trouble right now. Could you say that again?"


_T = TypeVar("_T")


def reap(reaping: set[asyncio.Task[Any]], task: asyncio.Task[_T]) -> None:
    """Track a background task until it finishes; log (never raise) on failure.

    Generic in the task's result because the tracking has nothing to do with it: the
    set exists so shutdown can await everything still in flight, and callers here
    discard the value (a Hermes-stop cleanup, a control-origin warm-up).
    """
    reaping.add(task)

    def _done(finished: asyncio.Task[_T]) -> None:
        reaping.discard(finished)
        if finished.cancelled():
            return
        exc = finished.exception()
        if exc is not None:
            logger.warning("background task failed error=%s", type(exc).__name__)

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


def _claim_ack_phrase(state: CallState, settings: Settings, used_this_turn: set[str]) -> str | None:
    """Claim an acknowledgement phrase from the call-global cooldown, or None.

    Returns the BARE phrase -- no `` <flush />`` suffix, no trailing whitespace.
    Delivery is a separate decision (:func:`_speak_ack`): via Vapi's Live Call
    Control ``say`` endpoint when available (the reliable path -- see
    ``vapi_control.py``), or, as a fallback, embedded in the SSE content stream
    with the flush token appended exactly as before.

    Returns None when :meth:`CallState.claim_acknowledgement` refuses -- i.e. this
    call already spoke one inside the cooldown window. Refusal costs nothing: no
    phrase is consumed from the picker and the caller simply stays silent.
    """
    phrase = state.claim_acknowledgement(
        min_gap_seconds=settings.filler_min_gap_seconds, exclude=used_this_turn
    )
    if phrase is None:
        return None
    used_this_turn.add(phrase)
    return phrase


async def _speak_ack(
    *,
    phrase: str,
    settings: Settings,
    control: VapiControlClient | None,
    control_url: str | None,
    call_ref: str,
) -> tuple[str | None, str]:
    """Deliver ``phrase`` to the callee. Returns ``(sse_text, channel)``.

    Tries Vapi's Live Call Control endpoint first when a ``control_url`` is on the
    request and the feature is enabled: that channel is proven immune to the fault
    the SSE path has (see ``vapi_control.py``). ``sse_text`` is None on success --
    the phrase must NOT also be written into the model stream, or the callee would
    hear it twice: once now via ``say``, once later merged with the real answer
    once Vapi's chunk-plan buffer for this stream eventually clears.

    Falls back to the old SSE-embedded delivery (phrase + optional `` <flush />``
    token, one atomic ``writer.content`` write at the caller's one call site) when
    there is no control URL, the feature is disabled, or the control POST itself
    fails -- never worse than the pre-existing behaviour. That fallback is only ever
    reached once the control POST has been given up on, so the acknowledgement can
    never be more timely than ``settings.ack_control_timeout``, which is exactly why
    that ceiling is derived from the acknowledgement budget rather than chosen for
    comfort (config.py) and enforced on the wall clock rather than per network phase
    (vapi_control.py). Worst case here is therefore ``filler_after_seconds +
    ack_control_timeout`` = 0.75 s on the shipped defaults, and the fallback write
    itself is in-process: no second network hop stands between it and the callee.
    """
    if settings.ack_use_call_control and control is not None and control_url is not None:
        delivered = await control.say(
            control_url, phrase, call_ref=call_ref, timeout=settings.ack_control_timeout
        )
        if delivered:
            return None, "control"
    text = phrase
    if settings.filler_use_flush:
        # <flush /> forces immediate TTS transmission (contracts section 1.6), but
        # is of limited effect once the stream itself stalls afterwards (same
        # section) -- exactly why the control channel above is tried first.
        text += " <flush />"
    return text + " ", "stream"


def _record_ack(
    journal: AckJournal | None,
    *,
    call_ref: str,
    phrase: str,
    channel: str,
    received_at: float,
) -> None:
    """Note that an acknowledgement went out: one log line, one journal entry.

    Both from here, from one ``elapsed_ms``, deliberately: the journal exists so an
    off-box observer can tell an adapter acknowledgement from a model-authored one
    (see ``ack_journal``), and a record that can disagree with the log is worse than
    no record -- the next person to read them would have to work out which lied.

    ``phrase`` is the BARE phrase, exactly what the callee hears. Not the
    ``sse_text`` :func:`_speak_ack` returns for the stream channel, which carries the
    `` <flush />`` audio-control token and a trailing space: those are transport
    framing, inaudible, and matching against them off-box would fail for no reason.
    """
    elapsed_ms = int((time.monotonic() - received_at) * 1000)
    logger.info(
        "turn filler call=%s elapsed_ms=%d channel=%s",
        call_ref,
        elapsed_ms,
        channel,
    )
    if journal is not None:
        journal.record(call_ref, text=phrase, channel=channel, elapsed_ms=elapsed_ms)


def _record_suppression(
    journal: AckJournal | None,
    spoken: SpokenTurn,
    *,
    call_ref: str,
    received_at: float,
) -> None:
    """Note that a MODEL-authored holding phrase was deleted before the callee heard it.

    Called after every ``feed``/``flush``, and no-ops until there is something to say,
    because a silent strip is the one thing this mechanism must not be. Off-box,
    "the model never wrote a holding phrase" and "it wrote one and we stripped it"
    are opposite facts about the model, and without this record they look identical --
    so a model that started producing them again would hide behind its own fix.
    """
    taken = spoken.take_suppressed_opening()
    if taken is None:
        return
    phrase, rule = taken
    elapsed_ms = int((time.monotonic() - received_at) * 1000)
    # The phrase itself is logged only as a length: it is model output, and this
    # module logs no content. The journal holds the text (see `note_suppressed` for
    # why that is safe there and needed there).
    logger.info(
        "turn model holding phrase suppressed call=%s elapsed_ms=%d rule=%s chars=%d",
        call_ref,
        elapsed_ms,
        rule,
        len(phrase),
    )
    if journal is not None:
        journal.note_suppressed(call_ref, text=phrase, reason=rule, elapsed_ms=elapsed_ms)


async def _finish_turn_via_control(
    agen: AsyncGenerator[HermesTurnEvent, None],
    next_task: asyncio.Task[HermesTurnEvent] | None,
    sanitizer: SpokenTurn,
    *,
    control: VapiControlClient,
    control_url: str,
    call_ref: str,
    timeout: float,
    journal: AckJournal | None = None,
    received_at: float,
) -> None:
    """Continue draining an already-running Hermes turn after its acknowledgement
    went out via Live Call Control, and speak whatever it produces through the
    SAME channel, once, when it concludes.

    Live evidence (adapter journal, both turns of a real call): once ``say``
    renders speech for a turn, Vapi treats that turn as answered and abandons the
    still-open model.url HTTP connection within roughly 1-2 s -- "turn end ...
    outcome=disconnected". The model.url SSE stream can therefore no longer be
    trusted to deliver the rest of the answer once an acknowledgement has been
    spoken this way, however long Hermes takes -- so it is delivered here instead,
    on a background task the request/response cycle does not depend on, exactly
    the way :func:`complete_turn` drains a turn to its natural conclusion.
    """
    pieces: list[str] = []
    final_text = ""
    try:
        while True:
            if next_task is None:
                next_task = asyncio.ensure_future(anext(agen))
            try:
                event = await next_task
            except StopAsyncIteration:
                event = HermesTurnEvent(kind="done")
            next_task = None
            if event.kind == "delta":
                pieces.append(sanitizer.feed(event.text))
                continue
            if event.kind == "tool_start":
                continue
            if event.kind == "done":
                pieces.append(sanitizer.flush())
                final_text = event.text
            else:  # error: already a safe, generic message by contract
                pieces.append(sanitizer.flush())
                pieces.append(event.text or APOLOGY_LINE)
            break
    finally:
        with contextlib.suppress(Exception):
            await agen.aclose()
    _record_suppression(journal, sanitizer, call_ref=call_ref, received_at=received_at)
    spoken = "".join(pieces).strip()
    if not spoken and final_text:
        spoken = sanitize_spoken(final_text).strip()
    if not spoken:
        return
    delivered = await control.say(control_url, spoken, call_ref=call_ref, timeout=timeout)
    if not delivered:
        logger.warning("post-ack control delivery failed call=%s", call_ref)


async def stream_turn(
    *,
    settings: Settings,
    hermes: HermesClient,
    control: VapiControlClient | None = None,
    control_url: str | None = None,
    state: CallState,
    instructions: str,
    history: list[dict[str, str]],
    user_input: str,
    reaping: set[asyncio.Task[Any]],
    journal: AckJournal | None = None,
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
    if journal is not None:
        # Note the call BEFORE anything can be acknowledged on it. A turn that stays
        # silent (fast answer, or the cooldown refusing) must leave an EMPTY record,
        # not no record: "we drove this turn and said nothing" is what lets an
        # off-box reader call a holding phrase the callee heard model-authored, and
        # "we have never heard of this call" must make the same reader say unknown.
        journal.open(state.call_ref)
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
    handed_off = False  # True once the rest of the turn was handed to a background
    # continuation (_finish_turn_via_control): the finally block below must then
    # never also cancel/close agen out from under it.
    try:
        yield writer.role()
        # The acknowledgement pool doubles as the suppression pool: a phrase the
        # adapter is configured to say is, by that fact, a holding phrase the MODEL
        # must not say (speech.SpokenTurn).
        sanitizer = SpokenTurn(settings.filler_phrases)
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
                    phrase = _claim_ack_phrase(state, settings, filler_used_this_turn)
                    if phrase is not None:
                        ack_text, channel = await _speak_ack(
                            phrase=phrase,
                            settings=settings,
                            control=control,
                            control_url=control_url,
                            call_ref=state.call_ref,
                        )
                        if ack_text is not None:
                            yield writer.content(ack_text)
                        _record_ack(
                            journal,
                            call_ref=state.call_ref,
                            phrase=phrase,
                            channel=channel,
                            received_at=received_at,
                        )
                        if channel == "control":
                            # Vapi abandons the still-open model.url connection
                            # shortly after `say` speaks for this turn (observed
                            # live: "turn end ... outcome=disconnected" within
                            # ~1-2s of the control-delivered ack on both turns of
                            # a real call). Racing that cancellation to deliver
                            # the rest of the answer in-stream is not viable, so
                            # hand the still-running Hermes turn to a background
                            # continuation that finishes it out-of-band and
                            # speaks the result through the same channel, and end
                            # this HTTP response cleanly right now.
                            assert control is not None and control_url is not None
                            handed_off = True
                            reap(
                                reaping,
                                asyncio.get_running_loop().create_task(
                                    _finish_turn_via_control(
                                        agen,
                                        next_task,
                                        sanitizer,
                                        control=control,
                                        control_url=control_url,
                                        call_ref=state.call_ref,
                                        # The ANSWER's ceiling, not the
                                        # acknowledgement's: this runs on a background
                                        # task with Hermes already finished and no
                                        # deadline on it, so borrowing the tight
                                        # acknowledgement budget here would truncate
                                        # the answer to protect a deadline it is not on.
                                        timeout=settings.control_answer_timeout_seconds,
                                        journal=journal,
                                        received_at=received_at,
                                    )
                                ),
                            )
                            yield writer.finish()
                            yield writer.done()
                            outcome = "handed_off_to_control"
                            return
                continue
            try:
                turn_event = next_task.result()
            except StopAsyncIteration:
                turn_event = HermesTurnEvent(kind="done")
            next_task = None
            if turn_event.kind == "delta":
                text = sanitizer.feed(turn_event.text)
                _record_suppression(
                    journal, sanitizer, call_ref=state.call_ref, received_at=received_at
                )
                if text or not sanitizer.holding_opening:
                    # Content has begun: cancel any pending filler atomically and, via
                    # content_started, permanently forbid future re-arms below.
                    #
                    # A delta the opening gate is STILL HOLDING does not count as the
                    # answer beginning. It may be nothing but a model-authored holding
                    # phrase, and that phrase is about to be deleted -- so letting it
                    # set this flag would cancel the adapter's own acknowledgement too
                    # and leave the callee with neither. Markdown the sanitizer is
                    # buffering is unaffected: only the holding-phrase gate reports
                    # `holding_opening`, so every pre-existing case still lands here.
                    filler_deadline = None
                    content_started = True
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
                _record_suppression(
                    journal, sanitizer, call_ref=state.call_ref, received_at=received_at
                )
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
                _record_suppression(
                    journal, sanitizer, call_ref=state.call_ref, received_at=received_at
                )
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
        # fresh task so a cancelled request can never abort the stop delivery. Skipped
        # when handed off: _finish_turn_via_control now owns agen/next_task and closes
        # them itself once the turn concludes -- cancelling/closing them here too would
        # race that background task's own await on the same next_task.
        if not handed_off:
            reap(reaping, asyncio.get_running_loop().create_task(_cleanup(agen, next_task)))
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
    journal: AckJournal | None = None,
) -> str:
    """Non-streaming variant (``"stream": false``): the full sanitized answer, no fillers.

    No acknowledgement is ever spoken here -- there is no stream to interleave one
    into -- but the model's own holding-phrase opener is suppressed exactly as it is
    on the streaming path. R2 is about what the callee hears, and this response is
    heard the same way.
    """
    received_at = time.monotonic()
    pieces: list[str] = []
    sanitizer = SpokenTurn(settings.filler_phrases)
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
                _record_suppression(
                    journal, sanitizer, call_ref=state.call_ref, received_at=received_at
                )
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
