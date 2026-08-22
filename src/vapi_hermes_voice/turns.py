"""Per-turn orchestration: one Vapi POST -> one Hermes run -> one OpenAI SSE stream.

Cleanup contract: whatever happens to the response -- completion, timeout, client
disconnect, task cancellation -- the underlying Hermes run is stopped. Abandoning the
Hermes events stream without ``POST /v1/runs/{id}/stop`` leaves the run executing
unboundedly (docs/integration-contracts.md section 2), so finalization never depends
on the (possibly already cancelled) request task: cleanup runs on a fresh background
task registered in ``reaping``, which the app lifespan drains on shutdown.

Acknowledgement delivery: a dead-air acknowledgement is always written into the
model.url SSE stream, and the response is ended immediately behind it -- see
``vapi_control.py`` for why ending the response is what makes that render reliably,
and why a Live Call Control POST is no longer tried on this critical path. Whatever
Hermes goes on to produce is delivered afterward through Live Call Control instead,
unconditionally, on a background continuation (``_finish_turn_via_control``): once
the model.url response has ended there is no other channel left to use.

That continuation is also the only place that can break a long silence, and it does:
``filler_min_gap_seconds`` is a FLOOR on the spacing of holding lines, not a budget
of one per turn, and an acknowledged turn whose answer takes half a minute used to
spend all of it silent. It now speaks again on the clock alone, from a second and
disjoint phrase pool, with the gaps doubling each time -- see
``_finish_turn_via_control`` for why nothing it says can land on or after the answer,
and ``config.reassure_after_seconds`` for the live measurements behind the numbers.
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
from .speech_feedback import Delivery, DeliveryKind
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
    Delivery is a separate decision (:func:`_ack_sse_text`): always the SSE-embedded
    content stream, with the flush token appended when enabled.

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


def _ack_sse_text(phrase: str, settings: Settings) -> str:
    """``phrase`` framed for the model.url stream: optional `` <flush />`` plus the
    trailing space every content chunk in this module carries.

    This is now the ONLY acknowledgement delivery path (see the module docstring for
    why Live Call Control was tried here and no longer is): the response is ended
    immediately behind this chunk, which is what makes the flush render reliably
    rather than risk the stall documented in ``vapi_control.py``.
    """
    text = phrase
    if settings.filler_use_flush:
        # <flush /> forces immediate TTS transmission (contracts section 1.6). The
        # response ending right behind it, not the token itself, is what makes this
        # reliable -- see the module docstring.
        text += " <flush />"
    return text + " "


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

    Called BEFORE the SSE chunk is yielded, not after: a client disconnect can land
    on that yield the instant it is written (Vapi is free to drop the model.url
    connection the moment it has something to speak), which would otherwise lose the
    record while the words were already on the wire -- and a record that says nothing
    was said is exactly the false "the model must have said it" reading this journal
    exists to prevent.

    ``phrase`` is the BARE phrase, exactly what the callee hears. Not the framed text
    :func:`_ack_sse_text` returns, which carries the `` <flush />`` audio-control
    token and a trailing space: those are transport framing, inaudible, and matching
    against them off-box would fail for no reason.
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


def register_delivery(
    state: CallState,
    journal: AckJournal | None,
    *,
    kind: DeliveryKind,
    text: str,
) -> Delivery:
    """Note that ``text`` has been handed to Vapi to speak, outcome unknown.

    Called at the moment of delivery and never later, for the same reason
    :func:`_record_ack` is called before its ``yield``: everything after this point
    is cancellable or fallible, and a delivery whose outcome is unknown has to be
    distinguishable from one that never happened (docs/integration-contracts.md
    section 6). Whether the callee actually HEARD it is decided later, by
    ``speech_feedback``, from evidence -- never from the fact that this returned.

    The journal record is opened here too and handed to the ledger, so the ledger's
    state and the published record are only ever written from one place and cannot
    come to disagree.
    """
    delivery = state.speech.register(kind=kind, text=text)
    if journal is not None:
        # No await since `register`, so the delivery cannot be adjudicated before its
        # record exists.
        delivery.record = journal.note_speech_outcome(
            state.call_ref,
            seq=delivery.seq,
            kind=kind,
            # An acknowledgement's text is an operator-configured pool phrase this
            # journal already publishes in `acks`; an answer's is arbitrary Hermes
            # content that may carry real call material and never leaves the process.
            text=text if kind == "ack" else None,
        )
    return delivery


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


# Spoken as a last resort when the real answer could not be delivered inside
# `control_answer_max_wait_seconds` of retrying: short, true, and actionable, so a
# caller who has already been told "one second" is not then met with silence. Chosen
# to never prefix-match `config._DEFAULT_FILLER_PHRASES` (asserted in
# test_answer_delivery_retry.py against the live default pool): a line the callee
# could mistake for a holding phrase, with no matching journal.record() emission
# behind it, would read off-box as a model-authored acknowledgement that never
# happened. It is journalled via `record()` regardless (see `_deliver_answer`),
# which is the invariant that actually protects against that -- the wording is a
# second, independent line of defense, not the load-bearing one.
ANSWER_DELIVERY_FAILED_LINE = (
    "Sorry, I'm having trouble reaching you with an answer right now. Please call back in a moment."
)


async def _deliver_answer(
    control: VapiControlClient,
    control_url: str,
    spoken: str,
    *,
    state: CallState,
    settings: Settings,
    journal: AckJournal | None,
    received_at: float,
) -> None:
    """Speak ``spoken`` through Live Call Control, retrying with real spacing for as
    long as the call plausibly still wants it, then a short honest apology if it
    never lands.

    Control is measurably bursty-unreliable, not merely occasionally slow: a live
    incident showed two attempts 3.0s apart -- back to back, the first attempt's own
    timeout -- BOTH fail inside the same multi-second bad window on the origin
    (vapi_control.py: it closes every connection it answers, so every POST pays a
    fresh handshake). Stacked retries are therefore two samples of the same outage,
    not two independent chances, which is why attempts here are spaced
    ``control_answer_retry_gap_seconds`` apart and continue for up to
    ``control_answer_max_wait_seconds`` -- bounded by TIME the call has plausibly
    stayed live, not by a fixed attempt count -- rather than giving up after one
    retry.

    Cancellation (``CallState.supersede_pending_answer``, called at the top of the
    NEXT turn on this call) stops this immediately, mid-attempt or mid-sleep: once
    the callee has asked something new, this answer is stale, and speaking it now
    would be worse than dropping it. The `except asyncio.CancelledError` below exists
    only to leave a record before propagating -- it does not swallow the
    cancellation, so the task's own cancelled state is unaffected.

    A WEAKER, non-cancelling signal (``CallState.caller_speaking``, set by the
    OPTIONAL speech-update webhook) instead PAUSES each attempt rather than
    abandoning the delivery -- see the loop below and ``set_caller_speaking``'s
    docstring for why the two are not the same guard.

    NOT retried when Vapi answers with a 4xx: that is Vapi rejecting the request
    outright, most plausibly because the call has already moved past this turn, and
    an identical POST is not going to change that answer. Logged at INFO, not
    WARNING -- there is nothing an operator can act on.

    Every attempt is journalled from the start (``note_answer_attempt``, BEFORE the
    first ``say``) and the SAME record is updated in place as the picture clears,
    exactly so a cancellation here leaves "attempted, outcome unknown" evidence
    rather than nothing -- see ``AnswerDeliveryRecord``.

    A 2xx from Vapi is NOT the end of the story, which is why the delivery is entered
    in ``state.speech`` the moment one arrives. Vapi accepts the POST, pushes it onto
    ``pipeline.sayQueuePush``, and then sometimes never renders it at all -- no error,
    no event, the callee simply hears nothing (see ``speech_feedback``). "Delivered"
    here means "Vapi took it", and only the ledger can say whether it was spoken.
    """
    call_ref = state.call_ref
    record = journal.note_answer_attempt(call_ref) if journal is not None else None
    attempts = 0

    def _finish(outcome: str) -> None:
        if record is not None:
            record.outcome = outcome
            record.attempts = attempts
            record.elapsed_ms = int((time.monotonic() - received_at) * 1000)

    deadline = time.monotonic() + settings.control_answer_max_wait_seconds
    try:
        while True:
            while state.caller_speaking:
                # Wait out the callee's speech instead of talking over it, and
                # instead of discarding an answer Hermes already spent real time
                # computing -- see CallState.set_caller_speaking. Bounded by the
                # same ceiling as every other wait in this loop: if the callee is
                # STILL flagged speaking once the ceiling is spent (a lost
                # "stopped" event, most plausibly), this falls through and speaks
                # anyway rather than hold forever.
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                await asyncio.sleep(min(0.25, remaining))
            attempts += 1
            t0 = time.monotonic()
            outcome = await control.say(
                control_url,
                spoken,
                call_ref=call_ref,
                timeout=settings.control_answer_timeout_seconds,
            )
            logger.info(
                "answer delivery call=%s attempt=%d delivered=%s status=%s elapsed_ms=%d",
                call_ref,
                attempts,
                outcome.delivered,
                outcome.status_code,
                int((time.monotonic() - t0) * 1000),
            )
            if outcome.delivered:
                # Vapi took it. Whether it SPEAKS it is a separate question with its
                # own evidence; entered in the ledger here so that question can be
                # answered later instead of assumed now.
                register_delivery(state, journal, kind="answer", text=spoken)
                _finish("delivered")
                return
            if outcome.status_code is not None and outcome.status_code < 500:
                logger.info(
                    "answer delivery declined call=%s status=%d after %d attempt(s)"
                    " (call likely past this turn)",
                    call_ref,
                    outcome.status_code,
                    attempts,
                )
                _finish("declined")
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            await asyncio.sleep(min(settings.control_answer_retry_gap_seconds, remaining))
    except asyncio.CancelledError:
        _finish("superseded")
        raise
    logger.warning(
        "answer delivery failed call=%s after %d attempts over %.1fs; speaking a fallback line",
        call_ref,
        attempts,
        settings.control_answer_max_wait_seconds,
    )
    fallback = await control.say(
        control_url,
        ANSWER_DELIVERY_FAILED_LINE,
        call_ref=call_ref,
        timeout=settings.control_answer_timeout_seconds,
    )
    if fallback.delivered:
        # Registered as an ACKNOWLEDGEMENT, not an answer, so it can never be
        # replayed: it is an apology for a failure, and re-speaking it after the fact
        # would compound the failure rather than recover from it.
        register_delivery(state, journal, kind="ack", text=ANSWER_DELIVERY_FAILED_LINE)
        if journal is not None:
            # The callee may hear this and it is pool-adjacent by construction (see the
            # constant above): recorded as an ordinary emission, exactly like any other
            # acknowledgement, so an off-box observer never mistakes it for a
            # model-authored line with no adapter emission behind it.
            journal.record(
                call_ref,
                text=ANSWER_DELIVERY_FAILED_LINE,
                channel="control",
                elapsed_ms=int((time.monotonic() - received_at) * 1000),
            )
    _finish("fallback_spoken" if fallback.delivered else "silent")


def _reassure_deadline(
    state: CallState, settings: Settings, *, gap: float, fallback_anchor: float
) -> float | None:
    """When the next reassurance falls due, or None when reassurance is switched off.

    Anchored on ``state.last_ack_at`` -- the moment the call-global slot was last
    taken -- and never on "now". Three things follow from that one choice:

    - The timer and the cooldown share a single origin, so with
      ``reassure_after_seconds >= filler_min_gap_seconds`` the timer can never fall
      due before :meth:`CallState.claim_reassurance` would grant it.
    - The cadence cannot drift outward. The caller awaits a network POST between
      deadlines; measuring the next one from the claim rather than from whenever that
      POST returned keeps the gaps at what they are configured to be.
    - A refused claim cannot spin. Refusal means the anchor is younger than
      ``filler_min_gap_seconds``, so the deadline this returns -- anchor plus a gap
      that is at least that large -- is strictly in the future, and the loop goes back
      to waiting on Hermes instead of re-asking immediately.

    ``fallback_anchor`` covers a call with no acknowledgement behind it at all. The
    only caller reaches here having just claimed one, so it is the degenerate/direct-
    call case rather than a live path.
    """
    if settings.reassure_after_seconds <= 0:
        return None
    anchor = state.last_ack_at if state.last_ack_at is not None else fallback_anchor
    return anchor + gap


async def _speak_reassurance(
    control: VapiControlClient,
    control_url: str,
    *,
    state: CallState,
    settings: Settings,
    journal: AckJournal | None,
    received_at: float,
) -> None:
    """Tell the callee the line is still up, if the call-global cooldown allows it.

    Journalled and entered in the ledger only once Vapi has taken it, which is the
    opposite order from the SSE acknowledgement in :func:`stream_turn` and the same
    order as the fallback line in :func:`_deliver_answer` -- because unlike the SSE
    path, this one can observe its own outcome, and the deciding question for the
    journal is which claims can be checked (see :func:`_record_ack`). Nothing awaits
    between the 2xx and both records, so a cancellation cannot land between them.

    A failed POST is logged at INFO and dropped, never retried. There is nothing to
    retry toward: the only content is "still here", it is worth less the later it
    lands, and a failure is evidence the control origin is inside one of the
    multi-second bad windows :func:`_deliver_answer` documents -- during which the
    thing not to do is POST again sooner. The cooldown slot stays spent, so the next
    attempt is pushed out by the backoff exactly as a successful one would be.
    """
    phrase = state.claim_reassurance(min_gap_seconds=settings.filler_min_gap_seconds)
    if phrase is None:
        return
    outcome = await control.say(
        control_url,
        phrase,
        call_ref=state.call_ref,
        timeout=settings.control_answer_timeout_seconds,
    )
    if not outcome.delivered:
        logger.info(
            "reassurance not delivered call=%s status=%s",
            state.call_ref,
            outcome.status_code,
        )
        return
    _record_ack(
        journal,
        call_ref=state.call_ref,
        phrase=phrase,
        channel="control",
        received_at=received_at,
    )
    register_delivery(state, journal, kind="ack", text=phrase)


async def _finish_turn_via_control(
    agen: AsyncGenerator[HermesTurnEvent, None],
    next_task: asyncio.Task[HermesTurnEvent] | None,
    sanitizer: SpokenTurn,
    *,
    control: VapiControlClient,
    control_url: str,
    state: CallState,
    settings: Settings,
    journal: AckJournal | None = None,
    received_at: float,
) -> None:
    """Continue draining an already-running Hermes turn after its acknowledgement
    ended the model.url response, reassuring the callee while that runs long, and hand
    whatever it produces to :func:`_deliver_answer` once it concludes.

    The response is ended as soon as the acknowledgement is flushed (see
    :func:`stream_turn`), so the model.url SSE stream is gone by the time this runs
    regardless of how long Hermes takes -- there is no longer any other channel left
    to deliver the answer through, so it always goes here, on a background task the
    request/response cycle does not depend on, exactly the way :func:`complete_turn`
    drains a turn to its natural conclusion.

    Reassurance is decided PURELY on the clock, in the timeout branch below, and this
    is the load-bearing property of the design rather than a simplification: what the
    callee is experiencing is silence, and no Hermes event is evidence about it. A
    `delta` here does not reach the callee -- nothing does until the whole answer is
    handed to control at the end -- so text arriving is not the silence ending, and a
    `tool_start` is not the silence continuing. Anything that consulted those events
    would be reporting on the adapter's inner life instead of on the phone call.

    Nothing spoken here can ever land on, or after, the answer, and that is structural
    rather than timed:

    - The reassurance is decided only when ``asyncio.wait`` returns nothing done --
      i.e. only while the event that would conclude the turn has NOT arrived.
    - Both POSTs come from this one coroutine, and the reassurance POST is fully
      awaited before the loop can go round again, let alone break. So the reassurance
      is handed to Vapi strictly before the answer is, and Vapi renders
      ``pipeline.sayQueuePush`` in push order (measured on 01a028f1: every push
      rendered 0.33-0.38 s after its POST, in order).
    - There is no second writer. The SSE response ended behind the acknowledgement,
      and :func:`_deliver_answer` runs after this loop, in this same coroutine.

    What remains is a bounded ordering cost, not a race: Hermes can conclude while a
    reassurance POST is in flight, in which case the callee hears the reassurance and
    then the answer close behind it. Over 34 live turns of adapter journal that would
    have happened once, at 0.9 s of separation.
    """
    pieces: list[str] = []
    final_text = ""
    reassure_gap = settings.reassure_after_seconds
    reassure_at = _reassure_deadline(state, settings, gap=reassure_gap, fallback_anchor=received_at)
    try:
        while True:
            if next_task is None:
                next_task = asyncio.ensure_future(anext(agen))
            timeout = None if reassure_at is None else max(0.0, reassure_at - time.monotonic())
            done, _pending = await asyncio.wait({next_task}, timeout=timeout)
            if not done:
                await _speak_reassurance(
                    control,
                    control_url,
                    state=state,
                    settings=settings,
                    journal=journal,
                    received_at=received_at,
                )
                # Each further wait is longer than the last, so the number of lines
                # grows with the logarithm of the silence and never with the silence
                # itself -- see `config.reassure_backoff` for why that is the right
                # shape and a fixed cadence is not.
                reassure_gap *= settings.reassure_backoff
                reassure_at = _reassure_deadline(
                    state, settings, gap=reassure_gap, fallback_anchor=time.monotonic()
                )
                continue
            try:
                event = next_task.result()
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
    _record_suppression(journal, sanitizer, call_ref=state.call_ref, received_at=received_at)
    spoken = "".join(pieces).strip()
    if not spoken and final_text:
        spoken = sanitize_spoken(final_text).strip()
    if not spoken:
        return
    await _deliver_answer(
        control,
        control_url,
        spoken,
        state=state,
        settings=settings,
        journal=journal,
        received_at=received_at,
    )


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
    # This turn is the callee speaking again: any answer this call is still trying
    # to deliver from an EARLIER turn is now stale. Delivering it after the callee
    # has already moved on to a new question would be worse than dropping it (see
    # `_deliver_answer`), so it is abandoned right here, before this turn does
    # anything else.
    state.supersede_pending_answer()
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
        # The adapter's holding-phrase vocabulary doubles as the suppression pool: a
        # phrase the adapter is configured to say is, by that fact, a holding phrase
        # the MODEL must not say (speech.SpokenTurn). BOTH pools, via
        # `Settings.holding_phrases` -- a reassurance line the adapter can speak but
        # the model is free to echo would defeat the cooldown from the only viewpoint
        # that counts, which is the callee's.
        sanitizer = SpokenTurn(settings.holding_phrases)
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
                        # Record BEFORE yielding: see _record_ack for why (a
                        # cancellation can land on the yield the instant it is
                        # written). The ledger entry goes in for the same reason and
                        # at the same moment -- Vapi echoing this chunk back as
                        # `voice-input` within ~1 ms is not evidence it was ever
                        # spoken (see speech_feedback), so the question stays open
                        # here and is settled later from evidence.
                        _record_ack(
                            journal,
                            call_ref=state.call_ref,
                            phrase=phrase,
                            channel="stream",
                            received_at=received_at,
                        )
                        register_delivery(state, journal, kind="ack", text=phrase)
                        yield writer.content(_ack_sse_text(phrase, settings))
                        if control is not None and control_url is not None:
                            # End the response NOW, right behind the flushed chunk:
                            # that is what makes it render reliably (module
                            # docstring in vapi_control.py; proven on an isolated
                            # probe and on turn 1 of call 01a02681, both under a
                            # second). Staying open to see what Hermes does next is
                            # what leaves the flush stranded (turn 2 of the SAME
                            # call: 11.5 s open, never rendered). Whatever Hermes
                            # goes on to produce is therefore delivered from here
                            # on out through Live Call Control instead, on a
                            # background task this response no longer waits for.
                            handed_off = True
                            answer_task = asyncio.get_running_loop().create_task(
                                _finish_turn_via_control(
                                    agen,
                                    next_task,
                                    sanitizer,
                                    control=control,
                                    control_url=control_url,
                                    state=state,
                                    settings=settings,
                                    journal=journal,
                                    received_at=received_at,
                                )
                            )
                            # So the NEXT turn on this call can abandon it if the
                            # callee speaks again before it finishes -- see
                            # `CallState.supersede_pending_answer` and
                            # `_deliver_answer`.
                            state.pending_answer_task = answer_task
                            reap(reaping, answer_task)
                            yield writer.finish()
                            yield writer.done()
                            outcome = "handed_off_to_control"
                            return
                        # No control channel at all on this request (Vapi's docs say
                        # call.monitor.controlUrl is always present; this is the
                        # degenerate case where it is not, or the caller passed
                        # none). There is then no channel left to deliver a later
                        # answer through, so the only option is what this adapter
                        # always did before this change: stay on the stream and let
                        # Hermes's own content keep arriving here, flush token and
                        # all -- the SAME risk the control channel exists to avoid,
                        # but no worse than the pre-existing behaviour.
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
    sanitizer = SpokenTurn(settings.holding_phrases)
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
