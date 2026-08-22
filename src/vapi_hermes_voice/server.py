"""FastAPI app factory: health endpoints and the Vapi Custom LLM chat endpoint."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets as secrets_mod
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from .ack_journal import CALL_REF_RE, AckJournal
from .call_state import CallState, CallStateRegistry
from .config import Settings
from .hermes_client import HermesClient, HermesUnavailableError
from .logredact import redact_phone
from .policy import (
    CallerPolicy,
    build_opening_nudge,
    build_reason_line,
    has_trailing_user_message,
    is_first_callee_turn,
    split_messages,
    truncate_history,
)
from .speech import build_instructions
from .speech_feedback import (
    Delivery,
    concat_assistant_text,
    spoken_coverage,
    user_text,
)
from .turns import complete_turn, reap, register_delivery, stream_turn
from .vapi_control import VapiControlClient
from .vapi_events import (
    ChunkWriter,
    OversizedPayloadError,
    VapiChatRequest,
    VapiProtocolError,
    completion_json,
    parse_caller_speech_event,
    parse_chat_request,
    parse_server_message,
)

logger = logging.getLogger(__name__)

BUSY_LINE = "I'm sorry, all lines are busy right now. Please call back shortly."
DENIED_LINE = "Sorry, this number isn't available. Goodbye."

_SSE_HEADERS = {"Cache-Control": "no-cache"}


def _authorized(header: str | None, settings: Settings) -> bool:
    """Constant-time check of the Authorization header against the adapter API key.

    Vapi's custom-llm credential sends the key in the Authorization header; whether it
    carries a ``Bearer `` prefix is unverified (docs/integration-contracts.md section
    4), so both forms are accepted. Each candidate is compared with
    ``secrets.compare_digest``; the header-shape branch reveals nothing secret.
    """
    if header is None:
        return False
    expected = settings.adapter_api_key.get_secret_value().encode()
    candidate = header.encode()
    prefix, _, rest = header.partition(" ")
    if prefix.lower() == "bearer" and rest:
        candidate = rest.strip().encode()
    return secrets_mod.compare_digest(candidate, expected)


def _speak_once(text: str) -> AsyncIterator[str]:
    """A complete one-utterance SSE stream (busy/denied lines)."""

    async def gen() -> AsyncIterator[str]:
        writer = ChunkWriter()
        yield writer.role()
        yield writer.content(text)
        yield writer.finish()
        yield writer.done()

    return gen()


async def _speak_replay(
    control: VapiControlClient,
    control_url: str,
    delivery: Delivery,
    *,
    state: CallState,
    settings: Settings,
    journal: AckJournal | None,
) -> None:
    """Re-deliver one confirmed-dropped answer, exactly once.

    The claim that authorises this (``SpeechLedger.claim_replay``) has already been
    made by the caller, synchronously, before this task existed -- so this coroutine
    never decides whether to speak, only carries out a decision already recorded. That
    split is deliberate: a decision made inside a task can be made twice by two tasks,
    and two tasks speaking is the failure mode this whole guard exists to avoid.

    One attempt, no retries. ``_deliver_answer``'s patient retry loop is right for a
    first delivery -- nobody has heard anything yet, so a late answer beats none -- but
    wrong here: a replay is already the second attempt at this text, and a retry loop
    on top of it turns one drop into an unbounded number of chances to say the same
    thing twice if any single confirmation is ever wrong.
    """
    outcome = await control.say(
        control_url,
        delivery.text,
        call_ref=state.call_ref,
        timeout=settings.control_answer_timeout_seconds,
    )
    logger.warning(
        "answer replayed after confirmed drop call=%s seq=%d delivered=%s status=%s",
        state.call_ref,
        delivery.seq,
        outcome.delivered,
        outcome.status_code,
    )
    if not outcome.delivered:
        return
    state.speech.mark_replayed(delivery)
    # The replay is itself a delivery, subject to the same confirmation rules as any
    # other -- including never being replayed again, because the record above can
    # never leave `replayed` and this new one starts from `unconfirmed`.
    register_delivery(state, journal, kind="answer", text=delivery.text)


def _reconcile_speech(
    chat: VapiChatRequest,
    state: CallState,
    settings: Settings,
) -> Delivery | None:
    """Audit this call's earlier deliveries against Vapi's own record of what was said,
    and claim a replay if one is due. Returns the delivery to re-speak, or None.

    This is the ZERO-CONFIG half of the guard and the load-bearing one: it needs no
    assistant configuration at all, because the evidence is already in the request
    body. The ``messages[]`` Vapi sends is derived from what was ACTUALLY SPOKEN, so
    the assistant text in it is a settled account of what the callee heard on every
    earlier turn -- and anything the adapter delivered that is missing from it was
    never rendered (see ``speech_feedback`` for the measurements behind that claim,
    and for why no timeout is ever allowed to reach the same verdict).

    Runs before the turn does anything else, so a recovery is spoken ahead of whatever
    this turn goes on to produce. Purely synchronous: it contains no ``await``, so the
    claim it makes cannot be raced by a concurrent request on the same call.
    """
    pairs = [(message.role, message.content) for message in chat.messages]
    prior = state.prior_user_input
    # Liveness: only condemn a delivery once the record demonstrably covers the turn
    # it was made on. Seeing the PREVIOUS turn's own input in this history is that
    # proof; without it, a truncated or differently-shaped history would read as
    # "nothing we ever said was spoken".
    history_advanced = prior is not None and spoken_coverage(prior, user_text(pairs)) >= (
        settings.speech_match_threshold
    )
    spoken, dropped = state.speech.reconcile_history(
        heard=concat_assistant_text(pairs),
        threshold=settings.speech_match_threshold,
        settled_before=time.monotonic() - settings.speech_confirm_window_seconds,
        history_advanced=history_advanced,
    )
    if dropped:
        logger.warning(
            "vapi accepted speech and never rendered it call=%s dropped=%s kinds=%s",
            state.call_ref,
            ",".join(str(d.seq) for d in dropped),
            ",".join(sorted({d.kind for d in dropped})),
        )
    if spoken:
        logger.debug(
            "speech confirmed spoken call=%s seqs=%s",
            state.call_ref,
            ",".join(str(d.seq) for d in spoken),
        )
    if not settings.speech_drop_replay:
        return None
    return state.speech.claim_replay(max_age_seconds=settings.speech_drop_replay_max_age_seconds)


def create_app(
    settings: Settings,
    hermes_transport: httpx.AsyncBaseTransport | None = None,
    vapi_control_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Build the ASGI app; one HermesClient and one turn-slot counter per app.

    ``hermes_transport`` lets tests mount a fake Hermes backend in-process;
    ``vapi_control_transport`` does the same for Vapi's Live Call Control endpoint
    (see ``vapi_control.py``).
    """
    # Plain counter, not a Semaphore: the busy check and the slot grab must be a
    # single atomic step (no await between them). asyncio is single-threaded, so
    # check+increment with no intervening await is race-free.
    active_turns = 0
    policy = CallerPolicy(settings.allowed_callers)
    registry = CallStateRegistry(settings)
    # None when VHV_DEBUG_ACK_JOURNAL is false: nothing is recorded and the route
    # below is never registered, so a disabled deployment 404s exactly like one that
    # never had the endpoint. See config.py for why it defaults to ON.
    journal = (
        AckJournal(
            max_calls=settings.debug_ack_journal_max_calls,
            max_entries_per_call=settings.debug_ack_journal_max_entries_per_call,
            ttl_seconds=settings.debug_ack_journal_ttl_seconds,
        )
        if settings.debug_ack_journal
        else None
    )
    if not policy.enforced:
        logger.warning("caller allowlist is DISABLED (VHV_ALLOWED_CALLERS empty): allowing all")

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        hermes = HermesClient(settings, transport=hermes_transport)
        app.state.hermes = hermes
        app.state.vapi_control = VapiControlClient(transport=vapi_control_transport)
        app.state.reaping = set()
        # The per-call registry, reachable from the app rather than only from this
        # closure. Two consumers need it: the speech-feedback webhook (a request with
        # no turn behind it) and tests that assert on ledger state.
        app.state.calls = registry
        warmup_task: asyncio.Task[None] | None = None
        if settings.warmup_on_start:
            warmup_task = asyncio.create_task(hermes.warmup(), name="vhv-warmup")
        try:
            yield
        finally:
            if warmup_task is not None:
                warmup_task.cancel()
                try:
                    await warmup_task
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    logger.warning(
                        "warmup cancel failed during shutdown error=%s", type(exc).__name__
                    )
            reaping: set[asyncio.Task[Any]] = app.state.reaping
            if reaping:
                # In-flight turn cleanups (Hermes stops) must complete before the
                # shared client closes -- no orphaned runs, even across shutdown.
                await asyncio.gather(*list(reaping), return_exceptions=True)
            await hermes.aclose()
            await app.state.vapi_control.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        hermes: HermesClient = app.state.hermes
        try:
            ready = await hermes.health()
        except HermesUnavailableError as exc:
            logger.warning("readiness check failed error=%s", type(exc).__name__)
            ready = False
        else:
            if not ready:
                logger.warning("readiness check failed error=unhealthy_status")
        if ready:
            return JSONResponse({"status": "ready"})
        return JSONResponse({"status": "degraded"}, status_code=503)

    async def handle_chat(request: Request) -> Response:
        nonlocal active_turns
        if not _authorized(request.headers.get("authorization"), settings):
            logger.warning("chat request rejected reason=bad_authorization")
            return JSONResponse({"error": {"message": "unauthorized"}}, status_code=401)
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > settings.max_body_bytes:
                logger.warning("chat request rejected reason=oversized_body")
                return JSONResponse(
                    {"error": {"message": "request body too large"}}, status_code=413
                )
        try:
            chat = parse_chat_request(bytes(raw), max_bytes=settings.max_body_bytes)
        except OversizedPayloadError:
            logger.warning("chat request rejected reason=oversized_body")
            return JSONResponse({"error": {"message": "request body too large"}}, status_code=413)
        except VapiProtocolError as exc:
            # Log the error type only; never body content (may carry transcript).
            logger.warning("chat request rejected reason=%s", type(exc).__name__)
            return JSONResponse({"error": {"message": "malformed request"}}, status_code=400)

        state = registry.get_or_create(chat.call_id)
        if chat.tools_present:
            logger.debug("request carries Vapi tools; ignored (Hermes owns tools)")

        # AUDIT FIRST. Vapi's `messages[]` is derived from what was actually spoken,
        # so this request body is a settled account of what the callee heard on every
        # earlier turn -- and it is the only feedback channel that needs no assistant
        # configuration at all. Anything this adapter delivered and Vapi never
        # rendered is identified here, journalled, and (for an answer) re-spoken once.
        # Done before the allowlist, the reason fast path and the capacity check
        # because it belongs to the PREVIOUS turn, not this one: a busy line or a
        # locally-answered opening must not cost the callee an answer that was already
        # produced and silently lost. It contains no await, so every check-and-set in
        # this handler stays atomic.
        replay = _reconcile_speech(chat, state, settings)
        recovered_answer: str | None = None
        if replay is not None and chat.control_url is not None:
            recovered_answer = replay.text
            reap(
                app.state.reaping,
                asyncio.get_running_loop().create_task(
                    _speak_replay(
                        app.state.vapi_control,
                        chat.control_url,
                        replay,
                        state=state,
                        settings=settings,
                        journal=journal,
                    )
                ),
            )
        if chat.control_url is not None:
            # For the webhook path, which has no turn behind it and so no other way
            # to reach this call's control endpoint.
            state.control_url = chat.control_url

        def speak(text: str) -> Response:
            if not chat.stream:
                return JSONResponse(completion_json(text))
            return StreamingResponse(
                _speak_once(text), media_type="text/event-stream", headers=_SSE_HEADERS
            )

        # INBOUND ONLY. `customer.number` is whoever is on the other end of the call:
        # the caller on an inbound call, but the CALLEE on an outbound one. Screening
        # an outbound number against a list of permitted *callers* denies every task
        # call the operator placed themselves the moment the allowlist is populated
        # (which the startup warning above tells them to do) -- and denies it before
        # Hermes is ever contacted, so the objective silently never runs.
        if (
            chat.direction == "inbound"
            and policy.enforced
            and not policy.is_allowed(chat.customer_number)
        ):
            # Fail closed: an enforced allowlist denies unknown AND absent caller
            # identity (metadataSendMode off, web calls) alike.
            logger.info(
                "call denied call=%s from=%s outcome=denied",
                state.call_ref,
                redact_phone(chat.customer_number) if chat.customer_number else "unknown",
            )
            return speak(DENIED_LINE)

        if chat.variables.has_values:
            # Fires for ANY non-empty variableValues, understood or not: a call whose
            # variables the adapter could make nothing of used to be indistinguishable
            # from a call that carried none.
            # Lengths and counts only: dynamic-variable text is untrusted, never logged.
            logger.info(
                "call has task variables call=%s direction=%s %s",
                state.call_ref,
                chat.direction,
                chat.variables.log_summary(),
            )
        if chat.variables.unknown_keys:
            # KEYS ONLY -- the values are untrusted call content. A live call sent
            # call_purpose/patient_name/patient_context and every one was dropped in
            # silence; a mis-named variable must now be visible in the log.
            logger.warning(
                "call variables not recognized as objective or callee call=%s keys=%s",
                state.call_ref,
                ",".join(chat.variables.unknown_keys),
            )
        callee_is_principal = chat.callee_is_principal(
            principal=settings.principal, principal_number=settings.principal_number
        )

        # WHY WE CALLED, said immediately. On an outbound task call the callee's first
        # utterance is answered from adapter-local text: no Hermes run, no tool, no
        # model. Nothing routed through Hermes can hold a one-to-two-second deadline
        # (1.6-2.2 s warm, 3.6-4.9 s cold, 14-17 s with a tool), and the live failure
        # was exactly that -- the callee said "Hello?", waited ~10 s, and was told
        # "give me a moment to find that" on a call we had placed ourselves.
        #
        # Placed ahead of the capacity check on purpose: this branch starts no Hermes
        # run and holds no turn slot, so answering "Hello?" with "all lines are busy"
        # would be wrong. It contains no await either, so the `active_turns`
        # check-and-increment below is still a single atomic step.
        reason_line: str | None = None
        if settings.outbound_reason_fast_path and chat.direction == "outbound":
            if not has_trailing_user_message(chat.messages):
                # Model-generated first-message mode: build_opening_nudge below has
                # Hermes state the reason on this very turn, so latch it now or the
                # callee would hear it a second time when they reply.
                state.reason_spoken = state.reason_spoken or bool(
                    chat.variables.spoken_reason or chat.variables.purpose
                )
            elif not state.reason_spoken and is_first_callee_turn(chat.messages):
                reason_line = build_reason_line(
                    settings,
                    variables=chat.variables,
                    callee_is_principal=callee_is_principal,
                )
        if reason_line is not None:
            state.reason_spoken = True  # no await since the check above: atomic
            logger.info(
                "outbound reason spoken locally call=%s chars=%d source=%s",
                state.call_ref,
                len(reason_line),
                "spoken_reason" if chat.variables.spoken_reason else "generic",
            )
            return speak(reason_line)

        if active_turns >= settings.max_concurrent_turns:
            logger.warning("turn rejected call=%s reason=busy", state.call_ref)
            return speak(BUSY_LINE)

        messages = truncate_history(chat.messages, settings.max_history_messages)
        history, user_input, extra = split_messages(
            messages,
            opening=build_opening_nudge(
                settings,
                direction=chat.direction,
                variables=chat.variables,
                callee_is_principal=callee_is_principal,
            ),
        )
        if recovered_answer is not None:
            # The replay is going out right now, so tell Hermes it was said. Vapi's
            # own history could not contain it -- that absence is precisely why it is
            # being re-spoken -- so without this the model would answer the callee's
            # new utterance as though it had never answered at all, and the callee
            # would hear the answer twice in two different forms. `history` is
            # everything except this turn's input, so appending puts it immediately
            # before the callee's new words, which is exactly where it belongs in
            # time.
            history.append({"role": "assistant", "content": recovered_answer})
        # The turn input this call is answering NOW, kept for the NEXT turn to use as
        # the liveness proof that licenses a drop verdict (see `_reconcile_speech`).
        # Latched after the split so it is the same string Hermes was given, and never
        # logged: it is callee speech.
        state.prior_user_input = user_input
        instructions = build_instructions(
            settings,
            direction=chat.direction,
            extra=extra,
            variables=chat.variables,
            callee_is_principal=callee_is_principal,
        )

        if not chat.stream:
            active_turns += 1
            try:
                text = await complete_turn(
                    settings=settings,
                    hermes=app.state.hermes,
                    state=state,
                    instructions=instructions,
                    history=history,
                    user_input=user_input,
                    journal=journal,
                )
            finally:
                active_turns -= 1
            return JSONResponse(completion_json(text))

        active_turns += 1  # atomic with the capacity check above: no await between

        async def stream() -> AsyncIterator[str]:
            nonlocal active_turns
            try:
                async for line in stream_turn(
                    settings=settings,
                    hermes=app.state.hermes,
                    control=app.state.vapi_control,
                    control_url=chat.control_url,
                    state=state,
                    instructions=instructions,
                    history=history,
                    user_input=user_input,
                    reaping=app.state.reaping,
                    journal=journal,
                ):
                    yield line
            finally:
                active_turns -= 1

        return StreamingResponse(stream(), media_type="text/event-stream", headers=_SSE_HEADERS)

    async def handle_debug_acks(request: Request, call_ref: str) -> Response:
        """The adapter's own record of what it said on one call, and what it stripped.

        Read-only, and bounded in what it can disclose. The acknowledgements are text
        this process chose from ``filler_phrases``. The suppressed model openings are
        model text, but only ever text that the gate in ``speech.SpokenTurn`` matched
        WHOLE against a closed acknowledgement/stall grammar or a verbatim
        ``filler_phrases`` member -- that is the suppression rule itself, not a hope,
        so such a string cannot contain caller speech and cannot carry information.
        Never a transcript, never a number, never a secret. Registered only when the
        journal is enabled (see config.py), behind the SAME bearer -- and the same
        optional route secret -- as /chat/completions, because a reader of this could
        already drive turns.

        404, never 403, for an unknown or malformed ``call_ref``: a caller holding the
        API key learns only whether this process still holds a record, and one shape
        of answer for "not a call reference", "never seen" and "aged out" is one fewer
        thing to reason about. The reference itself is checked against
        ``CALL_REF_RE`` before it is used, so an arbitrary path segment never becomes
        a lookup key -- and never reaches a log line.
        """
        assert journal is not None  # route is registered only when it is not
        if not _authorized(request.headers.get("authorization"), settings):
            logger.warning("debug acks request rejected reason=bad_authorization")
            return JSONResponse({"error": {"message": "unauthorized"}}, status_code=401)
        if CALL_REF_RE.fullmatch(call_ref) is None:
            return JSONResponse({"error": {"message": "not found"}}, status_code=404)
        snapshot = journal.snapshot(call_ref)
        if snapshot is None:
            return JSONResponse({"error": {"message": "not found"}}, status_code=404)
        return JSONResponse(
            {
                "call_ref": call_ref,
                "acks": [entry.as_dict() for entry in snapshot.acks],
                # How many of this call's acknowledgements this journal has LOST to a
                # cap or the TTL. Load-bearing for the consumer, not diagnostics: a
                # reader that cannot see the record is incomplete would read a missing
                # entry as "the model wrote that line", which is a false accusation.
                "dropped": snapshot.dropped,
                # Holding phrases the MODEL opened a turn with, which the adapter
                # deleted before the callee heard them. Present-and-empty means the
                # gate ran and found nothing, which is a different answer from absent
                # (an adapter old enough not to have the gate at all). Kept strictly
                # apart from `acks`/`dropped`, which answer "what did the callee
                # hear": a chatty model must not be able to make the acknowledgement
                # record look incomplete and so retire the attribution verdict.
                "suppressed_model_openings": [entry.as_dict() for entry in snapshot.suppressed],
                "suppressed_dropped": snapshot.suppressed_dropped,
                # What became of the answer that followed an acknowledgement, once one
                # was spoken and the model.url response ended behind it -- the only
                # channel left for it is Live Call Control (vapi_control.py), and it is
                # measurably unreliable in bursts. Present-and-empty means no turn on
                # this call ever needed a background delivery; absent means an adapter
                # too old to track this at all.
                "answer_deliveries": [e.as_dict() for e in snapshot.answer_deliveries],
                "answer_deliveries_dropped": snapshot.answer_deliveries_dropped,
                # Whether each thing the adapter delivered actually became AUDIO --
                # the only part of this record that is not the adapter's account of
                # its own actions. Vapi accepts text on both channels and sometimes
                # never renders it with no error anywhere, so every other field here
                # answers "what did we send" and this one answers "what did the callee
                # get". `unconfirmed` is a real answer and not a pending one: nothing
                # times out into `confirmed_dropped`, because no-audio-yet is unknown
                # rather than lost. Present-and-empty means the adapter delivered
                # nothing on this call; absent means an adapter too old to know.
                "speech_outcomes": [e.as_dict() for e in snapshot.speech_outcomes],
                "speech_outcomes_dropped": snapshot.speech_outcomes_dropped,
                "limits": journal.limits,
            }
        )

    async def handle_vapi_server(request: Request) -> Response:
        """OPTIONAL speech-feedback webhook: Vapi's server messages for a live call.

        Registered only when ``vapi_server_secret`` is set. That is the fail-closed
        direction and it matters: if the assistant is pointed at this URL and the
        secret is NOT configured here, every event gets a 404 and none is ever
        accepted -- rather than the adapter taking unauthenticated call events from
        anyone who guesses the path. Nothing degrades when that happens, because the
        load-bearing feedback channel is the conversation history Vapi already sends
        on every turn and needs no configuration on either side.

        404 on a bad or missing secret, never 401 or 403: this endpoint's existence is
        not something an unauthenticated caller gets to learn. Compared in constant
        time.

        The body is UNTRUSTED. Only four scalar fields are read out of it, the call id
        is matched against a strict shape before it is used as a lookup key, an unknown
        call allocates no state, and no text from it is ever logged.

        What this can and cannot do is deliberately asymmetric. It may confirm that
        something WAS spoken -- ``speech-update`` (which arrives with no
        ``serverMessages`` edit) and ``assistant.speechStarted`` (opt-in, text-bearing,
        and absent for Live Call Control ``say`` on this account). It may NOT conclude
        that anything was dropped: silence here is silence, and a stream held open once
        delayed a render by 20.3 s without losing it, so a missing event is evidence of
        unknown. Drop verdicts come only from Vapi's committed conversation history
        (``_reconcile_speech``), which is a settled account of a completed turn.

        It also carries the MIRROR event: ``speech-update`` about the CALLEE
        (``role: "user"``), which this route used to discard entirely (see
        ``parse_server_message``'s docstring for why it must never be used to confirm
        a delivery). That event still confirms nothing here -- it instead holds any
        answer or reassurance this call has queued through Live Call Control, so it
        is not spoken over a caller Vapi's own transcriber says is talking right now.
        See ``parse_caller_speech_event`` and ``CallState.set_caller_speaking``.
        """
        expected = settings.vapi_server_secret
        assert expected is not None  # route is registered only when it is set
        presented = request.headers.get("x-vapi-secret") or ""
        if not secrets_mod.compare_digest(presented.encode(), expected.get_secret_value().encode()):
            logger.warning("vapi server event rejected reason=bad_secret")
            return JSONResponse({"error": {"message": "not found"}}, status_code=404)
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > settings.max_body_bytes:
                logger.warning("vapi server event rejected reason=oversized_body")
                return JSONResponse({"error": {"message": "too large"}}, status_code=413)
        body = bytes(raw)
        event = parse_server_message(body)
        if event is not None:
            state = registry.peek(event.call_id)
            if state is None:
                # A call this process is not tracking (restarted, evicted, or never
                # ours). No state is created: a webhook must not be able to mint
                # call state.
                return JSONResponse({})
            if event.text:
                settled = state.speech.confirm_by_text(
                    event.text,
                    threshold=settings.speech_match_threshold,
                    evidence="assistant.speechStarted",
                )
            else:
                settled = state.speech.confirm_any_started(
                    before=time.monotonic(), evidence="speech-update"
                )
            if settled:
                logger.debug(
                    "speech confirmed by webhook call=%s type=%s seqs=%s",
                    state.call_ref,
                    event.type,
                    ",".join(str(d.seq) for d in settled),
                )
            return JSONResponse({})
        caller_event = parse_caller_speech_event(body)
        if caller_event is not None:
            state = registry.peek(caller_event.call_id)
            if state is not None:
                state.set_caller_speaking(caller_event.started)
        return JSONResponse({})

    if settings.route_secret is None:
        # Vapi's OpenAI client appends /chat/completions to the configured base URL;
        # the doubled path tolerates a URL configured WITH the suffix
        # (docs/integration-contracts.md section 1.1 doc conflict).
        app.post("/chat/completions")(handle_chat)
        app.post("/chat/completions/chat/completions")(handle_chat)
        if journal is not None:
            app.get("/debug/acks/{call_ref}")(handle_debug_acks)
        if settings.vapi_server_secret is not None:
            app.post("/vapi/server")(handle_vapi_server)
    else:
        expected_secret = settings.route_secret.get_secret_value()

        def _route_secret_ok(secret: str) -> bool:
            return secrets_mod.compare_digest(secret.encode(), expected_secret.encode())

        async def handle_chat_secret(request: Request, secret: str) -> Response:
            if not _route_secret_ok(secret):
                logger.warning("chat request rejected reason=bad_route_secret")
                return JSONResponse({"error": {"message": "not found"}}, status_code=404)
            return await handle_chat(request)

        async def handle_debug_acks_secret(
            request: Request, secret: str, call_ref: str
        ) -> Response:
            # The debug surface lives behind the route secret too: it must never be
            # reachable on a path the chat endpoint itself 404s.
            if not _route_secret_ok(secret):
                logger.warning("debug acks request rejected reason=bad_route_secret")
                return JSONResponse({"error": {"message": "not found"}}, status_code=404)
            return await handle_debug_acks(request, call_ref)

        async def handle_vapi_server_secret(request: Request, secret: str) -> Response:
            # Behind the route secret as well as its own: the webhook can make this
            # adapter stop re-speaking a dropped answer (by confirming it spoken), so
            # it must not be reachable on a path the chat endpoint itself 404s.
            if not _route_secret_ok(secret):
                logger.warning("vapi server event rejected reason=bad_route_secret")
                return JSONResponse({"error": {"message": "not found"}}, status_code=404)
            return await handle_vapi_server(request)

        app.post("/v/{secret}/chat/completions")(handle_chat_secret)
        app.post("/v/{secret}/chat/completions/chat/completions")(handle_chat_secret)
        if journal is not None:
            app.get("/v/{secret}/debug/acks/{call_ref}")(handle_debug_acks_secret)
        if settings.vapi_server_secret is not None:
            app.post("/v/{secret}/vapi/server")(handle_vapi_server_secret)

    return app
