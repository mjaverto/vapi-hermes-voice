"""FastAPI app factory: health endpoints and the Vapi Custom LLM chat endpoint."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets as secrets_mod
from collections.abc import AsyncIterator

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from .call_state import CallStateRegistry
from .config import Settings
from .hermes_client import HermesClient, HermesUnavailableError
from .logredact import redact_phone
from .policy import (
    CallerPolicy,
    build_opening_nudge,
    has_trailing_user_message,
    split_messages,
    truncate_history,
)
from .speech import build_instructions
from .turns import complete_turn, stream_turn
from .vapi_events import (
    ChunkWriter,
    OversizedPayloadError,
    VapiProtocolError,
    completion_json,
    parse_chat_request,
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


def create_app(
    settings: Settings, hermes_transport: httpx.AsyncBaseTransport | None = None
) -> FastAPI:
    """Build the ASGI app; one HermesClient and one turn-slot counter per app.

    ``hermes_transport`` lets tests mount a fake Hermes backend in-process.
    """
    # Plain counter, not a Semaphore: the busy check and the slot grab must be a
    # single atomic step (no await between them). asyncio is single-threaded, so
    # check+increment with no intervening await is race-free.
    active_turns = 0
    policy = CallerPolicy(settings.allowed_callers)
    registry = CallStateRegistry(settings)
    if not policy.enforced:
        logger.warning("caller allowlist is DISABLED (VHV_ALLOWED_CALLERS empty): allowing all")

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        hermes = HermesClient(settings, transport=hermes_transport)
        app.state.hermes = hermes
        app.state.reaping = set()
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
            reaping: set[asyncio.Task[None]] = app.state.reaping
            if reaping:
                # In-flight turn cleanups (Hermes stops) must complete before the
                # shared client closes -- no orphaned runs, even across shutdown.
                await asyncio.gather(*list(reaping), return_exceptions=True)
            await hermes.aclose()

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

        def speak(text: str) -> Response:
            if not chat.stream:
                return JSONResponse(completion_json(text))
            return StreamingResponse(
                _speak_once(text), media_type="text/event-stream", headers=_SSE_HEADERS
            )

        if policy.enforced and not policy.is_allowed(chat.customer_number):
            # Fail closed: an enforced allowlist denies unknown AND absent caller
            # identity (metadataSendMode off, web calls) alike.
            logger.info(
                "call denied call=%s from=%s outcome=denied",
                state.call_ref,
                redact_phone(chat.customer_number) if chat.customer_number else "unknown",
            )
            return speak(DENIED_LINE)
        if active_turns >= settings.max_concurrent_turns:
            logger.warning("turn rejected call=%s reason=busy", state.call_ref)
            return speak(BUSY_LINE)

        messages = truncate_history(chat.messages, settings.max_history_messages)
        if chat.variables.purpose:
            # Lengths only: dynamic-variable text is untrusted and never logged.
            logger.info(
                "call has task variables call=%s direction=%s %s",
                state.call_ref,
                chat.direction,
                chat.variables.log_summary(),
            )
        callee_is_principal = chat.callee_is_principal(
            principal=settings.principal, principal_number=settings.principal_number
        )
        # No trailing user utterance: the adapter synthesizes the opening, and nothing
        # is pending, so this turn must never speak a latency filler.
        opening_turn = not has_trailing_user_message(messages)
        history, user_input, extra = split_messages(
            messages,
            opening=build_opening_nudge(
                settings,
                direction=chat.direction,
                variables=chat.variables,
                callee_is_principal=callee_is_principal,
            ),
        )
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
                    state=state,
                    instructions=instructions,
                    history=history,
                    user_input=user_input,
                    reaping=app.state.reaping,
                    allow_fillers=not opening_turn,
                ):
                    yield line
            finally:
                active_turns -= 1

        return StreamingResponse(stream(), media_type="text/event-stream", headers=_SSE_HEADERS)

    if settings.route_secret is None:
        # Vapi's OpenAI client appends /chat/completions to the configured base URL;
        # the doubled path tolerates a URL configured WITH the suffix
        # (docs/integration-contracts.md section 1.1 doc conflict).
        app.post("/chat/completions")(handle_chat)
        app.post("/chat/completions/chat/completions")(handle_chat)
    else:
        expected_secret = settings.route_secret.get_secret_value()

        async def handle_chat_secret(request: Request, secret: str) -> Response:
            if not secrets_mod.compare_digest(secret.encode(), expected_secret.encode()):
                logger.warning("chat request rejected reason=bad_route_secret")
                return JSONResponse({"error": {"message": "not found"}}, status_code=404)
            return await handle_chat(request)

        app.post("/v/{secret}/chat/completions")(handle_chat_secret)
        app.post("/v/{secret}/chat/completions/chat/completions")(handle_chat_secret)

    return app
