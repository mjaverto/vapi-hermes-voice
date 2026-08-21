"""Regression tests for the acknowledgement delivery fix (accepted-but-silent ack).

Live reproduction (E2E harness, real Vapi, real adapter -- see
tests/e2e/README.md and docs/integration-contracts.md section 1.6): an
acknowledgement written into the model.url SSE stream, terminated with the
documented ``<flush />`` control token, is accepted by Vapi immediately (echoed
back as a ``model-output``/``voice-input`` event within ~1 ms) but is NOT
reliably turned into audio once that same stream then sits idle for more than a
few seconds -- exactly what happens whenever the Hermes turn behind it runs
long. Confirmed on an isolated probe carrying no Hermes traffic at all, and
confirmed neither omitting ``<flush />`` nor adding byte-level keepalive changes
it: the stall is about the SSE stream's own state, not the flush token.

Only a real Vapi call can prove the fix actually gets AUDIO out of Vapi -- that
proof is in the harness runs referenced from the PR description, not here. What
THIS layer can and must defend is the adapter's own contract: given a
``call.monitor.controlUrl`` on the request, a dead-air acknowledgement is
delivered through it (POST ``{"type": "say", "content": <phrase>}``) instead of
being written into the stalled SSE stream, so it is never left stranded behind a
long Hermes turn; and every existing fallback/atomicity guarantee still holds
when no control URL is available or the control request itself fails.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

import httpx

from test_turns import (
    _parse_chunk,
    _ScriptedHermes,
    content_chunks,
    delta,
    done,
    filler_chunks,
    make_settings,
    make_state,
    tool_start,
)
from vapi_hermes_voice.config import Settings
from vapi_hermes_voice.turns import stream_turn
from vapi_hermes_voice.vapi_control import VapiControlClient

CONTROL_URL = "https://phone-call-websocket.vapi.ai/call-1/control"


def make_control(
    handler: Any,
) -> tuple[VapiControlClient, list[httpx.Request]]:
    """A VapiControlClient wired to an in-memory transport; returns (client, requests)."""
    requests: list[httpx.Request] = []

    def _wrapped(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    return VapiControlClient(transport=httpx.MockTransport(_wrapped)), requests


async def run(
    events: list[tuple[float, Any]],
    settings: Settings,
    *,
    control: VapiControlClient | None = None,
    control_url: str | None = None,
) -> list[tuple[str, str] | str]:
    reaping: set[asyncio.Task[None]] = set()
    parsed: list[tuple[str, str] | str] = []
    agen = stream_turn(
        settings=settings,
        hermes=_ScriptedHermes(events),
        control=control,
        control_url=control_url,
        state=make_state(settings),
        instructions="instructions",
        history=[],
        user_input="hello",
        reaping=reaping,
    )
    async for chunk in agen:
        parsed.extend(_parse_chunk(chunk))
    for task in list(reaping):
        with contextlib.suppress(Exception):
            await task
    return parsed


# --- 1. control URL present + control call succeeds: ack goes out via `say`,
# --- never embedded in the SSE stream (would otherwise speak it twice) --------


async def test_dead_air_ack_delivered_via_control_when_url_present() -> None:
    settings = make_settings(filler_after_seconds=0.05, filler_phrases=["One moment."])
    control, requests = make_control(lambda r: httpx.Response(200, json={"status": "ok"}))
    events = [(0.20, delta("Paris.")), (0.02, done())]

    parsed = await run(events, settings, control=control, control_url=CONTROL_URL)
    await control.aclose()

    assert len(requests) == 1, "expected exactly one Live Call Control request"
    assert requests[0].url == httpx.URL(CONTROL_URL)
    body = requests[0].read()
    payload = json.loads(body)
    assert payload == {"type": "say", "content": "One moment."}

    # The ack must NOT also appear in the SSE stream: that would speak it twice
    # (once via `say`, once later if Vapi's stream buffer for the model.url
    # response ever catches up and merges it with the real content).
    chunks = content_chunks(parsed)
    assert not any("One moment." in c for c in chunks), (
        f"ack delivered via control channel leaked into the SSE stream too: {chunks!r}"
    )
    assert any("Paris." in c for c in chunks)


# --- 2. control call fails (non-2xx): falls back to the old SSE-embedded path -


async def test_control_failure_falls_back_to_sse_stream() -> None:
    settings = make_settings(
        filler_after_seconds=0.05, filler_use_flush=True, filler_phrases=["One moment."]
    )
    control, requests = make_control(lambda r: httpx.Response(500, text="internal error"))
    events = [(0.20, delta("Paris.")), (0.02, done())]

    parsed = await run(events, settings, control=control, control_url=CONTROL_URL)
    await control.aclose()

    assert len(requests) == 1, "the control channel must still be tried first"
    fillers = filler_chunks(parsed, settings.filler_phrases)
    assert len(fillers) == 1
    assert "<flush />" in fillers[0], "fallback delivery must keep the flush token"


# --- 3. control call raises (network error): falls back the same way ----------


async def test_control_network_error_falls_back_to_sse_stream() -> None:
    settings = make_settings(filler_after_seconds=0.05, filler_phrases=["One moment."])

    def _raise(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    control, requests = make_control(_raise)
    events = [(0.20, delta("Paris.")), (0.02, done())]

    parsed = await run(events, settings, control=control, control_url=CONTROL_URL)
    await control.aclose()

    assert len(requests) == 1
    fillers = filler_chunks(parsed, settings.filler_phrases)
    assert len(fillers) == 1


# --- 4. no control URL on the request: SSE-embedded path, exactly as before ---


async def test_no_control_url_uses_sse_stream_fallback() -> None:
    settings = make_settings(filler_after_seconds=0.05, filler_phrases=["One moment."])
    control, requests = make_control(lambda r: httpx.Response(200, json={"status": "ok"}))
    events = [(0.20, delta("Paris.")), (0.02, done())]

    # control client is available, but no control_url was on THIS request.
    parsed = await run(events, settings, control=control, control_url=None)
    await control.aclose()

    assert requests == [], "no control URL on the request: the control channel must not be used"
    fillers = filler_chunks(parsed, settings.filler_phrases)
    assert len(fillers) == 1


# --- 5. kill switch: ack_use_call_control=False always uses the SSE fallback --


async def test_ack_use_call_control_disabled_kill_switch() -> None:
    settings = make_settings(
        filler_after_seconds=0.05, filler_phrases=["One moment."], ack_use_call_control=False
    )
    control, requests = make_control(lambda r: httpx.Response(200, json={"status": "ok"}))
    events = [(0.20, delta("Paris.")), (0.02, done())]

    parsed = await run(events, settings, control=control, control_url=CONTROL_URL)
    await control.aclose()

    assert requests == [], "ack_use_call_control=False must never call the control endpoint"
    fillers = filler_chunks(parsed, settings.filler_phrases)
    assert len(fillers) == 1


# --- 6. multiple acks in one long turn: each independently tries control first


async def test_second_ack_in_a_long_turn_also_uses_control() -> None:
    settings = make_settings(
        filler_after_seconds=0.03,
        filler_min_gap_seconds=0.01,
        filler_phrases=["One moment.", "Still checking."],
    )
    control, requests = make_control(lambda r: httpx.Response(200, json={"status": "ok"}))
    events = [
        (0.20, tool_start()),
        (0.20, delta("Paris.")),
        (0.02, done()),
    ]

    parsed = await run(events, settings, control=control, control_url=CONTROL_URL)
    await control.aclose()

    assert len(requests) == 2, "both dead-air acknowledgements should use the control channel"
    chunks = content_chunks(parsed)
    assert not any("One moment." in c or "Still checking." in c for c in chunks)
    assert any("Paris." in c for c in chunks)


# --- 7. VapiControlClient.say itself: URL scheme guard --------------------


async def test_say_rejects_non_https_control_url() -> None:
    control, requests = make_control(lambda r: httpx.Response(200, json={"status": "ok"}))
    delivered = await control.say(
        "http://not-https.example/control", "hi", call_ref="ref-1", timeout=1.0
    )
    await control.aclose()
    assert delivered is False
    assert requests == [], "an unsafe control URL must never be requested"
