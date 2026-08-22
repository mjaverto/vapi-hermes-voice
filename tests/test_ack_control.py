"""Regression tests for acknowledgement delivery (accepted-but-silent ack, R2).

Live reproduction (E2E harness, real Vapi, real adapter -- see
tests/e2e/README.md and docs/integration-contracts.md section 1.6): an
acknowledgement written into the model.url SSE stream, terminated with the
documented ``<flush />`` control token, is accepted by Vapi immediately (echoed
back as a ``model-output``/``voice-input`` event within ~1 ms) but is NOT
reliably turned into audio once that same stream then sits idle for more than a
few seconds -- exactly what happens whenever the Hermes turn behind it runs
long. Confirmed on an isolated probe carrying no Hermes traffic at all, and
confirmed on a real call (01a02681): the SAME call's turn 1 rendered its
SSE-embedded ack in well under a second because the response ended right
behind the flush, while turn 2's identical ack was never rendered at all
because the response stayed open, draining Hermes, for 11.5 s behind it.

That is the fix this file defends: the acknowledgement is ALWAYS delivered by
writing it into the model.url stream and ending the response immediately
behind it -- never by waiting on an outbound network call first (an earlier
revision tried Vapi's Live Call Control endpoint on this path; see
vapi_control.py for the measurements that retired it: the control origin
closes every connection it answers, so a "warm" connection can never be
reused, and a cold handshake is a real, if rare, multi-second tail risk this
critical path has no business carrying once the SSE path alone is already
reliable). Whatever Hermes goes on to produce is delivered afterward through
Live Call Control instead, unconditionally, on a background continuation --
because once the response has ended there is no other channel left.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
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
from vapi_hermes_voice.turns import ANSWER_DELIVERY_FAILED_LINE, stream_turn
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


# --- 1. the ack is always SSE-embedded, whether or not control is available ---


async def test_ack_always_written_into_the_sse_stream() -> None:
    """The acknowledgement never waits on control, present or not, working or not."""
    settings = make_settings(filler_after_seconds=0.05, filler_phrases=["One moment."])
    control, requests = make_control(lambda r: httpx.Response(200, json={"status": "ok"}))
    events = [(0.20, delta("Paris.")), (0.02, done())]

    parsed = await run(events, settings, control=control, control_url=CONTROL_URL)
    await control.aclose()

    fillers = filler_chunks(parsed, settings.filler_phrases)
    assert len(fillers) == 1
    assert "<flush />" in fillers[0]
    # The ack itself must never be the content of a control POST: only the real
    # answer travels that way, once the ack has already ended the response.
    for r in requests:
        assert json.loads(r.read())["content"] != "One moment."


# --- 2. control available: response ends right behind the ack; the answer,
# --- not the ack, is what travels through control -----------------------------


async def test_ack_ends_the_response_and_the_answer_travels_via_control() -> None:
    settings = make_settings(filler_after_seconds=0.05, filler_phrases=["One moment."])
    control, requests = make_control(lambda r: httpx.Response(200, json={"status": "ok"}))
    events = [(0.20, delta("Paris.")), (0.02, done())]

    parsed = await run(events, settings, control=control, control_url=CONTROL_URL)
    await control.aclose()

    assert len(requests) == 1, "only the answer, never the ack, should reach control"
    payload = json.loads(requests[0].read())
    assert payload == {"type": "say", "content": "Paris."}

    # The response contains the ack and nothing else: no real content, because the
    # answer was diverted to control once the response ended behind the ack.
    chunks = content_chunks(parsed)
    assert len(chunks) == 1
    assert "One moment." in chunks[0]
    assert parsed[-1] == "[DONE]"


# --- 3. no control at all: exactly the pre-existing behaviour, unaffected ------


async def test_no_control_available_keeps_draining_into_the_same_stream() -> None:
    """No control_url on the request: the answer must still arrive, on the SAME
    response, exactly as it always did -- this is the one case nothing here can
    improve (there is no other channel to hand the answer to), so it must not
    regress either.
    """
    settings = make_settings(filler_after_seconds=0.05, filler_phrases=["One moment."])
    control, requests = make_control(lambda r: httpx.Response(200, json={"status": "ok"}))
    events = [(0.20, delta("Paris.")), (0.02, done())]

    parsed = await run(events, settings, control=control, control_url=None)
    await control.aclose()

    assert requests == [], "no control URL on the request: control must never be touched"
    fillers = filler_chunks(parsed, settings.filler_phrases)
    assert len(fillers) == 1
    reals = [c for c in content_chunks(parsed) if c not in fillers]
    assert "".join(reals).strip() == "Paris."


async def test_no_control_client_at_all_keeps_draining_into_the_same_stream() -> None:
    settings = make_settings(filler_after_seconds=0.05, filler_phrases=["One moment."])
    events = [(0.20, delta("Paris.")), (0.02, done())]

    parsed = await run(events, settings, control=None, control_url=CONTROL_URL)

    fillers = filler_chunks(parsed, settings.filler_phrases)
    assert len(fillers) == 1
    reals = [c for c in content_chunks(parsed) if c not in fillers]
    assert "".join(reals).strip() == "Paris."


# --- 4. once handed off, further Hermes events (e.g. a tool_start) are drained
# --- by the background continuation, into one final control call -------------


async def test_events_after_handoff_are_drained_into_one_final_control_call() -> None:
    """After the ack hands off, a subsequent tool_start must not reopen dead-air
    polling (the SSE generator has already returned) -- it is just one more event
    the background continuation drains on its way to the final ``say``.
    """
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

    assert len(requests) == 1, "exactly the drained-through answer, once"
    answer_payload = json.loads(requests[0].read())
    assert answer_payload == {"type": "say", "content": "Paris."}
    reals = [
        c for c in content_chunks(parsed) if c not in filler_chunks(parsed, settings.filler_phrases)
    ]
    assert reals == []


# --- 5. VapiControlClient.say itself: URL scheme guard ------------------------


async def test_say_rejects_non_https_control_url() -> None:
    control, requests = make_control(lambda r: httpx.Response(200, json={"status": "ok"}))
    outcome = await control.say(
        "http://not-https.example/control", "hi", call_ref="ref-1", timeout=1.0
    )
    await control.aclose()
    assert outcome.delivered is False
    assert requests == [], "an unsafe control URL must never be requested"


# --- 6. answer delivery: one retry on a plausibly transient failure -----------


async def test_answer_delivery_retries_once_after_a_timeout_then_succeeds() -> None:
    settings = make_settings(filler_after_seconds=0.03, filler_phrases=["One moment."])
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ReadTimeout("stalled", request=request)
        return httpx.Response(200, json={"status": "ok"})

    control, requests = make_control(handler)
    events = [(0.20, delta("Paris.")), (0.02, done())]

    parsed = await run(events, settings, control=control, control_url=CONTROL_URL)
    await control.aclose()

    assert attempts == 2, "a timeout must be retried exactly once"
    assert len(requests) == 2
    for r in requests:
        assert json.loads(r.read()) == {"type": "say", "content": "Paris."}
    assert content_chunks(parsed) == [
        c for c in content_chunks(parsed) if c in filler_chunks(parsed, settings.filler_phrases)
    ]


async def test_answer_delivery_gives_up_after_the_retry_window_and_speaks_a_fallback(
    caplog: Any,
) -> None:
    settings = make_settings(
        filler_after_seconds=0.03,
        filler_phrases=["One moment."],
        control_answer_timeout_seconds=0.05,
        control_answer_retry_gap_seconds=0.05,
        control_answer_max_wait_seconds=0.2,
    )
    control, requests = make_control(lambda r: httpx.Response(500, text="boom"))
    events = [(0.20, delta("Paris.")), (0.02, done())]

    with caplog.at_level(logging.WARNING):
        await run(events, settings, control=control, control_url=CONTROL_URL)
    await control.aclose()

    assert len(requests) >= 3, "a persistent 5xx must be retried more than once"
    contents = [json.loads(r.read())["content"] for r in requests]
    assert contents[-1] == ANSWER_DELIVERY_FAILED_LINE
    assert any(
        "answer delivery failed" in record.message and "speaking a fallback line" in record.message
        for record in caplog.records
    )


async def test_answer_delivery_does_not_retry_a_client_rejection(caplog: Any) -> None:
    """A 4xx is Vapi declining the request outright -- most plausibly because the
    call has already moved past this turn -- so a second, identical POST is not
    retried, and this is not logged as an operational warning: there is nothing
    an operator can act on.
    """
    settings = make_settings(filler_after_seconds=0.03, filler_phrases=["One moment."])
    control, requests = make_control(lambda r: httpx.Response(400, text="bad request"))
    events = [(0.20, delta("Paris.")), (0.02, done())]

    with caplog.at_level(logging.INFO):
        await run(events, settings, control=control, control_url=CONTROL_URL)
    await control.aclose()

    assert len(requests) == 1, "a 4xx must not be retried"
    assert not any(record.levelno >= logging.WARNING for record in caplog.records)
    assert any(
        "declined" in record.message and "status=400" in record.message for record in caplog.records
    )
