"""End-to-end smoke test for the Vapi <-> Hermes voice adapter.

Drives one realistic call -- two turns of Vapi-shaped POSTs with growing history, a
sanitized multi-delta answer, session continuity, and a mid-stream hangup -- through
the real ASGI stack (``create_app``) against the programmable fake Hermes in
``tests/fake_hermes.py``. No live Vapi or Hermes credentials are used anywhere.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from typing import Any

from starlette.testclient import TestClient

from fake_hermes import FakeHermesState, FakeScript, build_fake_hermes_transport
from vapi_hermes_voice.config import Settings
from vapi_hermes_voice.server import create_app

API_KEY = "adapter-key-0123456789"
AUTH = {"Authorization": f"Bearer {API_KEY}"}
ALLOWED_NUMBER = "+15551230000"
CALL_ID = "call-smoke-0001"


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "hermes_base_url": "http://fake-hermes.invalid",
        "hermes_api_key": "test-api-key",
        "adapter_api_key": API_KEY,
        "warmup_on_start": False,
        "hermes_first_token_timeout": 2.0,
        "hermes_turn_timeout": 5.0,
        "hermes_stop_timeout": 1.0,
        "filler_after_seconds": 5.0,
        "allowed_callers": [ALLOWED_NUMBER],
        "_env_file": None,
    }
    values.update(overrides)
    return Settings(**values)


@contextlib.contextmanager
def running_app(
    script: FakeScript, **settings_overrides: Any
) -> Iterator[tuple[TestClient, Settings, FakeHermesState]]:
    transport, state = build_fake_hermes_transport(script)
    settings = make_settings(**settings_overrides)
    app = create_app(settings, hermes_transport=transport)
    with TestClient(app) as client:
        yield client, settings, state


def vapi_request(messages: list[dict[str, str]]) -> dict[str, Any]:
    """Exactly what Vapi POSTs with metadataSendMode=variable (contracts section 1.2)."""
    return {
        "model": "hermes",
        "temperature": 0.5,
        "max_tokens": 250,
        "stream": True,
        "messages": messages,
        "call": {"id": CALL_ID, "type": "inboundPhoneCall", "orgId": "org-smoke"},
        "customer": {"number": ALLOWED_NUMBER},
        "phoneNumber": {"number": "+15559990000", "provider": "vapi"},
        "metadata": {},
    }


def spoken(sse_body: str) -> tuple[str, list[dict[str, Any] | str]]:
    events: list[dict[str, Any] | str] = []
    parts: list[str] = []
    for line in sse_body.splitlines():
        if not line.startswith("data: "):
            continue
        data = line[len("data: ") :]
        if data == "[DONE]":
            events.append("[DONE]")
            continue
        chunk = json.loads(data)
        events.append(chunk)
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            parts.append(delta["content"])
    return "".join(parts), events


def test_full_call_end_to_end_through_real_asgi_stack() -> None:
    """One realistic call against the real stack: readiness, two sanitized turns with
    session continuity, and a mid-stream hangup that stops the Hermes run."""
    script = FakeScript(
        deltas=["The capital is ", "**Paris**", ", built on the `Seine`."],
        delta_interval_s=0.02,
    )
    system = {"role": "system", "content": "You are Mike's home assistant."}
    with running_app(script) as (client, _settings, state):
        # Readiness gate: Hermes reachable through the fake.
        assert client.get("/readyz").status_code == 200

        # --- turn 1 ---
        messages: list[dict[str, str]] = [
            system,
            {"role": "assistant", "content": "Hi! How can I help?"},  # Vapi firstMessage
            {"role": "user", "content": "What's the capital of France?"},
        ]
        response = client.post("/chat/completions", json=vapi_request(messages), headers=AUTH)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        turn1_speech, turn1_events = spoken(response.text)
        assert turn1_events[-1] == "[DONE]"
        chunks = [e for e in turn1_events if isinstance(e, dict)]
        assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        assert "Paris" in turn1_speech
        assert "Seine" in turn1_speech
        assert "*" not in turn1_speech
        assert "`" not in turn1_speech

        # --- turn 2: Vapi resends the grown transcript ---
        messages = [
            *messages,
            {"role": "assistant", "content": turn1_speech},
            {"role": "user", "content": "Thanks, that's all."},
        ]
        response = client.post("/chat/completions", json=vapi_request(messages), headers=AUTH)
        assert response.status_code == 200
        turn2_speech, turn2_events = spoken(response.text)
        assert turn2_events[-1] == "[DONE]"
        assert turn2_speech

        # exactly one Hermes run per turn, both sharing the same stable per-call session
        assert len(state.runs) == 2
        assert {run["body"]["session_id"] for run in state.runs} == {
            state.runs[0]["body"]["session_id"]
        }
        assert {run["headers"]["x-hermes-session-key"] for run in state.runs} == {
            state.runs[0]["headers"]["x-hermes-session-key"]
        }
        # turn 2 carried turn 1's answer as history and the new utterance as input
        turn2_body = state.runs[1]["body"]
        assert turn2_body["input"] == "Thanks, that's all."
        assert {"role": "assistant", "content": turn1_speech} in turn2_body["conversation_history"]
        # the Vapi system prompt layered into instructions
        assert "Mike's home assistant" in turn2_body["instructions"]

        # both turns completed cleanly -- no stop was needed
        assert state.stops == []


def test_midcall_hangup_stops_hermes_run() -> None:
    """The caller hangs up mid-answer: the abandoned Hermes run MUST be stopped.

    The fake hangs after the first delta so the turn is genuinely mid-flight when
    the client walks away (a completed answer needs no stop and would race).
    """
    script = FakeScript(
        deltas=["Well, ", "the history of Paris"], delta_interval_s=0.02, hang_after=1
    )
    messages = [
        {"role": "system", "content": "You are Mike's home assistant."},
        {"role": "user", "content": "Tell me everything about Paris."},
    ]
    with running_app(script) as (client, _settings, state):
        with client.stream(
            "POST", "/chat/completions", json=vapi_request(messages), headers=AUTH
        ) as live:
            for line in live.iter_lines():
                if line.startswith("data: ") and '"content"' in line:
                    break  # heard the first words; hang up
        assert len(state.runs) == 1

    # Lifespan shutdown drained the cleanup tasks: the abandoned run was stopped.
    assert len(state.stops) == 1
    assert state.stops[0]["run_id"] == state.runs[0]["run_id"]
