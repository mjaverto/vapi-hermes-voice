"""Tests for vapi_hermes_voice.vapi_events: request parsing and SSE chunk emission."""

from __future__ import annotations

import json
from typing import Any

import pytest

from vapi_hermes_voice.vapi_events import (
    ChunkWriter,
    OversizedPayloadError,
    VapiProtocolError,
    completion_json,
    parse_chat_request,
)

MAX = 1_000_000


def _body(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload).encode()


def vapi_payload(**overrides: Any) -> dict[str, Any]:
    """A realistic Vapi custom-llm request body (metadataSendMode=variable)."""
    payload: dict[str, Any] = {
        "model": "hermes",
        "temperature": 0.5,
        "max_tokens": 250,
        "stream": True,
        "messages": [
            {"role": "system", "content": "You are Mike's assistant."},
            {"role": "assistant", "content": "Hi! How can I help?"},
            {"role": "user", "content": "What's on my calendar?"},
        ],
        "call": {"id": "call-abc-123", "type": "inboundPhoneCall", "orgId": "org-1"},
        "customer": {"number": "+15551234567"},
        "phoneNumber": {"number": "+15559990000", "provider": "vapi"},
        "metadata": {"tenant": "home"},
    }
    payload.update(overrides)
    return payload


class TestParseChatRequest:
    def test_happy_path(self) -> None:
        chat = parse_chat_request(_body(vapi_payload()), max_bytes=MAX)
        assert [m.role for m in chat.messages] == ["system", "assistant", "user"]
        assert chat.messages[-1].content == "What's on my calendar?"
        assert chat.stream is True
        assert chat.call_id == "call-abc-123"
        assert chat.call_type == "inboundPhoneCall"
        assert chat.direction == "inbound"
        assert chat.customer_number == "+15551234567"
        assert chat.metadata == {"tenant": "home"}
        assert chat.tools_present is False

    def test_outbound_direction(self) -> None:
        chat = parse_chat_request(
            _body(vapi_payload(call={"id": "c1", "type": "outboundPhoneCall"})), max_bytes=MAX
        )
        assert chat.direction == "outbound"

    def test_web_call_has_no_customer_number(self) -> None:
        payload = vapi_payload(call={"id": "c2", "type": "webCall"})
        del payload["customer"]
        chat = parse_chat_request(_body(payload), max_bytes=MAX)
        assert chat.customer_number is None
        assert chat.direction == "inbound"

    def test_metadata_send_mode_off_shape(self) -> None:
        # metadataSendMode "off": payload is just {messages, ...openai fields}.
        chat = parse_chat_request(
            _body({"model": "x", "messages": [{"role": "user", "content": "hi"}]}),
            max_bytes=MAX,
        )
        assert chat.call_id is None
        assert chat.customer_number is None
        assert chat.stream is True  # assumed streaming unless explicitly false

    def test_stream_false_honored(self) -> None:
        chat = parse_chat_request(_body(vapi_payload(stream=False)), max_bytes=MAX)
        assert chat.stream is False

    def test_tools_present_flag(self) -> None:
        chat = parse_chat_request(
            _body(vapi_payload(tools=[{"type": "function", "function": {"name": "t"}}])),
            max_bytes=MAX,
        )
        assert chat.tools_present is True

    def test_list_of_parts_content_joined(self) -> None:
        payload = vapi_payload(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "part one "},
                        {"type": "text", "text": "part two"},
                    ],
                }
            ]
        )
        chat = parse_chat_request(_body(payload), max_bytes=MAX)
        assert chat.messages[0].content == "part one part two"

    def test_malformed_message_entries_dropped(self) -> None:
        payload = vapi_payload(
            messages=["not-a-dict", {"content": "no role"}, {"role": "user", "content": "ok"}]
        )
        chat = parse_chat_request(_body(payload), max_bytes=MAX)
        assert len(chat.messages) == 1
        assert chat.messages[0].content == "ok"

    def test_oversized_rejected_before_parse(self) -> None:
        with pytest.raises(OversizedPayloadError):
            parse_chat_request(b"x" * 101, max_bytes=100)

    def test_invalid_json_rejected_without_content_leak(self) -> None:
        with pytest.raises(VapiProtocolError) as excinfo:
            parse_chat_request(b"secret-transcript-fragment{", max_bytes=MAX)
        assert "secret-transcript-fragment" not in str(excinfo.value)

    def test_non_object_rejected(self) -> None:
        with pytest.raises(VapiProtocolError):
            parse_chat_request(b"[1, 2, 3]", max_bytes=MAX)

    def test_missing_messages_rejected(self) -> None:
        with pytest.raises(VapiProtocolError):
            parse_chat_request(_body({"model": "x"}), max_bytes=MAX)


def _parse_sse(line: str) -> dict[str, Any]:
    assert line.startswith("data: ")
    assert line.endswith("\n\n")
    parsed = json.loads(line[len("data: ") : -2])
    assert isinstance(parsed, dict)
    return parsed


class TestChunkWriter:
    def test_openai_chunk_shape(self) -> None:
        writer = ChunkWriter()
        chunk = _parse_sse(writer.content("hello"))
        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["id"].startswith("chatcmpl-")
        assert isinstance(chunk["created"], int)
        assert chunk["choices"] == [
            {"index": 0, "delta": {"content": "hello"}, "finish_reason": None}
        ]

    def test_role_then_finish_then_done(self) -> None:
        writer = ChunkWriter()
        role = _parse_sse(writer.role())
        assert role["choices"][0]["delta"] == {"role": "assistant"}
        finish = _parse_sse(writer.finish())
        assert finish["choices"][0]["delta"] == {}
        assert finish["choices"][0]["finish_reason"] == "stop"
        assert writer.done() == "data: [DONE]\n\n"

    def test_stable_id_across_chunks(self) -> None:
        writer = ChunkWriter()
        first = _parse_sse(writer.role())
        second = _parse_sse(writer.content("x"))
        assert first["id"] == second["id"]

    def test_distinct_ids_across_turns(self) -> None:
        assert _parse_sse(ChunkWriter().role())["id"] != _parse_sse(ChunkWriter().role())["id"]


def test_completion_json_shape() -> None:
    body = completion_json("the answer")
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "the answer"}
    assert body["choices"][0]["finish_reason"] == "stop"
