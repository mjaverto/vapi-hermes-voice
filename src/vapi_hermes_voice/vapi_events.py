"""Vapi Custom LLM wire types: inbound chat request, outbound OpenAI SSE chunks.

Inbound: Vapi POSTs an OpenAI-chat-completions-shaped JSON body; with the default
``metadataSendMode: "variable"`` it additionally carries ``call``, ``customer``,
``phoneNumber``, and ``metadata`` objects (docs/integration-contracts.md section 1.2).
Outbound: OpenAI ``chat.completion.chunk`` SSE frames terminated by ``data: [DONE]``
(section 1.3).

Privacy: parse errors NEVER include payload content in their messages -- transcripts
are private.
"""

from __future__ import annotations

import json
import secrets
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

MODEL_LABEL = "hermes"


class VapiProtocolError(Exception):
    """A request body from Vapi could not be parsed into a chat request."""


class OversizedPayloadError(VapiProtocolError):
    """An inbound request body exceeded the configured byte limit."""


class ChatMessage(BaseModel):
    """One OpenAI-style conversation message. Extra fields are dropped."""

    model_config = ConfigDict(extra="ignore")

    role: str
    content: str | None = None


class VapiChatRequest(BaseModel):
    """The parsed, adapter-relevant view of one Vapi chat-completions request."""

    messages: list[ChatMessage]
    stream: bool = True
    call_id: str | None = None
    call_type: str | None = None  # inboundPhoneCall | outboundPhoneCall | webCall
    customer_number: str | None = None
    metadata: dict[str, Any] | None = None
    tools_present: bool = False

    @property
    def direction(self) -> str:
        """'outbound' for outboundPhoneCall, else 'inbound'."""
        return "outbound" if self.call_type == "outboundPhoneCall" else "inbound"


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _coerce_content(value: object) -> str | None:
    """Tolerate OpenAI list-of-parts content by joining its text parts."""
    if isinstance(value, str) or value is None:
        return value
    if isinstance(value, list):
        parts = [
            part["text"]
            for part in value
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        if parts:
            return "".join(parts)
    return None


def parse_chat_request(raw: bytes, *, max_bytes: int) -> VapiChatRequest:
    """Parse one inbound Vapi chat-completions body.

    Raises :class:`OversizedPayloadError` (before any JSON parsing) when the body
    exceeds ``max_bytes``; :class:`VapiProtocolError` on malformed JSON or a missing
    ``messages`` array. Individual malformed message entries are dropped, not fatal
    (defensive parse). Error messages never contain payload content.
    """
    if len(raw) > max_bytes:
        raise OversizedPayloadError(
            f"request body of {len(raw)} bytes exceeds limit of {max_bytes} bytes"
        )
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise VapiProtocolError("request body is not valid JSON") from None
    if not isinstance(payload, dict):
        raise VapiProtocolError("request body is not a JSON object")
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        raise VapiProtocolError("request body has no messages array")
    messages: list[ChatMessage] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        try:
            messages.append(
                ChatMessage.model_validate(
                    {**item, "content": _coerce_content(item.get("content"))}
                )
            )
        except ValidationError:
            continue  # deliberately unchained: entry content must never leak
    call = payload.get("call")
    call = call if isinstance(call, dict) else {}
    customer = payload.get("customer")
    customer = customer if isinstance(customer, dict) else {}
    metadata = payload.get("metadata")
    stream = payload.get("stream")
    tools = payload.get("tools")
    return VapiChatRequest(
        messages=messages,
        stream=stream if isinstance(stream, bool) else True,
        call_id=_string_or_none(call.get("id")),
        call_type=_string_or_none(call.get("type")),
        customer_number=_string_or_none(customer.get("number")),
        metadata=metadata if isinstance(metadata, dict) else None,
        tools_present=isinstance(tools, list) and len(tools) > 0,
    )


class ChunkWriter:
    """Builds the OpenAI SSE chunk lines for one streamed completion."""

    def __init__(self) -> None:
        self._id = "chatcmpl-" + secrets.token_hex(12)
        self._created = int(time.time())

    def _chunk(self, delta: dict[str, Any], finish_reason: str | None) -> str:
        payload = {
            "id": self._id,
            "object": "chat.completion.chunk",
            "created": self._created,
            "model": MODEL_LABEL,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    def role(self) -> str:
        return self._chunk({"role": "assistant"}, None)

    def content(self, text: str) -> str:
        return self._chunk({"content": text}, None)

    def finish(self) -> str:
        return self._chunk({}, "stop")

    @staticmethod
    def done() -> str:
        return "data: [DONE]\n\n"


def completion_json(text: str) -> dict[str, Any]:
    """A complete non-streaming ``chat.completion`` object (``"stream": false``)."""
    return {
        "id": "chatcmpl-" + secrets.token_hex(12),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_LABEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
