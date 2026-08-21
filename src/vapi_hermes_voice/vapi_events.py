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
import re
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


# --- Vapi dynamic variables (assistantOverrides.variableValues) ---
#
# These are UNTRUSTED free text: they arrive from whoever created the call, and a
# call can be created through a path a caller influences. Every value is stripped
# of control characters and whitespace-collapsed (so a value can never forge the
# paragraph breaks that separate sections of the prompt we build) and then length
# capped. Values are never logged; see CallVariables.log_summary.
MAX_PURPOSE_CHARS = 400
MAX_CALLEE_CHARS = 120
# The ready-to-speak reason for an outbound call. Separate from `purpose` because
# `purpose` is MODEL-FACING and routinely carries section labels, option lists and
# internal constraints ("Goal: ... Mike is free weekday mornings.") that must never
# reach a loudspeaker. Short by design: it is one spoken clause, not a brief.
MAX_SPOKEN_REASON_CHARS = 200

# Supplementary context: every entry this adapter has no dedicated field for. A
# live Hermes-issued call sent `call_purpose`, `patient_name` and
# `patient_context`; the literal-key lookup below understood none of them, so all
# three were discarded without a trace and the call ran with no objective at all.
# Unrecognized entries are now surfaced to the model as labelled data lines --
# bounded, because an unbounded prompt is latency and injection surface, not a
# feature.
MAX_CONTEXT_ENTRIES = 8
MAX_CONTEXT_LABEL_CHARS = 40
MAX_CONTEXT_VALUE_CHARS = 400

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_VARIABLE_WS_RE = re.compile(r"\s+")

# Keys the adapter understands, in precedence order: the first alias present in one
# variableValues object supplies the field. Matching is case- and
# separator-insensitive (see _normalize_key), so `call_purpose`, `callPurpose` and
# `Call-Purpose` are one key.
PURPOSE_ALIASES: tuple[str, ...] = ("purpose", "call_purpose", "objective", "task", "goal")
CALLEE_ALIASES: tuple[str, ...] = ("callee", "callee_name", "recipient", "calling")
# The spoken reason-for-calling clause (see MAX_SPOKEN_REASON_CHARS). `opening_line`
# is accepted as an alias so a caller written against either name works.
SPOKEN_REASON_ALIASES: tuple[str, ...] = (
    "spoken_reason",
    "reason",
    "reason_for_call",
    "opening_line",
    "spoken_purpose",
)

_SEPARATOR_RE = re.compile(r"[\s_\-.]+")


def _normalize_key(key: str) -> str:
    """Casefolded, separator-free form of a variable key, for alias matching."""
    return _SEPARATOR_RE.sub("", key).casefold()


_PURPOSE_KEYS: tuple[str, ...] = tuple(_normalize_key(alias) for alias in PURPOSE_ALIASES)
_CALLEE_KEYS: tuple[str, ...] = tuple(_normalize_key(alias) for alias in CALLEE_ALIASES)
_SPOKEN_REASON_KEYS: tuple[str, ...] = tuple(
    _normalize_key(alias) for alias in SPOKEN_REASON_ALIASES
)
_RESERVED_KEYS: frozenset[str] = frozenset(_PURPOSE_KEYS + _CALLEE_KEYS + _SPOKEN_REASON_KEYS)

# Where variableValues can appear in a Custom LLM request body, most specific
# first. `call.assistantOverrides.variableValues` is the documented location: the
# Call object carries `assistantOverrides`, which carries `variableValues` (see the
# POST https://api.vapi.ai/call reference). The rest are accepted because the same
# override object is echoed at different depths depending on how the call was made.
_VARIABLE_PATHS: tuple[tuple[str, ...], ...] = (
    ("call", "assistantOverrides", "variableValues"),
    ("assistantOverrides", "variableValues"),
    ("variableValues",),
    ("call", "variableValues"),
)


def _clean_variable(value: object, *, limit: int) -> str | None:
    """Sanitize one untrusted variable value; None when unusable.

    Non-strings (numbers, nested objects, null) are ignored rather than coerced:
    the adapter only understands free-form text for these keys.
    """
    if not isinstance(value, str):
        return None
    text = _VARIABLE_WS_RE.sub(" ", _CONTROL_RE.sub(" ", value)).strip()
    if not text:
        return None
    return text[:limit].rstrip() if len(text) > limit else text


class CallVariables(BaseModel):
    """The Vapi dynamic variables of one call, already sanitized.

    ``purpose`` is the objective of an outbound task call ("call Dr. Patel and move
    the cardiology recheck to Tuesday afternoon") and is written for the MODEL;
    ``spoken_reason`` is the operator's ready-to-speak version of it, the only field
    here intended to be read aloud as given; ``callee`` optionally describes who is
    being called; ``context`` carries every other entry the operator attached to the
    call as ``(label, value)`` pairs. All of it is untrusted text: it describes a
    task, and is never treated as configuration or as instructions that can relax
    the rules -- ``spoken_reason`` included, which is still scrubbed before it is
    spoken (see :func:`vapi_hermes_voice.speech.speakable_reason`).
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    purpose: str | None = None
    callee: str | None = None
    # The reason for the call, as the operator wants it said out loud. Optional: with
    # no value the adapter reduces `purpose` instead, conservatively.
    spoken_reason: str | None = None
    # Sanitized (label, value) pairs for entries with no dedicated field -- real
    # content (a live call's patient_context) that earlier versions dropped silently.
    context: tuple[tuple[str, str], ...] = ()
    # Labels of every entry with no dedicated field, including ones whose value was
    # unusable. Logged -- KEYS ONLY -- so a mis-named variable is visible, not silent.
    unknown_keys: tuple[str, ...] = ()
    # True when a variableValues object carrying at least one entry was present, even
    # if nothing in it was understood: the "call has task variables" log must fire for
    # a payload the adapter could make no use of at all.
    has_values: bool = False

    def log_summary(self) -> str:
        """Lengths and counts only: variable text is untrusted and never logged."""
        return (
            f"purpose_chars={len(self.purpose or '')}"
            f" spoken_reason_chars={len(self.spoken_reason or '')}"
            f" callee_chars={len(self.callee or '')}"
            f" context_entries={len(self.context)}"
            f" unknown_keys={len(self.unknown_keys)}"
        )


def _dig(payload: object, path: tuple[str, ...]) -> dict[str, Any] | None:
    node: object = payload
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node if isinstance(node, dict) else None


def _first_alias(
    folded: dict[str, tuple[str, object]], aliases: tuple[str, ...], *, limit: int
) -> str | None:
    """First usable value among ``aliases``, in alias-precedence order."""
    for alias in aliases:
        entry = folded.get(alias)
        if entry is None:
            continue
        text = _clean_variable(entry[1], limit=limit)
        if text is not None:
            return text
    return None


def extract_call_variables(payload: dict[str, Any]) -> CallVariables:
    """Pull the dynamic variables out of one request body.

    The objective, the spoken reason and the callee are each taken from the first
    alias present at the most specific location that supplies one, so a body
    carrying ``call_purpose`` and ``callee`` at different depths still yields both.
    Every remaining entry becomes supplementary ``context`` instead of being
    dropped, and its key is recorded in ``unknown_keys`` for a keys-only warning.
    Missing locations, unknown keys, and non-string values are ignored, never fatal.
    """
    purpose: str | None = None
    callee: str | None = None
    spoken_reason: str | None = None
    context: list[tuple[str, str]] = []
    unknown: list[str] = []
    seen: set[str] = set()
    has_values = False
    for path in _VARIABLE_PATHS:
        values = _dig(payload, path)
        if not values:
            continue
        has_values = True
        folded: dict[str, tuple[str, object]] = {}
        for key, value in values.items():
            if isinstance(key, str):
                folded.setdefault(_normalize_key(key), (key, value))
        if purpose is None:
            purpose = _first_alias(folded, _PURPOSE_KEYS, limit=MAX_PURPOSE_CHARS)
        if callee is None:
            callee = _first_alias(folded, _CALLEE_KEYS, limit=MAX_CALLEE_CHARS)
        if spoken_reason is None:
            spoken_reason = _first_alias(folded, _SPOKEN_REASON_KEYS, limit=MAX_SPOKEN_REASON_CHARS)
        for normalized, (key, value) in folded.items():
            if normalized in _RESERVED_KEYS or normalized in seen:
                continue
            seen.add(normalized)
            label = _clean_variable(key, limit=MAX_CONTEXT_LABEL_CHARS)
            if label is None:
                continue
            unknown.append(label)
            text = _clean_variable(value, limit=MAX_CONTEXT_VALUE_CHARS)
            if text is not None and len(context) < MAX_CONTEXT_ENTRIES:
                context.append((label, text))
    return CallVariables(
        purpose=purpose,
        spoken_reason=spoken_reason,
        callee=callee,
        context=tuple(context),
        unknown_keys=tuple(unknown),
        has_values=has_values,
    )


class VapiChatRequest(BaseModel):
    """The parsed, adapter-relevant view of one Vapi chat-completions request."""

    messages: list[ChatMessage]
    stream: bool = True
    call_id: str | None = None
    call_type: str | None = None  # inboundPhoneCall | outboundPhoneCall | webCall
    # Vapi's Live Call Control endpoint for THIS call (docs/integration-contracts.md
    # section 1.6): POSTing {"type": "say", "content": <text>} here speaks it, in
    # ~0.3s, regardless of what the model.url SSE stream is doing. It is present on
    # every Custom LLM request body already -- no assistant config change needed --
    # and is the delivery path for acknowledgements: a <flush />-terminated chunk
    # left alone in an otherwise idle stream for more than a few seconds is not
    # reliably rendered by Vapi's chunk-plan TTS pipeline (measured live and on an
    # isolated probe: unspoken for the whole stall, or spoken late, garbled, and
    # merged with a second buffered fragment once the stream finally progresses).
    control_url: str | None = None
    customer_number: str | None = None
    metadata: dict[str, Any] | None = None
    tools_present: bool = False
    variables: CallVariables = CallVariables()

    @property
    def direction(self) -> str:
        """'outbound' for outboundPhoneCall, else 'inbound'."""
        return "outbound" if self.call_type == "outboundPhoneCall" else "inbound"

    def callee_is_principal(self, *, principal: str, principal_number: str | None) -> bool:
        """True when the person on the other end of an outbound call IS the principal.

        Resolution order:

        1. ``principal_number`` vs ``customer.number``. Vapi's ``customer.number`` is
           signalling-derived, so when the operator configured the principal's own
           number it is authoritative and the free-text ``callee`` is not consulted.
        2. Otherwise an exact (case- and whitespace-insensitive) ``callee`` match.
           Deliberately not a substring test: "Mike's doctor" contains "Mike", and
           reading that as "we called Mike" would drop the AI disclosure owed to a
           third party and address a stranger by the principal's name.
        3. Otherwise False -- assume a third party, which is the safe default and
           preserves adapter behavior for operators who never set a principal number.

        Always False for inbound calls: there the counterparty is whoever dialed in.
        """
        if self.direction != "outbound":
            return False
        if principal_number is not None and self.customer_number is not None:
            return self.customer_number == principal_number
        callee = self.variables.callee
        if callee is None:
            return False
        return callee.strip().casefold() == principal.strip().casefold()


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
    monitor = call.get("monitor")
    monitor = monitor if isinstance(monitor, dict) else {}
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
        control_url=_string_or_none(monitor.get("controlUrl")),
        customer_number=_string_or_none(customer.get("number")),
        metadata=metadata if isinstance(metadata, dict) else None,
        tools_present=isinstance(tools, list) and len(tools) > 0,
        variables=extract_call_variables(payload),
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
