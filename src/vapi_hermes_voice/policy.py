"""Caller allowlisting, session id derivation, and message shaping."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from vapi_hermes_voice.vapi_events import ChatMessage

_E164_RE = re.compile(r"^\+[1-9]\d{6,14}$")

_HISTORY_ROLES = frozenset({"user", "assistant"})

# Synthetic turn input when the request carries no trailing user utterance -- e.g.
# Vapi's `assistant-speaks-first-with-model-generated-message` first-message mode
# asks the model to open the call before the caller has said anything.
OPENING_NUDGE = (
    "The call has just started and the caller has not spoken yet. "
    "Greet them briefly and ask how you can help."
)


class CallerPolicy:
    """Deny-by-default caller allowlist; disabled (allow all) when the list is empty."""

    def __init__(self, allowed: Sequence[str]) -> None:
        for number in allowed:
            if _E164_RE.fullmatch(number) is None:
                raise ValueError("allowlist entries must be E.164 numbers like +15551234567")
        self._allowed: frozenset[str] = frozenset(allowed)

    @property
    def enforced(self) -> bool:
        return bool(self._allowed)

    def is_allowed(self, from_number: str | None) -> bool:
        if not self.enforced:
            return True
        if from_number is None or _E164_RE.fullmatch(from_number) is None:
            return False
        return from_number in self._allowed


def derive_session_ids(call_id: str) -> tuple[str, str]:
    """(session_id, session_key) derived from the Vapi call id, never from phone numbers."""
    session_id = "vhv-" + hashlib.sha256(call_id.encode("utf-8")).hexdigest()[:24]
    session_key = "vhv-key-" + hashlib.sha256(("k" + call_id).encode("utf-8")).hexdigest()[:24]
    return session_id, session_key


def truncate_history(messages: list[ChatMessage], max_len: int) -> list[ChatMessage]:
    """Keep the most recent messages; always keep at least the last one."""
    if not messages:
        return []
    return messages[-max(max_len, 1) :]


def split_messages(messages: list[ChatMessage]) -> tuple[list[dict[str, str]], str, str]:
    """OpenAI messages -> (hermes_history, user_input, extra_instructions).

    System messages (the Vapi assistant's own prompt) are collected into
    ``extra_instructions`` -- they layer onto the adapter's voice instructions, which
    in turn layer onto Hermes's resident system prompt. ``tool``/``function`` frames
    are dropped (Hermes never sees Vapi-side tool plumbing). The trailing user
    message becomes the turn input; without one, :data:`OPENING_NUDGE` is used.
    """
    extra: list[str] = []
    history: list[dict[str, str]] = []
    for message in messages:
        content = message.content
        if content is None or not content.strip():
            continue
        if message.role == "system":
            extra.append(content)
        elif message.role in _HISTORY_ROLES:
            history.append({"role": message.role, "content": content})
    user_input = OPENING_NUDGE
    if history and history[-1]["role"] == "user":
        user_input = history[-1]["content"]
        history = history[:-1]
    return history, user_input, "\n\n".join(extra)
