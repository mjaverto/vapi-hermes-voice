"""Tests for vapi_hermes_voice.policy."""

from __future__ import annotations

import pytest

from vapi_hermes_voice.policy import (
    OPENING_NUDGE,
    CallerPolicy,
    derive_session_ids,
    split_messages,
    truncate_history,
)
from vapi_hermes_voice.vapi_events import ChatMessage


def _m(role: str, content: str | None) -> ChatMessage:
    return ChatMessage(role=role, content=content)


class TestCallerPolicy:
    def test_rejects_invalid_allowlist_entries(self) -> None:
        with pytest.raises(ValueError, match="E.164"):
            CallerPolicy(["not-a-number"])

    def test_empty_allowlist_not_enforced_allows_all(self) -> None:
        policy = CallerPolicy([])
        assert policy.enforced is False
        assert policy.is_allowed("+15551234567") is True
        assert policy.is_allowed(None) is True
        assert policy.is_allowed("garbage") is True

    def test_enforced_allows_listed_number(self) -> None:
        policy = CallerPolicy(["+15551234567", "+15559876543"])
        assert policy.enforced is True
        assert policy.is_allowed("+15551234567") is True
        assert policy.is_allowed("+15559876543") is True

    def test_enforced_denies_unlisted_number(self) -> None:
        policy = CallerPolicy(["+15551234567"])
        assert policy.is_allowed("+15550000000") is False

    def test_enforced_denies_missing_number(self) -> None:
        # Fail closed: web calls / metadataSendMode=off carry no caller identity.
        policy = CallerPolicy(["+15551234567"])
        assert policy.is_allowed(None) is False

    @pytest.mark.parametrize("bad", ["", "banana", "15551234567", "+0555123456", "+1"])
    def test_enforced_denies_malformed_number(self, bad: str) -> None:
        policy = CallerPolicy(["+15551234567"])
        assert policy.is_allowed(bad) is False


class TestDeriveSessionIds:
    def test_deterministic(self) -> None:
        assert derive_session_ids("call_abc123") == derive_session_ids("call_abc123")

    def test_id_and_key_distinct(self) -> None:
        session_id, session_key = derive_session_ids("call_abc123")
        assert session_id != session_key

    def test_distinct_across_calls(self) -> None:
        assert derive_session_ids("call_one") != derive_session_ids("call_two")

    def test_prefixes_and_hash_length(self) -> None:
        session_id, session_key = derive_session_ids("call_abc123")
        assert session_id.startswith("vhv-")
        assert session_key.startswith("vhv-key-")
        assert len(session_id) == len("vhv-") + 24
        assert len(session_key) == len("vhv-key-") + 24

    def test_raw_call_id_not_embedded(self) -> None:
        call_id = "call_super_secret_id"
        session_id, session_key = derive_session_ids(call_id)
        assert call_id not in session_id
        assert call_id not in session_key


class TestTruncateHistory:
    def test_empty(self) -> None:
        assert truncate_history([], 10) == []

    def test_under_limit_untouched(self) -> None:
        messages = [_m("user", "hi"), _m("assistant", "hello")]
        assert truncate_history(messages, 5) == messages

    def test_keeps_most_recent(self) -> None:
        messages = [_m("user", str(i)) for i in range(6)]
        result = truncate_history(messages, 2)
        assert [m.content for m in result] == ["4", "5"]

    def test_always_keeps_last_message(self) -> None:
        messages = [_m("user", "old"), _m("assistant", "latest")]
        result = truncate_history(messages, 0)
        assert [m.content for m in result] == ["latest"]


class TestSplitMessages:
    def test_trailing_user_message_becomes_input(self) -> None:
        history, user_input, extra = split_messages(
            [
                _m("system", "You are helpful."),
                _m("user", "hi"),
                _m("assistant", "hello, how can I help?"),
                _m("user", "what's the weather?"),
            ]
        )
        assert user_input == "what's the weather?"
        assert history == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello, how can I help?"},
        ]
        assert extra == "You are helpful."

    def test_no_trailing_user_message_uses_opening_nudge(self) -> None:
        history, user_input, extra = split_messages([_m("system", "Be nice.")])
        assert user_input == OPENING_NUDGE
        assert history == []
        assert extra == "Be nice."

    def test_assistant_last_keeps_history_and_nudges(self) -> None:
        history, user_input, _ = split_messages([_m("user", "hi"), _m("assistant", "hello there")])
        assert user_input == OPENING_NUDGE
        assert history == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello there"},
        ]

    def test_tool_and_function_frames_dropped(self) -> None:
        history, user_input, _ = split_messages(
            [
                _m("tool", '{"result": 42}'),
                _m("function", "ignored"),
                _m("user", "real question"),
            ]
        )
        assert user_input == "real question"
        assert history == []

    def test_multiple_system_messages_joined(self) -> None:
        _, _, extra = split_messages(
            [_m("system", "First."), _m("system", "Second."), _m("user", "q")]
        )
        assert extra == "First.\n\nSecond."

    def test_empty_and_null_content_skipped(self) -> None:
        history, user_input, extra = split_messages(
            [_m("user", ""), _m("assistant", None), _m("user", "  "), _m("user", "real")]
        )
        assert user_input == "real"
        assert history == []
        assert extra == ""
