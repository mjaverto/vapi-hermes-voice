"""Tests for vapi_hermes_voice.policy."""

from __future__ import annotations

import pytest

from vapi_hermes_voice.config import Settings
from vapi_hermes_voice.policy import (
    OPENING_NUDGE,
    CallerPolicy,
    build_opening_nudge,
    derive_session_ids,
    has_trailing_user_message,
    split_messages,
    truncate_history,
)
from vapi_hermes_voice.vapi_events import CallVariables, ChatMessage


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


# --- build_opening_nudge: direction-aware opening for a turn with no user utterance ---

PURPOSE = "reschedule Marvin's cardiology recheck to next Tuesday afternoon"


def make_settings(**overrides: object) -> Settings:
    kwargs: dict[str, object] = {
        "hermes_base_url": "http://127.0.0.1:9",
        "hermes_api_key": "unit-test-key",
        "adapter_api_key": "unit-test-secret-0123456789",
        "assistant_name": "Emma",
        "principal": "Mike",
        **overrides,
    }
    return Settings(**kwargs)  # type: ignore[arg-type]


class TestBuildOpeningNudge:
    def test_inbound_opening_unchanged(self) -> None:
        assert (
            build_opening_nudge(make_settings(), direction="inbound", variables=CallVariables())
            == OPENING_NUDGE
        )

    def test_inbound_ignores_a_purpose(self) -> None:
        # Inbound behavior stays exactly as it was, purpose or not.
        assert (
            build_opening_nudge(
                make_settings(), direction="inbound", variables=CallVariables(purpose=PURPOSE)
            )
            == OPENING_NUDGE
        )

    def test_outbound_without_purpose_unchanged(self) -> None:
        assert (
            build_opening_nudge(make_settings(), direction="outbound", variables=CallVariables())
            == OPENING_NUDGE
        )

    def test_outbound_with_purpose_states_who_behalf_and_reason(self) -> None:
        nudge = build_opening_nudge(
            make_settings(),
            direction="outbound",
            variables=CallVariables(purpose=PURPOSE, callee="Dr. Patel's office"),
        )
        assert "you are Emma" in nudge  # who is calling
        assert "on behalf of Mike" in nudge  # on whose behalf
        assert PURPOSE in nudge  # the reason
        assert "Dr. Patel's office" in nudge

    def test_outbound_with_purpose_never_greets_the_principal(self) -> None:
        # The live bug: the adapter opened with "Hi Mike, how can I help?" on a call
        # placed TO a third party.
        nudge = build_opening_nudge(
            make_settings(),
            direction="outbound",
            variables=CallVariables(purpose=PURPOSE, callee="Dr. Patel's office"),
        )
        assert nudge != OPENING_NUDGE
        assert "ask how you can help" not in nudge.lower().replace(
            "do not ask how you can help", ""
        )
        assert "Do not greet Mike" in nudge
        assert "caller has not spoken yet" not in nudge

    def test_outbound_purpose_gets_a_sentence_ending(self) -> None:
        nudge = build_opening_nudge(
            make_settings(),
            direction="outbound",
            variables=CallVariables(purpose="move the appointment"),
        )
        assert "move the appointment." in nudge

    def test_callee_sentence_omitted_when_absent(self) -> None:
        nudge = build_opening_nudge(
            make_settings(), direction="outbound", variables=CallVariables(purpose=PURPOSE)
        )
        assert "You are trying to reach" not in nudge
        assert "{callee}" not in nudge

    def test_discloses_ai_identity_by_default(self) -> None:
        nudge = build_opening_nudge(
            make_settings(),
            direction="outbound",
            variables=CallVariables(purpose=PURPOSE, callee="Dr. Patel's office"),
        )
        assert "you are an AI assistant calling for Mike" in nudge

    def test_disclosure_suppressible_by_config(self) -> None:
        nudge = build_opening_nudge(
            make_settings(outbound_disclose_ai=False),
            direction="outbound",
            variables=CallVariables(purpose=PURPOSE, callee="Dr. Patel's office"),
        )
        assert "AI assistant" not in nudge
        assert PURPOSE in nudge  # the objective still lands

    def test_no_disclosure_when_calling_the_principal(self) -> None:
        nudge = build_opening_nudge(
            make_settings(),
            direction="outbound",
            variables=CallVariables(purpose="remind him about the vet", callee="Mike"),
            callee_is_principal=True,
        )
        assert "AI assistant" not in nudge

    def test_disclosure_survives_a_template_that_omits_it(self) -> None:
        # Disclosure is a safety control appended outside the template, so an
        # operator cannot drop it just by rewriting VHV_OUTBOUND_OPENING.
        nudge = build_opening_nudge(
            make_settings(outbound_opening="Say this: {purpose}"),
            direction="outbound",
            variables=CallVariables(purpose=PURPOSE, callee="Dr. Patel"),
        )
        assert nudge.startswith("Say this: ")
        assert "you are an AI assistant calling for Mike" in nudge

    def test_custom_template_placeholders_substituted(self) -> None:
        nudge = build_opening_nudge(
            make_settings(
                outbound_opening="{assistant_name} for {principal} to {callee}: {purpose}"
            ),
            direction="outbound",
            variables=CallVariables(purpose=PURPOSE, callee="the clinic"),
        )
        assert nudge.startswith(f"Emma for Mike to the clinic: {PURPOSE}.")

    def test_unknown_placeholder_left_literal(self) -> None:
        nudge = build_opening_nudge(
            make_settings(outbound_opening="{purpose} and {mystery}"),
            direction="outbound",
            variables=CallVariables(purpose=PURPOSE),
        )
        assert "{mystery}" in nudge

    def test_missing_callee_renders_as_empty_string(self) -> None:
        nudge = build_opening_nudge(
            make_settings(outbound_opening="[{callee}] {purpose}"),
            direction="outbound",
            variables=CallVariables(purpose=PURPOSE),
        )
        assert nudge.startswith("[] ")

    def test_untrusted_purpose_is_not_re_expanded_as_a_template(self) -> None:
        # A hostile purpose must be inert data: no second substitution pass, and no
        # format-spec style reach into Python objects.
        hostile = "{principal} {assistant_name} {purpose} {0.__class__}"
        nudge = build_opening_nudge(
            make_settings(),
            direction="outbound",
            variables=CallVariables(purpose=hostile),
        )
        assert "{principal} {assistant_name} {purpose} {0.__class__}" in nudge
        assert "class 'str'" not in nudge


class TestSplitMessagesOpening:
    def test_provided_opening_used_when_no_user_message(self) -> None:
        history, user_input, extra = split_messages(
            [_m("system", "Be brief.")], opening="OPEN THE CALL"
        )
        assert user_input == "OPEN THE CALL"
        assert history == []
        assert extra == "Be brief."

    def test_default_opening_is_the_inbound_nudge(self) -> None:
        _history, user_input, _extra = split_messages([_m("system", "Be brief.")])
        assert user_input == OPENING_NUDGE

    def test_trailing_user_message_still_wins_over_opening(self) -> None:
        _history, user_input, _extra = split_messages(
            [_m("user", "actually I spoke first")], opening="OPEN THE CALL"
        )
        assert user_input == "actually I spoke first"


class TestOpeningNudgeToPrincipal:
    """Defect: outbound to Mike opened 'This is Emma calling for Mike' (third person)."""

    def test_greets_principal_directly_by_name(self) -> None:
        nudge = build_opening_nudge(
            make_settings(),
            direction="outbound",
            variables=CallVariables(purpose=PURPOSE, callee="Mike"),
            callee_is_principal=True,
        )
        assert "greeting Mike directly by name" in nudge
        assert "Hi Mike, Emma here." in nudge
        assert PURPOSE in nudge

    def test_drops_on_behalf_of_framing(self) -> None:
        nudge = build_opening_nudge(
            make_settings(),
            direction="outbound",
            variables=CallVariables(purpose=PURPOSE, callee="Mike"),
            callee_is_principal=True,
        )
        assert "on behalf of Mike, and state" not in nudge
        assert 'Do not say you are calling "for" or "on behalf of" Mike' in nudge

    def test_skips_ai_disclosure_and_ask_for_someone(self) -> None:
        nudge = build_opening_nudge(
            make_settings(),
            direction="outbound",
            variables=CallVariables(purpose=PURPOSE, callee="Mike"),
            callee_is_principal=True,
        )
        assert "AI assistant" not in nudge
        assert "Ask for the right person" not in nudge

    def test_third_party_branch_is_unaffected(self) -> None:
        nudge = build_opening_nudge(
            make_settings(),
            direction="outbound",
            variables=CallVariables(purpose=PURPOSE, callee="Dr. Patel"),
            callee_is_principal=False,
        )
        assert "on behalf of Mike" in nudge
        assert "you are an AI assistant calling for Mike" in nudge

    def test_principal_template_is_configurable(self) -> None:
        nudge = build_opening_nudge(
            make_settings(
                outbound_opening_principal="Yo {principal}, it's {assistant_name}: {purpose}"
            ),
            direction="outbound",
            variables=CallVariables(purpose=PURPOSE),
            callee_is_principal=True,
        )
        assert nudge == f"Yo Mike, it's Emma: {PURPOSE}."

    def test_principal_branch_needs_a_purpose(self) -> None:
        assert (
            build_opening_nudge(
                make_settings(),
                direction="outbound",
                variables=CallVariables(),
                callee_is_principal=True,
            )
            == OPENING_NUDGE
        )


class TestHasTrailingUserMessage:
    def test_trailing_user_message(self) -> None:
        assert has_trailing_user_message([_m("system", "x"), _m("user", "hello")]) is True

    def test_no_messages(self) -> None:
        assert has_trailing_user_message([]) is False

    def test_system_only_is_an_opening_turn(self) -> None:
        assert has_trailing_user_message([_m("system", "Be brief.")]) is False

    def test_trailing_assistant_message_is_an_opening_turn(self) -> None:
        assert has_trailing_user_message([_m("user", "hi"), _m("assistant", "hello")]) is False

    def test_blank_trailing_user_message_does_not_count(self) -> None:
        assert has_trailing_user_message([_m("user", "hi"), _m("user", "   ")]) is True
        assert has_trailing_user_message([_m("assistant", "hi"), _m("user", "   ")]) is False

    def test_system_message_after_user_still_counts_as_user_turn(self) -> None:
        # System frames are instructions, not utterances; they never end a turn.
        assert has_trailing_user_message([_m("user", "hello"), _m("system", "Be brief.")]) is True

    def test_agrees_with_split_messages(self) -> None:
        for messages in (
            [_m("system", "x")],
            [_m("user", "hi")],
            [_m("user", "hi"), _m("assistant", "yo")],
        ):
            _h, user_input, _e = split_messages(messages, opening="SYNTHETIC")
            assert has_trailing_user_message(messages) is (user_input != "SYNTHETIC")
