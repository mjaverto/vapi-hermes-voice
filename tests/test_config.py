"""Tests for vapi_hermes_voice.config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vapi_hermes_voice.config import Settings, ToolPolicy, get_settings

REQUIRED_ENV: dict[str, str] = {
    "VHV_HERMES_BASE_URL": "http://127.0.0.1:8642",
    "VHV_HERMES_API_KEY": "test-api-key",
    "VHV_ADAPTER_API_KEY": "0123456789abcdef",
}


def _make_settings(monkeypatch: pytest.MonkeyPatch, **extra: str) -> Settings:
    for key, value in {**REQUIRED_ENV, **extra}.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)


def test_env_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(monkeypatch, VHV_LISTEN_PORT="9001")
    assert settings.hermes_base_url == "http://127.0.0.1:8642"
    assert settings.hermes_api_key.get_secret_value() == "test-api-key"
    assert settings.adapter_api_key.get_secret_value() == "0123456789abcdef"
    assert settings.route_secret is None
    assert settings.listen_port == 9001
    assert settings.listen_host == "127.0.0.1"
    assert settings.session_retention == "none"
    assert settings.filler_use_flush is True


def test_secrets_not_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(monkeypatch, VHV_ROUTE_SECRET="route-secret-0123456789")
    assert "test-api-key" not in repr(settings)
    assert "0123456789abcdef" not in repr(settings)
    assert "route-secret-0123456789" not in repr(settings)


def test_adapter_api_key_too_short_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError):
        _make_settings(monkeypatch, VHV_ADAPTER_API_KEY="tooshort")


def test_route_secret_optional_but_min_length_when_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _make_settings(monkeypatch)
    assert settings.route_secret is None
    with pytest.raises(ValidationError):
        _make_settings(monkeypatch, VHV_ROUTE_SECRET="tooshort")


def test_allowed_callers_comma_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(monkeypatch, VHV_ALLOWED_CALLERS="+15551234567, +15559876543")
    assert settings.allowed_callers == ["+15551234567", "+15559876543"]


def test_allowed_callers_json_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(monkeypatch, VHV_ALLOWED_CALLERS='["+15551234567", "+15559876543"]')
    assert settings.allowed_callers == ["+15551234567", "+15559876543"]


def test_allowed_callers_empty_string_disables_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _make_settings(monkeypatch, VHV_ALLOWED_CALLERS="")
    assert settings.allowed_callers == []


@pytest.mark.parametrize(
    "bad",
    ["banana", "15551234567", "+05551234567", "+1555", "+1555123456789012345"],
)
def test_allowed_callers_rejects_non_e164(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    with pytest.raises(ValidationError):
        _make_settings(monkeypatch, VHV_ALLOWED_CALLERS=bad)


def test_filler_phrases_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(monkeypatch)
    assert len(settings.filler_phrases) >= 6
    assert len(set(settings.filler_phrases)) == len(settings.filler_phrases)
    assert all(phrase.strip() for phrase in settings.filler_phrases)


def test_filler_phrases_comma_override(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(monkeypatch, VHV_FILLER_PHRASES="One moment., Let me check.")
    assert settings.filler_phrases == ["One moment.", "Let me check."]


def test_filler_phrases_must_not_be_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError):
        _make_settings(monkeypatch, VHV_FILLER_PHRASES="")


def test_filler_max_per_turn_default(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(monkeypatch)
    assert settings.filler_max_per_turn == 1


def test_filler_max_per_turn_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(monkeypatch, VHV_FILLER_MAX_PER_TURN="2")
    assert settings.filler_max_per_turn == 2


@pytest.mark.parametrize("bad", ["0", "-1", "4", "100"])
def test_filler_max_per_turn_rejects_out_of_hard_cap_range(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    with pytest.raises(ValidationError):
        _make_settings(monkeypatch, VHV_FILLER_MAX_PER_TURN=bad)


def test_filler_min_gap_seconds_default(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(monkeypatch)
    assert settings.filler_min_gap_seconds == 8.0


def test_filler_min_gap_seconds_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(monkeypatch, VHV_FILLER_MIN_GAP_SECONDS="3.5")
    assert settings.filler_min_gap_seconds == 3.5


def test_tool_policy_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(monkeypatch)
    assert settings.tool_policy == ToolPolicy()
    assert settings.tool_policy.enabled_tools == []
    assert settings.tool_policy.max_tool_calls_per_turn == 3


def test_tool_policy_nested_env(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(monkeypatch, VHV_TOOL_POLICY__MAX_TOOL_CALLS_PER_TURN="5")
    assert settings.tool_policy.max_tool_calls_per_turn == 5


def test_tool_policy_list_env_comma_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(
        monkeypatch,
        VHV_TOOL_POLICY__ENABLED_TOOLS="recall,weather",
        VHV_TOOL_POLICY__CONFIRM_TOOLS="send_email, delete_file",
    )
    assert settings.tool_policy.enabled_tools == ["recall", "weather"]
    assert settings.tool_policy.confirm_tools == ["send_email", "delete_file"]


def test_tool_policy_list_env_json_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(monkeypatch, VHV_TOOL_POLICY__ENABLED_TOOLS='["recall", "weather"]')
    assert settings.tool_policy.enabled_tools == ["recall", "weather"]


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    try:
        first = get_settings()
        second = get_settings()
        assert first is second
        assert first.hermes_base_url == REQUIRED_ENV["VHV_HERMES_BASE_URL"]
    finally:
        get_settings.cache_clear()


def test_outbound_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(monkeypatch)
    assert "{purpose}" in settings.outbound_opening
    assert settings.outbound_disclose_ai is True


def test_outbound_opening_override(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(monkeypatch, VHV_OUTBOUND_OPENING="Open with: {purpose}")
    assert settings.outbound_opening == "Open with: {purpose}"


def test_outbound_opening_requires_purpose_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A template without {purpose} would silently drop the objective on every
    # outbound task call, so it fails at config load instead.
    with pytest.raises(ValidationError, match=r"\{purpose\} placeholder"):
        _make_settings(monkeypatch, VHV_OUTBOUND_OPENING="Just say hello.")


def test_outbound_disclosure_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(monkeypatch, VHV_OUTBOUND_DISCLOSE_AI="false")
    assert settings.outbound_disclose_ai is False


def test_principal_number_defaults_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _make_settings(monkeypatch).principal_number is None


def test_principal_number_accepts_e164(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(monkeypatch, VHV_PRINCIPAL_NUMBER="+15551230000")
    assert settings.principal_number == "+15551230000"


@pytest.mark.parametrize("bad", ["15551230000", "not-a-number", "+0555123", "+1"])
def test_principal_number_rejects_non_e164(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    with pytest.raises(ValidationError, match="E.164"):
        _make_settings(monkeypatch, VHV_PRINCIPAL_NUMBER=bad)


def test_principal_opening_template_requires_purpose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValidationError, match=r"\{purpose\} placeholder"):
        _make_settings(monkeypatch, VHV_OUTBOUND_OPENING_PRINCIPAL="Hi there.")


def test_principal_opening_default_greets_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = _make_settings(monkeypatch).outbound_opening_principal
    assert "{purpose}" in template
    assert "directly by name" in template
    assert "on behalf of" in template  # as a prohibition, not as framing
