"""Tests for vapi_hermes_voice.config."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from vapi_hermes_voice.config import (
    _DEFAULT_FILLER_PHRASES,
    _RETIRED_SETTINGS,
    Settings,
    ToolPolicy,
    get_settings,
)

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


def test_default_filler_phrases_do_not_claim_to_be_looking_something_up() -> None:
    """The default pool must read as "I heard you", not as "I am searching a database".

    An acknowledgement is spoken on every slow turn now, including turns where no
    tool runs at all and turns where the assistant is the one who placed the call.
    The previous pool was entirely lookup-flavoured, which produced the live
    absurdity of Emma saying "Give me a moment to find that" on the turn where she
    was introducing herself.
    """
    lookup_claims = (
        "pull that up",
        "information right here",
        "find that",
        "looking now",
        "checking that for you",
        "take a quick look",
    )
    for phrase in _DEFAULT_FILLER_PHRASES:
        lowered = phrase.lower()
        assert not any(claim in lowered for claim in lookup_claims), phrase
        # Short: every syllable delays the real answer.
        assert len(phrase.split()) <= 6, phrase
    assert "Okay, let me check." in _DEFAULT_FILLER_PHRASES


def test_filler_phrases_comma_override(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(monkeypatch, VHV_FILLER_PHRASES="One moment., Let me check.")
    assert settings.filler_phrases == ["One moment.", "Let me check."]


def test_filler_phrases_must_not_be_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError):
        _make_settings(monkeypatch, VHV_FILLER_PHRASES="")


def test_filler_after_seconds_default_fits_the_two_second_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vapi endpointing spends ~0.4-1.6 s before the adapter is invoked at all.

    The acknowledgement has to be audible within 2 s of the callee finishing their
    sentence, so the adapter's own share of that budget must stay under a second.
    """
    settings = _make_settings(monkeypatch)
    assert settings.filler_after_seconds <= 1.0


def test_filler_min_gap_seconds_default_is_the_ten_second_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _make_settings(monkeypatch)
    assert settings.filler_min_gap_seconds == 10.0


def test_filler_min_gap_seconds_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(monkeypatch, VHV_FILLER_MIN_GAP_SECONDS="3.5")
    assert settings.filler_min_gap_seconds == 3.5


def _write_env(tmp_path: Path, **values: str) -> Path:
    path = tmp_path / ".env"
    lines = (f"{key}={value}" for key, value in {**REQUIRED_ENV, **values}.items())
    path.write_text("\n".join(lines))
    return path


def test_dotenv_key_the_adapter_never_knew_is_still_rejected(tmp_path: Path) -> None:
    """The safety net is for keys we retired, not for typos: those must still fail."""
    env_file = _write_env(tmp_path, VHV_FILLER_MAXX_PER_TURN="1")
    with pytest.raises(ValidationError):
        Settings(_env_file=env_file)


def test_retired_dotenv_key_is_ignored_instead_of_crash_looping(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A retired key left in a deployed .env must not stop the service booting.

    ``VHV_FILLER_MAX_PER_TURN`` is set in the live deployment's .env. This model
    forbids extras, so without the retired-key path the first restart after the
    knob was removed would have crash-looped the unit -- i.e. taken the phone line
    down as the direct result of a settings cleanup.
    """
    env_file = _write_env(tmp_path, VHV_FILLER_MAX_PER_TURN="1")
    with caplog.at_level(logging.WARNING, logger="vapi_hermes_voice.config"):
        settings = Settings(_env_file=env_file)

    assert not hasattr(settings, "filler_max_per_turn")
    assert settings.filler_min_gap_seconds == 10.0  # the rest of the file still loaded
    assert "VHV_FILLER_MAX_PER_TURN" in caplog.text
    assert "remove it from .env" in caplog.text


def test_every_retired_setting_is_documented_and_gone() -> None:
    """No entry may name a live field, or it would silently drop real config."""
    assert _RETIRED_SETTINGS, "the map may be emptied, but then delete the code path too"
    for name, reason in _RETIRED_SETTINGS.items():
        assert name not in Settings.model_fields, name
        assert reason.strip(), name


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


# --- hermes routing guards ---


def test_voice_reasoning_effort_has_no_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # A default of "low" was sent on every turn even with no model/provider set --
    # i.e. on the default Hermes route, where it degrades multi-hop tool use.
    settings = _make_settings(monkeypatch)
    assert settings.voice_reasoning_effort is None


def test_voice_reasoning_effort_still_settable(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(monkeypatch, VHV_VOICE_REASONING_EFFORT="low")
    assert settings.voice_reasoning_effort == "low"


def test_voice_model_and_provider_accepted_together(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(
        monkeypatch,
        VHV_VOICE_MODEL="google/gemini-3.7-flash",
        VHV_VOICE_PROVIDER="openrouter",
    )
    assert settings.voice_model == "google/gemini-3.7-flash"
    assert settings.voice_provider == "openrouter"


def test_voice_model_without_provider_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    # Hermes silently mis-routes a bare model, so this fails at config load.
    with pytest.raises(ValidationError, match="must be set together"):
        _make_settings(monkeypatch, VHV_VOICE_MODEL="google/gemini-3.7-flash")


def test_voice_provider_without_model_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError, match="must be set together"):
        _make_settings(monkeypatch, VHV_VOICE_PROVIDER="openrouter")


def test_voice_routing_unset_is_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(monkeypatch)
    assert settings.voice_model is None
    assert settings.voice_provider is None
