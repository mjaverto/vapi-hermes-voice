"""Runtime settings for the Vapi <-> Hermes voice adapter."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import BaseModel, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

_E164_RE = re.compile(r"^\+[1-9]\d{6,14}$")

_MIN_SECRET_LENGTH = 16

# Absolute ceiling on VHV_FILLER_MAX_PER_TURN, independent of whatever an operator
# configures: a caller must never hear a machine-gun run of holding lines.
_MAX_FILLERS_PER_TURN_HARD_CAP = 3

_DEFAULT_FILLER_PHRASES: tuple[str, ...] = (
    "I have that information right here, give me a second.",
    "Let me pull that up for you.",
    "One moment while I check.",
    "Just a second, looking now.",
    "Give me a moment to find that.",
    "Hold on, checking that for you.",
    "Let me take a quick look.",
    "Bear with me one second.",
)


def _split_csv(value: object) -> object:
    """Allow list fields to be set from comma-separated env strings (or JSON arrays)."""
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            return json.loads(text)
        return [item.strip() for item in text.split(",") if item.strip()]
    return value


class ToolPolicy(BaseModel):
    """Which Hermes tools voice turns may use, and their budgets (advisory only)."""

    enabled_tools: Annotated[list[str], NoDecode] = []
    confirm_tools: Annotated[list[str], NoDecode] = []
    max_tool_calls_per_turn: int = 3
    max_tool_seconds_per_call: float = 60.0

    @field_validator("enabled_tools", "confirm_tools", mode="before")
    @classmethod
    def _parse_csv(cls, value: object) -> object:
        return _split_csv(value)


class Settings(BaseSettings):
    """Adapter configuration, loaded from VHV_-prefixed env vars and an optional .env file."""

    model_config = SettingsConfigDict(
        env_prefix="VHV_",
        env_file=".env",
        case_sensitive=False,
        env_nested_delimiter="__",
    )

    # hermes connection
    hermes_base_url: str
    hermes_api_key: SecretStr

    # adapter auth: the static API key Vapi's custom-llm credential must present in
    # the Authorization header (docs/integration-contracts.md section 1.4).
    adapter_api_key: SecretStr
    # optional defense-in-depth: when set, the endpoint moves to
    # POST /v/{route_secret}/chat/completions and the bare path 404s.
    route_secret: SecretStr | None = None

    listen_host: str = "127.0.0.1"
    listen_port: int = 8766

    # caller policy (E.164; empty list = allowlist DISABLED: allow all, warn)
    allowed_callers: Annotated[list[str], NoDecode] = []

    # voice behavior (the greeting itself is Vapi's job: assistant.firstMessage)
    assistant_name: str = "the assistant"
    principal: str = "the operator"
    filler_phrases: Annotated[list[str], NoDecode] = list(_DEFAULT_FILLER_PHRASES)
    filler_after_seconds: float = 1.5
    # Structural floor between the *end* of one filler and the start of the next,
    # independent of filler_after_seconds/tool_start re-arms: once a filler has
    # been spoken, no later dead-air window (however it was re-armed) may speak
    # another until this many seconds have passed. Default matches a caller
    # tolerating one holding line roughly every 8 s, never a rapid string of them.
    filler_min_gap_seconds: float = 8.0
    # Hard cap on how many holding lines one turn may speak. A long agentic Hermes
    # run can legitimately re-arm the filler timer many times (one per tool round
    # trip); left uncapped, a caller can hear a string of holding lines back to
    # back with no real content between them, which does not sound human. Bounded
    # to [1, _MAX_FILLERS_PER_TURN_HARD_CAP] regardless of configured value.
    # filler_min_gap_seconds and filler_max_per_turn both gate every filler: a
    # turn speaks a holding line only when BOTH the cap and the gap allow it.
    filler_max_per_turn: int = 1
    # Suffix fillers with the Vapi <flush /> audio-control token so they are spoken
    # immediately instead of sitting in the TTS buffer (contracts section 1.6).
    # Requires voice.chunkPlan.enabled (the Vapi default); disable if chunking is off.
    filler_use_flush: bool = True

    # hermes routing
    voice_model: str | None = None
    voice_provider: str | None = None
    voice_reasoning_effort: str | None = "low"
    warmup_on_start: bool = True

    # timeouts / limits
    hermes_connect_timeout: float = 5.0
    hermes_first_token_timeout: float = 15.0
    hermes_turn_timeout: float = 60.0
    hermes_stop_timeout: float = 3.0
    max_concurrent_turns: int = 5
    max_history_messages: int = 200
    max_body_bytes: int = 1_000_000
    session_retention: Literal["none", "hermes"] = "none"
    call_state_ttl_seconds: float = 14_400.0  # 4 h: longer than any sane phone call
    max_tracked_calls: int = 1024
    tool_policy: ToolPolicy = ToolPolicy()

    @field_validator("allowed_callers", "filler_phrases", mode="before")
    @classmethod
    def _parse_csv(cls, value: object) -> object:
        return _split_csv(value)

    @field_validator("adapter_api_key")
    @classmethod
    def _check_adapter_api_key(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value()) < _MIN_SECRET_LENGTH:
            raise ValueError("adapter_api_key must be at least 16 characters long")
        return value

    @field_validator("route_secret")
    @classmethod
    def _check_route_secret(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and len(value.get_secret_value()) < _MIN_SECRET_LENGTH:
            raise ValueError("route_secret must be at least 16 characters long when set")
        return value

    @field_validator("allowed_callers")
    @classmethod
    def _check_allowed_callers(cls, value: list[str]) -> list[str]:
        for number in value:
            if _E164_RE.fullmatch(number) is None:
                raise ValueError("allowed_callers entries must be E.164 numbers like +15551234567")
        return value

    @field_validator("filler_phrases")
    @classmethod
    def _check_filler_phrases(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("filler_phrases must not be empty")
        return value

    @field_validator("filler_max_per_turn")
    @classmethod
    def _check_filler_max_per_turn(cls, value: int) -> int:
        if not 1 <= value <= _MAX_FILLERS_PER_TURN_HARD_CAP:
            raise ValueError(
                f"filler_max_per_turn must be between 1 and {_MAX_FILLERS_PER_TURN_HARD_CAP}"
            )
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings instance."""
    return Settings()  # type: ignore[call-arg]  # required fields come from the environment
