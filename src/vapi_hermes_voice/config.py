"""Runtime settings for the Vapi <-> Hermes voice adapter."""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from typing import Annotated, Any, Literal

from pydantic import BaseModel, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

logger = logging.getLogger(__name__)

_E164_RE = re.compile(r"^\+[1-9]\d{6,14}$")

_MIN_SECRET_LENGTH = 16

# VHV_ env keys that used to configure the adapter and no longer do. They are
# accepted-and-ignored (with a warning naming them) rather than rejected: this
# model is extra="forbid", so a key left behind in a deployed .env would otherwise
# crash-loop the unit on its next restart -- retiring a knob must never be able to
# take the phone line down. Deleting the key from .env silences the warning; an
# entry may be dropped from this map once no deployment still sets it.
_RETIRED_SETTINGS: dict[str, str] = {
    "filler_max_per_turn": (
        "acknowledgements are now bounded by the call-global"
        " VHV_FILLER_MIN_GAP_SECONDS cooldown, which subsumes any per-turn cap"
    ),
}

# Acknowledgement lines: what the callee hears within ~1s of finishing a sentence,
# as an immediate human "I heard you", NOT as a claim to be searching a database.
# The adapter speaks one of these on any turn where the answer has not started yet,
# including turns on which no tool runs at all, so a phrase that promises a lookup
# ("I have that information right here") is wrong far more often than it is right.
# Keep them short: every syllable here is spoken before the real answer can begin.
_DEFAULT_FILLER_PHRASES: tuple[str, ...] = (
    "Okay, let me check.",
    "Okay, one moment.",
    "Got it, one second.",
    "Sure, give me a second.",
    "Alright, let me see.",
    "Okay, bear with me a moment.",
    "Right, one second.",
    "Understood, one moment.",
)

# Turn inputs used to open an outbound task call before the callee has spoken, one
# per counterparty. `{purpose}` is required in both: without it the objective never
# reaches the model. `{principal}`, `{assistant_name}`, and `{callee}` are
# substituted too, and any other `{name}` is left as literal text. See
# docs/integration-contracts.md.
_DEFAULT_OUTBOUND_OPENING = (
    "You have just placed this call and whoever answered has not spoken yet. "
    "Open the call yourself: say that you are {assistant_name}, say you are calling "
    "on behalf of {principal}, and state the reason you are calling. "
    "The reason for this call is: {purpose} "
    "Do not greet {principal} and do not ask how you can help: {principal} is not on "
    "this call, and you are the one who needs something."
)

# Used instead of _DEFAULT_OUTBOUND_OPENING when the call reaches the principal
# themselves. Third-person "this is Emma calling for Mike" framing is wrong when Mike
# is the one who picked up, and there is nobody to disclose to but the principal.
_DEFAULT_OUTBOUND_OPENING_PRINCIPAL = (
    "You have just placed this call to {principal} and they have not spoken yet. "
    "Open the call by greeting {principal} directly by name, as yourself -- for "
    'example "Hi {principal}, {assistant_name} here." -- and then say why you are '
    "calling. The reason for this call is: {purpose} "
    'Do not say you are calling "for" or "on behalf of" {principal}, and do not '
    "refer to {principal} in the third person: {principal} is the person you are "
    "speaking to, not somebody who answers on their behalf."
)

# The sentence that states WHY an outbound call was placed, spoken from
# adapter-local text on the first turn the callee speaks. It exists because nothing
# routed through Hermes can meet the deadline: a Hermes-composed utterance measures
# 1.6-2.2 s warm, 3.6-4.9 s cold and 14-17 s once a tool runs, against a target of
# one to two seconds after the callee stops talking. `{reason}` is required -- a
# reason-for-calling line with no reason in it is the bug, not the fix.
#
# The greeting and the AI-identity disclosure are NOT here: they are assembled in
# policy.build_reason_line so that no operator edit can drop them. Only the reason
# sentence itself is configurable.
#
# Note that this template's lead-in ("I am calling ...") cannot be spoken twice, even
# when the operator hands over a `spoken_reason` that is already a whole clause of
# its own ("I am calling about the MRI"): speech.speakable_reason deletes a redundant
# lead-in, so the value reduces to the connector-led clause this sentence expects.
# A rewritten template with no lead-in of its own ("Quick one, {reason}.") therefore
# reads correctly too.
_DEFAULT_OUTBOUND_REASON_SENTENCE = "I am calling {reason}. Is this a good moment?"

# Spoken instead when the call carries no `spoken_reason` the adapter can safely use.
# Purpose-free BY DESIGN: `purpose` is written for the model and routinely carries
# section labels, the options being weighed, and the principal's internal limits, so
# no fallback may be derived from it. Vague and true beats fluent and wrong.
_DEFAULT_OUTBOUND_REASON_SENTENCE_GENERIC = "Is this a good moment to talk?"


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
    # How long a turn may stay silent before the acknowledgement is spoken. The
    # requirement is measured from when the CALLEE STOPPED TALKING, and the adapter
    # does not own that whole budget: Vapi's transcriber endpointing (Deepgram Flux
    # eotThreshold/eotTimeoutMs) plus startSpeakingPlan.waitSeconds burn ~0.4-1.6 s
    # before this process is even invoked, and the spoken chunk still has to reach
    # the TTS. So against a 2 s ceiling the adapter's share is ~0.4 s worst case
    # (2.0 - 1.6) and ~1.6 s best case: 0.9 s keeps the typical call comfortably
    # inside 2 s while staying long enough that a genuinely fast Hermes answer
    # (measured TTFB 1.6-2.2 s warm) still does not get an acknowledgement in front
    # of it for nothing. Do NOT raise this above ~1.0 s without redoing this sum.
    filler_after_seconds: float = 0.9
    # Cooldown between acknowledgements, GLOBAL TO THE CALL, not to the turn: once
    # one is spoken, no other is spoken on this call until this many seconds have
    # passed, whatever happens in between -- new turns, tool_start re-arms, or a
    # barge-in retry storm re-POSTing the same turn six times in sixteen seconds.
    # The anchor lives on CallState (call_state.py), which is why it survives turn
    # boundaries and cancelled streams. 10 s is the requirement: the callee gets an
    # immediate "I heard you", then never hears one twice in the same breath.
    filler_min_gap_seconds: float = 10.0
    # Suffix fillers with the Vapi <flush /> audio-control token so they are spoken
    # immediately instead of sitting in the TTS buffer (contracts section 1.6).
    # Requires voice.chunkPlan.enabled (the Vapi default); disable if chunking is off.
    filler_use_flush: bool = True

    # outbound task calls: the objective arrives as a Vapi dynamic variable
    # (assistantOverrides.variableValues.purpose), never from the dashboard prompt.
    outbound_opening: str = _DEFAULT_OUTBOUND_OPENING
    outbound_opening_principal: str = _DEFAULT_OUTBOUND_OPENING_PRINCIPAL
    # The principal's own number, in E.164. When set it is the authoritative way to
    # tell "we called Mike" from "we called Mike's doctor": Vapi's customer.number is
    # signalling-derived, unlike the free-text `callee` variable. Unset (the default)
    # keeps the third-party framing unless `callee` names the principal outright.
    principal_number: str | None = None
    # Disclose "I am an AI assistant calling for <principal>" in the opening when the
    # callee is somebody other than the principal. On by default: a third party who
    # answered their own phone is owed that, so switching it off must be deliberate.
    # Never applies when the principal is the one who answered.
    outbound_disclose_ai: bool = True
    # Say why the call was placed from adapter-local text, the moment the callee's
    # first utterance arrives, instead of waiting for Hermes to compose it. Off
    # restores the previous behaviour exactly: the turn goes to Hermes like any other.
    outbound_reason_fast_path: bool = True
    outbound_reason_sentence: str = _DEFAULT_OUTBOUND_REASON_SENTENCE
    outbound_reason_sentence_generic: str = _DEFAULT_OUTBOUND_REASON_SENTENCE_GENERIC

    # hermes routing. model and provider are validated as a pair below: Hermes
    # silently mis-routes (or ignores) a model with no provider.
    voice_model: str | None = None
    voice_provider: str | None = None
    # model_options.reasoning_effort, applied to every voice turn when set. Unset by
    # default -- no opinion. A previous default of "low" was sent even with no model
    # or provider configured, i.e. on the default Hermes route, where low reasoning
    # effort degrades multi-hop tool use (docs/integration-contracts.md section 2).
    # Operators who want the measured low-latency route set it explicitly alongside
    # voice_model/voice_provider.
    voice_reasoning_effort: str | None = None
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

    @field_validator("outbound_opening", "outbound_opening_principal")
    @classmethod
    def _check_outbound_opening(cls, value: str) -> str:
        if "{purpose}" not in value:
            raise ValueError("outbound opening templates must contain the {purpose} placeholder")
        return value

    @field_validator("outbound_reason_sentence")
    @classmethod
    def _check_outbound_reason_sentence(cls, value: str) -> str:
        if "{reason}" not in value:
            raise ValueError(
                "outbound_reason_sentence must contain the {reason} placeholder: without"
                " it the callee is never told why the phone rang, which is the whole"
                " point of the line"
            )
        return value

    @field_validator("principal_number")
    @classmethod
    def _check_principal_number(cls, value: str | None) -> str | None:
        if value is not None and _E164_RE.fullmatch(value) is None:
            raise ValueError("principal_number must be an E.164 number like +15551234567")
        return value

    @model_validator(mode="before")
    @classmethod
    def _drop_retired_settings(cls, values: Any) -> Any:
        """Accept-and-warn on retired VHV_ keys instead of failing to start.

        This model forbids extras, so a ``.env`` key the adapter no longer
        understands is a hard validation error -- i.e. a service that will not boot.
        That is the right answer for a typo but the wrong one for a knob *we*
        removed while it was still sitting in a deployed ``.env``: the operator
        finds out by the phone line going dead on the next restart. Retired keys are
        therefore dropped here, before extra-field checking, and named in the log so
        they get cleaned up. See ``_RETIRED_SETTINGS`` for the per-key reason.

        ``.env`` keys that match no field reach validation with the ``VHV_`` prefix
        still attached (``vhv_filler_max_per_turn``), while a direct keyword
        argument arrives bare, so both spellings are recognized.
        """
        if not isinstance(values, dict):
            return values
        prefix = cls.model_config.get("env_prefix", "").lower()
        retired = {
            key: name
            for key in values
            if isinstance(key, str)
            for name in [key.lower().removeprefix(prefix)]
            if name in _RETIRED_SETTINGS
        }
        if not retired:
            return values
        for name in retired.values():
            logger.warning(
                "ignoring retired setting VHV_%s (%s); remove it from .env",
                name.upper(),
                _RETIRED_SETTINGS[name],
            )
        return {key: value for key, value in values.items() if key not in retired}

    @model_validator(mode="after")
    def _check_voice_routing_pair(self) -> Settings:
        """voice_model and voice_provider are all-or-nothing.

        A model with no provider is silently ignored or mis-routed by Hermes
        (docs/integration-contracts.md section 2), and a provider with no model
        routes nothing: either half alone is a configuration mistake that would
        otherwise show up only as unexplained latency or a wrong answer.
        """
        if (self.voice_model is None) != (self.voice_provider is None):
            raise ValueError(
                "voice_model and voice_provider must be set together (or both left"
                " unset): Hermes silently mis-routes a model with no provider"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings instance."""
    return Settings()  # type: ignore[call-arg]  # required fields come from the environment
