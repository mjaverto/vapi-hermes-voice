"""Runtime settings for the Vapi <-> Hermes voice adapter."""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    Field,
    SecretStr,
    ValidationInfo,
    field_validator,
    model_validator,
)
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
    "ack_use_call_control": (
        "the acknowledgement is always delivered via the SSE-embedded stream, with"
        " the response ended immediately behind it; Live Call Control is now used"
        " only for the answer that follows, unconditionally, so there is no longer a"
        " kill switch to set"
    ),
    "ack_control_timeout_seconds": (
        "the acknowledgement no longer makes a control POST at all, so it has no"
        " control timeout to configure; see VHV_CONTROL_ANSWER_TIMEOUT_SECONDS for"
        " the answer delivery that replaced it"
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

# Reassurance lines: what the callee hears when a silence this call has ALREADY
# acknowledged keeps running. A separate pool from `_DEFAULT_FILLER_PHRASES` rather
# than a second draw from it, because the two moments are audibly different: eleven
# seconds into an unbroken silence the callee said nothing eleven seconds ago, so
# "Got it, one second." and "Understood, one moment." acknowledge something that did
# not happen. On a call whose reported defect was the assistant answering half a
# sentence, an assistant that says "Got it" to silence is precisely the wrong
# impression to leave. Every line here has to read correctly as the SECOND thing said
# inside one silence, and still has to be short: there is no information in it beyond
# "the line is still up".
_DEFAULT_REASSURE_PHRASES: tuple[str, ...] = (
    "Still working on that.",
    "Almost there, bear with me.",
    "Still checking, one more moment.",
    "Sorry to keep you, nearly done.",
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
    # --- the R2 acknowledgement budget, decomposed -----------------------------
    # "The callee hears something within two seconds of finishing a sentence" is
    # measured on the CALLEE'S clock -- their microphone going quiet to their speaker
    # making a sound -- and only part of that interval belongs to this process.
    # Measured live on call 01a0262b, on the callee's own audio devices, independent
    # of Vapi's timeline:
    #
    #   callee stopped talking ..................... 15.516 s
    #   callee heard the first reply ............... 17.837 s  -> heard gap 2.321 s
    #   ack emitted, this process' own journal ..... +1.130 s after the turn arrived
    #
    # so 2.321 - 1.130 = 1.191 s was spent outside this repo entirely: Vapi's
    # transcriber endpointing (Deepgram Flux eotThreshold/eotTimeoutMs) plus
    # startSpeakingPlan.waitSeconds before the request is even delivered here, and
    # the TTS/transport hop after the acknowledgement is handed back. None of that
    # can be optimised from this side, so it is budgeted rather than wished away --
    # and budgeted UP from the measured 1.191 s, because one sample does not deserve
    # to be defended to three decimal places, and the spare covers both sample-to-
    # sample variance and the scheduling jitter of the hops we do own.
    #
    #   adapter's whole share = ack_budget_seconds - ack_platform_overhead_seconds
    #                         = 2.0 - 1.25
    #                         = 0.75 s
    #
    # That whole share is now spent on filler_after_seconds alone: the acknowledgement
    # is delivered by writing it into the model.url stream (the SSE-embedded path
    # below) and ending the response immediately behind it, which is in-process work
    # with no network round trip on the critical path at all -- see vapi_control.py
    # for why a live control POST was tried there once and no longer is. Both halves
    # are settings, so a deployment behind slower endpointing raises
    # ack_platform_overhead_seconds and leaves proportionally less room to raise
    # filler_after_seconds before `_check_ack_budget` below has something to say
    # about it.
    ack_budget_seconds: float = 2.0
    ack_platform_overhead_seconds: float = 1.25
    # How long a turn may stay silent before the acknowledgement is spoken: the
    # adapter's own contribution to the sum above, and now very nearly the WHOLE of
    # it (see above -- delivery itself no longer costs a network round trip). 0.9 s
    # was the previous default and it could not fit any budget that also had to pay
    # for a control POST; 0.3 s is kept because lowering it costs nothing in spurious
    # acknowledgements, which is the only thing waiting ever bought: the point of the
    # wait is to let a genuinely fast Hermes answer arrive first and go
    # unacknowledged, but measured Hermes TTFB is 1.6-2.2 s warm and 3.6-4.9 s cold,
    # so nothing was ever arriving inside 0.9 s either. Deliberately NOT gated on a
    # tool actually running: the requirement is an acknowledgement on every turn
    # whose answer has not started, tool or no tool.
    filler_after_seconds: float = 0.3
    # Cooldown between acknowledgements, GLOBAL TO THE CALL, not to the turn: once
    # one is spoken, no other is spoken on this call until this many seconds have
    # passed, whatever happens in between -- new turns, tool_start re-arms, or a
    # barge-in retry storm re-POSTing the same turn six times in sixteen seconds.
    # The anchor lives on CallState (call_state.py), which is why it survives turn
    # boundaries and cancelled streams. 10 s is the requirement: the callee gets an
    # immediate "I heard you", then never hears one twice in the same breath.
    filler_min_gap_seconds: float = 10.0
    # Suffix fillers with the Vapi <flush /> audio-control token so they are spoken
    # immediately instead of sitting in the TTS buffer (contracts section 1.6), and
    # so the response can be ended right behind them (turns.py) instead of staying
    # open to see what Hermes does next -- ending the response immediately after the
    # flush is what makes it render reliably (proven: an isolated probe with no
    # Hermes traffic, and turn 1 of call 01a02681, both rendered in well under a
    # second; a sibling turn of the SAME call that instead kept the connection open
    # never rendered at all). Requires voice.chunkPlan.enabled (the Vapi default);
    # disable if chunking is off.
    filler_use_flush: bool = True
    # --- reassurance while a wait this call already acknowledged keeps running ----
    #
    # `filler_min_gap_seconds` above is a FLOOR on how close two holding lines may be,
    # not a budget of one per turn. Before this existed the adapter had no way to spend
    # anything the floor allowed: the acknowledgement ends the model.url response and
    # the rest of the turn runs on a background continuation that spoke only once, at
    # the end, so an acknowledged turn whose answer took a long time was dead air for
    # all of it. Measured on live call 01a028f1 (Vapi's own log): acknowledgement
    # rendered at 70.47 s and stopped at 71.45 s, the answer rendered at 95.00 s --
    # 23.55 s of unbroken silence, with the callee saying nothing at all in it. A second
    # window on the same call ran 12.45 s. Across 34 turns of adapter journal the wait
    # between the acknowledgement and the answer had p50 9.0 s, p75 14.6 s, max 33.2 s,
    # so this is the ordinary shape of a calendar turn rather than one bad call.
    #
    # Why it matters beyond comfort: a callee met with half a minute of silence speaks
    # ("hello? are you there?"), and the callee speaking is a NEW TURN, which cancels
    # the answer this adapter spent that half minute computing
    # (`CallState.supersede_pending_answer`). Silence here does not merely feel bad, it
    # destroys work and buys another wait exactly like the one that caused it.
    reassure_phrases: Annotated[list[str], NoDecode] = list(_DEFAULT_REASSURE_PHRASES)
    # How long an acknowledged silence may run before the FIRST reassurance, measured
    # from the acknowledgement that opened it. <= 0 disables reassurance entirely and
    # restores the previous behaviour exactly.
    #
    # Deliberately ABOVE `filler_min_gap_seconds` rather than equal to it. The cooldown
    # stays the single hard authority on spacing (`CallState.claim_reassurance` shares
    # its anchor), so a timer sitting exactly on the floor would make every emission a
    # boundary case: what an auditor measures is speaker-to-speaker, and the two lines
    # reach the speaker down different paths whose render lags differ (measured on
    # 01a028f1: 0.28 s for a line streamed as model output, 0.33-0.38 s for a `say`
    # push), so a timer at exactly 10.0 s can be observed at 9.9 s. One second of
    # headroom costs nothing and makes the audible gap unambiguously past the floor.
    reassure_after_seconds: float = 11.0
    # Multiplier applied to the wait before each FURTHER reassurance: the gaps go
    # `reassure_after_seconds`, then twice that, then four times, so the number of
    # lines grows with the logarithm of the wait and never with the wait itself. At the
    # shipped defaults a 25 s wait hears exactly one reassurance, a 60 s wait two, and
    # a pathological five-minute wait five.
    #
    # 1.0 gives a fixed cadence, which is the wrong shape and is why this is not simply
    # hardcoded to it: a holding phrase carries no information beyond "the line is
    # still up", so its value decays with each repetition while its cost -- sounding
    # like nagging, and one more chance to talk over a callee -- does not. It also
    # cannot be a hard cap of one: capping at one leaves a 60 s wait with 49 s of
    # trailing silence, which is the failure above, unfixed, for the longest waits.
    reassure_backoff: Annotated[float, Field(ge=1.0)] = 2.0
    # Ceiling on ONE control POST that delivers the ANSWER, in the background, once
    # the acknowledgement has already ended the model.url response
    # (`turns._finish_turn_via_control`). On no caller's deadline at all (Hermes has
    # already finished by the time this runs), sized to comfortably cover the
    # measured ~0.22-0.33s round trip to Vapi's control origin with room to spare.
    control_answer_timeout_seconds: float = 3.0
    # How long to wait between two answer-delivery attempts that both failed for a
    # plausibly transient reason (timeout, network error, or a 5xx). Deliberately NOT
    # zero: a live incident showed two attempts back to back -- 3.0s apart, the first
    # attempt's own timeout -- both fail inside the SAME multi-second bad window on
    # the control origin (this origin closes every connection it answers, so every
    # POST pays a fresh handshake; see vapi_control.py). Spacing attempts further
    # apart than one attempt's own timeout gives a genuinely independent roll of the
    # dice instead of two samples of the same outage.
    control_answer_retry_gap_seconds: float = 4.0
    # Total time, from the moment the answer is ready to speak, this adapter keeps
    # retrying before giving up and speaking a short apology instead (see
    # `turns.ANSWER_DELIVERY_FAILED_LINE`). A caller who has already been told "one
    # second" is better served by a late answer than an on-time silence, so this is
    # generous rather than tight -- it is not on the R1/R2 deadline, which governs
    # only the acknowledgement (see `ack_budget_seconds` above) and is unaffected by
    # how long this runs. Retrying is abandoned immediately, before this deadline,
    # the moment the callee's NEXT turn arrives (`CallState.pending_answer_task`):
    # answering a question that is no longer the current one would be worse than not
    # answering it at all.
    control_answer_max_wait_seconds: float = 30.0

    # --- did what we delivered actually become audio? (speech_feedback.py) --------
    #
    # Vapi accepts text on both channels -- 200 from Live Call Control, bytes taken on
    # the model.url stream -- and then sometimes never renders it, with no error event
    # anywhere (call 01a026d8: pipeline.sayQueuePush -> botSpeechStarted of 6.755 s,
    # 5.759 s, DROPPED, 3.066 s, 0.487 s, 9.610 s, DROPPED). The callee just hears
    # nothing. These four knobs govern how that is detected and recovered from; none
    # of them can be set to a value that makes the adapter speak twice, and the reasons
    # are with each one.
    #
    # How long after a delivery its ABSENCE from Vapi's committed conversation history
    # is allowed to mean it was dropped. A PRECONDITION on the verdict, never a
    # trigger for it: nothing expires into "dropped" when this elapses -- a delivery
    # with no evidence stays unconfirmed forever, because "no audio yet" is evidence of
    # unknown, not of loss. Measured basis for 15 s: the worst observed sayQueuePush
    # -> botSpeechStarted delay that DID eventually render is 9.610 s (call 01a026d8),
    # plus a few seconds of the utterance's own audio, plus Vapi's ~0.3 s
    # conversation-update commit lag. Raising it makes detection later and safer;
    # lowering it below ~10 s makes a merely-late render indistinguishable from a drop,
    # which is how a guard against silence becomes a cause of double-speaking.
    speech_confirm_window_seconds: Annotated[float, Field(gt=0)] = 15.0
    # Fraction of a delivery's content words that must be discernible in Vapi's record
    # for it to count as spoken (`speech_feedback.spoken_coverage`). Vapi returns the
    # text RE-TRANSCRIBED, not echoed -- case and punctuation gone, acronyms spelled
    # out, digits read as words -- so this cannot be an equality test. Too LOW and a
    # genuine drop scores as a success and the guard silently disarms; too HIGH and a
    # spoken utterance is condemned and re-spoken. Measured on probe calls:
    # "ANSWER ALPHA IS FIFTY MILLIGRAMS." against its own re-transcription scores 1.00,
    # "ACK ONE PLEASE HOLD." against "a c k one please hold" scores 0.75, and a
    # genuinely dropped "PROBE CHARLIE THREE." against a record containing only
    # "probe alpha one Probe bravo two." scores 0.33 -- and that 0.33 is with
    # deliberately near-identical texts sharing the word "probe", so it is a floor on
    # the real margin rather than a typical case.
    speech_match_threshold: Annotated[float, Field(gt=0.0, le=1.0)] = 0.5
    # Re-deliver an answer once when Vapi's own record proves it never became audio.
    # False leaves detection and journalling fully intact and only declines to act,
    # which is the right setting for anyone who would rather read the record than have
    # the adapter speak on its own initiative.
    speech_drop_replay: bool = True
    # An answer older than this is not worth re-speaking: the callee has moved on and
    # a late answer to a superseded question is its own defect. A freshness limit on
    # the recovery, not a trigger for it. Same order as
    # `control_answer_max_wait_seconds`, which bounds the original delivery's own
    # patience for the same reason.
    speech_drop_replay_max_age_seconds: Annotated[float, Field(gt=0)] = 45.0
    # Hard ceiling on re-deliveries per call. This is the backstop for the one failure
    # this design cannot rule out by reasoning: if Vapi ever stops committing assistant
    # messages to the history it sends, every delivery on every call would look absent
    # and the guard would fire on all of them. Two is enough to recover the realistic
    # case (one dropped answer, one dropped replay) and small enough that a systematic
    # false positive costs two utterances, not a monologue.
    speech_drop_max_replays_per_call: Annotated[int, Field(ge=0)] = 2
    # Shared secret for the OPTIONAL speech-feedback webhook (POST /vapi/server).
    # Unset (the default) means the route is NOT REGISTERED AT ALL: an assistant that
    # has been pointed at this adapter's server URL without this being configured gets
    # a 404 for every event, and no unauthenticated event is ever accepted. That is the
    # fail-closed direction, and it costs nothing -- Vapi's committed conversation
    # history is the load-bearing feedback channel and needs no configuration on either
    # side. When set it must equal the assistant's `server.secret`, which Vapi sends
    # verbatim as the `x-vapi-secret` header (verified live, call 01a0272a).
    vapi_server_secret: SecretStr | None = None

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

    # --- the adapter's own record of the acknowledgements it emitted -------------
    # Serves GET /debug/acks/{call_ref} (server.py) from an in-memory ring
    # (ack_journal.py). ON by default, which for a debug surface wants justifying:
    #
    #  - It is not a data exposure. It holds only phrases the ADAPTER chose from
    #    `filler_phrases`, the channel each went out on, and two timestamps. No caller
    #    speech, no transcript, no phone number, no secret ever enters it.
    #  - It is not an authentication weakening. The endpoint is behind the same
    #    `adapter_api_key` bearer -- and the same optional `route_secret` path prefix --
    #    as /chat/completions, so whoever can read it can already drive turns.
    #  - It is not a memory risk: see the three caps below.
    #  - And a detector that must be armed before it can detect anything is off exactly
    #    when the regression happens. What this evidence exists to catch is the model
    #    re-acquiring its own holding-phrase habit in defiance of VOICE_SYSTEM_PROMPT,
    #    which is invisible from the spoken timeline alone (the phrases are verbatim
    #    members of the adapter's own pool). If the record has to be switched on first,
    #    the first live run after such a regression reports "unknown" all over again.
    #
    # Set false to switch off recording AND unregister the route entirely (a disabled
    # deployment 404s exactly like one that never had it).
    debug_ack_journal: bool = True
    # Worst case is the product of the first two: calls retained x entries per call.
    # 64 x 16 = 1024 records, each at most `MAX_TEXT_CHARS` of text -- well under a
    # megabyte in total (see docs/security.md for the arithmetic). Sized off what the
    # data is for rather than what fits: 16 entries is far more than one call can
    # produce (the acknowledgement cooldown is `filler_min_gap_seconds`, 10 s
    # call-globally, so a 2-minute call tops out at ~12), and 64 concurrent-ish calls
    # is an order of magnitude past `max_concurrent_turns`.
    debug_ack_journal_max_calls: Annotated[int, Field(ge=1)] = 64
    debug_ack_journal_max_entries_per_call: Annotated[int, Field(ge=1)] = 16
    # Entry age cap, the third bound and the one that holds when a process runs for
    # weeks with sporadic calls: 15 minutes is long after any call this could be
    # queried about has ended, and short enough that the steady state of an idle
    # adapter is an empty journal.
    debug_ack_journal_ttl_seconds: Annotated[float, Field(gt=0)] = 900.0

    @field_validator("allowed_callers", "filler_phrases", "reassure_phrases", mode="before")
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

    @field_validator("vapi_server_secret")
    @classmethod
    def _check_vapi_server_secret(cls, value: SecretStr | None) -> SecretStr | None:
        # Same floor as the other two credentials. This one guards an endpoint that
        # accepts live call events and can cause the adapter to speak, so a short
        # secret is worse here than a missing one: unset fails closed (no route at
        # all), while a guessable value is an open door that looks shut.
        if value is not None and len(value.get_secret_value()) < _MIN_SECRET_LENGTH:
            raise ValueError("vapi_server_secret must be at least 16 characters long when set")
        return value

    @field_validator("allowed_callers")
    @classmethod
    def _check_allowed_callers(cls, value: list[str]) -> list[str]:
        for number in value:
            if _E164_RE.fullmatch(number) is None:
                raise ValueError("allowed_callers entries must be E.164 numbers like +15551234567")
        return value

    @field_validator("filler_phrases", "reassure_phrases")
    @classmethod
    def _check_phrase_pool(cls, value: list[str], info: ValidationInfo) -> list[str]:
        # Empty is rejected rather than quietly meaning "off", for both pools. An
        # operator who cleared the list is far more likely to have mis-set the env key
        # than to have chosen the one spelling of "disabled" that produces no warning;
        # `VHV_REASSURE_AFTER_SECONDS=0` is the switch, and this error is where they
        # find that out.
        if not value:
            raise ValueError(f"{info.field_name} must not be empty")
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

    @model_validator(mode="after")
    def _check_ack_budget(self) -> Settings:
        """Warn -- never refuse to start -- when the acknowledgement cannot fit in R2.

        Same judgement as :meth:`_drop_retired_settings`: a budget that no longer adds
        up is a latency regression, not a broken deployment, and crash-looping the unit
        over a missed deadline would take the phone line down to protect it. So the
        arithmetic that failed is logged in full and the run continues anyway, which is
        still the best available answer.

        Just two terms now: the acknowledgement is delivered by writing it into the
        model.url stream and ending the response immediately behind it (turns.py), so
        the adapter's own share of the budget is `filler_after_seconds` and nothing
        else -- no network round trip sits on this critical path to add a third term
        for (see vapi_control.py for why one was tried here and no longer is).
        """
        worst_case = self.ack_platform_overhead_seconds + self.filler_after_seconds
        if worst_case > self.ack_budget_seconds:
            logger.warning(
                "acknowledgement worst case %.3fs exceeds the %.3fs budget:"
                " %.3fs platform overhead + %.3fs filler_after."
                " Lower VHV_FILLER_AFTER_SECONDS, or raise VHV_ACK_BUDGET_SECONDS if the"
                " requirement really did change",
                worst_case,
                self.ack_budget_seconds,
                self.ack_platform_overhead_seconds,
                self.filler_after_seconds,
            )
        return self

    @model_validator(mode="after")
    def _check_pools_are_disjoint(self) -> Settings:
        """The two holding-phrase pools may not share a line.

        Refused rather than warned, unlike the two validators above, because this one
        is not a missed target -- it silently breaks a stated guarantee. A phrase in
        both pools can be drawn as the acknowledgement and then, ten seconds later, as
        the reassurance: ``FillerPicker`` refuses to repeat its OWN previous pick and
        the two pools have separate pickers, so nothing else in the system stops it.
        The result is the callee hearing the same sentence twice in one silence, which
        is the exact complaint the cooldown was built for. There is also nothing to
        lose by refusing: an operator who wants a line in both moments can simply not
        want that, and every deployment that has never set either key is unaffected.
        """
        shared = sorted(set(self.filler_phrases) & set(self.reassure_phrases))
        if shared:
            raise ValueError(
                "filler_phrases and reassure_phrases must not share a line"
                f" (shared: {shared}): a phrase in both pools can be spoken twice in"
                " one silence, which is what the acknowledgement cooldown exists to"
                " prevent"
            )
        return self

    @property
    def holding_phrases(self) -> list[str]:
        """Every line the ADAPTER can speak as a holding phrase, both pools together.

        The union, not either pool, is what "a holding phrase" means to everything
        that does not choose one: the model is forbidden from producing any of them
        (``speech.SpokenTurn``, built from this), and the E2E harness matches spoken
        audio against this to score R2 (tests/e2e -- a pool it does not know about is
        a holding phrase it cannot see, which reads as "no acknowledgement came").
        Order is deterministic -- acknowledgements then reassurances -- so anything
        that hashes or prints it is stable across restarts.
        """
        return [*self.filler_phrases, *self.reassure_phrases]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings instance."""
    return Settings()  # type: ignore[call-arg]  # required fields come from the environment
