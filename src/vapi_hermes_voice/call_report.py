"""What the principal is told after a call placed on his behalf. His rule, verbatim:

    "it doesn't need to actually do anything else other than pick a time. It can pick
    a time, then come back, TELL ME THE TIME IT PICKED, and if I decide it doesn't
    work, it can always call back and change the time."

The first half shipped as prompt conduct (docs/integration-contracts.md §1.13): decide,
commit, and read the booking back on the line. This module is the second half. Without
it the spoken line in a transcript nobody reads was the whole report, and the veto the
rule depends on -- "if I decide it doesn't work" -- cannot be exercised over
information that never arrives.

WHO SENDS THIS. Not this process. The adapter has no channel to the principal and must
not be given one: its Hermes session runs under an ``enabled_tools`` allowlist with no
messaging tool, and Hermes's own ``api_server`` toolset -- the surface the adapter talks
to -- has no ``messaging`` either. Two independent allowlists say no, and they are right
to: a counterparty on a live call would otherwise be one prompt injection away from
sending the principal arbitrary text. Hermes delivers, over the Telegram conversation it
already owns. See §1.14 for the full argument.

So this module is deliberately a PURE RENDERER: payload in, the exact text the principal
receives out. No network, no I/O, no LLM, no clock. It is called from
``python -m vapi_hermes_voice.call_report`` (see ``__main__``) with a Vapi call object on
stdin, which is how the component that does own the channel gets a report it cannot
accidentally embellish.

WHY NOT LET AN LLM WRITE IT. Because the failure cases are the point. A model asked to
summarise a call that achieved nothing writes something that reads like progress. The
three verdicts here are computed, not composed, and the two that mean "you have nothing"
say so in their first two words.

MIKE-FACING ONLY. A sibling change made calendar tools withhold event details on
caller-facing turns. This text is the opposite case -- it is for the principal, so full
detail is correct -- and that asymmetry is exactly why it must never be routed anywhere
else. Nothing here is safe to speak on a call.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

__all__ = [
    "Booking",
    "BookingUnknown",
    "CallFacts",
    "CallReport",
    "NothingBooked",
    "build_report",
    "extract_call_facts",
]

# `purpose` and `callee` arrive as Vapi dynamic variables, i.e. UNTRUSTED free text set
# by whoever created the call. `vapi_events._clean_variable` already applies this
# treatment on the request path; the same discipline is applied again here because this
# module's input is a raw API payload that never went through it. Control characters are
# what would let a value forge the line structure of the report it is being placed into.
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_WS_RE = re.compile(r"\s+")

MAX_OBJECTIVE_CHARS = 300
MAX_PARTY_CHARS = 120
MAX_BOOKING_FIELD_CHARS = 200
# `endedReason` is a Vapi enum, not free text, but it is still their string in our
# message. Capped for the same reason and echoed verbatim otherwise: a reason we have
# never seen must survive to the principal legibly rather than be mapped to "other".
MAX_REASON_CHARS = 120


def _clean(value: object, *, limit: int) -> str | None:
    """One untrusted payload string, safe to place in a line-structured report."""
    if not isinstance(value, str):
        return None
    text = _WS_RE.sub(" ", _CONTROL_RE.sub(" ", value)).strip()
    if not text:
        return None
    return text[:limit].rstrip() if len(text) > limit else text


# --- what the reporter vouches for ------------------------------------------------
#
# Three cases, as three types rather than an optional. That is a deliberate refusal of
# the obvious `Booking | None`, because `None` would have to mean both "nothing was
# booked" and "I never found out", and §6 rule 1 exists precisely to stop those two
# collapsing into each other. Here the caller cannot express "nothing booked" by
# accident: it has a name, and the default is ignorance.


@dataclass(frozen=True, slots=True)
class Booking:
    """A commitment the reporter VOUCHES was made on the principal's behalf.

    Only the component that made the booking can supply this. Hermes can: it holds the
    objective the principal gave it and it performed the booking with its own calendar
    tools, so it reads this off a tool result rather than inferring it from a
    transcript. This adapter could not, which is a second reason the report is not
    delivered from here.
    """

    what: str
    when: str
    with_whom: str


@dataclass(frozen=True, slots=True)
class NothingBooked:
    """The reporter positively vouches that no commitment was made."""


@dataclass(frozen=True, slots=True)
class BookingUnknown:
    """The reporter does not know whether a commitment was made.

    The DEFAULT, and never an error. A report built on this says so in its first two
    words rather than guessing in either direction -- guessing "booked" invents an
    appointment, and guessing "nothing booked" invites a double booking.
    """


Claim = Booking | NothingBooked | BookingUnknown

Outcome = Literal["booked", "no_booking", "failed", "unknown"]

# --- reading the call --------------------------------------------------------------

# Substrings that mean the conversation never happened, matched against a casefolded
# `endedReason`. Substrings and not an enum on purpose: Vapi's reason list is long,
# undocumented in full, and grows, and an unrecognised reason must not silently read as
# a normal ending. Everything here fails the call CLOSED -- toward "you have nothing" --
# which is the safe direction for a report whose job is to prevent false confidence.
#
# `error` covers the whole `call.in-progress.error-*` and `pipeline-error-*` family,
# including the one observed live on this account
# (`call.in-progress.error-providerfault-transport-never-connected`).
_FAILURE_REASON_MARKERS: tuple[str, ...] = (
    "error",
    "failed",
    "did-not-answer",
    "no-answer",
    "busy",
    "voicemail",
    "rejected",
    "unallocated",
    "invalid",
    "forbidden",
    "not-found",
    "never-connected",
    "call-deleted",
    "customer-did-not-give-microphone-permission",
)


@dataclass(frozen=True, slots=True)
class CallFacts:
    """The parts of a settled Vapi call this report is allowed to rest on.

    Nothing derived and nothing guessed: every field is either present in the payload
    or ``None``, and ``None`` is carried through to the text as an admission rather
    than filled in with a default. ``transcript`` is held because "the call connected
    and nobody said anything" is a real and distinct outcome from "the call went fine";
    it is never quoted into the report, only measured.
    """

    call_id: str | None
    connected: bool
    ended: bool
    ended_reason: str | None
    duration_s: float | None
    transcript_chars: int
    objective: str | None
    counterparty: str | None
    # Vapi's own post-call analysis, when the account produces any. On this account it
    # does not: assistant b39379dc has `analysisPlan.summaryPlan.enabled = false` and
    # `successEvaluationPlan.enabled = false`, and every ended call sampled came back
    # with `analysis == {}` and `summary == ""` (§1.14). Read anyway, because reading a
    # field that is empty today costs nothing and turning those plans on is a one-field
    # assistant edit somebody may well make later. Never load-bearing: no verdict below
    # consults them.
    vapi_summary: str | None
    vapi_success_evaluation: str | None


def _first_mapping(payload: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    for key in keys:
        node = payload.get(key)
        if isinstance(node, Mapping):
            return node
    return {}


def _parse_iso(value: object) -> datetime | None:
    """Vapi timestamps are RFC 3339 with a `Z`, which `fromisoformat` rejects on 3.11."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def extract_call_facts(payload: Mapping[str, Any]) -> CallFacts:
    """Read one Vapi payload, in either of the two shapes that carry a finished call.

    Accepts BOTH a bare call object (``GET /call/{id}``) and an ``end-of-call-report``
    server message (``{"message": {...}}``), because they carry the same facts at
    different depths -- the same tolerance ``vapi_events._VARIABLE_PATHS`` already
    applies to ``variableValues``. One normaliser rather than two is not a convenience
    here: it is what stops a report assembled from a webhook push and one assembled
    from a poll disagreeing about the same call.

    Never raises. A payload missing everything yields facts that are honestly empty,
    and :func:`build_report` renders those as "outcome unknown" rather than as success.
    """
    message = _first_mapping(payload, "message")
    # In an end-of-call-report the call object is nested; in a poll it IS the payload.
    call = _first_mapping(message, "call") or payload
    artifact = _first_mapping(message, "artifact") or _first_mapping(call, "artifact")
    analysis = _first_mapping(message, "analysis") or _first_mapping(call, "analysis")

    def pick(key: str) -> Any:
        for source in (message, call):
            if key in source and source[key] is not None:
                return source[key]
        return None

    started = _parse_iso(pick("startedAt"))
    ended_at = _parse_iso(pick("endedAt"))
    duration = (ended_at - started).total_seconds() if started and ended_at else None
    if duration is None:
        raw_seconds = pick("durationSeconds")
        duration = float(raw_seconds) if isinstance(raw_seconds, (int, float)) else None

    transcript = pick("transcript") or artifact.get("transcript")
    variables = _first_mapping(artifact, "variableValues") or _first_mapping(
        _first_mapping(call, "assistantOverrides"), "variableValues"
    )
    ended_reason = _clean(pick("endedReason"), limit=MAX_REASON_CHARS)

    return CallFacts(
        call_id=_clean(pick("id") or call.get("id"), limit=64),
        # `startedAt` is Vapi's own record of the media path opening. Absent means the
        # call never became a conversation, whatever else the payload claims.
        connected=started is not None,
        # Any ONE of the three is proof the call is over. `endedReason` counts because
        # Vapi only sets it once it has stopped, and an `end-of-call-report` has been
        # seen carrying a reason with no `endedAt` -- without this such a call reads as
        # still running, and the report would tell the principal to go and check a call
        # that had already finished hours ago.
        ended=ended_at is not None or pick("status") == "ended" or ended_reason is not None,
        ended_reason=ended_reason,
        duration_s=duration,
        transcript_chars=len(transcript) if isinstance(transcript, str) else 0,
        objective=_objective_from(variables, limit=MAX_OBJECTIVE_CHARS),
        counterparty=_counterparty_from(variables, limit=MAX_PARTY_CHARS),
        vapi_summary=_clean(pick("summary") or analysis.get("summary"), limit=1000),
        vapi_success_evaluation=_clean(analysis.get("successEvaluation"), limit=200),
    )


def _lookup(variables: Mapping[str, Any], aliases: tuple[str, ...], *, limit: int) -> str | None:
    """First usable value among ``aliases``, matched the way the request path matches.

    Imported from ``vapi_events`` rather than reimplemented so a key that reaches the
    model as the objective is the same key that reaches the principal as the objective.
    A live call once sent ``call_purpose`` where a literal lookup wanted ``purpose``
    and the objective was silently discarded; that bug is not worth having twice.
    """
    from .vapi_events import _normalize_key

    folded = {_normalize_key(k): v for k, v in variables.items() if isinstance(k, str)}
    for alias in aliases:
        cleaned = _clean(folded.get(_normalize_key(alias)), limit=limit)
        if cleaned is not None:
            return cleaned
    return None


def _objective_from(variables: Mapping[str, Any], *, limit: int) -> str | None:
    from .vapi_events import PURPOSE_ALIASES

    return _lookup(variables, PURPOSE_ALIASES, limit=limit)


def _counterparty_from(variables: Mapping[str, Any], *, limit: int) -> str | None:
    from .vapi_events import CALLEE_ALIASES

    return _lookup(variables, CALLEE_ALIASES, limit=limit)


def _is_failure_reason(reason: str | None) -> bool:
    if reason is None:
        return False
    folded = reason.casefold()
    return any(marker in folded for marker in _FAILURE_REASON_MARKERS)


# --- the verdict ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CallReport:
    """One call's outcome and the exact text the principal is sent.

    ``outcome`` is the whole point and has four values, not three. ``"unknown"`` is a
    first-class answer under the §6 obligation: a call that was placed and whose end
    was never observed must be distinguishable from one that ended with nothing
    agreed, because the first means "go and check" and the second means "you are free".
    Collapsing them is how a reader ends up asserting a completeness nobody vouched
    for.
    """

    outcome: Outcome
    facts: CallFacts
    claim: Claim
    text: str

    @property
    def objective_met(self) -> bool | None:
        """Whether the call achieved what it was placed to achieve; ``None`` if unknown.

        For a call whose job is to settle a time, a vouched-for booking IS the
        objective met -- there is nothing further to judge. Everything else is either a
        definite no or a definite don't-know, and never a maybe.
        """
        return {"booked": True, "no_booking": False, "failed": False, "unknown": None}[self.outcome]


def _classify(facts: CallFacts, claim: Claim) -> Outcome:
    """Decide the verdict from evidence, failing toward "you have nothing".

    Every verdict except ``"unknown"`` requires POSITIVE evidence. That is the whole
    shape of this function, and it is not the obvious shape: the tempting version reads
    "no start timestamp, so the call failed", which turns an empty or truncated payload
    into a confident "CALL FAILED -- there was no conversation" about a call that may
    have gone perfectly. Absence of evidence is ``"unknown"``, which is a real answer
    here (§6 rule 1) and the only honest one.

    Order matters and encodes the precedence the principal needs:

    1. Failure, but only when something SAYS so: a failure-shaped ``endedReason``, or a
       call that is over and never connected, or a call that is over and connected and
       on which nobody said one word. A claim cannot override this -- a booking "made"
       on a call that never happened is a bug in the reporter, and rendering it as
       booked would hide that bug behind a plausible appointment he would show up for.
    2. A call not known to have ended is ``unknown``, even with a claim: the
       counterparty may still be talking and the time may still move.
    3. Only then does the claim decide, and only a claim can produce ``booked`` or
       ``no_booking``. An absent claim stays ``unknown``, because silence from the
       reporter is not evidence that nothing was booked.
    """
    if _is_failure_reason(facts.ended_reason):
        return "failed"
    if facts.ended and not facts.connected:
        return "failed"
    # Over, connected, and not one word from anybody. Nothing could have been agreed,
    # and calling this "nothing booked" would imply a conversation took place and went
    # nowhere, which is a different thing to tell someone.
    if facts.ended and facts.transcript_chars == 0:
        return "failed"
    if not facts.ended:
        return "unknown"
    if isinstance(claim, Booking):
        return "booked"
    if isinstance(claim, NothingBooked):
        return "no_booking"
    return "unknown"


def _duration_phrase(seconds: float | None) -> str:
    if seconds is None:
        return "duration not recorded"
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{int(seconds) // 60}m {int(seconds) % 60:02d}s"


def _call_line(facts: CallFacts) -> str:
    """The one line of provenance, so the principal can always find the call itself."""
    parts = [f"Call {facts.call_id}" if facts.call_id else "Call (id not recorded)"]
    if facts.connected:
        parts.append(_duration_phrase(facts.duration_s))
    else:
        parts.append("never connected")
    if facts.ended_reason:
        parts.append(facts.ended_reason)
    return ", ".join(parts) + "."


def _context_lines(facts: CallFacts, *, verdict: str) -> list[str]:
    lines: list[str] = []
    if facts.counterparty:
        lines.append(f"With: {facts.counterparty}")
    if facts.objective:
        lines.append(f"Objective: {facts.objective} -- {verdict}.")
    else:
        # No objective in the payload is itself worth saying. Staying silent about it
        # would let the report imply it had checked the call against a goal it never
        # saw, which is the "completeness signal it cannot vouch for" §6 forbids.
        lines.append(f"Objective: not recorded on this call -- {verdict}.")
    return lines


def render(outcome: Outcome, facts: CallFacts, claim: Claim) -> str:
    """The exact message the principal receives.

    Plain text, no markup: it is relayed over Telegram and any markup would be one
    more thing to get wrong in a message whose whole job is to be unambiguous. The
    verdict is the first word of the first line in every case, so a glance at a
    notification is enough to know whether anything is in the calendar.
    """
    if outcome == "booked":
        assert isinstance(claim, Booking)  # _classify returns "booked" only for these
        return "\n".join(
            [
                f"BOOKED -- {claim.when}",
                f"What: {claim.what}",
                f"With: {claim.with_whom}",
                *(
                    [f"Objective: {facts.objective} -- met."]
                    if facts.objective
                    else ["Objective: not recorded on this call -- a time was settled."]
                ),
                _call_line(facts),
                "",
                "If that time does not work, tell me and I will call back and change it.",
            ]
        )
    if outcome == "no_booking":
        return "\n".join(
            [
                "NOTHING BOOKED -- the call happened and no time was agreed.",
                *_context_lines(facts, verdict="NOT met"),
                _call_line(facts),
                "",
                "Nothing was added to your calendar. Tell me if you want me to try again.",
            ]
        )
    if outcome == "failed":
        return "\n".join(
            [
                "CALL FAILED -- there was no conversation, so nothing was agreed.",
                *_context_lines(facts, verdict="NOT met"),
                _call_line(facts),
                "",
                "Nothing was added to your calendar. Tell me if you want me to try again.",
            ]
        )
    # Two genuinely different ignorances, and telling them apart is the whole reason
    # this branch is not one string. "The call is still running or was never observed
    # to end" is a fact about the CALL and means go and look at it. "It ended and
    # nobody told me what came of it" is a fact about the REPORTER -- the call is over,
    # the transcript is sitting there, and what is missing is the one sentence Hermes
    # was supposed to add. Merging them would send the principal to check a finished
    # call as though it might still be live, and would hide the case that actually
    # needs chasing: the booking nobody wrote down.
    if not facts.ended:
        return "\n".join(
            [
                "OUTCOME UNKNOWN -- the call was placed and I never saw it end.",
                *_context_lines(facts, verdict="unknown"),
                _call_line(facts),
                "",
                "Do NOT assume a time was booked, and do not assume one was not."
                " Ask me to check this call before you rely on either.",
            ]
        )
    return "\n".join(
        [
            "OUTCOME UNKNOWN -- the call finished and I was not told whether anything was booked.",
            *_context_lines(facts, verdict="unknown"),
            _call_line(facts),
            "",
            "Do NOT assume a time was booked, and do not assume one was not."
            " Ask me to read this call back before you rely on either.",
        ]
    )


def build_report(payload: Mapping[str, Any], *, claim: Claim | None = None) -> CallReport:
    """One Vapi payload plus what the reporter vouches for -> the principal's message.

    ``claim`` defaults to :class:`BookingUnknown`, so a caller that forgets to say what
    it booked produces "outcome unknown" and not a confident silence. That default is
    the single most important line in this module: every other failure mode here is
    visible, and this is the one that would have been quiet.
    """
    facts = extract_call_facts(payload)
    settled: Claim = BookingUnknown() if claim is None else claim
    outcome = _classify(facts, settled)
    return CallReport(
        outcome=outcome, facts=facts, claim=settled, text=render(outcome, facts, settled)
    )
