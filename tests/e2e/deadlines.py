"""Deadline arithmetic over a Vapi call object. Pure: no I/O, no clock, no network.

Everything here takes a recorded ``GET /call/{id}`` payload and produces the verdicts
the harness prints. It is separated from the live driver precisely so it can be tested
deterministically against real recorded calls -- see ``tests/test_e2e_deadlines.py``.

Two traps this module exists to defuse
--------------------------------------

1. ``messages[].duration`` is in MILLISECONDS, even though Vapi's own OpenAPI
   description says "The duration of the message in seconds". Measured on live call
   ``01a025d2-3f8f-7dd8-a03a-7f03a6b68643``: a user message with
   ``time=1787340932548``, ``endTime=1787340932622`` (74 ms of wall clock) carries
   ``duration: 74``. Treating that as seconds turns a 0.8 s reply gap into a negative
   number and every deadline check silently passes. So the unit is *derived* from
   ``endTime - time`` on every decidable message and disagreement is a hard error.

2. An assistant with a static ``firstMessage`` and
   ``firstMessageMode="assistant-waits-for-user"`` answers the callee's first
   utterance from that fixed string -- no LLM, no adapter, no Hermes. Proven by
   pointing ``model.url`` at a guaranteed-404 endpoint and overriding the system
   prompt: the assistant still spoke the identical sentence, 0.56 s after the
   callee's "Hello?". A harness that only measured the R1 deadline would therefore
   report PASS with the adapter deleted. :func:`evaluate` reports *provenance* as a
   check of its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Literal

__all__ = [
    "Budgets",
    "Check",
    "Report",
    "TimelineUnitError",
    "Turn",
    "Utterance",
    "carries_flush_token",
    "duration_anomalies",
    "evaluate",
    "evaluate_transport",
    "load_utterances",
    "normalize_phrase",
    "r1_transport_scope",
    "render_table",
]

Verdict = Literal["pass", "fail", "skip"]

# Roles Vapi uses in `messages[]`. "bot" is the assistant; "system" is the prompt.
_USER = "user"
_BOT = "bot"

_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")
_WS_RE = re.compile(r"\s+")
# The Vapi audio-control token the adapter appends to fillers so they are spoken
# immediately (`filler_use_flush`). It is never audible, so it must not defeat
# phrase-pool matching.
_FLUSH_RE = re.compile(r"<\s*flush\s*/?\s*>", re.IGNORECASE)


class TimelineUnitError(RuntimeError):
    """``messages[].duration`` could not be resolved to a unit with confidence."""


def normalize_phrase(text: str) -> str:
    """Lowercase, de-punctuated, single-spaced form used for all text matching."""
    stripped = _FLUSH_RE.sub(" ", text or "")
    return _WS_RE.sub(" ", _PUNCT_RE.sub(" ", stripped.lower())).strip()


def carries_flush_token(text: str) -> bool:
    """Whether ``text`` is an adapter holding line, by its own structural marker.

    ``turns.build_filler`` is the only place that appends Vapi's ``<flush />`` audio
    control token, so its presence identifies a holding line without reference to any
    phrase list. That matters: the deployed pool drifts from the repo default, and a
    check that can only recognise the phrases it was told about cannot report the drift.
    """
    return bool(_FLUSH_RE.search(text or ""))


@dataclass(frozen=True, slots=True)
class Utterance:
    """One spoken turn, in seconds from the start of the call."""

    index: int
    role: str
    text: str
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True, slots=True)
class Turn:
    """A callee utterance and the assistant utterance that answered it."""

    user: Utterance
    reply: Utterance | None

    @property
    def gap_s(self) -> float | None:
        """Silence the callee actually sat through, end of their speech to start of ours."""
        if self.reply is None:
            return None
        return self.reply.start_s - self.user.end_s


@dataclass(frozen=True, slots=True)
class Budgets:
    """Wall-clock budgets, in seconds. Defaults are the two reported requirements."""

    # R1: callee's first utterance ends -> assistant starts saying why it called.
    reason_deadline_s: float = 2.0
    # R2: callee stops talking -> a short acknowledgement is spoken.
    ack_deadline_s: float = 2.0
    # R2: call-global floor between one acknowledgement and the next.
    ack_cooldown_s: float = 10.0
    # The reported ack-storm failure was six acknowledgements inside 16 s, at offsets
    # 0.05/0.22/0.40/0.57/0.74/0.91 s -- a genuine burst, not two ordinary
    # acknowledgements correctly spaced further apart than `ack_cooldown_s`. A 16 s
    # window is wide enough to hold more than one *correctly* spaced acknowledgement,
    # so `ack_storm_max` below must be derived from the cooldown, not hardcoded to 1:
    # a hardcoded 1 flags two acknowledgements 11.6 s apart (which the cooldown check
    # already, correctly, passes) as a storm.
    ack_storm_window_s: float = 16.0
    # Flux regression guard: no turn in the call may leave the callee waiting longer
    # than this. Live failure 01a02524 sat at 13.6 s.
    max_turn_gap_s: float = 3.0

    @property
    def ack_storm_max(self) -> int:
        """Most acknowledgements any `ack_storm_window_s` window can hold without also
        violating `ack_cooldown_s` -- DERIVED, never configured independently, so the
        storm check and the cooldown check can never disagree. A call that respects
        the cooldown always passes the storm check too; the only way to fail it is to
        pack strictly more acknowledgements into the window than the cooldown allows,
        which is exactly the reported failure this check exists to catch (see above).

        With ``k`` acknowledgements each at least ``ack_cooldown_s`` apart, the span
        from the first to the last is at least ``(k - 1) * ack_cooldown_s``. For all
        ``k`` to fit in `_max_in_window`'s half-open window of length
        `ack_storm_window_s`, that span must be strictly less than the window, which
        bounds ``k`` from above.
        """
        if self.ack_cooldown_s <= 0:
            return 1
        return int((self.ack_storm_window_s - 1e-9) // self.ack_cooldown_s) + 1


@dataclass(frozen=True, slots=True)
class Check:
    id: str
    title: str
    verdict: Verdict
    detail: str
    measured_s: float | None = None
    budget_s: float | None = None
    # Most checks measure seconds. The storm check measures a count of utterances and
    # provenance measures a similarity ratio; printing either as "6.000s" invites exactly
    # the unit confusion this module exists to prevent.
    unit: Literal["seconds", "count", "ratio"] = "seconds"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "verdict": self.verdict,
            "measured": None if self.measured_s is None else round(self.measured_s, 3),
            "unit": self.unit,
            "budget": self.budget_s,
            "detail": self.detail,
        }

    def render_measured(self) -> str:
        if self.measured_s is None:
            return "n/a"
        if self.unit == "count":
            return f"{self.measured_s:.0f}"
        if self.unit == "ratio":
            return f"{self.measured_s:.3f}"
        return f"{self.measured_s:.3f}s"


@dataclass
class Report:
    checks: list[Check]
    turns: list[Turn]
    utterances: list[Utterance]
    duration_unit: str
    acks: list[Utterance] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.verdict != "fail" for c in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "duration_unit": self.duration_unit,
            "checks": [c.as_dict() for c in self.checks],
            "acknowledgements": [
                {"index": a.index, "start_s": round(a.start_s, 3), "text": a.text}
                for a in self.acks
            ],
            "timeline": [
                {
                    "index": u.index,
                    "role": u.role,
                    "start_s": round(u.start_s, 3),
                    "end_s": round(u.end_s, 3),
                    "text": u.text,
                }
                for u in self.utterances
            ],
            "notes": list(self.notes),
        }


# --- unit resolution -------------------------------------------------------------


# The two hypotheses differ by a factor of 1000, so discriminating them needs nothing
# tighter than an order of magnitude. It must NOT be tighter: `duration` and
# `endTime - time` genuinely disagree on assistant messages, because one is the length of
# the synthesised audio and the other is a pair of pipeline timestamps. Measured live on
# call 01a025ea: a bot message with duration=2955 spanned 2228 ms of wall clock, a 33 %
# divergence. An early version of this function treated that as a format change and
# refused to score a perfectly good call.
_UNIT_RATIO_WINDOW = (0.25, 4.0)

# Divergence above this is reported as a note. It is information about Vapi's own
# bookkeeping, not a reason to throw the run away.
_DIVERGENCE_NOTE_THRESHOLD = 0.25


def _decide_unit(messages: list[dict[str, Any]]) -> str:
    """Resolve whether ``duration`` is seconds or milliseconds, or raise.

    Decided against ``endTime - time``, which is unambiguously milliseconds. Guessing
    would silently invert every measurement in the report, so an undecidable timeline is
    an error rather than a default.

    One message that fits neither hypothesis does not veto the run -- Vapi's per-message
    bookkeeping is not always self-consistent, and throwing away a whole billable call
    over one odd row is the wrong trade. The unit is taken from the messages that do
    decide, and only a timeline where *nothing* decides is an error.
    """
    low, high = _UNIT_RATIO_WINDOW
    votes: set[str] = set()
    rejected: list[str] = []
    for m in messages:
        dur, start, end = m.get("duration"), m.get("time"), m.get("endTime")
        if not isinstance(dur, int | float) or dur <= 0:
            continue
        if not isinstance(start, int | float) or not isinstance(end, int | float):
            continue
        wall_ms = float(end) - float(start)
        if wall_ms <= 0:
            continue
        as_ms = float(dur) / wall_ms
        as_s = float(dur) * 1000.0 / wall_ms
        ms_fits, s_fits = low <= as_ms <= high, low <= as_s <= high
        if ms_fits and s_fits:  # only reachable for sub-millisecond spans
            votes.add("ms" if abs(as_ms - 1.0) <= abs(as_s - 1.0) else "s")
        elif ms_fits:
            votes.add("ms")
        elif s_fits:
            votes.add("s")
        else:
            rejected.append(
                f"duration={dur} is neither ~{wall_ms:.0f} ms nor ~{wall_ms / 1000:.3f} s "
                f"(off by {as_ms:.2f}x and {as_s:.4f}x)"
            )
    if not votes:
        if rejected:
            raise TimelineUnitError(
                "no message's duration matches its own wall clock in either unit, so the "
                "Vapi timeline format has changed and every deadline below would be "
                f"wrong -- fix tests/e2e/deadlines.py before trusting a run. Samples: "
                f"{rejected[:3]}"
            )
        raise TimelineUnitError(
            "no message carried duration+time+endTime, so duration's unit cannot be "
            "derived; refusing to guess (Vapi documents seconds and emits milliseconds)"
        )
    if len(votes) != 1:
        raise TimelineUnitError(f"messages disagree on duration's unit: {sorted(votes)}")
    return votes.pop()


def duration_anomalies(call: dict[str, Any]) -> list[str]:
    """Messages whose ``duration`` disagrees materially with their own wall clock.

    Expected on assistant messages and worth printing: a holding line whose audio is
    much longer than its pipeline span is how the "Sure. Give me a second Sure. Give me
    a second" double-speak showed up on live call 01a025ea.
    """
    out: list[str] = []
    for m in call.get("messages") or []:
        if not isinstance(m, dict) or m.get("role") not in (_USER, _BOT):
            continue
        dur, start, end = m.get("duration"), m.get("time"), m.get("endTime")
        if not all(isinstance(v, int | float) for v in (dur, start, end)):
            continue
        wall_ms = float(end) - float(start)  # type: ignore[arg-type]
        if wall_ms <= 0:
            continue
        drift = abs(float(dur) - wall_ms) / wall_ms  # type: ignore[arg-type]
        if drift > _DIVERGENCE_NOTE_THRESHOLD:
            out.append(
                f"{m.get('role')} message {str(m.get('message'))[:40]!r} reports "
                f"duration={dur} ms but spans {wall_ms:.0f} ms of wall clock "
                f"({drift * 100:.0f}% divergence)"
            )
    return out


def load_utterances(call: dict[str, Any]) -> tuple[list[Utterance], str]:
    """``call["messages"]`` -> spoken utterances in seconds, plus the unit detected."""
    messages = [m for m in (call.get("messages") or []) if isinstance(m, dict)]
    spoken = [m for m in messages if m.get("role") in (_USER, _BOT)]
    unit = _decide_unit(spoken)
    divisor = 1000.0 if unit == "ms" else 1.0

    out: list[Utterance] = []
    for i, m in enumerate(spoken):
        start = m.get("secondsFromStart")
        if not isinstance(start, int | float):
            continue
        raw = m.get("duration")
        dur = float(raw) / divisor if isinstance(raw, int | float) else 0.0
        out.append(
            Utterance(
                index=i,
                role=str(m.get("role")),
                text=str(m.get("message") or ""),
                start_s=float(start),
                end_s=float(start) + max(0.0, dur),
            )
        )
    out.sort(key=lambda u: (u.start_s, u.index))
    return out, unit


# How alike a spoken utterance and the configured firstMessage must be to be judged the
# same utterance. The comparison is against a transcript, so it has to tolerate
# recognition noise; 0.85 sits far above the observed noise floor (0.994 for a one-letter
# surname error) and far below any genuinely different sentence.
_PROVENANCE_SIMILARITY = 0.85


def _utterance_similarity(spoken: str, static: str) -> float:
    """How likely ``spoken`` is a transcript of ``static``, in [0, 1].

    Compared over the shared head as well as in full, so an utterance cut short by a
    barge-in still matches the string it was reading from.
    """
    if not spoken or not static:
        return 0.0
    head = min(len(spoken), len(static))
    if head < 12:  # too little text for a ratio to mean anything
        return 1.0 if spoken == static else 0.0
    return max(
        SequenceMatcher(None, spoken, static).ratio(),
        SequenceMatcher(None, spoken[:head], static[:head]).ratio(),
    )


# --- turn pairing ----------------------------------------------------------------


def pair_turns(utterances: list[Utterance]) -> list[Turn]:
    """Each callee utterance with the first assistant utterance that starts after it.

    An assistant utterance already under way when the callee stopped talking is not a
    reply to it -- that is a barge-in -- so it is skipped rather than credited as a
    fast response.
    """
    bots = [u for u in utterances if u.role == _BOT]
    turns: list[Turn] = []
    for user in (u for u in utterances if u.role == _USER):
        reply = next((b for b in bots if b.start_s >= user.end_s), None)
        turns.append(Turn(user=user, reply=reply))
    return turns


def find_acks(utterances: list[Utterance], phrases: list[str]) -> list[Utterance]:
    """Assistant utterances that are one of the configured acknowledgement phrases.

    Matched on the normalized form, and on prefix as well as equality: an
    acknowledgement is routinely flushed ahead of real content in the same message
    ("One moment while I check. The vet prescribed ..."), and that still counts as the
    acknowledgement the callee heard.
    """
    pool = [p for p in (normalize_phrase(p) for p in phrases) if p]
    acks: list[Utterance] = []
    for u in utterances:
        if u.role != _BOT:
            continue
        norm = normalize_phrase(u.text)
        if any(norm == p or norm.startswith(p) for p in pool):
            acks.append(u)
    return acks


def _max_in_window(starts: list[float], window_s: float) -> tuple[int, float | None]:
    """Largest count of ``starts`` inside any half-open ``window_s`` window."""
    best, at = 0, None
    for i, left in enumerate(starts):
        count = sum(1 for s in starts[i:] if s - left < window_s)
        if count > best:
            best, at = count, left
    return best, at


# --- evaluation ------------------------------------------------------------------


def _mode(call: dict[str, Any]) -> str:
    """The call's ``firstMessageMode``, tolerating ``"assistant": null``.

    ``GET /call/{id}`` omits the embedded assistant on some calls and returns it as
    JSON null on others, so ``call.get("assistant", {})`` is not enough.
    """
    assistant = call.get("assistant")
    mode = assistant.get("firstMessageMode") if isinstance(assistant, dict) else None
    return str(mode) if mode else "assistant-waits-for-user"


def evaluate(
    call: dict[str, Any],
    *,
    phrases: list[str],
    budgets: Budgets | None = None,
    first_message: str | None = None,
    require_adapter_provenance: bool = True,
    ack_turn_index: int | None = -1,
    expected_callee_turns: int | None = None,
) -> Report:
    """Score a finished call against the two deadline requirements.

    ``first_message`` is the assistant's configured static ``firstMessage``; when the
    utterance that met the R1 deadline *is* that string, the deadline was met by Vapi
    and the adapter's fast path was never exercised. ``ack_turn_index`` selects which
    callee turn the R2 acknowledgement is expected on (default: the last one, which is
    the scripted lookup question); ``None`` means the script asked nothing that needs a
    lookup, so no acknowledgement is due and the deadline is not scored.

    ``expected_callee_turns`` is how many utterances the script spoke. When fewer turns
    come back than were spoken, the extra utterances landed inside assistant speech and
    were absorbed, so whatever the script was trying to provoke was never provoked --
    which must be reported rather than passing as a clean run.
    """
    budgets = budgets or Budgets()
    utterances, unit = load_utterances(call)
    turns = pair_turns(utterances)
    acks = find_acks(utterances, phrases)
    checks: list[Check] = []
    notes: list[str] = []

    ended = call.get("endedReason")
    if ended and ended not in ("customer-ended-call", "assistant-ended-call", "silence-timed-out"):
        notes.append(f"call endedReason={ended!r} -- the run may not be a clean measurement")
    notes.extend(duration_anomalies(call))

    # --- did the script actually exercise what it was written to exercise? -------
    if expected_callee_turns is not None:
        observed = len(turns)
        checks.append(
            Check(
                id="script_coverage",
                title="every scripted callee utterance became its own turn",
                verdict="pass" if observed >= expected_callee_turns else "fail",
                detail=(
                    f"the script spoke {expected_callee_turns} utterance(s) and Vapi "
                    f"recognised {observed} callee turn(s): the missing ones landed while "
                    "the assistant was talking and were absorbed, so no end-of-turn "
                    "detection was re-armed and THIS RUN DID NOT TEST THE STALL. That is a "
                    "gap in coverage, not a product regression -- lengthen the gaps, or "
                    "clear assistant.firstMessage so the assistant is not mid-sentence."
                    if observed < expected_callee_turns
                    else f"{observed} callee turn(s) for {expected_callee_turns} scripted"
                ),
                measured_s=float(observed),
                budget_s=float(expected_callee_turns),
                unit="count",
            )
        )

    # --- every turn got an answer at all -----------------------------------------
    unanswered = [t for t in turns if t.reply is None]
    checks.append(
        Check(
            id="replied",
            title="every callee turn was answered",
            verdict="fail" if unanswered else "pass",
            detail=(
                "assistant never spoke after: "
                + "; ".join(f"#{t.user.index} {t.user.text!r}" for t in unanswered)
                if unanswered
                else f"{len(turns)} callee turn(s), all answered"
            ),
        )
    )

    # --- R1: why we called, within the deadline ----------------------------------
    first = turns[0] if turns else None
    if first is None:
        checks.append(
            Check("r1_deadline", "R1 reason-for-calling deadline", "fail", "callee never spoke")
        )
        checks.append(Check("r1_provenance", "R1 answered by the adapter", "skip", "no R1 turn"))
    elif first.gap_s is None:
        checks.append(
            Check(
                "r1_deadline",
                "R1 reason-for-calling deadline",
                "fail",
                f"assistant never answered the callee's first utterance {first.user.text!r}",
                budget_s=budgets.reason_deadline_s,
            )
        )
        checks.append(
            Check("r1_provenance", "R1 answered by the adapter", "skip", "no reply to attribute")
        )
    else:
        gap = first.gap_s
        checks.append(
            Check(
                id="r1_deadline",
                title="R1 reason-for-calling deadline",
                verdict="pass" if gap <= budgets.reason_deadline_s else "fail",
                detail=(
                    f"callee's first utterance {first.user.text!r} ended at "
                    f"{first.user.end_s:.3f}s; assistant started at "
                    f"{first.reply.start_s:.3f}s"  # type: ignore[union-attr]
                ),
                measured_s=gap,
                budget_s=budgets.reason_deadline_s,
            )
        )
        # Provenance. This is the check that survives the adapter being switched off, so
        # it must not be fooled by transcription noise: `messages[]` records the
        # assistant's speech as recognised, not the configured string. Measured live on
        # call 01a025ee, Vapi spoke the static firstMessage and it came back with
        # "Mike Averdo" for "Mike Averto" -- one letter, and exact matching flipped the
        # most important check in this harness from FAIL to a false PASS.
        spoken = normalize_phrase(first.reply.text)  # type: ignore[union-attr]
        static = normalize_phrase(first_message or "")
        similarity = _utterance_similarity(spoken, static)
        if not first_message:
            checks.append(
                Check(
                    "r1_provenance",
                    "R1 answered by the adapter",
                    "pass",
                    "assistant has no static firstMessage, so the reply came from model.url",
                )
            )
        elif similarity >= _PROVENANCE_SIMILARITY:
            checks.append(
                Check(
                    id="r1_provenance",
                    title="R1 answered by the adapter",
                    verdict="fail" if require_adapter_provenance else "skip",
                    detail=(
                        "the R1 deadline was met by Vapi's static assistant.firstMessage, "
                        f"not by the adapter: with firstMessageMode={_mode(call)!r} Vapi "
                        "speaks that fixed string on the first callee turn and never calls "
                        "model.url. The adapter's reason-for-calling fast path is UNVERIFIED "
                        "by this run, and R1 would keep passing with the adapter switched "
                        "off. Clear assistant.firstMessage to measure the adapter."
                    ),
                    measured_s=similarity,
                    budget_s=_PROVENANCE_SIMILARITY,
                    unit="ratio",
                )
            )
        else:
            checks.append(
                Check(
                    id="r1_provenance",
                    title="R1 answered by the adapter",
                    verdict="pass",
                    detail=(
                        "the R1 reply is not the static firstMessage (similarity "
                        f"{similarity:.3f} < {_PROVENANCE_SIMILARITY}), so model.url "
                        "produced it"
                    ),
                    measured_s=similarity,
                    budget_s=_PROVENANCE_SIMILARITY,
                    unit="ratio",
                )
            )

    # --- R2: acknowledgement deadline on the lookup turn -------------------------
    if ack_turn_index is None:
        ack_turn = None
        checks.append(
            Check(
                "r2_ack_deadline",
                "R2 acknowledgement deadline",
                "skip",
                "this script asks nothing that needs a lookup, so no acknowledgement is due",
                budget_s=budgets.ack_deadline_s,
            )
        )
    elif not (bool(turns) and -len(turns) <= ack_turn_index < len(turns)):
        ack_turn = None
        checks.append(
            Check(
                "r2_ack_deadline",
                "R2 acknowledgement deadline",
                "fail",
                f"no callee turn at index {ack_turn_index} to score ({len(turns)} turn(s))",
                budget_s=budgets.ack_deadline_s,
            )
        )
    else:
        ack_turn = turns[ack_turn_index]
    if ack_turn is not None:
        after = [a for a in acks if a.start_s >= ack_turn.user.end_s]
        if not after:
            reply = ack_turn.reply
            hint = (
                f"the assistant did speak at {reply.start_s:.3f}s ({reply.text[:70]!r}) but it "
                "matched no configured acknowledgement phrase -- either no acknowledgement was "
                "emitted, or VHV_FILLER_PHRASES has drifted from the deployed adapter's pool"
                if reply is not None
                else "the assistant did not speak at all after this turn"
            )
            checks.append(
                Check(
                    id="r2_ack_deadline",
                    title="R2 acknowledgement deadline",
                    verdict="fail",
                    detail=(
                        f"no acknowledgement after the callee's turn {ack_turn.user.text!r} "
                        f"(ended {ack_turn.user.end_s:.3f}s): {hint}"
                    ),
                    budget_s=budgets.ack_deadline_s,
                )
            )
        else:
            ack = after[0]
            latency = ack.start_s - ack_turn.user.end_s
            checks.append(
                Check(
                    id="r2_ack_deadline",
                    title="R2 acknowledgement deadline",
                    verdict="pass" if latency <= budgets.ack_deadline_s else "fail",
                    detail=(
                        f"callee's turn ended at {ack_turn.user.end_s:.3f}s; "
                        f"{ack.text[:60]!r} started at {ack.start_s:.3f}s"
                    ),
                    measured_s=latency,
                    budget_s=budgets.ack_deadline_s,
                )
            )

    # --- R2: the cooldown is call-global -----------------------------------------
    if len(acks) < 2:
        checks.append(
            Check(
                id="r2_ack_cooldown",
                title="R2 acknowledgement cooldown is call-global",
                verdict="pass",
                detail=f"{len(acks)} acknowledgement(s) in the call, so no gap can be violated",
                budget_s=budgets.ack_cooldown_s,
            )
        )
    else:
        gaps = [
            (later.start_s - earlier.start_s, earlier, later)
            for earlier, later in zip(acks, acks[1:], strict=False)
        ]
        worst, earlier, later = min(gaps, key=lambda g: g[0])
        checks.append(
            Check(
                id="r2_ack_cooldown",
                title="R2 acknowledgement cooldown is call-global",
                verdict="pass" if worst >= budgets.ack_cooldown_s else "fail",
                detail=(
                    f"closest pair: {earlier.text[:40]!r} at {earlier.start_s:.3f}s then "
                    f"{later.text[:40]!r} at {later.start_s:.3f}s "
                    f"({len(acks)} acknowledgements total)"
                ),
                measured_s=worst,
                budget_s=budgets.ack_cooldown_s,
            )
        )

    # --- R2: the ack-storm failure, encoded literally ----------------------------
    burst, at = _max_in_window([a.start_s for a in acks], budgets.ack_storm_window_s)
    checks.append(
        Check(
            id="r2_ack_storm",
            title=f"no acknowledgement storm in any {budgets.ack_storm_window_s:g}s window",
            verdict="pass" if burst <= budgets.ack_storm_max else "fail",
            detail=(
                f"worst window starts at {at:.3f}s with {burst} acknowledgements"
                if at is not None
                else "no acknowledgements in the call"
            ),
            measured_s=float(burst),
            budget_s=float(budgets.ack_storm_max),
            unit="count",
        )
    )

    # --- the Flux hold-open guard ------------------------------------------------
    measurable = [t for t in turns if t.gap_s is not None]
    if not measurable:
        checks.append(
            Check(
                "max_turn_gap",
                "no turn leaves the callee waiting past the ceiling",
                "fail",
                "no turn in this call produced a measurable gap",
                budget_s=budgets.max_turn_gap_s,
            )
        )
    else:
        worst_turn = max(measurable, key=lambda t: t.gap_s or 0.0)
        worst_gap = worst_turn.gap_s or 0.0
        checks.append(
            Check(
                id="max_turn_gap",
                title="no turn leaves the callee waiting past the ceiling",
                verdict="pass" if worst_gap <= budgets.max_turn_gap_s else "fail",
                detail=(
                    f"worst turn: callee said {worst_turn.user.text[:50]!r} ending at "
                    f"{worst_turn.user.end_s:.3f}s, assistant started at "
                    f"{worst_turn.reply.start_s:.3f}s"  # type: ignore[union-attr]
                ),
                measured_s=worst_gap,
                budget_s=budgets.max_turn_gap_s,
            )
        )

    return Report(
        checks=checks,
        turns=turns,
        utterances=utterances,
        duration_unit=unit,
        acks=acks,
        notes=notes,
    )


# --- transport-side attribution --------------------------------------------------
#
# `evaluate` above scores only what Vapi recorded as spoken. That is the callee's
# experience, and it is the right thing to pass or fail on -- but on its own it cannot
# say WHERE a breach came from. The websocket transport carries `model-output` events,
# which are the adapter's text as Vapi received it, so comparing the two separates
# "the adapter was slow" from "the adapter answered and Vapi never spoke it".
#
# Measured live on call 01a025e5: the adapter emitted "Okay, bear with me a moment.
# <flush />" 1.29 s after the callee's question -- inside the 2 s budget -- and Vapi
# never turned it into audio. Scoring the spoken timeline alone reports "no
# acknowledgement" and points the next person at the adapter, which is the wrong place.


def _emitted_holding_lines(
    events: list[dict[str, Any]], phrases: list[str]
) -> list[tuple[float, str, bool]]:
    """``(at_s, text, matched_configured_pool)`` for each channel=stream holding line.

    Only ever sees channel=stream: the adapter's own logs record two delivery paths for
    an acknowledgement (``turn filler ... channel=stream|control``), but only the stream
    path puts text on the ``model.url`` SSE connection this reads. The control path
    (Vapi Live Call Control, ``POST call.monitor.controlUrl``) speaks directly and never
    touches model.url, so it leaves no event here at all -- see ``evaluate_transport``
    for how that channel is still counted.
    """
    pool = [p for p in (normalize_phrase(p) for p in phrases) if p]
    out: list[tuple[float, str, bool]] = []
    for entry in events:
        event = entry.get("event")
        if not isinstance(event, dict) or event.get("type") != "model-output":
            continue
        text = str(event.get("output") or "")
        if not carries_flush_token(text):
            continue
        norm = normalize_phrase(text)
        at = entry.get("at_s")
        out.append(
            (
                float(at) if isinstance(at, int | float) else 0.0,
                text.strip(),
                any(norm == p or norm.startswith(p) for p in pool),
            )
        )
    return out


def evaluate_transport(
    events: list[dict[str, Any]],
    *,
    callee_turn_end_s: float | None,
    spoken_ack_count: int,
    phrases: list[str],
    budgets: Budgets | None = None,
) -> tuple[list[Check], list[str]]:
    """Attribute R2 between the adapter and Vapi, from the websocket event stream.

    ``callee_turn_end_s`` is when the callee stopped talking on the lookup turn, on the
    same harness clock the events are stamped with. ``spoken_ack_count`` is how many
    acknowledgements Vapi actually spoke, from :func:`evaluate` -- Vapi's own record of
    what it said.

    ``_emitted_holding_lines`` can only ever see channel=stream (the ``model.url`` SSE
    connection). When Vapi spoke more acknowledgements than that explains, the surplus
    is reported as attribution UNKNOWN, never inferred as channel=control: several of
    the adapter's own filler phrases ("Sure, give me a second.", "Okay, bear with me a
    moment.") are ordinary enough that the model itself streaming the same words via
    model.url -- without the adapter's structural ``<flush />`` marker -- is
    indistinguishable from a genuine channel=control delivery on this transport alone.
    That ambiguity is exactly the regression PR #10's prohibition exists to catch (the
    model re-acquiring its own holding-phrase habit), so crediting the surplus to
    channel=control by inference would mask it rather than report it.
    See ``tests/e2e/README.md`` (Trap 5) for the unclaimed ``GET /debug/acks/{call_ref}``
    spec that would make this evidence-based instead of UNKNOWN -- not implemented, not
    currently claimed by anyone, and this function has no code path that depends on it.
    """
    budgets = budgets or Budgets()
    stream = _emitted_holding_lines(events, phrases)
    checks: list[Check] = []
    notes: list[str] = []

    drifted = [text for _, text, matched in stream if not matched]
    if drifted:
        notes.append(
            "the adapter emitted holding lines that are NOT in the phrase pool this run "
            f"matched against: {drifted}. The deployed VHV_FILLER_PHRASES has drifted "
            "from the pool given to this harness, so the spoken-timeline acknowledgement "
            "checks above may be scoring against the wrong list. Re-run with "
            "--ack-phrases-file taken from the deployed adapter."
        )

    stream_count = len(stream)
    # Spoken acknowledgements channel=stream cannot explain -- see the docstring above
    # for why this is reported as UNKNOWN rather than credited to channel=control.
    unattributed = max(0, spoken_ack_count - stream_count)

    if stream_count == 0 and spoken_ack_count == 0:
        checks.append(
            Check(
                "r2_ack_emitted",
                "the adapter emitted an acknowledgement",
                "fail" if callee_turn_end_s is not None else "skip",
                "no model-output carrying the <flush /> holding-line token reached the "
                "transport and nothing was spoken, so the adapter never produced an "
                "acknowledgement at all",
                budget_s=budgets.ack_deadline_s,
            )
        )
        return checks, notes

    if callee_turn_end_s is None:
        checks.append(
            Check(
                "r2_ack_emitted",
                "the adapter emitted an acknowledgement",
                "skip",
                f"{stream_count} channel=stream holding line(s) emitted"
                + (
                    f"; {unattributed} acknowledgement(s) spoken with no matching "
                    "channel=stream line (attribution UNKNOWN)"
                    if unattributed
                    else ""
                )
                + ", but no scripted callee turn to measure them from",
            )
        )
    else:
        after = [(at, text) for at, text, _ in stream if at >= callee_turn_end_s]
        if after:
            at, text = after[0]
            latency = at - callee_turn_end_s
            checks.append(
                Check(
                    "r2_ack_emitted",
                    "the adapter emitted an acknowledgement",
                    "pass" if latency <= budgets.ack_deadline_s else "fail",
                    f"{text[:60]!r} reached the transport at {at:.3f}s, "
                    f"{latency:.3f}s after the callee stopped talking [channel=stream]",
                    measured_s=latency,
                    budget_s=budgets.ack_deadline_s,
                )
            )
        elif unattributed > 0:
            # channel=stream produced nothing after the callee's turn, but Vapi spoke
            # more acknowledgements in total than channel=stream explains. This is NOT
            # evidence of channel=control -- see the docstring above -- so it is
            # reported as UNKNOWN, with neither a false "no acknowledgement" nor a
            # fabricated latency.
            checks.append(
                Check(
                    "r2_ack_emitted",
                    "the adapter emitted an acknowledgement",
                    "skip",
                    "no model-output line reached the transport after the callee's "
                    f"question ended at {callee_turn_end_s:.3f}s, but {spoken_ack_count} "
                    f"acknowledgement(s) were spoken in total against {stream_count} "
                    f"channel=stream line(s): the channel for {unattributed} of them is "
                    "UNKNOWN -- this transport cannot tell channel=control (Vapi Live "
                    "Call Control) apart from the model streaming its own "
                    "holding-phrase-shaped text via model.url without the adapter's "
                    "<flush /> marker. See r2_ack_deadline for the spoken-timeline "
                    "deadline; real channel attribution needs the adapter's own record "
                    "(GET /debug/acks/{call_ref}, not yet available to this harness).",
                    budget_s=budgets.ack_deadline_s,
                )
            )
        else:
            checks.append(
                Check(
                    "r2_ack_emitted",
                    "the adapter emitted an acknowledgement",
                    "fail",
                    f"the adapter emitted {stream_count} holding line(s) "
                    f"[channel=stream], all before the callee's question ended at "
                    f"{callee_turn_end_s:.3f}s",
                    budget_s=budgets.ack_deadline_s,
                )
            )

    # The attribution channel=stream evidence alone CAN make: a stream line with a
    # harness-clock timestamp that never became audio is a real drop. The other
    # direction (more spoken than channel=stream explains) is the ambiguity above, and
    # is never folded into this pass/fail -- it gets its own UNKNOWN note instead.
    dropped = stream_count - spoken_ack_count
    if dropped > 0:
        verdict = "fail"
        detail = (
            f"the adapter emitted {stream_count} holding line(s) via channel=stream "
            f"and Vapi spoke {spoken_ack_count}: {dropped} never became audio. The "
            "adapter met its deadline and the callee still heard silence, so this is "
            "a Vapi-side text-to-speech or turn-state fault, not an adapter latency "
            "regression."
        )
    else:
        verdict = "pass"
        detail = f"{stream_count} via channel=stream, Vapi spoke {spoken_ack_count}"
        if unattributed:
            detail += (
                f" ({unattributed} spoken with no matching channel=stream line -- "
                "attribution UNKNOWN, see r2_ack_emitted)"
            )
    checks.append(
        Check(
            "acks_reached_the_callee",
            "every emitted acknowledgement was actually spoken",
            verdict,
            detail,
            measured_s=float(max(0, dropped)),
            budget_s=0.0,
            unit="count",
        )
    )
    return checks, notes


# --- R1: transport verifiability ---------------------------------------------------
#
# The reason-for-calling fast path (server.py, gated on `chat.direction == "outbound"`)
# and the outbound-only "calling on behalf of..." framing built into the system prompt
# behind it (speech.py) both require `chat.call_type == "outboundPhoneCall"`
# (vapi_events.VapiChatRequest.direction). This harness never places a PSTN call -- see
# README -- so every call it creates is `vapi.websocketCall`, and `direction` is always
# "inbound". On this transport R1's own code paths structurally never run: whatever the
# assistant said to the callee's first utterance was an ordinary Hermes turn, not the
# reason-for-calling feature, no matter how fast or slow it was. r1_deadline above is a
# real measurement, but not of R1 -- reported here explicitly, the same disciplined way
# r1_provenance reports the firstMessage shortcut, rather than left to look like either a
# clean PASS or an adapter regression.


def r1_transport_scope(call: dict[str, Any], report: Report) -> Check:
    """Whether this call's transport could even exercise the R1 code path.

    A separate check from :func:`evaluate`'s ``r1_deadline``/``r1_provenance`` (append
    its result to ``report.checks`` after calling ``evaluate()``) so a live run can add
    it without changing what a recorded-call ``evaluate()`` reports for a call where R1
    genuinely was exercised (an ``outboundPhoneCall``).
    """
    call_type = call.get("type")
    if call_type == "outboundPhoneCall":
        return Check(
            "r1_transport_scope",
            "R1 is measurable on this transport",
            "pass",
            f"call.type={call_type!r} is an outbound phone call, so chat.direction is "
            "'outbound' and both the reason-for-calling fast path and the outbound "
            "system-prompt framing behind it are reachable: r1_deadline and "
            "r1_provenance above measure the real R1 feature.",
        )

    dominates_gap = False
    if report.turns and report.turns[0].gap_s is not None:
        measurable = [t for t in report.turns if t.gap_s is not None]
        worst = max(measurable, key=lambda t: t.gap_s or 0.0)
        dominates_gap = worst is report.turns[0]

    detail = (
        f"call.type={call_type!r} is never 'outboundPhoneCall' on the vapi.websocket "
        "transport this harness uses (it never places a PSTN call), so chat.direction "
        "is always 'inbound' and the reason-for-calling fast path (server.py, gated on "
        "direction == 'outbound') never fires -- nor does the outbound-only 'calling on "
        "behalf of...' framing speech.py builds into the system prompt off the same "
        "gate. r1_deadline above is a real measurement of an ordinary inbound Hermes "
        "turn, not of R1: it is UNVERIFIABLE-BY-THIS-TRANSPORT. Only a real "
        "outboundPhoneCall, which this harness refuses to place, can verify R1."
    )
    if dominates_gap:
        detail += (
            " max_turn_gap above is dominated by this same unverifiable turn (the "
            "callee's first utterance is its worst turn), so that verdict is not "
            "evidence of an R1 regression either -- only of ordinary turn latency."
        )
    return Check("r1_transport_scope", "R1 is measurable on this transport", "skip", detail)


# --- human-readable output -------------------------------------------------------

_GLYPH = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}


def render_table(report: Report, *, phrases: list[str] | None = None) -> str:
    """The per-turn timeline plus the verdicts, so a failure explains itself."""
    lines: list[str] = []
    lines.append("")
    lines.append(
        "per-turn timeline (seconds from call start; duration unit detected: "
        f"{report.duration_unit})"
    )
    lines.append(f"  {'role':<9} {'start':>8} {'end':>8} {'waited':>8}  text")
    lines.append(f"  {'-' * 9} {'-' * 8} {'-' * 8} {'-' * 8}  {'-' * 46}")
    # The wait belongs on the CALLEE's row: several callee turns can share one reply
    # (that is exactly what a stall looks like), and keying by reply would then show
    # only the last of them.
    waited = {id(t.user): t.gap_s for t in report.turns}
    ack_ids = {id(a) for a in report.acks}
    for u in report.utterances:
        gap = waited.get(id(u))
        mark = " <ack>" if id(u) in ack_ids else ""
        text = u.text.replace("\n", " ")
        if len(text) > 46:
            text = text[:43] + "..."
        if gap is not None:
            shown = f"{gap:.3f}"
        elif u.role == _USER:
            shown = "NO REPLY"
        else:
            shown = ""
        lines.append(f"  {u.role:<9} {u.start_s:>8.3f} {u.end_s:>8.3f} {shown:>8}  {text}{mark}")

    if phrases is not None:
        lines.append("")
        lines.append(f"acknowledgement phrase pool used for matching ({len(phrases)}):")
        for p in phrases:
            lines.append(f"  - {p}")

    lines.append("")
    lines.append("checks")
    for c in report.checks:
        measured = c.render_measured()
        budget = "" if c.budget_s is None else f" (budget {c.budget_s:g})"
        lines.append(f"  [{_GLYPH[c.verdict]}] {c.title}: {measured}{budget}")
        lines.append(f"         {c.detail}")
    for note in report.notes:
        lines.append(f"  [NOTE] {note}")
    lines.append("")
    return "\n".join(lines)
