"""Did what this adapter delivered actually become audio?

Until this module existed the adapter had no way to answer that. It saw a 200 from the
Live Call Control POST, or bytes accepted on the model.url SSE stream, and assumed
success. Vapi's own server-side log proves both can be accepted and then dropped with
nothing surfaced anywhere: on call ``01a026d8-ba00-744f-ae52-5de7e833cae6``,
``pipeline.sayQueuePush`` -> ``pipeline.botSpeechStarted`` ran 6.755 s, 5.759 s,
DROPPED, 3.066 s, 0.487 s, 9.610 s, DROPPED, while
``assistant.voice.connectionOpened`` kept reporting "vapi TTS WebSocket connected" and
no error event was written at any point. On a real call the callee simply hears
nothing.

FEEDBACK CHANNELS -- what Vapi is actually willing to tell a Custom LLM server. Both
established empirically on websocket-transport calls (no PSTN leg, nothing rang):

1. THE CONVERSATION HISTORY VAPI ALREADY SENDS US. No assistant config change; on by
   construction. The ``messages[]`` of every Custom LLM request is derived from what
   was ACTUALLY SPOKEN, not from what the adapter emitted. On call
   ``01a02723-3877-7dd8-a7da-59e56c42a744`` a streamed chunk was accepted (Vapi logged
   ``Voice input`` and fired ``assistant.speechStarted`` for it) and then cleared
   before any audio played: it is absent from the next turn's ``messages[]`` and from
   the call artifact, while everything audible -- including text delivered through
   Live Call Control -- is present. This is the load-bearing channel and it stands
   alone.

   Two fidelity caveats, both real and both handled below. The text comes back
   RE-TRANSCRIBED, not echoed: ``"ACK ONE PLEASE HOLD."`` was returned as
   ``"a c k one please hold"`` (case and punctuation gone, the acronym spelled out),
   and digits become words. And several deliveries MERGE into one assistant message
   -- call ``01a02727-41c8-7aa3-bc58-0115495583f1`` returned one entry reading
   ``"a c k one please hold Answer alpha is fifty milligrams."`` for a streamed
   acknowledgement plus a control-delivered answer. So matching is normalised and
   fractional (see :func:`spoken_coverage`) and is done against the CONCATENATION of
   the assistant messages, never per message.

2. ``speech-update`` WEBHOOKS. Needs one additive field on the assistant
   (``server: {url, secret}``) and is therefore OPTIONAL -- a strict upgrade, never a
   dependency. ``{"type": "speech-update", "status": "started", "role": "assistant",
   "turn": N}`` reaches a configured server URL 50-250 ms after audio really begins,
   and it arrives WITHOUT editing ``serverMessages`` (verified on call
   ``01a0272a-c3fc-744c-9c87-26d58bee0a40``, whose overrides set only
   ``server``). It is used here for one thing only: confirming that something was
   SPOKEN, sooner than the next turn would. It never produces a drop verdict -- see
   the next paragraph.

   ``assistant.speechStarted`` carries the exact text and would be better, but it is
   opt-in AND it did not fire at all for either Live Call Control ``say`` on call
   ``01a02727`` -- so the adapter's own answer channel gets no text-bearing event and
   the docs' ``source: "force-say"`` is not what this account emits. It is honoured
   when present, as strong per-utterance evidence, and nothing depends on it.

WHY NO TIMEOUT EVER MEANS "DROPPED". "No audio yet" is not evidence of a drop; it is
evidence of unknown. Measured: holding the model.url stream open for 20 s after a
flushed chunk DELAYED that chunk's render by 20.3 s (it played the instant the response
ended) rather than losing it -- call ``01a02723``, ``Voice input`` at t+0.6 s,
``pipeline.botSpeechStarted`` at t+20.9 s. A guard that called that a drop and re-spoke
it would have double-spoken on a live call, which is worse than the defect it is
fixing. So the ONLY thing that can move a delivery to ``dropped`` is POSITIVE ABSENCE
from a settled record: the ``messages[]`` of a LATER turn, which is a completed fact
about a completed turn, does not contain it.

``confirm_window_seconds`` is a PRECONDITION on that adjudication, never a trigger.
Absence is only allowed to mean "dropped" once enough time has passed that a render
which was going to happen would already have happened and been committed. Nothing
expires into ``dropped``; a delivery with no evidence either way stays ``unconfirmed``
for the life of the call, and is reported as exactly that.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

__all__ = [
    "Delivery",
    "DeliveryKind",
    "DeliveryState",
    "SpeechLedger",
    "SpeechOutcomeSink",
    "concat_assistant_text",
    "content_tokens",
    "spoken_coverage",
    "user_text",
]

DeliveryKind = Literal["ack", "answer"]

# `unconfirmed` -> `spoken` | `dropped` -> `replayed`.
#
# `spoken` is ABSORBING and is deliberately reachable on weaker evidence than
# `dropped`: over-claiming "spoken" only ever suppresses a recovery we would have
# liked to make, while over-claiming "dropped" puts a second utterance on a live call.
# The two errors are not symmetric, so the evidence bars are not either.
DeliveryState = Literal["unconfirmed", "spoken", "dropped", "replayed"]


class SpeechOutcomeSink(Protocol):
    """The journal record one :class:`Delivery` refines as evidence arrives.

    Declared structurally, here rather than in ``ack_journal``, so this module owns
    every write to it. That is the point: the ledger's state and the journal's record
    are updated from ONE place per transition, so they cannot come to disagree -- the
    same discipline ``turns._record_ack`` uses to keep the log and the journal from
    contradicting each other. A record that can disagree with the thing it describes
    is worse than no record, because the next reader has to work out which one lied.
    """

    outcome: str
    evidence: str
    settled_after_ms: int | None


_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Words that carry no evidence about WHICH utterance was spoken. Removing them is what
# stops "of the a to" scoring a match between two unrelated sentences. Kept small and
# closed on purpose: a long list starts deleting the short, distinctive words that
# acknowledgement phrases are almost entirely made of.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "then",
        "there",
        "they",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
    }
)


def content_tokens(text: str) -> tuple[str, ...]:
    """Lowercase alphanumeric tokens of ``text``, minus stopwords, order preserved.

    Everything Vapi's re-transcription destroys is destroyed here too -- case,
    punctuation, whitespace -- so the two sides are compared on what survives the
    round trip rather than on what was sent.

    Falls back to the un-filtered tokens when stopword removal would leave nothing:
    a delivery consisting only of stopwords must still be matchable rather than
    silently unjudgeable.
    """
    raw = _TOKEN_RE.findall(text.lower())
    kept = tuple(token for token in raw if token not in _STOPWORDS)
    return kept or tuple(raw)


def spoken_coverage(delivered: str, heard: str) -> float:
    """Fraction of ``delivered``'s content tokens discernible in ``heard`` (0.0-1.0).

    A FRACTION, not a similarity ratio, because the two known distortions are
    substitutions of individual words -- an acronym spelled out, a digit read as a
    word -- and a fraction degrades gracefully under those while a whole-string ratio
    collapses. Measured on the probe calls: ``"ANSWER ALPHA IS FIFTY MILLIGRAMS."``
    against its own re-transcription scores 1.00, and ``"ACK ONE PLEASE HOLD."``
    against ``"a c k one please hold"`` scores 0.75 -- the acronym is the only
    casualty.

    Duplicate tokens are counted once. ``heard`` is the concatenation of every
    assistant message in the record, because Vapi merges consecutive deliveries into
    one message and a per-message comparison would then miss the second one.
    """
    wanted = set(content_tokens(delivered))
    if not wanted:
        # Nothing to look for. Reported as fully covered rather than as absent: this
        # is the "no evidence either way" case, and the safe reading of no evidence is
        # never "dropped".
        return 1.0
    seen = set(content_tokens(heard))
    return len(wanted & seen) / len(wanted)


@dataclass(slots=True)
class Delivery:
    """One thing the adapter handed Vapi to speak, and what became of it.

    ``text`` is kept because it is the only way to recognise the delivery in a
    re-transcribed record -- and, for an answer, it is also what a replay would say.
    It never leaves the process except as a replay: the journal records the outcome
    and, for acknowledgements only, the phrase (which is already an operator-configured
    pool member the journal publishes anyway).
    """

    seq: int
    kind: DeliveryKind
    text: str
    at_monotonic_s: float
    state: DeliveryState = "unconfirmed"
    # WHICH observation settled it, so a reader never has to guess how confident to be.
    # "" while unconfirmed.
    evidence: str = ""
    # Set once a replay has been ISSUED for this delivery, and never cleared. The
    # at-most-once guarantee rests on this flag, not on the state, because a replay
    # that fails still counts as attempted -- retrying a failed replay is how one
    # delivery turns into three utterances.
    replay_issued: bool = False
    # The journal record this delivery refines in place, or None when the journal is
    # disabled. Never read by this module -- only written -- so a disabled journal
    # changes nothing about detection or recovery.
    record: SpeechOutcomeSink | None = None

    def _settle(self, state: DeliveryState, evidence: str, *, now: float) -> None:
        """The ONE place a delivery's verdict changes, ledger and journal together."""
        self.state = state
        self.evidence = evidence
        if self.record is not None:
            self.record.outcome = _JOURNAL_OUTCOMES[state]
            self.record.evidence = evidence
            self.record.settled_after_ms = int((now - self.at_monotonic_s) * 1000)


# The ledger's own vocabulary is internal; the journal's is a published wire contract
# read off-box by the E2E harness. Mapped rather than shared so renaming one cannot
# silently change the other.
_JOURNAL_OUTCOMES: dict[str, str] = {
    "unconfirmed": "unconfirmed",
    "spoken": "confirmed_spoken",
    "dropped": "confirmed_dropped",
    "replayed": "replayed",
}


@dataclass(slots=True)
class SpeechLedger:
    """Every delivery on ONE call, and the adjudication rules over them.

    Not thread-safe by design and it does not need to be: asyncio is single-threaded,
    every method here is synchronous with no ``await`` inside, so a check-then-set
    cannot interleave. That is the same argument ``CallStateRegistry``, ``AckJournal``
    and the ``active_turns`` counter already rest on.

    Bounded: at most ``max_deliveries`` are retained, oldest first. Evictions are
    COUNTED (``forgotten``) rather than silently dropped, because "we have no record
    of that delivery" and "that delivery was never made" are opposite facts and a
    reader that cannot tell them apart will pick the wrong one (see
    docs/integration-contracts.md section 6).
    """

    max_deliveries: int = 16
    max_replays: int = 2
    deliveries: list[Delivery] = field(default_factory=list)
    forgotten: int = 0
    replays_issued: int = 0
    _next_seq: int = 1

    def register(
        self,
        *,
        kind: DeliveryKind,
        text: str,
        record: SpeechOutcomeSink | None = None,
        now: float | None = None,
    ) -> Delivery:
        """Record that ``text`` has been handed to Vapi. Call this BEFORE it can be
        confirmed, cancelled or lost -- the record's whole job is to exist while the
        outcome is still unknown.
        """
        delivery = Delivery(
            seq=self._next_seq,
            kind=kind,
            text=text,
            at_monotonic_s=time.monotonic() if now is None else now,
            record=record,
        )
        self._next_seq += 1
        self.deliveries.append(delivery)
        while len(self.deliveries) > self.max_deliveries:
            del self.deliveries[0]
            self.forgotten += 1
        return delivery

    def outstanding(self) -> list[Delivery]:
        """Deliveries with no verdict yet."""
        return [d for d in self.deliveries if d.state == "unconfirmed"]

    # --- confirming that something WAS spoken ---------------------------------

    def confirm_spoken(
        self, delivery: Delivery, *, evidence: str, now: float | None = None
    ) -> None:
        """Mark one delivery spoken. Absorbing: nothing can undo it.

        Called from both feedback channels. It is the safe direction, so it is allowed
        to act on the weaker of them: a ``speech-update`` says "assistant audio
        started" without saying whose text, and crediting the wrong outstanding
        delivery costs at most a recovery we would have liked to make. The reverse
        mistake costs an extra utterance on a live call.
        """
        if delivery.state == "unconfirmed":
            delivery._settle("spoken", evidence, now=time.monotonic() if now is None else now)

    def confirm_any_started(self, *, before: float, evidence: str) -> list[Delivery]:
        """Credit every still-unconfirmed delivery made before ``before`` as spoken.

        This is the ``speech-update`` path, where Vapi tells us assistant audio began
        but not which text it was -- the only signal the Live Call Control ``say``
        channel produces at all (``assistant.speechStarted`` did not fire for either
        ``say`` on probe call ``01a02727``). Only deliveries OLDER than the event are
        eligible, so an event can never vouch for something delivered after it.
        """
        settled = [d for d in self.outstanding() if d.at_monotonic_s <= before]
        for delivery in settled:
            self.confirm_spoken(delivery, evidence=evidence)
        return settled

    def confirm_by_text(
        self, spoken_text: str, *, threshold: float, evidence: str
    ) -> list[Delivery]:
        """Credit deliveries whose text is discernible in ``spoken_text``.

        The ``assistant.speechStarted`` path: strong, per-utterance evidence, used
        when the assistant config opts into it. Nothing depends on it.
        """
        settled = [
            d for d in self.outstanding() if spoken_coverage(d.text, spoken_text) >= threshold
        ]
        for delivery in settled:
            self.confirm_spoken(delivery, evidence=evidence)
        return settled

    # --- the only path to a DROP verdict --------------------------------------

    def reconcile_history(
        self,
        *,
        heard: str,
        threshold: float,
        settled_before: float,
        history_advanced: bool,
    ) -> tuple[list[Delivery], list[Delivery]]:
        """Adjudicate outstanding deliveries against Vapi's own record of what was said.

        ``heard`` is the concatenation of every assistant message in the inbound
        ``messages[]`` -- Vapi's committed account of what the callee actually heard.

        Returns ``(spoken, dropped)``.

        Three interlocks, and all three must hold before an absence may be called a
        drop. Each one exists because without it a specific, observed situation reads
        as a drop when it is not:

        - ``settled_before``: the delivery must be older than this instant, i.e. older
          than ``confirm_window_seconds``. Without it, an utterance still sitting in
          ``pipeline.sayQueuePush`` (measured at up to 9.610 s on call ``01a026d8``,
          and up to 20.3 s when a stream was held open) reads as absent and gets
          re-spoken on top of itself.
        - ``history_advanced``: the inbound record must demonstrably cover the turn
          during which the delivery was made -- proven by the caller finding the
          PREVIOUS turn's own user input in this request's history. Without it, a
          record that is empty, truncated, or shaped differently than assumed
          (``metadataSendMode``, ``max_history_messages``) reads as "nothing we said
          was ever spoken" and every delivery on the call is condemned at once.
        - a positive absence, not a missing signal: the delivery scores below
          ``threshold`` against text Vapi says it DID speak. A record with no
          assistant text at all cannot clear ``history_advanced`` in practice, but
          this is also why the verdict is per-delivery rather than per-call.
        """
        spoken: list[Delivery] = []
        dropped: list[Delivery] = []
        now = time.monotonic()
        for delivery in self.outstanding():
            if spoken_coverage(delivery.text, heard) >= threshold:
                self.confirm_spoken(delivery, evidence="history", now=now)
                spoken.append(delivery)
                continue
            if not history_advanced or delivery.at_monotonic_s > settled_before:
                continue  # stays unconfirmed: absence is not yet evidence
            delivery._settle("dropped", "history_absent", now=now)
            dropped.append(delivery)
        return spoken, dropped

    # --- recovery -------------------------------------------------------------

    def claim_replay(self, *, max_age_seconds: float, now: float | None = None) -> Delivery | None:
        """Claim the one delivery to re-speak now, or None. AT MOST ONCE, EVER.

        This is the whole of the double-speaking guard, and every clause of it is
        load-bearing:

        - Only ``kind == "answer"``. An acknowledgement is a promise about the
          immediate future ("give me a second"), so a stale one is not merely
          redundant, it is FALSE -- re-speaking "give me a second" after the answer
          has already been delivered would be worse than the silence it was meant to
          cover. An answer is still true whenever it arrives, which is exactly why
          re-speaking one is safe and re-speaking the other is not. Do not relax this.
        - Only from ``dropped``, which only :meth:`reconcile_history` can produce, and
          only on positive absence from a settled record. No timeout reaches here.
        - ``replay_issued`` is set in the SAME synchronous block as the check, with no
          ``await`` between them, so two concurrent callers cannot both claim one
          delivery. Set before the replay is attempted, not after: a failed replay
          must not be retried, because that is how one delivery becomes three
          utterances.
        - ``max_replays`` per call bounds the damage if the matcher is ever wrong in
          the dangerous direction for structural reasons -- a Vapi change that stops
          committing assistant messages would otherwise condemn every delivery on
          every call.
        - ``max_age_seconds`` retires an answer that is too old to be worth speaking.
          It is a freshness limit on the recovery, not a trigger for it.
        """
        if self.replays_issued >= self.max_replays:
            return None
        moment = time.monotonic() if now is None else now
        for delivery in self.deliveries:
            if delivery.kind != "answer" or delivery.state != "dropped":
                continue
            if delivery.replay_issued:
                continue
            if moment - delivery.at_monotonic_s > max_age_seconds:
                continue
            delivery.replay_issued = True  # no await since the checks above: atomic
            self.replays_issued += 1
            return delivery
        return None

    def mark_replayed(self, delivery: Delivery) -> None:
        """Note that the replay for ``delivery`` reached Vapi.

        A new :class:`Delivery` is registered for the replay itself by the caller, so
        the replayed text is subject to exactly the same confirmation rules as any
        other delivery -- including never being replayed a second time, since that new
        record starts at ``unconfirmed`` and this one can never leave ``replayed``.
        """
        delivery._settle("replayed", "replayed", now=time.monotonic())


def concat_assistant_text(messages: Iterable[tuple[str, str | None]]) -> str:
    """Join the content of every assistant message in ``(role, content)`` pairs.

    Separate from the ledger so the ledger stays free of Vapi wire shapes, and so the
    merging caveat is handled in exactly one place: Vapi returns consecutive
    deliveries as a single assistant message, so what the ledger compares against is
    one string, not a list.
    """
    return " ".join(content for role, content in messages if role == "assistant" and content)


def user_text(messages: Sequence[tuple[str, str | None]]) -> str:
    """Join the content of every user message -- the liveness evidence.

    A delivery may only be condemned once this adapter can see, in the record Vapi
    just sent, the callee utterance from the turn during which that delivery was
    made. This is what that check reads.
    """
    return " ".join(content for role, content in messages if role == "user" and content)
