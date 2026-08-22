"""The adapter's own record of the acknowledgements it actually emitted.

Why a record is needed at all: an acknowledgement's TEXT cannot say who wrote it.
"Okay, one moment.", "Sure, give me a second." and "Okay, bear with me a moment." are
verbatim members of :data:`config._DEFAULT_FILLER_PHRASES`, and the voice system prompt
(``speech.VOICE_SYSTEM_PROMPT``) forbids the MODEL from producing lines of exactly that
shape -- because a model-authored holding phrase is indistinguishable to the person on
the line and so defeats the call-global cooldown, which cannot govern words the adapter
never wrote. An observer downstream (Vapi's transcript, a websocket transport, the E2E
harness) therefore cannot tell an adapter acknowledgement from a model one by reading
it, and inferring one from the other is precisely how that prohibition regressing stays
invisible. This journal is the missing evidence: a line that is NOT in here is, by
definition, not ours.

A second, separate record answers the question that one raises. Since the adapter now
DELETES a holding phrase the model opens a turn with (``speech.SpokenTurn``), "not in
here" has two possible causes: the model never wrote one, or it wrote one and we
stripped it. Those are opposite facts about the model's behaviour, so the suppressions
are journalled too -- never mixed into the emissions, because a reader of ``acks`` is
asking "what did the callee hear".

It is deliberately tiny in what it holds. Only text the adapter itself chose from the
configured phrase pool, the channel it went out on, and two times -- never caller
speech, never transcript content, never a phone number, never a secret. See
``AckRecord`` for why the exported timestamp is wall clock rather than monotonic.

Memory is bounded three ways at once (entries per call, calls retained, entry age), so
a long-lived process cannot grow without limit whatever a caller does. Every eviction
is COUNTED, not silently forgotten: a consumer that cannot see that the record was
truncated would read a missing entry as "the model wrote that line", which is a false
accusation. ``dropped`` is how a reader knows to say "unknown" instead.
"""

from __future__ import annotations

import re
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

# The two shapes `call_state.call_ref` can produce: sha256(call_id)[:12] for a real
# Vapi call id, and "anon-" + token_hex(4) when no call metadata arrived at all.
# Matched (never merely trusted) before a path parameter is used as a lookup key, so
# no arbitrary caller-supplied string can reach the map or a log line.
CALL_REF_RE = re.compile(r"[0-9a-f]{12}|anon-[0-9a-f]{8}")

# Acknowledgement phrases come from `filler_phrases`, which is operator-configured and
# has no length limit of its own. Truncating here is what turns "bounded number of
# entries" into "bounded bytes": the worst case stops depending on config.
MAX_TEXT_CHARS = 200


@dataclass(frozen=True, slots=True)
class AckRecord:
    """One acknowledgement this process emitted, as it went out.

    ``at_epoch_s`` (``time.time()``) is the exported timestamp and ``at_monotonic_s``
    (``time.monotonic()``) is deliberately NOT exported, which is the opposite of what
    a latency-minded reader expects, so: the only consumer that needs a timestamp here
    is off-box (the E2E harness runs on a different machine from the adapter), and a
    monotonic reading has an arbitrary per-boot origin -- it is not a point in time
    anywhere but inside this process, so it cannot be aligned to any other clock. Wall
    clock can be, to within the two hosts' NTP skew. Monotonic is kept internally for
    exactly the job it is good at, and wall clock is bad at: measuring the AGE of an
    entry for TTL eviction, immune to the clock being stepped underneath us.

    ``elapsed_ms`` is milliseconds from this turn's arrival at the adapter -- the same
    number as the ``turn filler ... elapsed_ms=`` log line, written from the same call
    site (``turns._record_ack``) so the log and this record cannot disagree. It is the
    adapter's OWN share of the acknowledgement budget, and needs no clock alignment to
    be meaningful; it is not the interval the callee experiences, which additionally
    includes Vapi's endpointing and TTS hops (see ``config.ack_platform_overhead_seconds``).
    """

    text: str
    channel: str
    at_epoch_s: float
    elapsed_ms: int
    at_monotonic_s: float

    def as_dict(self) -> dict[str, Any]:
        """The wire form. Note the absence of ``at_monotonic_s``; see the class docstring."""
        return {
            "text": self.text,
            "channel": self.channel,
            "at_epoch_s": round(self.at_epoch_s, 3),
            "elapsed_ms": self.elapsed_ms,
        }


# Suppressions are DIAGNOSTIC, not evidence: `acks` answers "what did the callee hear",
# and that is what an attribution verdict rests on. Eight per call is plenty to see a
# model misbehaving, and a small separate cap is what guarantees a chatty model can
# never push an acknowledgement out of the record it would then be judged against.
MAX_SUPPRESSED_PER_CALL = 8


# One per turn that hands its answer to a background continuation, so this can share
# the same order of magnitude as a typical `max_entries_per_call` without being tied
# to that operator-configured value: the two bound different things (acknowledgements
# vs the answers that follow them) and must be free to move independently.
MAX_ANSWER_DELIVERIES_PER_CALL = 16


@dataclass(slots=True)
class AnswerDeliveryRecord:
    """One turn's answer-delivery outcome -- MUTATED IN PLACE as attempts land, unlike
    every other record in this module.

    Once an acknowledgement has ended the model.url response, Live Call Control is the
    only channel left to deliver the answer through (vapi_control.py), and that
    channel is measurably unreliable in bursts. Retrying it means the outcome is not
    knowable at the moment delivery starts -- only after it succeeds, is declined,
    exhausts its retry window, or is cancelled outright because the callee spoke again
    before it finished. Appending a record with ``outcome="attempted"`` BEFORE the
    first ``say`` and then updating this SAME object as the picture clears -- rather
    than only ever appending once, at the end -- means a cancellation mid-delivery
    still leaves evidence behind: "we started this, outcome unknown" instead of
    nothing at all. That is exactly the shape of hole that turned a live, cancelled
    acknowledgement into a false MODEL-AUTHORED verdict on call 01a02681; this record
    is deliberately built so the same hole cannot open here.

    No answer TEXT is ever stored here, unlike ``AckRecord``: an acknowledgement is
    one of a handful of fixed, operator-configured phrases, but the answer this record
    describes is arbitrary Hermes-generated content that may carry real call content.
    Only the outcome and its timing are kept.

    ``outcome`` is one of:

    - ``"attempted"``  -- delivery started; no terminal state was ever reached
      (typically because the turn was cancelled mid-flight -- see above).
    - ``"delivered"``  -- the answer reached Vapi's control endpoint.
    - ``"declined"``   -- Vapi rejected the request outright (a 4xx), most plausibly
      because the call had already moved past this turn.
    - ``"fallback_spoken"`` -- the real answer could not be delivered inside its
      retry window, but a short apology was, so the callee heard something.
    - ``"silent"``      -- delivery was given up on and even the apology failed:
      nothing reached the callee for this turn's answer at all.
    - ``"superseded"``  -- the callee spoke again before this delivery finished, so it
      was abandoned deliberately rather than risk answering a question that was no
      longer the current one.
    """

    outcome: str
    attempts: int
    at_epoch_s: float
    # Milliseconds from this TURN'S ARRIVAL at the adapter -- the same origin as
    # ``AckRecord.elapsed_ms``, not from when this delivery or any one attempt
    # started. Sharing a name with a different origin is how a log line and a record
    # start disagreeing, which is the one thing this journal exists not to do.
    elapsed_ms: int
    at_monotonic_s: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "attempts": self.attempts,
            "at_epoch_s": round(self.at_epoch_s, 3),
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass(frozen=True, slots=True)
class JournalSnapshot:
    """One call's record: what was spoken, what was stripped, what was lost, and what
    happened to the answers that followed.

    ``dropped`` and ``suppressed_dropped`` are counted separately and must stay that
    way. ``dropped`` is load-bearing for exactly one decision -- whether an unmatched
    spoken phrase may be called model-authored -- so a suppression eviction bumping it
    would turn every verdict on a chatty model into "inconclusive", hiding the very
    regression the suppression record exists to expose. ``answer_deliveries_dropped``
    is a third, independent counter for the same reason: an evicted answer-delivery
    record must never make the acknowledgement record look incomplete either.
    """

    acks: list[AckRecord]
    dropped: int
    suppressed: list[AckRecord]
    suppressed_dropped: int
    answer_deliveries: list[AnswerDeliveryRecord]
    answer_deliveries_dropped: int


class _Timestamped(Protocol):
    """What :class:`_Bounded` needs of a record: when it was made, on the monotonic clock.

    A read-only property rather than a bare attribute so both a frozen record
    (:class:`AckRecord`) and a mutated-in-place one (:class:`AnswerDeliveryRecord`)
    satisfy it.
    """

    @property
    def at_monotonic_s(self) -> float: ...


_RecordT = TypeVar("_RecordT", bound=_Timestamped)


class _Bounded(Generic[_RecordT]):
    """One KIND of record for one call: a capped deque and its own loss counter.

    Each kind counts its own losses, and that separation is the point rather than
    tidiness. ``acks``'s counter decides whether an unmatched spoken phrase may be
    called model-authored, so a different kind of record aging out must never be able
    to inflate it and retire that verdict.
    """

    __slots__ = ("dropped", "entries")

    def __init__(self, max_entries: int) -> None:
        self.entries: deque[_RecordT] = deque(maxlen=max_entries)
        self.dropped = 0

    def append(self, record: _RecordT) -> None:
        if len(self.entries) == self.entries.maxlen:
            # deque(maxlen=) would discard the oldest silently; a silent discard reads
            # downstream as "that never happened", so count it instead.
            self.dropped += 1
        self.entries.append(record)

    def expire(self, now: float, ttl: float) -> None:
        """Drop entries older than ``ttl``, counting each loss against THIS kind.

        ``ttl`` is a PARAMETER and not state on this object, and that is a deliberate
        constraint rather than an oversight: it is where a per-kind TTL would be
        introduced, and introducing one has a consequence three lines up the call
        stack. See :meth:`AckJournal._expire` before adding one -- and expect
        ``test_every_kind_expires_against_the_same_ttl`` to go red, which is the point.
        """
        while self.entries and now - self.entries[0].at_monotonic_s > ttl:
            self.entries.popleft()
            self.dropped += 1


class _CallAcks:
    """One call's records, how many were lost, and when it was last touched.

    ``touched_at`` (monotonic) is what keeps an EMPTY bucket alive: a call the adapter
    handled and correctly said nothing on is a real, load-bearing fact -- it is what
    lets a reader call a spoken holding phrase model-authored -- and without a bucket
    timestamp of its own such a record would be indistinguishable from never having
    heard of the call.

    ``suppressed`` is the mirror image and is deliberately a SEPARATE kind: holding
    phrases the MODEL opened a turn with, which the adapter deleted before they
    reached the caller (``speech.SpokenTurn``). It must never be merged into ``acks``,
    whose meaning downstream is "the adapter spoke this". Without it, "we stripped
    one" and "the model never wrote one" look identical from off-box, and a model that
    started producing holding phrases again would hide behind its own fix.

    ``answer_deliveries`` is a third, independent kind for the same reason: what
    happened to the answer that followed an acknowledgement is a different fact from
    what the acknowledgement itself was, and must never be able to crowd either of the
    other two out of their own caps.

    ``kinds`` exists so eviction and the emptiness test are written ONCE, over
    whatever kinds happen to exist. A fourth kind is then one attribute and one entry
    here -- not another branch in ``_expire`` that the next author has to remember,
    which is how a record gets deleted out from under a reader whose kind was
    forgotten.
    """

    __slots__ = ("acks", "answer_deliveries", "suppressed", "touched_at")

    def __init__(self, max_entries: int, *, now: float) -> None:
        self.acks: _Bounded[AckRecord] = _Bounded(max_entries)
        self.suppressed: _Bounded[AckRecord] = _Bounded(MAX_SUPPRESSED_PER_CALL)
        self.answer_deliveries: _Bounded[AnswerDeliveryRecord] = _Bounded(
            MAX_ANSWER_DELIVERIES_PER_CALL
        )
        self.touched_at = now

    @property
    def kinds(self) -> tuple[_Bounded[Any], ...]:
        return (self.acks, self.suppressed, self.answer_deliveries)

    @property
    def empty(self) -> bool:
        """True when NO kind holds anything -- the only state a bucket may be swept in.

        Every kind counts, not just ``acks``. A call on which the adapter emitted
        nothing and the model tried a holding phrase and was stripped is the single
        most interesting record this journal can hold, and testing ``acks`` alone would
        delete it the moment its own TTL came due.
        """
        return all(not kind.entries for kind in self.kinds)


class AckJournal:
    """Bounded ``call_ref -> acknowledgements emitted`` map. Not thread-safe by design.

    asyncio is single-threaded and every method here is synchronous with no await
    inside, so there is no interleaving to guard against -- the same argument
    ``CallStateRegistry`` and the ``active_turns`` counter rest on.
    """

    def __init__(self, *, max_calls: int, max_entries_per_call: int, ttl_seconds: float) -> None:
        self._max_calls = max_calls
        self._max_entries_per_call = max_entries_per_call
        self._ttl_seconds = ttl_seconds
        self._calls: OrderedDict[str, _CallAcks] = OrderedDict()

    def __len__(self) -> int:
        return len(self._calls)

    @property
    def limits(self) -> dict[str, float | int]:
        """The caps in force, for the endpoint to report alongside a record."""
        return {
            "max_calls": self._max_calls,
            "max_entries_per_call": self._max_entries_per_call,
            "max_suppressed_per_call": MAX_SUPPRESSED_PER_CALL,
            "ttl_seconds": self._ttl_seconds,
            "max_text_chars": MAX_TEXT_CHARS,
        }

    def open(self, call_ref: str) -> None:
        """Note that this process is handling ``call_ref``, whether or not it ever
        acknowledges anything on it.

        Without this, a call on which the adapter correctly stayed silent (every turn
        answered fast, or the cooldown refusing) would be indistinguishable from a call
        this process never saw -- both "no bucket". Those two facts do opposite work
        downstream: "handled it, said nothing" is what licenses a reader to call a
        spoken holding phrase model-authored, and "never saw it" must make the same
        reader fall back to unknown. Cheap by design: one dict lookup on the hot path,
        and an empty bucket costs a key.
        """
        self._touch(call_ref, time.monotonic())

    def record(self, call_ref: str, *, text: str, channel: str, elapsed_ms: int) -> None:
        """Append one emitted acknowledgement. Never raises, never blocks.

        Called from the acknowledgement's own code path, so it must cost nothing that
        could show up in the callee's latency: a few dict/deque operations and, at
        most, a sweep of `max_calls` buckets.
        """
        now = time.monotonic()
        self._touch(call_ref, now).acks.append(
            AckRecord(
                text=text[:MAX_TEXT_CHARS],
                channel=channel,
                at_epoch_s=time.time(),
                elapsed_ms=elapsed_ms,
                at_monotonic_s=now,
            )
        )

    def note_suppressed(self, call_ref: str, *, text: str, reason: str, elapsed_ms: int) -> None:
        """Append one holding phrase the MODEL wrote and the adapter deleted.

        ``reason`` records WHICH rule fired -- ``"pool"`` for a verbatim member of the
        configured acknowledgement pool, ``"grammar"`` for a close variant -- because
        the two say different things about the model: echoing our own lines back is
        one failure, inventing new ones is a worse one.

        Kept out of :meth:`record`'s deque on purpose (see ``_CallAcks``): this is
        text the callee did NOT hear, and mixing it into the record of what was spoken
        would corrupt the attribution the journal exists to support.
        """
        now = time.monotonic()
        self._touch(call_ref, now).suppressed.append(
            AckRecord(
                text=text[:MAX_TEXT_CHARS],
                channel=reason,
                at_epoch_s=time.time(),
                elapsed_ms=elapsed_ms,
                at_monotonic_s=now,
            )
        )

    def note_answer_attempt(self, call_ref: str) -> AnswerDeliveryRecord:
        """Begin one turn's answer-delivery record, before the first ``say`` attempt.

        Returns the record BY REFERENCE so the caller (``turns._deliver_answer``) can
        update it in place as attempts land -- see ``AnswerDeliveryRecord`` for why
        that matters more here than anywhere else in this journal.
        """
        now = time.monotonic()
        record = AnswerDeliveryRecord(
            outcome="attempted",
            attempts=0,
            at_epoch_s=time.time(),
            elapsed_ms=0,
            at_monotonic_s=now,
        )
        self._touch(call_ref, now).answer_deliveries.append(record)
        return record

    def snapshot(self, call_ref: str) -> JournalSnapshot | None:
        """This call's record, or None when this journal holds none.

        None and an empty ``acks`` are different answers and must stay so: None means
        this journal has no record of the call (never seen, or aged out entirely) and a
        reader must fall back to "unknown"; an empty list with ``dropped == 0`` means
        the adapter genuinely emitted no acknowledgement on a call it did handle -- see
        :meth:`open` for why that distinction is the point. The same reading applies to
        ``suppressed`` and ``answer_deliveries``: empty means the gate ran, or no
        answer ever needed a background delivery, respectively.
        """
        now = time.monotonic()
        self._expire(now)
        bucket = self._calls.get(call_ref)
        if bucket is None:
            return None
        return JournalSnapshot(
            acks=list(bucket.acks.entries),
            dropped=bucket.acks.dropped,
            suppressed=list(bucket.suppressed.entries),
            suppressed_dropped=bucket.suppressed.dropped,
            answer_deliveries=list(bucket.answer_deliveries.entries),
            answer_deliveries_dropped=bucket.answer_deliveries.dropped,
        )

    def _touch(self, call_ref: str, now: float) -> _CallAcks:
        """This call's bucket, created if needed, with the caps re-applied.

        The sweep is told to keep this call: it must not remove the bucket we are about
        to write to even when it empties it, or the entries the TTL just took would be
        forgotten along with it and the record would go on to claim it was complete.
        """
        self._expire(now, keep=call_ref)
        bucket = self._calls.get(call_ref)
        if bucket is None:
            bucket = _CallAcks(self._max_entries_per_call, now=now)
            self._calls[call_ref] = bucket
            while len(self._calls) > self._max_calls:
                self._calls.popitem(last=False)  # oldest call, LRU by last touch
        else:
            self._calls.move_to_end(call_ref)
            bucket.touched_at = now
        return bucket

    def _expire(self, now: float, *, keep: str | None = None) -> None:
        """Drop entries older than the TTL, and calls with nothing left to say.

        Sweeps every bucket rather than only the LRU end: entries expire by their own
        age, and a call touched a moment ago can still be holding an entry from the
        start of a long call. Bounded by ``max_calls`` * ``max_entries_per_call``,
        which is small by construction.

        A bucket goes only when it is BOTH empty and itself older than the TTL, so a
        call in progress that has not needed an acknowledgement yet keeps its (empty,
        and meaningful -- see :meth:`open`) record. "Empty" means EVERY kind of record
        (``_CallAcks.empty``), not just the acknowledgements: a call whose only
        remaining evidence is a suppressed model opening, or an answer delivery still
        in flight, has something to say.

        Both the sweep and that test are written over ``bucket.kinds`` rather than
        naming the deques, so adding a kind of record cannot silently leave it
        unswept, uncounted, or deleted out from under its reader.

        INVARIANT: ONE TTL GOVERNS EVERY KIND. That is what makes "a record deleted
        out from under its reader" unreachable, and the reasoning is not obvious, so:
        every write bumps ``touched_at``, so a bucket's own age and its newest entry's
        age cross the TTL together, and the state that would lose evidence -- one kind
        empty, another still holding, bucket itself stale -- cannot exist. Probed
        against a 0.05 s TTL at 0.06/0.08/0.11 s across all three write orderings
        (acks-only, acks-then-suppressed, suppressed-then-acks): no ordering produces
        it.

        Give one kind a longer TTL and that argument collapses, because the bucket
        staleness test below still compares ``touched_at`` against a single ``ttl``:
        it would have to become the LONGEST TTL across the kinds, or a long-retention
        record gets swept the moment the short-retention ones age out. Someone will
        want to try this -- an answer-delivery outcome stays useful after the
        acknowledgements it followed have aged out -- so it is guarded by a test
        rather than by this paragraph:
        ``test_every_kind_expires_against_the_same_ttl`` fails the moment a kind stops
        expiring on the shared clock.
        """
        ttl = self._ttl_seconds
        expired: list[str] = []
        for key, bucket in self._calls.items():
            for kind in bucket.kinds:
                kind.expire(now, ttl)
            if bucket.empty and now - bucket.touched_at > ttl and key != keep:
                expired.append(key)
        for key in expired:
            # Nothing left to attribute with. Forgetting `dropped` too is deliberate:
            # a reader gets None ("no record") rather than "0 entries, N lost", which
            # is the same information with more ways to misread it.
            del self._calls[key]
