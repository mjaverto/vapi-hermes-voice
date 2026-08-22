"""Per-call state: session ids, the acknowledgement picker and its call-global
cooldown, keyed on the Vapi call id.

Vapi sends the full conversation on every request (docs/integration-contracts.md
section 1.2), so this is the ONLY cross-request state the adapter keeps. Entries are
evicted by TTL and an LRU cap; losing one mid-call is harmless with the default
``session_retention="none"`` (a fresh random Hermes session is minted and the full
history still arrives on every turn) -- the one consequence is that the
acknowledgement cooldown restarts, i.e. the callee may hear one sooner than the
configured gap. Fresh state can only ever cost an extra acknowledgement, never a
duplicated action.
"""

from __future__ import annotations

import asyncio
import hashlib
import secrets
import time
from collections import OrderedDict
from collections.abc import Collection
from dataclasses import dataclass, field

from .config import Settings
from .policy import derive_session_ids
from .speech import FillerPicker
from .speech_feedback import SpeechLedger


def call_ref(call_id: str) -> str:
    """Stable, non-reversible log reference for a call id."""
    return hashlib.sha256(call_id.encode()).hexdigest()[:12]


@dataclass
class CallState:
    """Everything the adapter remembers about one call between requests."""

    session_id: str
    session_key: str
    call_ref: str
    filler: FillerPicker
    # The second holding-phrase pool, drawn from only while a silence this call has
    # ALREADY acknowledged is still running (`claim_reassurance`). A separate picker
    # over a separate, disjoint pool (`Settings._check_pools_are_disjoint`) rather
    # than a second draw from `filler`: the wording that is right for answering a
    # callee who just stopped talking is wrong eleven seconds into their silence, and
    # keeping the pools apart is also what makes "a reassurance can never echo the
    # acknowledgement that opened this silence" true by construction rather than by
    # an `exclude` argument someone has to remember to pass.
    reassure: FillerPicker
    last_seen: float = field(default_factory=time.monotonic)
    # Latch: the reason for placing this call has been stated, so it is never stated
    # again. Set both when the adapter speaks it locally and when it delegates an
    # outbound opening to Hermes with a purpose-bearing nudge (which states it too).
    # Read-checked and written with no await in between, exactly like the
    # `active_turns` counter in server.py: asyncio is single-threaded, so a
    # check-then-set with no intervening await cannot interleave.
    reason_spoken: bool = False
    # ``time.monotonic()`` of the last acknowledgement CLAIMED on this call, or None
    # if none has been. This is the whole of the acknowledgement cooldown: it lives
    # here, not in a stream_turn local, because the requirement is a cooldown global
    # to the CALL. A turn-local timestamp is reinitialised by every HTTP POST and so
    # can only ever space out two acknowledgements inside one turn -- it does
    # nothing to stop turn N+1 speaking one three seconds after turn N did, which is
    # exactly the repetition the callee complained about.
    last_ack_at: float | None = None
    # The background task (if any) still trying to deliver a PREVIOUS turn's answer
    # through Live Call Control (turns._deliver_answer). Lives here, not in a
    # stream_turn local, for the same reason as `last_ack_at`: the decision to
    # abandon it belongs to the NEXT turn on this call, which is a different HTTP
    # request with no other handle on it. A turn that starts while this is still
    # running cancels it first -- see `stream_turn` -- because delivering a stale
    # answer once the callee has already moved on to a new question would be worse
    # than not delivering it at all.
    pending_answer_task: asyncio.Task[None] | None = None
    # Everything this call has handed Vapi to speak, and whether any of it became
    # AUDIO (speech_feedback.py). Per-call because both feedback channels are
    # per-call: Vapi's committed conversation history arrives on this call's next
    # request, and a `speech-update` webhook names this call's id. Replaced wholesale
    # is never correct -- the ledger IS the memory that makes a drop detectable, so
    # losing it (TTL, LRU) can only ever mean "we no longer know", which the journal
    # reports as `unconfirmed` rather than guessing.
    speech: SpeechLedger = field(default_factory=SpeechLedger)
    # `call.monitor.controlUrl` from the most recent request on this call. Held here
    # so the speech-feedback webhook -- which is a different HTTP request, with no
    # turn behind it -- can re-deliver a confirmed-dropped answer. Vapi supplies it on
    # every Custom LLM request with no assistant config change (vapi_control.py).
    control_url: str | None = None
    # The turn input of the PREVIOUS request on this call. This is the liveness proof
    # that licenses a drop verdict: a delivery may only be condemned once this
    # adapter can see, in the history Vapi has just sent, the callee utterance from
    # the turn during which that delivery was made. Without it a truncated or
    # differently-shaped history would read as "nothing we ever said was spoken" and
    # condemn every delivery on the call at once. Never logged: it is callee speech.
    prior_user_input: str | None = None
    # Set by the OPTIONAL speech-update(role="user") webhook (VHV_VAPI_SERVER_SECRET):
    # the callee is talking RIGHT NOW, as of Vapi's own transcriber/VAD, not a guess
    # from this process's own clock. `_deliver_answer` checks this before every `say`
    # attempt and WAITS rather than speaks while it is True -- see `set_caller_speaking`
    # for why that is a hold, not a cancellation. Bracketed by "started"/"stopped", so
    # it self-clears; a lost "stopped" event degrades to today's unmitigated behaviour
    # once `control_answer_max_wait_seconds` is spent, never to permanent silence.
    caller_speaking: bool = False

    def supersede_pending_answer(self) -> None:
        """Cancel and forget any answer delivery still running from an earlier turn.

        Cheap when there is nothing to do (the common case: most turns finish their
        own answer delivery, or never needed one), and safe to call unconditionally
        at the top of every turn -- cancelling an already-finished task is a no-op.
        """
        task = self.pending_answer_task
        if task is not None and not task.done():
            task.cancel()
        self.pending_answer_task = None

    def set_caller_speaking(self, speaking: bool) -> None:
        """Record a callee speech-update: they just started or just stopped talking.

        Deliberately NOT `supersede_pending_answer`. That call is right when Vapi's
        OWN endpointing has already decided a new turn began -- strong evidence the
        callee asked something new. A speech-update fires on far weaker evidence: a
        stray "hello?", "um", or background noise all set this exactly the way a real
        continuation does, and Hermes may already have spent real seconds -- a live
        incident measured 24s -- computing the answer that is now held. Cancelling on
        every such blip would discard that work and make the callee wait through an
        entire fresh turn for what may have been nothing; holding costs at most a
        brief delay on an answer that still gets spoken once the flag clears, or that
        `supersede_pending_answer` cancels properly once the callee's real utterance
        completes and a genuine next turn arrives. Holding is recoverable both ways;
        cancelling here is not.
        """
        self.caller_speaking = speaking

    def claim_acknowledgement(
        self, *, min_gap_seconds: float, exclude: Collection[str] = ()
    ) -> str | None:
        """Claim the call-global acknowledgement slot; return a phrase, or None.

        Returns None when an acknowledgement was claimed less than
        ``min_gap_seconds`` ago -- on this call, whichever turn or request claimed
        it. On success the anchor is stamped BEFORE the phrase is handed back, so
        the slot is already spent by the time the caller can suspend to write it:
        Vapi barge-in has been observed re-POSTing one turn six times in sixteen
        seconds with five of those streams cancelled mid-flight, and all six share
        this object. Stamping on claim rather than on confirmed delivery makes a
        torn-down stream count as spoken, which fails toward silence instead of
        toward six "okay, let me check"s in a row.

        asyncio is single-threaded and there is no await between the check and the
        stamp, so concurrent claims cannot interleave (the same argument
        ``CallStateRegistry`` relies on).
        """
        return self._claim(self.filler, min_gap_seconds=min_gap_seconds, exclude=exclude)

    def claim_reassurance(self, *, min_gap_seconds: float) -> str | None:
        """Claim the SAME call-global slot, for a phrase from the reassurance pool.

        Same slot, deliberately, and that is the whole substance of this method: the
        requirement is that the callee hears no second holding phrase inside
        ``min_gap_seconds``, and the callee cannot tell which pool a phrase was drawn
        from. Two anchors would be two independent budgets whose audible spacing was
        whatever their interleaving happened to produce, which is not a cooldown.

        No ``exclude``: the pools are disjoint (``Settings._check_pools_are_disjoint``)
        so a reassurance can never echo the acknowledgement that opened this silence,
        and ``FillerPicker`` already refuses to repeat its own previous pick, so a
        reassurance can never echo the one before it either.
        """
        return self._claim(self.reassure, min_gap_seconds=min_gap_seconds)

    def _claim(
        self,
        picker: FillerPicker,
        *,
        min_gap_seconds: float,
        exclude: Collection[str] = (),
    ) -> str | None:
        """The cooldown itself: one check, one stamp, one place.

        Both public claims route through here so ``last_ack_at`` is written from a
        single statement. A second copy of these four lines is how a cooldown that is
        global to the call quietly becomes global to a pool.
        """
        now = time.monotonic()
        if self.last_ack_at is not None and now - self.last_ack_at < min_gap_seconds:
            return None
        self.last_ack_at = now
        return picker.pick(exclude=exclude)


def _new_state(call_id: str | None, settings: Settings) -> CallState:
    if call_id is not None and settings.session_retention == "hermes":
        session_id, session_key = derive_session_ids(call_id)
    else:
        # "none" (default): per-call random ids; nothing links the Hermes session to
        # the call or the caller. Also the fallback when no call id arrived at all.
        session_id = f"vhv-{secrets.token_hex(12)}"
        session_key = f"vhv-key-{secrets.token_hex(12)}"
    ref = call_ref(call_id) if call_id is not None else "anon-" + secrets.token_hex(4)
    return CallState(
        session_id=session_id,
        session_key=session_key,
        call_ref=ref,
        filler=FillerPicker(settings.filler_phrases),
        reassure=FillerPicker(settings.reassure_phrases),
        speech=SpeechLedger(max_replays=settings.speech_drop_max_replays_per_call),
    )


class CallStateRegistry:
    """TTL + LRU map of ``call.id`` -> :class:`CallState`.

    asyncio is single-threaded and every method is synchronous (no await between
    check and mutate), so no locking is needed.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._states: OrderedDict[str, CallState] = OrderedDict()

    def __len__(self) -> int:
        return len(self._states)

    def get_or_create(self, call_id: str | None) -> CallState:
        """The state for ``call_id``; a fresh throwaway state when no id arrived."""
        self._evict()
        if call_id is None:
            # No call metadata (metadataSendMode "off" or a bare curl): no continuity
            # to preserve, so the state is not registered.
            return _new_state(None, self._settings)
        state = self._states.get(call_id)
        if state is None:
            state = _new_state(call_id, self._settings)
            self._states[call_id] = state
        else:
            self._states.move_to_end(call_id)
        state.last_seen = time.monotonic()
        return state

    def peek(self, call_id: str) -> CallState | None:
        """The state for ``call_id`` if this process is tracking it, else None.

        Deliberately does NOT create one, unlike :meth:`get_or_create`. Its caller is
        the speech-feedback webhook, which is driven by Vapi rather than by a turn: a
        request that arrives for a call this process never handled (or has since
        evicted) must be a no-op, not a way to mint call state. Otherwise an
        authenticated but misdirected event stream could fill the registry with calls
        that have no turns behind them and evict the ones that do.

        Does not bump ``last_seen`` either. Liveness is what the TTL measures, and a
        webhook is not evidence that the adapter is still driving this call -- only a
        turn is.
        """
        return self._states.get(call_id)

    def _evict(self) -> None:
        now = time.monotonic()
        ttl = self._settings.call_state_ttl_seconds
        while self._states:
            oldest_key = next(iter(self._states))
            if now - self._states[oldest_key].last_seen > ttl:
                del self._states[oldest_key]
                continue
            break
        while len(self._states) >= self._settings.max_tracked_calls:
            self._states.popitem(last=False)
