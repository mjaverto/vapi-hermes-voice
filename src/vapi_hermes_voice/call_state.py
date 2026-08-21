"""Per-call state: session ids and the filler picker, keyed on the Vapi call id.

Vapi sends the full conversation on every request (docs/integration-contracts.md
section 1.2), so this is the ONLY cross-request state the adapter keeps. Entries are
evicted by TTL and an LRU cap; losing one mid-call is harmless with the default
``session_retention="none"`` (a fresh random Hermes session is minted and the full
history still arrives on every turn).
"""

from __future__ import annotations

import hashlib
import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from .config import Settings
from .policy import derive_session_ids
from .speech import FillerPicker


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
    last_seen: float = field(default_factory=time.monotonic)
    # Latch: the reason for placing this call has been stated, so it is never stated
    # again. Set both when the adapter speaks it locally and when it delegates an
    # outbound opening to Hermes with a purpose-bearing nudge (which states it too).
    # Read-checked and written with no await in between, exactly like the
    # `active_turns` counter in server.py: asyncio is single-threaded, so a
    # check-then-set with no intervening await cannot interleave.
    reason_spoken: bool = False


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
