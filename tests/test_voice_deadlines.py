"""R1/R2 wall-clock deadline regression tests.

The two requirements under test both regressed repeatedly because every previous
fix was verified by hand once and never pinned down as an automated, wall-clock
assertion:

    R1. On an outbound call, within 1-2 SECONDS of the callee finishing their first
        utterance, Emma states WHY she is calling. Observed failure: 10+ seconds of
        silence (live call 01a02524: the callee's "Hello?" ended at 3.33s, the final
        transcript did not land until 14.29s, and Emma spoke at 16.92s).
    R2. After the callee stops talking on ANY turn, within 2 SECONDS Emma says a
        brief acknowledgement, then says nothing like it again for at least 10
        SECONDS -- a cooldown GLOBAL to the call, not per-turn. Observed failures:
        the acknowledgement arriving 10s late, and six of them in 16 seconds during
        a barge-in retry storm.

Most of R1's assertions live in test_reason_fast_path.py (it already owns the
running HTTP app and the fake-Hermes-request inspection this needs) and most of
R2's per-turn timing assertions live in test_turns.py (it already owns direct,
precisely-timed access to ``stream_turn``). This file holds the two R2 tests that
need a genuinely shared ``CallState`` driven at specific wall-clock offsets -- the
call-global cooldown and the historical retry storm -- which is done by injecting a
fake clock local to ``call_state``'s own module namespace, never by monkeypatching
the shared ``time`` module (that would also corrupt asyncio's own event-loop clock,
see ``_FakeMonotonic`` below), and never by sleeping out the real 10s window.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

import vapi_hermes_voice.call_state as call_state_module
from test_turns import (
    _ScriptedHermes,
    delta,
    done,
    make_settings as turns_make_settings,
    make_state as turns_make_state,
)
from vapi_hermes_voice.call_state import CallState
from vapi_hermes_voice.config import Settings
from vapi_hermes_voice.turns import stream_turn


class _FakeMonotonic:
    """A drop-in replacement for the ``time`` module, exposing only ``monotonic()``.

    Swapped in via ``monkeypatch.setattr(call_state_module, "time", ...)``, which
    rebinds the *name* ``time`` inside ``call_state``'s own module namespace only.
    Patching the attribute on the real, shared ``time`` module instead (e.g.
    ``monkeypatch.setattr(time, "monotonic", ...)``) would also corrupt
    ``asyncio``'s event loop, whose default ``BaseEventLoop.time()`` is
    ``time.monotonic()`` -- freezing that clock stalls every scheduled callback in
    the process, including this test's own ``asyncio.wait``/``asyncio.timeout``
    calls, and the test simply hangs. Rebinding the name in one importing module's
    namespace leaves the shared module, and therefore asyncio, untouched.
    """

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def monotonic(self) -> float:
        return self.t


async def _abandoned_attempt(
    settings: Settings, state: CallState, *, abandon_after: float = 0.10
) -> bool:
    """Drive one ``stream_turn`` attempt against a Hermes run that never answers,
    torn down after ``abandon_after`` seconds exactly like a Vapi barge-in retry
    that cancels the stream mid-flight. Returns whether it spoke an acknowledgement.
    """
    reaping: set[asyncio.Task[None]] = set()
    acked = False
    agen = stream_turn(
        settings=settings,
        hermes=_ScriptedHermes([(5.0, delta("never arrives")), (0.0, done())]),
        state=state,
        instructions="instructions",
        history=[],
        user_input="are you there?",
        reaping=reaping,
    )
    try:
        async with asyncio.timeout(abandon_after):
            async for chunk in agen:
                if any(phrase in chunk for phrase in settings.filler_phrases):
                    acked = True
    except TimeoutError:
        pass
    finally:
        await agen.aclose()
    for task in list(reaping):
        with contextlib.suppress(Exception):
            await task
    return acked


# --- R2 GLOBAL COOLDOWN: production 10s window, three turns 1s apart -----------


async def test_r2_global_cooldown_three_turns_one_second_apart_yield_exactly_one_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three POSTs for one call.id, 1s apart, must yield exactly one acknowledgement;
    a fourth POST once the real 10s cooldown has elapsed must get a new one.

    Requirement, verbatim: "she must then NOT say that or anything like it for at
    least 10 seconds, and that cooldown is global to the call, not per-turn." This
    drives the production ``filler_min_gap_seconds`` default (10.0, asserted below
    so a config edit that quietly narrows it is also caught) against three, then
    four, separate ``stream_turn`` calls sharing one ``CallState`` -- exactly how
    ``CallStateRegistry`` hands server.py the same object on every turn of a call --
    with the call-global clock pinned to t=0, t=1, t=2, then t=10.01, instead of
    genuinely waiting ten real seconds.
    """
    default_gap = Settings.model_fields["filler_min_gap_seconds"].default
    assert default_gap == 10.0, "test pins the documented production R2 cooldown"

    fake_clock = _FakeMonotonic()
    monkeypatch.setattr(call_state_module, "time", fake_clock)

    settings = turns_make_settings(filler_after_seconds=0.02, filler_min_gap_seconds=default_gap)
    shared_state = turns_make_state(settings)

    async def one_silent_turn() -> bool:
        # Hermes stays silent past filler_after (0.02s real), then answers with
        # nothing further -- a real, tiny wait that has nothing to do with the
        # (fake, injected) 10s cooldown clock.
        reaping: set[asyncio.Task[None]] = set()
        acked = False
        async for chunk in stream_turn(
            settings=settings,
            hermes=_ScriptedHermes([(0.05, done())]),
            state=shared_state,
            instructions="instructions",
            history=[],
            user_input="hello",
            reaping=reaping,
        ):
            if any(phrase in chunk for phrase in settings.filler_phrases):
                acked = True
        for task in list(reaping):
            with contextlib.suppress(Exception):
                await task
        return acked

    fake_clock.t = 0.0
    turn1_acked = await one_silent_turn()
    fake_clock.t = 1.0
    turn2_acked = await one_silent_turn()
    fake_clock.t = 2.0
    turn3_acked = await one_silent_turn()

    total_acks = sum((turn1_acked, turn2_acked, turn3_acked))
    assert total_acks == 1, (
        f"three turns fired 1s apart produced {total_acks} acknowledgements ("
        f"{turn1_acked=}, {turn2_acked=}, {turn3_acked=}); the 10s global cooldown "
        "must silence turns 2 and 3"
    )

    fake_clock.t = 10.01  # just past the 10s window measured from turn 1's claim
    turn4_acked = await one_silent_turn()
    assert turn4_acked, "a turn starting after the 10s cooldown window must speak again"


# --- R2 RETRY STORM: the exact live offsets, one acknowledgement total ---------

# The recorded offsets (seconds, from the callee's stop of speech) of the six
# barge-in re-POSTs of a single turn observed on one live call: the pre-fix adapter
# spoke a holding line on every one of them, six acknowledgements inside 16 seconds.
RETRY_STORM_OFFSETS = (0.05, 0.22, 0.40, 0.57, 0.74, 0.91)


async def test_r2_retry_storm_six_attempts_at_the_observed_live_offsets_yield_one_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay the exact six live retry offsets on one shared CallState: one ack total.

    This is the highest-value test in the R1/R2 set: at the pre-``fix/global-ack-
    cooldown`` commit this scenario produced six acknowledgements, one per attempt,
    because the cooldown lived on a per-turn local instead of the call-global
    ``CallState``. Every attempt here shares one ``CallState`` -- as
    ``CallStateRegistry`` does for one ``call.id`` -- and each is torn down
    mid-flight (Hermes never answers inside the attempt's window), matching "five
    of those streams cancelled mid-flight" from the live report. Only the call-
    global cooldown clock is faked (see ``_FakeMonotonic``); the real dead-air
    trigger and the per-attempt abandon window are both genuinely tiny waits.
    """
    fake_clock = _FakeMonotonic()
    monkeypatch.setattr(call_state_module, "time", fake_clock)

    settings = turns_make_settings(filler_after_seconds=0.02, filler_min_gap_seconds=10.0)
    shared_state = turns_make_state(settings)

    acks = 0
    for offset in RETRY_STORM_OFFSETS:
        fake_clock.t = offset
        if await _abandoned_attempt(settings, shared_state, abandon_after=0.08):
            acks += 1

    assert acks == 1, (
        f"the six-attempt retry storm at offsets {RETRY_STORM_OFFSETS} produced "
        f"{acks} acknowledgements; the live failure was six in sixteen seconds, "
        "the requirement is exactly one"
    )


# --- R2 atomic SSE frame, production defaults (reuses test_turns.py's raw-chunk --
# --- technique from test_filler_and_flush_token_are_one_atomic_sse_frame) -------


async def test_r2_ack_and_flush_token_stay_one_atomic_sse_frame_under_production_defaults() -> (
    None
):
    """Requirement 7, at the real 0.9s/10.0s defaults, not scaled-down test values.

    ``test_turns.test_filler_and_flush_token_are_one_atomic_sse_frame`` already
    proves an acknowledgement and its ``<flush />`` token can never be split across
    two SSE writes, with scaled timing; this reuses exactly that technique --
    collecting the raw ``data: ...`` strings ``stream_turn`` yields and counting
    ``data: `` occurrences, rather than inventing a second parsing helper -- against
    the shipped production constants, so a regression that only manifests at the
    true timing is still caught here too.
    """
    default_after = Settings.model_fields["filler_after_seconds"].default
    default_gap = Settings.model_fields["filler_min_gap_seconds"].default
    settings = turns_make_settings(
        filler_after_seconds=default_after, filler_min_gap_seconds=default_gap
    )
    reaping: set[asyncio.Task[None]] = set()
    raw_chunks: list[str] = []
    async for chunk in stream_turn(
        settings=settings,
        hermes=_ScriptedHermes([(default_after + 0.15, done())]),
        state=turns_make_state(settings),
        instructions="instructions",
        history=[],
        user_input="hello",
        reaping=reaping,
    ):
        raw_chunks.append(chunk)
    for task in list(reaping):
        with contextlib.suppress(Exception):
            await task

    filler_frames = [c for c in raw_chunks if any(p in c for p in settings.filler_phrases)]
    assert len(filler_frames) == 1, f"expected exactly one acknowledgement frame, got {len(filler_frames)}"
    frame = filler_frames[0]
    assert frame.count("data: ") == 1, (
        f"the acknowledgement and its flush token were split across SSE frames: {frame!r}"
    )
    assert "<flush />" in frame
    assert any(p in frame for p in settings.filler_phrases), (
        f"no configured filler phrase appears verbatim in the frame: {frame!r}"
    )
