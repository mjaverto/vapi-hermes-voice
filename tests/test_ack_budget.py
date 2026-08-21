"""R2 budget regression tests: the callee hears an acknowledgement inside two seconds
of finishing a sentence, INCLUDING when the Live Call Control POST that delivers it
never answers.

That deadline is measured on the CALLEE'S clock -- their microphone going quiet to
their speaker making a sound -- and only part of it belongs to this process. Measured
live on call 01a0262b, on the callee's own audio devices, independent of Vapi's
timeline:

    callee stopped talking ...................... 15.516 s
    callee heard the first reply ................ 17.837 s  -> heard gap 2.321 s
    adapter emitted its ack ..................... 1.130 s after the turn arrived here

so 2.321 - 1.130 = 1.191 s went on things outside this repo: Vapi's transcriber
endpointing (Deepgram Flux eotThreshold/eotTimeoutMs) plus
startSpeakingPlan.waitSeconds before the request is delivered, and the TTS/transport
hop after the acknowledgement is handed back. Nothing here can shrink that, so these
tests charge the adapter only for its own share of the two seconds -- and hold it to
that share, rather than to a 2 s figure it could always appear to meet by pretending
the platform is free.

The live failure pinned down here (adapter journal, same call, verbatim):

    17:13:15,133 WARNING vapi_control  msg=ack control request failed error=ReadTimeout
    17:13:15,133 INFO    turns         msg=turn filler elapsed_ms=3902 channel=stream

One slow control POST and the holding line landed at 3.902 s: 0.9 s of
``filler_after_seconds`` plus a control timeout of 3.0 s that was larger than the
entire budget it was spending -- on a request the caller only ever waits for so that
it can decide to give up and use the SSE fallback instead.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any, ClassVar

import httpx

from test_ack_control import CONTROL_URL
from test_turns import _parse_chunk, _ScriptedHermes, done, make_settings, make_state
from vapi_hermes_voice.config import Settings
from vapi_hermes_voice.turns import stream_turn
from vapi_hermes_voice.vapi_control import VapiControlClient

# The requirement: the callee hears something within two seconds of stopping talking.
R2_BUDGET_SECONDS = 2.0
# Measured (see module docstring): the 2.321 s heard gap minus the 1.130 s the
# adapter's own journal accounts for. Endpointing ahead of this process, TTS after it.
PLATFORM_OVERHEAD_SECONDS = 1.191
# What is therefore left for the adapter, end to end, from the turn arriving on the
# socket to the acknowledgement being handed back out: 0.809 s.
ADAPTER_SHARE_SECONDS = R2_BUDGET_SECONDS - PLATFORM_OVERHEAD_SECONDS
ACK_PHRASE = "Okay, let me check."


class SilentControlEndpoint(httpx.AsyncBaseTransport):
    """A control endpoint that accepts the request and then goes quiet, honouring the
    per-phase budget httpx hands it exactly as a real socket would.

    Deliberately NOT ``httpx.MockTransport``: that double enforces no timeouts at all,
    so a test written against it can only ever observe an unbounded hang and never the
    behaviour of the timeouts under discussion here.

    ``slow_phases`` consume their entire share and then SUCCEED, which is the part
    that surprises people: no exception is raised, and the request carries on into the
    next phase with a fresh full share. Only ``stall_phase`` runs out and raises. A
    caller who passed a bare float therefore waits for the SUM of the phases rather
    than for the float -- the arithmetic the acknowledgement budget has to survive.
    """

    _PHASE_ERRORS: ClassVar[dict[str, type[httpx.TimeoutException]]] = {
        "connect": httpx.ConnectTimeout,
        "write": httpx.WriteTimeout,
        "read": httpx.ReadTimeout,
        "pool": httpx.PoolTimeout,
    }

    def __init__(self, *, stall_phase: str = "read", slow_phases: tuple[str, ...] = ()) -> None:
        self.stall_phase = stall_phase
        self.slow_phases = slow_phases
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        budget: dict[str, float | None] = request.extensions["timeout"]
        for phase in (*self.slow_phases, self.stall_phase):
            share = budget.get(phase)
            if share is None:
                raise AssertionError(
                    f"httpx was given no {phase} budget: this request is unbounded"
                )
            await asyncio.sleep(share)
        raise self._PHASE_ERRORS[self.stall_phase](
            f"control endpoint stalled in {self.stall_phase}", request=request
        )


def production_settings(**overrides: Any) -> Settings:
    """``make_settings`` with the DEPLOYED dead-air default, not a fast test one.

    Every other test in this suite shrinks ``filler_after_seconds`` to keep itself
    quick; that is exactly wrong here, because the question these tests ask is whether
    the numbers we actually ship add up. Only the phrase list and the cooldown are
    pinned, so the acknowledgement text is predictable and no leftover cooldown can
    suppress the one line being measured.
    """
    values: dict[str, Any] = {
        "filler_after_seconds": Settings.model_fields["filler_after_seconds"].default,
        "filler_min_gap_seconds": 10.0,
        "filler_phrases": [ACK_PHRASE],
    }
    values.update(overrides)
    return make_settings(**values)


async def _seconds_until_ack_spoken(settings: Settings, control: VapiControlClient) -> float:
    """Seconds from the turn reaching this process to the acknowledgement going out.

    Returns at the first content chunk carrying the acknowledgement rather than
    draining the turn: the measurement is when the callee could first have heard
    something, and the scripted Hermes turn behind it is deliberately slower than any
    deadline under test, so waiting for it would only add dead time to the test run.
    """
    reaping: set[asyncio.Task[None]] = set()
    agen = stream_turn(
        settings=settings,
        # Far slower than any budget here: this turn exists only to guarantee dead
        # air, so the acknowledgement path is what is timed and nothing else.
        hermes=_ScriptedHermes([(30.0, done("The real answer."))]),
        control=control,
        control_url=CONTROL_URL,
        state=make_state(settings),
        instructions="instructions",
        history=[],
        user_input="hello",
        reaping=reaping,
    )
    started = time.monotonic()
    try:
        async for chunk in agen:
            for event in _parse_chunk(chunk):
                if isinstance(event, tuple) and event[0] == "content" and ACK_PHRASE in event[1]:
                    return time.monotonic() - started
    finally:
        with contextlib.suppress(Exception):
            await agen.aclose()
        for task in list(reaping):
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
    raise AssertionError("the turn ended without ever speaking an acknowledgement")


# --- 1. the acceptance case: control stalls, the callee still hears a line in time --


async def test_stalled_control_post_still_speaks_inside_the_adapter_share() -> None:
    """Worst case -- the primary delivery channel never answers -- and the callee
    still hears the holding line inside the adapter's slice of the two seconds.

    This is the regression the live call failed. The fallback is only reached once the
    control POST has been given up on, so the acknowledgement can never be more timely
    than that timeout, and a timeout larger than the whole budget makes the fallback
    worthless however fast it is.
    """
    settings = production_settings()
    transport = SilentControlEndpoint(stall_phase="read")
    control = VapiControlClient(transport=transport)
    try:
        elapsed = await _seconds_until_ack_spoken(settings, control)
    finally:
        await control.aclose()

    assert len(transport.requests) == 1, "the control channel must still be tried first"
    heard_gap = elapsed + PLATFORM_OVERHEAD_SECONDS
    assert elapsed <= ADAPTER_SHARE_SECONDS, (
        f"ack handed out {elapsed:.3f}s after the turn arrived, over the"
        f" {ADAPTER_SHARE_SECONDS:.3f}s the adapter may spend; the callee would hear it"
        f" {heard_gap:.3f}s after they stopped talking, against a"
        f" {R2_BUDGET_SECONDS:.1f}s requirement"
    )
    assert heard_gap <= R2_BUDGET_SECONDS


async def test_stalled_control_post_falls_back_with_no_extra_waiting() -> None:
    """The fallback is immediate: the acknowledgement lands at the control deadline,
    not at the deadline plus another round of anything.

    Guards the shape of the failover as well as its total -- a retry, a second POST or
    a sleep-and-see slipped in here would still satisfy the budget assertion above
    while spending the callee's headroom on nothing.
    """
    settings = production_settings()
    transport = SilentControlEndpoint(stall_phase="read")
    control = VapiControlClient(transport=transport)
    try:
        elapsed = await _seconds_until_ack_spoken(settings, control)
    finally:
        await control.aclose()

    handover = elapsed - settings.filler_after_seconds
    assert len(transport.requests) == 1, "failover must not retry the stalled channel"
    assert handover <= settings.ack_control_timeout * 1.25, (
        f"failover took {handover:.3f}s against a {settings.ack_control_timeout:.3f}s"
        " control deadline: something is waiting after the POST was abandoned"
    )


# --- 2. the same guarantee at its source: say() is bounded on the WALL CLOCK --------


async def test_say_is_bounded_on_the_wall_clock_not_per_phase() -> None:
    """``say(timeout=T)`` gives up after T seconds of real time, full stop.

    A bare float handed to httpx is not a request ceiling: it is applied to connect,
    read, write and pool separately, and a phase that consumes its whole share and
    then succeeds raises nothing at all. So "timeout=0.2" against an endpoint that is
    merely slow, then silent, costs 0.6 s. Any budget arithmetic resting on the float
    alone is wrong by a multiple, which is why the caller's deadline is enforced here
    on the clock instead of being delegated to the transport.
    """
    transport = SilentControlEndpoint(stall_phase="read", slow_phases=("connect", "write"))
    control = VapiControlClient(transport=transport)
    timeout = 0.2
    started = time.monotonic()
    try:
        delivered = await control.say(CONTROL_URL, "hello", call_ref="ref-1", timeout=timeout)
    finally:
        await control.aclose()
    elapsed = time.monotonic() - started

    assert delivered is False, "a stalled control POST is a failed delivery, not a silent one"
    assert elapsed < timeout * 1.5, (
        f"say() took {elapsed:.3f}s under a {timeout:.3f}s timeout: the caller's"
        " deadline is not actually bounded on the wall clock"
    )


async def test_connect_phase_is_bounded_more_tightly_than_the_read_phase() -> None:
    """A cold client, or a host whose route just changed, stalls in CONNECT, not read.

    Spending the entire budget on the handshake guarantees there is none left to send
    the request and read the reply, so connect gets the tighter share -- and no phase
    ever gets more time than the request itself has.
    """
    transport = SilentControlEndpoint(stall_phase="connect")
    control = VapiControlClient(transport=transport)
    total = 0.5
    try:
        assert await control.say(CONTROL_URL, "hi", call_ref="ref-1", timeout=total) is False
    finally:
        await control.aclose()

    budget = transport.requests[0].extensions["timeout"]
    assert budget["connect"] < budget["read"], "a connect stall must fail before a read stall"
    assert max(budget.values()) <= total, "no phase may outlive the request's own ceiling"


# --- 3. the derivation, so the number cannot drift away from the requirement -------


def test_control_timeout_is_derived_from_the_budget() -> None:
    """The shipped ceiling is the residue of the budget, not a hand-picked constant.

    2.0 s requirement - 1.25 s budgeted platform overhead - 0.3 s dead-air wait
    = 0.45 s, which is ~2x the measured 0.23 s round trip. Tuning any of the three
    moves it automatically; that is the whole reason it is not a literal.
    """
    settings = production_settings()
    expected = round(
        settings.ack_budget_seconds
        - settings.ack_platform_overhead_seconds
        - settings.filler_after_seconds,
        3,
    )
    assert settings.ack_control_timeout_seconds is None, "the shipped value must be derived"
    assert settings.ack_control_timeout == expected
    assert settings.ack_control_timeout == 0.45


def test_shipped_defaults_fit_the_requirement_in_the_worst_case() -> None:
    """filler_after + a fully spent control timeout + the platform overhead <= 2 s.

    The arithmetic the previous defaults failed: 0.9 + 3.0 + 1.191 = 5.09 s against a
    two second requirement. Asserted on the shipped values, against the MEASURED
    overhead rather than the budgeted one, so the headroom between them is real.
    """
    settings = production_settings()
    worst_case = settings.filler_after_seconds + settings.ack_control_timeout
    assert worst_case <= settings.ack_budget_seconds - settings.ack_platform_overhead_seconds
    assert worst_case + PLATFORM_OVERHEAD_SECONDS <= R2_BUDGET_SECONDS


def test_answer_delivery_keeps_its_own_generous_timeout() -> None:
    """The answer's control POST must not inherit the acknowledgement's tight ceiling.

    They are different deadlines: the acknowledgement is racing the callee's patience
    and must fail fast into the fallback, while the answer is spoken from a background
    task with Hermes already finished and nothing waiting on it. Cutting the answer
    short to protect a deadline it is not on would lose the actual reply.
    """
    settings = production_settings()
    assert settings.control_answer_timeout_seconds > settings.ack_control_timeout


def test_operator_override_wins_over_the_derivation() -> None:
    """An explicitly pinned VHV_ACK_CONTROL_TIMEOUT_SECONDS is obeyed, not clamped: an
    operator overriding a derived default is making a deliberate choice, and silently
    substituting our arithmetic for theirs would be the surprise.
    """
    settings = make_settings(ack_control_timeout_seconds=1.75)
    assert settings.ack_control_timeout == 1.75


def test_budget_that_no_longer_fits_is_logged_not_absorbed(caplog: Any) -> None:
    """A configuration whose worst case overruns the requirement warns, and boots.

    Refusing to start would take the phone line down to protect a latency deadline,
    which is the wrong trade; going quiet about it is the wrong trade too. Both the
    floor-clamped derivation and an oversized operator override land here.
    """
    with caplog.at_level("WARNING"):
        settings = make_settings(filler_after_seconds=0.79)
    worst_case = (
        settings.ack_platform_overhead_seconds
        + settings.filler_after_seconds
        + settings.ack_control_timeout
    )
    assert worst_case > settings.ack_budget_seconds, (
        "this configuration is meant to overrun -- that is what the warning is for"
    )
    assert settings.ack_control_timeout == 0.25, "the derivation must clamp at its floor"
    assert any("acknowledgement worst case" in record.message for record in caplog.records), (
        "an acknowledgement budget that no longer fits must be named in the log"
    )
