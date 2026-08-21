"""R2 budget regression tests: the callee hears an acknowledgement inside two seconds
of finishing a sentence, INCLUDING when the network the adapter would otherwise have
depended on never answers at all.

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

Two live incidents on the SAME call (01a02681) pin down why the acknowledgement no
longer makes a network call at all. Adapter journal, verbatim:

    18:47:22,501 turns        turn end call=5a6238eef3e8 ttfb_ms=- total_ms=391 outcome=disconnected
    18:47:37,546 vapi_control ack control request timed out call=5a6238eef3e8 after=0.450s
    18:47:37,548 turns        turn filler call=5a6238eef3e8 elapsed_ms=750 channel=stream
    18:47:41,793 vapi_control control origin warm-up failed origin=... error=TimeoutError
    18:47:48,365 httpx        POST https://phone-call-websocket.../control "HTTP/1.1 200 OK"
    18:47:48,365 turns        turn filler call=5a6238eef3e8 elapsed_ms=11567 channel=control

Turn 1 (the disconnected one, above) had already fallen back to the SSE stream and
rendered its "Alright, let me see." in well under a second, because the response
ended right behind the flushed chunk. Turn 2's identical fallback, sharing the same
call, was never heard: 0.45 s were spent on a control POST that never answered, and
the response then stayed open another 11 s waiting on Hermes, which is exactly the
shape the SSE path cannot survive (docs/integration-contracts.md 1.6). Probed live
against the real control origin afterward (GET /, HEAD /, OPTIONS /, GET of a
nonexistent path, POST to a nonexistent call id): every one of those answers in
~0.22-0.33 s, but every single response -- including the 400s -- carries
``Connection: close``, so no warm-up of any shape could ever have produced a
connection the real POST could reuse; one in ~40 cold-process trials also hung for
the warm-up's own 5 s timeout, which is the second log line above. Both failure
modes -- the slow POST, and the pointless warm-up racing to fail alongside it -- are
now impossible because the acknowledgement no longer makes a control POST at all.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any, ClassVar

import httpx

from test_ack_control import CONTROL_URL, make_control
from test_turns import _parse_chunk, _ScriptedHermes, done, make_settings, make_state
from vapi_hermes_voice.config import Settings
from vapi_hermes_voice.turns import stream_turn
from vapi_hermes_voice.vapi_control import VapiControlClient

# The requirement: the callee hears something within two seconds of stopping talking.
R2_BUDGET_SECONDS = 2.0
# Measured (see module docstring): the 2.321 s heard gap minus the 1.130 s the
# adapter's own journal accounts for. Endpointing ahead of this process, TTS after it.
PLATFORM_OVERHEAD_SECONDS = 1.191
ACK_PHRASE = "Okay, let me check."


class NeverAnswers(httpx.AsyncBaseTransport):
    """A control endpoint that never answers at all -- not slow, not erroring,
    simply silent for as long as the caller is willing to wait.

    Standing in for the class of failure the acknowledgement used to depend on: a
    lost SYN, a stalled TLS handshake, a backend that accepted the connection and
    then said nothing. The point of this file's first test is that it no longer
    matters how long this transport would take to answer, because nothing on the
    acknowledgement path ever asks it a question.
    """

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        await asyncio.sleep(3600)  # "never" -- any test that reaches this has already lost
        raise AssertionError("unreachable")


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


async def _time_ack(settings: Settings, control: VapiControlClient | None) -> float:
    """Seconds from the turn reaching this process to the acknowledgement chunk.

    Stops at the acknowledgement rather than draining the turn: the measurement is
    when the callee could first have heard something, and the scripted Hermes turn
    behind it is deliberately slower than any deadline under test.
    """
    reaping: set[asyncio.Task[Any]] = set()
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


# --- 1. the acceptance case: the acknowledgement never waits on the network ----


async def test_ack_never_waits_on_a_control_round_trip_at_all() -> None:
    """The regression this file exists to catch: however badly the control origin
    behaves, the acknowledgement lands at ``filler_after_seconds`` and nothing more,
    because nothing on this path ever asks that origin a question.

    A transport that never answers at all (worse than the live incident's 0.45 s
    timeout by a wide margin) is used deliberately: if anything on the
    acknowledgement path still touched it, this test would hang, not merely run
    slow -- there is no timeout value that would make a wrong implementation of this
    pass by accident.
    """
    settings = production_settings()
    transport = NeverAnswers()
    control = VapiControlClient(transport=transport)
    try:
        elapsed = await asyncio.wait_for(_time_ack(settings, control), timeout=5.0)
    finally:
        await control.aclose()

    assert transport.requests == [], "the acknowledgement must never touch the control origin"
    heard_gap = elapsed + PLATFORM_OVERHEAD_SECONDS
    assert elapsed <= settings.filler_after_seconds + 0.25, (
        f"ack handed out {elapsed:.3f}s after the turn arrived, against a"
        f" {settings.filler_after_seconds:.3f}s dead-air wait and no network hop at all"
    )
    assert heard_gap <= R2_BUDGET_SECONDS


async def test_ack_timing_is_identical_with_or_without_a_control_client() -> None:
    """The acknowledgement's timing must not depend on whether control exists,
    still be reachable, or answer quickly -- proof that it is not merely fast in
    this test's specific transport, but structurally independent of it.
    """
    settings = production_settings()

    with_control = VapiControlClient(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    try:
        elapsed_with = await _time_ack(settings, with_control)
    finally:
        await with_control.aclose()

    elapsed_without = await _time_ack(settings, None)

    assert abs(elapsed_with - elapsed_without) < 0.15, (
        f"ack timing differed by channel: {elapsed_with:.3f}s vs {elapsed_without:.3f}s"
    )


# --- 2. the derivation, so the number cannot drift away from the requirement -------


def test_shipped_defaults_fit_the_requirement_in_the_worst_case() -> None:
    """filler_after + the platform overhead <= the R2 budget, on the shipped values.

    Only two terms now: the acknowledgement no longer makes a network call of its
    own to budget a third term for (see the module docstring for why one was tried
    here and no longer is).
    """
    settings = production_settings()
    worst_case = settings.filler_after_seconds + settings.ack_platform_overhead_seconds
    assert worst_case <= settings.ack_budget_seconds
    assert settings.filler_after_seconds + PLATFORM_OVERHEAD_SECONDS <= R2_BUDGET_SECONDS


def test_budget_that_no_longer_fits_is_logged_not_absorbed(caplog: Any) -> None:
    """A configuration whose worst case overruns the requirement warns, and boots.

    Refusing to start would take the phone line down to protect a latency deadline,
    which is the wrong trade; going quiet about it is the wrong trade too.
    """
    with caplog.at_level("WARNING"):
        settings = make_settings(filler_after_seconds=1.5)
    worst_case = settings.ack_platform_overhead_seconds + settings.filler_after_seconds
    assert worst_case > settings.ack_budget_seconds, (
        "this configuration is meant to overrun -- that is what the warning is for"
    )
    assert any("acknowledgement worst case" in record.message for record in caplog.records), (
        "an acknowledgement budget that no longer fits must be named in the log"
    )


def test_retired_control_settings_do_not_crash_loop_a_deployed_env() -> None:
    """The two knobs this change removed (``ack_use_call_control``,
    ``ack_control_timeout_seconds``) must be accepted-and-ignored, not rejected --
    this model forbids extras, so a value left behind in a deployed .env would
    otherwise crash-loop the unit on its next restart.
    """
    settings = make_settings(
        ack_use_call_control=False,
        ack_control_timeout_seconds=1.75,  # type: ignore[call-arg]
    )
    assert not hasattr(settings, "ack_use_call_control")
    assert not hasattr(settings, "ack_control_timeout_seconds")


# --- 3. the answer's own delivery: generous, retried once, never on this deadline --


async def test_answer_delivery_keeps_a_generous_timeout_independent_of_the_ack() -> None:
    """The answer's control POST runs from a background task with Hermes already
    finished and nothing waiting on it, so it is not sized off the acknowledgement's
    dead-air wait at all -- unlike before this change, when both ceilings were
    derived from the same R2 budget and one could not move without the other.
    """
    settings = production_settings()
    assert settings.control_answer_timeout_seconds == 3.0
    assert settings.control_answer_timeout_seconds > settings.filler_after_seconds


async def test_answer_delivery_worst_case_is_two_attempts_of_the_configured_timeout() -> None:
    """Worst case for the answer, stated and proven: a plausibly-transient failure
    (timeout, network error, 5xx) is retried exactly once on a fresh connection
    before the answer is given up on as undeliverable -- so the wall-clock worst
    case is 2x ``control_answer_timeout_seconds``, never more.
    """
    settings = production_settings(filler_after_seconds=0.05, control_answer_timeout_seconds=0.2)
    control, requests = make_control(lambda r: httpx.Response(500, text="boom"))
    try:
        started = time.monotonic()
        reaping: set[asyncio.Task[Any]] = set()
        agen = stream_turn(
            settings=settings,
            hermes=_ScriptedHermes([(0.15, done("The answer."))]),
            control=control,
            control_url=CONTROL_URL,
            state=make_state(settings),
            instructions="instructions",
            history=[],
            user_input="hello",
            reaping=reaping,
        )
        async for _chunk in agen:
            pass
        for task in list(reaping):
            await task
        elapsed = time.monotonic() - started
    finally:
        await control.aclose()

    assert len(requests) == 2, "exactly one retry, not zero and not more"
    assert elapsed <= 2 * settings.control_answer_timeout_seconds + 0.5, (
        f"answer delivery took {elapsed:.3f}s against a worst case of"
        f" 2x{settings.control_answer_timeout_seconds:.3f}s"
    )


# --- 4. say() itself: bounded on the wall clock, no phase squeezed ------------


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
    than for the float -- the arithmetic ``say``'s wall-clock bound exists to survive.
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
        outcome = await control.say(CONTROL_URL, "hello", call_ref="ref-1", timeout=timeout)
    finally:
        await control.aclose()
    elapsed = time.monotonic() - started

    assert outcome.delivered is False, "a stalled control POST is a failed delivery"
    assert elapsed < timeout * 1.5, (
        f"say() took {elapsed:.3f}s under a {timeout:.3f}s timeout: the caller's"
        " deadline is not actually bounded on the wall clock"
    )


async def test_no_phase_is_squeezed_below_the_request_ceiling() -> None:
    """Connect is NOT given a tighter sub-share than the wall-clock ceiling.

    Sub-dividing buys nothing for the deadline -- the wall-clock bound already caps the
    total however the phases fall -- so every phase gets the whole budget.
    """
    transport = SilentControlEndpoint(stall_phase="connect")
    control = VapiControlClient(transport=transport)
    total = 0.5
    try:
        outcome = await control.say(CONTROL_URL, "hi", call_ref="ref-1", timeout=total)
        assert outcome.delivered is False
    finally:
        await control.aclose()

    budget = transport.requests[0].extensions["timeout"]
    assert min(budget.values()) == total, "no phase may be squeezed below the ceiling"
    assert max(budget.values()) <= total, "no phase may outlive the request's own ceiling"


async def test_say_reports_the_status_code_of_a_rejection() -> None:
    """A 4xx/5xx is a failed delivery either way, but the status code that came back
    is preserved so a caller can tell "Vapi rejected this" from "nothing answered".
    """
    control, _requests = make_control(lambda r: httpx.Response(400, text="bad"))
    try:
        outcome = await control.say(CONTROL_URL, "hi", call_ref="ref-1", timeout=1.0)
    finally:
        await control.aclose()
    assert outcome.delivered is False
    assert outcome.status_code == 400


async def test_say_reports_no_status_code_on_timeout() -> None:
    transport = SilentControlEndpoint(stall_phase="read")
    control = VapiControlClient(transport=transport)
    try:
        outcome = await control.say(CONTROL_URL, "hi", call_ref="ref-1", timeout=0.1)
    finally:
        await control.aclose()
    assert outcome.delivered is False
    assert outcome.status_code is None
