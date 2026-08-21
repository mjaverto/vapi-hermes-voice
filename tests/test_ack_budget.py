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
from starlette.testclient import TestClient

from fake_hermes import FakeScript, build_fake_hermes_transport
from test_ack_control import CONTROL_URL
from test_turns import _parse_chunk, _ScriptedHermes, done, make_settings, make_state
from vapi_hermes_voice.config import Settings
from vapi_hermes_voice.server import create_app
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
# The origin CONTROL_URL lives on. httpx pools by origin, not by path, so this is what
# a warm-up has to open in order for the later per-call POST to reuse it.
_ORIGIN = "https://phone-call-websocket.vapi.ai/"
# The same origin as a pool KEY (scheme://host[:port], no path), which is how httpx
# itself keys the connection pool and how the transport doubles below record handshakes.
_ORIGIN_KEY = "https://phone-call-websocket.vapi.ai"


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


class ColdConnectEndpoint(httpx.AsyncBaseTransport):
    """A control endpoint that charges a TCP+TLS handshake for the first request to an
    origin and then serves pooled requests without one, like any real HTTPS host.

    The handshake is charged against the CONNECT budget and the round trip against the
    READ budget, and either raises its own phase's timeout when it does not fit, exactly
    as httpx would. ``connects`` counts handshakes actually paid, which is what a test
    about connection reuse needs to see.
    """

    def __init__(self, *, handshake: float, round_trip: float = 0.0) -> None:
        self.handshake = handshake
        self.round_trip = round_trip
        self.connects: list[str] = []
        self.requests: list[httpx.Request] = []
        self._pooled: set[str] = set()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        budget: dict[str, float | None] = request.extensions["timeout"]
        origin = f"{request.url.scheme}://{request.url.netloc.decode()}"
        if origin not in self._pooled:
            allowed = budget["connect"]
            if allowed is not None and self.handshake > allowed:
                await asyncio.sleep(allowed)
                raise httpx.ConnectTimeout("handshake did not fit", request=request)
            await asyncio.sleep(self.handshake)
            self._pooled.add(origin)
            self.connects.append(origin)
        allowed = budget["read"]
        if allowed is not None and self.round_trip > allowed:
            await asyncio.sleep(allowed)
            raise httpx.ReadTimeout("round trip did not fit", request=request)
        await asyncio.sleep(self.round_trip)
        return httpx.Response(200, json={"status": "ok"})


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


async def _time_ack(settings: Settings, control: VapiControlClient) -> tuple[float, str]:
    """``(seconds, channel)`` from the turn reaching this process to the acknowledgement.

    ``channel`` is ``"control"`` or ``"stream"`` -- which one is not a detail. The
    stream fallback carries a live Vapi defect that can render no audio at all, so a
    test that only checked the timing would pass while the callee heard silence.

    Stops at the acknowledgement rather than draining the turn: the measurement is when
    the callee could first have heard something, and the scripted Hermes turn behind it
    is deliberately slower than any deadline under test. A control-delivered
    acknowledgement produces no content chunk at all -- the turn hands off to a
    background continuation and ends the response immediately -- so the ``[DONE]`` that
    follows the successful ``say`` is the timestamp in that case.
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
                    return time.monotonic() - started, "stream"
                if event == "[DONE]":
                    return time.monotonic() - started, "control"
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
        elapsed, channel = await _time_ack(settings, control)
    finally:
        await control.aclose()

    assert channel == "stream", "a stalled control POST must fall back, not hang on"
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
        elapsed, _channel = await _time_ack(settings, control)
    finally:
        await control.aclose()

    handover = elapsed - settings.filler_after_seconds
    assert len(transport.requests) == 1, "failover must not retry the stalled channel"
    assert handover <= settings.ack_control_timeout * 1.25, (
        f"failover took {handover:.3f}s against a {settings.ack_control_timeout:.3f}s"
        " control deadline: something is waiting after the POST was abandoned"
    )


# --- 1b. the handshake must not be what diverts the ack onto the defective path -----
#
# Failing over to the SSE stream is not a neutral second best. That path carries a live
# Vapi defect (contracts 1.6, root-caused on calls 01a025e5 and 01a025ee and reproduced
# on an isolated probe with no Hermes traffic): a `<flush />`-terminated chunk on a
# stream that then stalls is accepted and echoed back in ~1 ms and frequently never
# rendered to audio at all. So a handshake that misses the acknowledgement's deadline
# risks the callee hearing NOTHING -- and the first acknowledgement after an adapter
# restart, which is exactly where the requirement is judged, was the one paying for it.


async def test_cold_handshake_that_fits_the_ceiling_is_not_abandoned() -> None:
    """A 0.20 s handshake plus a 0.23 s round trip is 0.43 s: inside the 0.45 s ceiling,
    so it must be spent on the reliable channel rather than refused.

    This is what an earlier revision got wrong by giving connect a 40% sub-share: 0.20 s
    exceeded the 0.18 s that share allowed, so a request with 0.02 s of genuine headroom
    was abandoned into the path that can produce no audio.
    """
    settings = production_settings()
    transport = ColdConnectEndpoint(handshake=0.20, round_trip=0.23)
    control = VapiControlClient(transport=transport)
    try:
        elapsed, channel = await _time_ack(settings, control)
    finally:
        await control.aclose()

    assert channel == "control", (
        "a handshake with room to spare inside the ceiling was abandoned into the SSE"
        " fallback, which can render no audio at all"
    )
    assert elapsed <= ADAPTER_SHARE_SECONDS


async def test_warm_up_keeps_the_handshake_off_the_acknowledgement_deadline() -> None:
    """The proof for the warm-up: a handshake too big to fit ALONGSIDE the round trip
    inside the ceiling still leaves the acknowledgement on the reliable channel,
    because the warm-up already paid for it.

    0.40 s handshake + 0.23 s round trip = 0.63 s, well past the 0.45 s ceiling, so the
    acknowledgement cannot afford it and would fail over. Warmed first, the POST finds a
    pooled connection and pays only the round trip.
    """
    settings = production_settings()
    transport = ColdConnectEndpoint(handshake=0.40, round_trip=0.23)

    cold = VapiControlClient(transport=ColdConnectEndpoint(handshake=0.40, round_trip=0.23))
    try:
        _elapsed, cold_channel = await _time_ack(settings, cold)
    finally:
        await cold.aclose()
    assert cold_channel == "stream", (
        "this handshake is meant not to fit unwarmed -- otherwise the test below proves nothing"
    )

    control = VapiControlClient(transport=transport)
    try:
        # What server.py does at the top of every turn, off the critical path.
        assert await control.warm(CONTROL_URL) is True
        elapsed, channel = await _time_ack(settings, control)
    finally:
        await control.aclose()

    assert channel == "control", "the warm-up did not keep the acknowledgement off SSE"
    assert transport.connects == [_ORIGIN_KEY], "the handshake must be paid once, by the warm-up"
    assert elapsed <= ADAPTER_SHARE_SECONDS, (
        f"ack handed out {elapsed:.3f}s after the turn arrived, over the"
        f" {ADAPTER_SHARE_SECONDS:.3f}s the adapter may spend"
    )


async def test_warm_up_targets_the_origin_and_touches_no_call_resource() -> None:
    """The warm-up GETs the origin root, never the per-call control URL.

    httpx pools by origin, so the root shares the connection the later
    ``POST .../<call-id>/control`` will reuse -- which is the only reason warming is
    possible at all, since the control URL cannot be known before the call exists. It
    also means the warm-up cannot have any effect on a call.
    """
    transport = ColdConnectEndpoint(handshake=0.0)
    control = VapiControlClient(transport=transport)
    try:
        assert await control.warm(CONTROL_URL) is True
    finally:
        await control.aclose()

    assert len(transport.requests) == 1
    warmed = transport.requests[0]
    assert warmed.method == "GET", "the warm-up must not POST anything to a live call"
    assert str(warmed.url) == _ORIGIN
    assert "control" not in str(warmed.url)


async def test_warm_up_is_skipped_while_the_connection_is_known_live() -> None:
    """Called on every turn, it sends at most one request per origin per refresh window.

    A completed ``say`` counts as proof the connection is live, so the next turn's
    warm-up is free. Without that, a per-turn warm-up would be a per-turn extra request.
    """
    transport = ColdConnectEndpoint(handshake=0.0)
    control = VapiControlClient(transport=transport)
    try:
        assert await control.warm(CONTROL_URL) is True
        assert await control.warm(CONTROL_URL) is True
        assert len(transport.requests) == 1, "a fresh origin must not be re-warmed"

        assert await control.say(CONTROL_URL, "hi", call_ref="ref-1", timeout=0.45) is True
        assert len(transport.requests) == 2
        assert await control.warm(CONTROL_URL) is True
        assert len(transport.requests) == 2, "a completed POST proves the connection live"
    finally:
        await control.aclose()

    assert transport.connects == [_ORIGIN_KEY], "one handshake, then pooled reuse throughout"


async def test_pooled_connection_outlives_the_gap_between_turns() -> None:
    """httpx evicts idle pooled connections after FIVE SECONDS by default, which for
    this workload means the pool is decorative: acknowledgements are at least
    ``filler_min_gap_seconds`` (10 s) apart, so essentially every control POST was
    paying a fresh handshake -- not just the first one after a restart.
    """
    assert VapiControlClient.keepalive_expiry_seconds > make_settings().filler_min_gap_seconds, (
        "an idle pooled connection must outlive the gap between two acknowledgements,"
        " or it is never actually reused"
    )


def test_the_server_actually_fires_the_warm_up() -> None:
    """End-to-end wiring: a real request carrying ``call.monitor.controlUrl`` makes the
    app warm that origin, without the turn waiting for it.

    Unit-testing ``warm`` proves the method works; this proves it is CALLED. A warm-up
    nothing invokes is the most plausible way for this whole change to be silently
    worthless, and it would leave every other test in this file still passing.
    """
    warmed: list[tuple[str, str]] = []

    def record(request: httpx.Request) -> httpx.Response:
        warmed.append((request.method, str(request.url)))
        return httpx.Response(200, json={"status": "ok"})

    hermes_transport, _state = build_fake_hermes_transport(FakeScript(deltas=["Paris."]))
    settings = make_settings(
        adapter_api_key="unit-test-secret-0123456789",
        warmup_on_start=False,
        filler_after_seconds=5.0,  # no ack this turn: the warm-up is the subject
    )
    app = create_app(
        settings,
        hermes_transport=hermes_transport,
        vapi_control_transport=httpx.MockTransport(record),
    )
    body = {
        "model": "hermes",
        "stream": False,
        "messages": [{"role": "user", "content": "What's the capital of France?"}],
        "call": {
            "id": "call-warm-1",
            "type": "inboundPhoneCall",
            "monitor": {"controlUrl": CONTROL_URL},
        },
    }
    with TestClient(app) as client:
        response = client.post(
            "/chat/completions",
            json=body,
            headers={"authorization": "Bearer unit-test-secret-0123456789"},
        )
    assert response.status_code == 200

    assert warmed == [("GET", _ORIGIN)], (
        "the server must warm the control origin at the top of the turn, with a GET of"
        f" the origin root and nothing else; got {warmed}"
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


async def test_no_phase_is_squeezed_below_the_request_ceiling() -> None:
    """Connect is NOT given a tighter sub-share than the wall-clock ceiling.

    Sub-dividing buys nothing for the deadline -- the wall-clock bound already caps the
    total however the phases fall -- and it costs something real. A handshake that would
    have finished in 0.20 s, leaving ample room for the 0.23 s round trip inside a
    0.45 s ceiling, would instead be abandoned at the sub-share, and abandoned INTO the
    SSE fallback, which carries a live Vapi defect that can render no audio at all. So
    every phase gets the whole budget and no phase gets more.
    """
    transport = SilentControlEndpoint(stall_phase="connect")
    control = VapiControlClient(transport=transport)
    total = 0.5
    try:
        assert await control.say(CONTROL_URL, "hi", call_ref="ref-1", timeout=total) is False
    finally:
        await control.aclose()

    budget = transport.requests[0].extensions["timeout"]
    assert min(budget.values()) == total, "no phase may be squeezed below the ceiling"
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
