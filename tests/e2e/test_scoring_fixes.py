"""Regression cover for the three harness scoring bugs found against deployed main
(87a2102, call 01a0262b): ack-channel attribution, ack-storm arithmetic, and R1
transport verifiability. Pure arithmetic, no network, no Vapi call, no ``say``.

Not collected by a bare ``pytest`` run -- like the rest of ``tests/e2e``, this package
is opt-in only (see ``tests/e2e/__init__.py``). Run directly:

    uv run pytest tests/e2e/test_scoring_fixes.py

Each ``r1_transport_scope`` test imports it locally rather than at module scope: that
function did not exist before this fix, so those specific tests fail on import against
the pre-fix module while the ack-storm and ack-channel tests below fail on their own
assertions -- every fix here was proven to fail before it landed.
"""

from __future__ import annotations

from .deadlines import Budgets, Report, Turn, Utterance, evaluate, evaluate_transport

POOL = [
    "I have that information right here, give me a second.",
    "Let me pull that up for you.",
    "One moment while I check.",
    "Just a second, looking now.",
]


def verdicts(report: Report) -> dict[str, str]:
    return {c.id: c.verdict for c in report.checks}


def measured(report: Report, check_id: str) -> float | None:
    return next(c.measured_s for c in report.checks if c.id == check_id)


# --- ack-storm arithmetic must never disagree with the cooldown ------------------


def test_ack_storm_max_is_derived_from_the_cooldown() -> None:
    """Not configured independently, so the two checks can never disagree."""
    assert Budgets().ack_storm_max == 2  # floor(16 / 10) + 1
    assert Budgets(ack_cooldown_s=16.0, ack_storm_window_s=16.0).ack_storm_max == 1
    assert Budgets(ack_cooldown_s=5.0, ack_storm_window_s=16.0).ack_storm_max == 4
    assert Budgets(ack_cooldown_s=20.0, ack_storm_window_s=16.0).ack_storm_max == 1


def _ack_call(offsets_s: list[float]) -> dict:
    """A call whose only content is "Hello?" then one acknowledgement (1 s long) at
    each offset in ``offsets_s``, seconds from call start.
    """
    messages = [
        {
            "role": "user",
            "message": "Hello?",
            "time": 0,
            "endTime": 100,
            "secondsFromStart": 0.0,
            "duration": 100,
        },
    ]
    for i, at in enumerate(offsets_s):
        messages.append(
            {
                "role": "bot",
                "message": POOL[i % len(POOL)],
                "time": int(at * 1000),
                "endTime": int((at + 1.0) * 1000),
                "secondsFromStart": at,
                "duration": 1000,
            }
        )
    return {"messages": messages}


def test_two_correctly_spaced_acknowledgements_pass_the_storm_check() -> None:
    """The exact contradiction this fix resolves: two acknowledgements 11.6 s apart,
    both inside a 16 s window, respecting the 10 s cooldown -- must not read as a
    storm. Failed under the old hardcoded ``ack_storm_max=1``.
    """
    call = _ack_call([0.5, 12.1])
    report = evaluate(call, phrases=POOL, first_message=None)
    v = verdicts(report)
    assert v["r2_ack_cooldown"] == "pass"
    assert v["r2_ack_storm"] == "pass"


def test_original_ack_storm_failure_is_still_caught() -> None:
    """The literal reported failure: six acknowledgements inside 16 s, at offsets
    0.05/0.22/0.40/0.57/0.74/0.91 s -- a real burst, not a pair of correctly spaced
    acknowledgements. Must fail both the cooldown and the (now-derived) storm check.
    """
    call = _ack_call([0.05, 0.22, 0.40, 0.57, 0.74, 0.91])
    report = evaluate(call, phrases=POOL, first_message=None)
    v = verdicts(report)
    assert len(report.acks) == 6
    assert v["r2_ack_cooldown"] == "fail"
    assert v["r2_ack_storm"] == "fail"
    assert measured(report, "r2_ack_storm") == 6.0
    assert measured(report, "r2_ack_storm") > Budgets().ack_storm_max


# --- control-channel acknowledgement attribution ----------------------------------


def test_undercounted_channel_no_longer_contradicts_the_spoken_count() -> None:
    """Recorded live (call 01a0262b): one channel=stream acknowledgement visible in the
    websocket events, one channel=control acknowledgement only visible in what Vapi
    actually spoke. The old code reported "1 emitted" / "1 emitted, 2 spoken" -- more
    spoken than emitted, a contradiction since nothing else can produce speech.
    """
    events = [
        {
            "at_s": 3.902,
            "event": {"type": "model-output", "output": "One moment while I check. <flush />"},
        },
    ]
    checks, _ = evaluate_transport(events, callee_turn_end_s=0.0, spoken_ack_count=2, phrases=POOL)
    by_id = {c.id: c for c in checks}
    dropped = by_id["acks_reached_the_callee"]
    assert dropped.verdict == "pass"
    assert dropped.measured_s == 0.0
    assert "1 via channel=stream" in dropped.detail
    assert "1 via channel=control (inferred)" in dropped.detail
    assert "2 emitted, 2 spoken" in dropped.detail


def test_control_channel_ack_after_turn_end_is_not_falsely_reported_as_never_sent() -> None:
    """No model-output line ever reached the transport after the turn ended, but the
    spoken count proves one acknowledgement was delivered via channel=control -- must
    not claim "the adapter never produced an acknowledgement at all" (a live falsehood
    on this call), and must not fabricate a latency for a channel it cannot time.
    """
    checks, _ = evaluate_transport(
        events=[], callee_turn_end_s=5.0, spoken_ack_count=1, phrases=POOL
    )
    emitted = next(c for c in checks if c.id == "r2_ack_emitted")
    assert emitted.verdict == "skip"
    assert "channel=control" in emitted.detail
    assert "never produced an acknowledgement" not in emitted.detail
    assert emitted.measured_s is None


def test_zero_emitted_and_zero_spoken_is_still_reported_as_no_acknowledgement() -> None:
    """The pre-existing, correct behaviour must survive the fix: genuinely nothing on
    either channel is still a real failure, not an inferred control-channel pass.
    """
    checks, _ = evaluate_transport(
        events=[], callee_turn_end_s=5.0, spoken_ack_count=0, phrases=POOL
    )
    emitted = next(c for c in checks if c.id == "r2_ack_emitted")
    assert emitted.verdict == "fail"
    assert "never produced an acknowledgement at all" in emitted.detail


# --- R1 transport verifiability -----------------------------------------------------


def test_websocket_call_reports_r1_as_unverifiable_not_fail_or_pass() -> None:
    """The live evidence: r1_deadline FAIL 5.107s on a vapi.websocket call, where the
    reason-for-calling fast path (gated on call.type == "outboundPhoneCall") can never
    fire. Neither a silent pass nor a fail -- its own explicit, disciplined check.
    """
    from .deadlines import r1_transport_scope

    call = {"type": "vapi.websocketCall", "messages": []}
    report = Report(checks=[], turns=[], utterances=[], duration_unit="ms")
    check = r1_transport_scope(call, report)
    assert check.verdict == "skip"
    assert "UNVERIFIABLE-BY-THIS-TRANSPORT" in check.detail
    assert "outboundPhoneCall" in check.detail


def test_outbound_phone_call_reports_r1_as_measurable() -> None:
    from .deadlines import r1_transport_scope

    call = {"type": "outboundPhoneCall", "messages": []}
    report = Report(checks=[], turns=[], utterances=[], duration_unit="ms")
    check = r1_transport_scope(call, report)
    assert check.verdict == "pass"


def test_max_turn_gap_dominance_by_the_r1_turn_is_called_out() -> None:
    """When the worst turn max_turn_gap ever measures IS the unverifiable R1 turn, say
    so explicitly, so that FAIL is not mistaken for evidence of an R1 regression.
    """
    from .deadlines import r1_transport_scope

    user = Utterance(0, "user", "Hello?", 0.0, 0.1)
    reply = Utterance(1, "bot", "Hi there, how can I help?", 5.207, 6.0)
    turn = Turn(user=user, reply=reply)
    report = Report(checks=[], turns=[turn], utterances=[user, reply], duration_unit="ms")
    check = r1_transport_scope({"type": "vapi.websocketCall"}, report)
    assert "max_turn_gap above is dominated" in check.detail
