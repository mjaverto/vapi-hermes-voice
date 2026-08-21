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


def test_surplus_spoken_acks_are_reported_unknown_not_inferred_as_control() -> None:
    """Recorded live (call 01a0262b): one channel=stream acknowledgement visible in the
    websocket events, one further acknowledgement only visible in what Vapi actually
    spoke. The old code either undercounted this ("1 emitted" / "1 emitted, 2 spoken",
    a contradiction) or -- an earlier version of this fix -- confidently inferred the
    surplus as channel=control. Both are wrong: several of the adapter's own filler
    phrases are ordinary enough that the model streaming the same words itself (without
    the adapter's ``<flush />`` marker) is indistinguishable from a genuine
    channel=control delivery on this transport alone -- exactly the PR #10 regression
    (the model re-acquiring its own holding-phrase habit) this check must not mask by
    guessing. The surplus must be reported UNKNOWN, not credited to a channel.
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
    assert "attribution UNKNOWN" in dropped.detail
    assert "channel=control (inferred)" not in dropped.detail


def test_surplus_ack_after_turn_end_is_reported_unknown_not_never_sent_or_control() -> None:
    """No model-output line ever reached the transport after the turn ended, but the
    spoken count proves one acknowledgement happened on SOME channel -- must not claim
    "the adapter never produced an acknowledgement at all" (a live falsehood on this
    call), must not fabricate a latency for an untimed channel, and must not guess
    channel=control with false confidence (see the test above for why).
    """
    checks, _ = evaluate_transport(
        events=[], callee_turn_end_s=5.0, spoken_ack_count=1, phrases=POOL
    )
    emitted = next(c for c in checks if c.id == "r2_ack_emitted")
    assert emitted.verdict == "skip"
    assert "UNKNOWN" in emitted.detail
    assert "channel=control" in emitted.detail  # named as one possibility, not a verdict
    assert "never produced an acknowledgement" not in emitted.detail
    assert emitted.measured_s is None


def test_zero_emitted_and_zero_spoken_is_still_reported_as_no_acknowledgement() -> None:
    """The pre-existing, correct behaviour must survive the fix: genuinely nothing on
    either channel is still a real failure, not an unattributed-surplus pass.
    """
    checks, _ = evaluate_transport(
        events=[], callee_turn_end_s=5.0, spoken_ack_count=0, phrases=POOL
    )
    emitted = next(c for c in checks if c.id == "r2_ack_emitted")
    assert emitted.verdict == "fail"
    assert "never produced an acknowledgement at all" in emitted.detail


def test_stream_drop_is_still_a_real_fail_not_softened_to_unknown() -> None:
    """channel=stream evidence of a drop (a timestamped line that never became audio)
    is real, direct evidence -- unlike the surplus-in-the-other-direction ambiguity
    above -- and must keep failing exactly as before.
    """
    events = [
        {
            "at_s": 3.0,
            "event": {"type": "model-output", "output": "One moment while I check. <flush />"},
        },
        {
            "at_s": 8.0,
            "event": {"type": "model-output", "output": "Let me pull that up for you. <flush />"},
        },
    ]
    checks, _ = evaluate_transport(events, callee_turn_end_s=1.0, spoken_ack_count=0, phrases=POOL)
    dropped = next(c for c in checks if c.id == "acks_reached_the_callee")
    assert dropped.verdict == "fail"
    assert dropped.measured_s == 2.0
    assert "Vapi-side" in dropped.detail


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
