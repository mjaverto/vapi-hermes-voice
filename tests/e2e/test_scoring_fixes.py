"""Regression cover for the harness's own scoring: acknowledgement attribution (both
the evidence-based verdicts and the UNKNOWN fallback), ack-storm arithmetic, and R1
transport verifiability. Pure arithmetic, no network, no Vapi call, no ``say``.

Not collected by a bare ``pytest`` run -- like the rest of ``tests/e2e``, this package
is opt-in only (see ``tests/e2e/__init__.py``). Run directly:

    uv run pytest tests/e2e/test_scoring_fixes.py
"""

from __future__ import annotations

from .deadlines import (
    AdapterAck,
    AdapterAcks,
    Budgets,
    Report,
    Turn,
    Utterance,
    attribute_acks,
    evaluate,
    evaluate_transport,
    r1_transport_scope,
)

POOL = [
    "I have that information right here, give me a second.",
    "Let me pull that up for you.",
    "One moment while I check.",
    "Just a second, looking now.",
]


def spoken(text: str, at_s: float, *, index: int = 0) -> Utterance:
    """One assistant utterance Vapi recorded itself as having spoken."""
    return Utterance(index=index, role="bot", text=text, start_s=at_s, end_s=at_s + 1.0)


def emitted(
    text: str, *, channel: str, elapsed_ms: int = 1130, at_epoch_s: float = 0.0
) -> AdapterAck:
    """One acknowledgement the adapter's own record says it sent."""
    return AdapterAck(text=text, channel=channel, at_epoch_s=at_epoch_s, elapsed_ms=elapsed_ms)


def by_id(checks: list) -> dict:
    return {c.id: c for c in checks}


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


# --- attribution WITHOUT the adapter's record: the UNKNOWN fallback ----------------
#
# Everything in this block is what the harness reports when GET /debug/acks/{call_ref}
# is unreachable, disabled, or predates the deployed build. It must stay exactly as
# honest as it was before that endpoint existed: never a guess, never a fabricated
# latency, never a false "the adapter produced nothing".


def test_surplus_spoken_acks_are_reported_unknown_not_inferred_as_control() -> None:
    """Recorded live (call 01a0262b): one channel=stream acknowledgement visible in the
    websocket events, one further acknowledgement only visible in what Vapi actually
    spoke. The old code either undercounted this ("1 emitted" / "1 emitted, 2 spoken",
    a contradiction) or confidently inferred the surplus as channel=control. Both are
    wrong: several of the adapter's own filler phrases are ordinary enough that the
    model streaming the same words itself (without the adapter's ``<flush />`` marker)
    is indistinguishable from a genuine channel=control delivery on this transport
    alone -- exactly the PR #10 regression (the model re-acquiring its own
    holding-phrase habit) this check must not mask by guessing. With no adapter record
    to read, the surplus must be reported UNKNOWN, not credited to a channel.
    """
    events = [
        {
            "at_s": 3.902,
            "event": {"type": "model-output", "output": "One moment while I check. <flush />"},
        },
    ]
    checks, _ = evaluate_transport(
        events,
        callee_turn_end_s=0.0,
        spoken_acks=[
            spoken("One moment while I check.", 4.0),
            spoken("Let me pull that up for you.", 9.0),
        ],
        phrases=POOL,
    )
    checks_by_id = by_id(checks)
    dropped = checks_by_id["acks_reached_the_callee"]
    assert dropped.verdict == "pass"
    assert dropped.measured_s == 0.0
    assert "1 via channel=stream" in dropped.detail
    assert "attribution UNKNOWN" in dropped.detail
    assert "channel=control (inferred)" not in dropped.detail
    # And the dedicated attribution check says the same thing in its own right.
    attribution = checks_by_id["ack_attribution"]
    assert attribution.verdict == "skip"
    assert "UNKNOWN" in attribution.detail
    assert attribution.measured_s is None


def test_surplus_ack_after_turn_end_is_reported_unknown_not_never_sent_or_control() -> None:
    """No model-output line ever reached the transport after the turn ended, but the
    spoken count proves one acknowledgement happened on SOME channel -- must not claim
    "the adapter never produced an acknowledgement at all" (a live falsehood on this
    call), must not fabricate a latency for an untimed channel, and must not guess
    channel=control with false confidence (see the test above for why).
    """
    checks, _ = evaluate_transport(
        events=[],
        callee_turn_end_s=5.0,
        spoken_acks=[spoken("One moment while I check.", 7.0)],
        phrases=POOL,
    )
    emitted_check = by_id(checks)["r2_ack_emitted"]
    assert emitted_check.verdict == "skip"
    assert "UNKNOWN" in emitted_check.detail
    assert "channel=control" in emitted_check.detail  # one possibility, not a verdict
    assert "never produced an acknowledgement" not in emitted_check.detail
    assert emitted_check.measured_s is None


def test_zero_emitted_and_zero_spoken_is_still_reported_as_no_acknowledgement() -> None:
    """The pre-existing, correct behaviour must survive: genuinely nothing on either
    channel is still a real failure, not an unattributed-surplus pass.
    """
    checks, _ = evaluate_transport(events=[], callee_turn_end_s=5.0, spoken_acks=[], phrases=POOL)
    emitted_check = by_id(checks)["r2_ack_emitted"]
    assert emitted_check.verdict == "fail"
    assert "never produced an acknowledgement at all" in emitted_check.detail


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
    checks, _ = evaluate_transport(events, callee_turn_end_s=1.0, spoken_acks=[], phrases=POOL)
    dropped = by_id(checks)["acks_reached_the_callee"]
    assert dropped.verdict == "fail"
    assert dropped.measured_s == 2.0
    assert "Vapi-side" in dropped.detail


# --- attribution WITH the adapter's record: evidence, not inference ----------------
#
# `GET /debug/acks/{call_ref}` is the adapter's own record of every acknowledgement it
# emitted, and the channel each went out on. A spoken holding phrase that is not in it
# was not written here -- which is the one discrimination the transport cannot make.


def test_a_spoken_pool_phrase_with_no_adapter_emission_is_named_model_authored() -> None:
    """THE regression this endpoint exists to catch. The callee heard a holding phrase
    from the adapter's own pool, and the adapter's own complete record says it never
    sent it -- so the model wrote it, in defiance of the prohibition PR #10 put in
    VOICE_SYSTEM_PROMPT. A model-authored holding phrase is indistinguishable to the
    person on the line, so it defeats the call-global cooldown from the only viewpoint
    that counts. This must be a loud, named FAILURE, not a note.
    """
    record = AdapterAcks(acks=(emitted("One moment while I check.", channel="control"),))
    checks, _ = evaluate_transport(
        events=[],
        callee_turn_end_s=5.0,
        spoken_acks=[
            spoken("One moment while I check.", 6.5),
            spoken("Let me pull that up for you.", 11.0, index=1),
        ],
        phrases=POOL,
        adapter=record,
    )
    attribution = by_id(checks)["ack_attribution"]
    assert attribution.verdict == "fail"
    assert "MODEL-AUTHORED" in attribution.detail
    assert "Let me pull that up for you." in attribution.detail
    assert "VOICE_SYSTEM_PROMPT" in attribution.detail
    assert attribution.measured_s == 1.0
    # The one the adapter DID send is not accused.
    assert attribution.detail.count("at 11.000s") == 1
    assert "at 6.500s" not in attribution.detail


def test_every_spoken_ack_in_the_adapter_record_passes_with_its_channel() -> None:
    """The healthy case, and the one the transport alone could never report: a
    channel=control acknowledgement leaves nothing at all on model.url, so before the
    record existed this was indistinguishable from the failure above.
    """
    record = AdapterAcks(
        acks=(
            emitted("One moment while I check.", channel="control", elapsed_ms=1130),
            emitted("Let me pull that up for you.", channel="stream", elapsed_ms=305),
        )
    )
    checks, _ = evaluate_transport(
        events=[],
        callee_turn_end_s=5.0,
        spoken_acks=[
            spoken("One moment while I check.", 6.5),
            spoken("Let me pull that up for you.", 18.0, index=1),
        ],
        phrases=POOL,
        adapter=record,
    )
    checks_by_id = by_id(checks)
    attribution = checks_by_id["ack_attribution"]
    assert attribution.verdict == "pass"
    assert attribution.measured_s == 0.0
    assert "1 channel=control" in attribution.detail
    assert "1 channel=stream" in attribution.detail
    assert "UNKNOWN" not in attribution.detail
    # r2_ack_emitted stops claiming nothing was emitted, and says which channel and
    # what the adapter's own elapsed_ms was, without scoring a cross-host interval.
    emitted_check = checks_by_id["r2_ack_emitted"]
    assert emitted_check.verdict == "skip"
    assert "channel=stream" in emitted_check.detail
    assert "elapsed_ms=305" in emitted_check.detail
    assert emitted_check.measured_s is None


def test_an_ack_flushed_ahead_of_content_still_matches_its_emission() -> None:
    """The adapter's phrase arrives inside a longer spoken message when it is flushed
    ahead of the answer. Same tolerance ``find_acks`` uses to call it an ack at all.
    """
    record = AdapterAcks(acks=(emitted("One moment while I check.", channel="stream"),))
    attributions, orphans = attribute_acks(
        [spoken("One moment while I check. The vet prescribed gabapentin.", 6.0)], record
    )
    assert orphans == []
    assert attributions[0].emitted is not None
    assert attributions[0].emitted.channel == "stream"


def test_an_emission_nothing_spoke_fails_on_either_channel() -> None:
    """A channel=control acknowledgement that never became audio is invisible to this
    transport -- the adapter's record is the only way to see it dropped at all.
    """
    record = AdapterAcks(acks=(emitted("One moment while I check.", channel="control"),))
    checks, _ = evaluate_transport(
        events=[], callee_turn_end_s=5.0, spoken_acks=[], phrases=POOL, adapter=record
    )
    dropped = by_id(checks)["acks_reached_the_callee"]
    assert dropped.verdict == "fail"
    assert dropped.measured_s == 1.0
    assert "channel=control" in dropped.detail
    assert "Vapi-side" in dropped.detail


def test_an_empty_adapter_record_makes_a_spoken_holding_phrase_conclusive() -> None:
    """The purest form of the regression: the adapter drove the turn and emitted
    nothing at all (fast answers, or the cooldown refusing) and the callee still heard
    a holding phrase. Recorded live in speech.py's own comment.
    """
    checks, _ = evaluate_transport(
        events=[],
        callee_turn_end_s=5.0,
        spoken_acks=[spoken("One moment while I check.", 9.5)],
        phrases=POOL,
        adapter=AdapterAcks(acks=()),
    )
    checks_by_id = by_id(checks)
    assert checks_by_id["ack_attribution"].verdict == "fail"
    assert "MODEL-AUTHORED" in checks_by_id["ack_attribution"].detail
    emitted_check = checks_by_id["r2_ack_emitted"]
    assert emitted_check.verdict == "fail"
    assert "record for this call is EMPTY" in emitted_check.detail


def test_an_incomplete_adapter_record_may_not_accuse_anyone() -> None:
    """The adapter's ring lost entries for this call, so a phrase missing from it may
    simply be one of the lost ones. Refusing to accuse on incomplete evidence is the
    same discipline as refusing to infer channel=control from a spoken surplus.
    """
    record = AdapterAcks(acks=(emitted("One moment while I check.", channel="control"),), dropped=2)
    checks, _ = evaluate_transport(
        events=[],
        callee_turn_end_s=5.0,
        spoken_acks=[
            spoken("One moment while I check.", 6.5),
            spoken("Let me pull that up for you.", 20.0, index=1),
        ],
        phrases=POOL,
        adapter=record,
    )
    attribution = by_id(checks)["ack_attribution"]
    assert attribution.verdict == "skip"
    assert "UNKNOWN" in attribution.detail
    assert "INCOMPLETE" in attribution.detail
    assert "MODEL-AUTHORED" not in attribution.detail


def test_an_unavailable_record_falls_back_to_unknown_and_says_why() -> None:
    """An unreachable diagnostic surface must never become a verdict of its own."""
    checks, _ = evaluate_transport(
        events=[],
        callee_turn_end_s=5.0,
        spoken_acks=[spoken("One moment while I check.", 9.5)],
        phrases=POOL,
        adapter=AdapterAcks(unavailable="the tunnel rotated"),
    )
    attribution = by_id(checks)["ack_attribution"]
    assert attribution.verdict == "skip"
    assert "UNKNOWN" in attribution.detail
    assert "the tunnel rotated" in attribution.detail
    assert "MODEL-AUTHORED" not in attribution.detail


def test_the_adapter_record_exposes_phrase_pool_drift_authoritatively() -> None:
    """The record names the phrases the DEPLOYED adapter actually used, so a stale
    --ack-phrases-file stops being able to hide as "no acknowledgement".
    """
    record = AdapterAcks(acks=(emitted("Hang on a tick, I'll look.", channel="control"),))
    _checks, notes = evaluate_transport(
        events=[], callee_turn_end_s=5.0, spoken_acks=[], phrases=POOL, adapter=record
    )
    assert any("Hang on a tick" in note for note in notes)
    assert any("VHV_FILLER_PHRASES has drifted" in note for note in notes)


def test_alignment_is_reported_but_never_scored() -> None:
    """The adapter's wall clock can be placed on this harness's timeline, and that
    alignment is only as good as two hosts' NTP agreement -- so it is printed and no
    verdict rests on it.
    """
    record = AdapterAcks(
        acks=(emitted("One moment while I check.", channel="control", at_epoch_s=1_000_007.5),)
    )
    checks, _ = evaluate_transport(
        events=[],
        callee_turn_end_s=5.0,
        spoken_acks=[spoken("One moment while I check.", 6.5)],
        phrases=POOL,
        adapter=record,
        harness_epoch_origin_s=1_000_000.0,
    )
    emitted_check = by_id(checks)["r2_ack_emitted"]
    assert "7.500s on this harness's clock" in emitted_check.detail
    assert "not scored" in emitted_check.detail
    assert emitted_check.measured_s is None


# --- R1 transport verifiability -----------------------------------------------------


def test_websocket_call_reports_r1_as_unverifiable_not_fail_or_pass() -> None:
    """The live evidence: r1_deadline FAIL 5.107s on a vapi.websocket call, where the
    reason-for-calling fast path (gated on call.type == "outboundPhoneCall") can never
    fire. Neither a silent pass nor a fail -- its own explicit, disciplined check.
    """

    call = {"type": "vapi.websocketCall", "messages": []}
    report = Report(checks=[], turns=[], utterances=[], duration_unit="ms")
    check = r1_transport_scope(call, report)
    assert check.verdict == "skip"
    assert "UNVERIFIABLE-BY-THIS-TRANSPORT" in check.detail
    assert "outboundPhoneCall" in check.detail


def test_outbound_phone_call_reports_r1_as_measurable() -> None:

    call = {"type": "outboundPhoneCall", "messages": []}
    report = Report(checks=[], turns=[], utterances=[], duration_unit="ms")
    check = r1_transport_scope(call, report)
    assert check.verdict == "pass"


def test_max_turn_gap_dominance_by_the_r1_turn_is_called_out() -> None:
    """When the worst turn max_turn_gap ever measures IS the unverifiable R1 turn, say
    so explicitly, so that FAIL is not mistaken for evidence of an R1 regression.
    """

    user = Utterance(0, "user", "Hello?", 0.0, 0.1)
    reply = Utterance(1, "bot", "Hi there, how can I help?", 5.207, 6.0)
    turn = Turn(user=user, reply=reply)
    report = Report(checks=[], turns=[turn], utterances=[user, reply], duration_unit="ms")
    check = r1_transport_scope({"type": "vapi.websocketCall"}, report)
    assert "max_turn_gap above is dominated" in check.detail
