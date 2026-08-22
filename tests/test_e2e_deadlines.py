"""Deterministic cover for the live harness's arithmetic (``tests/e2e/deadlines.py``).

The live harness itself is excluded from this suite: it places real Vapi calls. Its
*scoring* is not, because a timing harness whose maths is wrong is worse than no harness
-- it manufactures confidence. Every fixture here is either a recorded live call or a
transcription of a reported failure, so these tests fail if the analysis stops catching
the bugs that actually happened.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from e2e.deadlines import (
    Budgets,
    TimelineUnitError,
    Utterance,
    carries_flush_token,
    duration_anomalies,
    evaluate,
    evaluate_transport,
    find_acks,
    load_utterances,
    normalize_phrase,
    pair_turns,
    render_table,
)

FIXTURES = Path(__file__).parent / "e2e" / "fixtures"

# The pool the adapter ships with (config._DEFAULT_FILLER_PHRASES). Duplicated rather
# than imported so that changing the adapter's phrases cannot silently change what these
# tests assert about matching behaviour.
POOL = [
    "I have that information right here, give me a second.",
    "Let me pull that up for you.",
    "One moment while I check.",
    "Just a second, looking now.",
    "Give me a moment to find that.",
    "Hold on, checking that for you.",
    "Let me take a quick look.",
    "Bear with me one second.",
]

STATIC_FIRST_MESSAGE = (
    "Hi, this is Emma, an AI assistant calling on behalf of Mike Averto. I am calling about a"
    " medical follow-up for him and I have his details in front of me. Is this a good moment?"
)


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def verdicts(report) -> dict[str, str]:
    return {c.id: c.verdict for c in report.checks}


def measured(report, check_id: str) -> float | None:
    return next(c.measured_s for c in report.checks if c.id == check_id)


# --- the millisecond trap --------------------------------------------------------


def test_duration_is_read_as_milliseconds_from_real_call() -> None:
    """Vapi documents ``duration`` in seconds and emits milliseconds.

    Recorded live: a user message spanning ``endTime - time == 74`` ms carries
    ``duration: 74``. Read as seconds it would put the end of a 1.286 s utterance at
    75.3 s, making every later gap negative and every deadline vacuously fine.
    """
    utterances, unit = load_utterances(load("call_static_first_message.json"))
    assert unit == "ms"
    user = next(u for u in utterances if u.role == "user")
    assert user.start_s == pytest.approx(1.286)
    assert user.end_s == pytest.approx(1.360, abs=1e-3)
    assert user.duration_s == pytest.approx(0.074, abs=1e-4)


def test_seconds_unit_is_also_recognised() -> None:
    """If Vapi ever matches its own docs, the harness must follow rather than break."""
    call = {
        "messages": [
            {
                "role": "user",
                "message": "Hello?",
                "time": 1_000_000,
                "endTime": 1_000_520,
                "secondsFromStart": 1.0,
                "duration": 0.52,
            },
            {
                "role": "bot",
                "message": "Hi.",
                "time": 1_001_400,
                "endTime": 1_003_400,
                "secondsFromStart": 2.4,
                "duration": 2.0,
            },
        ]
    }
    utterances, unit = load_utterances(call)
    assert unit == "s"
    assert utterances[0].end_s == pytest.approx(1.52)


def test_ambiguous_units_are_a_hard_error_not_a_guess() -> None:
    call = {
        "messages": [
            {
                "role": "user",
                "message": "a",
                "time": 0,
                "endTime": 100,
                "secondsFromStart": 0,
                "duration": 100,
            },
            {
                "role": "bot",
                "message": "b",
                "time": 200,
                "endTime": 2200,
                "secondsFromStart": 0.2,
                "duration": 2.0,
            },
        ]
    }
    with pytest.raises(TimelineUnitError, match="disagree on duration's unit"):
        load_utterances(call)


def test_timeline_where_nothing_decides_the_unit_is_a_hard_error() -> None:
    call = {
        "messages": [
            {
                "role": "user",
                "message": "a",
                "time": 0,
                "endTime": 1000,
                "secondsFromStart": 0,
                "duration": 17,
            },
        ]
    }
    with pytest.raises(TimelineUnitError, match="matches its own wall clock in either unit"):
        load_utterances(call)


def test_one_undecidable_message_does_not_veto_a_scorable_call() -> None:
    """Throwing away a whole billable call over one odd row is the wrong trade."""
    call = {
        "messages": [
            {
                "role": "user",
                "message": "Hello?",
                "time": 1000,
                "endTime": 1500,
                "secondsFromStart": 1.0,
                "duration": 500,
            },
            {
                "role": "bot",
                "message": "nonsense row",
                "time": 3000,
                "endTime": 4000,
                "secondsFromStart": 3.0,
                "duration": 17,
            },
        ]
    }
    utterances, unit = load_utterances(call)
    assert unit == "ms"
    assert utterances[0].end_s == pytest.approx(1.5)


def test_timeline_with_no_derivable_unit_refuses_to_score() -> None:
    call = {"messages": [{"role": "user", "message": "a", "secondsFromStart": 0, "duration": 500}]}
    with pytest.raises(TimelineUnitError, match="cannot be derived"):
        load_utterances(call)


# --- the provenance trap ---------------------------------------------------------


def test_static_first_message_meets_r1_deadline_but_fails_provenance() -> None:
    """The deadline is met and the adapter is untested. Both facts must be reported.

    Recorded live with ``model.url`` reachable: 0.797 s from the callee's "Hello."
    ending to the assistant starting. The words spoken are Vapi's configured
    ``firstMessage``, so nothing about the adapter was measured.
    """
    report = evaluate(
        load("call_static_first_message.json"), phrases=POOL, first_message=STATIC_FIRST_MESSAGE
    )
    v = verdicts(report)
    assert v["r1_deadline"] == "pass"
    assert measured(report, "r1_deadline") == pytest.approx(0.797, abs=1e-3)
    assert v["r1_provenance"] == "fail"
    assert "static assistant.firstMessage" in next(
        c.detail for c in report.checks if c.id == "r1_provenance"
    )
    assert report.ok is False


def test_dead_adapter_still_passes_the_r1_deadline() -> None:
    """The regression this harness exists to refuse to miss.

    Recorded with ``model.url`` pointed at a guaranteed-404 endpoint: the assistant
    answered in 0.556 s anyway, because Vapi read the static ``firstMessage``. Deadline
    checks alone would have called that a healthy R1.
    """
    report = evaluate(
        load("call_dead_adapter_control.json"), phrases=POOL, first_message=STATIC_FIRST_MESSAGE
    )
    assert verdicts(report)["r1_deadline"] == "pass"
    assert measured(report, "r1_deadline") == pytest.approx(0.556, abs=1e-3)
    assert verdicts(report)["r1_provenance"] == "fail"


def test_provenance_passes_when_the_reply_is_not_the_static_string() -> None:
    report = evaluate(load("call_healthy.json"), phrases=POOL, first_message=STATIC_FIRST_MESSAGE)
    assert verdicts(report)["r1_provenance"] == "pass"


def test_transcription_noise_does_not_flip_provenance_to_a_false_pass() -> None:
    """Recorded live (call 01a025ee): Vapi spoke the static firstMessage and the
    transcript came back with "Mike Averdo" for "Mike Averto".

    ``messages[]`` records the assistant's speech as recognised, not the configured
    string, so exact matching turned the most important check in this harness into a
    false PASS on a one-letter difference.
    """
    report = evaluate(
        load("call_first_message_asr_variant.json"),
        phrases=POOL,
        first_message=STATIC_FIRST_MESSAGE,
    )
    check = next(c for c in report.checks if c.id == "r1_provenance")
    assert check.verdict == "fail"
    assert check.unit == "ratio"
    assert check.measured_s == pytest.approx(0.994, abs=2e-3)


def test_a_genuinely_different_reply_scores_far_below_the_threshold() -> None:
    """The threshold has to separate recognition noise from a different sentence."""
    report = evaluate(load("call_healthy.json"), phrases=POOL, first_message=STATIC_FIRST_MESSAGE)
    check = next(c for c in report.checks if c.id == "r1_provenance")
    assert check.verdict == "pass"
    assert check.measured_s is not None and check.measured_s < 0.85


def test_a_reply_cut_short_still_matches_the_string_it_was_reading() -> None:
    """A barge-in truncates the utterance; it is still the static firstMessage."""
    call = dict(load("call_static_first_message.json"))
    call["messages"] = [
        dict(m, message=STATIC_FIRST_MESSAGE[:60]) if m["role"] == "bot" else m
        for m in call["messages"]
    ]
    report = evaluate(call, phrases=POOL, first_message=STATIC_FIRST_MESSAGE)
    assert verdicts(report)["r1_provenance"] == "fail"


def test_provenance_can_be_downgraded_but_never_silently_passes() -> None:
    report = evaluate(
        load("call_static_first_message.json"),
        phrases=POOL,
        first_message=STATIC_FIRST_MESSAGE,
        require_adapter_provenance=False,
    )
    assert verdicts(report)["r1_provenance"] == "skip"


def test_no_static_first_message_means_the_adapter_answered() -> None:
    report = evaluate(load("call_healthy.json"), phrases=POOL, first_message=None)
    assert verdicts(report)["r1_provenance"] == "pass"


# --- R1: the Flux hold-open stall ------------------------------------------------


def test_flux_stall_fails_r1_and_the_gap_ceiling() -> None:
    """Live call 01a02524: 96 ms of "Hello?" ending at 3.33 s, Emma speaking at 16.92 s."""
    report = evaluate(load("call_flux_stall.json"), phrases=POOL, first_message=None)
    v = verdicts(report)
    assert v["r1_deadline"] == "fail"
    assert measured(report, "r1_deadline") == pytest.approx(13.590, abs=2e-3)
    assert v["max_turn_gap"] == "fail"
    assert report.ok is False


def test_failure_table_shows_where_the_time_went() -> None:
    report = evaluate(load("call_flux_stall.json"), phrases=POOL, first_message=None)
    table = render_table(report, phrases=POOL)
    assert "per-turn timeline" in table
    assert "16.920" in table  # when the assistant finally spoke
    assert "13.590" in table  # the gap the callee sat through
    assert "[FAIL]" in table


def test_repeated_greetings_are_scored_against_the_last_one() -> None:
    """Each repeat re-arms end-of-turn detection, so only the final wait is the stall."""
    report = evaluate(load("call_flux_stall.json"), phrases=POOL, first_message=None)
    gaps = [t.gap_s for t in report.turns]
    assert gaps[0] == pytest.approx(13.590, abs=2e-3)
    assert gaps[-1] == pytest.approx(16.920 - (7.012 + 1.180), abs=2e-3)


# --- R2: acknowledgement deadline, cooldown, storm -------------------------------


def test_healthy_call_passes_every_check() -> None:
    report = evaluate(load("call_healthy.json"), phrases=POOL, first_message=STATIC_FIRST_MESSAGE)
    assert verdicts(report) == {
        "replied": "pass",
        "r1_deadline": "pass",
        "r1_provenance": "pass",
        "r2_ack_deadline": "pass",
        "r2_ack_cooldown": "pass",
        "r2_ack_storm": "pass",
        "max_turn_gap": "pass",
    }
    assert report.ok is True
    assert measured(report, "r2_ack_deadline") == pytest.approx(0.930, abs=2e-3)


def test_ack_storm_fails_cooldown_and_storm_checks() -> None:
    """Six acknowledgements in 16 s -- the reported barge-in retry failure."""
    report = evaluate(load("call_ack_storm.json"), phrases=POOL, first_message=None)
    v = verdicts(report)
    assert len(report.acks) == 6
    assert v["r2_ack_deadline"] == "pass"  # the first one was prompt
    assert v["r2_ack_cooldown"] == "fail"
    assert measured(report, "r2_ack_cooldown") == pytest.approx(2.6, abs=1e-3)
    assert v["r2_ack_storm"] == "fail"
    assert measured(report, "r2_ack_storm") == 6.0


def test_cooldown_is_global_not_per_turn() -> None:
    """Two acknowledgements 4 s apart on *different* callee turns is still a violation."""
    call = {
        "messages": [
            {
                "role": "user",
                "message": "Hello?",
                "time": 1000,
                "endTime": 1500,
                "secondsFromStart": 1.0,
                "duration": 500,
            },
            {
                "role": "bot",
                "message": "One moment while I check.",
                "time": 2000,
                "endTime": 3000,
                "secondsFromStart": 2.0,
                "duration": 1000,
            },
            {
                "role": "user",
                "message": "And the dose?",
                "time": 4000,
                "endTime": 4800,
                "secondsFromStart": 4.0,
                "duration": 800,
            },
            {
                "role": "bot",
                "message": "Let me take a quick look.",
                "time": 6000,
                "endTime": 7000,
                "secondsFromStart": 6.0,
                "duration": 1000,
            },
        ]
    }
    report = evaluate(call, phrases=POOL, first_message=None)
    assert verdicts(report)["r2_ack_cooldown"] == "fail"
    assert measured(report, "r2_ack_cooldown") == pytest.approx(4.0)


def test_ack_exactly_at_the_cooldown_boundary_passes() -> None:
    call = {
        "messages": [
            {
                "role": "user",
                "message": "Hello?",
                "time": 0,
                "endTime": 500,
                "secondsFromStart": 0.0,
                "duration": 500,
            },
            {
                "role": "bot",
                "message": "One moment while I check.",
                "time": 1000,
                "endTime": 2000,
                "secondsFromStart": 1.0,
                "duration": 1000,
            },
            {
                "role": "bot",
                "message": "Let me take a quick look.",
                "time": 11000,
                "endTime": 12000,
                "secondsFromStart": 11.0,
                "duration": 1000,
            },
        ]
    }
    report = evaluate(call, phrases=POOL, first_message=None, budgets=Budgets(max_turn_gap_s=99))
    assert verdicts(report)["r2_ack_cooldown"] == "pass"
    assert measured(report, "r2_ack_cooldown") == pytest.approx(10.0)


def test_late_acknowledgement_fails_the_deadline() -> None:
    """The other reported R2 failure: the holding line arriving ten seconds late."""
    call = {
        "messages": [
            {
                "role": "user",
                "message": "What medication did the vet prescribe Marvin?",
                "time": 5000,
                "endTime": 7400,
                "secondsFromStart": 5.0,
                "duration": 2400,
            },
            {
                "role": "bot",
                "message": "One moment while I check.",
                "time": 17600,
                "endTime": 19000,
                "secondsFromStart": 17.6,
                "duration": 1400,
            },
        ]
    }
    report = evaluate(call, phrases=POOL, first_message=None)
    assert verdicts(report)["r2_ack_deadline"] == "fail"
    assert measured(report, "r2_ack_deadline") == pytest.approx(10.2, abs=1e-3)


def test_unrecognised_reply_is_distinguished_from_no_reply() -> None:
    """A drifted phrase pool must not read as "the adapter said nothing"."""
    call = {
        "messages": [
            {
                "role": "user",
                "message": "What medication?",
                "time": 0,
                "endTime": 1000,
                "secondsFromStart": 0.0,
                "duration": 1000,
            },
            {
                "role": "bot",
                "message": "Righto, hang about a tick.",
                "time": 1500,
                "endTime": 2500,
                "secondsFromStart": 1.5,
                "duration": 1000,
            },
        ]
    }
    report = evaluate(call, phrases=POOL, first_message=None)
    detail = next(c.detail for c in report.checks if c.id == "r2_ack_deadline")
    assert verdicts(report)["r2_ack_deadline"] == "fail"
    assert "VHV_FILLER_PHRASES has drifted" in detail
    assert "Righto" in detail


# --- matching behaviour ----------------------------------------------------------


def test_flush_token_does_not_defeat_phrase_matching() -> None:
    """The adapter appends Vapi's ``<flush />`` control token to fillers."""
    utterances = [Utterance(0, "bot", "One moment while I check. <flush />", 1.0, 2.0)]
    assert find_acks(utterances, POOL) == utterances


def test_acknowledgement_prefixing_real_content_still_counts() -> None:
    utterances = [
        Utterance(0, "bot", "Let me take a quick look. The vet prescribed gabapentin.", 1.0, 4.0)
    ]
    assert find_acks(utterances, POOL) == utterances


def test_content_that_merely_contains_a_phrase_is_not_an_acknowledgement() -> None:
    text = "The vet said one moment while I check the chart, then hung up."
    utterances = [Utterance(0, "bot", text, 1.0, 4.0)]
    assert find_acks(utterances, POOL) == []


def test_user_utterances_are_never_acknowledgements() -> None:
    utterances = [Utterance(0, "user", "One moment while I check.", 1.0, 2.0)]
    assert find_acks(utterances, POOL) == []


def test_empty_phrase_pool_matches_nothing() -> None:
    utterances = [Utterance(0, "bot", "One moment while I check.", 1.0, 2.0)]
    assert find_acks(utterances, ["", "   "]) == []


def test_normalize_phrase_is_punctuation_and_case_insensitive() -> None:
    assert normalize_phrase("One MOMENT, while I check!!") == "one moment while i check"


# --- turn pairing ---------------------------------------------------------------


def test_assistant_speech_already_under_way_is_not_credited_as_a_reply() -> None:
    """A barge-in must not be scored as a sub-second response to the interruption."""
    utterances = [
        Utterance(0, "bot", "Long explanation that is still going.", 1.0, 9.0),
        Utterance(1, "user", "Wait, stop.", 4.0, 4.8),
        Utterance(2, "bot", "Sorry, go ahead.", 10.0, 11.0),
    ]
    turns = pair_turns(utterances)
    assert len(turns) == 1
    assert turns[0].reply is not None
    assert turns[0].reply.text == "Sorry, go ahead."
    assert turns[0].gap_s == pytest.approx(5.2)


def test_unanswered_turn_fails_the_replied_check() -> None:
    call = {
        "messages": [
            {
                "role": "user",
                "message": "Hello?",
                "time": 0,
                "endTime": 500,
                "secondsFromStart": 0.0,
                "duration": 500,
            },
            {
                "role": "bot",
                "message": "Hi there.",
                "time": 900,
                "endTime": 1900,
                "secondsFromStart": 0.9,
                "duration": 1000,
            },
            {
                "role": "user",
                "message": "What medication?",
                "time": 5000,
                "endTime": 6000,
                "secondsFromStart": 5.0,
                "duration": 1000,
            },
        ]
    }
    report = evaluate(call, phrases=POOL, first_message=None)
    assert verdicts(report)["replied"] == "fail"
    assert "What medication?" in next(c.detail for c in report.checks if c.id == "replied")


def test_out_of_order_messages_are_sorted_before_scoring() -> None:
    call = {
        "messages": [
            {
                "role": "bot",
                "message": "Hi there.",
                "time": 900,
                "endTime": 1900,
                "secondsFromStart": 0.9,
                "duration": 1000,
            },
            {
                "role": "user",
                "message": "Hello?",
                "time": 0,
                "endTime": 400,
                "secondsFromStart": 0.0,
                "duration": 400,
            },
        ]
    }
    utterances, _ = load_utterances(call)
    assert [u.role for u in utterances] == ["user", "bot"]
    assert verdicts(evaluate(call, phrases=POOL, first_message=None))["r1_deadline"] == "pass"


def test_abnormal_ended_reason_is_surfaced_as_a_note() -> None:
    call = dict(load("call_healthy.json"))
    call["endedReason"] = "pipeline-error-custom-llm-llm-failed"
    report = evaluate(call, phrases=POOL, first_message=None)
    assert any("pipeline-error" in n for n in report.notes)


def test_json_result_is_serialisable_and_carries_the_measurements() -> None:
    report = evaluate(load("call_ack_storm.json"), phrases=POOL, first_message=None)
    blob = json.loads(json.dumps(report.as_dict()))
    assert blob["ok"] is False
    assert blob["duration_unit"] == "ms"
    assert len(blob["acknowledgements"]) == 6
    assert {c["id"] for c in blob["checks"]} == {
        "replied",
        "r1_deadline",
        "r1_provenance",
        "r2_ack_deadline",
        "r2_ack_cooldown",
        "r2_ack_storm",
        "max_turn_gap",
    }
    storm = next(c for c in blob["checks"] if c["id"] == "r2_ack_storm")
    assert storm["unit"] == "count" and storm["measured"] == 6.0
    gap = next(c for c in blob["checks"] if c["id"] == "max_turn_gap")
    assert gap["unit"] == "seconds"


def test_budgets_are_honoured_rather_than_hardcoded() -> None:
    call = load("call_flux_stall.json")
    tolerant = evaluate(
        call,
        phrases=POOL,
        first_message=None,
        budgets=Budgets(reason_deadline_s=20.0, max_turn_gap_s=20.0),
    )
    assert verdicts(tolerant)["r1_deadline"] == "pass"
    assert verdicts(tolerant)["max_turn_gap"] == "pass"


def test_call_with_no_callee_speech_fails_rather_than_passing_vacuously() -> None:
    call = {
        "messages": [
            {
                "role": "bot",
                "message": "Hello, anyone there?",
                "time": 0,
                "endTime": 2000,
                "secondsFromStart": 0.0,
                "duration": 2000,
            },
        ]
    }
    report = evaluate(call, phrases=POOL, first_message=None)
    v = verdicts(report)
    assert v["r1_deadline"] == "fail"
    assert v["max_turn_gap"] == "fail"
    assert report.ok is False


def test_scenario_without_a_lookup_skips_the_acknowledgement_deadline() -> None:
    """flux_storm only says "Hello?", which warrants an answer, not a holding line."""
    report = evaluate(
        load("call_flux_stall.json"), phrases=POOL, first_message=None, ack_turn_index=None
    )
    assert verdicts(report)["r2_ack_deadline"] == "skip"
    assert verdicts(report)["r1_deadline"] == "fail"  # the stall is still caught


def test_out_of_range_ack_turn_fails_loudly() -> None:
    report = evaluate(load("call_healthy.json"), phrases=POOL, first_message=None, ack_turn_index=9)
    check = next(c for c in report.checks if c.id == "r2_ack_deadline")
    assert check.verdict == "fail"
    assert "no callee turn at index 9" in check.detail


def test_null_embedded_assistant_does_not_crash_provenance() -> None:
    """``GET /call/{id}`` returns ``"assistant": null`` on some calls."""
    call = dict(load("call_static_first_message.json"))
    call["assistant"] = None
    report = evaluate(call, phrases=POOL, first_message=STATIC_FIRST_MESSAGE)
    assert verdicts(report)["r1_provenance"] == "fail"
    assert "assistant-waits-for-user" in next(
        c.detail for c in report.checks if c.id == "r1_provenance"
    )


def test_embedded_assistant_mode_is_used_when_present() -> None:
    call = dict(load("call_static_first_message.json"))
    call["assistant"] = {"firstMessageMode": "assistant-speaks-first"}
    report = evaluate(call, phrases=POOL, first_message=STATIC_FIRST_MESSAGE)
    assert "assistant-speaks-first" in next(
        c.detail for c in report.checks if c.id == "r1_provenance"
    )


# --- transport-side attribution --------------------------------------------------


def _spoken_ack(text: str, at_s: float) -> Utterance:
    """One acknowledgement Vapi recorded itself as having spoken."""
    return Utterance(index=0, role="bot", text=text, start_s=at_s, end_s=at_s + 1.0)


def test_adapter_met_its_deadline_but_vapi_never_spoke_it() -> None:
    """Recorded live (call 01a025e5): the exact failure the spoken timeline mis-attributes.

    The callee asked a lookup question ending at 15.505 s. The adapter put
    "Okay, bear with me a moment. <flush />" on the wire at 17.554 s -- 2.049 s, over
    budget but in the right ballpark -- and Vapi turned neither that nor the second
    holding line into audio, then emitted ``hang``. Reading only what was spoken says
    "the adapter produced nothing", which sends the next person to the wrong component.
    """
    recorded = load("transport_ack_dropped_by_vapi.json")
    lookup = recorded["steps"][-1]
    checks, notes = evaluate_transport(
        recorded["events"],
        callee_turn_end_s=lookup["speech_end_s"],
        spoken_acks=[],
        spoken_utterances=[],
        phrases=POOL,
    )
    by_id = {c.id: c for c in checks}
    assert by_id["r2_ack_emitted"].measured_s == pytest.approx(2.049, abs=1e-3)
    dropped = by_id["acks_reached_the_callee"]
    assert dropped.verdict == "fail"
    assert dropped.measured_s == 2.0
    assert "Vapi-side" in dropped.detail


def test_drifted_deployed_phrase_pool_is_reported_not_hidden() -> None:
    """The recorded holding lines are not in the repo's default pool."""
    recorded = load("transport_ack_dropped_by_vapi.json")
    _, notes = evaluate_transport(
        recorded["events"],
        callee_turn_end_s=recorded["steps"][-1]["speech_end_s"],
        spoken_acks=[],
        spoken_utterances=[],
        phrases=POOL,
    )
    assert any("has drifted" in n for n in notes)
    assert any("bear with me a moment" in n for n in notes)


def test_no_drift_note_when_the_pool_matches() -> None:
    events = [
        {
            "at_s": 3.0,
            "event": {"type": "model-output", "output": "One moment while I check. <flush />"},
        },
    ]
    checks, notes = evaluate_transport(
        events,
        callee_turn_end_s=2.0,
        spoken_acks=[_spoken_ack("One moment while I check.", 3.5)],
        spoken_utterances=[_spoken_ack("One moment while I check.", 3.5)],
        phrases=POOL,
    )
    assert notes == []
    by_id = {c.id: c for c in checks}
    assert by_id["r2_ack_emitted"].verdict == "pass"
    assert by_id["r2_ack_emitted"].measured_s == pytest.approx(1.0)
    assert by_id["acks_reached_the_callee"].verdict == "pass"


def test_model_output_without_the_flush_token_is_content_not_a_holding_line() -> None:
    events = [
        {
            "at_s": 3.0,
            "event": {"type": "model-output", "output": "The vet prescribed gabapentin."},
        },
    ]
    checks, _ = evaluate_transport(
        events, callee_turn_end_s=2.0, spoken_acks=[], spoken_utterances=[], phrases=POOL
    )
    emitted = next(c for c in checks if c.id == "r2_ack_emitted")
    assert emitted.verdict == "fail"
    assert "never produced an acknowledgement at all" in emitted.detail


def test_holding_line_before_the_question_does_not_count() -> None:
    events = [
        {
            "at_s": 1.0,
            "event": {"type": "model-output", "output": "One moment while I check. <flush />"},
        },
    ]
    checks, _ = evaluate_transport(
        events,
        callee_turn_end_s=5.0,
        spoken_acks=[_spoken_ack("One moment while I check.", 1.5)],
        spoken_utterances=[_spoken_ack("One moment while I check.", 1.5)],
        phrases=POOL,
    )
    emitted = next(c for c in checks if c.id == "r2_ack_emitted")
    assert emitted.verdict == "fail"
    assert "all before the callee's question" in emitted.detail


def test_carries_flush_token_tolerates_spacing_variants() -> None:
    assert carries_flush_token("hold on <flush/>")
    assert carries_flush_token("hold on < flush />")
    assert carries_flush_token("hold on <FLUSH />")
    assert not carries_flush_token("hold on")
    assert not carries_flush_token("")


# --- duration divergence, from a real call ---------------------------------------


def test_bot_duration_diverging_from_wall_clock_still_scores() -> None:
    """Recorded live (call 01a025ea): duration=2955 over a 2228 ms span, 33 % apart.

    ``duration`` is the length of the synthesised audio and ``endTime - time`` is a pair
    of pipeline timestamps, so they legitimately disagree on assistant messages. An
    earlier, tighter unit check treated this as a changed timeline format and refused to
    score the call at all.
    """
    call = load("call_late_ack_duration_divergence.json")
    utterances, unit = load_utterances(call)
    assert unit == "ms"
    assert len(utterances) == 4
    report = evaluate(call, phrases=POOL, first_message=STATIC_FIRST_MESSAGE)
    assert verdicts(report)["r1_deadline"] == "pass"


def test_divergent_duration_is_reported_as_a_note() -> None:
    call = load("call_late_ack_duration_divergence.json")
    notes = duration_anomalies(call)
    assert len(notes) == 1
    assert "Sure. Give me a second" in notes[0]
    assert "33% divergence" in notes[0]
    assert notes[0] in evaluate(call, phrases=POOL, first_message=None).notes


def test_agreeing_durations_produce_no_notes() -> None:
    assert duration_anomalies(load("call_healthy.json")) == []


def test_the_live_late_acknowledgement_is_caught() -> None:
    """Same real call: the callee waited 14.17 s for "Sure. Give me a second"."""
    report = evaluate(
        load("call_late_ack_duration_divergence.json"),
        phrases=["Sure. Give me a second."],
        first_message=STATIC_FIRST_MESSAGE,
    )
    assert verdicts(report)["r2_ack_deadline"] == "fail"
    assert measured(report, "r2_ack_deadline") == pytest.approx(14.174, abs=2e-3)
    assert verdicts(report)["max_turn_gap"] == "fail"


def test_a_thousandfold_error_is_still_rejected() -> None:
    """The window is wide enough for real divergence and no wider."""
    call = {
        "messages": [
            {
                "role": "user",
                "message": "a",
                "time": 0,
                "endTime": 1000,
                "secondsFromStart": 0.0,
                "duration": 100_000,
            },
        ]
    }
    with pytest.raises(TimelineUnitError, match="neither"):
        load_utterances(call)


# --- did the script exercise what it was written to exercise? --------------------


def test_absorbed_repeats_are_reported_as_missing_coverage() -> None:
    """Live flux_storm run 01a025f1 spoke three greetings and Vapi saw one turn.

    Repeats two and three landed while the assistant was reading its firstMessage, so
    they were absorbed and no end-of-turn timer was ever re-armed. Reporting that run as
    a clean PASS would be false assurance about the exact bug the scenario exists for.
    """
    report = evaluate(
        load("call_healthy.json"), phrases=POOL, first_message=None, expected_callee_turns=3
    )
    check = next(c for c in report.checks if c.id == "script_coverage")
    assert check.verdict == "fail"
    assert check.measured_s == 2.0 and check.budget_s == 3.0
    assert "DID NOT TEST THE STALL" in check.detail


def test_full_coverage_passes() -> None:
    report = evaluate(
        load("call_healthy.json"), phrases=POOL, first_message=None, expected_callee_turns=2
    )
    assert verdicts(report)["script_coverage"] == "pass"


def test_coverage_check_is_absent_when_there_is_no_script() -> None:
    report = evaluate(load("call_healthy.json"), phrases=POOL, first_message=None)
    assert "script_coverage" not in verdicts(report)
