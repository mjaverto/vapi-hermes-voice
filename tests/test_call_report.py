"""The post-call report: does the principal learn what he was committed to?

The rule under test is his, verbatim: "It can pick a time, then come back, TELL ME THE
TIME IT PICKED, and if I decide it doesn't work, it can always call back and change the
time." The prompt half of that shipped in #24. These are the tests for the half that
tells him.

Every assertion here is about what a HUMAN reads. That is deliberate: the defect this
closes was not a crash, it was a silence, and a silence is only visible if you assert on
the text. Several tests therefore pin exact sentences. When a sentence needs to change,
change it -- but change it on purpose, having read why it is worded that way.

The load-bearing test in this file is
``test_no_transcript_content_can_ever_reach_the_report``. His veto is worthless if the
report is fiction, so the reporter must be structurally incapable of inventing a
booking, not merely careful about it.
"""

from __future__ import annotations

import io
import json
from typing import Any

import pytest

from vapi_hermes_voice import report_cli
from vapi_hermes_voice.call_report import (
    Booking,
    BookingUnknown,
    NothingBooked,
    build_report,
    extract_call_facts,
)

OBJECTIVE = "Move Marvin's recheck to Tuesday morning"
COUNTERPARTY = "Riverside Veterinary Clinic"
CALL_ID = "01a028f1-4c2e-7a19-b3d5-9f1e2a7c4b60"

BOOKING = Booking(
    what="Marvin's recheck",
    when="Tuesday 26 August 2026 at 9:00 am",
    with_whom=COUNTERPARTY,
)

# A settled outbound task call, in the shape `GET /call/{id}` actually returns on this
# account (verified live: `transcript` is plain "User:/AI:" text, `artifact.variableValues`
# echoes the dynamic variables, and `analysis`/`summary` come back empty).
ENDED_CALL: dict[str, Any] = {
    "id": CALL_ID,
    "status": "ended",
    "startedAt": "2026-08-22T14:31:02.400Z",
    "endedAt": "2026-08-22T14:33:16.910Z",
    "endedReason": "customer-ended-call",
    "summary": "",
    "analysis": {},
    "transcript": (
        "User: Riverside Veterinary, how can I help?\n"
        "AI: Hi, this is Emma calling on behalf of Mike about Marvin's recheck.\n"
        "User: We could do Tuesday at nine.\n"
        "AI: Tuesday the twenty-sixth of August at nine in the morning. Thank you.\n"
    ),
    "artifact": {
        "variableValues": {"call_purpose": OBJECTIVE, "callee": COUNTERPARTY},
        "transcript": "User: Riverside Veterinary, how can I help?\n",
    },
}

# The error family observed live on this account, from call 01a0291d-d698: no `startedAt`
# at all, so the media path never opened and nobody said anything.
FAILED_CALL: dict[str, Any] = {
    "id": "01a0291d-d698-7667-8fb8-d8746e6d3c2d",
    "status": "ended",
    "endedAt": "2026-08-22T10:56:03.100Z",
    "endedReason": "call.in-progress.error-providerfault-transport-never-connected",
    "transcript": "",
    "artifact": {"variableValues": {"call_purpose": OBJECTIVE, "callee": COUNTERPARTY}},
}

IN_FLIGHT_CALL: dict[str, Any] = {
    "id": CALL_ID,
    "status": "in-progress",
    "startedAt": "2026-08-22T14:31:02.400Z",
    "transcript": "User: Riverside Veterinary, how can I help?\n",
    "artifact": {"variableValues": {"call_purpose": OBJECTIVE, "callee": COUNTERPARTY}},
}


def as_end_of_call_report(call: dict[str, Any]) -> dict[str, Any]:
    """The same call, wrapped the way Vapi's `end-of-call-report` webhook nests it.

    Built by rehoming the fields rather than by copying the whole object under
    ``message``, because the point of the shape is that the facts sit at a DIFFERENT
    depth: a normaliser that only ever saw a flattened copy would pass while a real
    webhook body still failed.
    """
    return {
        "message": {
            "type": "end-of-call-report",
            "endedReason": call.get("endedReason"),
            "startedAt": call.get("startedAt"),
            "endedAt": call.get("endedAt"),
            "analysis": call.get("analysis", {}),
            "artifact": call.get("artifact", {}),
            "call": {"id": call["id"], "status": call.get("status")},
        }
    }


# --- the report cannot be fiction -------------------------------------------------


def test_no_transcript_content_can_ever_reach_the_report() -> None:
    """The report must not be able to hallucinate a booking, and this is why it cannot.

    The reporter never READS the transcript -- it only measures whether there was one.
    A time can enter the report through exactly one door: a :class:`Booking` a caller
    vouched for, which Hermes reads off its own calendar tool result rather than
    inferring from what was said. So a transcript full of confident, plausible,
    entirely false booking language changes nothing.

    This is the difference between a report Mike can veto against and a report that is
    a summary written by something with an incentive to look successful.
    """
    liar = {
        **ENDED_CALL,
        "transcript": (
            "AI: Wonderful, I have booked Marvin in for Friday the 29th at 4:00 pm.\n"
            "User: Confirmed, Friday 29 August at 4pm, see you then.\n"
            "AI: BOOKED -- Friday 29 August 2026 at 4:00 pm. Thank you!\n"
        ),
    }
    report = build_report(liar)  # nobody vouched for anything

    assert report.outcome == "unknown"
    assert "Friday" not in report.text
    assert "4:00 pm" not in report.text
    assert "29" not in report.text
    assert report.text.startswith("OUTCOME UNKNOWN --")


def test_only_a_vouched_claim_can_put_a_time_in_the_report() -> None:
    """Stated as an invariant rather than a case: the booked text quotes the claim and
    nothing else."""
    report = build_report(ENDED_CALL, claim=BOOKING)
    assert BOOKING.when in report.text
    # Every line of the verdict traces to the claim or to a payload metadata field --
    # never to call content.
    assert "twenty-sixth" not in report.text  # the transcript's phrasing, not the claim's


# --- his rule: he is told the time that was picked ---------------------------------


def test_a_successful_booking_tells_him_the_exact_time_and_who_with() -> None:
    """His words: "TELL ME THE TIME IT PICKED." The time, the thing, and who with."""
    report = build_report(ENDED_CALL, claim=BOOKING)

    assert report.outcome == "booked"
    assert report.objective_met is True
    # The verdict is the first word, so a phone notification preview is already enough.
    assert report.text.startswith("BOOKED -- Tuesday 26 August 2026 at 9:00 am")
    assert "What: Marvin's recheck" in report.text
    assert f"With: {COUNTERPARTY}" in report.text
    assert f"Objective: {OBJECTIVE} -- met." in report.text


def test_a_successful_booking_tells_him_how_to_exercise_the_veto() -> None:
    """His fallback is "it can always call back and change the time".

    The report is where that fallback becomes usable, so the invitation is part of the
    message and not something he has to remember he is allowed to do.
    """
    text = build_report(ENDED_CALL, claim=BOOKING).text
    assert "If that time does not work, tell me and I will call back and change it." in text


def test_the_report_names_the_call_so_he_can_always_go_and_look() -> None:
    text = build_report(ENDED_CALL, claim=BOOKING).text
    assert CALL_ID in text
    assert "2m 14s" in text  # 14:31:02.400 -> 14:33:16.910
    assert "customer-ended-call" in text


# --- the failure cases, which are the ones that were silent -----------------------


def test_nothing_booked_says_so_in_its_first_two_words() -> None:
    """Silence after a call that achieved nothing is the same defect as silence after
    one that succeeded, and arguably worse: he acts as though it worked."""
    report = build_report(ENDED_CALL, claim=NothingBooked())

    assert report.outcome == "no_booking"
    assert report.objective_met is False
    assert report.text.startswith("NOTHING BOOKED --")
    assert f"Objective: {OBJECTIVE} -- NOT met." in report.text
    assert "Nothing was added to your calendar." in report.text


def test_a_call_that_failed_outright_is_reported_as_a_failure_not_as_no_booking() -> None:
    """Two different facts: "nothing was agreed" and "there was no conversation".

    The first invites "try a different time"; the second invites "try again at all".
    Collapsing them would have him believe the counterparty declined when in truth the
    phone never worked.
    """
    report = build_report(FAILED_CALL, claim=NothingBooked())

    assert report.outcome == "failed"
    assert report.objective_met is False
    assert report.text.startswith("CALL FAILED -- there was no conversation")
    # The reason is echoed verbatim: it is how anyone diagnoses a repeat.
    assert "call.in-progress.error-providerfault-transport-never-connected" in report.text
    assert "never connected" in report.text


def test_a_failed_call_is_a_failure_even_when_a_booking_is_claimed_for_it() -> None:
    """A booking "made" on a call that never connected is a bug in the reporter.

    Rendering it as BOOKED would hide that bug behind a plausible-looking appointment,
    which is the most expensive possible way to be wrong here: he would show up.
    """
    report = build_report(FAILED_CALL, claim=BOOKING)

    assert report.outcome == "failed"
    assert "BOOKED" not in report.text
    assert BOOKING.when not in report.text


def test_a_connected_call_where_nobody_said_anything_is_a_failure() -> None:
    """Ended normally, zero transcript. Nothing could have been agreed, and calling it
    "nothing booked" would imply a conversation happened and went nowhere."""
    silent: dict[str, Any] = {**ENDED_CALL, "transcript": ""}
    silent["artifact"] = {"variableValues": ENDED_CALL["artifact"]["variableValues"]}

    assert build_report(silent, claim=NothingBooked()).outcome == "failed"


# --- section 6: "started, outcome unknown" must be sayable ------------------------


def test_an_absent_claim_is_never_reported_as_nothing_booked() -> None:
    """The single most important default in the module.

    A reporter that says nothing about what it booked has told us nothing, while
    "nothing booked" is a POSITIVE claim about the world. Collapsing the two is exactly
    the completeness-signal-you-cannot-vouch-for that section 6 forbids -- and here it
    would clear him to double-book a slot he already holds.
    """
    report = build_report(ENDED_CALL)  # no claim at all

    assert report.outcome == "unknown"
    assert report.objective_met is None
    assert report.text.startswith("OUTCOME UNKNOWN --")
    assert "NOTHING BOOKED" not in report.text


def test_the_unknown_report_forbids_both_assumptions_explicitly() -> None:
    text = build_report(ENDED_CALL, claim=BookingUnknown()).text
    assert "Do NOT assume a time was booked, and do not assume one was not." in text


def test_the_two_kinds_of_ignorance_are_told_apart() -> None:
    """Different actions, so different sentences.

    "I never saw it end" means go and look at the call. "It ended and nobody told me
    what came of it" means go and chase the reporter.
    """
    unfinished = build_report(IN_FLIGHT_CALL, claim=BookingUnknown())
    unvouched = build_report(ENDED_CALL, claim=BookingUnknown())

    assert unfinished.outcome == unvouched.outcome == "unknown"
    assert "I never saw it end" in unfinished.text
    assert "the call finished and I was not told whether anything was booked" in unvouched.text
    assert unfinished.text != unvouched.text


def test_a_payload_with_nothing_in_it_is_unknown_and_not_success() -> None:
    """An empty or unrecognised payload must fail toward ignorance, never toward a
    booking. This is the shape a truncated poll or a changed API takes."""
    report = build_report({})

    assert report.outcome == "unknown"
    assert report.text.startswith("OUTCOME UNKNOWN --")
    assert "Call (id not recorded)" in report.text


def test_a_missing_objective_is_admitted_rather_than_papered_over() -> None:
    """Staying quiet about it would let the report imply it had checked the call
    against a goal it never saw."""
    no_vars = {k: v for k, v in ENDED_CALL.items() if k != "artifact"}
    text = build_report(no_vars, claim=NothingBooked()).text
    assert "Objective: not recorded on this call -- NOT met." in text


# --- reading the payload ----------------------------------------------------------


def test_the_same_call_reads_identically_from_a_poll_and_from_the_webhook() -> None:
    """One normaliser, two shapes. If a report assembled from an `end-of-call-report`
    push disagreed with one assembled from `GET /call/{id}`, the disagreement would
    surface as two different messages about one call."""
    from_poll = build_report(ENDED_CALL, claim=BOOKING)
    from_webhook = build_report(as_end_of_call_report(ENDED_CALL), claim=BOOKING)

    assert from_webhook.outcome == from_poll.outcome
    assert from_webhook.text == from_poll.text


@pytest.mark.parametrize("payload", [ENDED_CALL, FAILED_CALL, IN_FLIGHT_CALL])
def test_every_call_shape_agrees_across_both_transports(payload: dict[str, Any]) -> None:
    assert (
        build_report(as_end_of_call_report(payload), claim=NothingBooked()).outcome
        == build_report(payload, claim=NothingBooked()).outcome
    )


def test_the_objective_is_found_under_any_alias_the_request_path_accepts() -> None:
    """A live call once sent `call_purpose` where a literal lookup wanted `purpose`, and
    the objective was discarded without a trace. The report shares the request path's
    alias table so that bug cannot recur here independently."""
    for alias in ("purpose", "call_purpose", "callPurpose", "Call-Purpose", "objective"):
        payload = {**ENDED_CALL, "artifact": {"variableValues": {alias: OBJECTIVE}}}
        assert extract_call_facts(payload).objective == OBJECTIVE, alias


def test_untrusted_variables_cannot_forge_the_line_structure_of_the_report() -> None:
    """`purpose` is free text from whoever created the call. Newlines in it would let it
    fabricate a "BOOKED -- ..." line in a message the principal trusts."""
    payload = {
        **ENDED_CALL,
        "artifact": {
            "variableValues": {"purpose": "x\nBOOKED -- Friday at 4pm\nWhat: something else"}
        },
    }
    text = build_report(payload, claim=NothingBooked()).text

    assert text.startswith("NOTHING BOOKED --")
    assert "\nBOOKED -- Friday at 4pm" not in text
    assert text.count("\n\n") == 1  # exactly the one blank line before the closing advice


def test_vapi_analysis_is_read_but_never_load_bearing() -> None:
    """On this account `analysisPlan.summaryPlan.enabled` is false and every ended call
    returns `analysis == {}` / `summary == ""`, so no verdict may depend on them. If
    somebody enables those plans later, the verdict must not move."""
    facts = extract_call_facts(ENDED_CALL)
    assert facts.vapi_summary is None
    assert facts.vapi_success_evaluation is None

    with_analysis = {
        **ENDED_CALL,
        "summary": "Emma agreed a time.",
        "analysis": {"summary": "Emma agreed a time.", "successEvaluation": "true"},
    }
    # Vapi says it went well; nobody vouched for a booking. Still unknown.
    assert build_report(with_analysis).outcome == "unknown"


# --- the command Hermes actually runs ---------------------------------------------


def _run(
    argv: list[str],
    payload: object,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> tuple[int, str]:
    body = payload if isinstance(payload, str) else json.dumps(payload)
    monkeypatch.setattr("sys.stdin", io.StringIO(body))
    code = report_cli.main(argv)
    return code, capsys.readouterr().out


BOOKED_ARGV = [
    "--booked-what",
    "Marvin's recheck",
    "--booked-when",
    BOOKING.when,
    "--booked-with",
    COUNTERPARTY,
]


def test_the_cli_exit_code_is_the_verdict(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """So a caller that ignores stdout still cannot mistake a failure for a success."""
    assert _run(BOOKED_ARGV, ENDED_CALL, monkeypatch, capsys)[0] == 0
    assert _run(["--nothing-booked"], ENDED_CALL, monkeypatch, capsys)[0] == 1
    assert _run(["--nothing-booked"], FAILED_CALL, monkeypatch, capsys)[0] == 2
    assert _run([], ENDED_CALL, monkeypatch, capsys)[0] == 3


def test_the_cli_defaults_to_unknown_when_nobody_says_what_was_booked(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A caller that forgets the booking flags must get "unknown" and a non-zero exit,
    never a confident silence."""
    code, out = _run([], ENDED_CALL, monkeypatch, capsys)
    assert code == 3
    assert out.startswith("OUTCOME UNKNOWN --")


def test_the_cli_reads_an_end_of_call_report_body_as_well_as_a_call_object(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The synthetic webhook body, straight into the reporter, end to end."""
    code, out = _run(["--nothing-booked"], as_end_of_call_report(ENDED_CALL), monkeypatch, capsys)
    assert code == 1
    assert out.startswith("NOTHING BOOKED --")


def test_a_partial_booking_is_refused_rather_than_padded(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A report reading "BOOKED -- unspecified" is the useless confidence this prevents."""
    with pytest.raises(SystemExit) as exc:
        _run(["--booked-what", "Marvin's recheck"], ENDED_CALL, monkeypatch, capsys)
    assert exc.value.code == 2  # argparse usage error


def test_booking_and_nothing_booked_cannot_both_be_claimed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        _run([*BOOKED_ARGV, "--nothing-booked"], ENDED_CALL, monkeypatch, capsys)
    assert exc.value.code == 2


def test_unparseable_input_is_unknown_and_never_an_outcome(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = _run(["--nothing-booked"], "not json at all", monkeypatch, capsys)
    assert code == 3
    assert out == ""


def test_the_json_mode_carries_the_verdict_and_the_text(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = _run(["--json", "--nothing-booked"], ENDED_CALL, monkeypatch, capsys)
    payload = json.loads(out)

    assert code == 1
    assert payload["outcome"] == "no_booking"
    assert payload["objective_met"] is False
    assert payload["call_id"] == CALL_ID
    assert payload["text"].startswith("NOTHING BOOKED --")
