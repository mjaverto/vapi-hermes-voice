"""Standing scheduling conduct: the three rules dictated after live call ``01a028f1``.

The operator answered that call himself, role-playing a doctor's office, and gave three
rules plus one prior defect. His words, verbatim:

    One is that it interrupted me. And then the 2nd problem is that it's telling a
    doctor's office what events are on my schedule. It should never do that. All it
    needs to say is those times work or those times don't work. And last thing, it
    doesn't need to actually do anything else other than pick a time. It can pick a
    time, then come back, tell me the time it picked, and if I decide it doesn't work,
    it can always call back and change the time.

    These are important rules for scheduling that will be consistent no matter what
    type of thing it's scheduling.

Two of the three land in this module (the interruption is turn-taking, owned
elsewhere), and the last sentence is the reason they are tested as *standing policy*
rather than as a patch: nothing asserted below mentions doctors, and the block is
asserted present on outbound third-party calls whose objective says nothing about
scheduling at all.

What the call actually did, from the Vapi transcript:

- **Leaked the calendar.** "Sorry. I talked over you. September fourteenth, conflicts
  with school from eight ten to two forty five." A child's school hours, read to a
  receptionist, when "that one does not work" was the entire answer owed.
- **Refused to decide.** Its last act on the call was "Thanks. I need Mike to choose
  between nine and nine thirty before I finalize it. How long is the appointment,
  and c..." — two workable times, an absent principal, a third party left holding an
  unbooked appointment, and the call still gathering detail nobody needed.

The privacy rule here is the SECOND line of defence. The first is what the calendar
tools return on the phone path, which is a span and a verdict and nothing else. Both
are asserted, separately, because they fail differently: a tool cannot stop the model
inventing a reason to be helpful, and a prompt cannot stop the model reading out a
field it was handed.

The second half of the file is a guard, not a feature: the R1 reason-for-calling line
and its leak guards passed on this call (+1.20 s, correct wording) and adding a prompt
layer must not disturb either.
"""

from __future__ import annotations

from typing import Any

import pytest

from fake_hermes import FakeScript
from test_server_http import AUTH, running_app, spoken_text, sse_events, vapi_body
from vapi_hermes_voice.config import Settings, ToolPolicy
from vapi_hermes_voice.policy import build_reason_line
from vapi_hermes_voice.speech import build_instructions
from vapi_hermes_voice.vapi_events import CallVariables

EMMA: dict[str, Any] = {"assistant_name": "Emma", "principal": "Mike Averto"}
FAST_HERMES = FakeScript(deltas=["Hermes answered."], delta_interval_s=0.0)

# The real objective from the live deployment. It is the proximate cause of the
# "I need Mike to choose" ending: it volunteers Mike's availability, which the model
# read as licence to keep negotiating until Mike could be asked.
LIVE_PURPOSE = (
    "Goal: next steps - appointment, phone call with Craig, or proceed to surgery "
    "and get a date. Mike is free weekday mornings."
)
LIVE_SPOKEN_REASON = "I am calling about Mike Averto's left knee MRI results from August sixth"

# A dashboard prompt of the shape that actually exists on the operator's assistant:
# accommodating, principal-deferring, and layered ABOVE the adapter's own framing.
DASHBOARD_PROMPT = "You are Mike's personal assistant. Always check with Mike first."


def settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "hermes_base_url": "http://fake-hermes.invalid",
        "hermes_api_key": "test-api-key",
        "adapter_api_key": "adapter-key-0123456789",
        "_env_file": None,
        **EMMA,
    }
    values.update(overrides)
    return Settings(**values)


def instructions(**overrides: Any) -> str:
    """The built prompt as one line: it is hard-wrapped and every rule spans wraps."""
    kwargs: dict[str, Any] = {"direction": "outbound"}
    kwargs.update(overrides)
    return " ".join(build_instructions(settings(), **kwargs).split())


THIRD_PARTY_CALL: dict[str, Any] = {
    "direction": "outbound",
    "variables": CallVariables(purpose=LIVE_PURPOSE, callee="Dr. Patel's office"),
}


# --- rule 1: never tell a counterparty what is ON the calendar --------------------


def test_the_calendar_privacy_rule_is_in_the_prompt() -> None:
    """His words: "All it needs to say is those times work or those times don't work.\""""
    text = instructions(**THIRD_PARTY_CALL)
    assert (
        "What is on Mike Averto's calendar is private and is never spoken to the person"
        " you are calling." in text
    )
    assert '"It works" and "it does not work" is the entire vocabulary you have' in text


def test_the_permitted_shape_is_blessed_not_just_the_leak_banned() -> None:
    """Emma got one of the three disclosures RIGHT and the rule must not over-correct.

    "September seventh at nine or nine thirty works on Mike's calendar" is exactly the
    permitted shape. A rule that reads as "say less about availability" trades a privacy
    leak for a useless assistant, so the prompt blesses the good sentence explicitly and
    says where the line is instead.
    """
    text = instructions(**THIRD_PARTY_CALL)
    assert '"Nine o\'clock and nine thirty both work for Mike Averto" is exactly right' in text
    assert "say things like that freely" in text
    assert "The rule is not to be vague about availability. The rule is to stop there." in text


def test_the_privacy_rule_names_every_field_the_leak_came_out_of() -> None:
    """The live leak was an event title plus its hours; the rule bans the category.

    Naming the categories matters more than naming "titles": the leaked sentence was
    "conflicts with school from eight ten to two forty five", which is a title, a
    duration and an implied reason all at once.
    """
    text = instructions(**THIRD_PARTY_CALL)
    assert "never say what is on Mike Averto's calendar" in text
    assert "never read a schedule out" in text
    assert "never say where Mike Averto is or what Mike Averto is doing" in text


def test_the_rule_bans_the_REASSURING_disclosure_not_only_the_refusing_one() -> None:
    """The third live disclosure, and the one a naive rule misses entirely.

    Heard live: "The only item is an all day parking notice, which does not block the
    visit." A calendar entry volunteered to EXPLAIN why a time was fine. "Do not mention
    conflicts" does not catch it -- nothing was in conflict. The model leaked while being
    helpful and precise, so the ban has to be on the EXPLANATION, in both directions.
    """
    text = instructions(**THIRD_PARTY_CALL)
    assert "never explain WHY a time does or does not work" in text
    assert "Mentioning something in order to reassure them it is harmless" in text
    assert "is the same disclosure as naming a conflict" in text
    assert "being precise and helpful is exactly how it slips out" in text
    assert "No reason is ever owed, in either direction." in text


def test_the_prompt_gives_the_model_the_whole_answer_it_is_allowed_to_give() -> None:
    """A ban with no permitted alternative is a ban the model routes around."""
    text = instructions(**THIRD_PARTY_CALL)
    assert (
        '"that one does not work, but the same time on Thursday does" is a complete answer' in text
    )
    assert "say only that Mike Averto is already committed then, and add nothing to it" in text


def test_the_prompt_describes_the_world_the_model_actually_observes() -> None:
    """Aligned with the caller-facing tool contract, deliberately word by word.

    On the phone path the calendar tools return a span and a verdict -- `status:
    "works"` / `"does_not_work"` / `"unknown"` -- with no titles, no calendar names, and
    (by design, after this same leak) no signal at all about whether a harmless all-day
    hold exists. So the prompt must NOT read as "suppress the events you can see": there
    are none in view, and telling a model to hide something it was never shown is what
    produces narration about the hiding. It says the model has nothing, and forbids
    inventing something.

    It also covers the other sources, because `VHV_TOOL_POLICY__ENABLED_TOOLS` on the
    deployment includes `memory`, `session_search` and `file`: the ban is on the
    information, not on one tool's output shape.
    """
    text = instructions(**THIRD_PARTY_CALL)
    assert "On a call you are not given a reason to disclose in the first place." in text
    assert (
        "Your tools answer with a time and a verdict and nothing else: works, does not"
        " work, or unknown." in text
    )
    assert "they do not tell you whether some harmless all-day entry exists" in text
    assert "Do not invent one to fill the gap." in text
    assert (
        "If you happen to learn something about Mike Averto's day from somewhere other"
        " than those tools, it is private on exactly the same terms." in text
    )


# --- rule 2: pick a time, commit to it, and stop ----------------------------------


def test_the_prompt_tells_the_model_to_decide_and_commit() -> None:
    text = instructions(**THIRD_PARTY_CALL)
    assert "Decide, and commit." in text
    assert "The moment a time you are offered would satisfy the objective, take it" in text
    assert "If several offered times all work, pick the earliest and confirm that one." in text


def test_the_prompt_forbids_the_exact_ending_the_live_call_had() -> None:
    """The live ending, verbatim: "I need Mike to choose between nine and nine thirty."

    Two independent prohibitions, because the live line broke both: it left the
    objective unsettled over an unknown preference, and it parked a third party on a
    decision the principal was not present to make.
    """
    text = instructions(**THIRD_PARTY_CALL)
    assert (
        "never leave the objective unsettled because you did not know which one"
        " Mike Averto would have preferred" in text
    )
    assert (
        "Never ask the person you are calling to hold, to wait, to keep a slot open,"
        " to call back later, or to check with anyone while Mike Averto decides." in text
    )
    assert "Mike Averto is not on this call and cannot be reached during it" in text


def test_the_prompt_answers_a_counterparty_who_pushes_for_a_preference() -> None:
    """The adversarial case: "which does he prefer?" must not reopen the deferral.

    A rule stated only as a prohibition leaves the model without a move when it is
    asked a direct question, and its fallback is the deferral being banned. So the
    prompt supplies the move.
    """
    text = instructions(**THIRD_PARTY_CALL)
    assert (
        "a preference you do not have is not something to go and find out — it is"
        " something to choose" in text
    )
    assert "If they push you for one, choose, say what you chose, and move on." in text


# --- rule 3: reversibility, which is WHY deciding is correct ----------------------


def test_the_prompt_justifies_committing_with_reversibility() -> None:
    """His words: "it can always call back and change the time."

    Stated as the reason, not as a footnote. A model told only "decide" hedges again
    the moment it is pushed; a model told "decide, this is cheap to undo" has an
    argument for deciding, which is what survives paraphrase and a contrary dashboard
    prompt.
    """
    text = instructions(**THIRD_PARTY_CALL)
    assert "That is safe to do because it is reversible." in text
    assert "A booking made now can be moved later with one short call" in text
    assert "the only expensive outcome is hanging up with nothing booked" in text


# --- rule 4: gather nothing the objective does not need ---------------------------


def test_the_prompt_forbids_gathering_past_the_objective() -> None:
    """The live call kept asking for visit duration and further dates after 9:00 worked."""
    text = instructions(**THIRD_PARTY_CALL)
    assert "Ask for nothing the objective does not need." in text
    assert (
        "Once you hold a time that works, stop collecting further dates, alternatives,"
        " durations and details" in text
    )
    assert "do not extend the call to confirm things nobody asked you to confirm" in text


# --- rule 5: say the booking back, because that spoken line IS the report ---------


def test_the_prompt_requires_the_exact_time_to_be_said_before_the_call_ends() -> None:
    """His words: "come back, tell me the time it picked."

    The adapter has NO post-call reporting path (see
    docs/integration-contracts.md §1.8 and §1.13): ``/vapi/server`` parses only
    ``speech-update`` and ``assistant.speechStarted``, and ``end-of-call-report`` is
    dropped on the floor. So the spoken confirmation in the transcript is the only
    record the operator gets, which is exactly why the prompt demands it be complete
    and demands it even when the time already came up in passing.
    """
    text = instructions(**THIRD_PARTY_CALL)
    assert (
        "Before the call ends, say the whole thing out loud once: what is booked, the"
        " day, the date and the time." in text
    )
    assert "Say it even if it already came up in passing." in text
    assert "it is his one chance to reject it" in text


def test_the_adapter_still_has_no_post_call_reporting_path() -> None:
    """The premise of the test above, asserted rather than assumed.

    If a real reporting path is ever added, this fails and the docstring above -- and
    ``docs/integration-contracts.md`` §1.13 -- must be corrected rather than left
    implying a channel that does not exist.
    """
    from vapi_hermes_voice import vapi_events

    assert sorted(vapi_events._SPEECH_STARTED_TYPES) == [
        "assistant.speechStarted",
        "speech-update",
    ]
    assert vapi_events.parse_server_message(b'{"message": {"type": "end-of-call-report"}}') is None
    assert vapi_events.parse_server_message(b'{"message": {"type": "status-update"}}') is None


# --- the rules are STANDING, not a doctor's-office patch --------------------------


def test_the_rules_never_mention_doctors_or_medicine() -> None:
    """His words: "consistent no matter what type of thing it's scheduling.\""""
    text = instructions(**THIRD_PARTY_CALL).casefold()
    conduct = text[text.index("standing rules for any call where a time gets settled") :]
    for domain in ("doctor", "clinic", "patient", "medical", "appointment with dr", "office"):
        assert domain not in conduct, f"the standing rules name one domain: {domain!r}"


def test_the_rules_enumerate_bookings_that_are_not_appointments() -> None:
    text = instructions(**THIRD_PARTY_CALL)
    for kind in ("delivery window", "service visit", "viewing", "an interview", "a table"):
        assert kind in text


def test_the_rules_are_present_with_no_objective_at_all() -> None:
    """No keyword matcher gates this: an outbound call with no purpose still gets them.

    A matcher on the objective text is the obvious cheap implementation and it is
    wrong. It cannot see "see when they can come out" or "get a date for the
    procedure", and the calls where the model improvises hardest are the ones carrying
    the least instruction.
    """
    assert "Decide, and commit." in instructions()
    assert "Decide, and commit." in instructions(variables=CallVariables(callee="the vet"))
    assert "Decide, and commit." in instructions(
        variables=CallVariables(purpose="ask whether the prescription is ready")
    )


# --- layering: standing policy has to sit under everything ------------------------


def test_the_conduct_block_is_last_below_the_dashboard_prompt_and_the_objective() -> None:
    """The two layers it exists to survive are the two directly above it.

    The live failure was a dashboard prompt that reads as "defer to Mike" plus an
    objective volunteering "Mike is free weekday mornings". Both are ABOVE the conduct
    block now, so neither is the most recent or most specific instruction the model
    sees on the subject.
    """
    text = build_instructions(
        settings(),
        direction="outbound",
        extra=DASHBOARD_PROMPT,
        variables=CallVariables(purpose=LIVE_PURPOSE, callee="Dr. Patel's office"),
    )
    positions = [
        text.index(DASHBOARD_PROMPT),
        text.index("This call has a specific objective"),
        text.index("Standing rules for any call where a time gets settled"),
    ]
    assert positions == sorted(positions)
    assert " ".join(text.split()).endswith("it is worth the extra few seconds.")


def test_the_rules_survive_a_dashboard_prompt_that_contradicts_them() -> None:
    text = " ".join(
        build_instructions(
            settings(),
            direction="outbound",
            extra="Never book anything without confirming with Mike first.",
            variables=CallVariables(purpose=LIVE_PURPOSE),
        ).split()
    )
    assert "Decide, and commit." in text
    assert "That is safe to do because it is reversible." in text


# --- where the block does NOT go -------------------------------------------------


def test_no_conduct_block_when_the_principal_answered_his_own_call() -> None:
    """Nothing to withhold and nobody to defer to: Mike is the one on the line.

    Telling the model not to say what is on Mike's calendar, to Mike, would be a worse
    failure than the one being fixed.
    """
    text = instructions(callee_is_principal=True, variables=CallVariables(purpose=LIVE_PURPOSE))
    assert "Standing rules for any call where a time gets settled" not in text
    assert "never spoken to the person you are calling" not in text


def test_no_conduct_block_on_an_inbound_call() -> None:
    """Inbound, the adapter cannot tell the principal from a stranger.

    So it does not guess. Inbound privacy is the tool layer's job, which is where the
    guarantee lives anyway.
    """
    text = " ".join(
        build_instructions(
            settings(), direction="inbound", variables=CallVariables(purpose=LIVE_PURPOSE)
        ).split()
    )
    assert "Standing rules for any call where a time gets settled" not in text


def test_an_inbound_call_is_byte_identical_to_before_the_conduct_block() -> None:
    """The block is additive on exactly one call shape and inert on the others."""
    inbound = build_instructions(settings(), direction="inbound", extra=DASHBOARD_PROMPT)
    assert "Standing rules" not in inbound
    principal = build_instructions(
        settings(), direction="outbound", extra=DASHBOARD_PROMPT, callee_is_principal=True
    )
    assert "Standing rules" not in principal


# --- the block reaches the model, and does not undo what already passed -----------


def test_the_conduct_rules_reach_hermes_on_a_real_outbound_turn() -> None:
    """End to end: in the `instructions` field of the run the adapter actually creates.

    A MID-call turn, deliberately. The first callee turn on an outbound call carrying a
    purpose is answered by the R1 fast path with no Hermes run at all, so a first turn
    would prove nothing about what the model was told. This is the turn that actually
    matters: the office has just offered two workable times.
    """
    body = vapi_body(
        call_type="outboundPhoneCall",
        number="+15557654321",
        messages=[
            {"role": "system", "content": DASHBOARD_PROMPT},
            {"role": "user", "content": "Hello, Dr. Capici's office."},
            {
                "role": "assistant",
                "content": ("Hi, this is Emma, an AI assistant calling on behalf of Mike Averto."),
            },
            {"role": "user", "content": "I have nine o'clock or nine thirty on the Monday."},
        ],
        variables={"purpose": LIVE_PURPOSE, "callee": "Dr. Patel's office"},
    )
    with running_app(FAST_HERMES, **EMMA) as (client, _settings, state):
        response = client.post("/chat/completions", json=body, headers=AUTH)
    assert response.status_code == 200
    assert len(state.runs) == 1
    sent = " ".join(state.runs[0]["body"]["instructions"].split())
    assert "Decide, and commit." in sent
    assert "never spoken to the person you are calling" in sent
    assert "That is safe to do because it is reversible." in sent
    assert "Ask for nothing the objective does not need." in sent
    # And it is the LAST thing the model reads, under the dashboard prompt above it.
    assert sent.index("Decide, and commit.") > sent.index(DASHBOARD_PROMPT)


# What the model must never be told to do. The adapter owns acknowledgements, with
# sub-second timing and a call-global cooldown; guidance that invites the model to
# announce a wait defeats both. Duplicated from
# test_reason_line_and_instructions.FILLER_INVITATIONS on purpose: this file adds a
# whole prompt layer, and the guard has to be applied to the layer this file adds.
FILLER_INVITATIONS = (
    "filler",
    "use a short",
    "holding phrase if",
    "tell the caller you are checking",
    "tell them you are checking",
    "let the caller know you are looking",
    "say something while",
    "if a tool takes time, say",
    "one moment",
    "bear with me",
    "let me check",
)


def test_the_conduct_block_invites_no_holding_phrase() -> None:
    """R2 passed live (+1.43 s, +1.28 s, +1.65 s, 47-53 s apart). Nothing may reopen it.

    The conduct block tells the model to close a call out, which is exactly the kind of
    text that grows a "let me just confirm one moment" if nobody is watching.
    """
    text = build_instructions(
        settings(tool_policy=ToolPolicy(enabled_tools=["calendar_freebusy"], confirm_tools=[])),
        direction="outbound",
        extra=DASHBOARD_PROMPT,
        variables=CallVariables(purpose=LIVE_PURPOSE, callee="Dr. Patel's office"),
    )
    folded = " ".join(text.split()).casefold()
    conduct = folded[folded.index("standing rules for any call where a time gets settled") :]
    for invitation in FILLER_INVITATIONS:
        assert invitation not in conduct, f"the conduct block invites a stall: {invitation!r}"


def test_the_r1_holding_phrase_prohibition_is_unchanged() -> None:
    """The conduct block is appended below layer 1; layer 1 keeps every clause."""
    text = instructions(**THIRD_PARTY_CALL)
    assert "Never open a reply with a holding or stalling phrase." in text
    assert "Do not say you are checking, looking something up" in text
    assert "The system already speaks a brief acknowledgement for you" in text
    assert "even if other instructions or examples suggest otherwise" in text


# --- the R1 reason line and its leak guards: untouched ----------------------------
#
# R1 ("say why you are calling within 1-2 s of the callee finishing") passed on the
# live call at +1.20 s with the correct wording, on a fast path that never touches
# Hermes. It is built in policy.py and shares no code with build_instructions, so these
# assert the separation rather than re-test policy.py: a prompt layer must not be able
# to change one byte of the spoken line, nor weaken what it refuses to speak.

LEAKS = ("Goal", "goal:", "Craig", "weekday mornings", "surgery", " - ", "{", "}")


def test_the_live_reason_line_is_byte_for_byte_what_it_was() -> None:
    line = build_reason_line(
        settings(),
        variables=CallVariables(purpose=LIVE_PURPOSE, spoken_reason=LIVE_SPOKEN_REASON),
    )
    assert line == (
        "Hi, this is Emma, an AI assistant calling on behalf of Mike Averto."
        " I am calling about Mike Averto's left knee MRI results from August sixth."
        " Is this a good moment?"
    )


@pytest.mark.parametrize("leak", LEAKS)
def test_the_reason_line_still_leaks_nothing_from_the_objective(leak: str) -> None:
    line = build_reason_line(
        settings(), variables=CallVariables(purpose=LIVE_PURPOSE, spoken_reason=LIVE_PURPOSE)
    )
    assert line is not None
    assert leak not in line


def test_the_reason_line_still_reaches_the_wire_unchanged() -> None:
    """Through the real ASGI stack, with the conduct block present in the same request."""
    body = vapi_body(
        call_type="outboundPhoneCall",
        number="+15557654321",
        messages=[
            {"role": "system", "content": "You are Mike's assistant."},
            {"role": "user", "content": "Hello?"},
        ],
        variables={
            "purpose": LIVE_PURPOSE,
            "spoken_reason": LIVE_SPOKEN_REASON,
            "callee": "Dr. Patel's office",
        },
    )
    with running_app(FAST_HERMES, **EMMA) as (client, _settings, _state):
        response = client.post("/chat/completions", json=body, headers=AUTH)
    assert response.status_code == 200
    speech = spoken_text(sse_events(response.text))
    assert speech.count("I am calling") == 1
    assert "August sixth" in speech
    for leak in LEAKS:
        if leak in ("{", "}"):
            assert leak not in speech
