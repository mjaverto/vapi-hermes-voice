"""The MODEL's own holding phrases: deterministic suppression, and its evidence.

R2 is "once something like 'okay, let me check' has been said, nothing of that kind
for at least 10 seconds", and the person on the line cannot tell who wrote the words.
The adapter's call-global cooldown governs only the phrases the ADAPTER speaks, so a
model-authored one defeats R2 from the only viewpoint that counts. The prohibition in
``speech.VOICE_SYSTEM_PROMPT`` is persuasion; these tests pin the enforcement.

Three layers, in the order a failure would be diagnosed:

1. the grammar (``holding_opener_length``): what counts as a holding phrase, and --
   far more important -- what does not. Over-reach here deletes a real answer.
2. the stream (``SpokenTurn``): the same decisions made on fragments arriving one at a
   time, without delaying an ordinary answer and without muting a turn.
3. the turn (``stream_turn``): that the phrase never reaches the callee, that
   suppressing it does not silently cancel the adapter's OWN acknowledgement, that
   the SSE framing guarantees still hold, and that the strip is recorded rather than
   silent -- a silent strip would let a worsening model hide behind its own fix.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections import deque
from typing import Any

import pytest
from starlette.testclient import TestClient

from fake_hermes import FakeScript, build_fake_hermes_transport
from test_server_http import API_KEY, AUTH, vapi_body
from test_turns import _ScriptedHermes, delta, done, error, make_settings, make_state, tool_start
from vapi_hermes_voice.ack_journal import MAX_TEXT_CHARS, AckJournal, AckRecord
from vapi_hermes_voice.call_state import call_ref
from vapi_hermes_voice.config import Settings
from vapi_hermes_voice.server import create_app
from vapi_hermes_voice.speech import (
    MAX_HOLDING_OPENER_CHARS,
    DeltaSanitizer,
    SpokenTurn,
    holding_opener_length,
    sanitize_spoken,
)
from vapi_hermes_voice.turns import stream_turn

POOL = [
    "Okay, let me check.",
    "Okay, one moment.",
    "Got it, one second.",
    "Sure, give me a second.",
    "Alright, let me see.",
    "Okay, bear with me a moment.",
    "Right, one second.",
    "Understood, one moment.",
]

# The exact words the callee heard at 2.642s on live call 01a02681 -- a verbatim member
# of the shipped pool, which is what makes "who said it" undecidable from the text alone.
LIVE_PHRASE = "Alright, let me see."
ANSWER = "The vet prescribed pimobendan, twice daily."


# --- 1. the grammar: what is a holding phrase, and what is an answer ---------------


@pytest.mark.parametrize("phrase", POOL)
def test_every_configured_phrase_is_recognized_verbatim(phrase: str) -> None:
    """A phrase the adapter is configured to SAY is one the model must not say.

    Pool membership is the one rule that needs no grammar: an operator putting a line
    in ``filler_phrases`` has stated that the line is a holding phrase.
    """
    text = f"{phrase} {ANSWER}"
    assert holding_opener_length(text, POOL) == len(phrase)


@pytest.mark.parametrize(
    ("stall", "rest"),
    [
        # Stalls the pool does not contain, worded the way a model actually words them.
        ("Okay, let me take a look.", "It is on Tuesday."),
        ("Hold on a second.", "He takes pimobendan."),
        ("Just a moment.", "Pimobendan."),
        ("Let me check.", "Pimobendan."),
        ("Checking now.", "Pimobendan."),
        ("Of course, one second.", "Pimobendan."),
        ("Right. One second.", "Pimobendan."),  # two sentences, one holding phrase
        ("Okay, bear with me.", "Pimobendan."),
        ("I'll check that for you.", "Pimobendan."),
    ],
)
def test_close_variants_are_recognized_too(stall: str, rest: str) -> None:
    """Suppression cannot depend on the model choosing our exact wording.

    Otherwise a model drifting to a novel stall walks straight past the mechanism,
    and the pool-only case would look like a clean win while R2 quietly broke again.
    """
    length = holding_opener_length(f"{stall} {rest}", POOL)
    assert length == len(stall), f"not recognized as a holding phrase: {stall!r}"


@pytest.mark.parametrize(
    "text",
    [
        # Every one of these says something. The whole-sentence rule is what protects
        # them: a leading sentence is deleted only when EVERY word in it is filler.
        "Let me check your calendar for Tuesday.",
        "Let me see what I can do about the appointment.",
        "I'll check the calendar and call you back.",
        "Right, the vet prescribed pimobendan.",
        "Sure, I can help with that.",
        "Of course, his appointment is Tuesday.",
        "One second is all it took, apparently.",
        "Looking good so far.",
        "Hold on to the receipt.",
        # A bare acknowledgement is not a holding phrase: it claims nothing and
        # delays nothing.
        "Okay.",
        "Sure.",
        "Alright. The vet prescribed pimobendan.",
        # Answers that happen to be built from words the grammar knows.
        "Yes, one moment is fine.",
        "No.",
        "Tuesday morning is partly booked.",
        "Pimobendan, twice daily.",
        # Emptiness and punctuation must not be mistaken for filler.
        "",
        "...",
        "42 degrees.",
    ],
)
def test_text_that_carries_information_is_never_stripped(text: str) -> None:
    assert holding_opener_length(text, POOL) == 0


def test_a_holding_phrase_mid_sentence_is_untouched() -> None:
    """Scope is the START of a turn. Nothing else."""
    text = f"{ANSWER} {LIVE_PHRASE}"
    assert holding_opener_length(text, POOL) == 0


def test_a_holding_phrase_mid_turn_is_untouched() -> None:
    text = f"He takes pimobendan. {LIVE_PHRASE} Twice daily."
    assert holding_opener_length(text, POOL) == 0


def test_only_the_leading_run_is_stripped_never_a_later_one() -> None:
    text = f"{LIVE_PHRASE} {ANSWER} Okay, one moment."
    length = holding_opener_length(text, POOL)
    assert text[length:].strip() == f"{ANSWER} Okay, one moment."


def test_what_may_be_deleted_is_bounded() -> None:
    """A cap on the deletion is a cap on the damage a grammar bug can do."""
    runaway = "Okay. " * 200 + ANSWER
    assert holding_opener_length(runaway, POOL) <= MAX_HOLDING_OPENER_CHARS


def test_the_deletion_cap_stays_inside_the_journal_text_cap() -> None:
    """The 96-char cap must be the SEMANTIC one, with MAX_TEXT_CHARS as the backstop.

    If the gate's cap ever exceeded the journal's, truncation would silently move to
    the journal and the recorded text would stop being the text the gate saw -- the
    evidence would quietly stop matching the event.
    """
    assert MAX_HOLDING_OPENER_CHARS <= MAX_TEXT_CHARS


# --- 2. the stream: the same decisions, on fragments -------------------------------


def drive(pieces: list[str], pool: list[str] | None = None) -> tuple[str, tuple[str, str] | None]:
    """Feed ``pieces`` through a SpokenTurn; return (spoken text, suppression)."""
    spoken = SpokenTurn(POOL if pool is None else pool)
    out = [spoken.feed(piece) for piece in pieces]
    out.append(spoken.flush())
    return "".join(out), spoken.take_suppressed_opening()


def test_a_streamed_holding_phrase_is_stripped_however_it_is_chunked() -> None:
    """Deltas arrive in arbitrary fragments; the decision must not depend on them."""
    whole = drive([f"{LIVE_PHRASE} {ANSWER}"])
    letters = drive(list(f"{LIVE_PHRASE} {ANSWER}"))
    words = drive([f"{w} " for w in f"{LIVE_PHRASE} {ANSWER}".split()])
    assert whole == letters == words == (ANSWER, (LIVE_PHRASE, "pool"))


def test_markdown_around_a_holding_phrase_does_not_hide_it() -> None:
    """The gate reads SANITIZED text, so emphasis cannot smuggle a phrase past it."""
    text, suppression = drive([f"**{LIVE_PHRASE}** ", ANSWER])
    assert text == ANSWER
    assert suppression == (LIVE_PHRASE, "pool")


def test_a_novel_stall_is_reported_as_grammar_not_pool() -> None:
    """WHICH rule fired is the diagnostic: inventing new stalls is the worse signal."""
    text, suppression = drive(["Hold on a second. ", ANSWER])
    assert text == ANSWER
    assert suppression == ("Hold on a second.", "grammar")


def test_a_turn_that_is_only_a_holding_phrase_is_spoken_not_muted() -> None:
    """Dead air was the original complaint; a repeated ack is the smaller failure.

    Nothing is reported as suppressed either, because nothing was: the callee heard
    the phrase, so the record must not claim we stopped it.
    """
    text, suppression = drive([LIVE_PHRASE])
    assert text == LIVE_PHRASE
    assert suppression is None


def test_an_ordinary_answer_is_not_held_back_waiting_for_a_decision() -> None:
    """Latency: a first word no holding phrase starts with releases immediately.

    Buffering the opening of every turn until its first sentence ended would put the
    gate on the critical path of the answer, which is the thing this whole subsystem
    exists to protect.
    """
    spoken = SpokenTurn(POOL)
    assert spoken.feed("Pimobendan") == "Pimobendan"
    assert spoken.holding_opening is False


def test_holding_opening_reports_only_while_a_decision_is_pending() -> None:
    spoken = SpokenTurn(POOL)
    assert spoken.feed("Alright, let me") == ""
    assert spoken.holding_opening is True, "a possible holding phrase must not count as content"
    assert spoken.feed(f" see. {ANSWER}") == ANSWER
    assert spoken.holding_opening is False, "the gate must open once the answer is known"


@pytest.mark.parametrize(
    "pieces",
    [
        [
            "Marvin's ",
            "**emerg",
            "ency vis",
            "it** ",
            "went well, ",
            "the vet said `no",
            " issues`.",
        ],
        ["A link: ", "https://example.com/x", " and more."],
        ["# Heading\n", "- one\n", "- two\n"],
    ],
)
def test_sanitizing_is_byte_identical_with_the_gate_in_front_of_it(pieces: list[str]) -> None:
    """The gate may not alter text that is not a holding phrase, in any way.

    Compared against ``DeltaSanitizer`` directly rather than against a literal, so
    this stays true as the sanitizer evolves.
    """
    bare = DeltaSanitizer()
    baseline = "".join(bare.feed(p) for p in pieces) + bare.flush()
    gated = SpokenTurn(POOL)
    assert "".join(gated.feed(p) for p in pieces) + gated.flush() == baseline
    assert gated.take_suppressed_opening() is None


def test_an_empty_pool_disables_suppression_entirely() -> None:
    """A deployment that clears its pool must behave exactly as it did before."""
    text, suppression = drive([f"{LIVE_PHRASE} {ANSWER}"], pool=[])
    assert text == sanitize_spoken(f"{LIVE_PHRASE} {ANSWER}")
    assert suppression is None


# --- 3. the turn: the callee, the acknowledgement, the framing, the record ---------


async def run_turn(
    events: list[tuple[float, Any]],
    settings: Settings,
    *,
    journal: AckJournal | None = None,
) -> list[str]:
    """Drive ``stream_turn`` and return the raw SSE chunks, one per yield."""
    reaping: set[asyncio.Task[Any]] = set()
    chunks: list[str] = []
    async for chunk in stream_turn(
        settings=settings,
        hermes=_ScriptedHermes(events),
        state=make_state(settings),
        instructions="instructions",
        history=[],
        user_input="What medication did the vet prescribe Marvin?",
        reaping=reaping,
        journal=journal,
    ):
        chunks.append(chunk)
    for task in list(reaping):
        with contextlib.suppress(Exception):
            await task  # a cleanup failure is not what any test here is about
    return chunks


def spoken_content(chunks: list[str]) -> str:
    """Everything the callee would hear, in order, framing removed."""
    out: list[str] = []
    for chunk in chunks:
        for line in chunk.splitlines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta_obj = json.loads(line[len("data: ") :])["choices"][0]["delta"]
            if "content" in delta_obj:
                out.append(delta_obj["content"])
    return "".join(out)


async def test_a_model_authored_holding_phrase_never_reaches_the_callee() -> None:
    """The live failure, as a test: the model opens with a verbatim pool phrase."""
    # filler_after_seconds well past the turn: the adapter's own ack never fires here.
    settings = make_settings(filler_after_seconds=5.0, filler_phrases=POOL)
    events = [
        (0.01, delta(f"{LIVE_PHRASE} ")),
        (0.01, delta(ANSWER)),
        (0.01, done()),
    ]
    heard = spoken_content(await run_turn(events, settings))

    assert LIVE_PHRASE not in heard
    assert heard.strip() == ANSWER


async def test_suppressing_the_model_phrase_does_not_cancel_our_own_acknowledgement() -> None:
    """The interaction that would have broken R2 in a NEW way.

    ``content_started`` forbids an acknowledgement once the answer has begun. A model
    holding phrase is not the answer beginning -- it is about to be deleted -- so if
    it set that flag, the gate would delete the model's phrase AND suppress ours, and
    the callee would hear nothing at all for the whole turn. Both must be true here:
    the model's phrase is gone, and ours was still spoken.

    The two phrases are deliberately different strings, and the adapter's pool holds
    only its own: with a shared pool the assertion cannot tell whose words were heard,
    which is the very ambiguity this whole subsystem exists to remove.
    """
    ours = "Understood, one moment."
    theirs = "Hold on a second."
    settings = make_settings(
        filler_after_seconds=0.08, filler_min_gap_seconds=0.01, filler_phrases=[ours]
    )
    events = [
        (0.01, delta(theirs)),  # model stalls immediately, well before our deadline
        (0.40, delta(f" {ANSWER}")),  # the real answer, long after
        (0.01, done()),
    ]
    heard = spoken_content(await run_turn(events, settings))

    assert ANSWER in heard
    assert theirs not in heard, "the model's holding phrase was spoken"
    assert ours in heard, "no acknowledgement at all: the callee heard nothing for 0.4s"


async def test_the_gate_never_strips_the_adapters_own_acknowledgement() -> None:
    """The scariest over-reach: the gate eating the very phrase it is protecting.

    Our acknowledgement is written straight to the SSE stream and never passes through
    the gate, which reads only MODEL text. That separation is load-bearing, so it is
    asserted with the acknowledgement pool and the model's opener sharing wording.
    """
    settings = make_settings(
        filler_after_seconds=0.05, filler_min_gap_seconds=10.0, filler_phrases=[LIVE_PHRASE]
    )
    events = [(0.30, delta(ANSWER)), (0.01, done())]
    heard = spoken_content(await run_turn(events, settings))

    assert LIVE_PHRASE in heard, "the adapter's own acknowledgement was stripped"
    assert ANSWER in heard


async def test_a_mid_turn_holding_phrase_is_left_alone_end_to_end() -> None:
    settings = make_settings(filler_after_seconds=5.0, filler_phrases=POOL)
    events = [
        (0.01, delta("He takes pimobendan. ")),
        (0.01, delta(f"{LIVE_PHRASE} Twice daily.")),
        (0.01, done()),
    ]
    heard = spoken_content(await run_turn(events, settings))

    assert heard.strip() == f"He takes pimobendan. {LIVE_PHRASE} Twice daily."


async def test_the_answer_is_never_split_mid_frame_by_the_gate() -> None:
    """Atomicity: the gate releases held text in ONE write, never a torn fragment.

    ``test_turns.test_filler_and_flush_token_are_one_atomic_sse_frame`` pins the same
    guarantee for the adapter's own acknowledgement; this is the model side of it,
    because the gate is the only thing that ever holds model text back and then lets
    it go.
    """
    settings = make_settings(filler_after_seconds=5.0, filler_phrases=POOL)
    events = [(0.005, delta(piece)) for piece in [LIVE_PHRASE, " He takes ", "pimobendan."]]
    events.append((0.005, done()))
    chunks = await run_turn(events, settings)

    content_frames = [c for c in chunks if '"content"' in c]
    for frame in content_frames:
        assert frame.count("data: ") == 1, f"more than one SSE write in a single yield: {frame!r}"
    released = next(c for c in content_frames if "He takes" in c)
    assert "pimobendan" not in released, "the gate must release only what it was holding"
    assert LIVE_PHRASE not in "".join(content_frames)


async def test_an_error_after_a_held_phrase_still_apologizes_and_loses_nothing() -> None:
    """A turn that fails while the gate is holding must not swallow the apology."""
    settings = make_settings(filler_after_seconds=5.0, filler_phrases=POOL)
    events = [(0.01, delta(LIVE_PHRASE)), (0.01, error("Sorry, I hit a problem."))]
    heard = spoken_content(await run_turn(events, settings))

    assert "Sorry, I hit a problem." in heard


async def test_the_suppression_is_recorded_not_silent() -> None:
    """A silent strip would hide a worsening model behind its own fix."""
    journal = AckJournal(max_calls=4, max_entries_per_call=4, ttl_seconds=60.0)
    settings = make_settings(filler_after_seconds=5.0, filler_phrases=POOL)
    events = [(0.01, delta(f"{LIVE_PHRASE} {ANSWER}")), (0.01, done())]
    state_ref = make_state(settings).call_ref  # same derivation run_turn uses

    await run_turn(events, settings, journal=journal)
    snapshot = journal.snapshot(state_ref)

    assert snapshot is not None
    assert [(e.text, e.channel) for e in snapshot.suppressed] == [(LIVE_PHRASE, "pool")]
    assert snapshot.suppressed_dropped == 0
    # The record of what the callee HEARD is untouched by the record of what it did not.
    assert snapshot.acks == []
    assert snapshot.dropped == 0


async def test_a_suppression_is_recorded_exactly_once_however_many_chunks_follow() -> None:
    journal = AckJournal(max_calls=4, max_entries_per_call=4, ttl_seconds=60.0)
    settings = make_settings(filler_after_seconds=5.0, filler_phrases=POOL)
    events = [(0.005, delta(p)) for p in [LIVE_PHRASE, " He takes ", "pimo", "bendan."]]
    events.extend([(0.005, tool_start()), (0.005, done())])

    await run_turn(events, settings, journal=journal)
    snapshot = journal.snapshot(make_state(settings).call_ref)

    assert snapshot is not None
    assert len(snapshot.suppressed) == 1


# --- the journal and the endpoint -------------------------------------------------


def test_suppressions_cannot_evict_an_acknowledgement_or_inflate_dropped() -> None:
    """``dropped`` decides whether an unmatched phrase may be called model-authored.

    A chatty model bumping it would turn every verdict into "inconclusive" -- the
    regression hiding behind the fix, which is exactly what this record exists to
    prevent.
    """
    journal = AckJournal(max_calls=4, max_entries_per_call=2, ttl_seconds=60.0)
    journal.record("abcdef012345", text="Okay, one moment.", channel="stream", elapsed_ms=1)
    for i in range(20):
        journal.note_suppressed("abcdef012345", text=f"Okay, {i}.", reason="pool", elapsed_ms=i)
    snapshot = journal.snapshot("abcdef012345")

    assert snapshot is not None
    assert [e.text for e in snapshot.acks] == ["Okay, one moment."]
    assert snapshot.dropped == 0, "a suppression evicted or shadowed an acknowledgement"
    assert snapshot.suppressed_dropped > 0
    assert len(snapshot.suppressed) == journal.limits["max_suppressed_per_call"]


def test_a_call_whose_only_evidence_is_a_suppression_still_has_a_record() -> None:
    journal = AckJournal(max_calls=4, max_entries_per_call=2, ttl_seconds=60.0)
    journal.note_suppressed("abcdef012345", text=LIVE_PHRASE, reason="pool", elapsed_ms=7)
    snapshot = journal.snapshot("abcdef012345")

    assert snapshot is not None
    assert snapshot.acks == []
    assert [e.text for e in snapshot.suppressed] == [LIVE_PHRASE]


def test_a_bucket_holding_only_a_suppression_is_not_empty() -> None:
    """The eviction emptiness test must span EVERY kind of record, not just ``acks``.

    A bucket is deleted once it is empty and itself older than the TTL. If "empty"
    meant "no acknowledgements", the record deleted would be exactly the one that
    matters most -- the adapter emitted nothing, and the model tried a holding phrase
    and was stripped -- and its reader would get None and fall back to "unknown",
    losing the evidence entirely.

    Asserted structurally, on ``_CallAcks.empty``, and NOT dressed up as a timing
    test, because with today's single shared TTL the deletion is unreachable: every
    write bumps ``touched_at``, so by the time a bucket's own age passes the TTL its
    newest entry has too, and the window in which "acks empty, suppression fresh,
    bucket stale" could exist is empty. Verified by probing all three write orderings
    at 0.06/0.08/0.11s against a 0.05s TTL -- no ordering produces it.

    So this guards the STRUCTURE for the next author, which is the real risk: a kind
    of record added with a different cap, a different TTL, or a write path that does
    not touch the bucket reopens the window immediately, and `empty`/`kinds` are what
    make that safe by default instead of one-more-branch-to-remember.
    """
    from vapi_hermes_voice.ack_journal import _Bounded, _CallAcks  # noqa: PLC0415 - structural

    bucket = _CallAcks(4, now=0.0)
    assert bucket.empty is True
    # The invariant, asserted as the invariant rather than as a count. A hardcoded
    # number here failed the moment somebody added a kind CORRECTLY, which taught the
    # next author to bump the number rather than to check the wiring -- the exact
    # reflex this test exists to prevent. Every bounded deque on the bucket must be in
    # `kinds`, or eviction and the emptiness test silently skip it.
    bounded = {name for name in _CallAcks.__slots__ if isinstance(getattr(bucket, name), _Bounded)}
    assert bounded, "no bounded record kinds found -- has _CallAcks been restructured?"
    assert {id(getattr(bucket, name)) for name in bounded} == {id(kind) for kind in bucket.kinds}, (
        "a new kind of record must be added to `kinds`"
    )

    bucket.suppressed.append(
        AckRecord(
            text=LIVE_PHRASE, channel="pool", at_epoch_s=1.0, elapsed_ms=7, at_monotonic_s=1.0
        )
    )
    assert bucket.acks.entries == deque()
    assert bucket.empty is False, "a bucket whose only evidence is a suppression is not empty"


def test_every_kind_expires_against_the_same_ttl() -> None:
    """One TTL governs every kind of record, and this is what enforces it.

    The whole safety argument for bucket deletion rests on it. Because every write
    bumps ``touched_at``, a bucket's own age and its newest entry's age cross the TTL
    together, so the state that would lose evidence -- one kind empty, another still
    holding, the bucket itself stale -- cannot exist. Probed against a 0.05 s TTL at
    0.06/0.08/0.11 s across all three write orderings: no ordering produces it.

    Give one kind a longer TTL and that collapses, because ``_expire``'s staleness
    test compares ``touched_at`` against a single ``ttl`` -- it would have to become
    the longest TTL across the kinds, or a long-retention record is swept the moment
    the short-retention ones age out. Someone will want to try exactly this: an
    answer-delivery outcome stays useful after the acknowledgements it followed have
    aged out.

    So this is deliberately a TIMING test, unlike its structural neighbour above. All
    three kinds are written at the same instant and the record is required to vanish
    WHOLE. A kind given a longer TTL keeps its entries, leaves the bucket non-empty,
    and turns the final assertion red at the moment the change is made -- which a
    comment in ``_expire`` would not.
    """
    journal = AckJournal(max_calls=4, max_entries_per_call=4, ttl_seconds=0.05)
    ref = "abcdef012345"
    journal.record(ref, text="Okay, one moment.", channel="stream", elapsed_ms=1)
    journal.note_suppressed(ref, text=LIVE_PHRASE, reason="pool", elapsed_ms=2)
    journal.note_answer_attempt(ref)

    before = journal.snapshot(ref)
    assert before is not None
    assert (len(before.acks), len(before.suppressed), len(before.answer_deliveries)) == (1, 1, 1), (
        "all three kinds must be populated, or this test cannot see a TTL diverge"
    )

    # Past the TTL for every kind. The sweep is driven by unrelated traffic rather
    # than by reading this call, because `_touch` protects the call it is writing to.
    time.sleep(0.08)
    journal.record("fedcba543210", text="Okay, one moment.", channel="stream", elapsed_ms=1)

    assert journal.snapshot(ref) is None, (
        "a kind outlived the shared TTL: see AckJournal._expire, the bucket staleness "
        "test must become the LONGEST TTL across kinds before any kind gets its own"
    )


def test_the_endpoint_reports_suppressions_apart_from_acknowledgements() -> None:
    """Off-box, "never said one" and "said one and we stripped it" must differ.

    Present-and-empty is itself an answer -- the gate ran and found nothing -- and is
    not the same as absent, which is an adapter too old to have a gate at all.
    """
    script = FakeScript(deltas=[f"{LIVE_PHRASE} ", ANSWER], delta_interval_s=0.01)
    transport, _state = build_fake_hermes_transport(script)
    settings = Settings(
        hermes_base_url="http://fake-hermes.invalid",
        hermes_api_key="test-api-key",
        adapter_api_key=API_KEY,
        warmup_on_start=False,
        filler_after_seconds=5.0,
        filler_phrases=POOL,
        _env_file=None,
    )
    app = create_app(settings, hermes_transport=transport)
    with TestClient(app) as client:
        with client.stream(
            "POST", "/chat/completions", json=vapi_body(call_id="c-1"), headers=AUTH
        ) as response:
            heard = "".join(response.iter_text())
        record = client.get(f"/debug/acks/{call_ref('c-1')}", headers=AUTH).json()

    assert LIVE_PHRASE not in heard
    assert record["acks"] == []
    assert [(e["text"], e["channel"]) for e in record["suppressed_model_openings"]] == [
        (LIVE_PHRASE, "pool")
    ]
    assert record["suppressed_dropped"] == 0
    assert "max_suppressed_per_call" in record["limits"]
