"""Deterministic cover for reading Vapi's TTS cache verdict out of a call log.

The live probe that produced these fixtures places billable calls and is excluded from
this suite (``tests/e2e``). Its *reading* of the log is not, for the same reason
``tests/test_e2e_deadlines.py`` covers the deadline harness's arithmetic: a classifier
that gets HIT and MISS the wrong way round would manufacture a design decision. The two
fixtures are reduced from real call logs -- only the rows and attributes
``e2e.call_logs`` reads, so no prompt text or model URL is committed -- and they encode
the finding recorded in ``docs/integration-contracts.md`` §1.12:

``call_log_cache_paths.json`` (call ``01a02728``, 2026-08-22): the same nonce delivered
by ``say`` twice and then as model output. The second ``say`` HITS the cache; the model
output MISSES on the identical key seven seconds later.

``call_log_cache_paths_no_payload.json`` (call ``01a0272b``, three minutes later, same
script): Vapi wrote the log WITHOUT its payload tier, so no key or cache marker exists
for any utterance. Every model-streamed utterance is still MISS, because
``assistant.voice.firstAudioReceived`` proves synthesis happened; the ``say`` utterances
read UNKNOWN rather than being guessed at.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from e2e.call_logs import classify, verdict

FIXTURES = Path(__file__).parent / "e2e" / "fixtures"


def _load(name: str) -> list[dict[str, Any]]:
    return list(json.loads((FIXTURES / name).read_text()))


@pytest.fixture(scope="module")
def full_log() -> list[dict[str, Any]]:
    return _load("call_log_cache_paths.json")


@pytest.fixture(scope="module")
def no_payload_log() -> list[dict[str, Any]]:
    return _load("call_log_cache_paths_no_payload.json")


def test_every_spoken_utterance_is_reported(full_log: list[dict[str, Any]]) -> None:
    """Six deliveries, six records, in order -- one per ``botSpeechStarted``."""
    assert [(u["path"], u["outcome"]) for u in classify(full_log)] == [
        ("say", "MISS"),
        ("say", "HIT"),
        ("llm", "MISS"),
        ("llm", "MISS"),
        ("llm", "MISS"),
        ("say", "MISS"),
    ]


def test_say_path_hit_is_an_order_of_magnitude_faster(full_log: list[dict[str, Any]]) -> None:
    """The reason any of this matters: a hit reaches speech in tens of ms, a miss in hundreds."""
    say = [u for u in classify(full_log) if u["path"] == "say"]
    hit = next(u for u in say if u["outcome"] == "HIT")
    misses = [u for u in say if u["outcome"] == "MISS"]
    assert hit["speech_start_ms"] < 50
    assert all(u["speech_start_ms"] > 300 for u in misses)


def test_model_path_misses_a_key_the_cache_is_holding(full_log: list[dict[str, Any]]) -> None:
    """The finding itself: an exact key that HIT earlier on this call, synthesised again."""
    resident = verdict(full_log)["resident_key_missed"]
    assert len(resident) == 1
    assert resident[0]["key"] == "Cache probe alpha 3 3 2 7."
    assert resident[0]["synthesis_ms"] == 350


def test_model_path_never_hits(full_log: list[dict[str, Any]]) -> None:
    per_path = verdict(full_log)["per_path"]
    assert per_path["llm"]["HIT"] == 0
    assert per_path["llm"]["MISS"] == 3
    assert per_path["say"]["HIT"] == 1


def test_model_path_does_not_populate_the_cache(full_log: list[dict[str, Any]]) -> None:
    """The last ``say`` repeats text the model path rendered twice, and still misses.

    So the model path neither reads the cache nor writes it: warming it from there is
    not an alternative to warming it with ``say``.
    """
    utterances = classify(full_log)
    model_keys = {u["key"] for u in utterances if u["path"] == "llm"}
    final = utterances[-1]
    assert final["path"] == "say"
    assert final["key"] in model_keys
    assert final["outcome"] == "MISS"


def test_synthesis_time_is_never_borrowed_from_a_later_utterance(
    full_log: list[dict[str, Any]],
) -> None:
    """A say-path miss reports no synthesis time rather than the next render's.

    Vapi does not emit ``firstAudioReceived`` for a ``say``, and an unbounded forward
    scan reported the model-streamed render 15 s later as this utterance's own -- a
    plausible-looking 350 ms that made a say-path miss indistinguishable from a
    model-path one.
    """
    say_misses = [u for u in classify(full_log) if u["path"] == "say" and u["outcome"] == "MISS"]
    assert say_misses
    assert all(u["synthesis_ms"] is None for u in say_misses)


def test_missing_payload_tier_reads_as_unmeasured_not_as_a_miss(
    no_payload_log: list[dict[str, Any]],
) -> None:
    """No key, no cache marker: ``say`` is UNKNOWN, the model path is still MISS."""
    utterances = classify(no_payload_log)
    assert len(utterances) == 6
    assert all(u["key"] is None for u in utterances)
    per_path = verdict(no_payload_log)["per_path"]
    assert per_path["say"] == {"HIT": 0, "MISS": 0, "UNKNOWN": 3}
    assert per_path["llm"] == {"HIT": 0, "MISS": 3, "UNKNOWN": 0}


def test_speech_start_separates_the_unknown_say_utterances(
    no_payload_log: list[dict[str, Any]],
) -> None:
    """UNKNOWN is not a dead end: the timing still tells a reader which was which."""
    say = [u["speech_start_ms"] for u in classify(no_payload_log) if u["path"] == "say"]
    assert sorted(say) == [5, 28, 397]


def test_a_dropped_render_does_not_lend_its_latency_to_the_next_utterance() -> None:
    """Synthetic: Vapi synthesised audio it then never spoke, and a HIT speaks next.

    The silent-drop defect produces exactly this shape -- an
    ``assistant.voice.firstAudioReceived`` with no ``botSpeechStarted`` behind it -- and
    the next utterance's window necessarily contains it. A record reading HIT with a
    synthesis time attached is self-contradictory: a cache hit synthesises nothing, so
    the number could only have come from the utterance that was dropped.
    """
    base = 1_700_000_000_000
    log = [
        {
            "time": base,
            "body": "LLM first token received",
            "attributes": {"category": "model", "event": "assistant.model.firstTokenReceived"},
        },
        {
            "time": base + 400,
            "body": "vapi TTS first audio received",
            "attributes": {
                "category": "voice",
                "event": "assistant.voice.firstAudioReceived",
                "latency": 400,
            },
        },
        {
            "time": base + 9_000,
            "body": "pipeline.sayQueuePush",
            "attributes": {"category": "pipeline", "event": "pipeline.sayQueuePush"},
        },
        {
            "time": base + 9_004,
            "body": "Voice cached",
            "attributes": {"category": "voice", "text": "Alright, let me see."},
        },
        {
            "time": base + 9_031,
            "body": "Bot started speaking",
            "attributes": {"category": "pipeline", "event": "pipeline.botSpeechStarted"},
        },
    ]
    (utterance,) = classify(log)
    assert utterance["path"] == "say"
    assert utterance["outcome"] == "HIT"
    assert utterance["synthesis_ms"] is None
    assert utterance["speech_start_ms"] == 31


def test_empty_log_is_not_a_verdict() -> None:
    assert classify([]) == []
    assert verdict([]) == {"per_path": {}, "resident_key_missed": []}
