"""Read Vapi's own server-side call log: which path spoke each utterance, and cache HIT/MISS.

``GET /call/{id}/call-logs`` (``VapiClient.call_logs``) is the only record that says what
the PLATFORM did with a piece of text, as opposed to what the adapter sent or what a
callee heard. It settles questions no client-side clock can:

* how long Vapi's TTS actually took (``assistant.voice.firstAudioReceived.latency``),
* the exact string the voice provider was given, post-formatting -- ``"3327."`` reaches
  it as ``"3 3 2 7."`` -- which is the TTS cache key as Vapi computes it,
* whether that key HIT the cache (``"Voice cached"``) or was synthesised again
  (``"Voice input"``),
* and which delivery path carried it: ``pipeline.sayQueuePush`` for a ``say`` control
  frame, ``assistant.model.*`` for text streamed as model output.

Pure and stdlib-only, over rows already fetched, so the collected test suite can cover it
without the live harness's dependencies -- the same split as ``deadlines.py`` (scoring,
covered) against ``ws_call.py`` (drives a billable call, not covered).
"""

from __future__ import annotations

from typing import Any

__all__ = ["classify", "verdict"]

_MODEL_TRIGGERS = ("assistant.model.requestStarted", "assistant.model.firstTokenReceived")


def classify(log_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One record per utterance Vapi spoke: which path delivered it, and cache HIT or MISS.

    Anchored on ``pipeline.botSpeechStarted`` rather than on the cache markers, because
    the markers are not always in the artifact. Vapi writes the log in two tiers: the
    ``event``-bearing rows (``pipeline.*``, ``assistant.*``) are always present, while
    the payload rows carrying the actual strings (``"Voice input"``, ``"Voice cached"``,
    ``"Model output"``) can be absent for a whole call -- observed on two calls placed
    three minutes apart from the same script, one with them and one without, still
    missing an hour later. Anchoring on the always-present tier means every utterance is
    reported; the strings enrich a record when present and are ``None`` when not.

    Each field says what was observed, so nothing here is an inference. Within an
    utterance's window -- the rows since the previous ``botSpeechStarted`` -- the LAST
    observation of a kind wins, because the log is a sequence of things that happened and
    the most recent one is what this utterance came out of. That is what keeps a render
    Vapi accepted and then never spoke (the silent-drop defect) from lending its synthesis
    time to whatever speaks next.

    * ``path`` -- ``"say"`` when ``pipeline.sayQueuePush`` opened this utterance,
      ``"llm"`` when an ``assistant.model.*`` event did, ``"?"`` when neither is in the
      window (Vapi's own ``firstMessage``, for one).
    * ``outcome`` -- ``"HIT"`` on a ``"Voice cached"`` row, ``"MISS"`` on a
      ``"Voice input"`` row or an ``assistant.voice.firstAudioReceived`` (emitted only
      when synthesis actually happened), else ``"UNKNOWN"``.
    * ``synthesis_ms`` -- Vapi's own ``latency`` for the render, when it reported one, and
      ``None`` on a HIT: a cache hit synthesises nothing, so any number there would
      belong to some other utterance.
    * ``speech_start_ms`` -- trigger to ``botSpeechStarted``. Reported for every
      utterance, including the UNKNOWN ones, because it separates the two cases by
      itself: a hit reaches speech in tens of milliseconds, a miss in hundreds.
    """
    seq = sorted(log_rows, key=lambda row: row.get("time") or 0)
    out: list[dict[str, Any]] = []
    window_start = 0
    for i, row in enumerate(seq):
        if row.get("attributes", {}).get("event") != "pipeline.botSpeechStarted":
            continue
        record: dict[str, Any] = {
            "at_ms": row["time"],
            "path": "?",
            "outcome": "UNKNOWN",
            "key": None,
            "synthesis_ms": None,
            "speech_start_ms": None,
        }
        trigger_ms: int | None = None
        for j in range(window_start, i):
            attrs = seq[j].get("attributes", {})
            event = attrs.get("event")
            body = seq[j].get("body")
            if event == "pipeline.sayQueuePush":
                record["path"] = "say"
                trigger_ms = seq[j]["time"]
            elif event in _MODEL_TRIGGERS:
                record["path"] = "llm"
                # The model's FIRST TOKEN, not the request, is when text could first
                # reach the voice pipeline; prefer it when both are in the window.
                if event == _MODEL_TRIGGERS[1] or trigger_ms is None:
                    trigger_ms = seq[j]["time"]
            elif event == "assistant.voice.firstAudioReceived":
                record["outcome"] = "MISS"
                record["synthesis_ms"] = attrs.get("latency")
            elif body == "Voice cached":
                record["outcome"] = "HIT"
                record["key"] = attrs.get("text")
                record["synthesis_ms"] = None
            elif body == "Voice input":
                record["outcome"] = "MISS"
                # Named ``input`` for model output and ``text`` for a say. Reading only
                # one silently drops every utterance the other path produced, and "no
                # misses in the log" and "that path never missed" are indistinguishable
                # to a reader.
                key = attrs.get("input")
                record["key"] = attrs.get("text") if key is None else key
        if trigger_ms is not None:
            record["speech_start_ms"] = row["time"] - trigger_ms
        out.append(record)
        window_start = i + 1
    return out


def verdict(log_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Cache hits per delivery path, plus every hit the model path declined to take.

    A ``say`` hit and a model-streamed hit are the same event to Vapi's cache, so a
    per-path tally is the whole answer. ``UNKNOWN`` is tallied alongside HIT and MISS
    rather than folded into either: a call whose log arrived without the payload tier
    must read as unmeasured, not as evidence of a miss.

    ``resident_key_missed`` is the load-bearing entry -- an utterance that MISSED on the
    model path carrying a key an EARLIER utterance on the same call HIT. Each one is a
    direct observation that the audio was sitting in the cache and the model path
    synthesised it again anyway, which is the difference between "the pool phrases happen
    to be cold" (warmable) and "this path does not read the cache" (not).
    """
    per_path: dict[str, dict[str, int]] = {}
    first_hit_at: dict[str, int] = {}
    resident_key_missed: list[dict[str, Any]] = []
    for entry in classify(log_rows):
        counts = per_path.setdefault(entry["path"], {"HIT": 0, "MISS": 0, "UNKNOWN": 0})
        counts[entry["outcome"]] += 1
        if entry["key"] is None:
            continue
        if entry["outcome"] == "HIT":
            first_hit_at.setdefault(entry["key"], entry["at_ms"])
        elif entry["outcome"] == "MISS" and entry["path"] == "llm":
            hit_at = first_hit_at.get(entry["key"])
            if hit_at is not None and hit_at < entry["at_ms"]:
                resident_key_missed.append(entry)
    return {"per_path": per_path, "resident_key_missed": resident_key_missed}
