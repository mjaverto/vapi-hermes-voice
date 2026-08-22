#!/usr/bin/env python3
"""Does Vapi's TTS cache serve the adapter's delivery path? A controlled A/B on one call.

Vapi caches synthesised audio (``voice.cachingEnabled``, default true) and a hit is
worth roughly half a second: on the ``say`` path a hit reaches ``botSpeechStarted`` in
tens of milliseconds, a miss in hundreds. The adapter does NOT use the ``say`` path for
acknowledgements -- it writes them into the model.url SSE stream and ends the response
behind them (``turns.py``) -- so whether that half second is available at all turns on
one question this probe answers: *is the cache consulted for text that arrives as model
output rather than as a ``say`` control frame?*

The measurement is Vapi's own, not this harness's inference. ``GET /call/{id}/call-logs``
returns gzipped JSONL in which each utterance appears as exactly one of:

* ``"Voice cached"`` -- a cache HIT. No synthesis happened, and
  ``pipeline.botSpeechStarted`` follows within tens of milliseconds.
* ``"Voice input"`` -- a cache MISS. The string is the EXACT text handed to the voice
  provider, which is the cache key as Vapi sees it, and the following
  ``assistant.voice.firstAudioReceived`` carries ``attributes.latency``, the synthesis
  time in milliseconds.

Vapi names that string differently per path -- ``attributes.input`` for model output,
``attributes.text`` for a ``say`` -- so ``classify`` accepts both. Reading only one
silently drops every utterance the other path produced, and "no misses in the log" and
"that path never missed" are indistinguishable to a reader.

So each step below is a labelled cause with an unambiguous recorded effect, and the
two delivery paths are distinguished by what precedes the outcome in the same log:
``pipeline.sayQueuePush`` for ``say``, ``assistant.model.*`` for model output.

Text is controlled exactly on both paths. On the ``say`` path the probe sends the
string itself. On the model path it injects an ``add-message`` user turn and an echo
custom-llm (``echo_llm.py``, reached through a throwaway tunnel) streams that same
string straight back, so the probe chooses the model's output byte for byte -- including
the adapter's live framing, `` <flush /> `` and the trailing space it carries.

Nothing rings: websocket transport only, no phone number. The assistant is used with a
per-call ``assistantOverrides.model`` and NOTHING else, so the voice under test is the
live one and no persisted configuration is touched.

Costs real money: one Vapi websocket call per run (~$0.05 for the default plan).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import websockets

if __package__ in (None, ""):  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "tests.e2e"

from .audio_script import SAMPLE_RATE
from .call_logs import classify, verdict
from .vapi_live import ASSISTANT_ID, VapiClient, load_api_key
from .ws_call import AUDIBLE_PEAK, FRAME_BYTES, FRAME_MS, _peak

__all__ = ["ProbeRun", "Step", "run_cache_probe"]

# An utterance that has produced no audio by here is a drop, not a slow render, and the
# step is reported as inconclusive rather than as a cache miss.
AUDIO_TIMEOUT_S = 20.0
# Audible gap that marks the end of an utterance.
UTTERANCE_END_QUIET_S = 1.0
# Breathing room so one render cannot bleed into the next step's measurement.
SETTLE_S = 3.0
# How long to wait for the log artifact, which Vapi writes asynchronously after the call
# ends. A run whose log never arrives reports no utterances rather than a false verdict.
LOG_WAIT_S = 180.0


@dataclass(slots=True)
class Step:
    """One labelled cause: a text delivered by one path, and what came back."""

    label: str
    path: str  # "say" | "llm"
    text: str
    sent_at_s: float
    first_audio_s: float | None = None

    @property
    def heard_gap_s(self) -> float | None:
        if self.first_audio_s is None:
            return None
        return self.first_audio_s - self.sent_at_s

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "path": self.path,
            "text": self.text,
            "sent_at_s": round(self.sent_at_s, 3),
            "first_audio_s": None if self.first_audio_s is None else round(self.first_audio_s, 3),
            "client_first_audio_s": None
            if self.heard_gap_s is None
            else round(self.heard_gap_s, 3),
        }


@dataclass
class ProbeRun:
    call_id: str
    steps: list[Step] = field(default_factory=list)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "error": self.error,
            "steps": [s.as_dict() for s in self.steps],
        }


async def _drive(ws_url: str, plan: list[tuple[str, str, str]], run: ProbeRun) -> None:
    """Stream silence, deliver each ``(label, path, text)`` and time the audio back."""
    loop_start = time.monotonic()

    def now() -> float:
        return time.monotonic() - loop_start

    audible_at: list[float] = []
    speaking = False
    closed = asyncio.Event()

    async with websockets.connect(ws_url, max_size=None) as ws:
        loop_start = time.monotonic()

        async def receiver() -> None:
            nonlocal speaking
            try:
                async for message in ws:
                    if isinstance(message, bytes | bytearray):
                        if _peak(bytes(message)) >= AUDIBLE_PEAK:
                            audible_at.append(now())
                        continue
                    try:
                        payload = json.loads(message)
                    except (TypeError, ValueError):
                        continue
                    if payload.get("type") == "speech-update" and (
                        payload.get("role") == "assistant"
                    ):
                        speaking = payload.get("status") == "started"
            except websockets.ConnectionClosed:
                pass
            finally:
                closed.set()

        async def uplink() -> None:
            """Digital silence at wall-clock speed: nothing is ever transcribed."""
            frame = b"\x00" * FRAME_BYTES
            deadline = time.monotonic()
            try:
                while not closed.is_set():
                    await ws.send(frame)
                    deadline += FRAME_MS / 1000
                    lateness = time.monotonic() - deadline
                    if lateness < 0:
                        await asyncio.sleep(-lateness)
            except (websockets.ConnectionClosed, OSError):
                pass

        recv_task = asyncio.create_task(receiver())
        send_task = asyncio.create_task(uplink())

        async def wait_idle(*, limit: float) -> None:
            hard = time.monotonic() + limit
            while time.monotonic() < hard:
                last = audible_at[-1] if audible_at else None
                if (last is None or now() - last >= UTTERANCE_END_QUIET_S) and not speaking:
                    return
                await asyncio.sleep(0.05)

        try:
            await asyncio.sleep(2.5)  # let the call finish coming up
            for label, path, text in plan:
                await wait_idle(limit=25.0)
                await asyncio.sleep(SETTLE_S)
                mark = len(audible_at)
                if path == "say":
                    frame = {"type": "say", "content": text, "endCallAfterSpoken": False}
                else:
                    # The echo custom-llm streams this content straight back, so the
                    # model's output -- and therefore the text the voice pipeline sees --
                    # is exactly this string.
                    frame = {
                        "type": "add-message",
                        "message": {"role": "user", "content": text},
                        "triggerResponseEnabled": True,
                    }
                step = Step(label=label, path=path, text=text, sent_at_s=now())
                await ws.send(json.dumps(frame))
                deadline = time.monotonic() + AUDIO_TIMEOUT_S
                while time.monotonic() < deadline:
                    if len(audible_at) > mark:
                        step.first_audio_s = audible_at[mark]
                        break
                    if closed.is_set():
                        break
                    await asyncio.sleep(0.004)
                run.steps.append(step)
                if closed.is_set():
                    run.error = "Vapi closed the call mid-probe"
                    break
            await wait_idle(limit=25.0)
        except websockets.ConnectionClosed as exc:
            run.error = f"Vapi closed the call mid-probe: {exc}"
        except OSError as exc:
            run.error = f"transport error: {type(exc).__name__}: {exc}"
        finally:
            send_task.cancel()
            recv_task.cancel()
            for task in (send_task, recv_task):
                with contextlib.suppress(
                    asyncio.CancelledError, websockets.ConnectionClosed, OSError
                ):
                    await task


def run_cache_probe(
    client: VapiClient,
    assistant_id: str,
    *,
    name: str,
    model_url: str,
    plan: list[tuple[str, str, str]],
) -> ProbeRun:
    """One websocket call. ``model`` is the only override: the voice is the live one."""
    created = client.create_websocket_call(
        assistant_id,
        name=name[:40],
        sample_rate=SAMPLE_RATE,
        assistant_overrides={
            "model": {"provider": "custom-llm", "model": "probe-echo", "url": model_url}
        },
    )
    run = ProbeRun(call_id=str(created["id"]))
    try:
        asyncio.run(_drive(str(created["transport"]["websocketCallUrl"]), plan, run))
    finally:
        client.end_call(run.call_id)
    return run


def _parse_plan(rows: list[str]) -> list[tuple[str, str, str]]:
    plan: list[tuple[str, str, str]] = []
    for row in rows:
        label, path, text = row.split("=", 2)
        if path not in ("say", "llm"):
            raise SystemExit(f"path must be say|llm, got {path!r}")
        plan.append((label, path, text))
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assistant", default=ASSISTANT_ID)
    parser.add_argument("--model-url", help="echo custom-llm base url; required to place a call")
    parser.add_argument("--name", default="vhv-cache-probe")
    parser.add_argument("--step", action="append", metavar="LABEL=PATH=TEXT", help="say|llm")
    parser.add_argument(
        "--from-call",
        help="classify an existing call id's server-side log instead of placing a call",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    with VapiClient(load_api_key()) as client:
        if args.from_call:
            run, call_id = None, args.from_call
        else:
            if not args.model_url or not args.step:
                parser.error("--model-url and --step are required unless --from-call is given")
            run = run_cache_probe(
                client,
                args.assistant,
                name=args.name,
                model_url=args.model_url,
                plan=_parse_plan(args.step),
            )
            call_id = run.call_id
            print(json.dumps(run.as_dict(), indent=2))
        # The log artifact is written asynchronously after the call ends, so a run that
        # just finished has to be waited for rather than read once and reported empty.
        # The wait ends on the first utterance whose outcome the log actually settles:
        # the artifact's payload tier can be absent for good (see `classify`), so
        # waiting for a key instead would burn the whole window on every such call.
        rows: list[dict[str, Any]] = []
        deadline = time.monotonic() + LOG_WAIT_S
        while True:
            rows = client.call_logs(call_id)
            if any(u["outcome"] != "UNKNOWN" for u in classify(rows)):
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(10.0)

    result = {
        "call_id": call_id,
        "run": None if run is None else run.as_dict(),
        "utterances": classify(rows),
        "verdict": verdict(rows),
    }
    print(json.dumps(result, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if run is None or not run.error else 1


if __name__ == "__main__":
    raise SystemExit(main())
