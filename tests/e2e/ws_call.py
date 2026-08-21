"""Drives the callee side of a Vapi websocket call and times what the callee hears.

This is the second, independent clock in the harness. ``deadlines.py`` scores Vapi's own
``messages[]`` timeline, which is authoritative but is Vapi grading its own homework.
Here the harness holds the microphone and the speaker: it knows to the frame when it
stopped emitting audible audio, and when audible audio came back. Those two numbers are
the callee's actual experience, including transcriber lag, LLM time, TTS time and
transport buffering, and they are reported alongside the API timeline so a divergence
between the two is visible rather than invisible.

Audio is streamed in real time. That is not incidental: sending 30 s of audio as fast as
the socket accepts it would give the transcriber no silence to detect end-of-turn with,
and end-of-turn detection is the thing under test.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from array import array
from dataclasses import dataclass, field
from typing import Any

import websockets

from .audio_script import SAMPLE_RATE, SAMPLE_WIDTH, Scenario, silence, synthesize

__all__ = ["StepObservation", "WsObservation", "drive_call"]

FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
FRAME_BYTES = FRAME_SAMPLES * SAMPLE_WIDTH

# 16-bit full scale is 32768. `say` emits digital silence between words, and Vapi's
# downstream silence is also at or near zero, so a low threshold is safe and a high one
# would clip the quiet onset of a word.
AUDIBLE_PEAK = 500

# Scheduling slop above which the send pacing itself has corrupted the measurement.
MAX_TOLERABLE_LATENESS_S = 0.100


def _peak(frame: bytes) -> int:
    samples = array("h")
    samples.frombytes(frame[: len(frame) - len(frame) % SAMPLE_WIDTH])
    return max((abs(s) for s in samples), default=0)


def _last_audible_offset_s(pcm: bytes) -> float:
    """Seconds into ``pcm`` at which the last audible frame ends.

    Text-to-speech output carries trailing silence. Counting it as speech would make
    every measured gap shorter than the callee's real wait, so the audible end is found
    rather than assumed.
    """
    total = len(pcm) // FRAME_BYTES
    for i in range(total - 1, -1, -1):
        if _peak(pcm[i * FRAME_BYTES : (i + 1) * FRAME_BYTES]) >= AUDIBLE_PEAK:
            return (i + 1) * FRAME_MS / 1000
    return len(pcm) / (SAMPLE_RATE * SAMPLE_WIDTH)


@dataclass
class StepObservation:
    label: str
    text: str
    send_start_s: float
    speech_end_s: float
    first_reply_audio_s: float | None = None

    @property
    def heard_gap_s(self) -> float | None:
        if self.first_reply_audio_s is None:
            return None
        return self.first_reply_audio_s - self.speech_end_s

    def as_dict(self) -> dict[str, Any]:
        gap = self.heard_gap_s
        return {
            "label": self.label,
            "text": self.text,
            "send_start_s": round(self.send_start_s, 3),
            "speech_end_s": round(self.speech_end_s, 3),
            "first_reply_audio_s": (
                None if self.first_reply_audio_s is None else round(self.first_reply_audio_s, 3)
            ),
            "heard_gap_s": None if gap is None else round(gap, 3),
        }


@dataclass
class WsObservation:
    """Everything the harness saw from the callee's seat, on its own monotonic clock."""

    steps: list[StepObservation] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    audio_frames_in: int = 0
    audible_frames_in: int = 0
    max_send_lateness_s: float = 0.0
    # ``time.time()`` at the instant `at_s`/`speech_end_s` were zeroed. Every other
    # number here is on this harness's own monotonic clock, which is a duration and not
    # a point in time -- so nothing measured on ANOTHER machine can be placed on it.
    # The adapter's acknowledgement record (GET /debug/acks/{call_ref}) is measured on
    # another machine, and this is the one anchor that lets its wall-clock stamps be
    # printed on this timeline. Only as accurate as the two hosts' NTP agreement, so it
    # is reported and never scored: see `deadlines.evaluate_transport`.
    epoch_origin_s: float = 0.0
    error: str | None = None

    @property
    def pacing_ok(self) -> bool:
        return self.max_send_lateness_s <= MAX_TOLERABLE_LATENESS_S

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": [s.as_dict() for s in self.steps],
            "audio_frames_in": self.audio_frames_in,
            "audible_frames_in": self.audible_frames_in,
            "max_send_lateness_s": round(self.max_send_lateness_s, 4),
            "epoch_origin_s": round(self.epoch_origin_s, 3),
            "pacing_ok": self.pacing_ok,
            "error": self.error,
            "events": self.events,
        }


async def drive_call(ws_url: str, scenario: Scenario, *, voice: str) -> WsObservation:
    """Stream ``scenario`` down ``ws_url`` in real time, recording both directions."""
    # Synthesise everything before opening the socket: a cold `say` invocation takes
    # hundreds of milliseconds and the call is already billing once connected.
    pcm = [(step, synthesize(step.text, voice=voice)) for step in scenario.steps]

    obs = WsObservation()
    loop_start = 0.0

    def now() -> float:
        return time.monotonic() - loop_start

    async with websockets.connect(ws_url, max_size=None) as ws:
        loop_start = time.monotonic()
        # Same instant, on the one clock another host can also read. See the field.
        obs.epoch_origin_s = time.time()
        stop = asyncio.Event()
        # The step currently awaiting a reply, so the receiver can stamp the first
        # audible downstream frame against the right utterance.
        pending: list[StepObservation] = []

        async def receiver() -> None:
            try:
                async for message in ws:
                    if isinstance(message, bytes | bytearray):
                        obs.audio_frames_in += 1
                        if _peak(bytes(message)) >= AUDIBLE_PEAK:
                            obs.audible_frames_in += 1
                            at = now()
                            for step in pending:
                                if step.first_reply_audio_s is None and at >= step.speech_end_s:
                                    step.first_reply_audio_s = at
                        continue
                    try:
                        payload = json.loads(message)
                    except (TypeError, ValueError):
                        payload = {"type": "unparsed", "raw": str(message)[:500]}
                    obs.events.append({"at_s": round(now(), 3), "event": payload})
            except websockets.ConnectionClosed:
                pass
            finally:
                stop.set()

        async def send_pcm(buf: bytes) -> None:
            """Feed ``buf`` at wall-clock speed, tracking how late we ever ran."""
            nonlocal_deadline = time.monotonic()
            for off in range(0, len(buf), FRAME_BYTES):
                frame = buf[off : off + FRAME_BYTES]
                if len(frame) < FRAME_BYTES:
                    frame = frame + b"\x00" * (FRAME_BYTES - len(frame))
                await ws.send(frame)
                nonlocal_deadline += FRAME_MS / 1000
                lateness = time.monotonic() - nonlocal_deadline
                obs.max_send_lateness_s = max(obs.max_send_lateness_s, lateness)
                if lateness < 0:
                    await asyncio.sleep(-lateness)

        recv_task = asyncio.create_task(receiver())
        try:
            await send_pcm(silence(scenario.lead_silence_s))
            for step, speech in pcm:
                start = now()
                audible_end = start + _last_audible_offset_s(speech)
                record = StepObservation(
                    label=step.label, text=step.text, send_start_s=start, speech_end_s=audible_end
                )
                obs.steps.append(record)
                pending.append(record)
                await send_pcm(speech)
                await send_pcm(silence(step.gap_after_s))
            await send_pcm(silence(scenario.tail_silence_s))
        except websockets.ConnectionClosed as exc:
            obs.error = f"Vapi closed the call mid-script: {exc}"
        except OSError as exc:  # transport died under us
            obs.error = f"transport error while streaming: {type(exc).__name__}: {exc}"

        # Let any in-flight downstream audio and the closing events land.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=2.0)
        recv_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await recv_task

    return obs
