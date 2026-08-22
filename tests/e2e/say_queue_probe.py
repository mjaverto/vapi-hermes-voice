"""What Vapi does with a ``say`` that arrives WHILE it is still speaking a previous one.

The question this answers, and why it is load-bearing: ``turns.py`` now speaks two
things down Live Call Control on one turn -- a reassurance during a long silence, then
the answer -- and the second can be POSTed while the first is still being rendered.
Three outcomes are possible and they are not equally acceptable:

    serialised  the second waits for the first to finish.        Fine.
    truncating  the second cuts the first off mid-word.          Acceptable: the
                                                                 answer beating a
                                                                 holding phrase is
                                                                 the right winner.
    overlapping two utterances mixed into one audio stream.      WORSE than the dead
                                                                 air being removed.
    dropped     the second is silently discarded.                CATASTROPHIC: the
                                                                 reassurance would be
                                                                 eating the answer.

Reasoning could not settle it. ``pipeline.sayQueuePush`` is named like a queue and 282
complete ``botSpeechStarted``/``botSpeechStopped`` pairs across 83 archived calls
contain zero overlaps -- but the tightest ``sayQueuePush``-to-``sayQueuePush`` gap in
that whole archive is 5.20 s, longer than any holding phrase takes to speak, so not one
of those samples ever put a push inside an utterance. The archive proves Vapi does not
overlap when it is not asked to.

So this asks it to. It places ONE websocket-transport call (no ``customer``, no
``phoneNumberId``: no PSTN leg can exist and nothing rings -- see README), streams
silence to keep it alive, and POSTs two ``say`` frames with a gap far shorter than the
first line takes to speak. Evidence is taken from two independent places:

  * the downstream audio this process receives, which is what a callee would hear, and
  * Vapi's own server-side call log, which names the events.

Run:  uv run python -m tests.e2e.say_queue_probe
Needs VAPI_API_KEY. Places a billable call of a few seconds. Not collected by pytest.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import websockets

from .vapi_live import ASSISTANT_ID, VapiClient, load_api_key

SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2
FRAME_MS = 20
FRAME_BYTES = SAMPLE_RATE * SAMPLE_WIDTH * FRAME_MS // 1000
# Same threshold ws_call.py uses to call a frame audible rather than line noise.
AUDIBLE_PEAK = 500

# Long enough that a second `say` sent 0.8 s later is unambiguously inside it (this
# reads as roughly four seconds of speech), and worded so the two lines cannot be
# mistaken for each other in a transcript.
FIRST_LINE = (
    "This is the first line of the probe, and it is deliberately long enough "
    "to still be speaking when the second one arrives."
)
SECOND_LINE = "Second line."


@dataclass
class Envelope:
    """When this process was receiving audible audio, in call-relative seconds."""

    frames: list[tuple[float, bool]] = field(default_factory=list)

    def runs(self, *, gap_s: float = 0.20) -> list[tuple[float, float]]:
        """Contiguous audible stretches, merging holes shorter than ``gap_s``.

        A hole is silence between frames. TTS output carries short internal pauses
        (between sentences, and inside them), so a run boundary has to be a hole big
        enough to be a gap between UTTERANCES rather than a comma.
        """
        out: list[tuple[float, float]] = []
        for at, audible in self.frames:
            if not audible:
                continue
            if out and at - out[-1][1] <= gap_s:
                out[-1] = (out[-1][0], at)
            else:
                out.append((at, at))
        return out


async def probe(*, name: str, gap_s: float) -> dict[str, Any]:
    api_key = load_api_key()
    client = VapiClient(api_key)
    call = client.create_websocket_call(ASSISTANT_ID, name=name, sample_rate=SAMPLE_RATE)
    call_id = str(call["id"])
    ws_url = str(call["transport"]["websocketCallUrl"])
    # `monitor` is absent from the POST /call response body and present on a GET of the
    # same call, so the control URL has to be read back rather than taken from the
    # creation response. (On the Custom LLM path this never comes up: Vapi puts
    # `call.monitor.controlUrl` in every request body it sends the adapter.)
    control_url = ""
    for _attempt in range(10):
        fetched = client.get_call(call_id)
        control_url = str((fetched.get("monitor") or {}).get("controlUrl") or "")
        if control_url:
            break
        time.sleep(1.0)
    if not control_url:
        client.end_call(call_id)
        raise SystemExit(f"call {call_id} never exposed monitor.controlUrl")
    print(f"call {call_id}")

    envelope = Envelope()
    posts: list[dict[str, Any]] = []
    silence = b"\x00" * FRAME_BYTES

    try:
        async with websockets.connect(ws_url, max_size=None) as ws:
            zero = time.monotonic()

            def now() -> float:
                return time.monotonic() - zero

            stop = asyncio.Event()

            async def receiver() -> None:
                try:
                    async for message in ws:
                        if isinstance(message, bytes | bytearray):
                            buf = bytes(message)
                            peak = max(
                                abs(int.from_bytes(buf[i : i + 2], "little", signed=True))
                                for i in range(0, max(len(buf) - 1, 1), 2)
                            )
                            envelope.frames.append((now(), peak >= AUDIBLE_PEAK))
                except websockets.ConnectionClosed:
                    pass
                finally:
                    stop.set()

            async def keepalive() -> None:
                """Real-time silence upstream, so the call stays connected."""
                deadline = time.monotonic()
                while not stop.is_set():
                    with contextlib.suppress(websockets.ConnectionClosed):
                        await ws.send(silence)
                    deadline += FRAME_MS / 1000
                    await asyncio.sleep(max(0.0, deadline - time.monotonic()))

            recv = asyncio.create_task(receiver())
            send = asyncio.create_task(keepalive())

            async def say(text: str, http: httpx.AsyncClient) -> None:
                at = now()
                response = await http.post(control_url, json={"type": "say", "content": text})
                posts.append(
                    {
                        "at_s": round(at, 3),
                        "returned_s": round(now(), 3),
                        "status": response.status_code,
                        "text": text,
                    }
                )
                print(f"  t={at:5.2f}s  say -> {response.status_code}  {text[:40]!r}")

            # Let any firstMessage the assistant has finish before the probe starts, so
            # the two lines under test are the last two utterances on the call.
            await asyncio.sleep(8.0)
            async with httpx.AsyncClient(timeout=10.0) as http:
                await say(FIRST_LINE, http)
                await asyncio.sleep(gap_s)
                await say(SECOND_LINE, http)
            await asyncio.sleep(12.0)

            stop.set()
            send.cancel()
            recv.cancel()
            for task in (send, recv):
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
    finally:
        client.end_call(call_id)

    await asyncio.sleep(20.0)  # let Vapi finish writing the artifact
    return {"call_id": call_id, "posts": posts, "envelope": envelope, "client": client}


def call_log(client: VapiClient, call_id: str, *, timeout_s: float = 90.0) -> list[dict[str, Any]]:
    """``VapiClient.call_logs``, polled: the artifact is written asynchronously after
    the call ends, so a fresh call id legitimately has no log for a minute or so.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        rows = client.call_logs(call_id)
        if rows or time.monotonic() >= deadline:
            return rows
        time.sleep(3.0)


def report(result: dict[str, Any]) -> None:
    client: VapiClient = result["client"]
    call_id: str = result["call_id"]
    envelope: Envelope = result["envelope"]

    print("\n--- what this process received (the callee's ear)")
    runs = envelope.runs()
    for start, end in runs:
        print(f"  audible {start:6.2f}s -> {end:6.2f}s   ({end - start:.2f}s)")
    print(f"  {len(runs)} audible run(s) over {len(envelope.frames)} frames")

    rows = call_log(client, call_id)
    print("\n--- Vapi's own call log")
    events = []
    for row in rows:
        event = (row.get("attributes") or {}).get("event") or ""
        if event.startswith(("pipeline.botSpeech", "pipeline.sayQueuePush", "assistant.voice")):
            events.append(
                (row.get("time") or row.get("timestamp"), event, str(row.get("body"))[:60])
            )
    for stamp, event, body in events:
        print(f"  {stamp}  {event:34} {body}")

    seq = [event for _stamp, event, _body in events]
    starts = seq.count("pipeline.botSpeechStarted")
    stops = seq.count("pipeline.botSpeechStopped")
    pushes = seq.count("pipeline.sayQueuePush")
    overlapped = False
    open_speech = False
    for event in seq:
        if event == "pipeline.botSpeechStarted":
            overlapped = overlapped or open_speech
            open_speech = True
        elif event == "pipeline.botSpeechStopped":
            open_speech = False

    print("\n--- verdict")
    print(f"  sayQueuePush        {pushes}")
    print(f"  botSpeechStarted    {starts}")
    print(f"  botSpeechStopped    {stops}")
    if pushes < 2:
        print("  INCONCLUSIVE: fewer than two pushes reached the log")
    elif overlapped:
        print("  OVERLAPPING: a second utterance began while the first was still open")
    elif starts >= 2:
        print("  SERIALISED or TRUNCATING: each utterance opened only after the last closed")
    else:
        print("  DROPPED: the second push never became speech")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gap",
        type=float,
        default=0.8,
        help="seconds between the two say POSTs (default 0.8, well inside the first line)",
    )
    parser.add_argument("--name", default="say-queue-overlap-probe")
    args = parser.parse_args()
    report(await probe(name=args.name, gap_s=args.gap))


if __name__ == "__main__":
    asyncio.run(main())
