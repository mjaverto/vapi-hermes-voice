"""What does Vapi ACTUALLY send at the end of a call, and when?

Run against a real websocket-transport call. No phone number is created and no E.164
number is dialled, so nothing rings; the callee is this process.

Why this exists rather than a reading of the docs. Twice in one day this account has
contradicted Vapi's published behaviour: ``assistant.speechStarted`` does not fire for a
Live Call Control ``say`` here, and the documented ``source: "force-say"`` is not what
this account emits. ``end-of-call-report`` is the payload an entire reporting design
would rest on, so it gets measured.

The obstacle is that assistant ``b39379dc`` carries no ``server`` field, so the real
adapter never receives one of these -- and editing that assistant is not this
workstream's decision. The way round it is that Vapi accepts ``server`` inside a
PER-CALL ``assistantOverrides`` (this probe is also the verification of that), so the
events can be pointed at a disposable sink for one call and nothing persistent changes.

    python -m tests.e2e.end_of_call_probe --sink https://<tunnel>/vapi/server \
        --secret-file /tmp/pcr-secret

What it reports: which message types arrive, in what order, how long after the call ends,
and -- the question the design turns on -- whether ``analysis``/``summary`` carry anything
on this account.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from .audio_script import Scenario, Step, synthesize
from .vapi_live import ASSISTANT_ID, VapiClient, load_api_key
from .ws_call import drive_call

# Short and purposeful: the point is to produce a real transcript with a real booking
# line in it, not to measure timing of anything the callee hears. One exchange is enough
# for `transcript` to be non-empty, which is the only property the reporter measures.
PROBE_SCENARIO = Scenario(
    name="end-of-call",
    description="one real exchange, so the settled call has a non-empty transcript",
    steps=(
        Step(text="Hello, Riverside Veterinary.", gap_after_s=9.0, label="greeting"),
        Step(text="Tuesday at nine in the morning works for us.", gap_after_s=9.0, label="offer"),
    ),
    lead_silence_s=0.4,
    tail_silence_s=1.0,
    # Nothing here asks a question that needs a Hermes lookup, so no acknowledgement is
    # due and the harness must not score one as missing.
    ack_turn_index=None,
)

# The dynamic variables an outbound task call carries. Present so the probe can confirm
# they survive into the end-of-call payload -- the report reads the objective and the
# counterparty from exactly here, and if Vapi does not echo them back the report has no
# objective to judge against.
PROBE_VARIABLES: dict[str, str] = {
    "call_purpose": "Move Marvin's recheck to Tuesday morning",
    "callee": "Riverside Veterinary Clinic",
    "spoken_reason": "I'm calling to move Marvin's recheck to Tuesday morning",
}


def _voice() -> str:
    return "Samantha" if sys.platform == "darwin" else "default"


async def _run(args: argparse.Namespace) -> int:
    secret = Path(args.secret_file).read_text().strip()
    if len(secret) < 16:
        print("refusing to run: the sink secret is shorter than 16 chars", file=sys.stderr)
        return 2

    try:
        synthesize(PROBE_SCENARIO.steps[0].text, voice=_voice())
    except Exception as exc:  # noqa: BLE001 - any TTS failure is the same verdict here
        print(f"no local text-to-speech, cannot drive the callee side: {exc}", file=sys.stderr)
        return 2

    client = VapiClient(load_api_key())
    with client:
        call = client.create_websocket_call(
            ASSISTANT_ID,
            name=f"pcr-eocr-{int(time.time()) % 100000}",
            sample_rate=16_000,
            assistant_overrides={
                # The whole point of the probe: no persistent edit to the assistant.
                "server": {"url": args.sink, "secret": secret},
                # Asked for explicitly. `end-of-call-report` is in Vapi's default set,
                # but this account's assistant has `serverMessages: null`, and "null
                # means the documented default" is exactly the assumption that has
                # already cost an afternoon here.
                "serverMessages": [
                    "end-of-call-report",
                    "status-update",
                    "speech-update",
                    "transcript",
                ],
                "variableValues": PROBE_VARIABLES,
            },
        )
        call_id = call["id"]
        ws_url = call["transport"]["websocketCallUrl"]
        print(f"call {call_id}")
        print(f"transcript will be driven for ~{PROBE_SCENARIO.scripted_seconds:.1f}s")

        hung_up_at = 0.0
        try:
            await drive_call(ws_url, PROBE_SCENARIO, voice=_voice())
        finally:
            hung_up_at = time.time()
            client.end_call(call_id)
        print(f"hung up at wall {hung_up_at:.3f}")

        # `messages[]` and the artifact are written asynchronously after the call ends.
        settled = client.await_transcript(call_id, timeout_s=args.settle_s)
        out: dict[str, Any] = {
            "call_id": call_id,
            "hung_up_at": hung_up_at,
            "settled_call": settled,
        }
        Path(args.out).write_text(json.dumps(out, indent=2, default=str))
        print(f"settled call object written to {args.out}")
        print(f"  status={settled.get('status')} endedReason={settled.get('endedReason')}")
        print(f"  summary={settled.get('summary')!r}")
        print(f"  analysis={settled.get('analysis')!r}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sink", required=True, help="public URL of the recorder")
    parser.add_argument("--secret-file", required=True, help="file holding the sink secret")
    parser.add_argument("--out", default="/tmp/pcr-settled-call.json")  # noqa: S108
    parser.add_argument("--settle-s", type=float, default=120.0)
    args = parser.parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
