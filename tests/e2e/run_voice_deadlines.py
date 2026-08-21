#!/usr/bin/env python3
"""Live voice-deadline regression harness. Places a real Vapi call; costs real money.

    uv run python -m tests.e2e.run_voice_deadlines --dry-run
    uv run python -m tests.e2e.run_voice_deadlines --scenario deadlines
    uv run python -m tests.e2e.run_voice_deadlines --from-call 01a02524-...   # re-score

No phone number is ever supplied, so nothing rings. See ``tests/e2e/README.md`` for the
cost per run, what this can prove, and what it structurally cannot.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

# Runnable as `python tests/e2e/run_voice_deadlines.py` as well as `python -m ...`.
if __package__ in (None, ""):  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "tests.e2e"

from .audio_script import SAMPLE_RATE, SCENARIOS, AudioUnavailable, Scenario, synthesize
from .deadlines import (
    Budgets,
    TimelineUnitError,
    evaluate,
    evaluate_transport,
    r1_transport_scope,
    render_table,
)
from .vapi_live import (
    ASSISTANT_ID,
    PreflightFailure,
    PreflightResult,
    VapiClient,
    VapiError,
    load_api_key,
    preflight,
)
from .ws_call import drive_call

# Measured on two live websocket calls of ~20 s: $0.0191 and $0.0189, almost all of it
# the Vapi platform minute rate plus Deepgram. The `deadlines` scenario is ~31 s.
COST_PER_RUN_USD = 0.05

# `Budgets` uses slots, so class-attribute access returns a descriptor rather than the
# field default. argparse needs the values, so read them off one instance.
DEFAULTS = Budgets()


def _ack_phrases(explicit: Path | None) -> tuple[list[str], str]:
    """The acknowledgement pool to match against, and where it came from.

    Never invented. The R2 check has to know the exact phrases the deployed adapter can
    say; guessing them would turn "the acknowledgement never came" and "my list is out
    of date" into the same result.
    """
    if explicit is not None:
        phrases = [ln.strip() for ln in explicit.read_text().splitlines() if ln.strip()]
        if not phrases:
            raise SystemExit(f"{explicit} contained no phrases")
        return phrases, f"file:{explicit}"
    try:
        from vapi_hermes_voice.config import Settings

        # Placeholders only, so that filler_phrases (and any VHV_ override of it)
        # resolves through the adapter's real config. They must satisfy the credential
        # validators, which enforce a 16-character minimum.
        settings = Settings(  # type: ignore[call-arg]
            hermes_base_url="http://unused.invalid",
            hermes_api_key="e2e-harness-placeholder",
            adapter_api_key="e2e-harness-placeholder",
        )
    except Exception as exc:  # noqa: BLE001 - any config error must be actionable here
        raise SystemExit(
            "could not resolve the acknowledgement phrase pool from adapter settings "
            f"({type(exc).__name__}: {exc}).\nPass --ack-phrases-file with one phrase per "
            "line, matching VHV_FILLER_PHRASES on the deployed adapter."
        ) from exc
    return list(settings.filler_phrases), "vapi_hermes_voice.config.Settings"


def _print_plan(scenario: Scenario, voice: str) -> float:
    """Synthesise the callee's audio and print the script. Returns total seconds."""
    print(f"scenario: {scenario.name} -- {scenario.description}")
    print(f"  lead silence      {scenario.lead_silence_s:>6.2f}s")
    total = scenario.lead_silence_s
    for step in scenario.steps:
        pcm = synthesize(step.text, voice=voice)
        speech_s = len(pcm) / (SAMPLE_RATE * 2)
        total += speech_s + step.gap_after_s
        print(f'  say  {speech_s:>6.2f}s   "{step.text}"  [{step.label}]')
        print(f"  gap  {step.gap_after_s:>6.2f}s   (silence the deadline is measured across)")
    total += scenario.tail_silence_s
    print(f"  tail silence      {scenario.tail_silence_s:>6.2f}s")
    print(f"  call duration     {total:>6.2f}s   ~${COST_PER_RUN_USD:.2f} per run")
    return total


def _report_preflight(pf: PreflightResult) -> None:
    print("preflight")
    print(f"  adapter origin  {pf.model_origin}")
    print(f"  GET /healthz    {pf.healthz}")
    print(f"  GET /readyz     {pf.readyz}")
    print(f"  firstMessageMode {pf.first_message_mode!r}")
    print(f"  transcriber     {json.dumps(pf.transcriber)[:200]}")
    for w in pf.warnings:
        print(f"  [WARN] {w}")


def _resolve_first_message(
    args: argparse.Namespace, call: dict[str, Any], pf: PreflightResult | None
) -> str | None:
    """The assistant's static ``firstMessage``, or a hard error rather than a guess.

    The r1_provenance check turns on this string: if it is wrongly taken to be absent,
    provenance passes and the run claims to have measured an adapter it never touched.
    So an unknown firstMessage is refused, not defaulted.
    """
    if args.first_message is not None:
        return args.first_message or None
    if pf is not None:
        return pf.first_message
    embedded = call.get("assistant")
    if isinstance(embedded, dict) and "firstMessage" in embedded:
        return embedded.get("firstMessage") or None
    assistant_id = call.get("assistantId") or args.assistant
    if args.from_file is None and assistant_id:
        with VapiClient(load_api_key()) as client:
            return client.get_assistant(str(assistant_id)).get("firstMessage") or None
    raise PreflightFailure(
        f"call {call.get('id')} carries no embedded assistant, so its static firstMessage "
        "is unknown and the R1 reply cannot be attributed to the adapter rather than to "
        "Vapi. Pass --first-message '<the configured string>' (or --first-message '' if "
        "the assistant has none) to score this payload."
    )


def _run_live(
    args: argparse.Namespace, scenario: Scenario, phrases: list[str]
) -> tuple[dict[str, Any], PreflightResult | None, Any]:
    client = VapiClient(load_api_key())
    with client:
        assistant = client.get_assistant(args.assistant)
        pf: PreflightResult | None = None
        if args.skip_preflight:
            print(
                "[WARN] --skip-preflight: if the adapter is down, R1 will still PASS from "
                "Vapi's static firstMessage and every number below will be a lie."
            )
        else:
            pf = preflight(assistant, require_ready=not args.allow_degraded)
            _report_preflight(pf)

        created = client.create_websocket_call(
            args.assistant,
            name=f"vhv-e2e {scenario.name} {time.strftime('%Y-%m-%dT%H:%M:%S')}",
            sample_rate=SAMPLE_RATE,
        )
        call_id = str(created["id"])
        ws_url = str(created["transport"]["websocketCallUrl"])
        print(f"\ncall {call_id} created (transport vapi.websocket, no phone number)")
        try:
            observation = asyncio.run(drive_call(ws_url, scenario, voice=args.voice))
        finally:
            client.end_call(call_id)
        if observation.error:
            print(f"[WARN] websocket driver: {observation.error}")
        if not observation.pacing_ok:
            print(
                f"[WARN] send pacing ran {observation.max_send_lateness_s * 1000:.0f} ms late; "
                "the callee's silence gaps were not the scripted length, so treat the "
                "harness-side clock as unreliable for this run"
            )
        print("waiting for Vapi to finalise the call transcript...")
        call = client.await_transcript(call_id)
        return call, pf, observation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure the voice deadlines a callee actually experiences.",
    )
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="deadlines")
    parser.add_argument("--assistant", default=ASSISTANT_ID)
    parser.add_argument("--voice", default="Samantha", help="macOS `say` voice for the callee")
    parser.add_argument("--json-out", type=Path, help="write the machine-readable result here")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="synthesise audio, print the script and run the preflight; place no call",
    )
    parser.add_argument("--from-call", help="score an existing Vapi call id instead of placing one")
    parser.add_argument(
        "--from-file", type=Path, help="score a saved GET /call/{id} payload; no network"
    )
    parser.add_argument("--ack-phrases-file", type=Path)
    parser.add_argument(
        "--first-message",
        help="the assistant's configured static firstMessage, for offline scoring; pass "
        "an empty string to assert the assistant has none",
    )
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument(
        "--allow-degraded",
        action="store_true",
        help="proceed when the adapter is up but Hermes is unreachable (/readyz 503)",
    )
    parser.add_argument(
        "--no-require-provenance",
        action="store_true",
        help="downgrade the r1_provenance check to a note instead of a failure",
    )
    parser.add_argument("--reason-deadline", type=float, default=DEFAULTS.reason_deadline_s)
    parser.add_argument("--ack-deadline", type=float, default=DEFAULTS.ack_deadline_s)
    parser.add_argument("--ack-cooldown", type=float, default=DEFAULTS.ack_cooldown_s)
    parser.add_argument("--max-turn-gap", type=float, default=DEFAULTS.max_turn_gap_s)
    args = parser.parse_args(argv)

    scenario = SCENARIOS[args.scenario]
    budgets = Budgets(
        reason_deadline_s=args.reason_deadline,
        ack_deadline_s=args.ack_deadline,
        ack_cooldown_s=args.ack_cooldown,
        max_turn_gap_s=args.max_turn_gap,
    )
    phrases, phrase_source = _ack_phrases(args.ack_phrases_file)

    pf: PreflightResult | None = None
    observation = None
    try:
        if args.from_file is not None:
            call = json.loads(args.from_file.read_text())
        elif args.from_call is not None:
            with VapiClient(load_api_key()) as client:
                call = client.get_call(args.from_call)
        else:
            try:
                total_s = _print_plan(scenario, args.voice)
            except AudioUnavailable as exc:
                print(f"cannot build the callee's audio: {exc}", file=sys.stderr)
                return 2
            if args.dry_run:
                if not args.skip_preflight:
                    with VapiClient(load_api_key()) as client:
                        _report_preflight(
                            preflight(
                                client.get_assistant(args.assistant),
                                require_ready=not args.allow_degraded,
                            )
                        )
                print(f"\n--dry-run: no call placed. A live run would last ~{total_s:.0f}s.")
                return 0
            call, pf, observation = _run_live(args, scenario, phrases)
    except PreflightFailure as exc:
        print(f"\nPREFLIGHT FAILED\n{exc}", file=sys.stderr)
        return 3
    except VapiError as exc:
        print(f"\nVAPI API ERROR\n{exc}", file=sys.stderr)
        return 4
    except AudioUnavailable as exc:
        print(f"\nAUDIO UNAVAILABLE\n{exc}", file=sys.stderr)
        return 2

    try:
        first_message = _resolve_first_message(args, call, pf)
    except PreflightFailure as exc:
        print(f"\nCANNOT ATTRIBUTE R1\n{exc}", file=sys.stderr)
        return 3
    try:
        report = evaluate(
            call,
            phrases=phrases,
            budgets=budgets,
            first_message=first_message,
            require_adapter_provenance=not args.no_require_provenance,
            ack_turn_index=scenario.ack_turn_index,
            # Only meaningful for a live run: a re-scored call has no script.
            expected_callee_turns=None if observation is None else len(scenario.steps),
        )
    except TimelineUnitError as exc:
        print(f"\nTIMELINE UNUSABLE\n{exc}", file=sys.stderr)
        return 5

    # Whether this call's transport could even exercise R1's own code path at all --
    # reported explicitly (see r1_transport_scope's docstring) rather than left to look
    # like either a clean PASS or an adapter regression on the vapi.websocket transport
    # this harness always uses (it never places a PSTN call).
    report.checks.append(r1_transport_scope(call, report))

    # Attribute R2 between the adapter and Vapi. Only possible with the transport clock,
    # so it runs only on a live call, and it may overturn the story the spoken timeline
    # tells: the adapter can meet its deadline and the callee still hear nothing.
    if observation is not None and scenario.ack_turn_index is not None:
        scripted = observation.steps[scenario.ack_turn_index] if observation.steps else None
        transport_checks, transport_notes = evaluate_transport(
            observation.events,
            callee_turn_end_s=None if scripted is None else scripted.speech_end_s,
            spoken_ack_count=len(report.acks),
            phrases=phrases,
            budgets=budgets,
        )
        report.checks.extend(transport_checks)
        report.notes.extend(transport_notes)

    print(render_table(report, phrases=phrases))
    if observation is not None:
        print("what the callee's own microphone/speaker measured (independent clock)")
        for step in observation.steps:
            gap = step.heard_gap_s
            print(
                f"  {step.label:<16} stopped speaking {step.speech_end_s:>7.3f}s  "
                f"first audible reply "
                f"{f'{step.first_reply_audio_s:.3f}s' if step.first_reply_audio_s else 'NONE':>9}"
                f"  heard gap {f'{gap:.3f}s' if gap is not None else 'n/a':>9}"
            )
        print()

    result: dict[str, Any] = {
        "harness": "tests/e2e/run_voice_deadlines.py",
        "scenario": scenario.name,
        "call_id": call.get("id"),
        "call_type": call.get("type"),
        "ended_reason": call.get("endedReason"),
        "cost_usd": call.get("cost"),
        "budgets": {
            "reason_deadline_s": budgets.reason_deadline_s,
            "ack_deadline_s": budgets.ack_deadline_s,
            "ack_cooldown_s": budgets.ack_cooldown_s,
            "ack_storm_window_s": budgets.ack_storm_window_s,
            # Derived from ack_cooldown_s, not independently configurable -- see
            # Budgets.ack_storm_max.
            "ack_storm_max": budgets.ack_storm_max,
            "max_turn_gap_s": budgets.max_turn_gap_s,
        },
        "ack_phrase_source": phrase_source,
        "ack_phrases": phrases,
        "preflight": None
        if pf is None
        else {
            "model_origin": pf.model_origin,
            "healthz": pf.healthz,
            "readyz": pf.readyz,
            "first_message_mode": pf.first_message_mode,
            "warnings": list(pf.warnings),
        },
        "callee_clock": None if observation is None else observation.as_dict(),
        **report.as_dict(),
    }
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"machine-readable result: {args.json_out}")

    failed = [c.id for c in report.checks if c.verdict == "fail"]
    print("RESULT: PASS" if report.ok else f"RESULT: FAIL ({', '.join(failed)})")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
