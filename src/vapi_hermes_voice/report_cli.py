"""``python -m vapi_hermes_voice.report_cli``: a Vapi call on stdin, the report on stdout.

This is the seam between the two components. Hermes owns the conversation with the
principal and owns the booking; it does NOT own an honest opinion about whether a call
achieved anything, because an agent summarising its own failed errand writes something
that reads like progress. So the division is: Hermes fetches the call (it placed it, and
it has the Vapi MCP server), Hermes says what it booked, and this decides the verdict and
the wording.

    hermes> get_call(id)  ->  call.json
    hermes> python -m vapi_hermes_voice.report_cli --booked-what "Marvin's recheck" \
                --booked-when "Tue 26 Aug, 9:00am" --booked-with "Riverside Vet" < call.json
    hermes> (send stdout to the principal over Telegram)

No network and no API key by design. Everything this needs is already in the payload
Hermes holds, and an adapter that could reach the Vapi management API would be a
privilege this process has no other use for.

EXIT CODES are the report's verdict, so a caller that ignores stdout still cannot mistake
a failure for a success -- and a shell `&&` chain does the right thing by default:

    0  something was booked; the text says what and when
    1  the call happened and nothing was booked
    2  the call failed; there was no conversation
    3  outcome unknown -- placed, never observed to end, or nothing vouched for

3 is the DEFAULT when no booking flags are passed, which is the important case: a caller
that forgets to say what it booked gets "unknown" and a non-zero exit, never a confident
silence.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from .call_report import Booking, BookingUnknown, Claim, NothingBooked, build_report

EXIT_CODES: dict[str, int] = {"booked": 0, "no_booking": 1, "failed": 2, "unknown": 3}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vapi_hermes_voice.report_cli",
        description="Render the post-call report for the principal from a Vapi call payload.",
        epilog=(
            "Accepts either a call object (GET /call/{id}) or an end-of-call-report"
            " server message. Exit code is the verdict: 0 booked, 1 nothing booked,"
            " 2 call failed, 3 outcome unknown."
        ),
    )
    # Three-way and mutually exclusive, mirroring call_report.Claim. There is no way to
    # spell "nothing booked" by omission: that requires --nothing-booked, and omitting
    # everything means "I do not know", which is the truth about a caller that said
    # nothing.
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--booked-what",
        metavar="TEXT",
        help="what was committed. Requires --booked-when and --booked-with.",
    )
    group.add_argument(
        "--nothing-booked",
        action="store_true",
        help="vouch that no commitment was made. Only pass this if you actually know.",
    )
    parser.add_argument("--booked-when", metavar="TEXT", help="the exact date and time agreed.")
    parser.add_argument("--booked-with", metavar="TEXT", help="who the commitment is with.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit {outcome, objective_met, text} instead of the bare message text.",
    )
    return parser


def _claim_from(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Claim:
    if args.nothing_booked:
        return NothingBooked()
    if args.booked_what is None:
        return BookingUnknown()
    # A partial booking is refused rather than padded with "unspecified". The whole
    # point of the report is that the principal can act on the time, and "BOOKED --
    # unspecified" is exactly the useless confidence this module exists to prevent.
    if not args.booked_when or not args.booked_with:
        parser.error("--booked-what requires both --booked-when and --booked-with")
    return Booking(what=args.booked_what, when=args.booked_when, with_whom=args.booked_with)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    claim = _claim_from(args, parser)

    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        # A malformed payload is NOT reported as "nothing booked": we know nothing about
        # the call, and saying anything about its outcome would be inventing it.
        print(f"could not parse the call payload as JSON: {exc}", file=sys.stderr)
        return EXIT_CODES["unknown"]
    if not isinstance(payload, dict):
        print("expected a JSON object (a Vapi call or an end-of-call-report)", file=sys.stderr)
        return EXIT_CODES["unknown"]

    report = build_report(payload, claim=claim)
    if args.json:
        out: dict[str, Any] = {
            "outcome": report.outcome,
            "objective_met": report.objective_met,
            "call_id": report.facts.call_id,
            "text": report.text,
        }
        print(json.dumps(out, indent=2))
    else:
        print(report.text)
    return EXIT_CODES[report.outcome]


if __name__ == "__main__":
    raise SystemExit(main())
