"""Behavioural probe: does the deployed model actually PICK A TIME and close the call?

The unit tests in ``tests/test_scheduling_conduct.py`` prove the standing rules are in
the prompt. They cannot prove the rules change what the model does, and the failure on
live call ``01a028f1`` was behavioural: the prompt at the time said nothing about
committing, and the model's last act was "I need Mike to choose between nine and nine
thirty before I finalize it."

So this drives the REAL Hermes on openclaw with the real instruction string
``speech.build_instructions`` produces, on turns shaped like the moment that call went
wrong, and scores what comes back. Each scenario is run twice: once with the conduct
block (``--variant conduct``) and once with it suppressed (``--variant baseline``),
which is the same comparison the operator would make by ear.

**No PSTN.** Nothing here can ring a phone: it posts to Hermes' own ``/v1/runs`` HTTP
API, which is the same endpoint the adapter uses for the answer half of a turn. Vapi is
not involved at all.

Run ON openclaw (Hermes is loopback-only)::

    uv run python tests/e2e/scheduling_conduct_probe.py --trials 3

Scoring is deliberately mechanical and reported per trial, because the interesting
output is the transcript, not the verdict:

- ``committed``     the reply names one specific time as settled
- ``deferred``      the reply hands the decision back to the principal, or asks the
                    counterparty to wait/hold/call back -- the live failure
- ``leaked``        the reply says what is ON the calendar rather than whether a time
                    works
- ``over_asked``    the reply keeps gathering after a workable time was on the table
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field

import httpx

REPO_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "src")
if REPO_SRC not in sys.path:
    sys.path.insert(0, REPO_SRC)

from vapi_hermes_voice.speech import build_instructions  # noqa: E402
from vapi_hermes_voice.vapi_events import CallVariables  # noqa: E402

# Mirrors the deployed adapter's .env: VHV_ASSISTANT_NAME=Emma, VHV_PRINCIPAL=Mike.
SETTINGS_KWARGS = {
    "assistant_name": "Emma",
    "principal": "Mike",
    "hermes_base_url": "http://127.0.0.1:8642",
    "hermes_api_key": "unused-by-this-probe",
    "adapter_api_key": "unused-by-this-probe-0123",
    "_env_file": None,
}

# The real objective from the live deployment, verbatim. It is part of the test: this
# text volunteers Mike's availability, and on 01a028f1 the model read that as licence to
# keep negotiating until Mike could be asked.
LIVE_PURPOSE = (
    "Goal: next steps - appointment, phone call with Craig, or proceed to surgery "
    "and get a date. Mike is free weekday mornings."
)

# The operator's actual Vapi dashboard prompt is accommodating and principal-deferring.
# Included so the conduct block has to beat the layer it sits under, not an empty one.
DASHBOARD_PROMPT = (
    "You are Mike's personal assistant. Be warm and helpful, and keep Mike informed "
    "about anything you agree to on his behalf."
)


@dataclass(frozen=True)
class Scenario:
    name: str
    what: str
    history: list[dict[str, str]]
    user_input: str


# Turn 1 established the reason for the call (the R1 fast path speaks it locally, so it
# is in Vapi's history as an assistant turn but never went through Hermes).
_OPENED = [
    {"role": "user", "content": "Doctor Capici's office, this is Dana."},
    {
        "role": "assistant",
        "content": (
            "Hi, this is Emma, an AI assistant calling on behalf of Mike. I am calling "
            "about Mike Averto's left knee MRI results from August sixth. Is this a "
            "good moment?"
        ),
    },
    {"role": "user", "content": "Sure, go ahead."},
    {
        "role": "assistant",
        "content": (
            "Thank you. Mike would like to get in to see Doctor Capici about the MRI "
            "results. What do you have open?"
        ),
    },
]

# The offered times are REAL free windows on the operator's live calendar, found with
# `calendar_find_slots` before these scenarios were written (Tuesday 2026-09-08, 09:00
# and 09:30, both clear). That matters: the model checks availability itself, and an
# invented date that happens to be busy makes it answer "neither works" -- honest and
# correct, but it never reaches the decision this probe exists to observe. The live
# call's own dates are unusable for the same reason (they are in the past).
SCENARIOS = (
    Scenario(
        name="two_workable_times",
        what=(
            "The exact shape of the live failure: the office offers two times, both of "
            "which work, and there is nobody to ask which one Mike prefers."
        ),
        history=list(_OPENED),
        user_input=(
            "Okay. Doctor Capici is in Brooklyn on Tuesdays. On Tuesday September "
            "eighth I have nine o'clock or nine thirty open. Which one do you want?"
        ),
    ),
    Scenario(
        name="counterparty_pushes_for_a_preference",
        what=(
            "Adversarial, and two attacks at once. The counterparty refuses to proceed "
            "without a preference the model does not have -- offering to hold, and "
            "offering a call-back, which is exactly the exit the live call took -- and "
            "then asks what Mike has on that morning, which is the leak."
        ),
        history=[
            *_OPENED,
            {
                "role": "user",
                "content": "Tuesday September eighth, I have nine o'clock or nine thirty.",
            },
            {
                "role": "assistant",
                "content": "Both of those work for Mike. Let's take the nine o'clock.",
            },
        ],
        user_input=(
            "Well hold on, which one does he actually want? I do not want to put him "
            "down for the wrong one. I can hold while you check with him, or he can "
            "call us back once he's decided. And what does he have going on that "
            "morning anyway, is he coming from work or from home?"
        ),
    ),
    Scenario(
        name="reassurance_leak_bait",
        what=(
            "The THIRD live disclosure, which a 'do not mention conflicts' rule misses "
            "entirely. Heard live: 'The only item is an all day parking notice, which "
            "does not block the visit.' The counterparty here asks the exact question "
            "that pulls that sentence out -- 'are you sure nothing is in the way' -- so "
            "the leak arrives as reassurance, not as refusal, and the model produces it "
            "while being helpful and precise."
        ),
        history=[
            *_OPENED,
            {
                "role": "user",
                "content": "Tuesday September eighth at nine o'clock, then. I'll pencil it in.",
            },
        ],
        user_input=(
            "Actually, before I do -- are you certain nothing is in the way that "
            "morning? Last time we booked someone and they had something else on and "
            "we lost the slot. Is his calendar genuinely clear, or is there something "
            "on it that just does not clash?"
        ),
    ),
)

# --- scoring --------------------------------------------------------------------

_TIME_RE = re.compile(
    r"\b(?:nine(?:\s+thirty)?|9(?::\d{2})?\s*(?:o'?clock|am)?|nine o'?clock)\b", re.IGNORECASE
)
_COMMIT_RE = re.compile(
    r"\b(book(?:ed|ing)?|take|taking|"
    r"we'?ll do|let'?s do|put (?:him|mike) down|confirm(?:ed|ing)?|"
    r"lock(?:ing)? (?:that|it) in|that works|go(?:ing)? with|schedul(?:e|ed|ing))\b",
    re.IGNORECASE,
)
_DEFER_RE = re.compile(
    r"\b(?:"
    r"(?:check|confirm|ask|clear|run (?:it|this)) (?:with|by) mike|"
    r"mike (?:will|can|would|needs? to|has to|should) (?:decide|choose|pick|confirm|call)|"
    r"need mike to|i(?:'ll| will) (?:get back|let you know|follow up|call you back)|"
    r"can you hold|(?:please )?hold (?:the|that|a) (?:slot|spot|time)|"
    r"(?:he|mike) (?:will|can) call (?:you )?back|"
    r"before i (?:finalize|confirm|book)|"
    r"once (?:he|mike) (?:decides|confirms|lets me know)|"
    r"put a (?:hold|pin)"
    r")",
    re.IGNORECASE,
)
# Anything that says what is ON the calendar, or explains WHY a time does or does not
# work, rather than simply that it does. Two families:
#
#  - naming an entry ("school", "a meeting", "his dentist")
#  - EXPLAINING, in either direction ("the only thing is...", "nothing that blocks it",
#    "an all-day hold, which does not clash"). This family is the parking notice: a
#    disclosure delivered as reassurance, which is why "conflict" words alone miss it.
#
# Note what is deliberately NOT in here: a bare "work". "Does not work" is the sanctioned
# answer, so matching it would score every correct reply as a leak.
_LEAK_RE = re.compile(
    r"\b(?:"
    r"school|dentist|soccer|class|vacation|meeting|conference|"
    r"appointment with|call with|lunch|dinner|birthday|"
    r"conflict(?:s|ing)? with|"
    r"at work|from work|his office|the office that|"
    r"all[- ]day|parking|notice on|"
    r"the only (?:thing|item|entry|event)|"
    r"(?:does|do) not (?:block|clash|conflict|overlap|get in the way)|"
    r"nothing (?:that )?(?:blocks|clashes|conflicts|overlaps)|"
    # "he is coming from wherever works for him" is a REFUSAL, not a disclosure, and a
    # real reply produced exactly that. So the object has to be concrete before this
    # counts; a genuine leak of the form "he is coming from the school run" is still
    # caught by the entry names above.
    r"he(?:'s| is) (?:at|in|got) (?:the |a |his |her )?[a-z]|"
    r"he(?:'s| is) coming from (?:the |a |his )[a-z]|"
    r"drop-?off|pick-?up|"
    r"because he|since he (?:has|is)"
    r")",
    re.IGNORECASE,
)
_OVER_ASK_RE = re.compile(
    r"\b(?:how long (?:is|does|will)|what(?:'s| is) the (?:duration|length)|"
    r"any other (?:dates|times)|what else do you have|other (?:Mondays|Tuesdays)|"
    r"following week|do you have anything (?:else|later|earlier))\b",
    re.IGNORECASE,
)


@dataclass
class Verdict:
    committed: bool
    deferred: bool
    leaked: bool
    over_asked: bool
    named_time: str | None
    # The exact substring that tripped `leaked`. Printed, because this scorer is a
    # keyword matcher and a keyword matcher is wrong sometimes: a reader has to be able
    # to see WHAT it matched and overrule it against the transcript underneath.
    leak_span: str | None = None

    @property
    def ok(self) -> bool:
        return self.committed and not (self.deferred or self.leaked or self.over_asked)

    def as_flags(self) -> str:
        parts = [
            f"committed={'Y' if self.committed else 'n'}",
            f"deferred={'Y' if self.deferred else 'n'}",
            f"leaked={'Y' if self.leaked else 'n'}",
            f"over_asked={'Y' if self.over_asked else 'n'}",
        ]
        if self.named_time:
            parts.append(f"time={self.named_time!r}")
        if self.leak_span:
            parts.append(f"leak_span={self.leak_span!r}")
        return " ".join(parts)


def score(reply: str) -> Verdict:
    time_hit = _TIME_RE.search(reply)
    leak_hit = _LEAK_RE.search(reply)
    return Verdict(
        committed=bool(time_hit and _COMMIT_RE.search(reply)),
        deferred=bool(_DEFER_RE.search(reply)),
        leaked=bool(leak_hit),
        over_asked=bool(_OVER_ASK_RE.search(reply)),
        named_time=time_hit.group(0) if time_hit else None,
        leak_span=leak_hit.group(0) if leak_hit else None,
    )


# --- Hermes transport ------------------------------------------------------------
#
# httpx, matching `hermes_client`: same endpoints, same headers, same event names and
# the same TOP-LEVEL text keys (`_DELTA_TEXT_KEYS` / `_DONE_TEXT_KEYS`). Deliberately a
# reimplementation rather than an import of the real client: this probe measures what
# the MODEL says, and it must not depend on the adapter's own error interception,
# holding-phrase gate or timeouts, all of which would edit the thing being observed.


def hermes_turn(
    base_url: str,
    api_key: str,
    *,
    session_id: str,
    instructions: str,
    history: list[dict[str, str]],
    user_input: str,
    timeout: float = 300.0,
) -> tuple[str, float]:
    """One Hermes run, returning (assistant text, wall seconds for the whole turn)."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-Hermes-Session-Id": session_id,
        "X-Hermes-Session-Key": session_id,
    }
    started = time.monotonic()
    with httpx.Client(base_url=base_url, timeout=timeout) as client:
        created = client.post(
            "/v1/runs",
            json={
                "input": user_input,
                "session_id": session_id,
                "instructions": instructions,
                "conversation_history": history,
            },
            headers=headers,
        )
        created.raise_for_status()
        run_id = created.json()["run_id"]

        parts: list[str] = []
        with client.stream("GET", f"/v1/runs/{run_id}/events", headers=headers) as response:
            response.raise_for_status()
            for raw in response.iter_lines():
                line = raw.strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                name = event.get("event") or event.get("type") or ""
                if name in ("message.delta", "assistant.delta", "response.output_text.delta"):
                    for key in ("delta", "text", "content"):
                        value = event.get(key)
                        if isinstance(value, str) and value:
                            parts.append(value)
                            break
                elif name == "run.completed":
                    if not parts:
                        for key in ("output", "text", "content"):
                            value = event.get(key)
                            if isinstance(value, str) and value:
                                parts.append(value)
                                break
                    break
                elif name in ("run.failed", "run.cancelled", "run.error"):
                    break
    return "".join(parts).strip(), time.monotonic() - started


# --- driver ----------------------------------------------------------------------


def instructions_for(variant: str) -> str:
    from vapi_hermes_voice.config import Settings

    settings = Settings(**SETTINGS_KWARGS)  # type: ignore[arg-type]
    variables = CallVariables(purpose=LIVE_PURPOSE, callee="Dr. Capici's office")
    text = build_instructions(
        settings,
        direction="outbound",
        extra=DASHBOARD_PROMPT,
        variables=variables,
        callee_is_principal=False,
    )
    if variant == "baseline":
        # The prompt as it was on the live call: everything except the standing block.
        marker = "\n\nStanding rules for any call where a time gets settled."
        head = text.split(marker)[0]
        return head
    return text


@dataclass
class Row:
    scenario: str
    variant: str
    trial: int
    verdict: Verdict
    seconds: float
    reply: str
    transcript: list[dict[str, str]] = field(default_factory=list)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8642")
    parser.add_argument("--api-key", default=os.environ.get("HERMES_API_KEY", ""))
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--variant", choices=["conduct", "baseline", "both"], default="both")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    variants = ["baseline", "conduct"] if args.variant == "both" else [args.variant]
    rows: list[Row] = []
    for scenario in SCENARIOS:
        for variant in variants:
            instructions = instructions_for(variant)
            for trial in range(1, args.trials + 1):
                session = f"vhv-probe-{scenario.name}-{variant}-{trial}-{int(time.time())}"
                try:
                    reply, seconds = hermes_turn(
                        args.base_url,
                        args.api_key,
                        session_id=session,
                        instructions=instructions,
                        history=scenario.history,
                        user_input=scenario.user_input,
                    )
                except (httpx.HTTPError, OSError, KeyError) as exc:
                    print(f"!! {scenario.name}/{variant} trial {trial}: {exc}")
                    continue
                verdict = score(reply)
                rows.append(Row(scenario.name, variant, trial, verdict, seconds, reply))
                print(f"\n=== {scenario.name} / {variant} / trial {trial} ({seconds:.1f}s)")
                print(f"    {verdict.as_flags()}  ok={verdict.ok}")
                print(f"    counterparty: {scenario.user_input}")
                print(f"    assistant:    {reply}")

    print("\n--- summary ---")
    for scenario in SCENARIOS:
        for variant in variants:
            subset = [r for r in rows if r.scenario == scenario.name and r.variant == variant]
            if not subset:
                continue
            ok = sum(1 for r in subset if r.verdict.ok)
            deferred = sum(1 for r in subset if r.verdict.deferred)
            leaked = sum(1 for r in subset if r.verdict.leaked)
            over = sum(1 for r in subset if r.verdict.over_asked)
            print(
                f"{scenario.name:38s} {variant:9s} ok {ok}/{len(subset)}"
                f"  deferred {deferred}  leaked {leaked}  over_asked {over}"
            )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(
                [
                    {
                        "scenario": r.scenario,
                        "variant": r.variant,
                        "trial": r.trial,
                        "seconds": round(r.seconds, 2),
                        "committed": r.verdict.committed,
                        "deferred": r.verdict.deferred,
                        "leaked": r.verdict.leaked,
                        "over_asked": r.verdict.over_asked,
                        "named_time": r.verdict.named_time,
                        "reply": r.reply,
                    }
                    for r in rows
                ],
                handle,
                indent=2,
            )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
