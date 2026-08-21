# Live voice-deadline harness

Measures the two voice requirements that keep regressing, **from the callee's seat**, on
a real Vapi call with real audio through the real transcriber — with no human on the line
and no phone number anywhere in the request.

```
R1  Within 2 s of the callee finishing their first utterance ("Hello?"), Emma says why
    she is calling.
R2  Within 2 s of the callee stopping on any turn, Emma says a brief acknowledgement —
    and then no other for at least 10 s, counted across the whole call, not per turn.
```

**This is not part of CI.** It places billable calls, so `pytest` never collects it
(`addopts = "-q --ignore=tests/e2e"`). Run it by hand or on a nightly schedule.

## Why this exists in addition to the unit tests

The unit tests assert *logic* — a cooldown field exists, a gate is consulted — and they
stop at the adapter's HTTP boundary. Both reported failures had causes on the far side of
that boundary:

- The biggest single contributor to "10 seconds of silence" was **Deepgram Flux holding
  the turn open**. On live call `01a02524`, `eotTimeoutMs` was 5000 and every repeated
  "Hello?" re-armed it while `eotThreshold` 0.8 was never reached: 96 ms of speech ending
  at 3.33 s produced no final transcript until 14.29 s, and Emma spoke at 16.92 s. That
  happens entirely upstream of the adapter. No adapter-level test can see it.
- The assistant's static `firstMessage` answers the callee's first utterance **without
  calling the adapter at all** (see [Trap 2](#trap-2-the-static-firstmessage)). An R1
  timing check alone therefore passes with the adapter switched off.

## Mechanism

`POST /call` with `transport.provider = "vapi.websocket"`. Vapi returns a
`websocketCallUrl`; the harness streams 16 kHz mono `pcm_s16le` up it in real time and
receives the assistant's audio back down it.

No `customer`, no `customer.number`, no `phoneNumberId` is ever sent, so **no PSTN leg can
exist and nothing rings**.

Holding both ends of the audio is the point. The harness knows to the 20 ms frame when it
stopped emitting audible samples, so the zero point of every deadline is exact rather than
inferred, and the silence gaps are exactly as scripted.

### Why not a browser `webCall`

Tried first; not usable with the credentials available:

| attempt | result |
| --- | --- |
| `POST /call` with `transport.provider="daily"` | `400 Couldn't Get Phone Number. Need Either `phoneNumberId` Or `phoneNumber`.` |
| `POST /call/web` with `VAPI_API_KEY` | `401 Invalid Key. Hot tip, you may be using the private key instead of the public key, or vice versa.` |

`POST /call/web` is the endpoint the browser SDK uses and it needs the org **public**
key — a different credential from `VAPI_API_KEY`, and not present on this workstation.
This is *not* the payment-method restriction that blocked `/chat`: the account's
subscription is active (`POST /call` returns
`subscriptionLimits: {concurrencyLimit: 10, ...}`).

If the public key is ever added, a browser runner driving Chromium with
`--use-fake-device-for-media-capture --use-file-for-fake-audio-capture` could reuse
`audio_script.py` unchanged. It would be strictly noisier than the websocket transport
(WebRTC jitter, no frame-accurate send clock), so there is no reason to prefer it.

## Running it

```bash
uv venv && uv pip install -e ".[dev,e2e]"
export VAPI_API_KEY=$(grep -i '^vapi_api_key=' ~/.env | cut -d= -f2-)

# Synthesise audio, print the script, run the preflight. Places no call, costs nothing.
uv run python -m tests.e2e.run_voice_deadlines --dry-run

# The real thing.
uv run python -m tests.e2e.run_voice_deadlines --scenario deadlines --json-out result.json

# Re-score a call that already happened — including a real production failure.
uv run python -m tests.e2e.run_voice_deadlines --from-call 01a02524-...
```

`VAPI_API_KEY` is read from the environment only. Nothing in the harness prints it, logs
a request header, or writes it to the JSON result.

### Cost

**~$0.05 per run** of the default `deadlines` scenario (~31 s of call). Measured on real
runs: $0.0191 and $0.0189 for ~20 s calls, $0.0371 for a 31 s call, almost all of it the
Vapi platform minute rate plus Deepgram STT.

### Exit codes

| code | meaning |
| --- | --- |
| 0 | every check passed |
| 1 | a deadline check failed — read the table |
| 2 | no local text-to-speech to build the callee's audio with |
| 3 | preflight refused to run, or the R1 reply could not be attributed |
| 4 | Vapi API error |
| 5 | Vapi's timeline was unreadable (see [Trap 1](#trap-1-duration-is-milliseconds)) |

### Scenarios

`--scenario deadlines` (default) — "Hello?", 12 s of silence, then "What medication did
the vet prescribe Marvin?", then 14 s of silence. The gaps *are* the instrument: 12 s so
that a 10 s stall is measured rather than clipped by the next utterance, and 14 s so the
10 s cooldown can actually be proven after an acknowledgement lands.

`--scenario flux_storm` — "Hello?" three times about a second apart, then silence. This
reproduces the re-arming that caused the live Flux stall. No lookup question, so no
acknowledgement is due and the R2 deadline is reported as `skip`.

## What it reports

Three clocks, printed together so a divergence between them is visible:

1. **Vapi's own timeline** (`messages[]` with `secondsFromStart` + `duration`) — the
   authoritative record of what was spoken and when. All pass/fail verdicts come from
   here, because it is what the callee heard.
2. **The callee's own microphone and speaker** — last audible frame sent to first audible
   frame received, on the harness's monotonic clock. Includes transcriber lag, LLM time,
   text-to-speech time and transport buffering.
3. **The transport event stream** (`model-output`, `voice-input`, `speech-update`,
   `hang`) — the adapter's text as Vapi received it. This is what makes attribution
   possible.

Checks, all with the measured value printed whether they pass or fail:

| id | what it means |
| --- | --- |
| `replied` | every callee turn got an answer at all |
| `r1_deadline` | callee's first utterance ends → assistant starts, ≤ 2 s |
| `r1_provenance` | that answer came from the adapter, not from `assistant.firstMessage` |
| `r1_transport_scope` | this transport's `call.type` can even exercise R1's own code path (live/scored calls only) |
| `r2_ack_deadline` | acknowledgement spoken within 2 s of the lookup question ending |
| `r2_ack_cooldown` | closest pair of acknowledgements in the call ≥ 10 s apart |
| `r2_ack_storm` | at most `floor(ack_storm_window_s / ack_cooldown_s) + 1` acknowledgements in any `ack_storm_window_s` window — **derived from the cooldown**, so it can never disagree with `r2_ack_cooldown` (see [the ack-storm arithmetic](#trap-4-the-ack-storm-threshold-must-agree-with-the-cooldown)) |
| `max_turn_gap` | no turn leaves the callee waiting past the ceiling (3 s) — the Flux guard |
| `r2_ack_emitted` | an acknowledgement reached the transport within 2 s (live only) |
| `acks_reached_the_callee` | every channel=stream acknowledgement was actually spoken (live only) |
| `script_coverage` | every scripted callee utterance became its own turn (live only) |

`r2_ack_emitted` and `acks_reached_the_callee` know about BOTH acknowledgement delivery
channels — see
[channel=control is invisible to model-output](#trap-5-channelcontrol-acknowledgements-are-invisible-to-model-output)
— but only ever *timestamp* channel=stream (the `model.url` SSE connection, visible as
`model-output` events). When Vapi spoke more acknowledgements than channel=stream
explains, the surplus is reported as attribution **UNKNOWN**, never guessed as
channel=control: several of the adapter's own filler phrases are ordinary enough that
the model streaming the same words itself, without the adapter's `<flush />` marker, is
indistinguishable from a genuine channel=control delivery on this transport alone —
exactly the model-freelancing-a-holding-phrase regression this harness exists to catch,
so guessing would mask it rather than report it. `r2_ack_deadline` above, on Vapi's own
clock, is still the deadline measurement for an unattributed acknowledgement; only the
*channel* is unknown, not whether or when it was spoken. Real channel attribution needs
the adapter's own record, not inference from the transport — see
[the debug-endpoint note](#trap-5-channelcontrol-acknowledgements-are-invisible-to-model-output).

`r1_transport_scope` exists because this harness never places a PSTN call (see
[Mechanism](#mechanism)), so `call.type` is never `outboundPhoneCall` and
`chat.direction` is always `"inbound"`. The adapter's reason-for-calling fast path (and
the outbound-only system-prompt framing behind it) is gated on `direction == "outbound"`
and so never runs on this transport, no matter how fast or slow the assistant answers.
`r1_deadline` above is still a real measurement of the callee's first turn, but reporting
it as R1's own PASS/FAIL would be misleading in both directions: `r1_transport_scope`
reports the gap explicitly, as `skip` with `UNVERIFIABLE-BY-THIS-TRANSPORT` in the
detail, the same disciplined way `r1_provenance` reports the `firstMessage` shortcut.
Only a real `outboundPhoneCall` — which this harness refuses to place — can verify R1.

The last several checks exist because of what live runs found. `script_coverage` catches
the opposite failure: a scenario that did not provoke what it was written to provoke.
On live `flux_storm` run `01a025f1`, greetings two and three landed while the assistant
was still reading its `firstMessage`, were absorbed as barge-in, and produced no separate
callee turns — so no end-of-turn timer was ever re-armed and the run tested nothing. It
reports FAIL, because passing would be false assurance about the exact bug the scenario
exists for.

Acknowledgements are matched against the **configured** phrase pool, never guessed. The
pool is resolved from `vapi_hermes_voice.config.Settings` (so `VHV_FILLER_PHRASES` in the
environment or `.env` applies) or from `--ack-phrases-file`, and the pool actually used is
printed and recorded in the JSON. Separately, any `model-output` carrying the `<flush />`
token is a holding line *structurally* — `turns.build_filler` is the only thing that
appends it — so the harness can detect that the deployed pool has drifted from the pool it
was told about, instead of silently reporting "no acknowledgement".

## Preflight

Refuses to measure a system that is not up, because with the adapter down the R1 deadline
still passes and every number is a lie.

- Derives the origin of `assistant.model.url` (origin only: `model.url` may carry a route
  secret path segment, which is never probed and never printed) and requires
  `GET /healthz` → 200.
- Requires `GET /readyz` → 200. That endpoint is 503 when Hermes is unreachable, and R2
  needs a real tool round trip to provoke an acknowledgement, so a degraded adapter
  cannot produce a meaningful R2 measurement. `--allow-degraded` downgrades this to a
  warning.
- Warns when `firstMessage` is set with `firstMessageMode="assistant-waits-for-user"`,
  and when `transcriber.eotTimeoutMs >= 3000`.

A failed preflight exits 3 and says, in words, that the run is an **outage and not a
latency regression**. `--skip-preflight` exists and prints a warning saying exactly what
you are giving up.

## Traps this harness had to be built around

### Trap 1: `duration` is milliseconds

Vapi's OpenAPI says "The duration of the message in seconds". It is milliseconds.
Measured on live call `01a025d2`: a user message with `time=1787340932548`,
`endTime=1787340932622` — 74 ms of wall clock — carries `duration: 74`.

Read as seconds, the end of a 1.286 s utterance lands at 75.3 s, every subsequent gap
goes negative, and **every deadline check passes vacuously**. So `deadlines.py` derives
the unit from `endTime - time` and refuses to score a timeline where nothing decides it.

The tolerance has to be loose, though. `duration` and `endTime - time` genuinely disagree
on assistant messages — one is the length of the synthesised audio, the other a pair of
pipeline timestamps. Live call `01a025ea` had `duration=2955` over a 2228 ms span, 33 %
apart. An earlier, tighter check called that a changed timeline format and threw away a
perfectly good billable call. Divergence over 25 % is now reported as a note; only a
thousandfold error is an error.

### Trap 2: the static `firstMessage`

The assistant has `firstMessageMode = "assistant-waits-for-user"` and a configured
`firstMessage`. That combination means Vapi speaks a fixed string on the callee's first
turn and **never calls `model.url`**.

Proven, not assumed: a control call with `model.url` overridden to a guaranteed-404
endpoint *and* the system prompt replaced still produced the identical utterance, 0.556 s
after the callee's "Hello." (fixture `call_dead_adapter_control.json`). The adapter's
reason-for-calling fast path was not involved.

So `r1_provenance` fails by default whenever the R1 reply matches `firstMessage`. It is
not a bug in the harness; it is the honest verdict: the deadline is met, the adapter is
untested, and R1 will regress silently the day `firstMessage` is edited. To measure the
adapter, clear `assistant.firstMessage`.

The match is a similarity ratio, not string equality, because `messages[]` records the
assistant's speech *as recognised*. On live call `01a025ee` Vapi spoke the static
`firstMessage` and the transcript came back with "Mike Averdo" for "Mike Averto" — one
letter, and exact matching flipped the most important check in this harness from FAIL to
a false PASS. The threshold is 0.85 against an observed noise floor of 0.994.

### Trap 3: `api.vapi.ai` blocks requests with no `User-Agent`

Cloudflare answers `403` with the body `error code: 1010`, which reads exactly like an
auth failure and is not one. Every request the harness makes sets a `User-Agent`.

### Trap 4: the ack-storm threshold must agree with the cooldown

`r2_ack_storm` counts acknowledgements in a rolling window; `r2_ack_cooldown` requires
the closest pair to be at least `ack_cooldown_s` apart. With the defaults (a 16 s window,
a 10 s cooldown) a hardcoded storm threshold of 1 flags **two acknowledgements 11.6 s
apart** — which pass the cooldown check correctly — as a storm. The two checks
disagreed with each other on the same call.

`Budgets.ack_storm_max` is now a property, derived from `ack_cooldown_s` and
`ack_storm_window_s`: with acknowledgements spaced at least `ack_cooldown_s` apart, the
span from the first to the last of `k` of them is at least `(k - 1) * ack_cooldown_s`,
which bounds how many can ever fit inside a window of length `ack_storm_window_s`. A
call that respects the cooldown now always passes the storm check too — the two checks
can never disagree — and the only way to fail the storm check is to genuinely pack more
acknowledgements into the window than the cooldown allows: the reported failure this
check exists for was **six acknowledgements inside 16 s, at offsets
0.05/0.22/0.40/0.57/0.74/0.91 s**, which still fails under the derived threshold exactly
as it should (see `tests/e2e/test_scoring_fixes.py`).

### Trap 5: channel=control acknowledgements are invisible to `model-output`

An acknowledgement can reach the callee by two different paths, both logged by the
adapter as `turn filler ... channel=stream|control`:

- **channel=stream** — the text rides the `model.url` SSE connection like any other
  reply. This is what `model-output` events on the websocket transport carry, and the
  only channel `_emitted_holding_lines` can see.
- **channel=control** — delivered directly via Vapi Live Call Control
  (`POST call.monitor.controlUrl {"type": "say", ...}`), out-of-band from `model.url`
  entirely. Nothing on this transport timestamps it before it becomes audio.

Recorded live on call `01a0262b-a1ce-733b-aad5-a93df060162e`: one channel=stream
acknowledgement and one further acknowledgement, both genuinely spoken. Because
`_emitted_holding_lines` only ever sees channel=stream, the harness reported "the
adapter emitted 1 holding line(s)" against "1 emitted, 2 spoken" — more spoken than
emitted, which is not possible unless a channel went uncounted.

The first fix for this **inferred** the surplus as channel=control, on the reasoning
that there is no third path a spoken-but-unexplained acknowledgement could have come
from. That reasoning is wrong, and was rejected on review: several of the adapter's own
`_DEFAULT_FILLER_PHRASES` ("Sure, give me a second.", "Okay, bear with me a moment.")
are ordinary enough that the model streaming the identical words itself over
`model.url` — without the adapter's structural `<flush />` marker — is
**indistinguishable, from this transport alone**, from a genuine channel=control
delivery. Confidently crediting the surplus to channel=control would silently mask
exactly the regression PR #10's prohibition exists to catch: the model re-acquiring its
own holding-phrase habit. A harness that cannot tell "the adapter used channel=control"
apart from "the model is freelancing filler again" must not pick one and call it a
PASS.

`evaluate_transport` now reports that surplus as attribution **UNKNOWN**, the same
disciplined way `r1_transport_scope` reports R1: `r2_ack_emitted` returns `skip` with
`UNKNOWN` in the detail rather than a false "the adapter never produced an
acknowledgement at all" or a fabricated latency, and `acks_reached_the_callee` names
the unattributed count without folding it into a pass/fail. `r2_ack_deadline` (Vapi's
own clock) remains the real deadline measurement regardless of channel — only the
*channel* is unknown, not whether or when the callee heard it.

**Real attribution needs evidence, not inference.** A spec exists for a small
read-only adapter endpoint that would provide it, `GET /debug/acks/{call_ref}`
(`call_ref = sha256(call_id)[:12]`, authenticated with the deployed
`VHV_ADAPTER_API_KEY` bearer, gated behind `VHV_DEBUG_ACK_LOG`, a bounded per-call
ring so it cannot grow without limit), returning the adapter's own record of every
acknowledgement it emitted: text, channel, and both wall-clock and monotonic emission
time. It is **not implemented and not currently claimed by anyone** -- proposed during
review, scoped, and then dropped for lack of run budget before any code landed. If it
is ever built, the harness should read that instead of guessing: a spoken line that is
NOT in the adapter's own record is by definition not ours, which is the discrimination
this transport cannot make on its own. Until and unless that happens, UNKNOWN is the
correct and **permanent** default -- not a placeholder for the inference this section
replaced, and this harness has no code path that depends on the endpoint existing.

## What a live run actually found

Run of `--scenario deadlines` against a healthy adapter (`/healthz` 200, `/readyz` 200),
call `01a025e5-c138-7449-a288-236c67a9341e`:

- **R1: 0.434 s** on Vapi's timeline, 0.693 s on the callee's own clock. Well inside
  budget — and `r1_provenance` failed, because it was the static `firstMessage`.
- **R2: the callee heard nothing at all.** They asked "What medication did the vet
  prescribe Marvin?" ending at 16.123 s and no assistant audio followed for the remaining
  14 s of the call.

The transport clock explains it, and this is precisely what the API timeline alone cannot
do. The adapter *did* respond: `model-output: "Okay, bear with me a moment. <flush />"` at
+2.05 s, accepted by Vapi as `voice-input` — then never spoken. Vapi emitted `hang` at
+21.2 s, and a second holding line at +29.7 s, also never spoken.

Scoring only what was spoken says "the adapter produced nothing", which points the next
person at the wrong component. Hence `r2_ack_emitted` and `acks_reached_the_callee`.

Two further findings from the same run, both recorded as fixtures:

- Neither emitted holding line is in `config._DEFAULT_FILLER_PHRASES` ("Bear with me one
  second." and "Just a second, looking now." are the near misses), so the deployed
  `VHV_FILLER_PHRASES` has drifted from the repo default — or those lines are
  LLM-generated. Either way the harness reports it rather than mis-scoring.
- The emitted acknowledgement was 2.049 s after the callee stopped talking, marginally
  over the 2 s budget, before it was dropped.

### A second run: call 01a0262b, the harness's own scoring bugs

A run against deployed `main` (87a2102), assistant `b39379dc-...279b4f` with
`firstMessage` cleared, produced `RESULT: FAIL (r1_deadline, r2_ack_deadline,
r2_ack_storm, max_turn_gap, r2_ack_emitted)`. Three of those five were the harness
mis-scoring, not the adapter:

- `r2_ack_emitted` read "1 emitted" against "1 emitted, 2 spoken" (Trap 5): the adapter
  journal showed one channel=stream and one further acknowledgement, and only the
  stream one is visible to `model-output`.
- `r2_ack_storm` failed at 11.604 s apart while `r2_ack_cooldown` **passed** at the same
  11.604 s (Trap 4): the hardcoded storm threshold disagreed with the cooldown it is
  supposed to defend.
- `r1_deadline` reported FAIL at 5.107 s on a `vapi.websocketCall`, where the
  reason-for-calling fast path this harness exists to verify can never fire (this
  section's `r1_transport_scope`). The number is real; it measured an ordinary inbound
  Hermes turn, not R1.

Re-scoring this exact call with the phrase pool actually deployed at the time
reproduces the original numbers precisely, and shows both harness fixes changing the
verdict table:

```
                                        before (buggy harness)   after (this fix)
r1_deadline           5.107s               FAIL                     FAIL
r2_ack_deadline        2.071s               FAIL                     FAIL
r2_ack_cooldown       11.604s               PASS (budget 10)         PASS (budget 10)
r2_ack_storm          2 acks                FAIL (budget 1)          PASS (budget 2)
max_turn_gap           5.107s               FAIL                     FAIL
r1_transport_scope    n/a                   (did not exist)          SKIP, UNVERIFIABLE-BY-THIS-TRANSPORT
```

`ack_control_timeout_seconds` defaulting to 3.0 s — longer than the entire 2 s R2
budget, so a single slow control POST blows the deadline before the stream fallback even
starts — is the one real defect this run found; it is an adapter fix
(`src/vapi_hermes_voice/config.py`), not a harness one. `r1_deadline`,
`r2_ack_deadline`, and `max_turn_gap` remain FAIL after this fix and are that adapter
defect, not a harness bug: this fix corrects *attribution*, not the underlying latency.

## Limits — what this cannot verify

- **It is not a phone line.** The transport is clean 16 kHz PCM; PSTN is 8 kHz µ-law with
  jitter and noise. Deepgram Flux's `eotThreshold` is reached *more easily* on clean
  audio, so absolute gap numbers here are a **floor**: a phone call can be worse than a
  passing run here, never better. What this does catch reliably is the *configuration*
  regression — an `eotTimeoutMs`/`eotThreshold` change that re-introduces the hold-open —
  because the scripted re-arming in `flux_storm` is transport-independent.
- **It cannot verify R1 at all, on any call this harness places.** It never places a
  PSTN call (see [Mechanism](#mechanism)), so `call.type` is never `outboundPhoneCall`
  and the adapter's reason-for-calling fast path (gated on that type) never runs.
  `r1_transport_scope` reports this explicitly on every run. This is true even with
  `assistant.firstMessage` cleared — a strictly stronger limit than the `firstMessage`
  shortcut below, which only applies while that field is set. This is the single most
  important thing in this document: only a real `outboundPhoneCall` can verify R1, and
  this harness will never place one.
- **It cannot verify R1 against the adapter while `assistant.firstMessage` is set,
  separately from the limit above.** It reports that it cannot, rather than passing.
- **It cannot verify R2's cooldown across calls**, only within one call. The cooldown is
  documented as call-global, and `CallState` is per-call, so that is the correct scope.
- **It does not verify barge-in behaviour.** The script never interrupts the assistant.
  The reported ack storm arose during a barge-in retry storm; `r2_ack_storm` would catch
  the symptom if a scenario provoked it, but no scenario here does.
- **A single run is one sample.** These are wall-clock measurements over a live network
  and a live LLM. Treat one failing run as a signal to re-run, and a repeatable failure as
  a regression. The default budgets have real headroom over the healthy measurements
  (0.43 s against a 2 s budget) precisely so that ordinary variance does not flap.

## Layout

| file | role |
| --- | --- |
| `run_voice_deadlines.py` | CLI: preflight, place call, drive it, score it, print, exit code |
| `vapi_live.py` | Vapi control plane + the preflight |
| `audio_script.py` | the callee's scripted utterances and silence gaps; local synthesis |
| `ws_call.py` | streams the audio, records both directions on a monotonic clock |
| `deadlines.py` | **pure** scoring: no I/O, no clock, no network |
| `fixtures/` | recorded live calls and transcribed reported failures |

`deadlines.py` is pure so that it can be tested deterministically. It is, by
`tests/test_e2e_deadlines.py`, which **does** run in CI: a timing harness whose own
arithmetic is untested manufactures confidence rather than providing it. Those tests fail
if the analysis stops catching the millisecond trap, the `firstMessage` trap, the Flux
stall, the ack storm, or the dropped acknowledgement.

`tests/e2e/test_scoring_fixes.py` covers the harness's own scoring fixes above (the
ack-storm arithmetic, the UNKNOWN channel attribution, `r1_transport_scope`) the same way
`tests/test_e2e_deadlines.py` covers everything else: deterministic, no network. It is
**not** collected by a bare `pytest` either — same reasoning as the rest of this
package — run it directly with `uv run pytest tests/e2e/test_scoring_fixes.py`.
