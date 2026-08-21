# vapi-hermes-voice

A Python 3.11 FastAPI adapter that bridges the [Vapi](https://vapi.ai) Custom LLM
HTTP/SSE protocol to a Hermes Agent API server, so a phone caller can talk to your
Hermes agent.

Successor to [retell-hermes-voice](https://github.com/mjaverto/retell-hermes-voice)
(now archived): same Hermes-side engineering, a different voice platform — and a real
improvement, because Vapi authenticates requests to the adapter (Retell's Custom LLM
WebSocket sent no auth at all).

```
                 PSTN / SIP               HTTP POST + SSE (Custom LLM)      HTTP + SSE
 +--------+    +-----------------+       +----------------------+       +---------------+
 | Caller | -> | Vapi cloud      | ----> | vapi-hermes-voice    | <---> | Hermes Agent  |
 +--------+    |  telephony      | HTTPS |  (this adapter)      |       |  API server   |
               |  STT / TTS      | <---- |  POST /chat/         |       |  /v1/runs     |
               |  turn-taking    |  SSE  |    completions       |       |  SSE events   |
               |  barge-in       |       |  Bearer auth         |       |               |
               +-----------------+       +----------------------+       +---------------+
```

## Division of labor

| Layer | Responsibility |
|---|---|
| **Vapi** | Telephony (numbers, SIP), speech-to-text, text-to-speech, turn-taking, barge-in detection, greeting (`firstMessage`), call recording |
| **This adapter** | Protocol translation, bearer-token auth (+ optional route secret), caller allowlist, latency management (fillers with `<flush />`, model routing, warmup), speech sanitization, run lifecycle (no orphaned Hermes runs), redacted logging |
| **Hermes Agent** | Reasoning, tools, memory, sessions — the actual "brain" |

### Non-goals

- Not a SIP stack or telephony system — Vapi owns the phone leg.
- Not a Hermes fork or plugin — it is a pure API client of the Hermes API server.
- No Vapi-side tools: do not attach Vapi tools to the assistant; the adapter never
  emits `tool_calls` and does not serve `/chat/completions/custom-tool`. Hermes owns
  all tools, and tool restriction is enforced on the Hermes side (see
  [Security](#security)); the adapter's tool policy is advisory.

## Supported versions

- Python **>= 3.11**
- Hermes Agent **0.20.4** (verified live)
- Vapi Custom LLM protocol as documented at docs.vapi.ai, fetched **2026-08-21**

The full verified wire contract — every payload shape, latency number, and failure
mode this adapter is built against, with a verification class per claim — lives in
[`docs/integration-contracts.md`](docs/integration-contracts.md). No live Vapi call
has been placed yet; Vapi-side claims are DOCS-class and the open items are listed in
its §4.

## Quick start

```sh
# 1. Install
uv venv
uv pip install -e '.[dev]'

# 2. Configure
cp .env.example .env
# edit .env: set VHV_HERMES_BASE_URL, VHV_HERMES_API_KEY, VHV_ADAPTER_API_KEY

# 3. Enable the Hermes API server (on the Hermes host)
#    In ~/.hermes/.env:
#      API_SERVER_ENABLED=true
#      API_SERVER_KEY=<a strong secret>       # this becomes VHV_HERMES_API_KEY
#    then start/restart the gateway:
hermes gateway

# 4. Run the adapter
python -m vapi_hermes_voice

# 5. Verify
curl http://127.0.0.1:8766/healthz   # {"status": "ok"}
curl http://127.0.0.1:8766/readyz    # 200 when Hermes is reachable, 503 otherwise

# 6. Exercise the endpoint like Vapi would
curl -N http://127.0.0.1:8766/chat/completions \
  -H "Authorization: Bearer $VHV_ADAPTER_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"hermes","stream":true,"messages":[{"role":"user","content":"Say hi."}],
       "call":{"id":"curl-test","type":"inboundPhoneCall"},
       "customer":{"number":"+15551234567"}}'
```

Generate the adapter API key (minimum 16 characters enforced; use 32+):

```sh
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## Exposing the endpoint

Vapi must reach the adapter over HTTPS. For development, tunnel the local port:

```sh
# cloudflared (also a legitimate production option — see docs/deployment.md)
cloudflared tunnel --url http://127.0.0.1:8766

# ngrok (development only)
ngrok http 8766
```

The URL you give Vapi is the **base** URL — Vapi's OpenAI client appends
`/chat/completions` itself:

```
https://<your-host>                     # without a route secret
https://<your-host>/v/<route-secret>    # with VHV_ROUTE_SECRET set
```

(If you paste a URL that already ends in `/chat/completions`, the adapter tolerates
the doubled path — see `docs/integration-contracts.md` §1.1.)

## Configuring the Vapi assistant

Via the dashboard: create an assistant, choose **Custom LLM** as the model provider,
paste the base URL, set the **API key** authentication to your `VHV_ADAPTER_API_KEY`
(Model section — this is what makes the endpoint non-public), set a `firstMessage`
greeting, pick a voice and transcriber, then bind a phone number.

Via the management API (`https://api.vapi.ai`, bearer auth with your **private**
API key — see `docs/integration-contracts.md` §3):

```sh
# Create the assistant (transient custom-llm credential attached inline)
curl -s https://api.vapi.ai/assistant \
  -H "Authorization: Bearer $VAPI_API_KEY" -H "Content-Type: application/json" \
  -d '{
    "name": "hermes-voice",
    "firstMessage": "Hi, this is the assistant. How can I help?",
    "model": {
      "provider": "custom-llm",
      "url": "https://your-host.example.com/v/<route-secret>",
      "model": "hermes",
      "metadataSendMode": "variable",
      "timeoutSeconds": 20
    },
    "credentials": [
      {"provider": "custom-llm", "apiKey": "<VHV_ADAPTER_API_KEY>"}
    ]
  }'

# Get a free Vapi number bound to the assistant
curl -s https://api.vapi.ai/phone-number \
  -H "Authorization: Bearer $VAPI_API_KEY" -H "Content-Type: application/json" \
  -d '{"provider": "vapi", "numberDesiredAreaCode": "212", "assistantId": "<assistant_id>"}'
```

Twilio/Telnyx/BYO-SIP numbers work too (`provider: "twilio"` etc. — see
https://docs.vapi.ai/phone-calling). Keep `metadataSendMode` at its default
`"variable"`: that is what delivers `call`/`customer` metadata, and the caller
allowlist **fails closed** without it.

There is also a hosted Vapi MCP server (`https://mcp.vapi.ai/mcp`, bearer
`VAPI_API_KEY`) exposing `create_assistant` / `create_call` / `list_phone_numbers`
tools — handy for managing the assistant from an agent, not needed by this adapter.

## Configuring Hermes

Enable the API server as in Quick start. Then, **strongly recommended**: run a
dedicated, least-privilege Hermes profile for voice, and disable dangerous toolsets
on the API-server platform:

```sh
hermes tools disable --platform api_server terminal process write_file patch \
  execute_code browser_exec cronjob delegate_task memory session_search
```

This is not optional hardening theater. It was verified live
(`docs/integration-contracts.md` §2) that **API clients cannot restrict Hermes
tools** — `"tools": []`, `"tool_choice": "none"`, toolset filters, and restriction
headers are all silently ignored. The adapter's `VHV_TOOL_POLICY__*` settings are
advisory prompt-shaping only. The Hermes profile config is the only real enforcement
point. Disabling `memory`/`session_search` also closes a verified cross-session
memory leak (see [docs/security.md](docs/security.md)).

## Voice behavior

- **Greeting** — Vapi speaks it: set `assistant.firstMessage` (or
  `firstMessageMode: "assistant-speaks-first-with-model-generated-message"` to have
  Hermes improvise one; the adapter sends an opening nudge when the request carries
  no user utterance yet). A static `firstMessage` never reaches the adapter, so it
  always wins over the adapter's own opening.
- **Outbound task calls** — an outbound call can be given a job to do. See
  [Outbound task calls](#outbound-task-calls) below.
- **Acknowledgements** — if no speakable text has arrived within
  `VHV_FILLER_AFTER_SECONDS` (default 0.3 s), the adapter speaks a brief
  acknowledgement from `VHV_FILLER_PHRASES` ("Okay, let me check."). It goes out over
  Vapi's Live Call Control endpoint, falling back to an SSE-embedded chunk suffixed
  with Vapi's `<flush />` token. The whole path is budgeted against the 2 s the callee
  actually experiences: 1.25 s of that is Vapi's own endpointing and TTS, leaving
  0.75 s here, which is why the dead-air wait is 0.3 s and the control POST's ceiling
  is *derived* rather than chosen (see `VHV_ACK_CONTROL_TIMEOUT_SECONDS`).
  This is an immediate answer to having been spoken to, so it fires on
  **every** turn including the first one, and it is deliberately **not** conditional
  on a tool running. Two gates gate it. It can never be spoken once the answer has
  started streaming — a turn Hermes answers in 300 ms gets none, because there was
  no dead air to cover. And `VHV_FILLER_MIN_GAP_SECONDS` (default 10 s) is a
  cooldown **global to the call**, not to the turn: once one is spoken, nothing else
  on that call speaks another until the gap has passed, whatever happens in between
  — new turns, tool-start re-arms, or a Vapi barge-in storm re-POSTing one turn six
  times. The cooldown anchor lives on the per-call state, so it survives turn
  boundaries and streams Vapi tears down mid-flight (a torn-down stream counts as
  spoken: the failure mode is silence, never repetition). The timer re-arms on
  tool-start events (each Hermes tool round trip adds roughly 2.9 s of dead air), so
  a turn long enough to outlive the cooldown does get a second line — the limit is a
  duration, not a count. Each line — phrase plus its `<flush />` token, when enabled
  — is always written as one atomic SSE chunk (never split across deltas), and each
  firing logs `turn filler call=<ref> elapsed_ms=<n>` for post-call diagnosis.

  The 0.9 s default is arithmetic, not taste: the requirement is that the callee
  hears something within 2 s of finishing their sentence, and Vapi's transcriber
  endpointing plus `startSpeakingPlan.waitSeconds` spend ~0.4-1.6 s of that budget
  before the adapter is invoked at all. Raising it above ~1 s breaks the ceiling.

  Because the adapter owns acknowledgements, the model is forbidden from speaking
  its own: the voice system prompt tells it never to open a reply with a holding or
  stalling phrase ("one moment", "bear with me", "let me check that first") and to
  begin with the substance. A model-authored holding phrase is indistinguishable
  from an adapter one to the person on the line, so it defeats the call-global
  cooldown from the only viewpoint that matters, and it spends the first tokens of
  the 2 s budget on filler instead of the answer. If you hear one anyway, check your
  Vapi dashboard system prompt (and the Hermes profile's own prompt) for standing
  guidance that asks for a filler while a tool runs, and delete it — the dashboard
  prompt layers on top of the adapter's framing.
- **Sanitization** — Hermes output is converted to speakable prose before it reaches
  TTS: markdown, code fences, tables, and emoji are stripped; URLs become
  "a link I can send you". Streaming-safe (constructs that span deltas are buffered).
- **Barge-in / hangup** — when Vapi abandons the request, the adapter stops the
  Hermes run (`POST /v1/runs/{id}/stop`, measured 0.255 s) — abandoned runs
  otherwise execute unboundedly.

### Outbound task calls

Give an outbound call an objective and the assistant opens by stating why it is
calling, instead of greeting whoever placed it. The objective travels as a Vapi
**dynamic variable**, set when the call is created:

```bash
curl -sS https://api.vapi.ai/call \
  -H "Authorization: Bearer $VAPI_PRIVATE_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
        "assistantId": "<your assistant id>",
        "phoneNumberId": "<your Vapi phone number id>",
        "customer": { "number": "+15551234567" },
        "assistantOverrides": {
          "variableValues": {
            "purpose": "reschedule Marvin'\''s cardiology recheck to next Tuesday afternoon",
            "spoken_reason": "about Marvin'\''s cardiology recheck",
            "callee": "Dr. Patel'\''s office"
          }
        }
      }'
```

- `purpose` (free text) is the objective, written **for the model**; it is injected
  into the Hermes instructions **after** the Vapi dashboard system prompt, so a
  generic dashboard persona cannot talk the model out of the job it was dialed to do.
  It is never spoken aloud.
- `spoken_reason` (optional but strongly recommended) is the reason for the call **as
  it should be said out loud**, one clause: `"about his left knee MRI results"`. The
  adapter speaks it within milliseconds of the callee's first word, with no model
  involved — see "Saying why you called" below. Writing it as a whole sentence
  instead (`"I am calling about his left knee MRI results"`) is also fine: the
  sentence template already supplies that lead-in, so a duplicate one is deleted
  rather than spoken twice. Aliases: `reason`, `reason_for_call`, `opening_line`,
  `spoken_purpose`.
- `callee` (optional, free text) describes who is being called.
- Omit `purpose` and `spoken_reason` both and the call behaves exactly as any other
  outbound call.

All values are treated as **untrusted**: control characters are stripped, whitespace
is collapsed so a value cannot forge prompt structure, lengths are capped (400 / 200 /
120 chars), and the text is never written to a log — only its length is. A key the
adapter does not recognize is warned about by **label only**, never silently dropped.

**Saying why you called.** On an outbound task call the callee's first utterance —
usually "Hello?" — is answered from adapter-local text, with no Hermes run and no
model: measured 2 ms in-process, against a requirement of one to two seconds. This
exists because nothing routed through Hermes can hold that deadline (1.6-2.2 s warm,
3.6-4.9 s cold, 14-17 s once a tool runs). What the callee hears is

> Hi, this is Emma, an AI assistant calling on behalf of Mike. I am calling about
> Marvin's cardiology recheck. Is this a good moment?

`purpose` is only a **trigger** for this line, never its content: it is instruction
prose that routinely carries section labels, the options being weighed, and the
principal's own constraints, so speaking it would read internal notes aloud to a
stranger. Only `spoken_reason` is spoken, and even it is stripped of labels, lists and
instruction phrasing first; if nothing safe survives, the line degrades to "Is this a
good moment to talk?" rather than guessing. `VHV_OUTBOUND_REASON_FAST_PATH=false`
restores the previous behaviour exactly.

The spoken clause keeps a length cap (24 words / 160 characters after the deletions
above), sized for a real reason as an operator writes it — a 12-word cap turned
"...results from August sixth" into "...results from August" on a live call, and an
ambiguous date on a clinical callback is worse than a slightly longer sentence.

**Third-party disclosure.** When the callee is not the principal, the opening states
that this is an AI assistant calling on `VHV_PRINCIPAL`'s behalf. On by default;
`VHV_OUTBOUND_DISCLOSE_AI=false` turns it off.

**Calling the principal.** Set `VHV_PRINCIPAL_NUMBER` to the principal's own E.164
number and the adapter recognizes calls to them: the assistant greets them directly by
name ("Hi Mike, Emma here.") instead of the third-person "this is Emma calling for
Mike", and skips the disclosure. Unset, every outbound call uses third-party framing
unless `callee` names the principal outright.

Full contract, precedence rules, and the request shapes the adapter parses:
`docs/integration-contracts.md` §1.10.

### Latency

Measured against Hermes 0.20.4 (`docs/integration-contracts.md` §2):

- Default Hermes route: **~3.0 s** median time-to-first-token. Routed
  `openrouter` / `google/gemini-3.7-flash` at low reasoning effort: **0.605 s**
  median — about 5× faster. Set the routing trio:

  ```sh
  VHV_VOICE_MODEL=google/gemini-3.7-flash
  VHV_VOICE_PROVIDER=openrouter
  VHV_VOICE_REASONING_EFFORT=low
  ```

  Model and provider must be set together — a bare model without a provider is
  silently ignored or mis-routed by Hermes — and the adapter now refuses to start
  with only one of them. `VHV_VOICE_REASONING_EFFORT` is unset by default: `low`
  buys the latency above but degrades multi-hop tool use, so it is opt-in.
- **Cold start**: the first call per provider was measured at up to **24 s**.
  `VHV_WARMUP_ON_START=true` (default) fires one tiny routed run at startup so the
  first real caller doesn't pay it. `/readyz` gates traffic on Hermes health.
- Vapi waits at most `model.timeoutSeconds` (default 20 s) for the first bytes; the
  adapter's fillers (from 1.5 s) and 15 s first-token timeout stay well inside it.

## Security

> **Authenticate the endpoint.** Unlike Retell's Custom LLM WebSocket, Vapi sends
> credentials: configure the custom-llm **API key** in Vapi and the same value as
> `VHV_ADAPTER_API_KEY`. The adapter rejects anything else with 401 (constant-time
> compare) before reading the body. You SHOULD combine:
>
> 1. **HTTPS** termination at a reverse proxy (the adapter binds loopback).
> 2. The **bearer API key** (required, ≥16 chars).
> 3. Optionally `VHV_ROUTE_SECRET` — an unguessable path prefix
>    (`/v/<secret>/chat/completions`) as defense-in-depth.
>
> Additionally: set `VHV_ALLOWED_CALLERS` to restrict who may reach the agent
> (empty list = allow all, with a warning; when set, calls without caller identity
> — web calls, `metadataSendMode: "off"` — are **denied**). It screens **inbound**
> calls only: outbound task calls are placed by the operator, and their
> `customer.number` is the callee. And run Hermes with
> **no tools enabled** for the API-server platform unless you have explicitly
> reviewed each one.

Full threat model: [docs/security.md](docs/security.md).
Reporting vulnerabilities: [SECURITY.md](SECURITY.md).

## Configuration reference

All settings load from `VHV_`-prefixed environment variables or a `.env` file
(case-insensitive). List fields accept comma-separated strings or JSON arrays.

| Env var | Default | Description |
|---|---|---|
| `VHV_HERMES_BASE_URL` | *(required)* | Hermes API server base URL, e.g. `http://127.0.0.1:8642` |
| `VHV_HERMES_API_KEY` | *(required)* | Hermes `API_SERVER_KEY` bearer token (secret) |
| `VHV_ADAPTER_API_KEY` | *(required)* | Static key Vapi presents in the Authorization header; minimum 16 characters enforced |
| `VHV_ROUTE_SECRET` | *(unset)* | Optional path secret; when set the endpoint moves to `/v/<secret>/chat/completions` and the bare path 404s |
| `VHV_LISTEN_HOST` | `127.0.0.1` | Bind address (keep loopback; terminate TLS at a proxy) |
| `VHV_LISTEN_PORT` | `8766` | Bind port |
| `VHV_ALLOWED_CALLERS` | `[]` | E.164 caller allowlist, applied to **inbound calls only** (on an outbound call `customer.number` is the callee, not a caller); empty = allowlist disabled (allow all, warn); non-empty fails closed on missing caller identity |
| `VHV_ASSISTANT_NAME` | `the assistant` | Name used in the voice instructions |
| `VHV_PRINCIPAL` | `the operator` | Whose assistant it says it is |
| `VHV_PRINCIPAL_NUMBER` | *(unset)* | Principal's own E.164 number; when set, a matching `customer.number` switches the opening to greeting the principal directly (no on-behalf-of framing, no AI disclosure) |
| `VHV_OUTBOUND_DISCLOSE_AI` | `true` | Disclose "I'm an AI assistant calling for &lt;principal&gt;" in the opening when the callee is not the principal |
| `VHV_OUTBOUND_OPENING` | *(built-in)* | Opening template for an outbound call with a `purpose` to a third party; must contain `{purpose}` |
| `VHV_OUTBOUND_OPENING_PRINCIPAL` | *(built-in)* | Opening template when the call reaches the principal themselves; must contain `{purpose}` |
| `VHV_OUTBOUND_REASON_FAST_PATH` | `true` | Answer the callee's first utterance on an outbound task call from adapter-local text, with no Hermes run. `false` restores the previous behaviour exactly |
| `VHV_OUTBOUND_REASON_SENTENCE` | `I am calling {reason}. Is this a good moment?` | The sentence that carries the reason; must contain `{reason}`. The greeting and the AI disclosure are built in code and are deliberately not templated |
| `VHV_OUTBOUND_REASON_SENTENCE_GENERIC` | `Is this a good moment to talk?` | Spoken instead when the call supplied no `spoken_reason` the adapter could safely use. Purpose-free by design |
| `VHV_FILLER_PHRASES` | 8 built-in phrases | Holding lines spoken during dead air; must be non-empty |
| `VHV_FILLER_AFTER_SECONDS` | `1.5` | Dead-air threshold before a filler opportunity (first one, and each tool-start re-arm) |
| `VHV_FILLER_MIN_GAP_SECONDS` | `8.0` | Structural floor between the end of one filler and the start of the next, checked when a filler would be spoken regardless of how it was re-armed |
| `VHV_FILLER_MAX_PER_TURN` | `1` | Hard cap on holding lines per turn (bounded to 1-3 regardless of configured value); once real content starts, no more are ever spoken |
| `VHV_FILLER_PHRASES` | 8 built-in phrases | Acknowledgement lines spoken during dead air; must be non-empty |
| `VHV_FILLER_AFTER_SECONDS` | `0.3` | Dead-air threshold before an acknowledgement opportunity (first one, and each tool-start re-arm). The adapter's own share of the 2 s budget is only `VHV_ACK_BUDGET_SECONDS - VHV_ACK_PLATFORM_OVERHEAD_SECONDS` = 0.75 s, and this plus the control POST has to fit inside it. Measured Hermes TTFB is 1.6-2.2 s warm, so lowering this suppresses no acknowledgement that would otherwise have been beaten by a real answer |
| `VHV_FILLER_MIN_GAP_SECONDS` | `10.0` | Cooldown between acknowledgements, **global to the call**: once one is spoken nothing on that call speaks another until it expires, across turns, re-arms and cancelled retries |
| `VHV_FILLER_USE_FLUSH` | `true` | Suffix fillers with `<flush />` for immediate TTS (requires Vapi's default `chunkPlan.enabled`) |
| `VHV_ACK_USE_CALL_CONTROL` | `true` | Speak acknowledgements via Vapi's Live Call Control endpoint (`call.monitor.controlUrl`) instead of the model.url SSE stream, which does not reliably render a flushed chunk left alone for more than a few seconds (docs/integration-contracts.md §1.6). Falls back to the SSE-embedded delivery when no control URL is present or the control request fails |
| `VHV_ACK_BUDGET_SECONDS` | `2.0` | The requirement itself: how long after the callee stops talking they may wait to hear something |
| `VHV_ACK_PLATFORM_OVERHEAD_SECONDS` | `1.25` | The part of that budget spent outside this process — Vapi transcriber endpointing and `startSpeakingPlan.waitSeconds` before the request arrives, TTS/transport after the ack is emitted. Measured 1.191 s live; budgeted up. Raise it for a slower region and the control timeout below shrinks to match |
| `VHV_ACK_CONTROL_TIMEOUT_SECONDS` | *(derived: `0.45`)* | Wall-clock ceiling on the acknowledgement control POST. Unset it is `ACK_BUDGET - ACK_PLATFORM_OVERHEAD - FILLER_AFTER` = 2.0 − 1.25 − 0.3 = 0.45 s, so it cannot drift out of agreement with the other three; set it to override deliberately. Enforced on the clock, not per network phase — a bare float is a *per-phase* value to httpx, so the old 3.0 s could be spent connecting and 3.0 s more reading |
| `VHV_CONTROL_ANSWER_TIMEOUT_SECONDS` | `3.0` | Ceiling on the control POST that speaks the **answer** after an acknowledgement went out the same way. Deliberately not the tight value above: it runs on a background task with Hermes already finished and no deadline on it |
| `VHV_VOICE_MODEL` | *(unset)* | Hermes model override for voice turns, e.g. `google/gemini-3.7-flash` |
| `VHV_VOICE_PROVIDER` | *(unset)* | Provider for `VHV_VOICE_MODEL`; always set together with it |
| `VHV_VOICE_REASONING_EFFORT` | *(unset)* | `model_options.reasoning_effort` sent to Hermes when set; `low` cuts first-token latency on the routed model but degrades multi-hop tool use, so there is no default |
| `VHV_WARMUP_ON_START` | `true` | Fire one tiny routed run at startup to absorb provider cold start |
| `VHV_HERMES_CONNECT_TIMEOUT` | `5.0` | Hermes HTTP connect timeout (s) |
| `VHV_HERMES_FIRST_TOKEN_TIMEOUT` | `15.0` | Max wait for the first Hermes token (s) |
| `VHV_HERMES_TURN_TIMEOUT` | `60.0` | Max wall time for one Hermes turn (s) |
| `VHV_HERMES_STOP_TIMEOUT` | `3.0` | Bound on the mandatory `POST /v1/runs/{id}/stop` (s) |
| `VHV_MAX_CONCURRENT_TURNS` | `5` | Adapter concurrent-turn cap; keep below Hermes `max_concurrent_runs` (default 10) |
| `VHV_MAX_HISTORY_MESSAGES` | `200` | Conversation-history truncation (keeps most recent) |
| `VHV_MAX_BODY_BYTES` | `1000000` | Inbound request body cap, enforced while reading |
| `VHV_SESSION_RETENTION` | `none` | `none` = per-call random session ids; `hermes` = let Hermes persist sessions |
| `VHV_CALL_STATE_TTL_SECONDS` | `14400` | Per-call state eviction TTL (session ids, acknowledgement picker, acknowledgement cooldown) |
| `VHV_MAX_TRACKED_CALLS` | `1024` | Per-call state LRU cap |
| `VHV_TOOL_POLICY__ENABLED_TOOLS` | `[]` | Advisory: tools voice turns may use (prompt-shaping only, not enforcement). Empty/unset = no client-side opinion (defers to the Hermes profile); set to `none` explicitly to tell the model to use no tools |
| `VHV_TOOL_POLICY__CONFIRM_TOOLS` | `[]` | Advisory: tools requiring spoken confirmation |
| `VHV_TOOL_POLICY__MAX_TOOL_CALLS_PER_TURN` | `3` | Advisory per-turn tool budget |
| `VHV_TOOL_POLICY__MAX_TOOL_SECONDS_PER_CALL` | `60.0` | Advisory per-call tool time budget |

The control connection is kept warm so the handshake never lands on that 0.45 s
ceiling: the shared client pools idle connections for 60 s (httpx's default is five
seconds, shorter than the gap between two acknowledgements, so the pool was doing
nothing), and every turn fire-and-forgets a `GET` of the control **origin** — httpx
pools by origin, not path, so it opens exactly the connection the later
`POST …/control` reuses, and touches no call resource. This is a correctness measure,
not a latency one: the SSE fallback is the path carrying the §1.6 defect where a
flushed chunk on a stalled stream is never rendered to audio, so an acknowledgement
diverted there by a slow handshake risks the callee hearing nothing at all.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `/readyz` returns 503 | Hermes API server unreachable — check `VHV_HERMES_BASE_URL`, that `API_SERVER_ENABLED=true`, and that `hermes gateway` is running |
| 401 from Hermes | `VHV_HERMES_API_KEY` doesn't match `API_SERVER_KEY` in `~/.hermes/.env` (Hermes returns identical bodies for missing and wrong keys) |
| Vapi says the model failed / caller hears a platform error | Adapter returned non-2xx: check for 401 (custom-llm credential in Vapi doesn't match `VHV_ADAPTER_API_KEY`) or 404 (`VHV_ROUTE_SECRET` set but URL configured without `/v/<secret>`) |
| Caller hears "this number isn't available" | Allowlist denial on an inbound call — the caller isn't in `VHV_ALLOWED_CALLERS`, or no caller identity arrived (web call, or `metadataSendMode` set to `off`) |
| First call is very slow | Provider cold start (up to 24 s measured) — leave `VHV_WARMUP_ON_START=true` and wait for `/readyz` before routing traffic |
| Caller hears "all lines are busy" | Adapter concurrent-turn cap reached (`VHV_MAX_CONCURRENT_TURNS`), or Hermes is at `max_concurrent_runs` — note the Hermes cap is shared with all other API work |
| Caller hears "flush" spoken aloud | `voice.chunkPlan.enabled` was set to `false` on the assistant — set `VHV_FILLER_USE_FLUSH=false` or re-enable chunking |
| Agent speaks markdown artifacts | Should not happen (sanitizer); if it does, file a bug with the raw Hermes output |
| Caller hears a string of holding lines with no answer in between | Should not happen: `VHV_FILLER_MIN_GAP_SECONDS` (default 10 s) is a call-global cooldown, so lines can never be closer together than that however many turns, `tool_start` re-arms or barge-in retries occur. If a long agentic turn is genuinely producing one line per cooldown window, raise the timeout budgets (`VHV_HERMES_FIRST_TOKEN_TIMEOUT`, `VHV_HERMES_TURN_TIMEOUT`) or the cooldown, not the threshold |
| Caller hears the same acknowledgement twice in quick succession | The per-call state was lost between turns (TTL `VHV_CALL_STATE_TTL_SECONDS`, LRU `VHV_MAX_TRACKED_CALLS`), or no `call.id` arrived so no state could be kept — check `metadataSendMode` is `variable` |
| Adapter's own log shows a filler was spoken (`turn filler ... channel=stream`) but the callee heard silence for many seconds | Known Vapi platform behaviour, not an adapter bug (docs/integration-contracts.md §1.6): a `<flush />`-terminated chunk left alone in the model.url stream for more than a few seconds is not reliably rendered, however long the adapter waits. Confirm `VHV_ACK_USE_CALL_CONTROL=true` (default) and check for `channel=control` in the log instead — the control channel is immune to this stall. If `channel=stream` appears even with the setting on, the request carried no `call.monitor.controlUrl` or the control POST itself failed (a `ack control request failed`/`rejected` warning will be adjacent in the log) |
| Service will not start after an upgrade, `extra_forbidden` in the log | A `.env` key the adapter no longer understands. Retired keys (see `_RETIRED_SETTINGS` in `config.py`) are ignored with a warning naming them; anything else is a typo and is rejected |
| Agent never uses tools it should have access to | `VHV_TOOL_POLICY__ENABLED_TOOLS` was left unset while relying on the old "empty = forbid everything" behavior — empty/unset now means "no client-side opinion" (defers to the Hermes profile); check the Hermes profile's own toolset config |
| Outbound assistant ignores the objective and asks "how can I help?" | The `purpose` variable never arrived. It must be set as `assistantOverrides.variableValues.purpose` on the `POST /call` body (not in the dashboard prompt, which cannot receive values), and the call must be an `outboundPhoneCall` |
| Outbound assistant greets the principal instead of the callee | Same cause — with no `purpose` the adapter keeps the old inbound-flavored opening. Add `purpose` (and ideally `callee`) to `variableValues` |
| Assistant says "this is Emma calling for Mike" when Mike answered | `VHV_PRINCIPAL_NUMBER` is unset, so the adapter assumes a third party. Set it to the principal's E.164 number |
| Adapter refuses to start: "outbound opening templates must contain the {purpose} placeholder" | A custom `VHV_OUTBOUND_OPENING` / `VHV_OUTBOUND_OPENING_PRINCIPAL` dropped `{purpose}`, which would silently discard every objective — put the placeholder back |
| Opening line starts with a holding phrase | Fixed: fillers are suppressed on the synthetic opening turn. If seen again, confirm the request really had no trailing `user` message |
| Callee says "Hello?" and hears nothing for several seconds | The reason line did not fire. Check the log for `outbound reason spoken locally`: absent means the call carried no `purpose`/`spoken_reason`, was not an `outboundPhoneCall`, was not the callee's first utterance, or `VHV_OUTBOUND_REASON_FAST_PATH` is off |
| Assistant opens with "Is this a good moment to talk?" and no reason | The call sent `purpose` but no `spoken_reason`, or the `spoken_reason` was instruction-shaped and was refused rather than guessed at. Add a plain one-clause `spoken_reason` such as `"about his knee MRI results"` |
| Assistant reads the objective out like a script ("Goal: next steps...") | Should be impossible: `purpose` is never spoken. If heard, the text was passed in `spoken_reason` (or an alias) rather than `purpose` — and even then it should have been reduced. File a bug with the `variableValues` keys used |
| Callee hears the reason twice | Both the `CallState` latch and the conversation-shape check would have to fail at once. Check whether `call.id` is reaching the adapter (`metadataSendMode: variable`); with no call id the latch cannot persist between turns |
| Callee hears the lead-in twice ("I am calling about I am calling about ...") | Fixed: a `spoken_reason` that is already a whole clause has its redundant lead-in deleted before the sentence is built. If seen again, file a bug with the `spoken_reason` value used |
| Callee hears a reason that stops mid-thought ("...results from August") | The spoken clause is capped (24 words / 160 chars). Shorten the `spoken_reason`, or raise `MAX_REASON_TOPIC_WORDS`/`MAX_REASON_TOPIC_CHARS` in `speech.py` — but note the deletion rules run first, so a cut can also mean a comma, dash or second sentence ended the clause |
| Callee hears a holding phrase from the assistant, not from the adapter's pool | The model wrote it. The voice system prompt forbids opening a reply with one, so check what is layered on top: the Vapi dashboard system prompt and the Hermes profile's own prompt. Attribute by log, not by text — `turn filler call=<ref> elapsed_ms=<n>` is the only proof the adapter spoke a line, and several of the built-in phrases are exactly what a model would improvise |

## Testing

```sh
pytest
```

The suite is fully offline: a programmable fake Hermes ASGI app is mounted via
`httpx.ASGITransport`, and no real network is touched. There are no live tests in CI;
any test requiring real Vapi or Hermes credentials is opt-in and local-only.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
