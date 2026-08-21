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
  no user utterance yet).
- **Filler phrases** — if no speakable text has arrived within
  `VHV_FILLER_AFTER_SECONDS` (default 1.5 s), the adapter speaks a non-repeating
  holding line from `VHV_FILLER_PHRASES`, suffixed with Vapi's `<flush />` token so
  it is voiced immediately instead of sitting in the TTS buffer. The timer re-arms
  on tool-start events (each Hermes tool round trip adds roughly 2.9 s of dead air),
  but a filler can never be spoken once the answer has started, at most
  `VHV_FILLER_MAX_PER_TURN` (default 1) play in one turn, no two fillers in the same
  turn are ever closer together than `VHV_FILLER_MIN_GAP_SECONDS` (default 8 s), and
  no phrase repeats within one turn — a caller never hears a machine-gun run of
  holding lines on a long agentic turn.
- **Sanitization** — Hermes output is converted to speakable prose before it reaches
  TTS: markdown, code fences, tables, and emoji are stripped; URLs become
  "a link I can send you". Streaming-safe (constructs that span deltas are buffered).
- **Barge-in / hangup** — when Vapi abandons the request, the adapter stops the
  Hermes run (`POST /v1/runs/{id}/stop`, measured 0.255 s) — abandoned runs
  otherwise execute unboundedly.

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

  Always set provider together with model — a bare model without a provider is
  silently ignored or mis-routed by Hermes.
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
> — web calls, `metadataSendMode: "off"` — are **denied**), and run Hermes with
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
| `VHV_ALLOWED_CALLERS` | `[]` | E.164 caller allowlist; empty = allowlist disabled (allow all, warn); non-empty fails closed on missing caller identity |
| `VHV_ASSISTANT_NAME` | `the assistant` | Name used in the voice instructions |
| `VHV_PRINCIPAL` | `the operator` | Whose assistant it says it is |
| `VHV_FILLER_PHRASES` | 8 built-in phrases | Holding lines spoken during dead air; must be non-empty |
| `VHV_FILLER_AFTER_SECONDS` | `1.5` | Dead-air threshold before a filler opportunity (first one, and each tool-start re-arm) |
| `VHV_FILLER_MIN_GAP_SECONDS` | `8.0` | Structural floor between the end of one filler and the start of the next, checked when a filler would be spoken regardless of how it was re-armed |
| `VHV_FILLER_MAX_PER_TURN` | `1` | Hard cap on holding lines per turn (bounded to 1-3 regardless of configured value); once real content starts, no more are ever spoken |
| `VHV_FILLER_USE_FLUSH` | `true` | Suffix fillers with `<flush />` for immediate TTS (requires Vapi's default `chunkPlan.enabled`) |
| `VHV_VOICE_MODEL` | *(unset)* | Hermes model override for voice turns, e.g. `google/gemini-3.7-flash` |
| `VHV_VOICE_PROVIDER` | *(unset)* | Provider for `VHV_VOICE_MODEL`; always set together with it |
| `VHV_VOICE_REASONING_EFFORT` | `low` | `model_options.reasoning_effort` sent to Hermes |
| `VHV_WARMUP_ON_START` | `true` | Fire one tiny routed run at startup to absorb provider cold start |
| `VHV_HERMES_CONNECT_TIMEOUT` | `5.0` | Hermes HTTP connect timeout (s) |
| `VHV_HERMES_FIRST_TOKEN_TIMEOUT` | `15.0` | Max wait for the first Hermes token (s) |
| `VHV_HERMES_TURN_TIMEOUT` | `60.0` | Max wall time for one Hermes turn (s) |
| `VHV_HERMES_STOP_TIMEOUT` | `3.0` | Bound on the mandatory `POST /v1/runs/{id}/stop` (s) |
| `VHV_MAX_CONCURRENT_TURNS` | `5` | Adapter concurrent-turn cap; keep below Hermes `max_concurrent_runs` (default 10) |
| `VHV_MAX_HISTORY_MESSAGES` | `200` | Conversation-history truncation (keeps most recent) |
| `VHV_MAX_BODY_BYTES` | `1000000` | Inbound request body cap, enforced while reading |
| `VHV_SESSION_RETENTION` | `none` | `none` = per-call random session ids; `hermes` = let Hermes persist sessions |
| `VHV_CALL_STATE_TTL_SECONDS` | `14400` | Per-call state eviction TTL (session ids + filler picker) |
| `VHV_MAX_TRACKED_CALLS` | `1024` | Per-call state LRU cap |
| `VHV_TOOL_POLICY__ENABLED_TOOLS` | `[]` | Advisory: tools voice turns may use (prompt-shaping only, not enforcement). Empty/unset = no client-side opinion (defers to the Hermes profile); set to `none` explicitly to tell the model to use no tools |
| `VHV_TOOL_POLICY__CONFIRM_TOOLS` | `[]` | Advisory: tools requiring spoken confirmation |
| `VHV_TOOL_POLICY__MAX_TOOL_CALLS_PER_TURN` | `3` | Advisory per-turn tool budget |
| `VHV_TOOL_POLICY__MAX_TOOL_SECONDS_PER_CALL` | `60.0` | Advisory per-call tool time budget |

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `/readyz` returns 503 | Hermes API server unreachable — check `VHV_HERMES_BASE_URL`, that `API_SERVER_ENABLED=true`, and that `hermes gateway` is running |
| 401 from Hermes | `VHV_HERMES_API_KEY` doesn't match `API_SERVER_KEY` in `~/.hermes/.env` (Hermes returns identical bodies for missing and wrong keys) |
| Vapi says the model failed / caller hears a platform error | Adapter returned non-2xx: check for 401 (custom-llm credential in Vapi doesn't match `VHV_ADAPTER_API_KEY`) or 404 (`VHV_ROUTE_SECRET` set but URL configured without `/v/<secret>`) |
| Caller hears "this number isn't available" | Allowlist denial — the caller isn't in `VHV_ALLOWED_CALLERS`, or no caller identity arrived (web call, or `metadataSendMode` set to `off`) |
| First call is very slow | Provider cold start (up to 24 s measured) — leave `VHV_WARMUP_ON_START=true` and wait for `/readyz` before routing traffic |
| Caller hears "all lines are busy" | Adapter concurrent-turn cap reached (`VHV_MAX_CONCURRENT_TURNS`), or Hermes is at `max_concurrent_runs` — note the Hermes cap is shared with all other API work |
| Caller hears "flush" spoken aloud | `voice.chunkPlan.enabled` was set to `false` on the assistant — set `VHV_FILLER_USE_FLUSH=false` or re-enable chunking |
| Agent speaks markdown artifacts | Should not happen (sanitizer); if it does, file a bug with the raw Hermes output |
| Caller hears a string of holding lines with no answer in between | A long agentic Hermes turn kept re-arming the filler on `tool_start`; capped by `VHV_FILLER_MAX_PER_TURN` (default 1) and `VHV_FILLER_MIN_GAP_SECONDS` (default 8 s) — raise the timeout budgets (`VHV_HERMES_FIRST_TOKEN_TIMEOUT`, `VHV_HERMES_TURN_TIMEOUT`) instead of the filler cap if the underlying tool call is just slow |
| Agent never uses tools it should have access to | `VHV_TOOL_POLICY__ENABLED_TOOLS` was left unset while relying on the old "empty = forbid everything" behavior — empty/unset now means "no client-side opinion" (defers to the Hermes profile); check the Hermes profile's own toolset config |

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
