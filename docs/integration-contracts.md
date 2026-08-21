# Integration contracts: Vapi Custom LLM ↔ Hermes Agent

Canonical distillation of the evidence this adapter is built against. Every claim
carries its verification class:

- **DOCS** — read from the official Vapi documentation at https://docs.vapi.ai
  (full-page fetches; append `.md` to any page URL for clean markdown) or from the
  published OpenAPI spec `https://api.vapi.ai/openapi/api-reference.json`.
- **LIVE** — observed by a real probe. All LIVE claims in §2 (Hermes) were measured
  against the operator's Hermes Agent 0.20.4 install during the retell-hermes-voice
  recon (2026-08-20) and are inherited unchanged — the Hermes side of this adapter is
  identical code. **No live Vapi call has been placed**; there are no LIVE Vapi claims.
- **UNVERIFIED / ASSUMED** — inferred; collected in §4 with the exact reason each one
  could not be verified without a live Vapi account.

| Verified at | Component | Version / revision | Method |
|---|---|---|---|
| 2026-08-21 | Vapi docs + OpenAPI spec | docs.vapi.ai live pages, OpenAPI 3.1 spec as published | DOCS (full-page fetches, spec downloads) |
| 2026-08-20 | Hermes Agent API server | **0.20.4** (aiohttp 3.14.3, Python 3.11) | LIVE probes + SOURCE reads (see retell-hermes-voice `docs/integration-contracts.md` §2 for full evidence) |

---

## 1. Vapi Custom LLM contract (DOCS)

### 1.1 Transport: HTTP POST + SSE, not WebSocket

Vapi's Custom LLM integration drives your server with the **OpenAI client**: it POSTs
an OpenAI-chat-completions-shaped JSON body to `{url}/chat/completions` and consumes
the response.

- Guide: https://docs.vapi.ai/customization/custom-llm/using-your-server
- Tool-calling guide (shows the streaming loop):
  https://docs.vapi.ai/customization/tool-calling-integration

The assistant's `model.url` field is documented as **"the URL we'll use for the OpenAI
client's `baseURL`"** (Create Assistant API reference, `CustomLLM.url`,
https://docs.vapi.ai/api-reference/assistants/create) — i.e. the OpenAI client appends
`/chat/completions`. Configure the **base** URL, without a `/chat/completions` suffix.

> **Doc conflict, flagged:** the tool-calling guide's PATCH example sets
> `"url": "https://custom-llm-url/chat/completions"` — with the suffix — which
> contradicts the schema's `baseURL` semantics. Whether Vapi normalizes a trailing
> `/chat/completions` is UNVERIFIED (§4). This adapter serves the handler at both
> `/chat/completions` and `/chat/completions/chat/completions` so either
> configuration works.

### 1.2 Request payload (Vapi → adapter)

OpenAI chat-completions fields, per the guides above: `model`, `messages`
(`{role, content}`; roles `assistant|function|user|system|tool` per the
`CustomLLM.messages` schema), `temperature` (assistant `model.temperature`, default
0.5), `max_tokens` (assistant `model.maxTokens`, default 250), `stream`, `tools`
(when the assistant has attached tools), plus Vapi extras (e.g. `destination` for
transfer flows) — all DOCS.

**Call metadata** is governed by the assistant's `model.metadataSendMode`
(Create Assistant API reference, quoted verbatim):

> `off` will not send any metadata. payload will look like `{ messages }` ·
> `variable` will send `assistant.metadata` as a variable on the payload. payload
> will look like `{ messages, metadata }` · `destructured` will send
> `assistant.metadata` fields directly on the payload ·
> **Further, `variable` and `destructured` will send `call`, `phoneNumber`, and
> `customer` objects in the payload. Default is `variable`.**

So with the default mode the adapter receives `call` (incl. `call.id`, `call.type` ∈
`inboundPhoneCall|outboundPhoneCall|webCall`), `customer` (incl. `customer.number`,
E.164 — the caller), and `phoneNumber` (the Vapi number dialed). This adapter keys
per-call session state on `call.id` and enforces the caller allowlist on
`customer.number` **for inbound calls only** — on an outbound call
`customer.number` is the callee, so screening it against a list of permitted
callers would deny the operator's own task calls. **Do not set `metadataSendMode: "off"`** — it removes the caller
identity the allowlist needs (the adapter then fails closed when an allowlist is
configured).

Timeout: assistant `model.timeoutSeconds` — "the timeout for the connection to the
custom provider without needing to stream any tokens back. **Default is 20
seconds.**" (Create Assistant API reference). The adapter's filler phrases and
15 s first-token timeout keep every response under that window.

### 1.3 Response: OpenAI SSE chunks

The documented pattern (tool-calling guide) re-streams OpenAI chunk objects verbatim:

```
Content-Type: text/event-stream

data: {"id":"…","object":"chat.completion.chunk","created":…,"model":"…",
       "choices":[{"index":0,"delta":{"content":"…"},"finish_reason":null}]}

data: [DONE]
```

This adapter emits exactly that: a first chunk with `delta: {"role": "assistant"}`,
content chunks, a final chunk with `finish_reason: "stop"`, then `data: [DONE]`.
The basic guide also shows a **non-streaming** JSON response (Flask `jsonify`)
being accepted; the adapter honors `"stream": false` with a complete
`chat.completion` object for curl-based debugging. Which chunk fields Vapi strictly
requires is UNVERIFIED (§4) — emitting the full OpenAI shape is the safe superset.

### 1.4 Auth (a real improvement over Retell)

Per https://docs.vapi.ai/customization/custom-llm/using-your-server (Authentication
section), Vapi supports two per-endpoint auth methods for custom LLMs:

1. **API key** — "Vapi sends your API key in the Authorization header to your custom
   LLM endpoint. Your server validates the API key before processing the request."
2. **OAuth2 client-credentials** — token endpoint + client id/secret, automatic
   refresh; Vapi presents the token in the Authorization header.

The credential object (OpenAPI `CreateCustomLLMCredentialDTO`):
`{"provider": "custom-llm", "apiKey": "<secret>", "name": "…"}`, with an optional
`authenticationPlan` (OAuth2 RFC 6749: `url`, `clientId`, `clientSecret`, `scope`).
The schema states: **"To use Bearer authentication, use apiKey."** Credentials can be
attached workspace-wide (dashboard → Model → Custom LLM auth), or per assistant via
`assistant.credentialIds` / transient `assistant.credentials` (both verified present
in the Create Assistant schema).

This adapter uses the **API-key mechanism as primary**: it requires
`Authorization: Bearer <VHV_ADAPTER_API_KEY>` (also accepts the raw key without the
`Bearer` prefix — the exact header format Vapi emits is UNVERIFIED, §4) and compares
in constant time. An optional **route secret** path prefix
(`/v/{VHV_ROUTE_SECRET}/chat/completions`) is available as defense-in-depth; unlike
retell-hermes-voice it is optional because real header auth exists here.

Custom headers: `model.headers` can add arbitrary headers **except** Authorization
("which should be specified using a custom-llm credential") — Create Assistant
API reference.

### 1.5 Interruptions / barge-in

Vapi owns telephony, STT, TTS, turn-taking, and barge-in (voice pipeline docs:
https://docs.vapi.ai/customization/voice-pipeline-configuration). **No Vapi document
found describes how an in-flight custom-LLM HTTP request is treated when the caller
interrupts.** The engineering-safe reading (§4, ASSUMED): Vapi aborts the in-flight
HTTP request (the only cancellation surface HTTP offers) and/or simply discards the
rest of the stream, then issues a fresh POST for the next turn.

Adapter behavior — safe under every possible Vapi behavior:

- Client disconnect or request cancellation mid-stream → the response generator is
  finalized → the Hermes run is **always stopped** (`POST /v1/runs/{id}/stop`,
  bounded by `VHV_HERMES_STOP_TIMEOUT`). No orphaned Hermes runs, ever (§2).
- A new POST for the same `call.id` while an old one is still draining is fine: runs
  have distinct `run_id`s and per-call session state is keyed on `call.id`.

### 1.6 Filler phrases and the `<flush />` audio-control token

Mid-stream filler injection **is** possible on Vapi — better than on Retell, in fact.
Anything the adapter emits as a content delta is speakable, and Vapi documents an
explicit audio-control token to defeat its own TTS buffering:

- https://docs.vapi.ai/assistants/flush-syntax — `<flush />` (also `<flush>`,
  `</flush>`, case-insensitive, regex `/<\s*flush\s*\/?>|<\s*\/\s*flush\s*>/i`)
  "forces immediate transmission of LLM output to voice providers, eliminating
  buffering delays". Explicitly recommended to "support custom LLM integrations
  with processing delays".
- Requires `voice.chunkPlan.enabled: true`, which is the **default** (OpenAPI
  `ChunkPlan.enabled`: "@default true"). If chunking is disabled the tags would be
  spoken; `VHV_FILLER_USE_FLUSH=false` turns the suffix off for such setups.

The adapter speaks a brief acknowledgement after `VHV_FILLER_AFTER_SECONDS`
(default 0.9 s) of dead air, suffixed with ` <flush />`, and re-arms the timer on
Hermes tool-start events (each Hermes tool round trip ≈ +2.9 s of dead air, §2).
It fires on every turn, including the first, and is not conditional on tool
activity: it exists so the callee gets an immediate answer to having been spoken
to, not to announce a lookup. Two limits keep a long agentic turn from producing a
machine-gun run (observed live: five in a row with no content between them):

- It can never be spoken once the answer has started (`content_started`, checked at
  the moment of speaking, not just when the deadline is armed). A turn Hermes
  answers inside the threshold therefore gets none.
- `VHV_FILLER_MIN_GAP_SECONDS` (default 10 s) is a cooldown **global to the call**,
  held on `CallState.last_ack_at` and claimed via
  `CallState.claim_acknowledgement()`. Once one is spoken, nothing on that call
  speaks another until the gap expires — across turn boundaries, tool-start re-arms
  and cancelled retries alike. The anchor is stamped at claim time, before the SSE
  frame is yielded, so a stream Vapi tears down mid-flight still spends the slot:
  the observed barge-in storm (six POSTs for one turn inside sixteen seconds, five
  cancelled) yields one line, not six. A refused claim spends nothing.

The 0.9 s threshold is arithmetic. The requirement is audible speech within 2 s of
the callee finishing their sentence, and Vapi spends ~0.4-1.6 s of that on
transcriber endpointing plus `startSpeakingPlan.waitSeconds` before this adapter is
invoked at all, so the adapter's own share has to stay under a second.

### 1.7 Tool calling

Three documented patterns (https://docs.vapi.ai/customization/tool-calling-integration):

1. **Native LLM tools** — tool_calls streamed by your LLM, executed by your server,
   result fed back. *This adapter's equivalent:* Hermes executes its own tools
   server-side inside the run; tool activity surfaces on the Hermes SSE stream as
   `tool.started` events, which the adapter uses only to re-arm the filler timer.
   The adapter never emits `tool_calls` chunks.
2. **Vapi-attached tools** (e.g. `transferCall`) — the server echoes a
   `function_call` frame naming the tool; Vapi executes it. Not used: do not attach
   Vapi tools to a hermes-voice assistant; the adapter will never invoke them.
3. **Custom tools** — a separate `/chat/completions/custom-tool` endpoint receiving
   `{"message": {"toolCallList": […]}}` and answering `{"results": […]}`. Not
   implemented — Hermes owns all tools (non-goal, see README).

Request `tools` arrays are accepted and ignored (logged at debug).

### 1.8 Call lifecycle webhooks

Vapi sends server events (`status-update` with `status: "ended"`,
`end-of-call-report`, etc.) to a configured Server URL —
https://docs.vapi.ai/server-url/events. The adapter does **not** require them:
per-call state is evicted by TTL (`VHV_CALL_STATE_TTL_SECONDS`) and LRU cap
(`VHV_MAX_TRACKED_CALLS`), and each request carries the full message history, so no
state loss can corrupt a call. Wiring the webhook for eager cleanup is optional
future work and would need its own auth
(https://docs.vapi.ai/server-url/server-authentication).

### 1.9 Greeting / first message

Vapi speaks the greeting itself: `assistant.firstMessage` ("If unspecified, assistant
will wait for user to speak") and `firstMessageMode`
(default `assistant-speaks-first`) — Create Assistant API reference. A **static**
`firstMessage` never reaches the adapter, so it always wins over anything below.

With `firstMessageMode: assistant-speaks-first-with-model-generated-message` the
adapter *is* asked to produce the opening: the request arrives with no trailing `user`
message, and `policy.split_messages` substitutes a synthetic turn input
(`policy.build_opening_nudge`). Two behaviors are specific to that path:

- **Direction-aware opening.** Inbound — and outbound with no `purpose` — use the
  unchanged `OPENING_NUDGE` ("greet them briefly and ask how you can help"). An
  outbound call carrying a `purpose` uses an outbound template instead; see §1.10.
- **Acknowledgements are not suppressed.** They used to be
  (`stream_turn(allow_fillers=False)`), on the theory that nothing is pending before
  the callee has spoken. Live evidence overturned it: the callee's report was "when
  it first calls, I pick up, and then it's like 10 seconds before it says anything",
  i.e. the one turn that could not cover dead air was the one that most needed to.
  The parameter is gone; every turn is treated alike. Where the opening line is
  produced locally instead of by Hermes, `content_started` suppresses the
  acknowledgement for free, without spending the call-global cooldown.

### 1.10 Dynamic variables: outbound task calls (DOCS + LIVE-verified 2026-08-21)

An outbound *task* call ("call my doctor and move Marvin's recheck to Tuesday") needs
the objective to reach the model. Vapi carries it as a **dynamic variable**, set via
`assistantOverrides.variableValues` when the call is created (Variables reference,
`POST https://api.vapi.ai/call`). The Call object echoes `assistantOverrides`, so the
adapter reads the values straight off the Custom LLM request body.

**Variables the adapter understands** (everything else is ignored, never an error):

| Key | Required | Meaning |
| --- | --- | --- |
| `purpose` | yes, for task calls | Free-text objective. Absent ⇒ behavior is exactly as before this feature. |
| `callee` | no | Free-text description of who is being called ("Dr. Patel's office"). |

**Locations searched**, most specific first; each key is taken from the first location
that supplies it, so a body splitting them across depths still yields both:

1. `call.assistantOverrides.variableValues` — the documented location, LIVE-verified.
2. `assistantOverrides.variableValues`
3. `variableValues`
4. `call.variableValues`

**These values are UNTRUSTED.** A call can be created through a path a caller
influences, so `vapi_events._clean_variable` strips control characters,
collapses all whitespace (a value can never forge the blank line that separates
authoritative prompt sections), and caps length — `purpose` at
`MAX_PURPOSE_CHARS` (400), `callee` at `MAX_CALLEE_CHARS` (120). Non-string values are
ignored rather than coerced. Values are **never logged**: only
`CallVariables.log_summary()` (`purpose_chars=… callee_chars=…`) reaches a log line.

#### Instruction precedence

`speech.build_instructions` layers paragraphs in this order (Hermes's own resident
system prompt sits under all of it):

1. `VOICE_SYSTEM_PROMPT` — phone/TTS style and safety rules.
2. Identity — who the assistant is, and which way the call went.
3. Tool policy, when configured.
4. `extra` — the Vapi dashboard system prompt. Standing operator configuration.
5. **Task paragraph — this call's `purpose`. Last.**

`purpose` goes *after* the dashboard prompt deliberately: it is the most specific
instruction in the prompt (the reason this one call exists), and a generic
Mike-flavored dashboard prompt must not be able to talk the model out of the job it was
dialed to do. The task paragraph closes by handing authority back to layer 1, so the
voice and safety rules stay authoritative and the objective is framed explicitly as
data describing a task rather than a source of new rules.

#### Who is on the other end

`VapiChatRequest.callee_is_principal` resolves this, and it changes both the opening
and the identity framing:

1. `VHV_PRINCIPAL_NUMBER` vs `customer.number`. `customer.number` is
   signalling-derived, so when the principal's own number is configured it is
   authoritative and `callee` is not consulted.
2. Otherwise an exact (case/whitespace-insensitive) `callee` match against
   `VHV_PRINCIPAL`. Deliberately not a substring test — "Mike's doctor" contains
   "Mike", and reading that as "we called Mike" would drop the disclosure owed to a
   third party and greet a stranger by the operator's name.
3. Otherwise **third party** — the safe default, and the behavior for operators who
   never set `VHV_PRINCIPAL_NUMBER`.

| Counterparty | Opening template | Framing | AI disclosure |
| --- | --- | --- | --- |
| Third party (default) | `VHV_OUTBOUND_OPENING` | who is calling, on whose behalf, the reason | yes, when `VHV_OUTBOUND_DISCLOSE_AI` |
| The principal | `VHV_OUTBOUND_OPENING_PRINCIPAL` | greeted directly by name, no on-behalf-of | no — nobody to disclose to but the operator |

Both templates **must** contain `{purpose}` (validated at config load; without it the
objective would silently never reach the model). `{principal}`, `{assistant_name}`, and
`{callee}` are also substituted; any other `{name}` is left as literal text.
Substitution is a single non-recursive pass (`policy._render`), never `str.format`, so
an untrusted value containing `{...}` cannot reach back into the template or into
Python objects.

The callee sentence and the AI disclosure are appended *outside* the template: the
disclosure is a safety control, and editing a template must not silently drop it.

#### The contract Hermes uses to launch a task call

`POST https://api.vapi.ai/call`, `Authorization: Bearer <VAPI_PRIVATE_API_KEY>`:

```json
{
  "assistantId": "b39379dc-ca93-48aa-a72a-a41d92279b4f",
  "phoneNumberId": "<your Vapi phone number id>",
  "customer": { "number": "+15551234567" },
  "assistantOverrides": {
    "variableValues": {
      "purpose": "reschedule Marvin's cardiology recheck to next Tuesday afternoon",
      "callee": "Dr. Patel's office"
    }
  }
}
```

`customer.number` is who gets dialed. `purpose` is the objective and is required for
task-call behavior; `callee` is optional but makes the opening concrete. Omit both and
the call behaves exactly like any other outbound call.

---

## 2. Hermes API server contract (v0.20.4, LIVE-verified 2026-08-20, inherited)

Full evidence with probe timings lives in the predecessor repo:
https://github.com/mjaverto/retell-hermes-voice/blob/main/docs/integration-contracts.md §2.
The Hermes-facing code in this adapter is a near-verbatim port; every claim below was
LIVE-verified there and the adapter preserves the exact same behavior.

- **Transport**: `POST /v1/runs` (202, `{"run_id", "status": "started"}`) →
  `GET /v1/runs/{id}/events` (SSE) → `POST /v1/runs/{id}/stop` on any abandon.
  Fields: `input`, `session_id`, `instructions` (layered on top of Hermes's own
  system prompt), `conversation_history`, `model`/`provider`/`model_options`.
  Headers `X-Hermes-Session-Id`, `X-Hermes-Session-Key`.
- **SSE framing**: bare `data:` frames, event name in the JSON `event` field:
  `message.delta` (`delta`), `tool.started`, `run.completed` (`output` + `usage`),
  `run.failed`, `run.cancelled`. Keepalive comment `: keepalive` after 30 s;
  terminator `: stream closed`. Single-subscriber, non-resumable.
- **Mandatory stop**: abandoning the events stream leaves the run executing
  **unboundedly** (LIVE: still running with live child processes at +30 s);
  `POST /v1/runs/{id}/stop` cancels in 0.255 s; stop on a finished run 404s
  harmlessly. This is why the adapter stops the run on every disconnect,
  cancellation, and timeout.
- **Fail-open hazard**: unknown provider → HTTP 200 with the error text as assistant
  content and zero usage; a bad model with valid provider silently falls back to the
  default route. The adapter intercepts known error prefixes (definite: `⚠️`,
  `provider authentication failed`; ambiguous: `error:` corroborated by zero/absent
  usage) and speaks a generic apology instead.
- **Concurrency**: `max_concurrent_runs` default 10, shared across all API work;
  429 with `Retry-After: 1`. The adapter caps itself below it
  (`VHV_MAX_CONCURRENT_TURNS`, default 5) and speaks a busy line at saturation.
- **Latency** (LIVE, n=3/row): default route **2.983 s** median TTFB; routed
  `openrouter`/`google/gemini-3.7-flash` + `reasoning_effort: low` **0.605 s**
  (~5×). Cold start up to **24 s** on a provider's first call → warmup run at
  startup. Provider must ALWAYS be set together with model (a bare model is
  silently ignored or mis-routed). Tool turns: +2.9 s each; first `tool.started`
  ≈ 2 s before first content.
- **Tool restriction is IMPOSSIBLE from the API client** (8/8 structural attempts
  silently ignored). The only enforcement point is the Hermes profile:
  `hermes tools disable --platform api_server …`. The adapter's tool policy is
  advisory prompt-shaping.
- **Cross-session memory leak** (reproduced 3/3): long-term memory is not
  session-scoped; the voice profile MUST disable `memory`/`session_search`.
- **Sessions**: `session_id` scopes transcript, `session_key` scopes memory; ids
  validated (max 256 chars, no `/`, `\`, `..`, CR/LF/NUL). The adapter derives
  non-reversible ids (`vhv-` + sha256 prefix) or per-call random ids, never
  phone-number-derived.

---

## 3. Vapi management API (DOCS, spec as published 2026-08-21)

Base `https://api.vapi.ai`, bearer auth with the **private** API key
(https://docs.vapi.ai/security-and-privacy/api-keys).

| Purpose | Method + path | Notes |
|---|---|---|
| Create assistant | `POST /assistant` | body incl. `model: {"provider": "custom-llm", "url": "https://host/v/<route-secret>", "model": "hermes", "metadataSendMode": "variable", "timeoutSeconds": 20}`, `firstMessage`, `voice`, `transcriber`; per-assistant auth via `credentialIds` or transient `credentials: [{"provider": "custom-llm", "apiKey": "<VHV_ADAPTER_API_KEY>"}]` |
| Update assistant | `PATCH /assistant/{id}` | same fields |
| List/get/delete | `GET /assistant`, `GET/DELETE /assistant/{id}` | |
| Phone numbers | `POST /phone-number` | providers `vapi` (free, `numberDesiredAreaCode`), `twilio`, `telnyx`, `byo-phone-number`; bind with `assistantId`. Free numbers: https://docs.vapi.ai/free-telephony |
| List calls | `GET /call` | artifacts per https://docs.vapi.ai/assistants/retrieve-call-artifacts |
| Outbound call | `POST /call` | `assistantId` + `customer.number` + `phoneNumberId` |

Workspace-wide custom-llm credentials are managed in the dashboard (Model → Custom
LLM → authentication; https://docs.vapi.ai/customization/custom-llm/using-your-server).
The credential-management REST endpoints are not present in the published OpenAPI
spec — use the dashboard or per-assistant `credentials`/`credentialIds`.

**Hosted MCP server** (for the later Hermes-side config swap, not used by this
adapter): `https://mcp.vapi.ai/mcp` (streamable-HTTP; also `/sse`), auth
`Authorization: Bearer <VAPI_API_KEY>`; tools include `list_assistants`,
`create_assistant`, `create_call`, `list_phone_numbers` —
https://docs.vapi.ai/sdk/mcp-server.

---

## 4. Assumed / unverified (requires a live Vapi account to close)

- **Exact Authorization header format for API-key credentials.** DOCS says "Vapi
  sends your API key in the Authorization header"; the credential schema says "To
  use Bearer authentication, use apiKey". Whether the header is `Bearer <key>` or
  the bare key is not spelled out. The adapter accepts both, constant-time either way.
- **Trailing `/chat/completions` normalization** in `model.url` (§1.1 doc conflict).
  Adapter serves both paths; recommend configuring the base URL.
- **Interruption/barge-in signaling to a custom LLM** (§1.5). Assumed: HTTP request
  abort and/or stream discard. The adapter's disconnect handling is correct under
  either; what a live call would confirm is *when* Vapi aborts.
- **Strictly required SSE chunk fields.** The docs re-stream full OpenAI chunks; a
  minimal `{choices:[{delta:{content}}]}` might also work. The adapter emits the
  full shape.
- **Whether Vapi sends `stream: true` explicitly** in the request body. Assumed yes
  (OpenAI client with `stream: true`); the adapter defaults to streaming unless
  `"stream": false` is explicit.
- **`(unintelligible audio)`-style magic transcript values.** Retell-specific; no
  Vapi equivalent documented. The voice prompt still tells the model to ask for a
  repeat when a transcript looks garbled.
- **Real call latency** end-to-end (Vapi STT + adapter + Hermes + TTS) and the
  effectiveness of `<flush />` with the chosen voice provider ("effectiveness varies
  by provider" — flush-syntax docs).
- **Vapi outbound IP addresses** for edge allowlisting. Static IPs exist as a
  paid/enterprise feature (https://docs.vapi.ai/security-and-privacy/static-ip-addresses);
  header auth replaces IP pinning in this design.

Everything in §4 is compensated for by design: header auth + optional route secret,
disconnect-safe run stops, full-shape SSE chunks, and both endpoint paths.

---

## 5. Consequences for this adapter (finding → design decision)

- **Header auth exists (§1.4)** → required `VHV_ADAPTER_API_KEY` (min 16 chars),
  constant-time compare, 401 before any parsing; optional `VHV_ROUTE_SECRET` path
  prefix as defense-in-depth. This closes retell-hermes-voice's biggest structural
  hole (an unauthenticated public WebSocket).
- **Vapi sends the whole conversation every turn (§1.2)** → the adapter is stateless
  per turn except for per-call session ids, the acknowledgement picker and its
  call-global cooldown anchor, keyed on `call.id` with TTL+LRU eviction; state loss
  can never corrupt a call (at worst the callee hears one acknowledgement sooner
  than the cooldown would have allowed).
- **Abandoned Hermes runs never stop on their own (§2, LIVE)** → the Hermes turn
  generator ALWAYS issues `POST /v1/runs/{id}/stop` on finalization unless the run
  reached a terminal event; the SSE response generator finalizes it on client
  disconnect, cancellation, and timeout alike.
- **20 s no-token connection timeout (§1.2)** → 15 s first-token timeout plus
  fillers from 1.5 s guarantee bytes on the wire well inside the window.
- **`<flush />` verified (§1.6)** → fillers are spoken immediately instead of
  sitting in Vapi's TTS buffer; configurable off.
- **Fail-open Hermes errors return HTTP 200 + error prose (§2, LIVE)** → intercepted
  and replaced with one safe apology sentence; provider/internal error text is never
  spoken or logged unredacted.
- **Client-side tool restriction impossible (§2, LIVE)** → ToolPolicy documented as
  advisory; the README's Hermes hardening section is a deployment requirement.
- **Cross-session memory leak (§2, LIVE)** → per-call random session ids by default
  (`VHV_SESSION_RETENTION=none`), never phone-derived; deployment requires
  `memory`/`session_search` disabled on the voice profile.
- **Caller identity arrives as `customer.number` (§1.2)** → allowlist enforced on
  it for inbound calls; with an allowlist configured and no metadata present, the
  adapter **fails closed** and speaks a denial line. Outbound calls are exempt:
  there the same field is the callee.
- **429 at Hermes concurrency cap (§2, LIVE)** → adapter caps its own concurrent
  turns and speaks a busy line as normal SSE content (a 5xx would just make Vapi
  retry or read a platform error).
