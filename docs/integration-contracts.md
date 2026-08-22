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
(default 0.3 s) of dead air, and re-arms the timer on
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

The 0.3 s threshold is arithmetic, and so is the ceiling on the control POST that
delivers the line. The requirement is audible speech within 2 s of the callee
finishing their sentence, measured on the callee's clock, and most of that interval
belongs to Vapi rather than to this process. Measured on the callee's own audio
devices, call `01a0262b`:

| | |
|---|---|
| callee stopped talking | 15.516 s |
| callee heard the first reply | 17.837 s |
| **heard gap** | **2.321 s** — a miss |
| adapter emitted its acknowledgement | 1.130 s after the turn reached it |

so 2.321 − 1.130 = **1.191 s** went on transcriber endpointing plus
`startSpeakingPlan.waitSeconds` ahead of this process and the TTS/transport hop after
it. None of that is optimisable from here, so it is budgeted — at 1.25 s, rounded up,
because one sample does not deserve three decimal places of trust —
as `VHV_ACK_PLATFORM_OVERHEAD_SECONDS`, leaving the adapter
`VHV_ACK_BUDGET_SECONDS − VHV_ACK_PLATFORM_OVERHEAD_SECONDS` = **0.75 s** for the
dead-air wait *and* the delivery POST together:

| | common case | worst case (control POST stalls) |
|---|---|---|
| dead-air wait | 0.300 s | 0.300 s |
| control delivery | 0.230 s measured | 0.450 s, the full ceiling, then the SSE fallback in-process |
| adapter total | 0.530 s | 0.750 s |
| + measured platform overhead | **1.721 s heard** | **1.941 s heard** |

The previous defaults could not fit: 0.9 s of dead-air wait plus a 3.0 s control
ceiling is 3.9 s before the platform is even charged for, and that is not a
hypothetical — it is the live journal, `turn filler ... elapsed_ms=3902 channel=stream`
after `ack control request failed error=ReadTimeout`. The POST is waited on *solely*
in order to give up on it and use the SSE fallback, so a ceiling larger than the whole
budget makes the fallback worthless however fast it is.

`VHV_ACK_CONTROL_TIMEOUT_SECONDS` is therefore **derived** from the other three rather
than chosen, and enforced on the **wall clock** rather than delegated to httpx: a bare
float handed to httpx is a *per-phase* value (`{"connect": t, "read": t, "write": t,
"pool": t}`), and a phase that consumes its whole share and then succeeds raises
nothing, so the caller waits for the sum. No phase is given a *tighter* sub-share than
the whole, though, and specifically not connect. Sub-dividing buys nothing for the
deadline — the wall-clock bound caps the total however the phases fall — and it costs
something real: a 0.20 s handshake leaving ample room for the 0.23 s round trip inside
the 0.45 s ceiling would be abandoned at a 0.18 s connect sub-share, and abandoned
*into* the SSE path below.

**Failing over is not a neutral second best**, which is what makes the handshake worth
engineering around. The SSE path is the one carrying the defect described in §1.6: a
flushed chunk on a stream that then stalls is accepted, echoed back in ~1 ms, and
frequently never rendered to audio at all. So an acknowledgement diverted there risks
the callee hearing *nothing*, not merely hearing it late.

Two things therefore keep the TCP+TLS handshake off the acknowledgement's deadline:

- **The connection pool is configured to actually pool.** httpx evicts idle
  connections after **five seconds** by default, and acknowledgements are at least
  `VHV_FILLER_MIN_GAP_SECONDS` (10 s) apart, with conversational turns often much
  further — so essentially *every* control POST was paying a fresh handshake, not just
  the first after a restart. The client (one per process, never rebuilt per turn) now
  keeps idle connections for 60 s: long enough to span a turn gap, short enough to stay
  under the idle timeouts load balancers typically enforce.
- **The origin is warmed at the top of every turn**, fire-and-forget, off the critical
  path. The control URL is per-call
  (`https://phone-call-websocket.<region>-backend-productionN.vapi.ai/<call-id>/control`)
  and cannot be known before the call exists — but httpx pools by *origin*, not path,
  and the origin arrives on the call's first request. So a `GET` of the origin root
  opens exactly the connection the later `POST …/control` reuses, while touching no
  call resource. It no-ops when the origin was used recently and re-warms when it went
  cold, so an arbitrary silence in the conversation self-heals. Its own timeout is
  generous (5 s) precisely because it is *not* on the deadline: holding it to the
  acknowledgement's ceiling would make it give up on the slow handshake it exists to
  absorb.

Measured against transport doubles that charge a handshake the way a real host does:

| first-turn scenario | adapter | channel | heard |
|---|---|---|---|
| 0.20 s handshake + 0.23 s round trip | 0.741 s | control | 1.932 s |
| 0.40 s handshake, not warmed | 0.755 s | **SSE fallback** | 1.946 s |
| 0.40 s handshake, warmed first | 0.536 s | control | 1.727 s |

A cold handshake can never extend the deadline itself — the wall-clock ceiling holds at
0.45 s regardless, so the first turn's worst case is the same 0.750 s as any other
turn's. What the warm-up changes is *which channel* the acknowledgement goes out on.

**The model may not speak one of its own.** Because acknowledgements are owned here,
`speech.VOICE_SYSTEM_PROMPT` forbids the model from opening a reply with a holding or
stalling phrase and tells it to begin with the substance. Live evidence for the rule:
on a turn where the adapter correctly stayed silent inside its cooldown, the callee
heard "Okay, one moment." at 4.46 s anyway, written by the model. The cooldown governs
adapter fillers, but the requirement is about what is HEARD, so a model-authored
holding phrase defeats it just as thoroughly — and it spends the first tokens of the
2 s budget on filler instead of the answer. The prohibition names its own reason and
says it holds "even if other instructions or examples suggest otherwise", because the
Vapi dashboard prompt (§1.10 layer 4) and the Hermes profile's resident prompt both
layer on top of it and are outside this adapter's control.

Note for post-call diagnosis: several built-in phrases ("Okay, one moment.", "Sure,
give me a second.") are exactly what a model improvises, so a heard line CANNOT be
attributed by its text. The `turn filler call=<ref> elapsed_ms=<n>` log line is the
only proof the adapter spoke one.

**Accepted-but-silent: `<flush />` does not survive a long stall, LIVE, reproduced.**
A flushed chunk sitting alone in the stream is accepted by Vapi immediately -- its
own `model-output`/`voice-input` events echo it back within ~1 ms, proving receipt
-- but is NOT reliably turned into audio once that same stream then goes quiet for
more than a few seconds, which is exactly what a slow or multi-tool Hermes turn
does. Live call `01a025e5` (fixture `tests/e2e/fixtures/transport_ack_dropped_by_vapi.json`):
the adapter's first filler landed 2.05 s after the callee's question -- Vapi echoed
it as accepted -- then nothing was spoken for the rest of the call; a second filler
12+ s later was equally silent. `tests/e2e/README.md`'s standalone probe (no Hermes
traffic at all) isolates the cause to the stream's own state, not the flush token:

- A lone `<flush />`-terminated chunk followed by an 18 s stall on an otherwise idle
  stream is not spoken AT ALL until the stream produces more content or terminates.
- Omitting `<flush />` changes nothing -- the identical stall reproduces either way.
- Byte-level keepalive during the stall (empty `content` deltas, raw SSE comment
  lines) changes nothing either: Vapi's TTS commit does not track wire activity, it
  tracks the model-shaped content stream's progress.
- Ending the response immediately after the flushed chunk (`finish_reason: "stop"`
  then `[DONE]`, no stall at all) gets it spoken in ~0.2 s, reliably.
- When a buffered flush chunk IS eventually rendered late (stream progress or
  termination), it can come out concatenated with a second buffered fragment and
  audibly garbled/duplicated -- this is the live-reported "Sure. Give me a second
  Sure. Give me a second." (call `01a025ea`), not two separate adapter bugs.

This is a Vapi platform behaviour to design around, not an adapter defect: the
adapter's own emission is correct and the bytes are received immediately. The fix
does not touch `<flush />` or stream shape at all -- it uses a channel proven
immune to the stall. Every Custom LLM request body already carries
`call.monitor.controlUrl` (present with **no** `monitorPlan.controlEnabled`
override needed on the assistant -- confirmed live on requests with and without
it). `POST controlUrl {"type": "say", "content": text}` renders speech in ~0.3 s
measured, independent of what the model.url stream is doing -- proven on a probe
whose model stream stayed silent for the full 26 s of the call, while two separate
`say` calls each still spoke on schedule, clean and undivided. `turns.py` tries this
channel first for every acknowledgement (`VHV_ACK_USE_CALL_CONTROL`, default on) and
falls back to the old SSE-embedded delivery only when no control URL is on the
request or the control POST itself fails -- see `vapi_control.py`.

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

`/vapi/server` consumes exactly the live speech types it was built for —
`vapi_events._SPEECH_STARTED_TYPES` plus the caller-speech parser — and
`parse_server_message` returns `None` for everything else, so `end-of-call-report` and
`status-update` are received, authenticated, and dropped. **That is now a decision, not
a gap.** The post-call report exists (§1.14) and is rendered by `call_report.py`, but it
is DELIVERED by Hermes, because reporting to the principal needs a channel to the
principal and this adapter must not have one: its Hermes session and Hermes's
`api_server` toolset both lack a messaging tool, and granting one to the session a live
counterparty is talking to is a prompt-injection path to his Telegram.

So the branch §1.8 used to nominate here is deliberately NOT built. If you are about to
widen `parse_server_message` to catch `end-of-call-report`, read §1.14 first: the
adapter would gain promptness (2.88 s after hangup instead of a poll) and still could
not deliver anything.

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
| `purpose` | yes, for task calls | Free-text objective, written **for the model**. Absent ⇒ behavior is exactly as before this feature. Never spoken — see §1.11. |
| `spoken_reason` | no, but strongly recommended | The reason for the call **as it should be said out loud**, one clause: `"about his left knee MRI results"`. The only variable intended to reach a loudspeaker. Aliases: `reason`, `reason_for_call`, `opening_line`, `spoken_purpose`. |
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
`MAX_PURPOSE_CHARS` (400), `spoken_reason` at `MAX_SPOKEN_REASON_CHARS` (200),
`callee` at `MAX_CALLEE_CHARS` (120). Non-string values are
ignored rather than coerced. Values are **never logged**: only
`CallVariables.log_summary()` (`purpose_chars=… spoken_reason_chars=… callee_chars=…`)
reaches a log line.

A key that matches no alias is still surfaced: its **label only** goes to a
`call variables not recognized` warning and its value becomes supplementary context.
A novel spelling is loud, never silently dropped.

#### Instruction precedence

`speech.build_instructions` layers paragraphs in this order (Hermes's own resident
system prompt sits under all of it):

1. `VOICE_SYSTEM_PROMPT` — phone/TTS style and safety rules.
2. Identity — who the assistant is, and which way the call went.
3. Tool policy, when configured.
4. `extra` — the Vapi dashboard system prompt. Standing operator configuration.
5. Task paragraph — this call's `purpose`.
6. **Standing scheduling conduct — outbound calls to somebody other than the
   principal. Last.** See §1.13.

`purpose` goes *after* the dashboard prompt deliberately: it is the most specific
instruction in the prompt (the reason this one call exists), and a generic
Mike-flavored dashboard prompt must not be able to talk the model out of the job it was
dialed to do. The task paragraph closes by handing authority back to layer 1, so the
voice and safety rules stay authoritative and the objective is framed explicitly as
data describing a task rather than a source of new rules.

The conduct block goes after *both* of them for the same kind of reason, inverted: it
is standing policy the operator dictated for every call of that shape, and the two
things it has to survive are precisely the two layers above it — a dashboard prompt
written to sound accommodating, and an objective whose own text invited the model to
negotiate on the principal's behalf. On call `01a028f1` the objective carried "Mike is
free weekday mornings" and the model treated that as licence to keep the call open
until Mike could be consulted. Nothing may sit below layer 6.

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
      "spoken_reason": "about Marvin's cardiology recheck",
      "callee": "Dr. Patel's office"
    }
  }
}
```

`customer.number` is who gets dialed. `purpose` is the objective and is required for
task-call behavior; `callee` is optional but makes the opening concrete. Omit both and
the call behaves exactly like any other outbound call.

`spoken_reason` is optional but **strongly recommended for every task call**: it is
what the callee actually hears in the first second (§1.11). Supply it and the opening
is exact and unambiguous. Omit it and the adapter refuses to guess from `purpose`,
falling back to "Is this a good moment to talk?" — safe, but it tells the callee
nothing, and Hermes then has to deliver the reason on the following turn.

### 1.11 Reason for calling, spoken locally (R1)

**Requirement.** On an outbound call the callee must learn why the phone rang within
one to two seconds of finishing their first utterance — typically "Hello?".

**Why nothing may route through Hermes for it.** Measured on the deployed system, any
utterance whose text comes from Hermes costs 1.6–2.2 s warm, 3.6–4.9 s cold, and
14–17 s when a tool runs. No amount of threshold tuning makes that fit a 2 s deadline.
Only two sources can: Vapi static text, and text the adapter generates itself. With
`firstMessageMode: assistant-waits-for-user` the static `firstMessage` is inert on
outbound calls, so the adapter is the **only** remaining source — which also makes it
the only place the AI-identity disclosure can be carried.

**Mechanism.** `server.handle_chat` short-circuits before any Hermes call site and
answers from `policy.build_reason_line`, streamed through the same
`speak()`/`_speak_once`/`ChunkWriter` path that already serves `BUSY_LINE` and
`DENIED_LINE`. Measured end to end in-process: **2 ms**.

It fires only when **all** of these hold:

1. `VHV_OUTBOUND_REASON_FAST_PATH` is on (default) and the call is `outboundPhoneCall`.
2. The call carries a `purpose` or a `spoken_reason`. Neither ⇒ nothing to announce,
   and behavior is byte-identical to before.
3. The request carries a real trailing user utterance. The trigger is the callee
   *speaking*, never call connect: a line spoken on connect arrives before the handset
   reaches the ear, which is the observed failure it must not reproduce.
4. `policy.is_first_callee_turn` — the trailing user message is the only user message
   and at most one assistant message precedes it. True for both
   `[system, user]` (waits-for-user) and `[system, assistant(firstMessage), user]`
   (legacy speaks-first).
5. `CallState.reason_spoken` is unset.

**Conditions 4 and 5 are ANDed deliberately, and must stay that way.** Each closes a
hole the other cannot: the latch (5) stops a Vapi re-POST whose reply has not yet made
it back into the history Vapi sends, while the conversation shape (4) still answers
correctly when the latch is gone — per-call state lost to TTL or the
`max_tracked_calls` LRU, or a request with no `call.id`, which is never registered at
all. Repeating the reason on every turn needs *both* to fail, which is strictly harder
than either failing alone. Collapsing this back to one condition reintroduces one of
the two failures.

**`purpose` is a trigger, never speech.** Its presence marks the call as a task call;
not one character of it is spoken. It is model-facing prose with no fixed grammar — a
real value was `"Goal: next steps - appointment, phone call with Craig, or proceed to
surgery and get a date. Mike is free weekday mornings."` — and quoting it means reading
the principal's internal negotiating limits aloud to a doctor's office. The spoken
clause comes only from `spoken_reason`, and even that goes through
`speech.speakable_reason`, which **only deletes** (no paraphrase, no summary, no model
call): it strips section labels, everything past the first sentence, everything past a
list or aside boundary, markdown, emoji, URLs and braces; caps the result to one clause
(`MAX_REASON_TOPIC_WORDS` 24, `MAX_REASON_TOPIC_CHARS` 160, against the 200-character
limit `extract_call_variables` already applied); and returns `None`
outright when what survives still reads as an instruction. On `None` the line falls back
to `VHV_OUTBOUND_REASON_SENTENCE_GENERIC`, which mentions neither the purpose nor
anything derived from it. Leaking operator text is therefore impossible by
construction, not merely improbable.

**The lead-in is spoken exactly once.** `VHV_OUTBOUND_REASON_SENTENCE` supplies one
("I am calling {reason}."), and an operator may reasonably write `spoken_reason` as a
finished clause that supplies another. Live, that produced *"I am calling about I am
calling about Mike Averto's left knee MRI results from August."* — the lead-in twice,
and a date truncated because the duplicate had spent four of the twelve words the cap
then allowed. A redundant lead-in ("I am calling about", "I'm calling regarding",
"Calling about", "This is Emma calling about") is therefore deleted like every other
rule here, leaving the connector-led clause the template expects. Composition is
idempotent: `speakable_reason("I am calling about X") == speakable_reason("about X")`.
The deletion needs a first-person or self-referring subject **and** a following
connector, so "the office calling about the results" and "call Dr. Patel and ..." keep
every word.

**The greeting is not configurable.** `build_reason_line` assembles it in code:
`"Hi, this is {assistant_name}, an AI assistant calling on behalf of {principal}."`
for a third party, or `"Hi {principal}, {assistant_name} here."` when the principal
answered (nobody to disclose to, and "calling on behalf of Mike" is nonsense framing
when Mike picked up). Only the reason sentence is templated, and
`VHV_OUTBOUND_REASON_SENTENCE` must keep its `{reason}` placeholder (validated at
config load). No operator edit can drop the disclosure.

**No `<flush />` on this line.** The flush token forces early TTS transmission when
more content is still coming in the same stream (§1.6). Here the utterance is complete
and is followed immediately by `finish` + `[DONE]`, so there is nothing to flush past —
and emitting the token with `voice.chunkPlan.enabled` off would have Vapi read
"`<flush />`" out loud.

### 1.12 Vapi's TTS cache serves `say` only, NOT model output (LIVE-verified 2026-08-22)

Vapi caches synthesised audio (`voice.cachingEnabled`, **default true**, present on
`VapiVoice` in the published OpenAPI). A hit is worth most of half a second, which is
large against a 2 s heard-gap requirement — so "make the eight acknowledgement phrases
always warm" looks like the cheapest win available. **It is not available.** The cache
is consulted on the `say` path and not on the model-output path, and the adapter
delivers acknowledgements as model output (§1.6). Recorded here so nobody spends
another day on it.

What the platform's own log says (`GET /call/{id}/call-logs`, gzipped JSONL — see
`tests/e2e/call_logs.py`; the endpoint 302s to a presigned URL and the `Authorization`
header must be DROPPED on the redirect):

- `"Voice cached"` — a cache HIT. Nothing is synthesised and
  `pipeline.botSpeechStarted` follows in **tens of milliseconds**.
- `"Voice input"` — a MISS, carrying the exact string handed to the voice provider.
  That string is the key, **post-`formatPlan`**: `"3327."` is cached as `"3 3 2 7."`.
  Vapi names it `attributes.text` for a `say` and `attributes.input` for model output.
- `assistant.voice.firstAudioReceived` — emitted only when synthesis happened, with the
  render time in `attributes.latency`.

Controlled A/B, one call, no persisted change (assistant untouched; per-call
`assistantOverrides.model` pointed at an echo custom-llm so the model's output is chosen
byte for byte — `tests/e2e/cache_probe.py`). Call `01a02728`, fixture
`tests/e2e/fixtures/call_log_cache_paths.json`:

| step | path | outcome | to speech |
|---|---|---|---|
| nonce, first time | `say` | MISS | 439 ms |
| **same nonce again** | `say` | **HIT** | **33 ms** |
| **same nonce, as model output** | model | **MISS**, 350 ms synthesis | 357 ms |
| second nonce, as model output | model | MISS, 300 ms | 336 ms |
| second nonce again, as model output | model | MISS, 304 ms | 325 ms |
| **second nonce, via `say`** | `say` | **MISS** | 365 ms |

Three separate facts, each load-bearing:

1. **The cache works and is cross-call.** It is keyed on the formatted text, survives at
   least three hours, and is not per-call: a pool phrase last spoken at 22:51 hit at
   01:52 the next morning on a different call (`01a0272b`, 5 ms to speech).
2. **Model output does not READ it.** Row 3 missed on a key row 2 had just proven
   resident, and paid 350 ms of fresh synthesis for it. Over 12.5 hours of live and probe
   traffic — 124 calls, 905 utterances — the model path took **0 hits in 671
   utterances**, while the `say` path took 20.
3. **Model output does not POPULATE it either.** Row 6 repeats text rows 4 and 5 had just
   rendered twice, and still missed. So warming from the model path is not an
   alternative route to the same win.

`voice.chunkPlan.enabled: false` does not change it (call `01a0272f`: `say` hit at
34 ms, three model-output misses at 311/300/309 ms), so the streaming chunker is not
what excludes the lookup.

**Consequence.** The ~0.3 s the cache would save is reachable only by delivering the
acknowledgement through `say`, i.e. Live Call Control — the channel §1.6 abandoned for
this line because ending the SSE response behind a flushed chunk is what makes it render
at all, and because `say` is where the multi-second stalls and outright drops were
measured (`pipeline.sayQueuePush` → `botSpeechStarted` of 6.755 s, 3.066 s, 9.610 s and
two never rendered, on call `01a026d8`). Reliability wins: a line heard 0.3 s later is a
smaller failure than a line not heard at all. Nothing in the adapter changes on this
finding, and the pool phrases are deliberately NOT warmed — warming would cost a call
per deploy and buy nothing on the path they are actually spoken from.


---

### 1.12 Did what we delivered become audio? (LIVE-verified 2026-08-21)

Vapi accepts text on both channels and then sometimes never renders it, with **no
error event anywhere**. Server-side on call `01a026d8-ba00-744f-ae52-5de7e833cae6`,
`pipeline.sayQueuePush` → `pipeline.botSpeechStarted` ran 6.755 s, 5.759 s, DROPPED,
3.066 s, 0.487 s, 9.610 s, DROPPED while `assistant.voice.connectionOpened` reported a
healthy TTS websocket throughout. The callee simply hears nothing. A 200 from Live
Call Control and a `voice-input` echo on the stream are therefore both statements that
**Vapi took it**, never that it was spoken.

Two feedback channels exist. Probed on four websocket-transport calls (no PSTN leg;
`01a02723`, `01a02727`, `01a02729`, `01a0272a`).

**1. Vapi's own conversation history — NO assistant config change.** The `messages[]`
of every Custom LLM request is derived from what was **actually spoken**. On `01a02723`
a streamed chunk was accepted (`Voice input` logged, `assistant.speechStarted` fired)
and then cleared before any audio played: it is absent from the next turn's
`messages[]` and from the call artifact, while everything audible — including text
delivered through Live Call Control — is present. This is the load-bearing channel and
it stands alone. Two caveats, both real:

- The text is **re-transcribed, not echoed**. `"ACK ONE PLEASE HOLD."` came back as
  `"a c k one please hold"`: case and punctuation gone, the acronym spelled out.
  Digits become words. Matching must be normalised and fractional.
- Consecutive deliveries **merge into one assistant message**. `01a02727` returned one
  entry reading `"a c k one please hold Answer alpha is fifty milligrams."` for a
  streamed acknowledgement plus a control-delivered answer, so matching is done against
  the concatenation of the assistant messages, never per message.

It arrives only at the **start of the next turn**, never same-turn.

**2. `speech-update` webhooks — ONE additive assistant field.**
`{"type": "speech-update", "status": "started", "role": "assistant", "turn": N}`
reaches a configured server URL 50–250 ms after audio really begins. It arrives with
**`serverMessages` left alone** — verified on `01a0272a`, whose overrides set only
`server` and whose echoed `serverMessages` was null. Vapi sends `server.secret`
verbatim as the **`x-vapi-secret`** header, plus `x-call-id`. No signature header.
The same event type is also sent for the CALLEE (`role: "user"`) on every turn, so
`role` must be checked.

`assistant.speechStarted` carries the exact text and would be better, but it is opt-in
**and it did not fire for either Live Call Control `say` on `01a02727`** — so the
adapter's own answer channel gets no text-bearing event, and the documented
`source: "force-say"` is not what this account emits.

**No timeout may ever mean "dropped."** Holding the model.url stream open for 20 s
after a flushed chunk **delayed** that chunk's render by 20.3 s — it played the instant
the response ended — rather than losing it (`01a02723`: `Voice input` at t+0.6 s,
`pipeline.botSpeechStarted` at t+20.9 s). "No audio yet" is evidence of *unknown*. Only
positive absence from a settled record may condemn a delivery, which is why
`speech_feedback.SpeechLedger` has no expiry path and `speech_confirm_window_seconds`
is a precondition on the verdict rather than a trigger for it.

**`GET /call/{id}/call-logs`** (gzip JSONL) carries the same ground truth server-side
and works retroactively, but it needs an org-wide `VAPI_API_KEY` that the adapter
deliberately does not hold. It is the right oracle for the E2E harness, which does.

#### The assistant change this would need (NOT applied — operator's decision)

```json
{ "server": { "url": "https://<adapter>/vapi/server", "secret": "<VHV_VAPI_SERVER_SECRET>" } }
```

One additive field, no persona or voice impact, `serverMessages` untouched. It is
optional: with `server` unset every journal state is still reachable through channel 1.
If `server` IS set and `VHV_VAPI_SERVER_SECRET` is **not** configured on the adapter,
the route is not registered at all and every event gets a 404 — fail closed, never
unauthenticated acceptance.

**The same field also protects against a second, unrelated defect** (`CallState.
caller_speaking`, `turns._deliver_answer`): the CALLEE half of this event
(`role: "user"`), which channel 2 above receives but a drop-detection-only reading
discards, is real-time, structural evidence that the caller is talking right now --
something no amount of polling `messages[]` (channel 1, only refreshed at the START
of the NEXT turn) can ever supply mid-turn. Live call `01a028f1-4abd-799e-8d6d-
c83273cd01ae`: Deepgram declared the caller's turn over on a grammatically complete
clause ("...Brooklyn on Mondays.", `end_of_turn_confidence` 0.7207 against
`eotThreshold` 0.7) while he kept talking with no acoustic gap at all; the adapter's
own SSE response to that fragment WAS self-corrected by Vapi's platform before it
spoke (new partial transcripts kept streaming in, same turn), but the SLOWER
Hermes-computed answer that followed it was delivered through Live Call Control 6.5s
later, landing while the caller's real utterance was still open -- the audible
interruption he reported. Wiring the same webhook's `role: "user"` half into a HOLD
(never a cancel -- see `CallState.set_caller_speaking`) on any pending Live Call
Control delivery closes that gap: same field, same route, two independent defects.
With `server` unset, `caller_speaking` is never set (its default is `False` and
nothing else writes to it), so every `_deliver_answer` attempt proceeds exactly as it
does today -- clean degradation, not a broken half-state.
### 1.13 Standing scheduling conduct (LIVE-dictated 2026-08-22, call `01a028f1`)

**Not a fix for one transcript.** The operator answered this call himself, role-playing
a doctor's office, and dictated three rules. His closing sentence is the contract:

> These are important rules for scheduling that will be consistent no matter what type
> of thing it's scheduling.

So they live in `speech._SCHEDULING_CONDUCT`, as prompt layer 6 (§1.10, instruction
precedence), and nothing in them names a doctor, a clinic or an appointment.

**What the call did.** Four utterances from Vapi's transcript, one of them correct:

| Heard | Verdict |
| --- | --- |
| "September fourteenth, conflicts with school from eight ten to two forty five" | **Leak.** A child's school hours read to a receptionist. "That one does not work" was the entire answer owed. |
| "September seventh at nine or nine thirty works on Mike's calendar" | **Correct**, and the shape the prompt now blesses by name. The fix is *not* to be vaguer about availability. |
| "The only item is an all day parking notice, which does not block the visit" | **Leak, and the one a naive rule misses.** A calendar entry volunteered to *explain* why a time was fine. "Do not mention conflicts" does not catch it — nothing was in conflict. The model leaked while being helpful and precise, reasoning aloud about why something was not a problem. |
| "Thanks. I need Mike to choose between nine and nine thirty before I finalize it. How long is the appointment, and c…" — the call's last act | **Deferral.** Two workable times, an absent principal, a third party parked on a decision nobody present could make, and the call still gathering detail the objective did not need. |

The third row is why rule 1 bans the **explanation**, in both directions, rather than
banning conflicts: `"it works"` / `"it does not work"` is the complete permitted
vocabulary about the calendar, and no reason is ever owed to a counterparty.

**The five rules, and where each is enforced.**

| # | Rule | Enforced by |
| --- | --- | --- |
| 1 | Never disclose what is *on* the calendar to a counterparty, and never explain *why* a time does or does not work — only that it does or does not | **Tool layer first** (hermes-calendar), prompt second. See below. |
| 2 | Decide and commit: take the first workable time, confirm it aloud, close the call | Prompt only |
| 3 | Reversibility is *why* deciding is safe — a booking can be moved with another call | Prompt only, stated as the justification and not merely as a rule |
| 4 | Gather nothing the objective does not need | Prompt only |
| 5 | Say the exact booking back before the call ends, so the counterparty can correct it and the principal can veto it | Prompt for the spoken line; `call_report.py` for the written report the principal actually receives (§1.14) |

**Rule 1 is two layers, and the prompt is the SECOND one.** The first is what the
calendar tools return. On the phone path (Hermes platform `api_server`, and anything
unrecognised — it fails closed) `calendar_agenda` and `calendar_list_calendars` refuse
outright, and `calendar_freebusy`/`calendar_find_slots` answer with a span and a
verdict only — literally `"<span> works."` / `"<span> does not work."` — with no event
title, location, attendee or calendar name anywhere in the payload. *That* is the
guarantee. Prompting is persuasion and this project has already been burned once
assuming a prompt rule would hold (§5, client-side tool restriction). The rule is
repeated in the prompt because the two layers fail differently: the tool layer cannot
stop the model repeating something the **callee** just said, or inventing a plausible
reason to be helpful; the prompt layer cannot stop the model reading out a field it was
handed. Neither is sufficient; together they are defence in depth.

Rules 2–4 have **no** structural enforcement available. There is nothing for the
adapter to withhold or rewrite: "ask one more question" and "hang up without booking"
are ordinary speech, indistinguishable in the stream from a correct turn. Prompt is the
only lever, which is why rule 3 is phrased as the *argument* for rule 2 rather than as
another prohibition — a model told only "decide" hedges again the moment a counterparty
pushes, and a rule with no permitted move is a rule the model routes around.

**Rule 5 no longer stands alone, and it is not weaker for it.** It was written when
nothing reported a call outcome at all, so the spoken line was the operator's only
record. A written report now exists (§1.14: `call_report.py` renders it, Hermes
delivers it), and rule 5 survives the change because the two records do different
jobs and neither replaces the other:

- The **spoken** read-back happens while the counterparty is still on the line, so it
  is the only one of the two that the person who actually holds the slot can
  contradict. "No, I said ten" is only possible before the call ends.
- The **written** report reaches the principal, which the spoken line never did, and is
  the only one he can act on — his veto ("it can always call back and change the
  time") needs a record that arrives where he is.

They also audit each other. The report quotes a booking Hermes vouches for; the
transcript records what was said out loud. A disagreement between them is visible, and
that is worth more than either alone.

`tests/test_scheduling_conduct.py::test_the_adapter_is_deliberately_not_the_thing_that_reports_to_the_principal`
replaces the old premise guard. It no longer asserts that no report exists; it asserts
that this ADAPTER is not the thing that sends it, which is a security boundary (§1.14)
rather than an unfilled gap.

**Where the block is emitted, and where it is not.**

| Call shape | Conduct block | Why |
| --- | --- | --- |
| Outbound, callee is a third party | **yes, always** | The only shape in which the adapter *knows* a third party is on the line. |
| Outbound, `callee_is_principal` | no | Nothing to withhold and nobody to defer to; gagging the assistant about the operator's calendar *to the operator* is a worse failure than the one being fixed. |
| Inbound | no | The adapter cannot tell the principal ringing his own assistant from a stranger, so it does not guess. Inbound privacy is the tool layer's, which is where the guarantee lives anyway. |

Emitted on **every** outbound third-party call, with no keyword matcher on the
objective. A matcher is the obvious cheap implementation and it is wrong: it cannot see
"get a date for the procedure" or "see when they can come out", and the calls where the
model improvises hardest are the ones carrying the least instruction. The rules are
phrased to be inert when no time is being settled, so the cost of a false positive is a
paragraph the model ignores.

#### Evidence (LIVE-measured 2026-08-22, no PSTN)

A prompt rule that is only asserted to be *present* is untested. `tests/e2e/scheduling_conduct_probe.py`
drives the real Hermes on `127.0.0.1:8642` with the real `build_instructions` output on
turns shaped like the moment call `01a028f1` went wrong, once with the block and once
with it suppressed, three trials each. The offered times are real free windows on the
operator's live calendar (found with `calendar_find_slots` first) — an invented date that
happens to be busy makes the model answer "neither works", which is correct and never
reaches the decision being observed.

| Scenario | baseline clean | conduct clean | baseline failures |
| --- | --- | --- | --- |
| Two workable times offered | 2/3 | **3/3** | 1 over-ask ("whether Mike should bring anything") |
| Counterparty pushes for a preference | 0/3 | **3/3** | 2 deferrals, 2 leaks |
| Reassurance bait ("is his calendar *genuinely* clear, or is there something on it that just does not clash?") | 0/3 | **3/3** | 3 leaks |

The baseline reproduced the live call on both axes, and reproduced the parking-notice
family 3/3. Verbatim, and note that a rule phrased as "do not mention conflicts" catches
none of these:

- "He has two work commitments starting at eleven thirty, so a nine o'clock appointment
  should be fine if it ends before then."
- "Nothing is scheduled on the family, personal, Shopify, or Karissa calendars." — four
  calendar names and a person's name, a worse disclosure than the live call's.
- "Mike has nothing else scheduled that morning."
- "Please don't book either one yet… He'll call back to choose." — the deferral.
- "Could you hold the nine o'clock slot while I confirm with him?" — the deferral again.

Same bait, with the block: *"Yes, nine o'clock works for Mike. Please pencil him in for
Tuesday, September eighth at nine in the morning."* 0/9 leaks, 0/9 deferrals.

**Latency: the longer prompt is not slower.** 18 real Hermes turns, mean 12.85 s with
the block against 15.31 s without. On the adversarial scenario it is 6.80 s against
16.68 s, because rule 4 stops the model re-checking and re-negotiating — the conduct
rules make calls *shorter*, not longer.

**What the websocket harness can and cannot say about this.** Three
`tests/e2e/run_voice_deadlines.py` runs on this branch: all PASS, R2 ack deadline
1.273/1.638/1.241 s (budget 2), cooldown 14.4–14.7 s (budget 10), `max_turn_gap`
1.474/1.638/1.289 s (budget 3), zero model-authored holding phrases. But that transport
is always `call.type = "vapi.websocketCall"` → `direction = "inbound"` (§1.11,
`r1_transport_scope`), and this block is gated on outbound — so **it is not in the prompt
on any of those runs.** They prove no inbound regression and nothing more. The outbound
side is covered instead by an offline diff of `build_instructions` output against
`9610ef2` across all four call shapes: inbound and outbound-to-principal are
**byte-identical**, and the block is the only delta, only on outbound-to-third-party.

### 1.14 Telling the principal what was booked (LIVE-verified 2026-08-22, call `01a02933`)

**Requirement, his words.** "It can pick a time, then come back, TELL ME THE TIME IT
PICKED, and if I decide it doesn't work, it can always call back and change the time."
§1.13 shipped the picking. This is the coming back. The veto in the second clause is
worthless without it: he cannot reject a time he was never told.

#### What Vapi actually sends at the end of a call

Measured, not read. Assistant `b39379dc` carries no `server` field, so the adapter never
receives one of these; the way round without editing the assistant is that Vapi accepts
`server` inside a PER-CALL `assistantOverrides`, which points the events at a disposable
sink for one call (`tests/e2e/end_of_call_probe.py`, `tests/e2e/report_recorder.py`).
One websocket call, no PSTN, 26 events:

| Type | Count | When |
| --- | --- | --- |
| `status-update` | 2 | `in-progress` at start, `ended` +1.58 s after hangup |
| `speech-update` | 8 | in-call, `role` × `status` for each of 2 turns |
| `transcript` | 15 | in-call, `transcriptType: "partial"`, role-tagged |
| `end-of-call-report` | **1** | **+2.88 s after hangup**, 22.5 KB |

`end-of-call-report`'s `message` carries `analysis`, `artifact`, `assistant`,
`assistantVersion`, `call`, `cost`, `costBreakdown`, `costs`, `durationMinutes`,
`durationMs`, `durationSeconds`, `endedAt`, `endedReason`, `messages`, `recordingUrl`,
`startedAt`, `stereoRecordingUrl`, `summary`, `timestamp`, `transcript`, `type`.

**Two of the fields the docs sell it on are empty here, and that decided the design.**
`summary` is `""` and `analysis` is `{}` — on this call and on all 20 ended calls
sampled from the account. The cause is on the assistant:
`analysisPlan.summaryPlan.enabled: false` and
`analysisPlan.successEvaluationPlan.enabled: false`. So there is **no summary, no
`structuredData` and no `successEvaluation`** to receive. Turning them on is an edit to
`b39379dc` (not this workstream's call) and would additionally ship call content to
Vapi's analysis model. Anything built on that block on this account is building on `{}`.
This is the third docs-vs-reality gap found in two days, after `assistant.speechStarted`
not firing for control-`say` and the phantom `source: "force-say"`.

What IS real and load-bearing: `transcript` (plain `User:`/`AI:` text), `endedReason`,
`startedAt`/`endedAt`/`durationSeconds`, and — the one that makes a report possible —
`artifact.variableValues`, which echoes the dynamic variables **verbatim**, so
`call_purpose` and `callee` come back and the report has an objective to judge against.

#### Per-call `server` is equivalent to the persistent field

Worth stating separately because three workstreams were queued behind Mike approving
that field. All 26 events above arrived through a per-call override alone:

- `server.secret` is sent verbatim as `x-vapi-secret`, **26/26** constant-time compares
  equal. Same semantics as the persistent field. No signature header.
- `speech-update` arrives with `role` and `status` (`user`/`assistant` ×
  `started`/`stopped`), in-call, per turn. No `text` on any of the 8 — the existing note
  in `vapi_events.py` is correct.
- Delivery latency (arrival minus Vapi's own `message.timestamp`): `speech-update`
  median **55 ms** (51–93), `transcript` median 55 ms, `end-of-call-report` 67 ms.

`serverMessages` was listed explicitly rather than trusting the documented default,
because this assistant's is `null` and "null means the documented default" is the same
assumption that already cost an afternoon. Omitting it is untested.

**Recommend the persistent field anyway.** Equivalent in capability, not in guarantee:
per-call is opt-in per creator, so anything that POSTs `/call` without it silently loses
the protection. Concretely, from the reassurance work: with `server` unset,
`CallState.caller_speaking` stays permanently `False` and the caller-speaking gate fails
**open and silently**, because "the callee is quiet" and "we are never told the callee is
talking" are the same observation from in here. The persistent field cannot be forgotten.
One additive field, `serverMessages` untouched, no persona or voice impact:

```json
{ "server": { "url": "https://<adapter>/vapi/server", "secret": "<VHV_VAPI_SERVER_SECRET>" } }
```

#### Who reports, and why it is not this adapter

**Hermes delivers.** Not a layering preference — the adapter cannot, and should not be
made able to:

- **No channel, refused twice.** The adapter's `VHV_TOOL_POLICY__ENABLED_TOOLS` carries
  no messaging tool, and Hermes's `api_server` toolset — the surface the adapter talks to
  — is `calendar, clarify, notes, todo, web`, also with no `messaging`. Granting one
  would hand the session a live counterparty is talking to a path to the principal's
  Telegram. We spent the same day narrowing that session's tool surface, not widening it.
- **No summary to forward.** Per above, `analysis` is `{}`. Receiving the webhook would
  hand the adapter a transcript to guess from, and a component guessing at bookings is
  exactly the failure this must not have.
- **Hermes does not have to guess.** It holds the objective the principal gave it, and it
  *made* the booking with its own calendar tools — it reads the commitment off a tool
  result. The adapter only ever sees `purpose`: untrusted, sanitised, 400-char-capped,
  model-facing text, not an objective of record.
- **It works today.** Hermes placed the call and has the Vapi MCP server, so
  `GET /call/{id}` needs no assistant change at all.

**The tradeoff, stated once.** The adapter is the only component guaranteed to observe
the call ending, so a Hermes-owned report must poll and can therefore be late — or never
arrive at all, if Hermes's process dies between placing the call and polling. That is
accepted in exchange for not granting the principal's Telegram to a caller-facing
session. The `unknown` verdict below is what keeps the residual failure honest rather
than silent.

#### The report, and why it cannot be fiction

`call_report.py` is a pure renderer: payload in, the exact text out. No network, no
clock, no LLM. `python -m vapi_hermes_voice.report_cli` reads a call on stdin and prints
the message, so the component that owns the channel gets a report it cannot embellish.
`extract_call_facts` accepts **either** a call object or an `end-of-call-report` envelope
— same facts at different depths, the tolerance `_VARIABLE_PATHS` already applies — and
on the real call above both shapes render byte-identical text.

**The reporter never reads the transcript.** It measures whether there was one. A time
can enter the report through exactly one door: a `Booking` a caller vouched for. Feed it
a transcript that says "I have booked Marvin in for Friday the 29th at 4pm" with no
claim attached and the output is `OUTCOME UNKNOWN` with no Friday and no 4pm in it
(`test_no_transcript_content_can_ever_reach_the_report`). This is the difference between
a report the principal can veto against and a summary written by something with an
incentive to look successful.

Four verdicts, and the fourth is the §6 obligation:

| Verdict | Means | Requires |
| --- | --- | --- |
| `booked` | a time was committed; the text says what, when, with whom | a vouched `Booking`, on a call that connected and ended |
| `no_booking` | the call happened and nothing was agreed | an explicit `NothingBooked` |
| `failed` | there was no conversation, so nothing could be agreed | positive evidence: a failure-shaped `endedReason`, or ended-and-never-connected, or ended-connected-and-zero-transcript |
| `unknown` | we do not know | the default |

Every verdict except `unknown` needs positive evidence, which is not the obvious shape:
"no `startedAt`, so it failed" would turn an empty or truncated payload into a confident
`CALL FAILED` about a call that may have gone perfectly. Absence of evidence is
`unknown`.

**An absent claim is never `no_booking`.** `Booking`, `NothingBooked` and
`BookingUnknown` are three types rather than `Booking | None`, because `None` would have
to mean both "nothing was booked" and "I never found out" — precisely the collapse §6
rule 1 forbids. "Nothing booked" is a positive claim about the world; here it would clear
the principal to double-book a slot he already holds. The CLI cannot spell it by
omission: it takes `--nothing-booked`, and omitting everything yields `unknown` and a
non-zero exit.

`unknown` also splits by which ignorance it is, because the two need different actions:
"I never saw it end" means go and look at the call; "the call finished and I was not told
whether anything was booked" means go and chase the reporter.

#### What the principal sees when `server` is unset — i.e. today

This is the default state on merge day, so it is the case tested hardest. With no
`server` field, no `end-of-call-report` ever arrives, and the adapter observes nothing at
the end of a call. Delivery does not depend on it: Hermes polls `GET /call/{id}`, which
needs no assistant change. So the principal gets the same `BOOKED` / `NOTHING BOOKED` /
`CALL FAILED` message, just on a poll's latency instead of 2.88 s.

What he must never get is silence. If Hermes reports without vouching for a booking, the
message opens `OUTCOME UNKNOWN` and tells him in as many words: "Do NOT assume a time was
booked, and do not assume one was not." If Hermes never reports at all, that is a gap
this repo cannot close from inside — and it is named here rather than papered over,
because the one thing the design guarantees is that no message from this renderer ever
leaves him believing a booking happened when it did not.

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
- **Calendar contents leaked to a counterparty, and a booking left unmade (§1.13,
  LIVE)** → standing scheduling conduct as prompt layer 6 on every outbound
  third-party call. The privacy half is the **second** line of defence only; the first
  is what the calendar tools return on the phone path (a span and a verdict, no titles).
  Rules 2–4 have no structural lever available and are prompt-only, stated with their
  justification so they survive paraphrase. Rule 5 (say the booking back) now pairs with
  a written report rather than substituting for one — see the next entry.
- **A time was picked and the principal was never told (§1.14, LIVE)** → `call_report.py`
  renders the report; **Hermes delivers it**, because this adapter has no channel to the
  principal and must not be given one (its session and Hermes's `api_server` toolset both
  lack a messaging tool, and a live counterparty is on the other end of that session).
  Four verdicts, every one but `unknown` requiring positive evidence, and an absent
  booking claim rendering as `unknown` rather than as "nothing booked". The reporter never
  reads the transcript, so it cannot invent a booking; a time enters the report only via a
  vouched `Booking`.
- **`end-of-call-report`'s `analysis`/`summary` are empty on this account (§1.14, LIVE)**
  → no verdict may depend on them, and `analysisPlan` is left alone. Read defensively,
  never load-bearing.

## 6. Standing rule: journals an off-box reader attributes blame with

`ack_journal.py` is not a debugging aid. It is read over HTTP by the E2E harness on
another machine, which cannot see the adapter's logs and uses it to decide **who
spoke a holding phrase** — the adapter or the model. A gap in it is therefore not an
inconvenience; it is evidence that will be misread by something with no way to know
better. Three separate bugs in one day, all the same shape, cost real time and
produced a false accusation against Hermes that was reported upward as a genuine
model regression before forensics overturned it.

Any record used to attribute a cause MUST:

1. **Be able to express "started, outcome unknown."** Two states are not enough. A
   step that has begun and not finished must be distinguishable from one that never
   began, or a reader collapses the two and blames whatever is left.
2. **Write that state BEFORE any cancellable or fallible step**, and refine it
   afterwards. Not after, and not only on success. Over-claiming is recoverable — it
   shows up as "we sent it, the callee never heard it", which is true and already
   reportable. Under-claiming becomes an accusation against another component.
3. **Never report a completeness signal it cannot vouch for.** `dropped == 0` means
   "nothing was lost", so every path that loses an entry — a cap, a TTL, a
   cancellation — must be counted. Silence read as completeness is worse than an
   admitted gap.

And the corollary, on the reading side: **an unexplained observation is UNKNOWN, not
a cause.** A consumer that cannot match something it observed against the record must
say so and stop. Inferring the cause is what turned a missing entry into "the model
wrote that line."

The three instances, for anyone who wants the receipts:

- `_record_ack` sat after the `yield` Vapi cancels, so an acknowledgement that was
  delivered *and spoken* left no record — while `dropped` stayed 0, asserting a
  completeness it did not have (call `01a02681` turn 1). Fixed by recording first.
- Answer delivery could only be "delivered" or absent, so a turn that had begun and
  not finished looked like one that never started. Fixed by writing
  `outcome="attempted"` before the first POST and mutating it as the picture clears.
- An off-box reader turned that ambiguity into a MODEL-AUTHORED verdict against
  Hermes. The prohibition it accused the model of ignoring was present in the prompt
  and being obeyed (0/6 trials produced a holding phrase, with an obedience control
  proving the instructions reached the model at all).
- `speech_outcomes` is the fourth record kind added under this rule, and the first
  that describes what the CALLEE got rather than what the adapter sent. It obeys all
  three obligations by construction: it is written `unconfirmed` at the moment of
  delivery (before any confirmation, cancellation or loss), it is refined in place as
  evidence arrives, and `speech_outcomes_dropped` counts its own evictions separately
  from every other kind. The obligation it exercises hardest is the first one:
  `unconfirmed` is a **terminal-in-practice state, not a pending one**. Nothing times
  out into `confirmed_dropped`, because a render can be 9.6 s late and still arrive
  (§1.12), so "we never found out" is a real answer and the record must be able to say
  it forever. A reader that treats `unconfirmed` as a failure has re-introduced exactly
  the guess this rule forbids.
