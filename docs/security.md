# Threat model

Scope: the vapi-hermes-voice adapter and the trust boundaries it sits on. Wire-level
evidence for every "verified"/"measured" claim below is in
[`integration-contracts.md`](integration-contracts.md) (cited by section).

## Assets

- **The Hermes agent itself** — a high-privilege backend. A default Hermes install
  exposes `terminal`, `process`, `write_file`, `patch`, `execute_code`,
  `browser_exec`, `cronjob`, `delegate_task`, `memory`, and home-automation control
  to any prompt that reaches the API server (§2). Whoever can talk to it can
  potentially act as the operator.
- **Caller data** — phone numbers, spoken content, transcripts.
- **Operator data** — anything in Hermes sessions, long-term memory, and the
  operator's machine.
- **Secrets** — `VHV_HERMES_API_KEY`, `VHV_ADAPTER_API_KEY`, `VHV_ROUTE_SECRET`,
  the Vapi private API key.
- **Availability** — the phone line staying answerable.

## Trust boundaries

```
 UNTRUSTED                semi-trusted transport        enforcement point         high privilege
+-----------+  speech   +------------------------+   +---------------------+   +----------------+
|  Caller   | --------> |  Vapi cloud            |-->| vapi-hermes-voice   |-->|  Hermes Agent  |
| (anyone   |           |  STT text of whatever  |TLS|  bearer API key     |   |  tools, memory |
|  who dials|           |  the caller said;      |   |  (+ route secret)   |   |  operator env  |
|  the      |           |  presents the adapter  |   |  allowlist          |   |                |
|  number)  |           |  API key per request   |   |  sanitization       |   |                |
+-----------+           +------------------------+   |  session scoping    |   +----------------+
                                                     |  run lifecycle      |
                                                     +---------------------+
```

- **Caller speech is attacker-controlled input.** Everything in the `messages`
  transcript is untrusted.
- **Vapi is a semi-trusted transport**: it faithfully carries requests and — unlike
  Retell's Custom LLM WebSocket — authenticates itself with the configured
  custom-llm API key on every request (§1.4). The exact header shape is unverified
  (§4), so the adapter accepts `Bearer <key>` and the bare key, both constant-time.
- **The adapter is the enforcement point** for everything it can enforce; several
  critical controls (tool restriction, memory scoping) are only enforceable on the
  Hermes side and are called out as such below.
- **Hermes is a high-privilege backend** reached with a bearer key that grants full
  agent access.

## Threats and mitigations

| Threat | Mitigations | Residual risk |
|---|---|---|
| **Unauthenticated access to the endpoint** | Required `VHV_ADAPTER_API_KEY` (≥16 chars) checked with `secrets.compare_digest` before the body is read; optional `VHV_ROUTE_SECRET` path prefix; TLS termination at the proxy; adapter binds loopback | Key leakage (Vapi dashboard, operator env). Rotate by updating both sides |
| **Prompt injection via caller speech** — caller talks the agent into abusing tools or revealing data | Voice system prompt instructs plain prose, no tool/prompt/error disclosure; `ToolPolicy` is advisory prompt-shaping only — empty/unset `enabled_tools` says nothing about tools (defers entirely to the Hermes profile; an explicit `enabled_tools=none` tells the model to use none); **real control**: dedicated least-privilege Hermes profile with dangerous toolsets disabled via `hermes tools disable --platform api_server` | Prompt-level controls are model compliance, not enforcement (§2: 8/8 client-side restriction attempts silently ignored). The Hermes profile is the only actual boundary — misconfiguring it (not this adapter) is the real risk |
| **Prompt injection via tool output / retrieved content** | Instructions tell the model to treat tool output as data, not instructions; the adapter executes no tools itself | Hermes-side risk: the adapter cannot inspect or filter tool output. Documented, not solved here |
| **Data leakage in logs / spoken output** | `RedactionFilter` scrubs secret values and E.164 numbers from all logs; transcript content is never logged; Hermes fail-open error bodies (HTTP 200 + error prose + zero usage, §2) are intercepted and replaced with a generic apology | Operators who capture transcripts in their own tooling accept that exposure in their log pipeline |
| **Cross-caller data leakage via Hermes memory** — **VERIFIED in Hermes 0.20.4** (§2): a fact planted with "remember this" phrasing in session A was recalled verbatim in a fresh session B; session *keys* did not partition it | Per-call random `session_id`/`session_key` never derived from phone numbers; `session_retention="none"` default | **Headers do not contain this leak.** Operators MUST disable the `memory`/`session_search` toolsets in the voice profile, or explicitly accept that caller A's data can surface in caller B's call |
| **Runaway / abusive tool use** | Advisory budgets `max_tool_calls_per_turn` (3) and `max_tool_seconds_per_call` (60); turn timeout hard-stops the run | Budgets are advisory prompt-shaping. The hard control is the Hermes profile's toolset config. **Do not rely on Hermes approvals**: they are off by default, and a pending approval mid-call is silent dead air anyway |
| **DoS / abuse of the endpoint** | 401 before body read on bad auth; `max_body_bytes` enforced while streaming the body in; `max_concurrent_turns` cap (full → spoken busy line); history truncated at `max_history_messages`; per-call state bounded by TTL + LRU | Volumetric attacks are handled at the edge (reverse proxy), not by the adapter |
| **Caller spoofing / unwanted callers** | `VHV_ALLOWED_CALLERS` E.164 allowlist on `customer.number`; **fails closed** when caller identity is absent (web calls, `metadataSendMode: "off"`) | Caller-ID spoofing upstream of Vapi is out of scope; the allowlist is a policy filter, not authentication |
| **Orphaned Hermes runs** — abandoning the SSE stream leaves the run alive **unboundedly** (§2, measured) | `POST /v1/runs/{id}/stop` on every disconnect, cancellation, timeout, and shutdown — delivered on a background task so a cancelled request can never abort it; bounded by `hermes_stop_timeout`; lifespan drains pending stops before closing the client | None significant; `/stop` on an already-finished run 404s harmlessly |
| **Phone-number privacy** | E.164 numbers redacted in logs (`+1******1234`); call ids logged only as truncated SHA-256 refs; Hermes session ids/keys never derived from phone numbers; retention default `none` | `session_retention="hermes"` persists sessions under Hermes's own retention — an explicit operator choice |

## Deployment-level requirements

These are not adapter features; they are prerequisites for a safe deployment
(details in [`deployment.md`](deployment.md)):

1. TLS termination at a reverse proxy; the adapter binds loopback.
2. A strong `VHV_ADAPTER_API_KEY`, configured identically in the Vapi custom-llm
   credential; rotate by updating both sides.
3. A dedicated Hermes voice profile: own `API_SERVER_KEY`, dangerous toolsets
   disabled (including `memory`/`session_search`), approvals irrelevant because the
   toolsets are gone.
4. `VHV_ALLOWED_CALLERS` set whenever the caller population is known — and
   `metadataSendMode` left at `"variable"` so caller identity actually arrives.
5. Keep `voice.chunkPlan.enabled` (Vapi default) if fillers use `<flush />`.
