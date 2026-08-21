# Deployment

The adapter is a single stateless-per-turn ASGI process. A production deployment is:

```
Vapi cloud --HTTPS--> reverse proxy (TLS) --HTTP--> vapi-hermes-voice (loopback)
                                                         |
                                                         v
                                              Hermes API server (loopback)
```

## Checklist

1. **Bind loopback** (`VHV_LISTEN_HOST=127.0.0.1`, the default) and terminate TLS at
   a reverse proxy (Caddy, nginx, or a cloudflared tunnel). Vapi requires HTTPS.
2. **Set the three required secrets** in the environment or a root-owned `.env`:
   `VHV_HERMES_BASE_URL`, `VHV_HERMES_API_KEY`, `VHV_ADAPTER_API_KEY`. Optionally
   `VHV_ROUTE_SECRET` for a secret path prefix.
3. **Configure Vapi**: custom-llm model URL = your public base URL (plus
   `/v/<route-secret>` if set); custom-llm API-key credential = `VHV_ADAPTER_API_KEY`;
   `metadataSendMode` left at `"variable"`; `firstMessage` set;
   `voice.chunkPlan.enabled` left on (default) so `<flush />` fillers work.
4. **Harden Hermes** (the adapter cannot do this for you — see README "Configuring
   Hermes"): dedicated voice profile, dangerous toolsets disabled including
   `memory`/`session_search`.
5. **Route for latency**: set `VHV_VOICE_MODEL`/`VHV_VOICE_PROVIDER`/
   `VHV_VOICE_REASONING_EFFORT` (see README "Latency"), keep
   `VHV_WARMUP_ON_START=true`, and gate traffic on `/readyz`.
6. **Monitor**: `/healthz` for liveness, `/readyz` for Hermes reachability; logs are
   key=value on stderr with secrets and phone numbers redacted.

## cloudflared example

```sh
cloudflared tunnel --url http://127.0.0.1:8766
# paste https://<tunnel-host>            into the Vapi model URL (no /chat/completions)
# or    https://<tunnel-host>/v/<secret> when VHV_ROUTE_SECRET is set
```

## systemd example

```ini
[Unit]
Description=vapi-hermes-voice adapter
After=network-online.target

[Service]
User=voice
WorkingDirectory=/opt/vapi-hermes-voice
EnvironmentFile=/etc/vapi-hermes-voice.env
ExecStart=/opt/vapi-hermes-voice/.venv/bin/python -m vapi_hermes_voice
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

## Rotation

- **Adapter API key**: generate a new value, update the Vapi custom-llm credential,
  then restart the adapter with the new `VHV_ADAPTER_API_KEY`. (Do it in this order
  during a quiet window; there is no dual-key mode.)
- **Route secret**: update `VHV_ROUTE_SECRET`, restart, then PATCH the assistant's
  `model.url` to the new `/v/<secret>` prefix.
- **Hermes key**: rotate `API_SERVER_KEY` in the Hermes profile and
  `VHV_HERMES_API_KEY` together.
