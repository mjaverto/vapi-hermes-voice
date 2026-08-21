"""Vapi control plane + the preflight that keeps an outage from looking like a regression.

Transport choice
----------------
Calls are placed with ``transport.provider = "vapi.websocket"``. No phone number is
involved, nothing rings, and the harness holds both ends of the audio: it decides
exactly when the callee stops talking, which is the zero point every deadline is
measured from.

The browser/Daily ``webCall`` route was tried first and is not usable here:

* ``POST /call`` with ``transport.provider="daily"`` is rejected outright ->
  ``400 "Couldn't Get Phone Number. Need Either phoneNumberId Or phoneNumber."``
* ``POST /call/web`` exists but answers ``401 "Invalid Key. Hot tip, you may be using
  the private key instead of the public key"``. It needs the org PUBLIC key, which is a
  separate credential from ``VAPI_API_KEY`` and is not present on this workstation.

Two operational details that are easy to lose and expensive to rediscover:

* Every request MUST carry a ``User-Agent``. Without one, Cloudflare in front of
  ``api.vapi.ai`` answers ``403`` with the body ``error code: 1010`` -- which reads
  exactly like an auth failure and is not one.
* ``VAPI_API_KEY`` is read from the environment and never logged. Nothing here prints
  a request header.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx

__all__ = [
    "ASSISTANT_ID",
    "PreflightFailure",
    "PreflightResult",
    "VapiClient",
    "VapiError",
    "load_api_key",
    "preflight",
]

API_BASE = "https://api.vapi.ai"
USER_AGENT = "vapi-hermes-voice-e2e/1.0"
ASSISTANT_ID = "b39379dc-ca93-48aa-a72a-a41d92279b4f"


class VapiError(RuntimeError):
    """A Vapi API call failed."""


class PreflightFailure(RuntimeError):
    """The system under test is not in a state where a measurement would mean anything."""


def load_api_key(env: dict[str, str] | None = None) -> str:
    """``VAPI_API_KEY`` from the environment. Never echoed, not even truncated."""
    source = env if env is not None else dict(os.environ)
    for name in ("VAPI_API_KEY", "vapi_api_key"):
        value = (source.get(name) or "").strip()
        if value:
            return value
    raise PreflightFailure(
        "VAPI_API_KEY is not set. Export it from your secret store, e.g.\n"
        "    export VAPI_API_KEY=$(grep -i '^vapi_api_key=' ~/.env | cut -d= -f2-)\n"
        "It is read from the environment only and is never written to the report."
    )


@dataclass(frozen=True, slots=True)
class PreflightResult:
    model_origin: str
    healthz: str
    readyz: str
    first_message: str | None
    first_message_mode: str | None
    transcriber: dict[str, Any]
    warnings: tuple[str, ...]


class VapiClient:
    """Thin, synchronous Vapi client. Only the five endpoints this harness needs."""

    def __init__(self, api_key: str, *, base_url: str = API_BASE, timeout: float = 30.0) -> None:
        self._http = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                # Mandatory: without it Cloudflare answers 403 / "error code: 1010".
                "User-Agent": USER_AGENT,
            },
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> VapiClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        try:
            r = self._http.request(method, path, json=body)
        except httpx.HTTPError as exc:
            raise VapiError(f"{method} {path} failed: {type(exc).__name__}: {exc}") from exc
        if r.status_code >= 400:
            raise VapiError(f"{method} {path} -> {r.status_code}: {r.text[:400]}")
        return r.json() if r.content else None

    def get_assistant(self, assistant_id: str) -> dict[str, Any]:
        return self._request("GET", f"/assistant/{assistant_id}")

    def create_websocket_call(
        self,
        assistant_id: str,
        *,
        name: str,
        sample_rate: int,
        assistant_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a websocket-transport call. No phone number, so nothing can ring."""
        body: dict[str, Any] = {
            "assistantId": assistant_id,
            "name": name,
            "transport": {
                "provider": "vapi.websocket",
                "audioFormat": {
                    "sampleRate": sample_rate,
                    "format": "pcm_s16le",
                    "container": "raw",
                },
            },
        }
        if assistant_overrides:
            body["assistantOverrides"] = assistant_overrides
        call = self._request("POST", "/call", body)
        url = ((call or {}).get("transport") or {}).get("websocketCallUrl")
        if not url:
            raise VapiError(f"call {(call or {}).get('id')} came back with no websocketCallUrl")
        return call

    def get_call(self, call_id: str) -> dict[str, Any]:
        return self._request("GET", f"/call/{call_id}")

    def end_call(self, call_id: str) -> None:
        """Best effort: a call left running bills until it times out."""
        try:
            self._request("DELETE", f"/call/{call_id}")
        except VapiError:
            pass

    def await_transcript(
        self, call_id: str, *, timeout_s: float = 90.0, poll_s: float = 3.0
    ) -> dict[str, Any]:
        """Poll until the call has ended and its ``messages[]`` has settled.

        ``messages`` is written asynchronously after the call ends, so reading it the
        instant the websocket closes yields a truncated timeline -- and a truncated
        timeline reads as "the assistant never replied", which is a false failure.
        """
        deadline = time.monotonic() + timeout_s
        last: dict[str, Any] = {}
        stable_at: float | None = None
        seen = -1
        while time.monotonic() < deadline:
            last = self.get_call(call_id)
            count = len(last.get("messages") or [])
            if last.get("status") == "ended":
                if count != seen:
                    seen, stable_at = count, time.monotonic()
                elif stable_at is not None and time.monotonic() - stable_at >= poll_s:
                    return last
            time.sleep(poll_s)
        return last


def _probe(client: httpx.Client, url: str) -> str:
    """One-line outcome for a health URL, distinguishing DNS from refusal from status."""
    try:
        r = client.get(url)
    except httpx.ConnectError as exc:
        detail = str(exc)
        if "nodename nor servname" in detail or "Name or service not known" in detail:
            return f"DNS-NXDOMAIN ({detail[:80]})"
        return f"CONNECT-REFUSED ({detail[:80]})"
    except httpx.HTTPError as exc:
        return f"{type(exc).__name__} ({str(exc)[:80]})"
    return f"HTTP {r.status_code} {r.text[:80]!r}"


def preflight(
    assistant: dict[str, Any], *, timeout_s: float = 10.0, require_ready: bool = True
) -> PreflightResult:
    """Refuse to measure a system that is not up.

    The adapter behind the assistant is a separate host reachable only over a tunnel.
    When it is down, Vapi still answers the callee's first utterance -- from the static
    ``assistant.firstMessage`` -- and the R1 deadline still passes. Every number in the
    report is then meaningless, so the run stops here instead.
    """
    model = assistant.get("model") or {}
    if model.get("provider") != "custom-llm":
        raise PreflightFailure(
            f"assistant model.provider is {model.get('provider')!r}, not 'custom-llm': this "
            "harness measures the adapter behind a custom LLM and has nothing to measure here"
        )
    raw_url = str(model.get("url") or "")
    parts = urlsplit(raw_url)
    if not parts.scheme or not parts.netloc:
        raise PreflightFailure(f"assistant model.url is not a usable URL: {raw_url!r}")
    # Origin only: model.url may carry a route secret path segment, which must not be
    # probed and must never be printed.
    origin = f"{parts.scheme}://{parts.netloc}"

    warnings: list[str] = []
    with httpx.Client(timeout=timeout_s, headers={"User-Agent": USER_AGENT}) as http:
        healthz = _probe(http, f"{origin}/healthz")
        readyz = _probe(http, f"{origin}/readyz")

    if not healthz.startswith("HTTP 200"):
        raise PreflightFailure(
            f"the adapter is not answering: GET {origin}/healthz -> {healthz}\n"
            "This is an OUTAGE, not a latency regression. Nothing below would have been\n"
            "measured against the adapter: with firstMessageMode="
            f"{assistant.get('firstMessageMode')!r} Vapi answers the callee's first\n"
            "utterance from the static assistant.firstMessage even with model.url dead,\n"
            "so the R1 deadline would have reported PASS. Bring the adapter up and re-run."
        )
    if not readyz.startswith("HTTP 200"):
        message = (
            f"the adapter is up but degraded: GET {origin}/readyz -> {readyz}\n"
            "That endpoint is 503 when Hermes is unreachable. R2 needs a real tool round\n"
            "trip to provoke an acknowledgement, so it cannot be measured in this state."
        )
        if require_ready:
            raise PreflightFailure(message)
        warnings.append(message)

    first_message = assistant.get("firstMessage") or None
    mode = assistant.get("firstMessageMode")
    if first_message and mode == "assistant-waits-for-user":
        warnings.append(
            "assistant.firstMessage is set and firstMessageMode is "
            "'assistant-waits-for-user': Vapi will speak that fixed string on the first "
            "callee turn without calling model.url, so the R1 check below measures Vapi, "
            "not the adapter's reason-for-calling fast path. The r1_provenance check "
            "reports this."
        )

    transcriber = assistant.get("transcriber") or {}
    if transcriber.get("provider") == "deepgram" and str(transcriber.get("model", "")).startswith(
        "flux"
    ):
        eot_timeout = transcriber.get("eotTimeoutMs")
        if isinstance(eot_timeout, int | float) and eot_timeout >= 3000:
            warnings.append(
                f"transcriber.eotTimeoutMs={eot_timeout}: each repeated callee utterance "
                "re-arms this timer, which is the mechanism behind the 10 s stall on live "
                "call 01a02524. Expect the flux_storm scenario to fail."
            )

    return PreflightResult(
        model_origin=origin,
        healthz=healthz,
        readyz=readyz,
        first_message=first_message,
        first_message_mode=mode if isinstance(mode, str) else None,
        transcriber=transcriber,
        warnings=tuple(warnings),
    )
