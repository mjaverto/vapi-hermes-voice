"""Vapi's Live Call Control endpoint: the acknowledgement delivery path that is
immune to the model.url stream's own buffering.

Root cause (docs/integration-contracts.md section 1.6): a filler chunk written into
the streamed chat-completion response, terminated with ``<flush />``, is accepted by
Vapi immediately (echoed back as a ``model-output``/``voice-input`` event within
~1 ms of being sent) but is NOT reliably turned into audio when that same stream then
sits idle for more than a few seconds -- which is exactly what happens whenever the
Hermes turn behind it runs long (a tool round trip, a multi-tool agentic run, or
simply a slow provider). Measured live and confirmed on an isolated probe carrying no
Hermes traffic at all: a lone flushed chunk followed by an 18 s stall on an otherwise
identical stream is not spoken at all until the stream produces more content or
terminates, and when it finally is, it can be spoken late and concatenated with a
second buffered fragment (the audible duplication reported live: "Sure. Give me a
second Sure. Give me a second."). Neither omitting `<flush />` nor sending
byte-level keepalive (empty content deltas, raw SSE comment lines) changes this --
the same probe reproduces the identical stall regardless.

Every Custom LLM request body Vapi sends already carries ``call.monitor.controlUrl``
(present with no assistant config change: no ``monitorPlan.controlEnabled`` override
needed, confirmed live). ``POST controlUrl {"type": "say", "content": text}`` renders
speech in ~0.3 s measured, independent of the model.url stream's state -- proven on a
probe whose model stream stayed completely silent for the full 26 s duration of the
call, while `say` still spoke two separate acknowledgements right on schedule, clean
and undivided. This module wraps that call; ``turns.py`` uses it as the primary
acknowledgement channel and falls back to the old SSE-embedded delivery only when a
control URL is unavailable or the request itself fails.
"""

from __future__ import annotations

import logging
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger(__name__)


def _is_safe_control_url(url: str) -> bool:
    """https with a real host. ``call.monitor.controlUrl`` is Vapi-platform-supplied,
    not caller-influenced, but validating before an outbound POST is nearly free and
    stops a malformed or unexpected value from being handed to httpx unexamined.
    """
    parts = urlsplit(url)
    return parts.scheme == "https" and bool(parts.netloc)


class VapiControlClient:
    """One shared HTTP client for Vapi's Live Call Control endpoint, never rebuilt
    per turn (same lifecycle pattern as :class:`hermes_client.HermesClient`).
    """

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._client = httpx.AsyncClient(transport=transport)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def say(self, control_url: str, text: str, *, call_ref: str, timeout: float) -> bool:
        """POST ``{"type": "say", "content": text}`` to ``control_url``.

        True iff Vapi accepted the request (HTTP < 400). Never raises: any failure
        (unsafe URL, network error, timeout, non-2xx) is logged and reported as
        False, so the caller can fall back to the SSE-embedded delivery -- an ack
        control failure must never be worse than the pre-existing behaviour.
        """
        if not _is_safe_control_url(control_url):
            logger.warning("ack control url rejected call=%s (not an https URL)", call_ref)
            return False
        try:
            response = await self._client.post(
                control_url, json={"type": "say", "content": text}, timeout=timeout
            )
        except httpx.HTTPError as exc:
            logger.warning(
                "ack control request failed call=%s error=%s", call_ref, type(exc).__name__
            )
            return False
        if response.status_code >= 400:
            logger.warning(
                "ack control request rejected call=%s status=%d", call_ref, response.status_code
            )
            return False
        return True
