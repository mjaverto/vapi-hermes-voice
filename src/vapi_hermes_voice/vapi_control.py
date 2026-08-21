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

import asyncio
import logging
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger(__name__)

# A bare float handed to httpx is NOT a ceiling on the request: it is applied to the
# connect, read, write and pool phases SEPARATELY (verified -- `timeout=0.5` arrives at
# the transport as `{"connect": 0.5, "read": 0.5, "write": 0.5, "pool": 0.5}`). Each
# phase may consume its whole share and still succeed, so the caller waits for the SUM.
# Under the old 3.0 s default that is up to ~9 s on a request whose entire purpose is to
# be given up on quickly; one ReadTimeout alone put a live holding line at
# elapsed_ms=3902. `say` therefore bounds itself on the wall clock and treats the
# per-phase values as a fail-fast refinement rather than as the guarantee.
#
# Connect gets the tighter share on purpose. A cold client, or a host whose route just
# changed, stalls in CONNECT rather than in read, and spending the whole budget on the
# handshake guarantees there is none left to send the request and read the reply -- so
# it fails over to the SSE path with time still on the clock instead of being cut off
# anonymously by the outer bound. The remainder goes to read, where the measured 0.23 s
# round trip is actually spent. The client is process-lifetime, so connections are warm
# after the first acknowledgement of the process and connect costs nothing at all.
_CONNECT_BUDGET_SHARE = 0.4
# ...but never so small that a handshake is hopeless on a budget that has room for one.
_MIN_CONNECT_TIMEOUT_SECONDS = 0.1


def phase_timeouts(total: float) -> httpx.Timeout:
    """Per-phase httpx budget in which no single phase may outlive ``total``."""
    connect = min(total, max(_MIN_CONNECT_TIMEOUT_SECONDS, total * _CONNECT_BUDGET_SHARE))
    return httpx.Timeout(total, connect=connect)


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

        Returns within ``timeout`` seconds of wall clock, whatever the network does.
        That is a promise the caller's budget depends on and that httpx alone does not
        make (see ``_CONNECT_BUDGET_SHARE`` above): the acknowledgement path only ever
        waits here in order to decide to give up, so overshooting the deadline is
        strictly worse than failing at it.
        """
        if not _is_safe_control_url(control_url):
            logger.warning("ack control url rejected call=%s (not an https URL)", call_ref)
            return False
        try:
            async with asyncio.timeout(timeout):
                response = await self._client.post(
                    control_url,
                    json={"type": "say", "content": text},
                    timeout=phase_timeouts(timeout),
                )
        except TimeoutError:
            # The wall-clock bound fired: every phase stayed inside its own share but
            # together they did not. Reported as an ordinary failure so the caller
            # falls straight through to the fallback with no further waiting.
            logger.warning("ack control request timed out call=%s after=%.3fs", call_ref, timeout)
            return False
        except httpx.HTTPError as exc:
            # httpx's own timeout exceptions land here (httpx.TimeoutException does not
            # inherit from the builtin TimeoutError), which is the fast path: a phase
            # gave up before the outer bound had to.
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
