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
import time
from urllib.parse import urlsplit, urlunsplit

import httpx

logger = logging.getLogger(__name__)

# A bare float handed to httpx is NOT a ceiling on the request: it is applied to the
# connect, read, write and pool phases SEPARATELY (verified -- `timeout=0.5` arrives at
# the transport as `{"connect": 0.5, "read": 0.5, "write": 0.5, "pool": 0.5}`). Each
# phase may consume its whole share and still succeed, so the caller waits for the SUM.
# Under the old 3.0 s default that is up to ~9 s on a request whose entire purpose is to
# be given up on quickly; one ReadTimeout alone put a live holding line at
# elapsed_ms=3902. `say` therefore bounds itself on the WALL CLOCK -- that is the
# guarantee its caller's budget rests on -- and hands httpx the same figure per phase
# purely so a doomed phase raises a specific error instead of being cut off anonymously.
#
# No phase is given a tighter sub-share than the whole, and specifically not connect,
# which an earlier revision of this module did get wrong. Sub-dividing buys nothing for
# the deadline (the wall-clock bound already caps the total however the phases fall) and
# it costs something real: a handshake that would have finished in 0.20 s, leaving ample
# time for the 0.23 s round trip inside a 0.45 s ceiling, is instead abandoned at the
# sub-share -- and abandoned INTO the SSE fallback, which is not a neutral second best.
# That path carries a live Vapi defect (module docstring above, contracts 1.6): a
# flushed chunk on a stream that then stalls is accepted, echoed back in ~1 ms, and
# frequently never rendered to audio at all. Failing over is therefore a real risk of
# the callee hearing NOTHING, not merely of hearing it late, so the reliable channel is
# worth every millisecond the deadline can legitimately give it.


def phase_timeouts(total: float) -> httpx.Timeout:
    """Per-phase httpx budget in which no single phase may outlive ``total``.

    Every phase gets the whole budget: see above for why connect is deliberately not
    squeezed. ``say`` bounds the total on the wall clock regardless.
    """
    return httpx.Timeout(total)


# How long an idle pooled connection is kept. httpx's default is FIVE SECONDS, which for
# this workload means the pool is decorative: acknowledgements are at least
# `filler_min_gap_seconds` (10 s) apart and conversational turns are often much further,
# so essentially every control POST was paying a fresh TCP+TLS handshake -- not just the
# first one after a restart. 60 s spans a normal turn gap while staying under the idle
# timeouts load balancers typically enforce, so we are not left holding connections the
# far end has already closed.
_KEEPALIVE_EXPIRY_SECONDS = 60.0
# Re-warm an origin once its last known-good use is older than this. Half the expiry, so
# a warm-up always lands while the pooled connection is still alive (and is then a cheap
# reused round trip rather than a handshake) instead of racing its eviction.
_WARM_REFRESH_SECONDS = _KEEPALIVE_EXPIRY_SECONDS / 2
# The warm-up is off the critical path -- nothing in the turn waits for it -- so it gets
# a generous bound rather than the acknowledgement's tight one. Deliberately NOT
# `ack_control_timeout`: that ceiling is sized so a handshake AND a round trip together
# fit the callee's patience, and a warm-up held to it would give up on exactly the slow
# handshake it exists to absorb. It is bounded at all only so a hung connect cannot
# outlive the call that triggered it.
_WARM_TIMEOUT_SECONDS = 5.0


def _origin_of(url: str) -> str | None:
    """``https://host[:port]/`` for ``url``, or None when it is not a safe control URL.

    httpx pools connections by origin -- scheme, host, port -- and not by path, so a
    request to this root shares the pooled connection a later
    ``POST https://host/<call-id>/control`` will use. That is what makes warming possible
    at all: the control URL is per-call
    (``https://phone-call-websocket.<region>-backend-productionN.vapi.ai/<id>/control``)
    and cannot be known before the call exists, but its ORIGIN is shared by every call on
    that backend and is known the moment the first request of a call arrives.
    """
    if not _is_safe_control_url(url):
        return None
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, "/", "", ""))


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

    Sharing the client is what makes connection reuse possible, and reuse is what keeps
    a TCP+TLS handshake off the acknowledgement's critical path -- but sharing alone was
    not enough, because httpx evicts idle pooled connections after five seconds by
    default. See :attr:`keepalive_expiry_seconds`.
    """

    keepalive_expiry_seconds = _KEEPALIVE_EXPIRY_SECONDS

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._client = httpx.AsyncClient(
            transport=transport,
            limits=httpx.Limits(keepalive_expiry=self.keepalive_expiry_seconds),
        )
        # origin -> monotonic time we last completed a request to it. Drives `warm`'s
        # freshness check; bounded by the number of Vapi backend origins a process ever
        # talks to, which is a handful of regional hostnames, not a per-call key.
        self._last_used: dict[str, float] = {}

    async def aclose(self) -> None:
        await self._client.aclose()

    async def warm(self, control_url: str, *, timeout: float = _WARM_TIMEOUT_SECONDS) -> bool:
        """Pay the TCP+TLS handshake to ``control_url``'s origin ahead of needing it.

        Called at the top of a turn, off the critical path, so the acknowledgement due
        ``filler_after_seconds`` later finds a live pooled connection instead of
        handshaking inside its own deadline. Returns True when the origin is warm --
        including when it was already warm and nothing was sent.

        This matters more than latency. A cold handshake that does not fit the
        acknowledgement's deadline does not merely make the line late, it diverts it to
        the SSE fallback, which carries a live Vapi defect that can render no audio at
        all (see the module docstring). The first acknowledgement after every adapter
        restart is exactly where the requirement is judged, so it must not be the one
        paying for the handshake.

        A GET of the origin ROOT, never the control URL itself: all that is wanted is a
        pooled connection, and httpx pools by origin, so this touches no call resource
        and can have no effect on any call. Whatever it answers -- 404, 405, anything --
        is success as far as this is concerned; only the connection matters.

        Never raises. A failed warm-up is not an error, just a handshake still to pay.
        """
        origin = _origin_of(control_url)
        if origin is None:
            return False
        last_used = self._last_used.get(origin)
        now = time.monotonic()
        if last_used is not None and now - last_used < _WARM_REFRESH_SECONDS:
            return True
        try:
            async with asyncio.timeout(timeout):
                await self._client.get(origin, timeout=phase_timeouts(timeout))
        except (TimeoutError, httpx.HTTPError) as exc:
            logger.info(
                "control origin warm-up failed origin=%s error=%s", origin, type(exc).__name__
            )
            return False
        self._last_used[origin] = time.monotonic()
        logger.debug("control origin warmed origin=%s", origin)
        return True

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
        # A completed round trip means the pooled connection is live right now, so the
        # next turn's warm-up can skip. Recorded regardless of the HTTP status: this
        # tracks the CONNECTION, and a 404 proves one just as well as a 200.
        origin = _origin_of(control_url)
        if origin is not None:
            self._last_used[origin] = time.monotonic()
        if response.status_code >= 400:
            logger.warning(
                "ack control request rejected call=%s status=%d", call_ref, response.status_code
            )
            return False
        return True
