"""Vapi's Live Call Control endpoint: how the adapter speaks once the model.url
stream for a turn has already been ended.

Root cause of why the model.url stream cannot be left open (docs/integration-
contracts.md section 1.6): a filler chunk written into the streamed chat-completion
response, terminated with ``<flush />``, is accepted by Vapi immediately (echoed
back as a ``model-output``/``voice-input`` event within ~1 ms of being sent) but is
NOT reliably turned into audio when that same stream then sits idle for more than a
few seconds -- which is exactly what happens whenever the Hermes turn behind it runs
long (a tool round trip, a multi-tool agentic run, or simply a slow provider).
Measured live and confirmed on an isolated probe carrying no Hermes traffic at all: a
lone flushed chunk followed by an 18 s stall on an otherwise identical stream is not
spoken at all until the stream produces more content or terminates -- and ending the
response immediately after the flushed chunk got it spoken in ~0.2 s on every trial.
So ``turns.py`` no longer waits for Vapi to notice the stream is done: it ends the
response itself the instant the acknowledgement is flushed, and delivers whatever
Hermes goes on to produce through THIS module instead, on a background task the
caller's response no longer depends on.

Every Custom LLM request body Vapi sends already carries ``call.monitor.controlUrl``
(present with no assistant config change: no ``monitorPlan.controlEnabled`` override
needed, confirmed live). ``POST controlUrl {"type": "say", "content": text}`` renders
speech in ~0.3 s measured, independent of the model.url stream's state.

What this module does NOT do, and measured why: pre-warm a connection. Probed live
against the real origin (GET /, HEAD /, OPTIONS /, GET of a nonexistent path, POST to
a nonexistent call id) every one of those answers in ~0.22-0.33 s -- but every single
response, including the 400s, carries ``Connection: close``. httpx pools connections
by origin, but a connection the SERVER closes the instant it answers can never be
reused by a later request to that origin, whatever shape opened it. A prior warm-up
that GETs the origin root therefore pays a whole extra TCP+TLS handshake for a
connection that is already dead before the real POST needs one -- it cannot help, by
construction, and it previously had its own bug on top of achieving nothing: one in
roughly 40 cold-process trials hung for the full length of its own timeout (a
transient SYN-level stall, reproduced once live), which is exactly the failure the
adapter journal caught. Dropped rather than fixed: there is no request shape this
origin will keep alive for us to warm with.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from urllib.parse import urlsplit

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
# the deadline (the wall-clock bound already caps the total however the phases fall).


def phase_timeouts(total: float) -> httpx.Timeout:
    """Per-phase httpx budget in which no single phase may outlive ``total``.

    Every phase gets the whole budget: see above for why connect is deliberately not
    squeezed. ``say`` bounds the total on the wall clock regardless.
    """
    return httpx.Timeout(total)


# How long an idle pooled connection is kept, IF the far end ever leaves one open long
# enough to matter -- unverified for a genuine `say` response (every probed response
# this module's docstring describes was a 400 to a deliberately invalid request, and
# those all closed immediately). httpx's own default is five seconds; raising it to a
# minute costs nothing if the far end never keeps a connection alive at all, and could
# only help if a real `say` response someday does.
_KEEPALIVE_EXPIRY_SECONDS = 60.0


def _is_safe_control_url(url: str) -> bool:
    """https with a real host. ``call.monitor.controlUrl`` is Vapi-platform-supplied,
    not caller-influenced, but validating before an outbound POST is nearly free and
    stops a malformed or unexpected value from being handed to httpx unexamined.
    """
    parts = urlsplit(url)
    return parts.scheme == "https" and bool(parts.netloc)


@dataclass(frozen=True, slots=True)
class SayOutcome:
    """The result of one ``say`` POST: whether it landed, and why not if it did not.

    ``status_code`` is None for a timeout or a network-level error (no response was
    ever read) and the HTTP status Vapi returned otherwise. That distinction is what
    lets a caller retry a plausibly-transient failure (timeout, network error, 5xx)
    without also retrying an identical POST into a rejection Vapi is not going to
    reconsider (a 4xx -- most plausibly the call has already moved past this turn).
    """

    delivered: bool
    status_code: int | None = None


class VapiControlClient:
    """One shared HTTP client for Vapi's Live Call Control endpoint, never rebuilt
    per turn (same lifecycle pattern as :class:`hermes_client.HermesClient`).
    """

    keepalive_expiry_seconds = _KEEPALIVE_EXPIRY_SECONDS

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._client = httpx.AsyncClient(
            transport=transport,
            limits=httpx.Limits(keepalive_expiry=self.keepalive_expiry_seconds),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def say(
        self, control_url: str, text: str, *, call_ref: str, timeout: float
    ) -> SayOutcome:
        """POST ``{"type": "say", "content": text}`` to ``control_url``.

        Never raises: any failure (unsafe URL, network error, timeout, non-2xx) is
        reported through the returned :class:`SayOutcome`. Logged at WARNING for a
        timeout, network error or 5xx -- all plausibly transient, all worth an
        operator's attention if they recur -- and at DEBUG for a 4xx, which is Vapi
        declining the request outright (most plausibly because the call has already
        moved past this turn) and not, by itself, evidence of anything wrong here.
        The caller (``turns._finish_turn_via_control``) makes the retry decision that
        follows from that same distinction; this is just where it is first observed.

        Returns within ``timeout`` seconds of wall clock, whatever the network does.
        That is a promise the caller's budget depends on and that httpx alone does not
        make (see the phase-budget note above): the caller waits here in order to
        decide whether to retry or give up, so overshooting the deadline is strictly
        worse than failing at it.
        """
        if not _is_safe_control_url(control_url):
            logger.warning("ack control url rejected call=%s (not an https URL)", call_ref)
            return SayOutcome(delivered=False)
        try:
            async with asyncio.timeout(timeout):
                response = await self._client.post(
                    control_url,
                    json={"type": "say", "content": text},
                    timeout=phase_timeouts(timeout),
                )
        except TimeoutError:
            # The wall-clock bound fired: every phase stayed inside its own share but
            # together they did not.
            logger.warning("ack control request timed out call=%s after=%.3fs", call_ref, timeout)
            return SayOutcome(delivered=False)
        except httpx.HTTPError as exc:
            # httpx's own timeout exceptions land here (httpx.TimeoutException does not
            # inherit from the builtin TimeoutError), which is the fast path: a phase
            # gave up before the outer bound had to.
            logger.warning(
                "ack control request failed call=%s error=%s", call_ref, type(exc).__name__
            )
            return SayOutcome(delivered=False)
        if response.status_code >= 500:
            logger.warning(
                "ack control request failed call=%s status=%d", call_ref, response.status_code
            )
            return SayOutcome(delivered=False, status_code=response.status_code)
        if response.status_code >= 400:
            logger.debug(
                "ack control request declined call=%s status=%d", call_ref, response.status_code
            )
            return SayOutcome(delivered=False, status_code=response.status_code)
        return SayOutcome(delivered=True, status_code=response.status_code)
