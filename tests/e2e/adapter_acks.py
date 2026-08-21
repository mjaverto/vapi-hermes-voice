"""Read the adapter's own record of the acknowledgements it emitted.

This is the only module in the harness that talks to the adapter directly, and it
exists for one reason: an acknowledgement's TEXT cannot say who wrote it. "Okay, one
moment." and "Sure, give me a second." are verbatim members of the adapter's own
``_DEFAULT_FILLER_PHRASES``, and ``speech.VOICE_SYSTEM_PROMPT`` forbids the MODEL from
producing lines of that shape -- so a spoken holding phrase is either the adapter's or
a regression of that prohibition, and nothing on Vapi's transcript or on the websocket
transport can tell which. ``GET /debug/acks/{call_ref}`` can: a phrase that is not in
the adapter's record is, by definition, not the adapter's.

Failure is never fatal here. Every way this can go wrong -- endpoint absent, disabled,
tunnel rotated, key missing, record aged out -- comes back as
``AdapterAcks(unavailable=...)``, and the scoring layer degrades that to UNKNOWN, which
is exactly what it reported before this endpoint existed. A live run must never fail
because a diagnostic surface was unreachable.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from .deadlines import AdapterAck, AdapterAcks

__all__ = ["ADAPTER_KEY_ENV", "adapter_call_ref", "debug_acks_url", "fetch_adapter_acks"]

ADAPTER_KEY_ENV = "VHV_ADAPTER_API_KEY"

# Vapi's OpenAI client appends /chat/completions to model.url, and the adapter also
# serves the doubled path for a URL configured WITH the suffix (server.py). Either way
# the debug endpoint is a sibling of the chat endpoint under the same optional
# /v/{route_secret} prefix, so the prefix is recovered by stripping the suffix rather
# than by asking the operator to configure a second URL that could disagree with it.
_CHAT_SUFFIX = "/chat/completions"


def adapter_call_ref(call_id: str) -> str:
    """The adapter's own ``call_ref`` for ``call_id``, from the adapter's own code.

    Imported, never reimplemented. It is a truncated hash
    (``call_state.call_ref``: ``sha256(call_id).hexdigest()[:12]``) and a harness copy
    of that arithmetic is a second definition that can drift: the day the adapter
    changes it, a reimplementation here starts silently 404ing and the harness reports
    "attribution unknown" for a reason that has nothing to do with the adapter's
    behaviour. The import is not optional cleverness either -- the harness already
    resolves the acknowledgement phrase pool out of ``vapi_hermes_voice.config`` for
    the same reason (``run_voice_deadlines._ack_phrases``).
    """
    from vapi_hermes_voice.call_state import call_ref

    return call_ref(call_id)


def debug_acks_url(model_url: str, call_ref: str) -> str:
    """``GET`` URL for one call's record, derived from the assistant's ``model.url``.

    NEVER printed or logged by callers: ``model.url`` may carry a route secret path
    segment (``/v/{secret}/chat/completions``), which is a credential.
    """
    parts = urlsplit(model_url)
    path = parts.path
    while path.endswith(_CHAT_SUFFIX):
        path = path[: -len(_CHAT_SUFFIX)]
    return urlunsplit((parts.scheme, parts.netloc, f"{path}/debug/acks/{call_ref}", "", ""))


def _parse(payload: Any) -> AdapterAcks:
    """The wire form -> :class:`AdapterAcks`, refusing anything malformed.

    A body that does not have the expected shape is reported as unavailable, not
    coerced: a half-understood record would attribute acknowledgements by accident,
    which is worse than declining to attribute them at all.
    """
    if not isinstance(payload, dict):
        return AdapterAcks(unavailable="the endpoint returned something that is not an object")
    raw = payload.get("acks")
    dropped = payload.get("dropped")
    if not isinstance(raw, list) or not isinstance(dropped, int):
        return AdapterAcks(
            unavailable="the endpoint's response is missing 'acks' or 'dropped': the "
            "field contract has changed and this harness would misread it"
        )
    acks: list[AdapterAck] = []
    for item in raw:
        if not isinstance(item, dict):
            return AdapterAcks(unavailable="an entry in 'acks' is not an object")
        text = item.get("text")
        channel = item.get("channel")
        at_epoch_s = item.get("at_epoch_s")
        elapsed_ms = item.get("elapsed_ms")
        if (
            not isinstance(text, str)
            or not isinstance(channel, str)
            or not isinstance(at_epoch_s, int | float)
            or not isinstance(elapsed_ms, int)
        ):
            return AdapterAcks(
                unavailable=f"an entry in 'acks' has the wrong shape: {sorted(item)}"
            )
        acks.append(
            AdapterAck(
                text=text,
                channel=channel,
                at_epoch_s=float(at_epoch_s),
                elapsed_ms=elapsed_ms,
            )
        )
    return AdapterAcks(acks=tuple(acks), dropped=dropped)


def fetch_adapter_acks(
    model_url: str,
    call_id: str,
    *,
    api_key: str | None = None,
    timeout_s: float = 10.0,
    env: dict[str, str] | None = None,
    transport: httpx.BaseTransport | None = None,
) -> AdapterAcks:
    """The adapter's record for ``call_id``, or an ``unavailable`` explaining why not.

    ``api_key`` defaults to ``$VHV_ADAPTER_API_KEY`` -- the same key the deployed
    adapter checks on ``/chat/completions``, because the debug endpoint is behind the
    same bearer (and the same optional route secret). Read from the environment only
    and never echoed, not even truncated.

    Never raises. Every outcome is a report, because the harness's own verdicts must
    not depend on a diagnostic surface being up.
    """
    source = env if env is not None else dict(os.environ)
    key = (api_key if api_key is not None else source.get(ADAPTER_KEY_ENV, "")).strip()
    if not key:
        return AdapterAcks(
            unavailable=f"{ADAPTER_KEY_ENV} is not set, so the adapter's acknowledgement "
            "record cannot be read. Export it from the deployed adapter's .env, e.g.\n"
            f"    export {ADAPTER_KEY_ENV}=$(ssh openclaw \"grep '^{ADAPTER_KEY_ENV}=' "
            '/home/mja311/src/vapi-hermes-voice/.env | cut -d= -f2-")'
        )
    try:
        url = debug_acks_url(model_url, adapter_call_ref(call_id))
    except ImportError as exc:  # pragma: no cover - only when the package is absent
        return AdapterAcks(
            unavailable="the vapi_hermes_voice package is not importable, so this "
            f"harness cannot derive the adapter's call_ref ({exc}). Run from the repo "
            "root with the project installed."
        )
    try:
        with httpx.Client(transport=transport, timeout=timeout_s) as http:
            # `transport` mounts a fake adapter in-process for this module's own tests,
            # the same way `server.create_app(hermes_transport=...)` does on the other
            # side of the wire. Unset in every real run.
            response = http.get(
                url,
                headers={
                    "Authorization": f"Bearer {key}",
                    "User-Agent": "vapi-hermes-voice-e2e/1.0",
                },
            )
    except httpx.HTTPError as exc:
        # The URL is deliberately absent from this message: it may carry a route secret.
        return AdapterAcks(
            unavailable=f"GET /debug/acks/{{call_ref}} on the adapter origin failed: "
            f"{type(exc).__name__}"
        )
    if response.status_code == 404:
        # NOT "the adapter emitted nothing": a call it drove and stayed silent on comes
        # back 200 with an empty `acks` (turns.stream_turn opens the record before
        # anything can be acknowledged). 404 means there is no record to read at all.
        return AdapterAcks(
            unavailable="the adapter holds no acknowledgement record for this call "
            "(404): VHV_DEBUG_ACK_JOURNAL is off, the record aged out of its TTL, the "
            "adapter restarted since the call, or this build predates the endpoint"
        )
    if response.status_code == 401:
        return AdapterAcks(
            unavailable=f"the adapter rejected the {ADAPTER_KEY_ENV} bearer (401): the "
            "key this harness holds is not the one the deployed adapter checks"
        )
    if response.status_code != 200:
        return AdapterAcks(
            unavailable=f"the adapter answered HTTP {response.status_code} for its "
            "acknowledgement record"
        )
    try:
        payload = response.json()
    except ValueError:
        return AdapterAcks(unavailable="the endpoint's response was not JSON")
    return _parse(payload)
