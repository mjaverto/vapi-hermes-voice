"""Cover for the one module in the harness that talks to the adapter.

Pure and in-process: the adapter is mounted behind an ``httpx.MockTransport``, the same
way ``server.create_app(hermes_transport=...)`` mounts a fake Hermes on the other side
of the wire. No network, no Vapi call, no ``say``.

Not collected by a bare ``pytest`` run -- like the rest of ``tests/e2e``. Run directly:

    uv run pytest tests/e2e/test_adapter_acks.py
"""

from __future__ import annotations

from typing import Any

import httpx

from vapi_hermes_voice.call_state import call_ref as adapter_call_ref_source

from .adapter_acks import ADAPTER_KEY_ENV, adapter_call_ref, debug_acks_url, fetch_adapter_acks

CALL_ID = "01a0262b-a1ce-733b-aad5-a93df060162e"
KEY = {ADAPTER_KEY_ENV: "adapter-key-0123456789"}
BARE_URL = "https://room-kde-her-disclaimers.trycloudflare.com/chat/completions"
SECRET_URL = (
    "https://room-kde-her-disclaimers.trycloudflare.com/v/route-secret-0123456789/chat/completions"
)

RECORD = {
    "call_ref": adapter_call_ref_source(CALL_ID),
    "acks": [
        {
            "text": "Okay, one moment.",
            "channel": "control",
            "at_epoch_s": 1_755_823_456.123,
            "elapsed_ms": 1130,
        }
    ],
    "dropped": 0,
    "limits": {"max_calls": 64, "max_entries_per_call": 16, "ttl_seconds": 900.0},
}


def responder(
    status: int = 200, body: Any = None, *, seen: list[httpx.Request] | None = None
) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        if isinstance(body, str):
            return httpx.Response(status, text=body)
        return httpx.Response(status, json=RECORD if body is None else body)

    return httpx.MockTransport(handle)


# --- call_ref: the adapter's own arithmetic, not a copy of it ----------------------


def test_call_ref_is_the_adapters_own_function_not_a_reimplementation() -> None:
    """A harness copy of ``sha256(call_id)[:12]`` is a second definition that can
    drift, and the day the adapter changes it the harness would 404 and blame the
    adapter's behaviour for a bug of its own. So it is imported.
    """
    assert adapter_call_ref(CALL_ID) == adapter_call_ref_source(CALL_ID)
    # The shape the deployed adapter's journal will accept, and the value seen live in
    # its own log line for this exact call id: `turn filler call=6196615ba4e6`.
    assert adapter_call_ref(CALL_ID) == "6196615ba4e6"


# --- URL derivation: the route secret must survive, the suffix must not -------------


def test_url_is_a_sibling_of_the_chat_endpoint() -> None:
    ref = adapter_call_ref(CALL_ID)
    assert debug_acks_url(BARE_URL, ref).endswith(f"/debug/acks/{ref}")
    assert "/chat/completions" not in debug_acks_url(BARE_URL, ref)


def test_url_keeps_a_route_secret_prefix() -> None:
    """The debug endpoint lives behind the same optional ``/v/{secret}`` prefix as the
    chat endpoint (server.py), so dropping the prefix would 404 on a hardened
    deployment and be reported as "the endpoint does not exist".
    """
    url = debug_acks_url(SECRET_URL, "6196615ba4e6")
    assert url.endswith("/v/route-secret-0123456789/debug/acks/6196615ba4e6")


def test_url_tolerates_a_doubled_chat_suffix() -> None:
    """A model.url configured WITH the suffix double-appends; the adapter serves both
    (server.py), so this must resolve the same base either way.
    """
    doubled = f"{BARE_URL}/chat/completions"
    assert debug_acks_url(doubled, "abc") == debug_acks_url(BARE_URL, "abc")


# --- reading the record ------------------------------------------------------------


def test_a_healthy_record_is_parsed_and_authenticated() -> None:
    seen: list[httpx.Request] = []
    record = fetch_adapter_acks(BARE_URL, CALL_ID, env=KEY, transport=responder(seen=seen))

    assert record.usable and record.conclusive
    assert [(a.text, a.channel, a.elapsed_ms) for a in record.acks] == [
        ("Okay, one moment.", "control", 1130)
    ]
    assert record.acks[0].at_epoch_s == 1_755_823_456.123
    assert seen[0].headers["authorization"] == f"Bearer {KEY[ADAPTER_KEY_ENV]}"
    assert seen[0].url.path.endswith(f"/debug/acks/{adapter_call_ref(CALL_ID)}")


def test_a_dropped_record_is_usable_but_not_conclusive() -> None:
    body = dict(RECORD, dropped=3)
    record = fetch_adapter_acks(BARE_URL, CALL_ID, env=KEY, transport=responder(body=body))
    assert record.usable
    assert not record.conclusive  # an incomplete record may not accuse anyone
    assert record.dropped == 3


def test_an_empty_record_is_evidence_not_an_absence() -> None:
    """The load-bearing case: the adapter drove the call and emitted nothing. This is
    what lets the scoring layer call a spoken holding phrase model-authored, so it must
    NOT be flattened into "unavailable".
    """
    body = {"call_ref": adapter_call_ref(CALL_ID), "acks": [], "dropped": 0}
    record = fetch_adapter_acks(BARE_URL, CALL_ID, env=KEY, transport=responder(body=body))
    assert record.usable and record.conclusive
    assert record.acks == ()


# --- every way this can fail is a report, never an exception ------------------------


def test_a_missing_key_is_reported_with_the_command_that_fixes_it() -> None:
    record = fetch_adapter_acks(BARE_URL, CALL_ID, env={}, transport=responder())
    assert not record.usable
    assert ADAPTER_KEY_ENV in str(record.unavailable)


def test_a_404_is_no_record_and_says_so_without_claiming_silence() -> None:
    record = fetch_adapter_acks(BARE_URL, CALL_ID, env=KEY, transport=responder(404, {}))
    assert not record.usable
    assert "no acknowledgement record" in str(record.unavailable)
    assert "VHV_DEBUG_ACK_JOURNAL is off" in str(record.unavailable)


def test_a_401_names_the_key_mismatch() -> None:
    record = fetch_adapter_acks(BARE_URL, CALL_ID, env=KEY, transport=responder(401, {}))
    assert not record.usable
    assert "401" in str(record.unavailable)


def test_a_transport_error_never_leaks_the_url() -> None:
    """``model.url`` may carry a route secret, so no failure message may echo it."""

    def boom(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nodename nor servname provided")

    record = fetch_adapter_acks(SECRET_URL, CALL_ID, env=KEY, transport=httpx.MockTransport(boom))
    assert not record.usable
    assert "route-secret-0123456789" not in str(record.unavailable)
    assert "ConnectError" in str(record.unavailable)


def test_a_changed_field_contract_is_refused_not_half_understood() -> None:
    """A body this module does not fully recognise must not be coerced: attributing
    acknowledgements off a half-read record is worse than declining to attribute them.
    """
    for body in (
        {"acks": [{"text": "hi"}], "dropped": 0},  # entry missing channel/times
        {"acks": "not a list", "dropped": 0},
        {"acks": [], "dropped": "none"},
        [],
    ):
        record = fetch_adapter_acks(BARE_URL, CALL_ID, env=KEY, transport=responder(body=body))
        assert not record.usable, body


def test_a_non_json_body_is_refused() -> None:
    record = fetch_adapter_acks(
        BARE_URL, CALL_ID, env=KEY, transport=responder(body="<html>gateway error</html>")
    )
    assert not record.usable
    assert "not JSON" in str(record.unavailable)


def test_an_unexpected_status_is_reported_with_its_code() -> None:
    record = fetch_adapter_acks(BARE_URL, CALL_ID, env=KEY, transport=responder(502, {}))
    assert not record.usable
    assert "502" in str(record.unavailable)
