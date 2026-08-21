"""Logging setup with redaction of secrets and phone numbers."""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Iterable

from vapi_hermes_voice.config import Settings

SECRET_ENV_NAMES: tuple[str, ...] = (
    "VHV_HERMES_API_KEY",
    "VHV_ADAPTER_API_KEY",
    "VHV_ROUTE_SECRET",
)

_E164_RE = re.compile(r"\+[1-9]\d{6,14}")
# Route-secret path segments as they appear in access logs, e.g. uvicorn's own
# 'POST /v/<secret>/chat/completions' line, which bypasses app-level hashing.
_ROUTE_PATH_RE = re.compile(r"/v/([^/\s\"']+)/chat/completions")
_CONFIGURED_ATTR = "_vhv_logging_configured"


def redact_phone(number: str) -> str:
    """Mask the middle digits of an E.164 number: '+13475551234' -> '+1******1234'."""
    if _E164_RE.fullmatch(number) is None:
        return "*" * len(number)
    return number[:2] + "*" * (len(number) - 6) + number[-4:]


class RedactionFilter(logging.Filter):
    """Replaces occurrences of secret values and E.164-looking numbers in log records.

    The record message is formatted eagerly (``record.getMessage()``) and rewritten so
    downstream handlers never see raw secrets or full phone numbers. Request paths of
    the form ``/v/<route-secret>/chat/completions`` are rewritten too: the route
    secret becomes ``[REDACTED]``.
    """

    def __init__(self, secrets: Iterable[str]) -> None:
        super().__init__()
        self._secrets: tuple[str, ...] = tuple(s for s in secrets if s)

    def _redact(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, "[REDACTED]")
        text = _ROUTE_PATH_RE.sub("/v/[REDACTED]/chat/completions", text)
        return _E164_RE.sub(lambda match: redact_phone(match.group(0)), text)

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # bad %-format call: scrub what we have, never raise
            message = str(record.msg)
        record.msg = self._redact(message)
        record.args = None
        return True


def configure_logging(settings: Settings) -> None:
    """Install key=value stderr logging with redaction on the root logger. Idempotent."""
    root = logging.getLogger()
    if getattr(root, _CONFIGURED_ATTR, False):
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("ts=%(asctime)s level=%(levelname)s logger=%(name)s msg=%(message)s")
    )
    secrets = [
        settings.hermes_api_key.get_secret_value(),
        settings.adapter_api_key.get_secret_value(),
    ]
    if settings.route_secret is not None:
        secrets.append(settings.route_secret.get_secret_value())
    handler.addFilter(RedactionFilter(secrets))
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    setattr(root, _CONFIGURED_ATTR, True)
