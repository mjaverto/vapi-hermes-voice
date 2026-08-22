"""A throwaway public webhook sink, for finding out what Vapi actually sends at end of call.

Not part of the adapter and not imported by it. This exists because
``end-of-call-report`` could not be characterised any other way: assistant
``b39379dc`` carries no ``server`` field (so the real adapter never receives one), and
editing that assistant is not this workstream's call to make. Vapi does accept
``server`` inside a per-call ``assistantOverrides`` (verified live, see §1.14), which
makes a disposable sink like this the whole verification story -- no assistant edit, no
phone number, nothing rings.

Deliberately dependency-free (``http.server`` only) so it can be scp'd to a box that
has ``cloudflared`` but not this project's virtualenv.

Every request is appended to the log as one JSON line carrying the wall clock, the
monotonic clock, the message ``type`` and the VERBATIM body. Verbatim matters: the
point of the exercise is to discover fields nobody thought to ask for, and a recorder
that keeps only the fields it was told about cannot make a discovery. The presented
secret is compared and then reported as a BOOLEAN -- the header value itself is never
written to disk, because this log gets read back over ssh and pasted into notes.
"""

from __future__ import annotations

import json
import os
import secrets
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG_PATH = os.environ.get("RECORDER_LOG", "/tmp/vapi-report-recorder.jsonl")  # noqa: S108
EXPECTED_SECRET = os.environ.get("RECORDER_SECRET", "")
MAX_BODY = 4 * 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:
        """Silence stderr access logging: it would carry the path, and the path is a secret."""

    def do_POST(self) -> None:  # noqa: N802 - http.server's required spelling
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(min(length, MAX_BODY)) if length else b""
        presented = self.headers.get("x-vapi-secret") or ""
        # compare_digest, not ==: this is a habit worth keeping even in a throwaway.
        ok = bool(EXPECTED_SECRET) and secrets.compare_digest(presented, EXPECTED_SECRET)
        try:
            payload = json.loads(body)
        except ValueError:
            payload = None
        message = (payload or {}).get("message") if isinstance(payload, dict) else None
        row = {
            "wall": time.time(),
            "mono": time.monotonic(),
            "path": self.path,
            "secret_ok": ok,
            "bytes": len(body),
            "type": (message or {}).get("type") if isinstance(message, dict) else None,
            # Verbatim, so a field nobody predicted is still there to be found later.
            "body": payload if payload is not None else body.decode("utf-8", "replace"),
        }
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", "2")
        self.end_headers()
        self.wfile.write(b"{}")


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8791
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
