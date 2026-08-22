#!/usr/bin/env python3
"""Echo custom-llm: streams the last user message back verbatim as one SSE delta.

The companion to ``cache_probe.py``, and the reason that probe can attribute anything:
it gives a measurement exact control over the text Vapi's TTS receives on the
MODEL-STREAMED path -- the path the adapter actually delivers acknowledgements on -- so a
cache hit or miss there is a fact about the text and nothing else.

Contract, deliberately minimal:

    POST <any path>  ->  one SSE ``content`` delta carrying the last user message,
                         then ``[DONE]``, then the response ENDS.
    GET  <any path>  ->  ``{"ok": true}``, so a tunnel can be smoke-tested.

Two properties matter and are not incidental:

- **Control tokens pass through untouched.** A probe can inject
  ``"Alright, let me see. <flush /> "`` and reproduce the adapter's live framing byte for
  byte, trailing space included -- which is what Vapi's cache key would have to match.
- **The response ends immediately behind the delta**, which is what ``turns.py`` does
  today. A probe that stayed open would be measuring a different pipeline behaviour (see
  contracts §1.6: a flushed chunk on a stalled stream is frequently never rendered).

Bind it to localhost and expose it with a throwaway tunnel; it authenticates nothing and
must never be reachable as anything but a probe target::

    python tests/e2e/echo_llm.py 9098
    cloudflared tunnel --url http://127.0.0.1:9098
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

DEFAULT_PORT = 9098


def _last_user(body: dict[str, Any]) -> str:
    for message in reversed(body.get("messages") or []):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


class EchoHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        # One timestamped line per request: the probe's own record of when text was
        # handed to Vapi, independent of Vapi's log.
        sys.stderr.write(f"{time.time():.3f} {fmt % args}\n")

    def do_GET(self) -> None:
        payload = b'{"ok":true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            body = {}
        text = _last_user(body if isinstance(body, dict) else {})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        chunk = {
            "id": "probe",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "probe-echo",
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
        }
        sys.stderr.write(f"{time.time():.3f} echo {text!r}\n")
        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port", nargs="?", type=int, default=DEFAULT_PORT)
    port = parser.parse_args(argv).port
    server = ThreadingHTTPServer(("127.0.0.1", port), EchoHandler)
    sys.stderr.write(f"echo custom-llm on http://127.0.0.1:{port}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
