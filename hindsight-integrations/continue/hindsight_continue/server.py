"""A small stdlib HTTP server exposing the Continue context-provider endpoint.

Run it alongside your editor and point Continue's ``http`` context provider at
``http://<host>:<port>/`` (see the README). It has no third-party server
dependency — only :mod:`http.server` from the standard library — so it stays a
lightweight sidecar.
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from hindsight_client import Hindsight

from ._client import resolve_client
from .config import HindsightContinueConfig, get_config
from .errors import HindsightError
from .provider import build_context_items, serialize

logger = logging.getLogger("hindsight_continue.server")

_MAX_BODY_BYTES = 1_000_000


def make_handler(client: Hindsight, config: HindsightContinueConfig):
    """Build a request-handler class bound to a client and config."""

    class _Handler(BaseHTTPRequestHandler):
        # Quiet the default per-request stderr logging; route through our logger.
        def log_message(self, fmt: str, *args) -> None:  # noqa: A002 - stdlib signature
            logger.debug("%s - %s", self.address_string(), fmt % args)

        def _write_json(self, status: int, payload) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - stdlib signature
            if self.path.rstrip("/") in ("", "/health"):
                self._write_json(200, {"status": "ok", "service": "hindsight-continue"})
            else:
                self._write_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - stdlib signature
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            if length > _MAX_BODY_BYTES:
                self._write_json(413, {"error": "request body too large"})
                return

            raw = self.rfile.read(length) if length else b""
            try:
                payload = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                self._write_json(400, {"error": "invalid JSON body"})
                return
            if not isinstance(payload, dict):
                self._write_json(400, {"error": "request body must be a JSON object"})
                return

            try:
                items = build_context_items(payload, client=client, config=config)
            except HindsightError as e:
                # Surface the failure so it shows up in Continue's warnings rather
                # than silently returning no memory.
                self._write_json(502, {"error": str(e)})
                return
            except Exception as e:  # pragma: no cover - defensive
                logger.exception("Unexpected error handling context request")
                self._write_json(500, {"error": f"internal error: {e}"})
                return

            self._write_json(200, serialize(items))

    return _Handler


def build_server(
    config: Optional[HindsightContinueConfig] = None, client: Optional[Hindsight] = None
) -> ThreadingHTTPServer:
    """Create (but do not start) the adapter HTTP server."""
    config = config or get_config()
    client = client or resolve_client()
    handler = make_handler(client, config)
    return ThreadingHTTPServer((config.host, config.port), handler)


def run(config: Optional[HindsightContinueConfig] = None) -> None:
    """Create and serve the adapter forever."""
    config = config or get_config()
    server = build_server(config)
    logger.info(
        "hindsight-continue context provider listening on http://%s:%s (bank=%s, api=%s)",
        config.host,
        config.port,
        config.bank_id or "<per-request>",
        config.hindsight_api_url,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        pass
    finally:
        server.server_close()
