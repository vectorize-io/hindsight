"""Localhost HTTP server for the control center.

A tiny stdlib ``http.server`` (no web framework, no Node) that serves the
single-file wizard SPA and a small JSON API over ``service.py``. Bound to
127.0.0.1 only and gated by the control token: every ``/api/*`` route except
``/api/health`` requires the ``X-Hindsight-Control-Token`` header to match.

Requiring a custom header (which browsers can't set cross-origin without a CORS
preflight the server never grants) is the CSRF defense — a malicious page can't
forge a daemon-stopping request even though we're on localhost.
"""

import argparse
import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import lifecycle, providers, service

_STATIC_DIR = Path(__file__).parent / "static"


class ControlCenterHandler(BaseHTTPRequestHandler):
    """Routes control-center requests; token-gated on /api/* (except health)."""

    # Set by serve() before the server starts.
    token: str = ""
    version: str = "unknown"

    server_version = "HindsightControlCenter"

    # Quieter logs — default BaseHTTPRequestHandler spams stderr per request.
    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return

    # ----- helpers -------------------------------------------------------
    def _send_json(self, status: int, payload: object) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        # Never cache API responses or the UI — this is a live control surface
        # and the SPA is iterated frequently; stale caches read as "broken".
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, status: int, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _token_ok(self) -> bool:
        provided = self.headers.get("X-Hindsight-Control-Token", "")
        return bool(self.token) and secrets.compare_digest(provided, self.token)

    @staticmethod
    def _segments(path: str) -> list[str]:
        return [s for s in urlparse(path).path.split("/") if s]

    # ----- dispatch ------------------------------------------------------
    def do_GET(self) -> None:
        segs = self._segments(self.path)

        if not segs or segs == ["index.html"]:
            self._serve_index()
            return

        # Static brand assets (logo, favicon, self-hosted fonts, css) — served
        # unauthenticated; they carry no secrets.
        if segs[0] != "api" and self._try_static(segs):
            return

        if segs[:2] == ["api", "health"]:
            self._send_json(200, {"status": "ok", "version": self.version})
            return

        if segs and segs[0] == "api":
            if not self._token_ok():
                self._send_json(401, {"error": "invalid or missing control token"})
                return
            try:
                self._route_get_api(segs)
            except Exception as exc:  # surface as JSON, never a stack-trace page
                self._send_json(500, {"error": str(exc)})
            return

        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        segs = self._segments(self.path)
        if not segs or segs[0] != "api":
            self._send_json(404, {"error": "not found"})
            return
        if not self._token_ok():
            self._send_json(401, {"error": "invalid or missing control token"})
            return
        try:
            self._route_post_api(segs, self._read_body())
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    # ----- routes --------------------------------------------------------
    def _route_get_api(self, segs: list[str]) -> None:
        if segs == ["api", "providers"]:
            self._send_json(200, {"providers": [service.to_json(p) for p in providers.PROVIDER_CATALOG]})
            return
        if segs == ["api", "profiles"]:
            self._send_json(200, {"profiles": [service.to_json(p) for p in service.list_profiles()]})
            return
        if len(segs) == 4 and segs[:2] == ["api", "profiles"]:
            name, resource = segs[2], segs[3]
            getters = {
                "config": lambda: service.get_profile_config(name),
                "env": lambda: service.read_env_file(name),
                "paths": lambda: service.get_profile_paths(name),
                "daemon": lambda: service.daemon_status(name),
                "ui": lambda: service.ui_status(name),
                "health": lambda: service.health(name),
                "logs": lambda: service.tail_log(
                    name, self._int_query("lines", 200), self._str_query("source", "daemon")
                ),
            }
            getter = getters.get(resource)
            if getter is not None:
                self._send_json(200, service.to_json(getter()))
                return
        self._send_json(404, {"error": "not found"})

    def _route_post_api(self, segs: list[str], body: dict) -> None:
        if len(segs) == 4 and segs[:2] == ["api", "profiles"] and segs[3] == "config":
            result = service.save_llm_config(
                name=segs[2],
                provider=body.get("provider", ""),
                api_key=body.get("api_key"),
                model=body.get("model"),
                base_url=body.get("base_url"),
                api_port=body.get("api_port"),
                ui_port=body.get("ui_port"),
                api_version=body.get("api_version"),
                cp_version=body.get("cp_version"),
            )
            self._send_json(200, service.to_json(result))
            return

        if len(segs) == 4 and segs[:2] == ["api", "profiles"] and segs[3] == "env":
            self._send_json(200, service.to_json(service.write_env_file(segs[2], body.get("content", ""))))
            return

        if len(segs) == 4 and segs[:2] == ["api", "profiles"] and segs[3] == "delete":
            self._send_json(200, service.to_json(service.delete_profile(segs[2])))
            return

        if len(segs) == 5 and segs[:2] == ["api", "profiles"] and segs[3] == "daemon":
            actions = {
                "start": service.start_daemon,
                "stop": service.stop_daemon,
                "restart": service.restart_daemon,
            }
            handler = actions.get(segs[4])
            if handler is None:
                self._send_json(404, {"error": f"unknown daemon action '{segs[4]}'"})
                return
            self._send_json(200, service.to_json(handler(segs[2])))
            return

        if len(segs) == 5 and segs[:2] == ["api", "profiles"] and segs[3] == "ui":
            ui_actions = {"start": service.start_ui, "stop": service.stop_ui, "restart": service.restart_ui}
            handler = ui_actions.get(segs[4])
            if handler is None:
                self._send_json(404, {"error": f"unknown ui action '{segs[4]}'"})
                return
            self._send_json(200, service.to_json(handler(segs[2])))
            return

        self._send_json(404, {"error": "not found"})

    def _int_query(self, key: str, default: int) -> int:
        """Read an integer query-string parameter, falling back on bad input."""
        values = parse_qs(urlparse(self.path).query).get(key)
        if not values:
            return default
        try:
            return int(values[0])
        except ValueError:
            return default

    def _str_query(self, key: str, default: str) -> str:
        """Read a string query-string parameter, falling back on absence."""
        values = parse_qs(urlparse(self.path).query).get(key)
        return values[0] if values else default

    def _serve_index(self) -> None:
        index = _STATIC_DIR / "index.html"
        if not index.exists():
            self._send_html(500, "<h1>Control center UI not found</h1>")
            return
        self._send_html(200, index.read_text())

    _CONTENT_TYPES = {
        ".png": "image/png",
        ".ico": "image/x-icon",
        ".css": "text/css; charset=utf-8",
        ".woff2": "font/woff2",
        ".js": "text/javascript; charset=utf-8",
        ".svg": "image/svg+xml",
    }

    def _try_static(self, segs: list[str]) -> bool:
        """Serve a whitelisted static file under static/. Returns True if handled."""
        rel = "/".join(segs)
        target = (_STATIC_DIR / rel).resolve()
        # Path-traversal guard: the resolved path must stay inside static/.
        if _STATIC_DIR.resolve() not in target.parents or not target.is_file():
            return False
        content_type = self._CONTENT_TYPES.get(target.suffix)
        if content_type is None:
            return False
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        return True


def serve(port: int) -> None:
    """Run the control-center HTTP server (blocking)."""
    from .. import __version__

    ControlCenterHandler.token = lifecycle.get_or_create_token()
    ControlCenterHandler.version = __version__
    # Record our own pid so `control stop` can find us even if the launcher's
    # Popen handle is gone.
    import os

    lifecycle.pid_file().parent.mkdir(parents=True, exist_ok=True)
    lifecycle.pid_file().write_text(str(os.getpid()))

    httpd = ThreadingHTTPServer(("127.0.0.1", port), ControlCenterHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(prog="hindsight-embed control-center server")
    parser.add_argument("--port", type=int, default=lifecycle.resolve_control_port())
    args = parser.parse_args()
    serve(args.port)


if __name__ == "__main__":
    main()
