"""Parity tests for the container readiness probe.

Each case is pinned to what `curl -sf` (without -L) does for the same response,
because that is what `hindsight_api.http_probe` replaced in
`docker/standalone/start-all.sh`. The redirect cases are the point: the obvious
`urllib.request.urlopen` implementation follows redirects and fails on a 404
behind one, where curl reports success - so a healthy service that redirects
would be reported as "not ready".
"""

from __future__ import annotations

import base64
import http.server
import json
import socket
import subprocess
import sys
import threading
from collections.abc import Iterator

import pytest

from hindsight_api.http_probe import main, probe

_EXPECTED_AUTH = "Basic " + base64.b64encode(b"user:pa ss").decode()


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        if self.path == "/ok":
            self.send_response(204)
        elif self.path == "/redirect-to-ok":
            self.send_response(302)
            self.send_header("Location", "/ok")
        elif self.path == "/redirect-to-missing":
            self.send_response(302)
            self.send_header("Location", "/missing")
        elif self.path == "/server-error":
            self.send_response(500)
        elif self.path == "/query":
            self.send_response(204 if "expected=1" in self.path else 400)
        elif self.path.startswith("/query?"):
            self.send_response(204 if "expected=1" in self.path else 400)
        elif self.path == "/auth":
            got = self.headers.get("Authorization")
            self.send_response(204 if got == _EXPECTED_AUTH else 401)
        else:
            self.send_response(404)
        self.end_headers()

    def log_message(self, *_args: object) -> None:
        """Silence the default stderr access log."""


@pytest.fixture(scope="module")
def base_url() -> Iterator[str]:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture(scope="module")
def closed_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


@pytest.mark.parametrize(
    ("path", "expected", "why"),
    [
        ("/ok", True, "2xx succeeds"),
        ("/redirect-to-ok", True, "curl does not follow redirects; a 302 is a completed transfer"),
        (
            "/redirect-to-missing",
            True,
            "still a 302 to curl, which never sees the 404 behind it - urlopen would fail here",
        ),
        ("/missing", False, "curl -f fails on 4xx"),
        ("/server-error", False, "curl -f fails on 5xx"),
        ("/query?expected=1", True, "query string is preserved"),
        ("/query?expected=0", False, "query string is preserved"),
    ],
)
def test_matches_curl_sf(base_url: str, path: str, expected: bool, why: str) -> None:
    assert probe(f"{base_url}{path}", 5) is expected, why


def test_sends_url_credentials_as_basic_auth(base_url: str) -> None:
    host = base_url.removeprefix("http://")
    assert probe(f"http://user:pa%20ss@{host}/auth", 5) is True


def test_connection_refused_fails(closed_port: int) -> None:
    assert probe(f"http://127.0.0.1:{closed_port}/ok", 2) is False


def test_unsupported_scheme_fails() -> None:
    assert probe("ftp://127.0.0.1/ok", 2) is False


def test_main_exit_codes(base_url: str, closed_port: int) -> None:
    assert main([f"{base_url}/ok"]) == 0
    assert main([f"{base_url}/missing"]) == 1
    assert main([f"http://127.0.0.1:{closed_port}/ok", "2"]) == 1
    assert main([]) == 2
    assert main([f"{base_url}/ok", "not-a-number"]) == 2


# The probe must not drag the application in behind it. It lives inside
# hindsight_api, so this cannot assert "no hindsight_api" - it asserts the part
# that actually matters: no third-party package, and none of the engine, config
# or API surface. hindsight_api/__init__ is cheap by design (PEP 562 lazy
# attributes, see its docstring) and this test is what keeps the probe from
# being the thing that makes it expensive again.
_IMPORT_AUDIT = """
import json, sys

before = set(sys.modules)
import hindsight_api.http_probe  # noqa: F401
loaded = set(sys.modules) - before

third_party = {
    name.split(".")[0] for name in loaded
    if name.split(".")[0] not in sys.stdlib_module_names
    and not name.startswith("hindsight_api")
    and not name.split(".")[0].startswith("_sysconfigdata")
}
heavy = {
    name for name in loaded
    if name.startswith(("hindsight_api.engine", "hindsight_api.api", "hindsight_api.config"))
}
print(json.dumps({"third_party": sorted(third_party), "heavy": sorted(heavy)}))
"""


def test_imports_nothing_heavy() -> None:
    """Importing the probe must pull in no third-party package and none of the engine."""
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_AUDIT],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    loaded = json.loads(result.stdout)
    assert loaded["third_party"] == [], (
        f"the readiness probe must not import third-party packages; it pulled in {loaded['third_party']}"
    )
    assert loaded["heavy"] == [], (
        f"the readiness probe must not import the engine/API/config; it pulled in {loaded['heavy']}"
    )


def test_runnable_as_a_module() -> None:
    """`python -m hindsight_api.http_probe` is how start-all.sh invokes it."""
    result = subprocess.run(
        [sys.executable, "-m", "hindsight_api.http_probe"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 2
    assert "usage: python -m hindsight_api.http_probe" in result.stderr
