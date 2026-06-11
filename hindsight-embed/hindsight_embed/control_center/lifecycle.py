"""Control-center process lifecycle, port resolution, and access token.

The control center runs as its own detached, long-lived process (separate from
the daemon, which it supervises via lock-files). Stopping/restarting it never
affects a running daemon. Access is gated by a local token persisted to
``~/.hindsight/control.token`` so a stray browser tab can't drive the daemon.
"""

import os
import secrets
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

ENV_CONTROL_PORT = "HINDSIGHT_EMBED_CONTROL_PORT"
CONTROL_PORT_DEFAULT = 7878

# How long the launcher waits for the spawned server to answer /api/health.
_STARTUP_TIMEOUT_SECONDS = 20.0
_HEALTH_INTERVAL_SECONDS = 0.25


def _control_dir() -> Path:
    """Resolve ~/.hindsight dynamically (supports tests with a temp HOME)."""
    return Path.home() / ".hindsight"


def token_file() -> Path:
    return _control_dir() / "control.token"


def pid_file() -> Path:
    return _control_dir() / "control.pid"


def log_file() -> Path:
    return _control_dir() / "control.log"


def resolve_control_port() -> int:
    """Control-center port from env, falling back to the default."""
    raw = os.getenv(ENV_CONTROL_PORT)
    if not raw:
        return CONTROL_PORT_DEFAULT
    try:
        port = int(raw)
    except ValueError:
        return CONTROL_PORT_DEFAULT
    return port if 1024 <= port <= 65535 else CONTROL_PORT_DEFAULT


def read_token() -> str | None:
    """Read the persisted control token, or None if not set up yet."""
    path = token_file()
    if not path.exists():
        return None
    token = path.read_text().strip()
    return token or None


def get_or_create_token() -> str:
    """Return the control token, generating and persisting one if absent."""
    existing = read_token()
    if existing:
        return existing
    token = secrets.token_urlsafe(32)
    path = token_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token)
    path.chmod(0o600)
    return token


def build_url(port: int, token: str | None = None) -> str:
    """The localhost URL for the control center, with the token if available.

    We present `localhost` (not 127.0.0.1) everywhere user-facing; the server
    still binds the 127.0.0.1 loopback, which localhost resolves to.
    """
    base = f"http://localhost:{port}"
    token = token if token is not None else read_token()
    return f"{base}/?token={token}" if token else base


@dataclass(frozen=True)
class ControlStatus:
    """Liveness of the control-center process."""

    running: bool
    port: int
    url: str | None


@dataclass(frozen=True)
class ControlStartResult:
    """Outcome of starting (or reusing) the control center."""

    ok: bool
    already_running: bool
    port: int
    url: str | None
    message: str


def _health_ok(port: int) -> bool:
    try:
        with httpx.Client(timeout=2.0) as client:
            resp = client.get(f"http://127.0.0.1:{port}/api/health")
            return resp.status_code == 200 and resp.json().get("status") == "ok"
    except Exception:
        return False


def control_status(port: int | None = None) -> ControlStatus:
    """Report whether the control center is up on the given/default port."""
    port = port if port is not None else resolve_control_port()
    running = _health_ok(port)
    return ControlStatus(running=running, port=port, url=build_url(port) if running else None)


def _kill_pid(pid: int) -> bool:
    """SIGTERM a pid and wait briefly for it to exit. True if it's gone."""
    import signal

    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return True  # already gone
    for _ in range(50):
        time.sleep(0.1)
        try:
            os.kill(pid, 0)
        except OSError:
            return True
    return False


def start_control_center(port: int | None = None, *, open_browser: bool = False) -> ControlStartResult:
    """Start the control center as a detached process (idempotent).

    If a healthy control center is already listening on the port, this reuses
    it. Otherwise it spawns the server detached, waits for it to become
    healthy, and (optionally) opens the browser at the tokenized URL.
    """
    import subprocess

    from ..daemon_embed_manager import _detach_popen_kwargs

    port = port if port is not None else resolve_control_port()

    # Ensure the token exists before spawning so parent and child agree on it.
    token = get_or_create_token()

    if _health_ok(port):
        url = build_url(port, token)
        if open_browser:
            _open_browser(url)
        return ControlStartResult(
            ok=True, already_running=True, port=port, url=url, message="Control center already running."
        )

    log = log_file()
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "hindsight_embed.control_center.server", "--port", str(port)]
    with open(log, "ab") as log_handle:
        proc = subprocess.Popen(cmd, **_detach_popen_kwargs(log_handle))
    pid_file().write_text(str(proc.pid))

    deadline = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if _health_ok(port):
            url = build_url(port, token)
            if open_browser:
                _open_browser(url)
            return ControlStartResult(
                ok=True, already_running=False, port=port, url=url, message="Control center started."
            )
        time.sleep(_HEALTH_INTERVAL_SECONDS)

    return ControlStartResult(
        ok=False,
        already_running=False,
        port=port,
        url=None,
        message=f"Control center did not become healthy within {int(_STARTUP_TIMEOUT_SECONDS)}s (see {log}).",
    )


def stop_control_center(port: int | None = None) -> bool:
    """Stop the control center via its pid file. True if it's no longer up."""
    port = port if port is not None else resolve_control_port()
    pid_path = pid_file()
    stopped = True
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text().strip())
        except ValueError:
            pid = None
        if pid is not None:
            stopped = _kill_pid(pid)
        pid_path.unlink(missing_ok=True)
    # Whatever the pid file said, success means nothing healthy remains.
    return stopped and not _health_ok(port)


def _open_browser(url: str) -> None:
    import webbrowser

    try:
        webbrowser.open(url)
    except Exception:
        # Headless or no browser configured — the caller still prints the URL.
        pass
