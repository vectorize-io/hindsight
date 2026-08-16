"""
Daemon mode support for Hindsight API.

Provides idle timeout for running as a background daemon.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import IO

logger = logging.getLogger(__name__)

# Default daemon configuration
DEFAULT_DAEMON_PORT = 8888
DEFAULT_IDLE_TIMEOUT = 0  # 0 = no auto-exit (hindsight-embed passes its own timeout)

# Allow override via environment variable for profile-specific logs
DAEMON_LOG_PATH = Path(os.getenv("HINDSIGHT_API_DAEMON_LOG", str(Path.home() / ".hindsight" / "daemon.log")))

# Internal env var: set by daemonize() in the re-exec'd child so the child
# skips re-exec and just redirects stdio.  Also set by hindsight-embed's
# DaemonEmbedManager so the daemon launched via Popen skips re-exec entirely
# (hindsight-embed's Popen already provides a clean, detached process).
ENV_DAEMON_CHILD = "_HINDSIGHT_DAEMON_CHILD"
ENV_DAEMON_PID_FILE = "HINDSIGHT_EMBED_DAEMON_PID_FILE"


def _process_birth_marker(pid: int) -> str | None:
    """Return an OS-reported process creation marker for a PID."""
    try:
        system = platform.system()
        if system == "Linux":
            stat = Path(f"/proc/{pid}/stat").read_text()
            # Field 2 (comm) is parenthesized and may contain spaces. Fields
            # after its final ')' start at field 3; starttime is field 22.
            fields = stat[stat.rfind(")") + 2 :].split()
            return f"linux:{fields[19]}" if len(fields) > 19 else None
        if system == "Windows":
            create_no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            result = subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    f"(Get-CimInstance Win32_Process -Filter 'ProcessId = {pid}').CreationDate",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=create_no_window,
            )
        else:
            result = subprocess.run(
                ["ps", "-o", "lstart=", "-p", str(pid)],
                capture_output=True,
                text=True,
                timeout=5,
            )
        marker = " ".join(result.stdout.split())
        return marker if result.returncode == 0 and marker else None
    except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
        return None


def _pid_receipt(pid: int) -> str | None:
    birth_marker = _process_birth_marker(pid)
    if birth_marker is None:
        return None
    return json.dumps({"version": 1, "pid": pid, "birth_marker": birth_marker}, separators=(",", ":"))


def write_daemon_pid_receipt() -> Path | None:
    """Write the daemon's PID and creation marker for its embed manager."""
    raw_path = os.environ.get(ENV_DAEMON_PID_FILE)
    if not raw_path:
        return None
    receipt = _pid_receipt(os.getpid())
    if receipt is None:
        logger.warning("Could not determine daemon process creation time; ownership receipt not written")
        return None
    pid_path = Path(raw_path)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(receipt)
    return pid_path


def remove_daemon_pid_receipt(pid_path: Path | None) -> None:
    """Remove our receipt without deleting one replaced by a newer daemon."""
    if pid_path is None:
        return
    try:
        if pid_path.read_text().strip() == _pid_receipt(os.getpid()):
            pid_path.unlink()
    except OSError:
        pass


class IdleTimeoutMiddleware:
    """ASGI middleware that tracks activity and exits after idle timeout."""

    def __init__(self, app, idle_timeout: int = DEFAULT_IDLE_TIMEOUT):
        self.app = app
        self.idle_timeout = idle_timeout
        self.last_activity = time.time()

    async def __call__(self, scope, receive, send):
        self.last_activity = time.time()
        await self.app(scope, receive, send)

    async def _check_idle(self):
        """Exit the daemon after the configured period without requests."""
        if self.idle_timeout <= 0:
            return

        while True:
            await asyncio.sleep(30)
            idle_time = time.time() - self.last_activity
            if idle_time > self.idle_timeout:
                logger.info(f"Idle timeout reached ({self.idle_timeout}s), shutting down daemon")
                await asyncio.sleep(1)
                import signal

                os.kill(os.getpid(), signal.SIGTERM)


def _detach_popen_kwargs(log_handle: IO[bytes]) -> dict:
    """Cross-platform kwargs to spawn a subprocess detached from the caller.

    On POSIX, ``start_new_session=True`` calls ``setsid(2)`` so the child
    survives the parent's terminal.  On Windows we use
    ``DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP``.

    ``log_handle`` receives the child's stdout/stderr so output never leaks
    into the parent's terminal.
    """
    if platform.system() == "Windows":
        detached_process = getattr(subprocess, "DETACHED_PROCESS", 0)
        create_new_process_group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        return {
            "creationflags": detached_process | create_new_process_group,
            "stdin": subprocess.DEVNULL,
            "stdout": log_handle,
            "stderr": subprocess.STDOUT,
            "close_fds": True,
        }
    return {
        "start_new_session": True,
        "stdin": subprocess.DEVNULL,
        "stdout": log_handle,
        "stderr": log_handle,
    }


def _redirect_stdio_to_log() -> None:
    """Redirect stdin/stdout/stderr to the daemon log file.

    Called in the daemon child process after re-exec.
    """
    DAEMON_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    sys.stdout.flush()
    sys.stderr.flush()

    with open(os.devnull, "r") as devnull:
        os.dup2(devnull.fileno(), sys.stdin.fileno())

    log_fd = open(DAEMON_LOG_PATH, "a")
    os.dup2(log_fd.fileno(), sys.stdout.fileno())
    os.dup2(log_fd.fileno(), sys.stderr.fileno())


def daemonize():
    """Detach the current process into a background daemon.

    Uses ``subprocess.Popen`` (which maps to ``posix_spawn`` on macOS) to
    re-exec the current command in a detached session.  This replaces the
    traditional double-fork pattern because ``os.fork()`` without ``exec()``
    corrupts Apple framework state (XPC, Metal/MPS, ObjC runtime) on macOS,
    causing SIGBUS crashes when PyTorch uses the MPS backend.

    The function has two code paths controlled by the ``_HINDSIGHT_DAEMON_CHILD``
    environment variable:

    * **Parent** (env var not set): re-exec the same command via Popen with
      ``start_new_session=True``, stripping ``--daemon`` from argv and setting
      ``_HINDSIGHT_DAEMON_CHILD=1``.  Then ``sys.exit(0)``.
    * **Child** (env var set): redirect stdio to the daemon log file and return.
      No fork, no re-exec.

    On Windows there is no fork model: the spawning parent is expected to
    detach us via ``CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS`` and to
    redirect stdout/stderr to ``HINDSIGHT_API_DAEMON_LOG`` before exec.
    We still ensure the log directory exists.
    """
    if sys.platform == "win32":
        DAEMON_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        return

    # If we are already the daemon child (re-exec'd by a previous daemonize()
    # call, or launched by hindsight-embed with the env var set), just redirect
    # stdio and return — no re-exec needed.
    if os.environ.get(ENV_DAEMON_CHILD) == "1":
        _redirect_stdio_to_log()
        return

    # --- Parent path: re-exec ourselves as a detached background process ---

    # Build child command: same Python, same module entry point, all args
    # except --daemon (replaced by the env var).
    child_args = [a for a in sys.argv[1:] if a != "--daemon"]
    cmd = [sys.executable, "-m", "hindsight_api.main"] + child_args

    env = os.environ.copy()
    env[ENV_DAEMON_CHILD] = "1"
    env["HINDSIGHT_API_DAEMON_LOG"] = str(DAEMON_LOG_PATH)

    DAEMON_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(DAEMON_LOG_PATH, "ab") as log_handle:
        subprocess.Popen(cmd, env=env, **_detach_popen_kwargs(log_handle))

    sys.exit(0)
