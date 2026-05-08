"""
Daemon mode support for Hindsight API.

Provides idle timeout for running as a background daemon.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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


class IdleTimeoutMiddleware:
    """ASGI middleware that tracks activity and exits after idle timeout."""

    def __init__(self, app, idle_timeout: int = DEFAULT_IDLE_TIMEOUT):
        self.app = app
        self.idle_timeout = idle_timeout
        self.last_activity = time.time()
        self._checker_task = None

    async def __call__(self, scope, receive, send):
        # Update activity timestamp on each request
        self.last_activity = time.time()
        await self.app(scope, receive, send)

    def start_idle_checker(self):
        """Start the background task that checks for idle timeout."""
        self._checker_task = asyncio.create_task(self._check_idle())

    async def _check_idle(self):
        """Background task that exits the process after idle timeout."""
        # If idle_timeout is 0, don't auto-exit
        if self.idle_timeout <= 0:
            return

        while True:
            await asyncio.sleep(30)  # Check every 30 seconds
            idle_time = time.time() - self.last_activity
            if idle_time > self.idle_timeout:
                logger.info(f"Idle timeout reached ({self.idle_timeout}s), shutting down daemon")
                # Give a moment for any in-flight requests
                await asyncio.sleep(1)
                # Send SIGTERM to ourselves to trigger graceful shutdown
                import signal

                os.kill(os.getpid(), signal.SIGTERM)


def _detach_popen_kwargs(log_handle: "IO[bytes]") -> dict:
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


def check_daemon_running(port: int = DEFAULT_DAEMON_PORT) -> bool:
    """Check if a daemon is running and responsive on the given port."""
    import socket

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(("127.0.0.1", port))
        sock.close()
        return result == 0
    except Exception:
        return False
