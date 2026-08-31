"""
Daemon mode support for Hindsight API.

Provides process detachment for running as a background daemon.

The daemon used to auto-exit after a configurable idle period.  That was
removed: idleness was measured as "time since the last request *started*", so a
long retain/reflect/consolidation call that outlived the timeout got SIGTERM'd
mid-flight (#3903).  A daemon now runs until it is stopped.  ``--idle-timeout``
is still accepted so existing launchers keep working, but it does nothing.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import IO

# Default daemon configuration
DEFAULT_DAEMON_PORT = 8888

# Allow override via environment variable for profile-specific logs
DAEMON_LOG_PATH = Path(os.getenv("HINDSIGHT_API_DAEMON_LOG", str(Path.home() / ".hindsight" / "daemon.log")))

# Internal env var: set by daemonize() in the re-exec'd child so the child
# skips re-exec and just redirects stdio.  Also set by hindsight-embed's
# DaemonEmbedManager so the daemon launched via Popen skips re-exec entirely
# (hindsight-embed's Popen already provides a clean, detached process).
ENV_DAEMON_CHILD = "_HINDSIGHT_DAEMON_CHILD"


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
