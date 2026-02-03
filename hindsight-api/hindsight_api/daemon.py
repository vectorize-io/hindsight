"""
Daemon mode support for Hindsight API.

Provides idle timeout for running as a background daemon.
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Default daemon configuration
DEFAULT_DAEMON_PORT = 8888
DEFAULT_IDLE_TIMEOUT = 0  # 0 = no auto-exit (hindsight-embed passes its own timeout)

# Allow override via environment variable for profile-specific logs
DAEMON_LOG_PATH = Path(os.getenv("HINDSIGHT_API_DAEMON_LOG", str(Path.home() / ".hindsight" / "daemon.log")))


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


def daemonize():
    """
    Fork the current process into a background daemon.

    Uses double-fork technique to properly detach from terminal.
    """
    # First fork - detach from parent
    try:
        pid = os.fork()
        if pid > 0:
            sys.exit(0)
    except OSError as e:
        sys.stderr.write(f"fork #1 failed: {e}\n")
        sys.exit(1)

    # Decouple from parent environment
    os.chdir("/")
    os.setsid()
    os.umask(0)

    # Second fork - prevent zombie
    pid = os.fork()
    if pid > 0:
        sys.exit(0)

    # Redirect standard file descriptors to log file
    DAEMON_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    sys.stdout.flush()
    sys.stderr.flush()

    # Redirect stdin to /dev/null
    with open("/dev/null", "r") as devnull:
        os.dup2(devnull.fileno(), sys.stdin.fileno())

    # Redirect stdout/stderr to log file
    log_fd = open(DAEMON_LOG_PATH, "a")
    os.dup2(log_fd.fileno(), sys.stdout.fileno())
    os.dup2(log_fd.fileno(), sys.stderr.fileno())


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
