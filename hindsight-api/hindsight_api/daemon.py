"""
Daemon mode support for Hindsight API.

Provides idle timeout and lockfile management for running as a background daemon.
"""

import asyncio
import fcntl
import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Default daemon configuration
DEFAULT_DAEMON_PORT = 8889
DEFAULT_IDLE_TIMEOUT = 0  # 0 = no auto-exit (hindsight-embed passes its own timeout)
LOCKFILE_PATH = Path.home() / ".hindsight" / "daemon.lock"
DAEMON_LOG_PATH = Path.home() / ".hindsight" / "daemon.log"


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


class DaemonLock:
    """
    File-based lock to prevent multiple daemon instances.

    Uses fcntl.flock for atomic locking on Unix systems.
    """

    def __init__(self, lockfile: Path = LOCKFILE_PATH):
        self.lockfile = lockfile
        self._fd = None

    def acquire(self) -> bool:
        """
        Try to acquire the daemon lock.

        Returns True if lock acquired, False if another daemon is running.
        """
        self.lockfile.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._fd = open(self.lockfile, "w")
            fcntl.flock(self._fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Write PID for debugging
            self._fd.write(str(os.getpid()))
            self._fd.flush()
            return True
        except (IOError, OSError):
            # Lock is held by another process
            if self._fd:
                self._fd.close()
                self._fd = None
            return False

    def release(self):
        """Release the daemon lock."""
        if self._fd:
            try:
                fcntl.flock(self._fd.fileno(), fcntl.LOCK_UN)
                self._fd.close()
            except Exception:
                pass
            finally:
                self._fd = None
                # Remove lockfile
                try:
                    self.lockfile.unlink()
                except Exception:
                    pass

    def is_locked(self) -> bool:
        """Check if the lock is held by another process."""
        if not self.lockfile.exists():
            return False

        try:
            fd = open(self.lockfile, "r")
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            # We got the lock, so no one else has it
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
            fd.close()
            return False
        except (IOError, OSError):
            return True

    def get_pid(self) -> int | None:
        """Get the PID of the daemon holding the lock."""
        if not self.lockfile.exists():
            return None
        try:
            with open(self.lockfile, "r") as f:
                return int(f.read().strip())
        except (ValueError, IOError):
            return None


def daemonize():
    """
    Fork the current process into a background daemon.

    Uses double-fork technique to properly detach from terminal.
    """
    # First fork
    pid = os.fork()
    if pid > 0:
        # Parent exits
        sys.exit(0)

    # Create new session
    os.setsid()

    # Second fork to prevent zombie processes
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


def stop_daemon(port: int = DEFAULT_DAEMON_PORT) -> bool:
    """Stop a running daemon by sending SIGTERM to the process."""
    lock = DaemonLock()
    pid = lock.get_pid()

    if pid is None:
        return False

    try:
        import signal

        os.kill(pid, signal.SIGTERM)
        # Wait for process to exit
        for _ in range(50):  # Wait up to 5 seconds
            time.sleep(0.1)
            try:
                os.kill(pid, 0)  # Check if process exists
            except OSError:
                return True  # Process exited
        return False
    except OSError:
        return False
