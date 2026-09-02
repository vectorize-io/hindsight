"""
Concrete implementation of EmbedManager using daemon-based architecture.

This module provides the production implementation of the embed management interface,
consolidating daemon lifecycle, profile management, and database URL resolution.
"""

import logging
import math
import os
import platform
import re
import subprocess
import sys
import sysconfig
import time
from collections.abc import Mapping
from importlib.util import find_spec
from pathlib import Path
from typing import IO, Optional

import httpx
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from .embed_manager import EmbedManager
from .profile_manager import ProfileLockTimeout, ProfileManager, lock_file, unlock_file

logger = logging.getLogger(__name__)
console = Console(stderr=True)

# Suppress noisy httpx logs
logging.getLogger("httpx").setLevel(logging.WARNING)


def _parse_float_env(name: str, default: float) -> float:
    """Parse a float environment variable, falling back on invalid values."""
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _safe_non_negative_float(value: float, fallback: float) -> float:
    """Return a finite non-negative float, or fallback for invalid values."""
    return value if math.isfinite(value) and value >= 0 else fallback


def _safe_positive_float(value: float, fallback: float) -> float:
    """Return a finite positive float, or fallback for invalid values."""
    return value if math.isfinite(value) and value > 0 else fallback


def _parse_non_negative_int(value: str | None, default: int, name: str) -> int:
    """Parse a non-negative integer, falling back for invalid or negative values."""
    try:
        parsed = int(value) if value is not None else default
    except ValueError:
        logger.warning("Invalid %s=%r; using %d", name, value, default)
        return default
    if parsed < 0:
        logger.warning("Invalid %s=%r; using %d", name, value, default)
        return default
    return parsed


# Constants
# Allow CI/Windows to extend the startup budget — pg0-embedded's Windows wheel
# unpacks and runs initdb on first boot, which takes noticeably longer on cold
# runners than POSIX.
DAEMON_STARTUP_TIMEOUT = int(os.getenv("HINDSIGHT_EMBED_DAEMON_STARTUP_TIMEOUT", "180"))
DEFAULT_DAEMON_IDLE_TIMEOUT = 0  # 0 = disabled (no auto-exit)
ENV_DAEMON_LOG_MAX_BYTES = "HINDSIGHT_EMBED_DAEMON_LOG_MAX_BYTES"
ENV_DAEMON_LOG_BACKUP_COUNT = "HINDSIGHT_EMBED_DAEMON_LOG_BACKUP_COUNT"
DEFAULT_DAEMON_LOG_MAX_BYTES = 10 * 1024 * 1024
DEFAULT_DAEMON_LOG_BACKUP_COUNT = 3
# When another process is concurrently starting the daemon, the TCP port can be
# bound before /health returns 200. Give that warming daemon a short grace window
# before treating the listener as stale/foreign and attempting to reclaim it.
PORT_HEALTH_GRACE_TIMEOUT = _safe_non_negative_float(
    _parse_float_env("HINDSIGHT_EMBED_PORT_HEALTH_GRACE_TIMEOUT", 30.0),
    30.0,
)
PORT_HEALTH_CHECK_INTERVAL = _safe_positive_float(
    _parse_float_env("HINDSIGHT_EMBED_PORT_HEALTH_CHECK_INTERVAL", 0.5),
    0.5,
)
# Two probe budgets, split by what a wrong answer costs.
#
# The daemon serves /health on the same asyncio event loop that runs LLM calls
# and embeddings, so a slow provider call can stall the response for tens of
# seconds. Where a false negative makes us *kill* the listener — the reclaim
# decision in _port_health_ok — 2s was not enough to tell "busy" from "dead"
# (issue #3099), so that probe waits as long as the worker-side liveness
# threshold does.
HEALTH_PROBE_TIMEOUT = _safe_positive_float(
    _parse_float_env("HINDSIGHT_EMBED_HEALTH_PROBE_TIMEOUT", 10.0),
    10.0,
)
# Everywhere else the question is only "is it up?", asked on hot paths (every
# _ensure_started, profile delete, CLI status) and often on ports where nothing
# is listening at all. A false negative there is cheap — the caller re-runs
# ensure_running, which consults the long probe above before touching anything —
# so these stay short. They were briefly raised to HEALTH_PROBE_TIMEOUT, which
# pushed the control center's delete handler (one daemon probe + one UI probe
# per loopback family) past the 5s default client timeout on Windows.
LIVENESS_PROBE_TIMEOUT = 2.0
# Connecting is not the slow part for a busy daemon — being answered is. Cap
# connect separately so an unreachable address cannot spend the whole read
# budget, which is what turns two loopback families into double the wait.
PROBE_CONNECT_TIMEOUT = 1.0

# Both loopback families are probed: Next.js binds ::1 only when started with
# `--hostname localhost`, so an IPv4-only health check reported a perfectly
# healthy control plane as down (issue #3527).
LOOPBACK_HOSTS = ("127.0.0.1", "::1")

# Before signalling a PID we require its command line to identify it as one of
# ours. Selecting the victim purely by "who holds the port" killed unrelated
# services that happened to be on the same port (issue #3520).
DAEMON_PROCESS_MARKERS = ("hindsight-api", "hindsight_api")
# The control plane runs as `npx @vectorize-io/hindsight-control-plane` (or
# `node .../hindsight-control-plane/bin/cli.js` in the monorepo), but Next.js
# rewrites argv to "next-server (vX.Y.Z)" once it is serving, so the package
# name is not always still visible in the command line.
UI_PROCESS_MARKERS = ("hindsight-control-plane", "next-server", "next start")

# Deadline for the short read-only probes (netstat, ps, wmic, PowerShell) used to
# identify processes. Five seconds is ample for the POSIX ones, which read procfs
# or run `ps`.
_PROBE_TIMEOUT_S = 5.0

# Windows needs more. Its command-line lookup goes through WMI, and the wmic
# front-end is absent from current Windows images, so the fallback is a fresh
# PowerShell — whose startup alone can pass five seconds on a loaded machine
# before Get-CimInstance runs. Under the old shared deadline that surfaced as a
# daemon we own being classified as foreign, and as a flaky test-embed-windows.
_WINDOWS_PROBE_TIMEOUT_S = 30.0


def _probe_timeout(read: float) -> httpx.Timeout:
    """Timeout with a short connect and a caller-chosen read budget."""
    return httpx.Timeout(read, connect=min(read, PROBE_CONNECT_TIMEOUT))


def _detach_popen_kwargs(log_handle: IO[bytes]) -> dict:
    """Cross-platform kwargs to spawn a subprocess detached from the caller.

    On POSIX, `start_new_session=True` calls setsid(2) so the child
    survives the parent's terminal. On Windows there is no setsid: we use
    `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`, which also means the
    child has no console, so stdin/stdout/stderr MUST be redirected or any
    write from the child crashes with "handle is invalid".

    `log_handle` receives the child's stdout/stderr on both platforms so
    output never leaks into the parent's terminal (which would corrupt a
    TUI parent communicating over stdio).
    """
    if platform.system() == "Windows":
        # Windows-only constants; use getattr so type checkers (e.g. ty) running
        # on Linux don't flag the attribute access. They're guaranteed present
        # at runtime because of the platform.system() guard above.
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
        "stdout": log_handle,
        "stderr": log_handle,
    }


class DaemonEmbedManager(EmbedManager):
    """Production embed manager using daemon-based architecture with profile isolation."""

    def __init__(self):
        """Initialize the daemon embed manager."""
        self._profile_manager = ProfileManager()

    @staticmethod
    def _rotate_daemon_log(log_path: Path, max_bytes: int, backup_count: int) -> None:
        """Rotate a full daemon log before a new daemon opens it.

        Startup is serialized by the profile lock, and this runs only after
        any stale daemon has been stopped. That keeps child processes from
        writing to a renamed inode during rotation. Rotation only occurs at
        startup; an already-running daemon is left untouched.
        """
        if max_bytes == 0 or not log_path.exists() or log_path.stat().st_size < max_bytes:
            return

        if backup_count == 0:
            log_path.write_bytes(b"")
            return

        oldest = log_path.with_name(f"{log_path.name}.{backup_count}")
        oldest.unlink(missing_ok=True)
        for index in range(backup_count - 1, 0, -1):
            source = log_path.with_name(f"{log_path.name}.{index}")
            if source.exists():
                source.replace(log_path.with_name(f"{log_path.name}.{index + 1}"))
        log_path.replace(log_path.with_name(f"{log_path.name}.1"))

    def _sanitize_profile_name(self, profile: str | None) -> str:
        """Sanitize profile name for use in database names and file paths."""
        if profile is None:
            return "default"
        return re.sub(r"[^a-zA-Z0-9_-]", "-", profile)

    def get_database_url(self, profile: str, db_url: Optional[str] = None) -> str:
        """
        Get the database URL for this profile.

        Args:
            profile: Profile name
            db_url: Optional override database URL

        Returns:
            Database connection string
        """
        if db_url and db_url != "pg0":
            return db_url
        safe_profile = self._sanitize_profile_name(profile)
        return f"pg0://hindsight-embed-{safe_profile}"

    def get_url(self, profile: str) -> str:
        """
        Get the URL for the daemon serving this profile.

        Args:
            profile: Profile name

        Returns:
            URL string (e.g., "http://127.0.0.1:54321")

        Raises:
            RuntimeError: If daemon is not running
        """
        paths = self._profile_manager.resolve_profile_paths(profile)
        return f"http://127.0.0.1:{paths.port}"

    def is_running(self, profile: str) -> bool:
        """Check if daemon is running and responsive.

        Uses the short liveness budget, not HEALTH_PROBE_TIMEOUT: this runs on
        hot paths and a false negative only costs a re-run of ensure_running,
        which consults the long probe before deciding anything destructive.
        """
        daemon_url = self.get_url(profile)
        try:
            with httpx.Client(timeout=_probe_timeout(LIVENESS_PROBE_TIMEOUT)) as client:
                response = client.get(f"{daemon_url}/health")
                return response.status_code == 200
        except Exception:
            return False

    def _dev_api_command(self) -> list[str] | None:
        """Return the dev-mode launch command when running inside the monorepo."""
        dev_api_path = Path(__file__).parent.parent.parent / "hindsight-api-slim"
        if dev_api_path.exists() and (dev_api_path / "pyproject.toml").exists():
            return ["uv", "run", "--project", str(dev_api_path), "--extra", "all", "hindsight-api"]
        return None

    @staticmethod
    def _windows_gui_interpreter(preferred_dir: Path | None = None) -> str | None:
        """Path to the GUI-subsystem Python (pythonw.exe), or None.

        Returns None on non-Windows, or when pythonw.exe can't be located next
        to the preferred scripts directory or running interpreter.

        On Windows 11 with Windows Terminal as the default terminal app,
        spawning the console-subsystem (CUI, subsystem 3) hindsight-api.exe
        launcher makes ConPTY pop a visible terminal tab on daemon start, even
        with DETACHED_PROCESS / CREATE_NO_WINDOW (issue #1885). Launching the
        daemon through the GUI-subsystem (subsystem 2) pythonw.exe interpreter
        never allocates a console, so no window appears. Prefer the scripts dir
        that contains hindsight-api.exe because wrapper entry points can make
        sys.executable point at a different launcher directory (issue #2389).
        """
        if platform.system() != "Windows":
            return None
        if preferred_dir is not None:
            pythonw = preferred_dir / "pythonw.exe"
            if pythonw.exists():
                return str(pythonw)
        pythonw = Path(sys.executable).with_name("pythonw.exe")
        return str(pythonw) if pythonw.exists() else None

    def _component_version(self, profile: str, env_key: str) -> str:
        """Resolve a component version: profile .env override > env var > embed version.

        Lets the API and control-plane package versions be pinned per profile
        (e.g. from the control center), falling back to the process env var and
        finally the embed's own version so the stack stays in lockstep.
        """
        from . import __version__

        override = self._profile_manager.load_profile_config(profile).get(env_key)
        return override or os.getenv(env_key) or __version__

    @staticmethod
    def _needs_local_sentence_transformers(env: Mapping[str, str] | None = None) -> bool:
        """Return True when the daemon config will load the local ST providers."""
        env = env or os.environ
        embeddings_provider = env.get("HINDSIGHT_API_EMBEDDINGS_PROVIDER", "local")
        reranker_provider = env.get("HINDSIGHT_API_RERANKER_PROVIDER", "local")
        return embeddings_provider == "local" or reranker_provider == "local"

    @staticmethod
    def _can_use_installed_api_binary(env: Mapping[str, str] | None = None) -> bool:
        """Avoid slim sibling binaries when default local ML deps are unavailable."""
        if not DaemonEmbedManager._needs_local_sentence_transformers(env):
            return True
        if find_spec("sentence_transformers") is not None:
            return True
        logger.warning(
            "Installed hindsight-api binary is missing local ML dependencies; "
            "falling back to uvx hindsight-api for the local daemon. Set "
            "HINDSIGHT_API_EMBEDDINGS_PROVIDER and HINDSIGHT_API_RERANKER_PROVIDER "
            "to non-local providers to use a slim install."
        )
        return False

    def _find_api_command(self, api_version: str, env: Mapping[str, str] | None = None) -> list[str]:
        """Find the command to run hindsight-api (api_version used only for the uvx fallback)."""
        # Check if we're in development mode
        dev_command = self._dev_api_command()
        if dev_command is not None:
            return dev_command

        # Prefer a hindsight-api entry point installed alongside hindsight-embed.
        # Try two strategies:
        #
        # 1. sysconfig: resolves <venv>/bin or <venv>/Scripts for standard
        #    pip/venv installs (issue #1401).
        # 2. __file__-relative: resolves <target>/bin or <target>/Scripts for
        #    `pip install --target` layouts where sysconfig still points at the
        #    system/venv scripts dir (issue #1240).
        binary_name = "hindsight-api.exe" if platform.system() == "Windows" else "hindsight-api"

        scripts_dir = Path(sysconfig.get_path("scripts"))
        candidate = scripts_dir / binary_name
        if candidate.exists() and self._can_use_installed_api_binary(env):
            # The console exe lives in sys.executable's scripts dir, so
            # hindsight_api is importable by the GUI interpreter; prefer it on
            # Windows to avoid ConPTY popping a terminal tab (issue #1885).
            gui_python = self._windows_gui_interpreter(scripts_dir)
            if gui_python is not None:
                return [gui_python, "-m", "hindsight_api.main"]
            return [str(candidate)]

        # --target installs place binaries alongside site-packages contents.
        # The running interpreter usually can't import hindsight_api here (it's
        # under the --target dir, not on sys.path), so we can't substitute
        # pythonw.exe; run the console exe directly.
        package_root = Path(__file__).parent.parent
        for bin_dir in ("bin", "Scripts"):
            candidate = package_root / bin_dir / binary_name
            if candidate.exists():
                # A hindsight-api binary bundled into a --target layout is a
                # deliberate slim-bundle install; use it unconditionally. Falling
                # back to uvx here reintroduces issue #1240 (the Windows embed
                # smoke test enforces this). The #2676 "missing local ML deps ->
                # uvx" fallback intentionally applies only to the sysconfig-scripts
                # path above (standard venv installs), not to a --target bundle.
                return [str(candidate)]

        # Fall back to uvx for the installed version (resolved by the caller).
        return ["uvx", f"hindsight-api@{api_version}"]

    @staticmethod
    def _is_port_in_use(port: int) -> bool:
        """Check if a port is in use using a socket connection (cross-platform).

        Probes both loopback families: a server bound to ::1 only (issue #3527)
        is invisible to an IPv4-only connect, which would make the caller treat
        an occupied port as free.
        """
        import socket

        for host in LOOPBACK_HOSTS:
            family = socket.AF_INET6 if ":" in host else socket.AF_INET
            try:
                with socket.socket(family, socket.SOCK_STREAM) as sock:
                    sock.settimeout(1)
                    if sock.connect_ex((host, port)) == 0:
                        return True
            except OSError:
                # The family isn't available on this host (e.g. IPv6 disabled).
                continue
        return False

    @staticmethod
    def _run_probe(cmd: list[str], timeout: float = _PROBE_TIMEOUT_S) -> str | None:
        """Run a short read-only probe command, returning stdout or None."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (subprocess.TimeoutExpired, OSError):
            return None
        return result.stdout if result.returncode == 0 else None

    @staticmethod
    def _windows_listening_pids(port: int) -> list[int]:
        """PIDs listening on `port` according to netstat."""
        output = DaemonEmbedManager._run_probe(["netstat", "-ano", "-p", "TCP"])
        if output is None:
            return []
        pids: list[int] = []
        for line in output.splitlines():
            fields = line.split()
            # proto, local address, foreign address, state, pid
            if len(fields) < 5 or fields[-2] != "LISTENING":
                continue
            # Match on the local address' port only: the daemon binds 127.0.0.1
            # but the UI can bind 0.0.0.0 or [::1] (issue #3527), and an
            # IPv4-literal match would miss those listeners entirely.
            local_address = fields[1]
            if local_address.rsplit(":", 1)[-1] != str(port):
                continue
            try:
                pids.append(int(fields[-1]))
            except ValueError:
                continue
        return pids

    @staticmethod
    def _posix_listening_pids(port: int) -> list[int]:
        """PIDs listening on `port`, via lsof, falling back to ss.

        `lsof` is the default on macOS but is absent from minimal containers and
        several Linux distributions, where it used to leave the caller with no
        PID at all (issue #3517). `ss` (iproute2) covers those hosts.
        """
        output = DaemonEmbedManager._run_probe(["lsof", "-ti", f":{port}", "-sTCP:LISTEN"])
        pids: list[int] = []
        if output:
            for token in output.split():
                try:
                    pids.append(int(token))
                except ValueError:
                    continue
        if pids:
            return pids

        # Lines look like:
        #   LISTEN 0 4096 127.0.0.1:9177 0.0.0.0:* users:(("hindsight-api",pid=15774,fd=19))
        output = DaemonEmbedManager._run_probe(["ss", "-tlnp", f"sport = :{port}"])
        if output:
            for line in output.splitlines():
                if "users:" not in line:
                    continue
                pids.extend(int(match) for match in re.findall(r"pid=(\d+)", line))
        return pids

    @staticmethod
    def _listening_pids(port: int) -> list[int]:
        """All PIDs listening on a local port, de-duplicated, in discovery order."""
        if platform.system() == "Windows":
            found = DaemonEmbedManager._windows_listening_pids(port)
        else:
            found = DaemonEmbedManager._posix_listening_pids(port)
        return list(dict.fromkeys(found))

    @staticmethod
    def _process_command_line(pid: int) -> str | None:
        """The command line of `pid`, or None when it cannot be determined."""
        if platform.system() == "Windows":
            # Both Windows probes get a longer deadline than the default. wmic is
            # gone from current Windows images, so it usually fails instantly and
            # everything rests on PowerShell — whose cold start alone can exceed
            # five seconds on a loaded machine, before Get-CimInstance has asked
            # WMI anything. Timing out here is not harmless: the caller reads None
            # as "not one of our processes" (see _process_matches), so the daemon
            # manager refuses to stop a daemon it does in fact own.
            output = DaemonEmbedManager._run_probe(
                ["wmic", "process", "where", f"ProcessId={pid}", "get", "CommandLine", "/format:list"],
                timeout=_WINDOWS_PROBE_TIMEOUT_S,
            )
            if output is None:
                output = DaemonEmbedManager._run_probe(
                    [
                        "powershell",
                        "-NoProfile",
                        "-NonInteractive",
                        "-Command",
                        f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine",
                    ],
                    timeout=_WINDOWS_PROBE_TIMEOUT_S,
                )
            if output is None:
                return None
            text = output.replace("CommandLine=", " ").strip()
            return text or None

        proc_cmdline = Path(f"/proc/{pid}/cmdline")
        try:
            raw = proc_cmdline.read_bytes()
        except OSError:
            raw = None
        if raw:
            return raw.replace(b"\x00", b" ").decode("utf-8", "replace").strip()

        # macOS (and Linux hosts without procfs).
        output = DaemonEmbedManager._run_probe(["ps", "-p", str(pid), "-o", "args="])
        return output.strip() if output and output.strip() else None

    @staticmethod
    def _process_matches(pid: int, markers: tuple[str, ...]) -> bool:
        """True when `pid`'s command line identifies it as one of our processes."""
        cmdline = DaemonEmbedManager._process_command_line(pid)
        if cmdline is None:
            return False
        lowered = cmdline.lower()
        return any(marker in lowered for marker in markers)

    @staticmethod
    def _owned_pid_on_port(port: int, markers: tuple[str, ...], description: str) -> int | None:
        """PID listening on `port` that we can positively identify as ours.

        Returns None — refusing to signal anything — when the listener cannot be
        identified. Failing to reclaim a port is recoverable; SIGTERMing an
        unrelated service that merely holds the port is not (issue #3520).
        """
        pids = DaemonEmbedManager._listening_pids(port)
        if not pids:
            logger.warning(f"Could not find PID for port {port}")
            return None
        for pid in pids:
            if DaemonEmbedManager._process_matches(pid, markers):
                return pid
        logger.warning(
            f"Port {port} is held by PID(s) {', '.join(str(p) for p in pids)}, which do not look like "
            f"a {description}; refusing to signal them"
        )
        return None

    @staticmethod
    def _kill_process(pid: int) -> bool:
        """Kill a process by PID and wait for it to exit. Returns True if process is gone."""
        import signal

        try:
            os.kill(pid, signal.SIGTERM)
            for _ in range(50):
                time.sleep(0.1)
                try:
                    os.kill(pid, 0)
                except OSError:
                    return True
        except OSError:
            return True  # Already gone
        return False

    @staticmethod
    def _port_health_ok(port: int) -> bool:
        """Return True when the listener on port responds like initialized Hindsight."""
        try:
            with httpx.Client(timeout=_probe_timeout(HEALTH_PROBE_TIMEOUT)) as client:
                response = client.get(f"http://127.0.0.1:{port}/health")
                if response.status_code != 200:
                    return False
                try:
                    health = response.json()
                except Exception:
                    return False
                return health.get("status") == "healthy" and health.get("database") == "connected"
        except Exception:
            return False

    def _wait_for_port_health(self, port: int, timeout: float | None = None) -> bool:
        """Wait briefly for a just-bound daemon port to become healthy.

        `timeout` bounds when the last probe may *start*, not when it returns, so
        a listener that accepts and hangs can push the wall clock to roughly
        timeout + HEALTH_PROBE_TIMEOUT. That slack is deliberate: it is what lets
        a daemon stalled on a slow LLM call answer before we call it stale.
        """
        timeout = _safe_non_negative_float(
            PORT_HEALTH_GRACE_TIMEOUT if timeout is None else timeout,
            0.0,
        )
        interval = _safe_positive_float(PORT_HEALTH_CHECK_INTERVAL, 0.5)
        deadline = time.monotonic() + timeout
        while True:
            if not self._is_port_in_use(port):
                return False
            if self._port_health_ok(port):
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(interval, remaining))

    def _clear_port(self, port: int) -> bool:
        """
        Ensure the port is free before starting a daemon.

        Behavior:
          * Port free → True (nothing to do).
          * Port occupied by a *healthy* hindsight daemon → True without killing.
            The caller's "start" is effectively a no-op: the daemon is already up.
            Killing it would race concurrent starts (one process kills the other's
            freshly-started daemon, both rush to rebind the port).
          * Port occupied but /health is unreachable, non-200, or does not return
            Hindsight's initialized health payload → treat as a stale daemon and
            attempt to reclaim it, but only after the listener's command line
            confirms it is a Hindsight daemon. This preserves the original intent
            of clearing stale daemons from version upgrades without SIGTERMing a
            foreign service that merely holds the port (issue #3520).
          * Kill failed, or unidentifiable/foreign process occupying the port → False.
        """
        if not self._is_port_in_use(port):
            return True

        # Port is occupied — check if it's a healthy Hindsight daemon. In
        # concurrent startup races the socket can bind before /health returns
        # 200, so wait briefly before deciding the listener is stale/foreign.
        if self._wait_for_port_health(port):
            logger.debug(f"Port {port} already serving a healthy hindsight daemon; reusing it")
            return True

        # Unhealthy after grace window — attempt to reclaim by killing the listener.
        if not self._is_port_in_use(port):
            return True

        pid = self._owned_pid_on_port(port, DAEMON_PROCESS_MARKERS, "hindsight daemon")
        if pid is None:
            logger.warning(f"Port {port} is in use by another process")
            return False

        logger.info(f"Clearing unhealthy process on port {port} (PID {pid})")
        if self._kill_process(pid):
            logger.info(f"Stale process (PID {pid}) stopped")
            return True

        logger.warning(f"Process (PID {pid}) did not stop in time")
        return False

    def _start_daemon(self, config: dict, profile: str, extra_args: list[str] | None = None) -> bool:
        """Start the daemon in background.

        Serializes concurrent start attempts via an exclusive flock on the
        profile's lock file, so two processes calling `start()` at the same
        time cannot race into `_clear_port` and kill each other's daemons.
        Inside the lock we re-check `is_running()`; the second caller sees
        the first caller's daemon and short-circuits.

        The lock wait is bounded (see HINDSIGHT_EMBED_LOCK_TIMEOUT): a caller
        that cannot get in reports a startup failure instead of raising an
        opaque OS error (issue #3100).
        """
        paths = self._profile_manager.resolve_profile_paths(profile)
        paths.lock.parent.mkdir(parents=True, exist_ok=True)

        # Hold the per-profile start lock for the full startup sequence.
        # lock_file() retries until the lock is acquired or the wait budget
        # expires, on both Unix (flock) and Windows (msvcrt), so concurrent
        # callers serialize here.
        with open(paths.lock, "w") as lock_fd:
            try:
                lock_file(lock_fd)
            except ProfileLockTimeout as exc:
                logger.error(f"Cannot start daemon for profile '{profile}': {exc}")
                return False
            try:
                if self.is_running(profile):
                    logger.debug(f"Daemon for profile '{profile}' came up while waiting for start lock")
                    if profile:
                        self._register_profile(profile, paths.port, config)
                    return True
                return self._start_daemon_locked(config, profile, paths, extra_args=extra_args)
            finally:
                unlock_file(lock_fd)

    def _start_daemon_locked(
        self,
        config: dict,
        profile: str,
        paths,
        extra_args: list[str] | None = None,
    ) -> bool:
        """Perform the actual daemon startup. Caller must hold paths.lock."""
        profile_label = f"profile '{profile}'" if profile else "default profile"
        daemon_log = paths.log
        port = paths.port

        # Ensure port is free before starting (handles stale daemons from version upgrades)
        if not self._clear_port(port):
            logger.error(f"Cannot start daemon: port {port} is in use by a non-hindsight process")
            return False

        # _clear_port returns True without killing when a healthy hindsight daemon
        # already owns the port (started out-of-band, e.g. by another user or a
        # previous version upgrade). Re-check is_running so we don't spawn a
        # second daemon that would fail to bind.
        if self.is_running(profile):
            logger.debug(f"Daemon for profile '{profile}' already healthy; skipping spawn")
            if profile:
                self._register_profile(profile, port, config)
            return True

        # Load profile's .env file and merge with provided config
        # This fixes issue #305 where profile env vars were ignored
        profile_config = self._profile_manager.load_profile_config(profile)
        # Merge: profile config first, then override with explicitly provided config
        merged_config = {**profile_config, **config}
        config = merged_config

        # Build environment with LLM config
        # Support both formats: simple keys ("llm_api_key") and env var format ("HINDSIGHT_API_LLM_API_KEY")
        env = os.environ.copy()

        # Map of simple key -> env var key
        key_mapping = {
            "llm_api_key": "HINDSIGHT_API_LLM_API_KEY",
            "llm_provider": "HINDSIGHT_API_LLM_PROVIDER",
            "llm_model": "HINDSIGHT_API_LLM_MODEL",
            "llm_base_url": "HINDSIGHT_API_LLM_BASE_URL",
            "log_level": "HINDSIGHT_API_LOG_LEVEL",
            "idle_timeout": "HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT",
        }

        for simple_key, env_key in key_mapping.items():
            # Check both simple format and env var format
            value = config.get(simple_key) or config.get(env_key)
            if value:
                env[env_key] = str(value)

        # Propagate any other HINDSIGHT_* keys from the merged profile/explicit
        # config into the daemon env. Without this, arbitrary settings in the
        # profile's .env file (e.g. HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU,
        # HINDSIGHT_API_EMBEDDINGS_PROVIDER) are silently dropped because the
        # whitelist above only covers LLM/log/idle_timeout keys.
        for key, value in config.items():
            if key.startswith("HINDSIGHT_") and value is not None:
                env[key] = str(value)

        # Use profile-specific database (check config for override)
        db_override = config.get("HINDSIGHT_EMBED_API_DATABASE_URL") or env.get("HINDSIGHT_EMBED_API_DATABASE_URL")
        if db_override:
            env["HINDSIGHT_API_DATABASE_URL"] = db_override
        else:
            env["HINDSIGHT_API_DATABASE_URL"] = self.get_database_url(profile)

        database_url = env["HINDSIGHT_API_DATABASE_URL"]
        is_pg0 = database_url.startswith("pg0://")

        # Set defaults if not provided
        if "HINDSIGHT_API_LOG_LEVEL" not in env:
            env["HINDSIGHT_API_LOG_LEVEL"] = "info"
        if "HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT" not in env:
            env["HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT"] = str(DEFAULT_DAEMON_IDLE_TIMEOUT)

        # Tell the daemon child it was already launched in a detached session
        # (via our Popen below) so daemonize() skips the redundant re-exec.
        env["_HINDSIGHT_DAEMON_CHILD"] = "1"

        # Get idle timeout from env
        idle_timeout = int(env.get("HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT", str(DEFAULT_DAEMON_IDLE_TIMEOUT)))

        # Create log directory
        daemon_log.parent.mkdir(parents=True, exist_ok=True)
        max_bytes = _parse_non_negative_int(
            env.get(ENV_DAEMON_LOG_MAX_BYTES),
            DEFAULT_DAEMON_LOG_MAX_BYTES,
            ENV_DAEMON_LOG_MAX_BYTES,
        )
        backup_count = _parse_non_negative_int(
            env.get(ENV_DAEMON_LOG_BACKUP_COUNT),
            DEFAULT_DAEMON_LOG_BACKUP_COUNT,
            ENV_DAEMON_LOG_BACKUP_COUNT,
        )
        try:
            self._rotate_daemon_log(daemon_log, max_bytes=max_bytes, backup_count=backup_count)
        except OSError as exc:
            logger.warning("Could not rotate daemon log %s: %s", daemon_log, exc)
        env["HINDSIGHT_API_DAEMON_LOG"] = str(daemon_log)

        # Build command
        cmd = self._find_api_command(self._component_version(profile, "HINDSIGHT_EMBED_API_VERSION"), env=env) + [
            "--daemon",
            "--idle-timeout",
            str(idle_timeout),
            "--port",
            str(port),
        ]
        if extra_args:
            cmd.extend(extra_args)

        try:
            # Start daemon. Redirect stdout/stderr to the daemon log so that
            # any output from the subprocess (e.g. uvx download progress,
            # Python library init messages) does not leak into the parent
            # process terminal. Python's own logging (via
            # HINDSIGHT_API_DAEMON_LOG) will append to the same path.
            # Popen dups the fd into the child during spawn, so the parent
            # can close its handle as soon as Popen returns.
            with open(daemon_log, "ab") as daemon_log_handle:
                subprocess.Popen(cmd, env=env, **_detach_popen_kwargs(daemon_log_handle))

            # Wait for daemon to be ready with rich UI
            start_time = time.time()
            last_check_time = start_time
            last_log_position = 0
            log_lines = [f"Starting daemon for {profile_label}...", ""]

            title = f"[bold cyan]Starting Daemon[/bold cyan] [dim]({profile} @ :{port})[/dim]"

            with Live(console=console, auto_refresh=False) as live:
                content = Text("\n".join(log_lines), style="dim")
                panel = Panel(content, title=title, border_style="cyan", padding=(1, 2))
                live.update(panel)
                live.refresh()

                while time.time() - start_time < DAEMON_STARTUP_TIMEOUT:
                    # Tail daemon logs
                    if daemon_log.exists():
                        try:
                            with open(daemon_log, "r", encoding="utf-8", errors="replace") as f:
                                f.seek(last_log_position)
                                new_lines = f.readlines()
                                last_log_position = f.tell()
                                for line in new_lines:
                                    line = line.rstrip()
                                    if line:
                                        log_lines.append(line)
                                log_lines = log_lines[-4:]
                        except Exception:
                            pass

                    if self.is_running(profile):
                        log_lines.append("")
                        log_lines.append("✓ Daemon responding, verifying stability...")
                        content = Text("\n".join(log_lines), style="dim")
                        panel = Panel(content, title=title, border_style="cyan", padding=(1, 2))
                        live.update(panel)
                        live.refresh()

                        time.sleep(2)
                        if self.is_running(profile):
                            log_lines.append("✓ Daemon started successfully!")
                            log_lines.append("")
                            log_lines.append(f"Logs: {daemon_log}")

                            if is_pg0:
                                pg0_name = database_url.replace("pg0://", "")
                                pg0_path = Path.home() / ".pg0" / "instances" / pg0_name
                                log_lines.append(f"Database: {pg0_path}")

                            content = Text("\n".join(log_lines), style="dim")
                            success_title = (
                                f"[bold green]✓ Daemon Started[/bold green] [dim]({profile} @ :{port})[/dim]"
                            )
                            panel = Panel(content, title=success_title, border_style="green", padding=(1, 2))
                            live.update(panel)
                            live.refresh()
                            console.print()
                            # Register profile in metadata so CLI can discover it
                            if profile:
                                self._register_profile(profile, port, config)
                            return True
                        else:
                            log_lines.append("")
                            log_lines.append("✗ Daemon crashed during initialization")
                            content = Text("\n".join(log_lines), style="dim")
                            fail_title = f"[bold red]✗ Daemon Failed[/bold red] [dim]({profile} @ :{port})[/dim]"
                            panel = Panel(content, title=fail_title, border_style="red", padding=(1, 2))
                            live.update(panel)
                            live.refresh()
                            console.print()
                            break

                    # Periodic progress
                    if time.time() - last_check_time > 3:
                        elapsed = int(time.time() - start_time)
                        status_msg = f"⏳ Waiting for daemon... ({elapsed}s elapsed)"
                        if log_lines and log_lines[-1].startswith("⏳"):
                            log_lines[-1] = status_msg
                        else:
                            log_lines.append(status_msg)
                        last_check_time = time.time()

                    content = Text("\n".join(log_lines), style="dim")
                    panel = Panel(content, title=title, border_style="cyan", padding=(1, 2))
                    live.update(panel)
                    live.refresh()
                    time.sleep(0.5)

            # Timeout
            log_lines.append("")
            log_lines.append("✗ Daemon failed to start (timeout)")
            log_lines.append("")
            log_lines.append(f"See full log: {daemon_log}")
            content = Text("\n".join(log_lines), style="dim")
            timeout_title = f"[bold red]✗ Daemon Failed (Timeout)[/bold red] [dim]({profile} @ :{port})[/dim]"
            panel = Panel(content, title=timeout_title, border_style="red", padding=(1, 2))
            console.print(panel)
            console.print()
            return False

        except FileNotFoundError:
            error_msg = (
                f"Command not found: {cmd[0]}\nFull command: {' '.join(cmd)}\n\n"
                "Install hindsight-api with: pip install hindsight-api"
            )
            error_panel = Panel(
                Text(error_msg, style="red"),
                title="[bold red]✗ Command Not Found[/bold red]",
                border_style="red",
                padding=(1, 2),
            )
            console.print(error_panel)
            console.print()
            return False
        except Exception as e:
            error_msg = f"Failed to start daemon: {e}\n\nCommand: {' '.join(cmd)}\nLog file: {daemon_log}"
            error_panel = Panel(
                Text(error_msg, style="red"),
                title="[bold red]✗ Startup Error[/bold red]",
                border_style="red",
                padding=(1, 2),
            )
            console.print(error_panel)
            console.print()
            return False

    def _register_profile(self, profile: str, port: int, config: dict) -> None:
        """Register a named profile in metadata so it's discoverable by the CLI.

        Only saves HINDSIGHT_API_* config keys (not internal daemon keys).
        Silently ignores errors to avoid blocking daemon startup.
        """
        try:
            api_config = {k: v for k, v in config.items() if k.startswith("HINDSIGHT_API_")}
            if not api_config:
                return
            # Merge onto the existing .env so persisted keys (UI port, idle
            # timeout, etc.) survive — create_profile overwrites the file, so we
            # must carry the existing non-alias keys forward.
            existing = self._profile_manager.load_profile_config(profile)
            merged = {k: v for k, v in existing.items() if not k.islower()}
            merged.update(api_config)
            self._profile_manager.create_profile(profile, port, merged)
        except Exception as e:
            logger.debug(f"Failed to register profile '{profile}' in metadata: {e}")

    def _find_ui_command(self, cp_version: str) -> list[str]:
        """Find the command to run the control plane UI (cp_version used only for the npx fallback)."""
        import shutil

        # Check if we're in development mode (monorepo)
        dev_cp_path = Path(__file__).parent.parent.parent / "hindsight-control-plane"
        cli_js = dev_cp_path / "bin" / "cli.js"
        if cli_js.exists():
            return ["node", str(cli_js)]

        # Use npx to run the published control plane package (version resolved by the caller).
        # `npx` prompts before installing missing packages on first run unless `-y` is set.
        # The UI starts in the background with stdout/stderr redirected to a log file, so an
        # interactive prompt would be invisible to users and the health-check loop would time out.
        # On Windows, detached processes may not inherit the parent's PATH, so resolve the
        # absolute path to npx to avoid "Command not found" errors.
        npx_path = shutil.which("npx")
        if npx_path is None:
            # Fallback to bare command so the FileNotFoundError handler can report it cleanly
            return ["npx", "-y", f"@vectorize-io/hindsight-control-plane@{cp_version}"]
        return [npx_path, "-y", f"@vectorize-io/hindsight-control-plane@{cp_version}"]

    def get_ui_url(self, profile: str, ui_port: int | None = None) -> str:
        """Get the URL for the UI serving this profile (callers render it for display)."""
        if ui_port is None:
            paths = self._profile_manager.resolve_profile_paths(profile)
            ui_port = paths.ui_port
        return f"http://0.0.0.0:{ui_port}"

    def is_ui_running(self, profile: str, ui_port: int | None = None) -> bool:
        """Check if the UI is running and responsive."""
        return self._reachable_ui_host(profile, ui_port) is not None

    def _reachable_ui_host(self, profile: str, ui_port: int | None = None) -> str | None:
        """Loopback host on which the UI answers /api/health, or None.

        The UI is probed on every loopback family rather than on 127.0.0.1
        alone: Next.js started with `--hostname localhost` binds ::1 only, so
        the IPv4 probe refused the connection and `ui start`/`ui status`
        reported a healthy control plane as down (issue #3527).
        """
        if ui_port is None:
            ui_port = self._profile_manager.resolve_profile_paths(profile).ui_port
        for host in LOOPBACK_HOSTS:
            # IPv6 literals have to be bracketed in a URL authority.
            base = f"http://[{host}]:{ui_port}" if ":" in host else f"http://{host}:{ui_port}"
            try:
                with httpx.Client(timeout=_probe_timeout(LIVENESS_PROBE_TIMEOUT)) as client:
                    if client.get(f"{base}/api/health").status_code == 200:
                        return host
            except Exception:
                continue
        return None

    @staticmethod
    def _ui_port_file(paths) -> Path:
        """Path of the file recording the UI's actually-bound port (so stop/restart
        can find it even after the configured port changed). e.g. <name>.ui.port."""
        return paths.ui_log.with_suffix(".port")

    def _read_recorded_ui_port(self, paths) -> int | None:
        f = self._ui_port_file(paths)
        if not f.exists():
            return None
        try:
            return int(f.read_text().strip())
        except (ValueError, OSError):
            return None

    def _record_ui_port(self, paths, port: int) -> None:
        try:
            self._ui_port_file(paths).write_text(str(port))
        except OSError:
            pass

    def start_ui(self, profile: str, ui_port: int | None = None, hostname: str = "0.0.0.0") -> bool:
        """Start the control plane UI in background.

        Args:
            profile: Profile name.
            ui_port: Port for the UI. Defaults to daemon_port + 10000.
            hostname: Hostname to bind to. Defaults to 0.0.0.0.

        Returns:
            True if UI started successfully.
        """
        paths = self._profile_manager.resolve_profile_paths(profile)
        if ui_port is None:
            ui_port = paths.ui_port

        if self.is_ui_running(profile, ui_port):
            logger.debug(f"UI already running for profile '{profile}' on port {ui_port}")
            self._record_ui_port(paths, ui_port)
            return True

        profile_label = f"profile '{profile}'" if profile else "default profile"
        api_url = self.get_url(profile)
        ui_log = paths.ui_log

        # Build environment
        env = os.environ.copy()
        env["PORT"] = str(ui_port)
        env["HOSTNAME"] = hostname
        env["HINDSIGHT_CP_DATAPLANE_API_URL"] = api_url

        # Create log directory
        ui_log.parent.mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = self._find_ui_command(self._component_version(profile, "HINDSIGHT_EMBED_CP_VERSION")) + [
            "--port",
            str(ui_port),
            "--hostname",
            hostname,
            "--api-url",
            api_url,
        ]

        try:
            with open(ui_log, "wb") as log_file:
                subprocess.Popen(cmd, env=env, **_detach_popen_kwargs(log_file))

            # Wait for UI to be ready
            start_time = time.time()
            title = f"[bold cyan]Starting UI[/bold cyan] [dim]({profile or 'default'} @ :{ui_port})[/dim]"
            log_lines = [f"Starting UI for {profile_label}...", ""]

            with Live(console=console, auto_refresh=False) as live:
                content = Text("\n".join(log_lines), style="dim")
                panel = Panel(content, title=title, border_style="cyan", padding=(1, 2))
                live.update(panel)
                live.refresh()

                while time.time() - start_time < 30:
                    if self._reachable_ui_host(profile, ui_port) is not None:
                        self._record_ui_port(paths, ui_port)
                        # "localhost" rather than a loopback literal: the UI may
                        # have bound ::1 only (issue #3527), and name resolution
                        # covers both families for whoever opens the link.
                        log_lines.append(f"✓ UI started at http://localhost:{ui_port}")
                        log_lines.append(f"Logs: {ui_log}")
                        content = Text("\n".join(log_lines), style="dim")
                        success_title = (
                            f"[bold green]✓ UI Started[/bold green] [dim]({profile or 'default'} @ :{ui_port})[/dim]"
                        )
                        panel = Panel(content, title=success_title, border_style="green", padding=(1, 2))
                        live.update(panel)
                        live.refresh()
                        console.print()
                        return True

                    elapsed = int(time.time() - start_time)
                    status_msg = f"⏳ Waiting for UI... ({elapsed}s elapsed)"
                    if log_lines and log_lines[-1].startswith("⏳"):
                        log_lines[-1] = status_msg
                    else:
                        log_lines.append(status_msg)

                    content = Text("\n".join(log_lines), style="dim")
                    panel = Panel(content, title=title, border_style="cyan", padding=(1, 2))
                    live.update(panel)
                    live.refresh()
                    time.sleep(0.5)

            # Timeout
            console.print(
                Panel(
                    Text(f"UI failed to start (timeout)\n\nSee full log: {ui_log}", style="dim"),
                    title=f"[bold red]✗ UI Failed (Timeout)[/bold red] [dim](:{ui_port})[/dim]",
                    border_style="red",
                    padding=(1, 2),
                )
            )
            console.print()
            return False

        except FileNotFoundError:
            error_msg = (
                f"Command not found: {cmd[0]}\nFull command: {' '.join(cmd)}\n\nInstall Node.js and npx to run the UI."
            )
            console.print(
                Panel(
                    Text(error_msg, style="red"),
                    title="[bold red]✗ Command Not Found[/bold red]",
                    border_style="red",
                    padding=(1, 2),
                )
            )
            console.print()
            return False
        except Exception as e:
            error_msg = f"Failed to start UI: {e}\n\nCommand: {' '.join(cmd)}\nLog file: {ui_log}"
            console.print(
                Panel(
                    Text(error_msg, style="red"),
                    title="[bold red]✗ UI Startup Error[/bold red]",
                    border_style="red",
                    padding=(1, 2),
                )
            )
            console.print()
            return False

    def stop_ui(self, profile: str, ui_port: int | None = None) -> bool:
        """Stop the UI for this profile.

        Args:
            profile: Profile name.
            ui_port: Port the UI is running on. Defaults to daemon_port + 10000.

        Returns:
            True if stopped successfully.
        """
        paths = self._profile_manager.resolve_profile_paths(profile)
        configured = paths.ui_port if ui_port is None else ui_port

        # Kill the UI on both the configured port AND the actually-recorded port.
        # They differ when the UI port was changed while the old UI kept running
        # on the previous port — otherwise that old process is orphaned.
        recorded = self._read_recorded_ui_port(paths)
        targets = {configured}
        if recorded is not None:
            targets.add(recorded)

        for port in targets:
            pid = self._owned_pid_on_port(port, UI_PROCESS_MARKERS, "control plane UI")
            if pid is not None:
                logger.debug(f"Found UI PID {pid} on port {port}")
                self._kill_process(pid)

        self._ui_port_file(paths).unlink(missing_ok=True)

        # Wait until nothing on any target port is still listening.
        for _ in range(30):
            if not any(self._is_port_in_use(p) for p in targets):
                return True
            time.sleep(0.1)

        return not any(self._is_port_in_use(p) for p in targets)

    def ensure_running(self, config: dict, profile: str, extra_args: list[str] | None = None) -> bool:
        """
        Ensure daemon is running, starting it if needed.

        Args:
            config: Environment configuration dict (HINDSIGHT_API_* vars)
            profile: Profile name for isolation
            extra_args: Extra CLI arguments to pass to hindsight-api (e.g. ["--offline"])

        Returns:
            True if daemon is running (started or already running), False on failure
        """
        if self.is_running(profile):
            logger.debug(f"Daemon already running for profile '{profile}'")
            if profile:
                paths = self._profile_manager.resolve_profile_paths(profile)
                self._register_profile(profile, paths.port, config)
            return True
        return self._start_daemon(config, profile, extra_args=extra_args)

    def stop(self, profile: str) -> bool:
        """
        Stop the daemon for this profile.

        Args:
            profile: Profile name

        Returns:
            True if stopped successfully, False otherwise
        """
        paths = self._profile_manager.resolve_profile_paths(profile)
        port = paths.port

        # Every decision here is based on port occupancy, never on /health.
        # A daemon that is alive but busy fails the responsiveness probe
        # (issue #3169), so using it as the already-stopped guard made stop()
        # report success without sending any signal. Reclaiming the profile's
        # port from an unresponsive listener is the same policy _clear_port()
        # applies on the start path.
        if not self._is_port_in_use(port):
            logger.debug(f"Daemon not running for profile '{profile}'")
            return True

        pid = self._owned_pid_on_port(port, DAEMON_PROCESS_MARKERS, "hindsight daemon")
        if pid is None:
            logger.warning(f"Port {port} is bound but no hindsight daemon could be identified on it")
            return False

        logger.debug(f"Found daemon PID {pid} on port {port}")
        if not self._kill_process(pid):
            logger.warning(f"Daemon process (PID {pid}) did not stop in time")
            return False

        # The process is gone; wait for the listener to disappear so a
        # follow-up start doesn't race the closing socket.
        for _ in range(30):
            if not self._is_port_in_use(port):
                return True
            time.sleep(0.1)

        return not self._is_port_in_use(port)
