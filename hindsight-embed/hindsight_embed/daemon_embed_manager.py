"""
Concrete implementation of EmbedManager using daemon-based architecture.

This module provides the production implementation of the embed management interface,
consolidating daemon lifecycle, profile management, and database URL resolution.
"""

import logging
import os
import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

import httpx
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from .embed_manager import EmbedManager
from .profile_manager import UI_PORT_OFFSET, ProfileManager, lock_file, resolve_active_profile, unlock_file

logger = logging.getLogger(__name__)
console = Console(stderr=True)

# Suppress noisy httpx logs
logging.getLogger("httpx").setLevel(logging.WARNING)

# Constants
# Allow CI/Windows to extend the startup budget — pg0-embedded's Windows wheel
# unpacks and runs initdb on first boot, which takes noticeably longer on cold
# runners than POSIX.
DAEMON_STARTUP_TIMEOUT = int(os.getenv("HINDSIGHT_EMBED_DAEMON_STARTUP_TIMEOUT", "180"))
DEFAULT_DAEMON_IDLE_TIMEOUT = 0  # 0 = disabled (no auto-exit)


def _detach_popen_kwargs(log_handle) -> dict:
    """Cross-platform kwargs to spawn a subprocess detached from the caller.

    On POSIX, `start_new_session=True` calls setsid(2) so the child
    survives the parent's terminal. On Windows there is no setsid: we use
    `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`, which also means the
    child has no console, so stdin/stdout/stderr MUST be redirected or any
    write from the child crashes with "handle is invalid".

    If a `log_handle` is supplied, stdout/stderr are routed to it on both
    platforms (matches the existing UI-spawn behavior). If `None`, the
    POSIX child inherits the parent's fds (legacy daemon-spawn behavior).
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
    kwargs: dict = {"start_new_session": True}
    if log_handle is not None:
        kwargs["stdout"] = log_handle
        kwargs["stderr"] = log_handle
    return kwargs


class DaemonEmbedManager(EmbedManager):
    """Production embed manager using daemon-based architecture with profile isolation."""

    def __init__(self):
        """Initialize the daemon embed manager."""
        self._profile_manager = ProfileManager()

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
        """Check if daemon is running and responsive."""
        daemon_url = self.get_url(profile)
        try:
            with httpx.Client(timeout=2) as client:
                response = client.get(f"{daemon_url}/health")
                return response.status_code == 200
        except Exception:
            return False

    def _find_api_command(self) -> list[str]:
        """Find the command to run hindsight-api."""
        # Check if we're in development mode
        dev_api_path = Path(__file__).parent.parent.parent / "hindsight-api-slim"
        if dev_api_path.exists() and (dev_api_path / "pyproject.toml").exists():
            return ["uv", "run", "--project", str(dev_api_path), "--extra", "all", "hindsight-api"]

        # Prefer a hindsight-api entry point installed alongside hindsight-embed
        # (e.g. `uv pip install hindsight-all` or `--target`). Falling through
        # to uvx in that case downloads a standalone Python whose ABI won't
        # match the sibling site-packages' C extensions (issue #1240).
        package_root = Path(__file__).parent.parent
        binary_name = "hindsight-api.exe" if platform.system() == "Windows" else "hindsight-api"
        for bin_dir in ("bin", "Scripts"):
            candidate = package_root / bin_dir / binary_name
            if candidate.exists():
                return [str(candidate)]

        # Fall back to uvx for installed version
        from . import __version__

        api_version = os.getenv("HINDSIGHT_EMBED_API_VERSION", __version__)
        return ["uvx", f"hindsight-api@{api_version}"]

    @staticmethod
    def _is_port_in_use(port: int) -> bool:
        """Check if a port is in use using a socket connection (cross-platform)."""
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            return sock.connect_ex(("127.0.0.1", port)) == 0

    @staticmethod
    def _find_pid_on_port(port: int) -> int | None:
        """Find the PID of the process listening on a port."""
        import platform

        try:
            if platform.system() == "Windows":
                # Use netstat on Windows
                result = subprocess.run(
                    ["netstat", "-ano", "-p", "TCP"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    for line in result.stdout.splitlines():
                        if f"127.0.0.1:{port}" in line and "LISTENING" in line:
                            return int(line.strip().split()[-1])
            else:
                # Use lsof on macOS/Linux
                result = subprocess.run(
                    ["lsof", "-ti", f":{port}", "-sTCP:LISTEN"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return int(result.stdout.strip().split()[0])
        except (subprocess.TimeoutExpired, ValueError, OSError, FileNotFoundError):
            pass
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

    def _clear_port(self, port: int) -> bool:
        """
        Ensure the port is free before starting a daemon.

        Behavior:
          * Port free → True (nothing to do).
          * Port occupied by a *healthy* hindsight daemon → True without killing.
            The caller's "start" is effectively a no-op: the daemon is already up.
            Killing it would race concurrent starts (one process kills the other's
            freshly-started daemon, both rush to rebind the port).
          * Port occupied but /health is unreachable / non-200 → treat as a stale
            hindsight daemon (or foreign process) and attempt to reclaim by killing
            the PID listening on the port. This preserves the original intent of
            clearing stale daemons from version upgrades.
          * Kill failed, or non-hindsight process occupying the port → False.
        """
        if not self._is_port_in_use(port):
            return True

        # Port is occupied — check if it's a healthy hindsight daemon.
        health_ok = False
        try:
            with httpx.Client(timeout=2) as client:
                response = client.get(f"http://127.0.0.1:{port}/health")
                health_ok = response.status_code == 200
        except Exception:
            health_ok = False

        if health_ok:
            logger.debug(f"Port {port} already serving a healthy hindsight daemon; reusing it")
            return True

        # Unhealthy — attempt to reclaim by killing the listener.
        pid = self._find_pid_on_port(port)
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
        """
        paths = self._profile_manager.resolve_profile_paths(profile)
        paths.lock.parent.mkdir(parents=True, exist_ok=True)

        # Hold the per-profile start lock for the full startup sequence.
        # lock_file() blocks until the lock is acquired on Unix (flock) and
        # Windows (msvcrt), so concurrent callers serialize here.
        with open(paths.lock, "w") as lock_fd:
            lock_file(lock_fd)
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

        # On macOS, force CPU for local embeddings/reranker to avoid MPS/XPC
        # hangs during sentence-transformers init in daemon mode (issue #962).
        # Users can opt back into MPS by explicitly setting these to "0".
        if platform.system() == "Darwin":
            if "HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU" not in env:
                env["HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU"] = "1"
            if "HINDSIGHT_API_RERANKER_LOCAL_FORCE_CPU" not in env:
                env["HINDSIGHT_API_RERANKER_LOCAL_FORCE_CPU"] = "1"

        # Get idle timeout from env
        idle_timeout = int(env.get("HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT", str(DEFAULT_DAEMON_IDLE_TIMEOUT)))

        # Create log directory
        daemon_log.parent.mkdir(parents=True, exist_ok=True)
        env["HINDSIGHT_API_DAEMON_LOG"] = str(daemon_log)

        # Build command
        cmd = self._find_api_command() + [
            "--daemon",
            "--idle-timeout",
            str(idle_timeout),
            "--port",
            str(port),
        ]
        if extra_args:
            cmd.extend(extra_args)

        try:
            # Start daemon. On Windows we must redirect stdout/stderr before
            # the child Python runs; DETACHED_PROCESS leaves it without a
            # console so any print/log to stdio would raise. We open the
            # daemon log for append here and hand it to the child; Python's
            # own logging (via HINDSIGHT_API_DAEMON_LOG) will append to the
            # same path once logging initializes. On POSIX we keep the
            # existing inherit-fd behavior to avoid touching working code.
            if platform.system() == "Windows":
                daemon_log_handle = open(daemon_log, "ab")
                popen_kwargs = _detach_popen_kwargs(daemon_log_handle)
            else:
                daemon_log_handle = None
                popen_kwargs = _detach_popen_kwargs(None)
            subprocess.Popen(cmd, env=env, **popen_kwargs)

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
                            with open(daemon_log, "r") as f:
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

        except FileNotFoundError as e:
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
            self._profile_manager.create_profile(profile, port, api_config)
        except Exception as e:
            logger.debug(f"Failed to register profile '{profile}' in metadata: {e}")

    def _find_ui_command(self) -> list[str]:
        """Find the command to run the control plane UI."""
        # Check if we're in development mode (monorepo)
        dev_cp_path = Path(__file__).parent.parent.parent / "hindsight-control-plane"
        cli_js = dev_cp_path / "bin" / "cli.js"
        if cli_js.exists():
            return ["node", str(cli_js)]

        # Use npx to run the published control plane package
        from . import __version__

        cp_version = os.getenv("HINDSIGHT_EMBED_CP_VERSION", __version__)
        # `npx` prompts before installing missing packages on first run unless `-y` is set.
        # The UI starts in the background with stdout/stderr redirected to a log file, so an
        # interactive prompt would be invisible to users and the health-check loop would time out.
        return ["npx", "-y", f"@vectorize-io/hindsight-control-plane@{cp_version}"]

    def get_ui_url(self, profile: str, ui_port: int | None = None, hostname: str | None = None) -> str:
        """Get the URL for the UI serving this profile."""
        if ui_port is None:
            paths = self._profile_manager.resolve_profile_paths(profile)
            ui_port = paths.port + UI_PORT_OFFSET
        host = hostname or "0.0.0.0"
        return f"http://{host}:{ui_port}"

    def is_ui_running(self, profile: str, ui_port: int | None = None) -> bool:
        """Check if the UI is running and responsive."""
        # Always health-check on 127.0.0.1 regardless of bind hostname
        ui_url = self.get_ui_url(profile, ui_port, hostname="127.0.0.1")
        try:
            with httpx.Client(timeout=2) as client:
                response = client.get(f"{ui_url}/api/health")
                return response.status_code == 200
        except Exception:
            return False

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
            ui_port = paths.port + UI_PORT_OFFSET

        if self.is_ui_running(profile, ui_port):
            logger.debug(f"UI already running for profile '{profile}' on port {ui_port}")
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
        cmd = self._find_ui_command() + [
            "--port",
            str(ui_port),
            "--hostname",
            hostname,
            "--api-url",
            api_url,
        ]

        try:
            log_file = open(ui_log, "w")
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
                    if self.is_ui_running(profile, ui_port):
                        log_lines.append(f"✓ UI started at http://127.0.0.1:{ui_port}")
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
        if ui_port is None:
            ui_port = paths.port + UI_PORT_OFFSET

        if not self.is_ui_running(profile, ui_port):
            logger.debug(f"UI not running for profile '{profile}'")
            return True

        pid = self._find_pid_on_port(ui_port)
        if pid is not None:
            logger.debug(f"Found UI PID {pid} on port {ui_port}")
            self._kill_process(pid)
        else:
            logger.warning(f"Could not find PID for UI port {ui_port}")

        # Wait for health check to fail
        for _ in range(30):
            if not self.is_ui_running(profile, ui_port):
                return True
            time.sleep(0.1)

        return not self.is_ui_running(profile, ui_port)

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
        if not self.is_running(profile):
            logger.debug(f"Daemon not running for profile '{profile}'")
            return True

        # Get port
        paths = self._profile_manager.resolve_profile_paths(profile)
        port = paths.port

        pid = self._find_pid_on_port(port)
        if pid is not None:
            logger.debug(f"Found daemon PID {pid} on port {port}")
            self._kill_process(pid)
        else:
            logger.warning(f"Could not find PID for port {port}")

        # Wait for health check to fail
        for _ in range(30):
            if not self.is_running(profile):
                return True
            time.sleep(0.1)

        return not self.is_running(profile)
