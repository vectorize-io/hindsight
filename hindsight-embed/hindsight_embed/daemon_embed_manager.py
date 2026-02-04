"""
Concrete implementation of EmbedManager using daemon-based architecture.

This module provides the production implementation of the embed management interface,
consolidating daemon lifecycle, profile management, and database URL resolution.
"""

import logging
import os
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
from .profile_manager import ProfileManager, resolve_active_profile

logger = logging.getLogger(__name__)
console = Console(stderr=True)

# Suppress noisy httpx logs
logging.getLogger("httpx").setLevel(logging.WARNING)

# Constants
DAEMON_STARTUP_TIMEOUT = 180  # seconds
DEFAULT_DAEMON_IDLE_TIMEOUT = 300  # 5 minutes


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
        dev_api_path = Path(__file__).parent.parent.parent / "hindsight-api"
        if dev_api_path.exists() and (dev_api_path / "pyproject.toml").exists():
            return ["uv", "run", "--project", str(dev_api_path), "hindsight-api"]

        # Fall back to uvx for installed version
        from . import __version__

        api_version = os.getenv("HINDSIGHT_EMBED_API_VERSION", __version__)
        return ["uvx", f"hindsight-api@{api_version}"]

    def _start_daemon(self, config: dict, profile: str) -> bool:
        """Start the daemon in background."""
        paths = self._profile_manager.resolve_profile_paths(profile)
        profile_label = f"profile '{profile}'" if profile else "default profile"
        daemon_log = paths.log
        port = paths.port

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

        # On macOS, force CPU for embeddings/reranker to avoid MPS issues
        import platform

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

        try:
            # Start daemon
            subprocess.Popen(
                cmd,
                env=env,
                start_new_session=True,
            )

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

    def ensure_running(self, config: dict, profile: str) -> bool:
        """
        Ensure daemon is running, starting it if needed.

        Args:
            config: Environment configuration dict (HINDSIGHT_API_* vars)
            profile: Profile name for isolation

        Returns:
            True if daemon is running (started or already running), False on failure
        """
        if self.is_running(profile):
            logger.debug(f"Daemon already running for profile '{profile}'")
            return True
        return self._start_daemon(config, profile)

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

        # Find PID by port
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{port}", "-sTCP:LISTEN"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                pid = int(result.stdout.strip().split()[0])
                logger.debug(f"Found daemon PID {pid} on port {port}")

                # Send SIGTERM
                os.kill(pid, 15)

                # Wait for process to exit
                for _ in range(50):
                    time.sleep(0.1)
                    try:
                        os.kill(pid, 0)
                    except OSError:
                        break
            else:
                logger.warning(f"Could not find PID for port {port}")
        except (subprocess.TimeoutExpired, ValueError, OSError, FileNotFoundError) as e:
            logger.warning(f"Could not find/kill daemon by port: {e}")

        # Wait for health check to fail
        for _ in range(30):
            if not self.is_running(profile):
                return True
            time.sleep(0.1)

        return not self.is_running(profile)
