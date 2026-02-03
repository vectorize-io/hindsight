"""
Client for communicating with the Hindsight daemon.

Handles daemon lifecycle (start if needed) and API requests via the Python client.
"""

import logging
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

import httpx  # Used only for health check
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from .profile_manager import ProfileManager, resolve_active_profile

console = Console(stderr=True)

logger = logging.getLogger(__name__)

# Suppress noisy httpx logs
logging.getLogger("httpx").setLevel(logging.WARNING)

# Default port for default profile
DEFAULT_DAEMON_PORT = 8888
DAEMON_PORT = DEFAULT_DAEMON_PORT  # Backward compatibility
DAEMON_STARTUP_TIMEOUT = 180  # seconds - needs to be long for first run (downloads dependencies)
# Default idle timeout: 5 minutes - users can override with HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT env var
DEFAULT_DAEMON_IDLE_TIMEOUT = 300


def get_daemon_port(profile: str | None = None) -> int:
    """Get daemon port for a profile.

    Args:
        profile: Profile name (None = resolve from priority).

    Returns:
        Port number for daemon.
    """
    if profile is None:
        profile = resolve_active_profile()

    pm = ProfileManager()
    paths = pm.resolve_profile_paths(profile)
    return paths.port


def get_daemon_url(profile: str | None = None) -> str:
    """Get daemon URL for a profile.

    Args:
        profile: Profile name (None = resolve from priority).

    Returns:
        URL for daemon.
    """
    port = get_daemon_port(profile)
    return f"http://127.0.0.1:{port}"


# CLI paths - check multiple locations
CLI_INSTALL_DIRS = [
    Path.home() / ".local" / "bin",  # Standard location from get-cli installer
    Path.home() / ".hindsight" / "bin",  # Alternative location
]
CLI_INSTALLER_URL = "https://hindsight.vectorize.io/get-cli"


def _find_hindsight_api_command() -> list[str]:
    """Find the command to run hindsight-api."""
    # Check if we're in development mode (local hindsight-api available)
    # Path: daemon_client.py -> hindsight_embed/ -> hindsight-embed/ -> memory-poc/
    dev_api_path = Path(__file__).parent.parent.parent / "hindsight-api"
    if dev_api_path.exists() and (dev_api_path / "pyproject.toml").exists():
        # Use uv run with the local project
        return ["uv", "run", "--project", str(dev_api_path), "hindsight-api"]

    # Fall back to uvx for installed version
    # Allow version override via environment variable (defaults to matching embed version)
    from . import __version__

    api_version = os.getenv("HINDSIGHT_EMBED_API_VERSION", __version__)
    return ["uvx", f"hindsight-api@{api_version}"]


def _is_daemon_running(profile: str | None = None) -> bool:
    """Check if daemon is running and responsive.

    Args:
        profile: Profile name (None = resolve from priority).

    Returns:
        True if daemon is running and responsive.
    """
    daemon_url = get_daemon_url(profile)
    try:
        with httpx.Client(timeout=2) as client:
            response = client.get(f"{daemon_url}/health")
            return response.status_code == 200
    except Exception:
        return False


def _start_daemon(config: dict, profile: str | None = None) -> bool:
    """
    Start the daemon in background.

    Args:
        config: Configuration dict with LLM settings.
        profile: Profile name (None = resolve from priority).

    Returns:
        True if daemon started successfully.
    """
    import sys

    if profile is None:
        profile = resolve_active_profile()

    # Get profile-specific paths
    pm = ProfileManager()
    paths = pm.resolve_profile_paths(profile)

    profile_label = f"profile '{profile}'" if profile else "default profile"
    daemon_log = paths.log
    port = paths.port

    # Build environment with LLM config
    env = os.environ.copy()
    if config.get("llm_api_key"):
        env["HINDSIGHT_API_LLM_API_KEY"] = config["llm_api_key"]
    if config.get("llm_provider"):
        env["HINDSIGHT_API_LLM_PROVIDER"] = config["llm_provider"]
    if config.get("llm_model"):
        env["HINDSIGHT_API_LLM_MODEL"] = config["llm_model"]

    # Use profile-specific pg0 database for isolation
    # Allow override via HINDSIGHT_EMBED_API_DATABASE_URL for external PostgreSQL
    # (e.g. when running as root where embedded pg0 cannot use initdb)
    if "HINDSIGHT_EMBED_API_DATABASE_URL" not in env:
        # Sanitize profile name for use in database name (allow only alphanumeric, dash, underscore)
        safe_profile = re.sub(r"[^a-zA-Z0-9_-]", "-", profile or "default")
        env["HINDSIGHT_API_DATABASE_URL"] = f"pg0://hindsight-embed-{safe_profile}"
    else:
        # Pass through the embed-specific env var to the daemon as the standard API env var
        env["HINDSIGHT_API_DATABASE_URL"] = env["HINDSIGHT_EMBED_API_DATABASE_URL"]

    # Store database URL for display later
    database_url = env["HINDSIGHT_API_DATABASE_URL"]
    is_pg0 = database_url.startswith("pg0://")

    env["HINDSIGHT_API_LOG_LEVEL"] = "info"

    # On macOS, force CPU for embeddings/reranker to avoid MPS/Metal/XPC issues in daemon mode
    # Only set if not already configured by user
    import platform

    if platform.system() == "Darwin":
        if "HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU" not in env:
            env["HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU"] = "1"
        if "HINDSIGHT_API_RERANKER_LOCAL_FORCE_CPU" not in env:
            env["HINDSIGHT_API_RERANKER_LOCAL_FORCE_CPU"] = "1"

    # Get idle timeout from environment or use default
    idle_timeout = int(os.getenv("HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT", str(DEFAULT_DAEMON_IDLE_TIMEOUT)))

    # Use profile-specific log file
    daemon_log.parent.mkdir(parents=True, exist_ok=True)

    # Tell hindsight-api daemon where to write its logs
    env["HINDSIGHT_API_DAEMON_LOG"] = str(daemon_log)

    # Pass profile-specific port (no lockfile - we use port-based discovery)
    cmd = _find_hindsight_api_command() + [
        "--daemon",
        "--idle-timeout",
        str(idle_timeout),
        "--port",
        str(paths.port),
    ]

    try:
        # Start daemon directly (hindsight-api handles its own log redirection via HINDSIGHT_API_DAEMON_LOG)
        subprocess.Popen(
            cmd,
            env=env,
            start_new_session=True,
        )

        # Wait for daemon to be ready
        # Note: With --daemon flag, the parent process forks and exits immediately (code 0).
        # The child process (actual daemon) continues running. So we can't rely on process.poll()
        # to detect failures - we must use the health check.
        start_time = time.time()
        last_check_time = start_time
        last_log_position = 0  # Track position in log file for tailing
        log_lines = [f"Starting daemon for {profile_label}...", ""]  # Accumulate log lines for display

        # Build title with profile and port info
        if profile:
            title = f"[bold cyan]Starting Daemon[/bold cyan] [dim]({profile} @ :{port})[/dim]"
        else:
            title = f"[bold cyan]Starting Daemon[/bold cyan] [dim](:{port})[/dim]"

        # Use Rich Live display for beautiful real-time updates
        with Live(console=console, auto_refresh=False) as live:
            # Show initial panel
            content = Text("\n".join(log_lines), style="dim")
            panel = Panel(
                content,
                title=title,
                border_style="cyan",
                padding=(1, 2),
            )
            live.update(panel)
            live.refresh()

            while time.time() - start_time < DAEMON_STARTUP_TIMEOUT:
                # Tail daemon logs if available
                if daemon_log.exists():
                    try:
                        with open(daemon_log, "r") as f:
                            f.seek(last_log_position)
                            new_lines = f.readlines()
                            last_log_position = f.tell()

                            # Add new log lines (keep last 4 for display)
                            for line in new_lines:
                                line = line.rstrip()
                                if line:
                                    log_lines.append(line)
                            # Keep only last 4 lines
                            log_lines = log_lines[-4:]
                    except Exception:
                        pass  # Silently ignore log read errors

                if _is_daemon_running(profile):
                    # Health check passed - but daemon might crash during initialization
                    # Add status message to logs
                    log_lines.append("")
                    log_lines.append("✓ Daemon responding, verifying stability...")

                    # Update display with success status
                    content = Text("\n".join(log_lines), style="dim")
                    panel = Panel(
                        content,
                        title=title,
                        border_style="cyan",
                        padding=(1, 2),
                    )
                    live.update(panel)
                    live.refresh()

                    time.sleep(2)
                    if _is_daemon_running(profile):
                        log_lines.append("✓ Daemon started successfully!")
                        log_lines.append("")
                        log_lines.append(f"Logs: {daemon_log}")

                        # Show pg0 location if using pg0
                        if is_pg0:
                            # pg0 stores data in ~/.pg0/instances/<database_name>
                            pg0_name = database_url.replace("pg0://", "")
                            pg0_path = Path.home() / ".pg0" / "instances" / pg0_name
                            log_lines.append(f"Database: {pg0_path}")

                        content = Text("\n".join(log_lines), style="dim")

                        # Build success title with profile and port
                        if profile:
                            success_title = (
                                f"[bold green]✓ Daemon Started[/bold green] [dim]({profile} @ :{port})[/dim]"
                            )
                        else:
                            success_title = f"[bold green]✓ Daemon Started[/bold green] [dim](:{port})[/dim]"

                        panel = Panel(
                            content,
                            title=success_title,
                            border_style="green",
                            padding=(1, 2),
                        )
                        live.update(panel)
                        live.refresh()
                        console.print()  # Add newline after panel
                        return True
                    else:
                        # Daemon crashed after initial health check
                        log_lines.append("")
                        log_lines.append("✗ Daemon crashed during initialization")
                        content = Text("\n".join(log_lines), style="dim")

                        # Build failure title with profile and port
                        if profile:
                            fail_title = f"[bold red]✗ Daemon Failed[/bold red] [dim]({profile} @ :{port})[/dim]"
                        else:
                            fail_title = f"[bold red]✗ Daemon Failed[/bold red] [dim](:{port})[/dim]"

                        panel = Panel(
                            content,
                            title=fail_title,
                            border_style="red",
                            padding=(1, 2),
                        )
                        live.update(panel)
                        live.refresh()
                        console.print()
                        break

                # Periodically log progress
                if time.time() - last_check_time > 3:
                    elapsed = int(time.time() - start_time)
                    # Update last status line or add new one
                    status_msg = f"⏳ Waiting for daemon... ({elapsed}s elapsed)"
                    if log_lines and log_lines[-1].startswith("⏳"):
                        log_lines[-1] = status_msg
                    else:
                        log_lines.append(status_msg)
                    last_check_time = time.time()

                # Update the live display
                content = Text("\n".join(log_lines), style="dim")
                panel = Panel(
                    content,
                    title=title,
                    border_style="cyan",
                    padding=(1, 2),
                )
                live.update(panel)
                live.refresh()

                time.sleep(0.5)

        # Timeout - show failure
        log_lines.append("")
        log_lines.append("✗ Daemon failed to start (timeout)")
        log_lines.append("")
        log_lines.append(f"See full log: {daemon_log}")

        content = Text("\n".join(log_lines), style="dim")

        # Build timeout title with profile and port
        if profile:
            timeout_title = f"[bold red]✗ Daemon Failed (Timeout)[/bold red] [dim]({profile} @ :{port})[/dim]"
        else:
            timeout_title = f"[bold red]✗ Daemon Failed (Timeout)[/bold red] [dim](:{port})[/dim]"

        panel = Panel(
            content,
            title=timeout_title,
            border_style="red",
            padding=(1, 2),
        )
        console.print(panel)
        console.print()

        return False

    except FileNotFoundError as e:
        error_msg = f"Command not found: {cmd[0]}\nFull command: {' '.join(cmd)}\n\nInstall hindsight-api with: pip install hindsight-api"
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


def ensure_daemon_running(config: dict, profile: str | None = None) -> bool:
    """
    Ensure daemon is running, starting it if needed.

    Args:
        config: Configuration dict with LLM settings.
        profile: Profile name (None = resolve from priority).

    Returns:
        True if daemon is running.
    """
    if profile is None:
        profile = resolve_active_profile()

    if _is_daemon_running(profile):
        logger.debug(f"Daemon already running for profile '{profile or 'default'}'")
        return True

    return _start_daemon(config, profile)


def stop_daemon(profile: str | None = None) -> bool:
    """Stop the running daemon and wait for it to fully stop.

    Args:
        profile: Profile name (None = resolve from priority).

    Returns:
        True if daemon stopped successfully.
    """
    import subprocess

    if profile is None:
        profile = resolve_active_profile()

    # Check if daemon is actually running via health check
    if not _is_daemon_running(profile):
        logger.debug(f"Daemon not running for profile '{profile or 'default'}'")
        return True

    # Get profile-specific port
    pm = ProfileManager()
    paths = pm.resolve_profile_paths(profile)
    port = paths.port

    # Find PID by port using lsof (works on macOS/Linux, handles stale lockfiles)
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
                    break  # Process exited
        else:
            logger.warning(f"Could not find PID for port {port}")
    except (subprocess.TimeoutExpired, ValueError, OSError, FileNotFoundError) as e:
        logger.warning(f"Could not find/kill daemon by port: {e}")

    # Wait for health check to fail (daemon fully stopped)
    for _ in range(30):  # Wait up to 3 seconds
        if not _is_daemon_running(profile):
            return True
        time.sleep(0.1)

    return not _is_daemon_running(profile)


def find_cli_binary() -> Path | None:
    """Find the hindsight CLI binary in known locations or PATH."""
    import shutil

    # Check standard install locations
    for install_dir in CLI_INSTALL_DIRS:
        binary = install_dir / "hindsight"
        if binary.exists() and os.access(binary, os.X_OK):
            return binary

    # Check PATH
    path_binary = shutil.which("hindsight")
    if path_binary:
        return Path(path_binary)

    return None


def is_cli_installed() -> bool:
    """Check if the hindsight CLI is installed."""
    return find_cli_binary() is not None


def install_cli() -> bool:
    """
    Install the hindsight CLI using the official installer.

    Returns True if installation succeeded.
    """
    import subprocess
    import sys

    from . import __version__

    # Determine CLI version (use env var or match embed version)
    cli_version = os.getenv("HINDSIGHT_EMBED_CLI_VERSION", __version__)

    print(f"Installing hindsight CLI (version {cli_version})...")
    print(f"  Installer URL: {CLI_INSTALLER_URL}")

    try:
        # Download and run installer with version env var
        env = os.environ.copy()
        env["HINDSIGHT_CLI_VERSION"] = cli_version

        result = subprocess.run(
            ["bash", "-c", f"curl -fsSL {CLI_INSTALLER_URL} | bash"],
            capture_output=True,
            text=True,
            env=env,
        )

        if result.returncode != 0:
            print(f"CLI installation failed (exit code {result.returncode}):", file=sys.stderr)
            if result.stdout:
                print(f"  stdout: {result.stdout}", file=sys.stderr)
            if result.stderr:
                print(f"  stderr: {result.stderr}", file=sys.stderr)
            return False

        cli_binary = find_cli_binary()
        if cli_binary:
            print(f"CLI installed to {cli_binary}")
            return True
        else:
            print("CLI installation completed but binary not found", file=sys.stderr)
            print(f"  stdout: {result.stdout}", file=sys.stderr)
            print(f"  stderr: {result.stderr}", file=sys.stderr)
            # Check known locations
            for install_dir in CLI_INSTALL_DIRS:
                binary = install_dir / "hindsight"
                print(f"  Checking {binary}: exists={binary.exists()}", file=sys.stderr)
            return False

    except Exception as e:
        print(f"CLI installation failed: {e}", file=sys.stderr)
        return False


def ensure_cli_installed() -> bool:
    """Ensure CLI is installed, installing if needed."""
    if is_cli_installed():
        return True
    return install_cli()


def run_cli(args: list[str], config: dict, profile: str | None = None) -> int:
    """
    Run the hindsight CLI with the given arguments.

    Ensures daemon is running (unless HINDSIGHT_API_URL is already set) and passes the API URL.

    Args:
        args: CLI arguments (e.g., ["memory", "retain", "bank", "content"])
        config: Configuration dict with llm settings
        profile: Profile name (None = resolve from priority)

    Returns:
        Exit code from CLI
    """
    import subprocess
    import sys

    if profile is None:
        profile = resolve_active_profile()

    # Ensure CLI is installed
    if not ensure_cli_installed():
        return 1

    cli_binary = find_cli_binary()
    if not cli_binary:
        print("Error: hindsight CLI not found", file=sys.stderr)
        return 1

    # Build environment
    env = os.environ.copy()

    # Check if user wants to use external API
    api_url = env.get("HINDSIGHT_EMBED_API_URL")

    if not api_url:
        # No external API specified - ensure our daemon is running
        if not ensure_daemon_running(config, profile):
            print("Error: Failed to start daemon", file=sys.stderr)
            return 1
        api_url = get_daemon_url(profile)
    else:
        # Using external API - skip daemon startup
        logger.debug(f"Using external API at {api_url}")

    # Set the API URL for the CLI (using the standard HINDSIGHT_API_URL var that the CLI expects)
    env["HINDSIGHT_API_URL"] = api_url

    # Pass through API token if set (using the standard HINDSIGHT_API_KEY var that the CLI expects)
    api_token = env.get("HINDSIGHT_EMBED_API_TOKEN")
    if api_token:
        env["HINDSIGHT_API_KEY"] = api_token

    # Run CLI
    try:
        result = subprocess.run(
            [str(cli_binary)] + args,
            env=env,
        )
        return result.returncode
    except Exception as e:
        print(f"Error running CLI: {e}", file=sys.stderr)
        return 1
