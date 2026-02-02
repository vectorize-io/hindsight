"""
Client for communicating with the Hindsight daemon.

Handles daemon lifecycle (start if needed) and API requests via the Python client.
"""

import logging
import os
import shlex
import subprocess
import time
from pathlib import Path

import httpx  # Used only for health check

from .profile_manager import ProfileManager, resolve_active_profile

logger = logging.getLogger(__name__)

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
    logger.info(f"Starting daemon for {profile_label}...")

    # Build environment with LLM config
    env = os.environ.copy()
    if config.get("llm_api_key"):
        env["HINDSIGHT_API_LLM_API_KEY"] = config["llm_api_key"]
    if config.get("llm_provider"):
        env["HINDSIGHT_API_LLM_PROVIDER"] = config["llm_provider"]
    if config.get("llm_model"):
        env["HINDSIGHT_API_LLM_MODEL"] = config["llm_model"]

    # Use single shared pg0 database for all banks (banks are isolated within the database)
    # Allow override via HINDSIGHT_EMBED_API_DATABASE_URL for external PostgreSQL
    # (e.g. when running as root where embedded pg0 cannot use initdb)
    if "HINDSIGHT_EMBED_API_DATABASE_URL" not in env:
        env["HINDSIGHT_API_DATABASE_URL"] = "pg0://hindsight-embed"
    else:
        # Pass through the embed-specific env var to the daemon as the standard API env var
        env["HINDSIGHT_API_DATABASE_URL"] = env["HINDSIGHT_EMBED_API_DATABASE_URL"]
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

    # Pass profile-specific port and lockfile
    cmd = _find_hindsight_api_command() + [
        "--daemon",
        "--idle-timeout",
        str(idle_timeout),
        "--port",
        str(paths.port),
        "--lockfile",
        str(paths.lock),
    ]

    # Use profile-specific log file
    daemon_log = paths.log
    daemon_log.parent.mkdir(parents=True, exist_ok=True)

    print(f"Starting daemon with command: {' '.join(cmd)}", file=sys.stderr)
    print(f"  Log file: {daemon_log}", file=sys.stderr)

    try:
        # Start daemon with shell redirection (works across forks)
        # The >> appends to log file, 2>&1 redirects stderr to stdout
        shell_cmd = f"{' '.join(cmd)} >> {shlex.quote(str(daemon_log))} 2>&1"
        subprocess.Popen(
            shell_cmd,
            shell=True,
            env=env,
            start_new_session=True,
        )

        # Wait for daemon to be ready
        # Note: With --daemon flag, the parent process forks and exits immediately (code 0).
        # The child process (actual daemon) continues running. So we can't rely on process.poll()
        # to detect failures - we must use the health check.
        start_time = time.time()
        last_check_time = start_time
        while time.time() - start_time < DAEMON_STARTUP_TIMEOUT:
            if _is_daemon_running(profile):
                # Health check passed - but daemon might crash during initialization
                # Wait a moment and verify it's still healthy (stability check)
                print("  Daemon responding, verifying stability...", file=sys.stderr)
                time.sleep(2)
                if _is_daemon_running(profile):
                    logger.info(f"Daemon started successfully for {profile_label}")
                    return True
                else:
                    # Daemon crashed after initial health check
                    print("  Daemon crashed during initialization", file=sys.stderr)
                    break

            # Periodically log progress
            if time.time() - last_check_time > 5:
                elapsed = int(time.time() - start_time)
                print(f"  Still waiting for daemon... ({elapsed}s elapsed)", file=sys.stderr)
                last_check_time = time.time()

            time.sleep(0.5)

        logger.error("Daemon failed to start")
        # Show logs on failure
        if daemon_log.exists():
            log_content = daemon_log.read_text()
            if log_content:
                # Show last 3000 chars of log
                print(f"\n  Daemon log ({daemon_log}):", file=sys.stderr)
                print(f"{log_content[-3000:]}", file=sys.stderr)
        return False

    except FileNotFoundError as e:
        print(f"Command not found: {cmd[0]}", file=sys.stderr)
        print(f"  Full command: {' '.join(cmd)}", file=sys.stderr)
        logger.error("hindsight-api command not found. Install with: pip install hindsight-api")
        return False
    except Exception as e:
        print(f"Failed to start daemon: {e}", file=sys.stderr)
        logger.error(f"Failed to start daemon: {e}")
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
    if profile is None:
        profile = resolve_active_profile()

    # Get profile-specific lockfile
    pm = ProfileManager()
    paths = pm.resolve_profile_paths(profile)
    lockfile = paths.lock

    # Try to kill by PID from lockfile
    if lockfile.exists():
        try:
            pid = int(lockfile.read_text().strip())
            os.kill(pid, 15)  # SIGTERM
            # Wait for process to exit
            for _ in range(50):
                time.sleep(0.1)
                try:
                    os.kill(pid, 0)
                except OSError:
                    break  # Process exited
        except (ValueError, OSError):
            pass

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
