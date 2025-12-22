"""
Client for communicating with the Hindsight daemon.

Handles daemon lifecycle (start if needed) and API requests via the Python client.
"""

import logging
import os
import subprocess
import time
from pathlib import Path

import httpx  # Used only for health check

logger = logging.getLogger(__name__)

DAEMON_PORT = 8889
DAEMON_URL = f"http://127.0.0.1:{DAEMON_PORT}"
DAEMON_STARTUP_TIMEOUT = 30  # seconds
DAEMON_IDLE_TIMEOUT = 300  # 5 minutes - auto-exit after idle

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
    return ["uvx", "hindsight-api"]


def _is_daemon_running() -> bool:
    """Check if daemon is running and responsive."""
    try:
        with httpx.Client(timeout=2) as client:
            response = client.get(f"{DAEMON_URL}/health")
            return response.status_code == 200
    except Exception:
        return False


def _start_daemon(config: dict) -> bool:
    """
    Start the daemon in background.

    Returns True if daemon started successfully.
    """
    logger.info("Starting daemon...")

    # Build environment with LLM config
    env = os.environ.copy()
    if config.get("llm_api_key"):
        env["HINDSIGHT_API_LLM_API_KEY"] = config["llm_api_key"]
    if config.get("llm_provider"):
        env["HINDSIGHT_API_LLM_PROVIDER"] = config["llm_provider"]
    if config.get("llm_model"):
        env["HINDSIGHT_API_LLM_MODEL"] = config["llm_model"]

    # Use pg0 database specific to bank
    bank_id = config.get("bank_id", "default")
    env["HINDSIGHT_API_DATABASE_URL"] = f"pg0://hindsight-embed-{bank_id}"

    # Optimization flags for faster startup
    env["HINDSIGHT_API_SKIP_LLM_VERIFICATION"] = "true"
    env["HINDSIGHT_API_LOG_LEVEL"] = "warning"

    cmd = _find_hindsight_api_command() + ["--daemon", "--idle-timeout", str(DAEMON_IDLE_TIMEOUT)]

    try:
        # Start daemon in background
        # On Unix, we don't need to do anything special since --daemon does the fork
        subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        # Wait for daemon to be ready
        start_time = time.time()
        while time.time() - start_time < DAEMON_STARTUP_TIMEOUT:
            if _is_daemon_running():
                logger.info("Daemon started successfully")
                return True
            time.sleep(0.5)

        logger.error("Daemon failed to start within timeout")
        return False

    except FileNotFoundError:
        logger.error("hindsight-api command not found. Install with: pip install hindsight-api")
        return False
    except Exception as e:
        logger.error(f"Failed to start daemon: {e}")
        return False


def ensure_daemon_running(config: dict) -> bool:
    """
    Ensure daemon is running, starting it if needed.

    Returns True if daemon is running.
    """
    if _is_daemon_running():
        logger.debug("Daemon already running")
        return True

    return _start_daemon(config)


def stop_daemon() -> bool:
    """Stop the running daemon."""
    # Try to kill by PID from lockfile
    lockfile = Path.home() / ".hindsight" / "daemon.lock"
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
                    return True
        except (ValueError, OSError):
            pass

    return not _is_daemon_running()


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

    print("Installing hindsight CLI...")

    try:
        # Download and run installer
        result = subprocess.run(
            ["bash", "-c", f"curl -fsSL {CLI_INSTALLER_URL} | bash"],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"CLI installation failed: {result.stderr}", file=sys.stderr)
            return False

        cli_binary = find_cli_binary()
        if cli_binary:
            print(f"CLI installed to {cli_binary}")
            return True
        else:
            print("CLI installation completed but binary not found", file=sys.stderr)
            return False

    except Exception as e:
        print(f"CLI installation failed: {e}", file=sys.stderr)
        return False


def ensure_cli_installed() -> bool:
    """Ensure CLI is installed, installing if needed."""
    if is_cli_installed():
        return True
    return install_cli()


def run_cli(args: list[str], config: dict) -> int:
    """
    Run the hindsight CLI with the given arguments.

    Ensures daemon is running and passes the API URL.

    Args:
        args: CLI arguments (e.g., ["memory", "retain", "bank", "content"])
        config: Configuration dict with llm settings

    Returns:
        Exit code from CLI
    """
    import subprocess
    import sys

    # Ensure CLI is installed
    if not ensure_cli_installed():
        return 1

    cli_binary = find_cli_binary()
    if not cli_binary:
        print("Error: hindsight CLI not found", file=sys.stderr)
        return 1

    # Ensure daemon is running
    if not ensure_daemon_running(config):
        print("Error: Failed to start daemon", file=sys.stderr)
        return 1

    # Build environment with API URL pointing to daemon
    env = os.environ.copy()
    env["HINDSIGHT_API_URL"] = DAEMON_URL

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
