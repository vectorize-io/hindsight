"""
CLI utilities for daemon and CLI management.

This module provides CLI-specific functions for managing the daemon
and the hindsight Rust CLI binary.
"""

import logging
import os
from pathlib import Path

from .daemon_embed_manager import DaemonEmbedManager
from .profile_manager import ProfileManager, resolve_active_profile

logger = logging.getLogger(__name__)

# Singleton manager instance
_manager = DaemonEmbedManager()

# CLI paths - check multiple locations
CLI_INSTALL_DIRS = [
    Path.home() / ".local" / "bin",  # Standard location from get-cli installer
    Path.home() / ".hindsight" / "bin",  # Alternative location
]
CLI_INSTALLER_URL = "https://hindsight.vectorize.io/get-cli"


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
    if profile is None:
        profile = resolve_active_profile()
    return _manager.get_url(profile)


def ensure_daemon_running(config: dict, profile: str | None = None) -> bool:
    """
    Ensure daemon is running, starting it if needed.

    Args:
        config: Configuration dict with LLM settings (accepts both simple keys
                like "llm_api_key" and env var format like "HINDSIGHT_API_LLM_API_KEY").
        profile: Profile name (None = resolve from priority).

    Returns:
        True if daemon is running.
    """
    if profile is None:
        profile = resolve_active_profile()

    return _manager.ensure_running(config, profile)


def stop_daemon(profile: str | None = None) -> bool:
    """Stop the running daemon and wait for it to fully stop.

    Args:
        profile: Profile name (None = resolve from priority).

    Returns:
        True if daemon stopped successfully.
    """
    if profile is None:
        profile = resolve_active_profile()

    return _manager.stop(profile)


def is_daemon_running(profile: str | None = None) -> bool:
    """Check if daemon is running for a profile.

    Args:
        profile: Profile name (None = resolve from priority).

    Returns:
        True if daemon is running and responsive.
    """
    if profile is None:
        profile = resolve_active_profile()

    return _manager.is_running(profile)


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
