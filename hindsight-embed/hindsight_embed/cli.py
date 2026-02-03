"""
Hindsight Embedded CLI.

A wrapper CLI that manages a local daemon and forwards commands to hindsight-cli.
No external server required - runs everything locally with automatic daemon management.

Usage:
    hindsight-embed configure              # Interactive setup
    hindsight-embed retain "User prefers dark mode"
    hindsight-embed recall "What are user preferences?"
    hindsight-embed daemon status          # Check daemon status

Environment variables:
    HINDSIGHT_EMBED_LLM_API_KEY: Required. API key for LLM provider.
    HINDSIGHT_EMBED_LLM_PROVIDER: Optional. LLM provider (default: "openai").
    HINDSIGHT_EMBED_LLM_MODEL: Optional. LLM model (default: "gpt-4o-mini").
    HINDSIGHT_EMBED_BANK_ID: Optional. Memory bank ID (default: "default").
    HINDSIGHT_EMBED_API_URL: Optional. Use external API server instead of starting local daemon.
    HINDSIGHT_EMBED_API_TOKEN: Optional. Authentication token for external API (sent as Bearer token).
    HINDSIGHT_EMBED_API_DATABASE_URL: Optional. Database URL for daemon (default: "pg0://hindsight-embed").
    HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT: Optional. Seconds before daemon auto-exits when idle (default: 300).
    HINDSIGHT_EMBED_API_VERSION: Optional. hindsight-api version to use (default: matches embed version).
                                 Note: Only applies when starting daemon. To change version, stop daemon first.
    HINDSIGHT_EMBED_CLI_VERSION: Optional. hindsight CLI version to install (default: {embed_version}).
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from .profile_manager import (
    ProfileManager,
    resolve_active_profile,
    validate_profile_exists,
)

CONFIG_DIR = Path.home() / ".hindsight"
CONFIG_FILE = CONFIG_DIR / "embed"
CONFIG_FILE_ALT = CONFIG_DIR / "config.env"  # Alternative config file location

# Global profile context (set by --profile flag)
_cli_profile_override: str | None = None


def set_cli_profile_override(profile: str | None):
    """Set the CLI profile override (from --profile flag)."""
    global _cli_profile_override
    _cli_profile_override = profile


def get_cli_profile_override() -> str | None:
    """Get the CLI profile override."""
    return _cli_profile_override


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level_str = os.environ.get("HINDSIGHT_EMBED_LOG_LEVEL", "info").lower()
    if verbose:
        level_str = "debug"

    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    level = level_map.get(level_str, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        stream=sys.stderr,
    )
    return logging.getLogger(__name__)


def load_config_file(profile: str | None = None):
    """Load configuration from file if it exists.

    Args:
        profile: Profile name to load (None = resolve from priority).
    """
    # Resolve profile if not specified
    if profile is None:
        profile = resolve_active_profile()

    # Validate profile exists
    validate_profile_exists(profile)

    # Get config file path for profile
    pm = ProfileManager()
    paths = pm.resolve_profile_paths(profile)
    config_path = paths.config

    # For default profile, also check alternative location
    config_files = [config_path]
    if not profile:  # Default profile
        config_files.append(CONFIG_FILE_ALT)

    for config_file in config_files:
        if config_file.exists():
            with open(config_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        # Handle 'export VAR=value' format
                        if line.startswith("export "):
                            line = line[7:]
                        key, value = line.split("=", 1)
                        if key not in os.environ:  # Don't override env vars
                            os.environ[key] = value
            break  # Only load first existing config file


def get_config(profile: str | None = None):
    """Get configuration from environment variables.

    Args:
        profile: Profile name to load (None = resolve from priority).

    Returns:
        Config dict with LLM settings.
    """
    load_config_file(profile)
    return {
        "llm_api_key": os.environ.get("HINDSIGHT_EMBED_LLM_API_KEY")
        or os.environ.get("HINDSIGHT_API_LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY"),
        "llm_provider": os.environ.get("HINDSIGHT_EMBED_LLM_PROVIDER")
        or os.environ.get("HINDSIGHT_API_LLM_PROVIDER", "openai"),
        "llm_model": os.environ.get("HINDSIGHT_EMBED_LLM_MODEL")
        or os.environ.get("HINDSIGHT_API_LLM_MODEL", "gpt-4o-mini"),
        "bank_id": os.environ.get("HINDSIGHT_EMBED_BANK_ID", "default"),
        "profile": profile or resolve_active_profile(),
    }


# Provider defaults: (provider_id, default_model, env_key_name)
PROVIDER_DEFAULTS = {
    "openai": ("openai", "o3-mini", "OPENAI_API_KEY"),
    "groq": ("groq", "openai/gpt-oss-20b", "GROQ_API_KEY"),
    "google": ("google", "gemini-2.0-flash", "GOOGLE_API_KEY"),
    "ollama": ("ollama", "llama3.2", None),
}


def do_configure(args):
    """Configuration setup with optional profile and env vars support.

    Args:
        args: Parsed arguments with optional --profile and --env flags.
    """
    # Get profile and env vars from args
    profile = getattr(args, "profile", None)
    env_vars = getattr(args, "env", None)

    # Check if we're creating a named profile with --env flags
    if profile and env_vars:
        return _do_configure_profile_with_env(profile, env_vars)

    # Check if we're creating a named profile interactively
    if profile:
        return _do_configure_profile_interactive(profile)

    # Default behavior: interactive configuration for default profile
    # If stdin is not a terminal (e.g., running via curl | bash),
    # redirect stdin from /dev/tty for interactive prompts
    original_stdin = None
    if not sys.stdin.isatty():
        try:
            original_stdin = sys.stdin
            sys.stdin = open("/dev/tty", "r")
        except OSError:
            # No terminal available - try non-interactive mode with env vars
            return _do_configure_from_env(None)

    try:
        return _do_configure_interactive(None)
    finally:
        if original_stdin is not None:
            sys.stdin.close()
            sys.stdin = original_stdin


def _do_configure_profile_with_env(profile_name: str, env_vars: list[str]) -> int:
    """Configure a named profile with environment variables (non-interactive).

    Args:
        profile_name: Name of the profile to create/update.
        env_vars: List of KEY=VALUE strings.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    # Parse env vars
    config = {}
    for env_str in env_vars:
        if "=" not in env_str:
            print(f"Error: Invalid --env format '{env_str}'. Expected KEY=VALUE", file=sys.stderr)
            return 1

        key, value = env_str.split("=", 1)
        key = key.strip()
        value = value.strip()

        # Validate key format (should start with HINDSIGHT_EMBED_)
        if not key.startswith("HINDSIGHT_EMBED_") and not key.startswith("HINDSIGHT_API_"):
            print(
                f"Warning: Key '{key}' doesn't start with HINDSIGHT_EMBED_ or HINDSIGHT_API_",
                file=sys.stderr,
            )

        config[key] = value

    # Create profile
    pm = ProfileManager()
    try:
        pm.create_profile(profile_name, config)
    except ValueError as e:
        print(f"Error creating profile: {e}", file=sys.stderr)
        return 1

    print()
    print(f"\033[32m✓ Profile '{profile_name}' configured successfully!\033[0m")
    print()
    profile_path = CONFIG_DIR / "profiles" / f"{profile_name}.env"
    print(f"  \033[2mConfig:\033[0m {profile_path}")
    print(f"  \033[2mPort:\033[0m {pm.resolve_profile_paths(profile_name).port}")
    print()
    print("  \033[2mUse with:\033[0m")
    print(f'    \033[36mHINDSIGHT_EMBED_PROFILE={profile_name} hindsight-embed memory retain default "text"\033[0m')
    print(f'    \033[36mhindsight-embed --profile {profile_name} memory recall default "query"\033[0m')
    print()

    return 0


def _do_configure_profile_interactive(profile_name: str) -> int:
    """Configure a named profile interactively.

    Args:
        profile_name: Name of the profile to create/update.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    print()
    print(f"\033[1m\033[36m  Configuring profile '{profile_name}'\033[0m")
    print()

    # Use the same interactive flow as default profile
    return _do_configure_interactive(profile_name)


def _do_configure_from_env(profile_name: str | None = None):
    """Non-interactive configuration from environment variables (for CI).

    Args:
        profile_name: Optional profile name. If None, configures default profile.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    # Check for required environment variables
    api_key = os.environ.get("HINDSIGHT_EMBED_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    provider = os.environ.get("HINDSIGHT_EMBED_LLM_PROVIDER", "openai")

    if provider not in PROVIDER_DEFAULTS:
        print(
            f"Error: Unknown provider '{provider}'. Supported: {', '.join(PROVIDER_DEFAULTS.keys())}", file=sys.stderr
        )
        return 1

    _, default_model, env_key = PROVIDER_DEFAULTS[provider]

    # Check for API key (required for non-ollama providers)
    if not api_key and provider != "ollama":
        print("Error: Cannot run interactive configuration without a terminal.", file=sys.stderr)
        print("", file=sys.stderr)
        print("For non-interactive (CI) mode, set environment variables:", file=sys.stderr)
        print("  HINDSIGHT_EMBED_LLM_API_KEY=<your-api-key>", file=sys.stderr)
        print(f"  HINDSIGHT_EMBED_LLM_PROVIDER={provider}  # optional, default: openai", file=sys.stderr)
        print(f"  HINDSIGHT_EMBED_LLM_MODEL=<model>  # optional, default: {default_model}", file=sys.stderr)
        return 1

    model = os.environ.get("HINDSIGHT_EMBED_LLM_MODEL", default_model)
    bank_id = os.environ.get("HINDSIGHT_EMBED_BANK_ID", "default")

    print()
    profile_label = f"profile '{profile_name}'" if profile_name else "default profile"
    print(f"\033[1m\033[36m  Hindsight Embed - Non-interactive Configuration ({profile_label})\033[0m")
    print()
    print(f"  \033[2mProvider:\033[0m {provider}")
    print(f"  \033[2mModel:\033[0m {model}")
    print(f"  \033[2mBank ID:\033[0m {bank_id}")

    # Build configuration
    config = {
        "HINDSIGHT_EMBED_LLM_PROVIDER": provider,
        "HINDSIGHT_EMBED_LLM_MODEL": model,
        "HINDSIGHT_EMBED_BANK_ID": bank_id,
    }
    if api_key:
        config["HINDSIGHT_EMBED_LLM_API_KEY"] = api_key

    # Force CPU mode for embeddings/reranker on macOS to avoid MPS/XPC crashes in daemon mode
    import platform

    if platform.system() == "Darwin":  # macOS
        config["HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU"] = "1"
        config["HINDSIGHT_API_RERANKER_LOCAL_FORCE_CPU"] = "1"

    # Save configuration
    if profile_name:
        # Named profile
        pm = ProfileManager()
        try:
            pm.create_profile(profile_name, config)
        except ValueError as e:
            print(f"Error creating profile: {e}", file=sys.stderr)
            return 1
        config_path = CONFIG_DIR / "profiles" / f"{profile_name}.env"
    else:
        # Default profile
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        config_path = CONFIG_FILE

        with open(config_path, "w") as f:
            f.write("# Hindsight Embed Configuration\n")
            f.write("# Generated by hindsight-embed configure (non-interactive)\n\n")
            for key, value in config.items():
                f.write(f"{key}={value}\n")

        config_path.chmod(0o600)

    print()
    display_name = profile_name if profile_name else "default"
    print(f"\033[32m  ✓ Profile '{display_name}' configured successfully!\033[0m")
    print(f"  \033[2mConfig:\033[0m {config_path}")
    pm = ProfileManager()
    port = pm.resolve_profile_paths(profile_name or "").port
    print(f"  \033[2mPort:\033[0m {port}")
    print()

    return 0


def _prompt_choice(prompt: str, choices: list[tuple[str, str]], default: int = 1) -> str | None:
    """Simple choice prompt that works with /dev/tty."""
    print(f"\033[1m{prompt}\033[0m")
    print()
    for i, (label, _) in enumerate(choices, 1):
        print(f"  \033[36m{i})\033[0m {label}")
    print()
    try:
        response = input(f"Enter choice [{default}]: ").strip()
        if not response:
            return choices[default - 1][1]
        idx = int(response)
        if 1 <= idx <= len(choices):
            return choices[idx - 1][1]
        return choices[default - 1][1]
    except (ValueError, EOFError, KeyboardInterrupt):
        return None


def _prompt_text(prompt: str, default: str = "") -> str | None:
    """Simple text prompt."""
    try:
        suffix = f" [{default}]" if default else ""
        response = input(f"\033[1m{prompt}\033[0m{suffix}: ").strip()
        return response if response else default
    except (EOFError, KeyboardInterrupt):
        return None


def _prompt_password(prompt: str) -> str | None:
    """Simple password prompt that works with /dev/tty."""
    import termios
    import tty

    # Read password with echo disabled (works because sys.stdin is already /dev/tty)
    fd = sys.stdin.fileno()
    print(f"\033[1m{prompt}\033[0m: ", end="", flush=True)
    try:
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd, termios.TCSADRAIN)
            # Read character by character until newline
            password = []
            while True:
                ch = sys.stdin.read(1)
                if ch in ("\n", "\r"):
                    break
                elif ch == "\x7f":  # Backspace
                    if password:
                        password.pop()
                        # Erase character on screen
                        sys.stdout.write("\b \b")
                        sys.stdout.flush()
                elif ch == "\x03":  # Ctrl+C
                    raise KeyboardInterrupt
                elif ch >= " ":  # Printable character
                    password.append(ch)
            print()  # Newline after password
            return "".join(password)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    except Exception:
        # Fallback to simple input if termios fails
        try:
            return input("")
        except (EOFError, KeyboardInterrupt):
            return None


def _prompt_confirm(prompt: str, default: bool = True) -> bool | None:
    """Simple yes/no prompt."""
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        response = input(f"\033[1m{prompt}\033[0m {suffix}: ").strip().lower()
        if not response:
            return default
        return response in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return None


def _do_configure_interactive(profile_name: str | None = None):
    """Internal interactive configuration.

    Args:
        profile_name: Optional profile name. If None, configures default profile.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    print()
    display_name = profile_name if profile_name else "default"
    profile_label = f" - Profile '{display_name}'"
    print("\033[1m\033[36m  ╭─────────────────────────────────────╮\033[0m")
    print(f"\033[1m\033[36m  │   Hindsight Embed Configuration{profile_label:4s}│\033[0m")
    print("\033[1m\033[36m  ╰─────────────────────────────────────╯\033[0m")
    print()

    # Check existing config
    if profile_name:
        config_path = CONFIG_DIR / "profiles" / f"{profile_name}.env"
    else:
        config_path = CONFIG_FILE

    if config_path.exists():
        if not _prompt_confirm("Existing configuration found. Reconfigure?", default=False):
            print("\n\033[32m✓\033[0m Keeping existing configuration.")
            return 0
        print()

    # Provider selection
    providers = [
        ("OpenAI (recommended)", "openai"),
        ("Groq (fast & free tier)", "groq"),
        ("Google Gemini", "google"),
        ("Ollama (local, no API key)", "ollama"),
    ]

    provider = _prompt_choice("Select your LLM provider:", providers, default=1)
    if provider is None:
        print("\n\033[33m⚠\033[0m Configuration cancelled.")
        return 1

    _, default_model, env_key = PROVIDER_DEFAULTS[provider]
    print()

    # API key
    api_key = ""
    if env_key:
        existing = os.environ.get(env_key, "")

        if existing:
            masked = existing[:8] + "..." + existing[-4:] if len(existing) > 12 else "***"
            if _prompt_confirm(f"Found API key in ${env_key} ({masked}). Use it?", default=True):
                api_key = existing
            print()

        if not api_key:
            api_key = _prompt_password("Enter your API key")
            if not api_key:
                print("\n\033[31m✗\033[0m API key is required.", file=sys.stderr)
                return 1
            print()

    # Model selection
    model = _prompt_text("Model name", default=default_model)
    if model is None:
        return 1
    print()

    # Bank ID
    bank_id = _prompt_text("Memory bank ID", default="default")
    if bank_id is None:
        return 1

    # Build configuration
    config = {
        "HINDSIGHT_EMBED_LLM_PROVIDER": provider,
        "HINDSIGHT_EMBED_LLM_MODEL": model,
        "HINDSIGHT_EMBED_BANK_ID": bank_id,
    }
    if api_key:
        config["HINDSIGHT_EMBED_LLM_API_KEY"] = api_key

    # Force CPU mode for embeddings/reranker on macOS to avoid MPS/XPC crashes in daemon mode
    import platform

    if platform.system() == "Darwin":  # macOS
        config["HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU"] = "1"
        config["HINDSIGHT_API_RERANKER_LOCAL_FORCE_CPU"] = "1"

    # Save configuration
    if profile_name:
        # Named profile
        pm = ProfileManager()
        try:
            pm.create_profile(profile_name, config)
        except ValueError as e:
            print(f"\n\033[31m✗\033[0m Error creating profile: {e}", file=sys.stderr)
            return 1
        config_path = CONFIG_DIR / "profiles" / f"{profile_name}.env"
    else:
        # Default profile
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        config_path = CONFIG_FILE

        with open(config_path, "w") as f:
            f.write("# Hindsight Embed Configuration\n")
            f.write("# Generated by hindsight-embed configure\n\n")
            for key, value in config.items():
                f.write(f"{key}={value}\n")

        config_path.chmod(0o600)

    # For default profile only: stop existing daemon if running (it needs to pick up new config)
    if not profile_name:
        from . import daemon_client

        profile = ""  # Default profile
        if daemon_client._is_daemon_running(profile):
            print("\n  \033[2mRestarting daemon with new configuration...\033[0m")
            daemon_client.stop_daemon(profile)

        # Start daemon with new config
        new_config = {
            "llm_api_key": api_key,
            "llm_provider": provider,
            "llm_model": model,
            "bank_id": bank_id,
            "profile": profile,
        }
        if daemon_client.ensure_daemon_running(new_config, profile):
            print("  \033[32m✓ Daemon started\033[0m")
        else:
            print("  \033[33m⚠ Failed to start daemon (will start on first command)\033[0m")

    print()
    display_name = profile_name if profile_name else "default"
    print("\033[32m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m")
    print(f"\033[32m  ✓ Profile '{display_name}' configured successfully!\033[0m")
    print("\033[32m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m")
    print()
    print(f"  \033[2mConfig:\033[0m {config_path}")
    pm = ProfileManager()
    port = pm.resolve_profile_paths(profile_name or "").port
    print(f"  \033[2mPort:\033[0m {port}")
    print()

    if profile_name:
        print("  \033[2mUse with:\033[0m")
        print(f'    \033[36mHINDSIGHT_EMBED_PROFILE={profile_name} hindsight-embed memory retain default "text"\033[0m')
        print(f'    \033[36mhindsight-embed --profile {profile_name} memory recall default "query"\033[0m')
    else:
        print("  \033[2mTest with:\033[0m")
        print('    \033[36mhindsight-embed memory retain default "Alice works at Google as a software engineer"\033[0m')
        print('    \033[36mhindsight-embed memory recall default "Alice"\033[0m')
    print()

    return 0


def do_daemon(args, config: dict, logger):
    """Handle daemon subcommands."""
    from pathlib import Path

    from . import daemon_client

    # Get profile from config
    profile = config.get("profile", "")

    # Get profile-specific paths
    pm = ProfileManager()
    paths = pm.resolve_profile_paths(profile)
    daemon_log_path = paths.log

    if args.daemon_command == "start":
        if daemon_client._is_daemon_running(profile):
            print("Daemon is already running")
            return 0

        print("Starting daemon...")
        if daemon_client.ensure_daemon_running(config, profile):
            print("Daemon started successfully")
            print(f"  Port: {paths.port}")
            print(f"  Logs: {daemon_log_path}")
            return 0
        else:
            print("Failed to start daemon", file=sys.stderr)
            return 1

    elif args.daemon_command == "stop":
        if not daemon_client._is_daemon_running(profile):
            print("Daemon is not running")
            return 0

        print("Stopping daemon...")
        if daemon_client.stop_daemon(profile):
            print("Daemon stopped")
            return 0
        else:
            print("Failed to stop daemon", file=sys.stderr)
            return 1

    elif args.daemon_command == "status":
        if daemon_client._is_daemon_running(profile):
            # Get PID from lockfile
            lockfile = paths.lock
            pid = "unknown"
            if lockfile.exists():
                try:
                    pid = lockfile.read_text().strip()
                except Exception:
                    pass
            print(f"Daemon is running (PID: {pid})")
            print(f"  URL: {daemon_client.get_daemon_url(profile)}")
            print(f"  Logs: {daemon_log_path}")
            return 0
        else:
            print("Daemon is not running")
            return 1

    elif args.daemon_command == "logs":
        if not daemon_log_path.exists():
            print("No daemon logs found", file=sys.stderr)
            print(f"  Expected at: {daemon_log_path}")
            return 1

        if args.follow:
            # Follow mode - like tail -f
            import subprocess

            try:
                subprocess.run(["tail", "-f", str(daemon_log_path)])
            except KeyboardInterrupt:
                pass
            return 0
        else:
            # Show last N lines
            try:
                with open(daemon_log_path) as f:
                    lines = f.readlines()
                    for line in lines[-args.lines :]:
                        print(line, end="")
                return 0
            except Exception as e:
                print(f"Error reading logs: {e}", file=sys.stderr)
                return 1

    else:
        print("Usage: hindsight-embed daemon {start|stop|status|logs}", file=sys.stderr)
        return 1


def do_profile_command(args: list[str]) -> int:
    """Handle profile subcommands.

    Args:
        args: Command arguments (after 'profile').

    Returns:
        Exit code (0 = success, 1 = error).
    """
    parser = argparse.ArgumentParser(prog="hindsight-embed profile")
    subparsers = parser.add_subparsers(dest="profile_command", required=True)

    # List command
    subparsers.add_parser("list", help="List all profiles")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a profile")
    delete_parser.add_argument("name", help="Profile name to delete")

    # Set-active command
    set_active_parser = subparsers.add_parser("set-active", help="Set active profile")
    set_active_parser.add_argument("name", nargs="?", help="Profile name (omit to clear)")
    set_active_parser.add_argument("--none", action="store_true", help="Clear active profile")

    # Show command
    subparsers.add_parser("show", help="Show current active profile")

    try:
        parsed_args = parser.parse_args(args)
    except SystemExit as e:
        return e.code or 1

    pm = ProfileManager()

    if parsed_args.profile_command == "list":
        # List all profiles
        profiles = pm.list_profiles()
        if not profiles:
            print("No profiles configured.")
            print()
            print("Create one with:")
            print("  hindsight-embed configure --profile my-app --env HINDSIGHT_EMBED_LLM_API_KEY=...")
            return 0

        print()
        print("\033[1mProfiles:\033[0m")
        print()
        for profile in profiles:
            name = profile.name or "default"
            active_marker = " \033[32m✓ active\033[0m" if profile.is_active else ""
            daemon_marker = " \033[36m● running\033[0m" if profile.daemon_running else ""
            print(f"  \033[1m{name}\033[0m{active_marker}{daemon_marker}")
            print(f"    Port: {profile.port}")
            if profile.name:  # Named profile
                config_path = CONFIG_DIR / "profiles" / f"{profile.name}.env"
                print(f"    Config: {config_path}")
            else:  # Default profile
                config_path = CONFIG_FILE
                print(f"    Config: {config_path}")
            print()

        return 0

    elif parsed_args.profile_command == "delete":
        # Delete profile
        profile_name = parsed_args.name

        if not pm.profile_exists(profile_name):
            print(f"Error: Profile '{profile_name}' does not exist.", file=sys.stderr)
            return 1

        # Check if daemon is running
        profile_info = pm.get_profile(profile_name)
        if profile_info and profile_info.daemon_running:
            print(f"Warning: Daemon is running for profile '{profile_name}'")
            try:
                confirm = input("Stop daemon and delete profile? [y/N]: ").strip().lower()
                if confirm not in ("y", "yes"):
                    print("Cancelled.")
                    return 0
            except (EOFError, KeyboardInterrupt):
                print("\nCancelled.")
                return 0

            # Stop daemon
            from . import daemon_client

            daemon_client.stop_daemon(profile_name)

        # Delete profile
        try:
            pm.delete_profile(profile_name)
            print(f"\033[32m✓\033[0m Profile '{profile_name}' deleted.")
            return 0
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    elif parsed_args.profile_command == "set-active":
        # Set active profile
        if parsed_args.none:
            pm.set_active_profile(None)
            print("\033[32m✓\033[0m Active profile cleared.")
            return 0

        if not parsed_args.name:
            print("Error: Specify profile name or use --none to clear.", file=sys.stderr)
            return 1

        profile_name = parsed_args.name
        try:
            pm.set_active_profile(profile_name)
            print(f"\033[32m✓\033[0m Active profile set to '{profile_name}'.")
            return 0
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    elif parsed_args.profile_command == "show":
        # Show current active profile
        # Resolve using full priority chain
        active_profile = resolve_active_profile()

        # Validate profile exists
        validate_profile_exists(active_profile)

        print()
        display_name = active_profile if active_profile else "default"
        print(f"\033[1mActive profile:\033[0m {display_name}")
        print()

        # Determine source
        if not active_profile:
            print("  \033[2mSource:\033[0m Default (no profile specified)")
        elif os.getenv("HINDSIGHT_EMBED_PROFILE"):
            print("  \033[2mSource:\033[0m HINDSIGHT_EMBED_PROFILE environment variable")
        elif get_cli_profile_override():
            print("  \033[2mSource:\033[0m --profile flag")
        elif pm.get_active_profile():
            print("  \033[2mSource:\033[0m Active profile file")
        else:
            print("  \033[2mSource:\033[0m Default")

        # Show config path
        paths = pm.resolve_profile_paths(active_profile)
        print(f"  \033[2mConfig:\033[0m {paths.config}")
        print(f"  \033[2mPort:\033[0m {paths.port}")
        print()

        return 0

    return 1


def main():
    """Main entry point."""
    # Parse global --profile flag first
    # NOTE: For 'configure' command, we DON'T consume the --profile flag here
    # because it has special meaning for that command (which profile to create)
    profile_from_flag = None
    remaining_args = []
    i = 1

    # Peek at the command to determine if we should consume --profile
    command = sys.argv[1] if len(sys.argv) > 1 else None
    should_consume_profile = command != "configure"

    while i < len(sys.argv):
        arg = sys.argv[i]
        if should_consume_profile and arg == "--profile" and i + 1 < len(sys.argv):
            profile_from_flag = sys.argv[i + 1]
            i += 2
            continue
        elif should_consume_profile and arg.startswith("--profile="):
            profile_from_flag = arg.split("=", 1)[1]
            i += 1
            continue
        remaining_args.append(arg)
        i += 1

    # Set global profile override if --profile was provided
    if profile_from_flag:
        set_cli_profile_override(profile_from_flag)

    # Check for built-in commands first (before argparse)
    # This allows us to forward unknown commands to hindsight-cli
    if len(remaining_args) > 0:
        command = remaining_args[0]

        # Handle configure
        if command == "configure":
            # Parse configure arguments
            parser = argparse.ArgumentParser(prog="hindsight-embed configure")
            parser.add_argument("--profile", help="Profile name to create/update")
            parser.add_argument(
                "--env",
                action="append",
                help="Environment variable (KEY=VALUE, can be repeated)",
            )
            # Parse only the configure arguments (skip 'configure' command)
            args = parser.parse_args(remaining_args[1:])
            logger = setup_logging(False)
            exit_code = do_configure(args)
            sys.exit(exit_code)

        # Handle daemon subcommands
        if command == "daemon":
            # Parse daemon subcommand
            parser = argparse.ArgumentParser(prog="hindsight-embed daemon")
            subparsers = parser.add_subparsers(dest="daemon_command")
            subparsers.add_parser("start", help="Start the daemon")
            subparsers.add_parser("stop", help="Stop the daemon")
            subparsers.add_parser("status", help="Check daemon status")
            logs_parser = subparsers.add_parser("logs", help="View daemon logs")
            logs_parser.add_argument("--follow", "-f", action="store_true")
            logs_parser.add_argument("--lines", "-n", type=int, default=50)

            args = parser.parse_args(remaining_args[1:])
            logger = setup_logging(False)
            config = get_config()
            exit_code = do_daemon(args, config, logger)
            sys.exit(exit_code)

        # Handle profile subcommands
        if command == "profile":
            exit_code = do_profile_command(remaining_args[1:])
            sys.exit(exit_code)

        # Handle --help / -h
        if command in ("--help", "-h"):
            print_help()
            sys.exit(0)

        # Forward all other commands to hindsight-cli
        config = get_config()

        # Check for LLM API key
        if not config["llm_api_key"]:
            print("Error: LLM API key is required.", file=sys.stderr)
            print("Run 'hindsight-embed configure' to set up.", file=sys.stderr)
            sys.exit(1)

        from . import daemon_client

        # Forward to hindsight-cli (handles daemon startup and CLI installation)
        profile = config.get("profile", "")
        exit_code = daemon_client.run_cli(remaining_args, config, profile)
        sys.exit(exit_code)

    # No command - show help
    print_help()
    sys.exit(1)


def print_help():
    """Print help message."""
    print("""Hindsight Embedded CLI - local memory operations with automatic daemon management.

Usage: hindsight-embed <command> [options]

Built-in commands:
    configure              Interactive configuration setup
    daemon start           Start the background daemon
    daemon stop            Stop the daemon
    daemon status          Check daemon status
    daemon logs [-f] [-n]  View daemon logs

CLI commands (forwarded to hindsight-cli):
    memory retain <bank> <content>   Store a memory
    memory recall <bank> <query>     Search memories
    memory reflect <bank> <query>    Generate contextual answer
    bank list                        List memory banks
    ...                              Run 'hindsight --help' for all commands

Examples:
    hindsight-embed configure
    hindsight-embed memory retain default "User prefers dark mode"
    hindsight-embed memory recall default "user preferences"
    hindsight-embed daemon status
    hindsight-embed daemon logs -f
""")


if __name__ == "__main__":
    main()
