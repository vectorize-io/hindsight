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
    HINDSIGHT_API_LLM_API_KEY: Required. API key for LLM provider.
    HINDSIGHT_API_LLM_PROVIDER: Optional. LLM provider (default: "openai").
    HINDSIGHT_API_LLM_MODEL: Optional. LLM model (default: "gpt-4o-mini").
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

from . import get_embed_manager

CONFIG_DIR = Path.home() / ".hindsight"
CONFIG_FILE = CONFIG_DIR / "embed"
CONFIG_FILE_ALT = CONFIG_DIR / "config.env"  # Alternative config file location

# Module-level variable to store CLI profile override (set by argparse)
_cli_profile_override: str | None = None


def get_cli_profile_override() -> str | None:
    """Get the profile override from CLI flag (--profile).

    Returns:
        Profile name if set via CLI flag, None otherwise.
    """
    return _cli_profile_override


def set_cli_profile_override(profile: str | None) -> None:
    """Set the profile override from CLI flag (--profile).

    Args:
        profile: Profile name to set, or None to clear.
    """
    global _cli_profile_override
    _cli_profile_override = profile


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

    # Set httpx to warning level to reduce noise
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return logging.getLogger(__name__)


def load_config_file():
    """Load configuration from file if it exists."""
    # Check both config file locations
    config_files = [CONFIG_FILE, CONFIG_FILE_ALT]
    for config_path in config_files:
        if config_path.exists():
            with open(config_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        # Handle 'export VAR=value' format
                        if line.startswith("export "):
                            line = line[7:]
                        key, value = line.split("=", 1)
                        if key not in os.environ:  # Don't override env vars
                            os.environ[key] = value


def get_config():
    """Get configuration from environment variables."""
    load_config_file()
    return {
        "llm_api_key": os.environ.get("HINDSIGHT_API_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        "llm_provider": os.environ.get("HINDSIGHT_API_LLM_PROVIDER", "openai"),
        "llm_model": os.environ.get("HINDSIGHT_API_LLM_MODEL", "gpt-4o-mini"),
        "bank_id": os.environ.get("HINDSIGHT_EMBED_BANK_ID", "default"),
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
        args: Parsed arguments with optional --profile, --port, and --env flags.
    """
    # Get profile, port, and env vars from args
    profile = getattr(args, "profile", None)
    port = getattr(args, "port", None)
    env_vars = getattr(args, "env", None)

    # Check if we're creating a named profile with --env flags
    if profile and env_vars:
        # Pass port (may be None for auto-allocation/reuse)
        return _do_configure_profile_with_env(profile, port, env_vars)

    # Check if we're creating a named profile interactively
    if profile:
        # Pass port (may be None for auto-allocation/reuse)
        return _do_configure_profile_interactive(profile, port)

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
            return _do_configure_from_env()

    try:
        return _do_configure_interactive()
    finally:
        if original_stdin is not None:
            sys.stdin.close()
            sys.stdin = original_stdin


def _do_configure_from_env():
    """Non-interactive configuration from environment variables (for CI)."""
    # Check for required environment variables
    api_key = os.environ.get("HINDSIGHT_API_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    provider = os.environ.get("HINDSIGHT_API_LLM_PROVIDER", "openai")

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
        print("  HINDSIGHT_API_LLM_API_KEY=<your-api-key>", file=sys.stderr)
        print(f"  HINDSIGHT_API_LLM_PROVIDER={provider}  # optional, default: openai", file=sys.stderr)
        print(f"  HINDSIGHT_API_LLM_MODEL=<model>  # optional, default: {default_model}", file=sys.stderr)
        return 1

    model = os.environ.get("HINDSIGHT_API_LLM_MODEL", default_model)
    bank_id = os.environ.get("HINDSIGHT_EMBED_BANK_ID", "default")

    print()
    print("\033[1m\033[36m  Hindsight Embed - Non-interactive Configuration\033[0m")
    print()
    print(f"  \033[2mProvider:\033[0m {provider}")
    print(f"  \033[2mModel:\033[0m {model}")
    print(f"  \033[2mBank ID:\033[0m {bank_id}")

    # Save configuration
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_FILE, "w") as f:
        f.write("# Hindsight Embed Configuration\n")
        f.write("# Generated by hindsight-embed configure (non-interactive)\n\n")
        f.write(f"HINDSIGHT_API_LLM_PROVIDER={provider}\n")
        f.write(f"HINDSIGHT_API_LLM_MODEL={model}\n")
        f.write(f"HINDSIGHT_EMBED_BANK_ID={bank_id}\n")
        if api_key:
            f.write(f"HINDSIGHT_API_LLM_API_KEY={api_key}\n")

        # Force CPU mode for embeddings/reranker on macOS to avoid MPS/XPC crashes in daemon mode
        # On Linux, users can set these to 0 to use CUDA if available
        import platform

        if platform.system() == "Darwin":  # macOS
            f.write("\n# Daemon settings (macOS: force CPU to avoid MPS/XPC issues)\n")
            f.write("HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU=1\n")
            f.write("HINDSIGHT_API_RERANKER_LOCAL_FORCE_CPU=1\n")

    CONFIG_FILE.chmod(0o600)

    print()
    print("\033[32m  ✓ Configuration saved!\033[0m")
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


def _do_configure_interactive(profile_name: str | None = None, port: int | None = None):
    """Internal interactive configuration.

    Args:
        profile_name: Optional profile name. If None, configures default profile.
        port: Optional port for named profile. Required if profile_name is provided.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    print()
    if profile_name:
        print(f"\033[1m\033[36m  Configuring profile '{profile_name}' (port {port})\033[0m")
    else:
        print("\033[1m\033[36m  ╭─────────────────────────────────────╮\033[0m")
        print("\033[1m\033[36m  │   Hindsight Embed Configuration    │\033[0m")
        print("\033[1m\033[36m  ╰─────────────────────────────────────╯\033[0m")
    print()

    # Check existing config
    config_file = CONFIG_DIR / "profiles" / f"{profile_name}.env" if profile_name else CONFIG_FILE
    if config_file.exists():
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

    # Save configuration
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Prepare config dict
    config_dict = {
        "HINDSIGHT_API_LLM_PROVIDER": provider,
        "HINDSIGHT_API_LLM_MODEL": model,
        "HINDSIGHT_EMBED_BANK_ID": bank_id,
    }
    if api_key:
        config_dict["HINDSIGHT_API_LLM_API_KEY"] = api_key

    # Force CPU mode for embeddings/reranker on macOS to avoid MPS/XPC crashes in daemon mode
    import platform

    if platform.system() == "Darwin":  # macOS
        config_dict["HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU"] = "1"
        config_dict["HINDSIGHT_API_RERANKER_LOCAL_FORCE_CPU"] = "1"

    if profile_name:
        # Create named profile
        from .profile_manager import ProfileManager

        pm = ProfileManager()
        try:
            pm.create_profile(profile_name, port, config_dict)
        except ValueError as e:
            print(f"\n\033[31m✗\033[0m Error creating profile: {e}", file=sys.stderr)
            return 1
    else:
        # Save to default profile
        with open(CONFIG_FILE, "w") as f:
            f.write("# Hindsight Embed Configuration\n")
            f.write("# Generated by hindsight-embed configure\n\n")
            for key, value in config_dict.items():
                f.write(f"{key}={value}\n")
        CONFIG_FILE.chmod(0o600)

    # Stop existing daemon if running (it needs to pick up new config)
    from . import daemon_client

    daemon_profile = profile_name if profile_name else None
    if daemon_client.is_daemon_running(daemon_profile):
        print("\n  \033[2mRestarting daemon with new configuration...\033[0m")
        daemon_client.stop_daemon(daemon_profile)

    # Start daemon with new config
    new_config = {
        "llm_api_key": api_key,
        "llm_provider": provider,
        "llm_model": model,
        "bank_id": bank_id,
    }
    if daemon_client.ensure_daemon_running(new_config, daemon_profile):
        print("  \033[32m✓ Daemon started\033[0m")
    else:
        print("  \033[33m⚠ Failed to start daemon (will start on first command)\033[0m")

    print()
    print("\033[32m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m")
    print("\033[32m  ✓ Configuration saved!\033[0m")
    print("\033[32m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m")
    print()
    print(f"  \033[2mConfig:\033[0m {CONFIG_FILE}")
    print()
    print("  \033[2mTest with:\033[0m")
    print('    \033[36mhindsight-embed retain "Alice works at Google as a software engineer"\033[0m')
    print('    \033[36mhindsight-embed recall "Alice"\033[0m')
    print()

    return 0


def do_daemon(args, config: dict, logger):
    """Handle daemon subcommands."""
    from pathlib import Path

    from . import daemon_client
    from .profile_manager import ProfileManager

    # Get profile from args
    profile = args.profile

    # Get profile-specific paths
    pm = ProfileManager()
    paths = pm.resolve_profile_paths(profile or "")

    daemon_log_path = paths.log
    port = paths.port

    if args.daemon_command == "start":
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        console = Console()

        if daemon_client.is_daemon_running(profile):
            # Build title with profile and port
            if profile:
                already_running_title = (
                    f"[bold yellow]Daemon Already Running[/bold yellow] [dim]({profile} @ :{port})[/dim]"
                )
            else:
                already_running_title = f"[bold yellow]Daemon Already Running[/bold yellow] [dim](:{port})[/dim]"

            console.print(
                Panel(
                    Text("Daemon is already running", style="yellow"),
                    title=already_running_title,
                    border_style="yellow",
                )
            )
            return 0

        if daemon_client.ensure_daemon_running(config, profile):
            return 0
        else:
            console.print(
                Panel(
                    Text("Failed to start daemon", style="red"),
                    title="[bold red]✗ Error[/bold red]",
                    border_style="red",
                )
            )
            return 1

    elif args.daemon_command == "stop":
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        console = Console()

        if not daemon_client.is_daemon_running(profile):
            # Build title for not running status
            if profile:
                not_running_title = f"[bold]Daemon Status[/bold] [dim]({profile})[/dim]"
            else:
                not_running_title = "[bold]Daemon Status[/bold]"

            console.print(
                Panel(
                    Text("Daemon is not running", style="dim"),
                    title=not_running_title,
                    border_style="dim",
                )
            )
            return 0

        if daemon_client.stop_daemon(profile):
            # Build title with profile
            if profile:
                stopped_title = f"[bold green]✓ Daemon Stopped[/bold green] [dim]({profile})[/dim]"
            else:
                stopped_title = "[bold green]✓ Daemon Stopped[/bold green]"

            console.print(
                Panel(
                    Text("Daemon stopped successfully", style="green"),
                    title=stopped_title,
                    border_style="green",
                )
            )
            return 0
        else:
            console.print(
                Panel(
                    Text("Failed to stop daemon", style="red"),
                    title="[bold red]✗ Error[/bold red]",
                    border_style="red",
                )
            )
            return 1

    elif args.daemon_command == "status":
        import os
        from pathlib import Path

        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        console = Console()

        if daemon_client.is_daemon_running(profile):
            status_text = Text()
            status_text.append("Daemon is running\n\n", style="green bold")
            status_text.append("  URL: ", style="dim")
            status_text.append(f"http://127.0.0.1:{port}\n", style="cyan")
            status_text.append("  Logs: ", style="dim")
            status_text.append(f"{daemon_log_path}\n", style="")

            # Check if using pg0 and show database location
            database_url = os.getenv("HINDSIGHT_EMBED_API_DATABASE_URL")
            if not database_url:
                # Default: use profile-specific pg0 (shared utility ensures consistency)
                database_url = get_embed_manager().get_database_url(profile)

            if database_url.startswith("pg0://"):
                pg0_name = database_url.replace("pg0://", "")
                pg0_path = Path.home() / ".pg0" / "instances" / pg0_name
                status_text.append("  Database: ", style="dim")
                status_text.append(f"{pg0_path}", style="")

            # Build title with profile and port
            if profile:
                status_title = f"[bold green]✓ Daemon Running[/bold green] [dim]({profile} @ :{port})[/dim]"
            else:
                status_title = f"[bold green]✓ Daemon Running[/bold green] [dim](:{port})[/dim]"

            console.print(
                Panel(
                    status_text,
                    title=status_title,
                    border_style="green",
                    padding=(1, 2),
                )
            )
            return 0
        else:
            # Build title for not running status
            if profile:
                not_running_title = f"[bold]Daemon Status[/bold] [dim]({profile})[/dim]"
            else:
                not_running_title = "[bold]Daemon Status[/bold]"

            console.print(
                Panel(
                    Text("Daemon is not running", style="dim"),
                    title=not_running_title,
                    border_style="dim",
                )
            )
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


def _do_configure_profile_with_env(profile_name: str, port: int | None, env_vars: list[str]) -> int:
    """Configure a named profile with environment variables (non-interactive).

    Args:
        profile_name: Name of the profile to create/update.
        port: Port number for the daemon (None to auto-allocate/reuse existing).
        env_vars: List of KEY=VALUE strings.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    from .profile_manager import ProfileManager

    # Parse env vars
    config = {}
    for env_str in env_vars:
        if "=" not in env_str:
            print(f"Error: Invalid --env format '{env_str}'. Expected KEY=VALUE", file=sys.stderr)
            return 1

        key, value = env_str.split("=", 1)
        key = key.strip()
        value = value.strip()

        # Validate key format
        if not key.startswith("HINDSIGHT_EMBED_") and not key.startswith("HINDSIGHT_API_"):
            print(
                f"Warning: Key '{key}' doesn't start with HINDSIGHT_EMBED_ or HINDSIGHT_API_",
                file=sys.stderr,
            )

        config[key] = value

    # Create profile
    pm = ProfileManager()

    # Determine port: use provided, reuse existing, or allocate new
    if port is None:
        # Check if profile exists and get its port
        existing_profile = pm.get_profile(profile_name)
        if existing_profile:
            port = existing_profile.port
        else:
            port = pm._allocate_port(profile_name)

    try:
        pm.create_profile(profile_name, port, config)
    except ValueError as e:
        print(f"Error creating profile: {e}", file=sys.stderr)
        return 1

    print()
    print(f"\033[32m✓ Profile '{profile_name}' configured successfully!\033[0m")
    print()
    profile_path = CONFIG_DIR / "profiles" / f"{profile_name}.env"
    print(f"  \033[2mConfig:\033[0m {profile_path}")
    print(f"  \033[2mPort:\033[0m {port}")
    print()
    print("  \033[2mUse with:\033[0m")
    print(f"    \033[36mhindsight-embed daemon start --profile {profile_name}\033[0m")
    print(f'    \033[36mhindsight-embed --profile {profile_name} memory recall default "query"\033[0m')
    print()

    return 0


def _do_configure_profile_interactive(profile_name: str, port: int | None) -> int:
    """Configure a named profile interactively.

    Args:
        profile_name: Name of the profile to create/update.
        port: Port number for the daemon (None to auto-allocate/reuse existing).

    Returns:
        Exit code (0 = success, 1 = error).
    """
    from .profile_manager import ProfileManager

    # Determine port: use provided, reuse existing, or allocate new
    if port is None:
        pm = ProfileManager()
        # Check if profile exists and get its port
        existing_profile = pm.get_profile(profile_name)
        if existing_profile:
            port = existing_profile.port
        else:
            port = pm._allocate_port(profile_name)

    print()
    print(f"\033[1m\033[36m  Configuring profile '{profile_name}' (port {port})\033[0m")
    print()

    # Use the same interactive flow as default profile but save to named profile
    return _do_configure_interactive(profile_name, port)


def do_profile_command(args: list[str]) -> int:
    """Handle profile subcommands.

    Args:
        args: Command arguments (after 'profile').

    Returns:
        Exit code (0 = success, 1 = error).
    """
    from .profile_manager import ProfileManager, resolve_active_profile, validate_profile_exists

    parser = argparse.ArgumentParser(prog="hindsight-embed profile")
    subparsers = parser.add_subparsers(dest="profile_command", required=True)

    # List command
    list_parser = subparsers.add_parser("list", help="List all profiles")
    list_parser.add_argument(
        "-o", "--output", choices=["text", "json"], default="text", help="Output format (text or json)"
    )

    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new profile")
    create_parser.add_argument("name", help="Profile name")
    create_parser.add_argument("--port", type=int, required=True, help="Port for the daemon")
    create_parser.add_argument("--env", action="append", help="Environment variable (KEY=VALUE, can be repeated)")
    create_parser.add_argument("--merge", action="store_true", help="Merge with existing profile if it exists")

    # Set-env command
    set_env_parser = subparsers.add_parser("set-env", help="Set/update an environment variable in a profile")
    set_env_parser.add_argument("name", help="Profile name")
    set_env_parser.add_argument("key", help="Environment variable key")
    set_env_parser.add_argument("value", help="Environment variable value")

    # Remove-env command
    remove_env_parser = subparsers.add_parser("remove-env", help="Remove an environment variable from a profile")
    remove_env_parser.add_argument("name", help="Profile name")
    remove_env_parser.add_argument("key", help="Environment variable key to remove")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a profile")
    delete_parser.add_argument("name", help="Profile name to delete")

    # Set-active command
    set_active_parser = subparsers.add_parser("set-active", help="Set active profile")
    set_active_parser.add_argument("name", nargs="?", help="Profile name (omit to clear)")
    set_active_parser.add_argument("--none", action="store_true", help="Clear active profile")

    # Show command
    show_parser = subparsers.add_parser("show", help="Show current active profile")
    show_parser.add_argument(
        "-o", "--output", choices=["text", "json"], default="text", help="Output format (text or json)"
    )

    try:
        parsed_args = parser.parse_args(args)
    except SystemExit as e:
        return e.code or 1

    pm = ProfileManager()

    if parsed_args.profile_command == "list":
        # List all profiles
        profiles = pm.list_profiles()

        if parsed_args.output == "json":
            # JSON output
            import json

            profiles_data = []
            for profile in profiles:
                config_path = str(CONFIG_DIR / "profiles" / f"{profile.name}.env") if profile.name else str(CONFIG_FILE)
                profiles_data.append(
                    {
                        "name": profile.name or "default",
                        "port": profile.port,
                        "config": config_path,
                        "created_at": profile.created_at,
                        "last_used": profile.last_used,
                        "is_active": profile.is_active,
                        "daemon_running": profile.daemon_running,
                    }
                )
            print(json.dumps(profiles_data, indent=2))
            return 0

        # Text output
        if not profiles:
            print("No profiles configured.")
            print()
            print("Create one with:")
            print("  hindsight-embed configure --profile my-app --port 9100 --env HINDSIGHT_API_LLM_PROVIDER=...")
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

    elif parsed_args.profile_command == "create":
        # Create new profile
        profile_name = parsed_args.name
        port = parsed_args.port
        env_vars = parsed_args.env or []
        merge = parsed_args.merge

        # Normalize "default" to empty string
        if profile_name == "default":
            profile_name = ""

        # Check if profile exists
        profile_exists = pm.profile_exists(profile_name)
        if profile_exists and not merge:
            display_name = profile_name or "default"
            print(f"Error: Profile '{display_name}' already exists.", file=sys.stderr)
            print("  Use --merge to update the profile, or delete it first with:", file=sys.stderr)
            print(f"  hindsight-embed profile delete {display_name}", file=sys.stderr)
            return 1

        # Parse new env vars
        new_config = {}
        for env_str in env_vars:
            if "=" not in env_str:
                print(f"Error: Invalid --env format '{env_str}'. Expected KEY=VALUE", file=sys.stderr)
                return 1
            key, value = env_str.split("=", 1)
            new_config[key.strip()] = value.strip()

        # If merging, read existing config and merge
        config = {}
        if merge and profile_exists:
            if profile_name:
                config_path = CONFIG_DIR / "profiles" / f"{profile_name}.env"
            else:
                config_path = CONFIG_FILE

            if config_path.exists():
                for line in config_path.read_text().splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        if k != "PORT":  # Don't copy PORT, we'll set it explicitly
                            config[k] = v

        # Merge new config into existing
        config.update(new_config)

        # Create/update profile
        try:
            pm.create_profile(profile_name, port, config)
            display_name = profile_name or "default"
            action = "updated" if (merge and profile_exists) else "created"
            print(f"\033[32m✓\033[0m Profile '{display_name}' {action} successfully!")
            print()
            if profile_name:
                config_path = CONFIG_DIR / "profiles" / f"{profile_name}.env"
            else:
                config_path = CONFIG_FILE
            print(f"  \033[2mConfig:\033[0m {config_path}")
            print(f"  \033[2mPort:\033[0m {port}")
            return 0
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    elif parsed_args.profile_command == "set-env":
        # Set/update env variable in profile
        profile_name = parsed_args.name
        key = parsed_args.key
        value = parsed_args.value

        # Normalize "default" to empty string
        if profile_name == "default":
            profile_name = ""

        if not pm.profile_exists(profile_name):
            display_name = profile_name or "default"
            print(f"Error: Profile '{display_name}' does not exist.", file=sys.stderr)
            return 1

        # Read current config
        if profile_name:
            config_path = CONFIG_DIR / "profiles" / f"{profile_name}.env"
        else:
            config_path = CONFIG_FILE

        # Parse existing config
        config = {}
        if config_path.exists():
            for line in config_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    config[k] = v

        # Update the key
        config[key] = value

        # Get port from config or resolve from profile
        port = int(config.get("PORT", pm.resolve_profile_paths(profile_name).port))

        # Write back
        try:
            pm.create_profile(profile_name, port, {k: v for k, v in config.items() if k != "PORT"})
            display_name = profile_name or "default"
            print(f"\033[32m✓\033[0m Set {key}={value} in profile '{display_name}'")
            return 0
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    elif parsed_args.profile_command == "remove-env":
        # Remove env variable from profile
        profile_name = parsed_args.name
        key = parsed_args.key

        # Normalize "default" to empty string
        if profile_name == "default":
            profile_name = ""

        if not pm.profile_exists(profile_name):
            display_name = profile_name or "default"
            print(f"Error: Profile '{display_name}' does not exist.", file=sys.stderr)
            return 1

        # Read current config
        if profile_name:
            config_path = CONFIG_DIR / "profiles" / f"{profile_name}.env"
        else:
            config_path = CONFIG_FILE

        # Parse existing config
        config = {}
        if config_path.exists():
            for line in config_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    config[k] = v

        # Remove the key
        if key not in config:
            display_name = profile_name or "default"
            print(f"Error: Key '{key}' not found in profile '{display_name}'", file=sys.stderr)
            return 1

        del config[key]

        # Get port from config or resolve from profile
        port = int(config.get("PORT", pm.resolve_profile_paths(profile_name).port))

        # Write back
        try:
            pm.create_profile(profile_name, port, {k: v for k, v in config.items() if k != "PORT"})
            display_name = profile_name or "default"
            print(f"\033[32m✓\033[0m Removed {key} from profile '{display_name}'")
            return 0
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    elif parsed_args.profile_command == "delete":
        # Delete profile
        profile_name = parsed_args.name
        # Normalize "default" to empty string
        if profile_name == "default":
            profile_name = ""

        if not pm.profile_exists(profile_name):
            display_name = profile_name or "default"
            print(f"Error: Profile '{display_name}' does not exist.", file=sys.stderr)
            return 1

        # Check if daemon is running
        profile_info = pm.get_profile(profile_name)
        if profile_info and profile_info.daemon_running:
            display_name = profile_name or "default"
            print(f"Warning: Daemon is running for profile '{display_name}'")
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
            display_name = profile_name or "default"
            print(f"\033[32m✓\033[0m Profile '{display_name}' deleted.")
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

        display_name = active_profile if active_profile else "default"

        # Determine source
        source = "default"
        if not active_profile:
            source = "default"
        elif os.getenv("HINDSIGHT_EMBED_PROFILE"):
            source = "HINDSIGHT_EMBED_PROFILE"
        elif get_cli_profile_override():
            source = "cli_flag"
        elif pm.get_active_profile():
            source = "active_profile_file"

        # Get config path
        paths = pm.resolve_profile_paths(active_profile)

        if parsed_args.output == "json":
            # JSON output
            import json

            data = {
                "name": display_name,
                "source": source,
                "config": str(paths.config),
                "port": paths.port,
            }
            print(json.dumps(data, indent=2))
            return 0

        # Text output
        print()
        print(f"\033[1mActive profile:\033[0m {display_name}")
        print()

        if source == "default":
            print("  \033[2mSource:\033[0m Default (no profile specified)")
        elif source == "HINDSIGHT_EMBED_PROFILE":
            print("  \033[2mSource:\033[0m HINDSIGHT_EMBED_PROFILE environment variable")
        elif source == "cli_flag":
            print("  \033[2mSource:\033[0m --profile flag")
        elif source == "active_profile_file":
            print("  \033[2mSource:\033[0m Active profile file")

        print(f"  \033[2mConfig:\033[0m {paths.config}")
        print(f"  \033[2mPort:\033[0m {paths.port}")
        print()

        return 0

    return 1


def main():
    """Main entry point."""
    # Use argparse to properly parse global flags
    # Create a parent parser for global --profile/-p flag
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("-p", "--profile", help="Profile name to use")

    # Parse known args to extract --profile value
    global_args, remaining_args = parent_parser.parse_known_args()
    global_profile = global_args.profile
    if global_profile == "default":
        global_profile = None

    # Check for built-in commands first
    # Find the first non-flag argument (the actual command)
    command = None
    if remaining_args:
        command = remaining_args[0]

        # Handle configure
        if command == "configure":
            # Parse configure arguments
            parser = argparse.ArgumentParser(prog="hindsight-embed configure")
            parser.add_argument("-p", "--profile", help="Profile name to create/update")
            parser.add_argument(
                "--port",
                type=int,
                help="Port for the daemon (required for named profiles, default profile uses 8888)",
            )
            parser.add_argument(
                "--env",
                action="append",
                help="Environment variable (KEY=VALUE, can be repeated)",
            )
            args = parser.parse_args(remaining_args[1:])  # Skip 'configure' itself

            # If --profile was consumed by parent_parser, use global_profile
            if not args.profile and global_profile:
                args.profile = global_profile

            logger = setup_logging(False)
            exit_code = do_configure(args)
            sys.exit(exit_code)

        # Handle profile subcommands
        if command == "profile":
            exit_code = do_profile_command(remaining_args[1:])  # Skip 'profile' itself
            sys.exit(exit_code)

        # Handle daemon subcommands
        if command == "daemon":
            # Parse daemon subcommand (profile already extracted globally)
            parser = argparse.ArgumentParser(prog="hindsight-embed daemon")
            subparsers = parser.add_subparsers(dest="daemon_command")
            subparsers.add_parser("start", help="Start the daemon")
            subparsers.add_parser("stop", help="Stop the daemon")
            subparsers.add_parser("status", help="Check daemon status")
            logs_parser = subparsers.add_parser("logs", help="View daemon logs")
            logs_parser.add_argument("--follow", "-f", action="store_true")
            logs_parser.add_argument("--lines", "-n", type=int, default=50)

            args = parser.parse_args(remaining_args[1:])  # Skip 'daemon' itself
            # Use globally extracted profile
            args.profile = global_profile
            logger = setup_logging(False)
            config = get_config()
            exit_code = do_daemon(args, config, logger)
            sys.exit(exit_code)

        # Handle --help / -h
        if command in ("--help", "-h"):
            print_help()
            sys.exit(0)

        # Check for common mistakes - these are daemon subcommands, not top-level commands
        if command in ("start", "stop", "status", "logs"):
            print(f"error: '{command}' is not a direct command", file=sys.stderr)
            print(f"\nDid you mean: hindsight-embed daemon {command}", file=sys.stderr)
            if global_profile:
                print(f"              (with --profile {global_profile})", file=sys.stderr)
            sys.exit(1)

        # Forward all other commands to hindsight-cli
        config = get_config()

        # Check for LLM API key
        if not config["llm_api_key"]:
            print("Error: LLM API key is required.", file=sys.stderr)
            print("Run 'hindsight-embed configure' to set up.", file=sys.stderr)
            sys.exit(1)

        from . import daemon_client

        # Forward to hindsight-cli (handles daemon startup and CLI installation)
        # Pass the globally extracted profile
        # remaining_args already has --profile/-p filtered out
        exit_code = daemon_client.run_cli(remaining_args, config, global_profile)
        sys.exit(exit_code)

    # No command - show help
    print_help()
    sys.exit(1)


def print_help():
    """Print help message."""
    print("""Hindsight Embedded CLI - local memory operations with automatic daemon management.

Usage: hindsight-embed [-p PROFILE] <command> [options]

Profile management:
    profile create NAME --port PORT [--env KEY=VALUE ...]   Create a new profile
    profile set-env NAME KEY VALUE                          Set/update environment variable
    profile remove-env NAME KEY                             Remove environment variable
    profile list [-o json]                                  List all profiles
    profile show [-o json]                                  Show current active profile
    profile set-active NAME                                 Set active profile
    profile delete NAME                                     Delete a profile

Daemon management:
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

Global options:
    -p, --profile PROFILE  Profile to use for commands

Examples:
    # Create a profile
    hindsight-embed profile create my-app --port 9100 --env HINDSIGHT_API_LLM_PROVIDER=openai

    # Manage environment variables
    hindsight-embed profile set-env my-app HINDSIGHT_API_LLM_MODEL gpt-4
    hindsight-embed profile remove-env my-app HINDSIGHT_API_LLM_MODEL

    # Use profile with commands
    hindsight-embed -p my-app daemon start
    hindsight-embed -p my-app memory retain default "User prefers dark mode"
    hindsight-embed --profile my-app bank list

Note: 'configure' command is deprecated, use 'profile create' instead.
""")


if __name__ == "__main__":
    main()
