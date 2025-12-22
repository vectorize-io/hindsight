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
"""

import argparse
import logging
import os
import sys
from pathlib import Path

CONFIG_DIR = Path.home() / ".hindsight"
CONFIG_FILE = CONFIG_DIR / "embed"
CONFIG_FILE_ALT = CONFIG_DIR / "config.env"  # Alternative config file location


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
        "llm_api_key": os.environ.get("HINDSIGHT_EMBED_LLM_API_KEY")
            or os.environ.get("HINDSIGHT_API_LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY"),
        "llm_provider": os.environ.get("HINDSIGHT_EMBED_LLM_PROVIDER")
            or os.environ.get("HINDSIGHT_API_LLM_PROVIDER", "openai"),
        "llm_model": os.environ.get("HINDSIGHT_EMBED_LLM_MODEL")
            or os.environ.get("HINDSIGHT_API_LLM_MODEL", "gpt-4o-mini"),
        "bank_id": os.environ.get("HINDSIGHT_EMBED_BANK_ID", "default"),
    }


def do_configure(args):
    """Interactive configuration setup with beautiful TUI."""
    import questionary
    from questionary import Style

    # If stdin is not a terminal (e.g., running via curl | bash),
    # reopen stdin from /dev/tty for interactive prompts
    if not sys.stdin.isatty():
        try:
            sys.stdin = open('/dev/tty', 'r')
        except OSError:
            print("Error: Cannot run interactive configuration without a terminal.", file=sys.stderr)
            print("Run directly: uvx hindsight-embed configure", file=sys.stderr)
            return 1

    # Custom style for the prompts
    custom_style = Style([
        ('qmark', 'fg:cyan bold'),
        ('question', 'fg:white bold'),
        ('answer', 'fg:cyan'),
        ('pointer', 'fg:cyan bold'),
        ('highlighted', 'fg:cyan bold'),
        ('selected', 'fg:green'),
        ('text', 'fg:white'),
    ])

    print()
    print("\033[1m\033[36m  ╭─────────────────────────────────────╮\033[0m")
    print("\033[1m\033[36m  │   Hindsight Embed Configuration    │\033[0m")
    print("\033[1m\033[36m  ╰─────────────────────────────────────╯\033[0m")
    print()

    # Check existing config
    if CONFIG_FILE.exists():
        if not questionary.confirm(
            "Existing configuration found. Reconfigure?",
            default=False,
            style=custom_style,
        ).ask():
            print("\n\033[32m✓\033[0m Keeping existing configuration.")
            return 0
        print()

    # Provider selection with descriptions
    providers = [
        questionary.Choice("OpenAI (recommended)", value=("openai", "o3-mini", "OpenAI")),
        questionary.Choice("Groq (fast & free tier)", value=("groq", "openai/gpt-oss-20b", "Groq")),
        questionary.Choice("Google Gemini", value=("google", "gemini-2.0-flash", "Google")),
        questionary.Choice("Ollama (local, no API key)", value=("ollama", "llama3.2", None)),
    ]

    result = questionary.select(
        "Select your LLM provider:",
        choices=providers,
        style=custom_style,
    ).ask()

    if result is None:  # User cancelled
        print("\n\033[33m⚠\033[0m Configuration cancelled.")
        return 1

    provider, default_model, key_name = result

    # API key
    api_key = ""
    if key_name:
        env_keys = {
            "OpenAI": "OPENAI_API_KEY",
            "Groq": "GROQ_API_KEY",
            "Google": "GOOGLE_API_KEY",
        }
        env_key = env_keys.get(key_name, "")
        existing = os.environ.get(env_key, "")

        if existing:
            masked = existing[:8] + "..." + existing[-4:] if len(existing) > 12 else "***"
            if questionary.confirm(
                f"Found {key_name} key in ${env_key} ({masked}). Use it?",
                default=True,
                style=custom_style,
            ).ask():
                api_key = existing

        if not api_key:
            api_key = questionary.password(
                f"Enter your {key_name} API key:",
                style=custom_style,
            ).ask()

            if not api_key:
                print("\n\033[31m✗\033[0m API key is required.", file=sys.stderr)
                return 1

    # Model selection
    model = questionary.text(
        "Model name:",
        default=default_model,
        style=custom_style,
    ).ask()

    if model is None:
        return 1

    # Bank ID
    bank_id = questionary.text(
        "Memory bank ID:",
        default="default",
        style=custom_style,
    ).ask()

    if bank_id is None:
        return 1

    # Save configuration
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_FILE, "w") as f:
        f.write("# Hindsight Embed Configuration\n")
        f.write(f"# Generated by hindsight-embed configure\n\n")
        f.write(f"HINDSIGHT_EMBED_LLM_PROVIDER={provider}\n")
        f.write(f"HINDSIGHT_EMBED_LLM_MODEL={model}\n")
        f.write(f"HINDSIGHT_EMBED_BANK_ID={bank_id}\n")
        if api_key:
            f.write(f"HINDSIGHT_EMBED_LLM_API_KEY={api_key}\n")

    CONFIG_FILE.chmod(0o600)

    # Stop existing daemon if running (it needs to pick up new config)
    from . import daemon_client

    if daemon_client._is_daemon_running():
        print("\n  \033[2mRestarting daemon with new configuration...\033[0m")
        daemon_client.stop_daemon()

    # Start daemon with new config
    new_config = {
        "llm_api_key": api_key,
        "llm_provider": provider,
        "llm_model": model,
        "bank_id": bank_id,
    }
    if daemon_client.ensure_daemon_running(new_config):
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
    print('    \033[36mhindsight-embed retain "Test memory"\033[0m')
    print('    \033[36mhindsight-embed recall "test"\033[0m')
    print()

    return 0


def do_daemon(args, config: dict, logger):
    """Handle daemon subcommands."""
    from pathlib import Path
    from . import daemon_client

    daemon_log_path = Path.home() / ".hindsight" / "daemon.log"

    if args.daemon_command == "start":
        if daemon_client._is_daemon_running():
            print("Daemon is already running")
            return 0

        print("Starting daemon...")
        if daemon_client.ensure_daemon_running(config):
            print("Daemon started successfully")
            print(f"  Port: {daemon_client.DAEMON_PORT}")
            print(f"  Logs: {daemon_log_path}")
            return 0
        else:
            print("Failed to start daemon", file=sys.stderr)
            return 1

    elif args.daemon_command == "stop":
        if not daemon_client._is_daemon_running():
            print("Daemon is not running")
            return 0

        print("Stopping daemon...")
        if daemon_client.stop_daemon():
            print("Daemon stopped")
            return 0
        else:
            print("Failed to stop daemon", file=sys.stderr)
            return 1

    elif args.daemon_command == "status":
        if daemon_client._is_daemon_running():
            # Get PID from lockfile
            lockfile = Path.home() / ".hindsight" / "daemon.lock"
            pid = "unknown"
            if lockfile.exists():
                try:
                    pid = lockfile.read_text().strip()
                except Exception:
                    pass
            print(f"Daemon is running (PID: {pid})")
            print(f"  URL: http://127.0.0.1:{daemon_client.DAEMON_PORT}")
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
                    for line in lines[-args.lines:]:
                        print(line, end="")
                return 0
            except Exception as e:
                print(f"Error reading logs: {e}", file=sys.stderr)
                return 1

    else:
        print("Usage: hindsight-embed daemon {start|stop|status|logs}", file=sys.stderr)
        return 1


def main():
    """Main entry point."""
    # Check for built-in commands first (before argparse)
    # This allows us to forward unknown commands to hindsight-cli
    if len(sys.argv) > 1:
        command = sys.argv[1]

        # Handle configure
        if command == "configure":
            logger = setup_logging(False)
            exit_code = do_configure(None)
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

            args = parser.parse_args(sys.argv[2:])
            logger = setup_logging(False)
            config = get_config()
            exit_code = do_daemon(args, config, logger)
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
        exit_code = daemon_client.run_cli(sys.argv[1:], config)
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
