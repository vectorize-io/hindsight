"""CLI for the Hindsight Zed integration.

One-time setup installs a small background daemon that watches Zed's thread
database and, per project, keeps a recalled-memory block fresh in the
instruction file Zed always reads (auto-recall) and retains conversations
(auto-retain). After ``init`` it is fully hands-off — no per-project steps.
"""

import argparse
import json
import plistlib
import subprocess
import sys
from pathlib import Path
from typing import Optional

from . import __version__
from .config import USER_CONFIG_FILE, load_config

LAUNCHD_LABEL = "io.vectorize.hindsight-zed"
LAUNCHD_PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"
SYSTEMD_UNIT = Path.home() / ".config" / "systemd" / "user" / "hindsight-zed.service"
LOG_DIR = Path.home() / ".hindsight"


def _daemon_command() -> list:
    """The command the service runs: this CLI's ``run`` subcommand."""
    # Use the same interpreter that's running so the install works under uv/venv.
    return [sys.executable, "-m", "hindsight_zed.cli", "run"]


def _scaffold_config(api_url: Optional[str], api_token: Optional[str], fixed_bank_id: Optional[str]) -> None:
    if USER_CONFIG_FILE.is_file():
        print(f"  Config already exists at {USER_CONFIG_FILE}, leaving as-is.")
        return
    config: dict = {}
    if api_url:
        config["hindsightApiUrl"] = api_url
    if api_token:
        config["hindsightApiToken"] = api_token
    if fixed_bank_id:
        config["fixedBankId"] = fixed_bank_id
    USER_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    USER_CONFIG_FILE.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"  Wrote config to {USER_CONFIG_FILE}")


def _install_launchd() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    plist = {
        "Label": LAUNCHD_LABEL,
        "ProgramArguments": _daemon_command(),
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(LOG_DIR / "zed-daemon.log"),
        "StandardErrorPath": str(LOG_DIR / "zed-daemon.log"),
    }
    LAUNCHD_PLIST.parent.mkdir(parents=True, exist_ok=True)
    with open(LAUNCHD_PLIST, "wb") as f:
        plistlib.dump(plist, f)
    # Reload if already loaded, then load.
    subprocess.run(["launchctl", "unload", str(LAUNCHD_PLIST)], capture_output=True)
    result = subprocess.run(["launchctl", "load", str(LAUNCHD_PLIST)], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  warning: launchctl load failed: {result.stderr.strip()}", file=sys.stderr)
    else:
        print(f"  Installed and started launchd agent ({LAUNCHD_LABEL})")


def _install_systemd() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cmd = " ".join(_daemon_command())
    unit = (
        "[Unit]\n"
        "Description=Hindsight memory daemon for Zed\n\n"
        "[Service]\n"
        f"ExecStart={cmd}\n"
        "Restart=always\n"
        "RestartSec=5\n\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )
    SYSTEMD_UNIT.parent.mkdir(parents=True, exist_ok=True)
    SYSTEMD_UNIT.write_text(unit, encoding="utf-8")
    subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
    result = subprocess.run(
        ["systemctl", "--user", "enable", "--now", "hindsight-zed.service"], capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  warning: systemctl enable failed: {result.stderr.strip()}", file=sys.stderr)
        print("  (you may need: systemctl --user enable --now hindsight-zed.service)")
    else:
        print("  Installed and started systemd user service (hindsight-zed.service)")


def _install_daemon() -> None:
    if sys.platform == "darwin":
        _install_launchd()
    elif sys.platform.startswith("linux"):
        _install_systemd()
    else:
        print(f"  Automatic daemon install isn't supported on {sys.platform}.")
        print(f"  Run it yourself (e.g. at login): {' '.join(_daemon_command())}")


def _uninstall_daemon() -> None:
    if sys.platform == "darwin" and LAUNCHD_PLIST.exists():
        subprocess.run(["launchctl", "unload", str(LAUNCHD_PLIST)], capture_output=True)
        LAUNCHD_PLIST.unlink()
        print(f"  Removed launchd agent ({LAUNCHD_LABEL})")
    elif sys.platform.startswith("linux") and SYSTEMD_UNIT.exists():
        subprocess.run(["systemctl", "--user", "disable", "--now", "hindsight-zed.service"], capture_output=True)
        SYSTEMD_UNIT.unlink()
        subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
        print("  Removed systemd user service")
    else:
        print("  No installed daemon found.")


def cmd_init(args: argparse.Namespace) -> None:
    print("Setting up Hindsight for Zed ...")
    _scaffold_config(args.api_url, args.api_token, args.fixed_bank_id)

    # Fail fast if the server is unreachable, so the user fixes config now.
    cfg = load_config()
    client_ok = True
    try:
        from .client import HindsightClient

        client_ok = HindsightClient(cfg.hindsight_api_url, cfg.hindsight_api_token).health_check()
    except Exception:
        client_ok = False
    if not client_ok:
        print(f"  warning: could not reach Hindsight at {cfg.hindsight_api_url}.")
        print("  Check --api-url / --api-token; the daemon will keep retrying.")

    if not args.no_daemon:
        _install_daemon()

    print()
    print("Done. Hindsight now runs automatically for every project you open in Zed.")
    print("Open a project, chat with the Agent Panel, and relevant memory is injected")
    print("via the project's instruction file; conversations are retained as you go.")


def cmd_uninstall(args: argparse.Namespace) -> None:
    _uninstall_daemon()


def cmd_run(args: argparse.Namespace) -> None:
    from .daemon import run

    run()


def cmd_status(args: argparse.Namespace) -> None:
    if sys.platform == "darwin":
        out = subprocess.run(["launchctl", "list", LAUNCHD_LABEL], capture_output=True, text=True)
        print("running" if out.returncode == 0 else "not running")
    elif sys.platform.startswith("linux"):
        out = subprocess.run(
            ["systemctl", "--user", "is-active", "hindsight-zed.service"], capture_output=True, text=True
        )
        print(out.stdout.strip() or "unknown")
    else:
        print("status unavailable on this platform")


def main() -> None:
    parser = argparse.ArgumentParser(prog="hindsight-zed", description="Hindsight memory for Zed")
    parser.add_argument("--version", action="version", version=f"hindsight-zed {__version__}")
    sub = parser.add_subparsers(dest="command")

    init_p = sub.add_parser("init", help="Set up the daemon and config (one-time)")
    init_p.add_argument("--api-url", default=None, help="Hindsight API URL (default: cloud)")
    init_p.add_argument("--api-token", default=None, help="Hindsight API token")
    init_p.add_argument(
        "--fixed-bank-id", default=None, help="Use one shared bank for all projects (default: per-project)"
    )
    init_p.add_argument("--no-daemon", action="store_true", help="Write config but don't install the daemon")
    init_p.set_defaults(func=cmd_init)

    run_p = sub.add_parser("run", help="Run the daemon in the foreground (used by the service)")
    run_p.set_defaults(func=cmd_run)

    status_p = sub.add_parser("status", help="Show whether the daemon is running")
    status_p.set_defaults(func=cmd_status)

    uninst_p = sub.add_parser("uninstall", help="Stop and remove the daemon")
    uninst_p.set_defaults(func=cmd_uninstall)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
