#!/usr/bin/env python3
"""Install Hindsight memory for Cline.

Copies the lifecycle-hook scripts (+ their lib and settings) into Cline's
hooks directory and records your connection settings. Cline runs the hooks to
auto-recall relevant memories before each task/prompt and auto-retain what
happened when a task ends.

Usage:
    python install.py                              # into ./.clinerules/hooks/
    python install.py --api-url https://api.hindsight.vectorize.io --api-token KEY
    python install.py --project-dir /path/to/project
    python install.py --global                     # into ~/Documents/Cline/Rules/Hooks/

After installing, enable hooks in Cline: Settings → Features → Hooks.
Cline hooks run on macOS and Linux only.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
HOOKS_SRC = SCRIPT_DIR / "hooks"
SETTINGS_SRC = SCRIPT_DIR / "settings.json"
HOOK_FILES = ["TaskStart", "UserPromptSubmit", "TaskComplete", "TaskCancel"]

DEFAULT_API_URL = "https://api.hindsight.vectorize.io"


def get_hooks_dir(project_dir: Path, global_install: bool) -> Path:
    if global_install:
        return Path.home() / "Documents" / "Cline" / "Rules" / "Hooks"
    return project_dir / ".clinerules" / "hooks"


def install_hooks(hooks_dir: Path) -> None:
    hooks_dir.mkdir(parents=True, exist_ok=True)

    for name in HOOK_FILES:
        dest = hooks_dir / name
        shutil.copy2(HOOKS_SRC / name, dest)
        dest.chmod(0o755)  # Cline only runs executable hook files

    # Shared library + plugin defaults sit alongside the hook files; the hook
    # scripts add this directory to sys.path and read settings.json from here.
    lib_dest = hooks_dir / "lib"
    if lib_dest.exists():
        shutil.rmtree(lib_dest)
    shutil.copytree(HOOKS_SRC / "lib", lib_dest)
    shutil.copy2(SETTINGS_SRC, hooks_dir / "settings.json")
    print(f"Hooks installed: {hooks_dir}")


def write_user_config(api_url: str, api_token: str) -> None:
    """Persist connection settings to ~/.hindsight/cline.json (stable across updates)."""
    if not api_url and not api_token:
        return
    config_dir = Path.home() / ".hindsight"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "cline.json"

    existing: dict = {}
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text())
        except json.JSONDecodeError as e:
            print(f"Warning: could not parse {config_path}: {e}. Overwriting.", file=sys.stderr)

    if api_url:
        existing["hindsightApiUrl"] = api_url.rstrip("/")
    if api_token:
        existing["hindsightApiToken"] = api_token

    config_path.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"Connection settings written: {config_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Install Hindsight memory for Cline.")
    parser.add_argument(
        "--api-url",
        default=os.environ.get("HINDSIGHT_API_URL", ""),
        help=f"Hindsight API base URL (e.g. {DEFAULT_API_URL})",
    )
    parser.add_argument("--api-token", default=os.environ.get("HINDSIGHT_API_TOKEN", ""), help="Hindsight API key")
    parser.add_argument("--project-dir", default=".", help="Project directory (default: current)")
    parser.add_argument(
        "--global",
        dest="global_install",
        action="store_true",
        help="Install to ~/Documents/Cline/Rules/Hooks/ instead of the project",
    )
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    hooks_dir = get_hooks_dir(project_dir, args.global_install)

    print("Installing Hindsight memory for Cline...")
    print(f"  Hooks dir : {hooks_dir}")
    print(f"  API URL   : {args.api_url or '(set later in ~/.hindsight/cline.json)'}")
    print()

    install_hooks(hooks_dir)
    write_user_config(args.api_url, args.api_token)

    print()
    print("Done. Final step — enable hooks in Cline:")
    print("  Settings → Features → Hooks (toggle on)")
    print()
    print("Note: Cline hooks run on macOS and Linux only.")


if __name__ == "__main__":
    main()
