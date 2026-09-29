"""Install logic for the Cline Hindsight integration.

Copies the lifecycle-hook scripts (+ their ``lib/`` and ``settings.json``) into
Cline's hooks directory and records your connection settings. Cline runs the
hooks to auto-recall relevant memories before each task/prompt and auto-retain
what happened when a task ends.

The hook payload (the four hook scripts, ``lib/``, ``settings.json``) ships as
package data under ``hindsight_cline/hooks`` and is read via
``importlib.resources`` so it resolves whether installed as a wheel or run from
a source checkout.

Cline hooks run on macOS and Linux only.
"""

import json
import shutil
import sys
from importlib import resources
from pathlib import Path

PACKAGE = "hindsight_cline"
HOOK_FILES = ["TaskStart", "UserPromptSubmit", "TaskComplete", "TaskCancel"]

DEFAULT_API_URL = "https://api.hindsight.vectorize.io"


def _payload_root():
    """The packaged hook payload (hook scripts + ``lib/`` + ``settings.json``)."""
    return resources.files(PACKAGE).joinpath("hooks")


def get_hooks_dir(project_dir: Path, global_install: bool) -> Path:
    if global_install:
        return Path.home() / "Documents" / "Cline" / "Rules" / "Hooks"
    return project_dir / ".clinerules" / "hooks"


def install_hooks(hooks_dir: Path) -> None:
    """Deploy the packaged hook scripts, ``lib/`` and ``settings.json``."""
    hooks_dir.mkdir(parents=True, exist_ok=True)
    payload = _payload_root()

    for name in HOOK_FILES:
        dest = hooks_dir / name
        dest.write_text(payload.joinpath(name).read_text())
        dest.chmod(0o755)  # Cline only runs executable hook files

    # Shared library sits alongside the hook files; the hook scripts add this
    # directory to sys.path and read settings.json from here.
    lib_dest = hooks_dir / "lib"
    if lib_dest.exists():
        shutil.rmtree(lib_dest)
    # as_file materialises the packaged dir on disk (no-op for a source install,
    # a real extraction for a zipped wheel) so we can copytree it.
    with resources.as_file(payload.joinpath("lib")) as lib_src:
        shutil.copytree(lib_src, lib_dest)

    (hooks_dir / "settings.json").write_text(payload.joinpath("settings.json").read_text())
    print(f"Hooks installed: {hooks_dir}")


def write_user_config(api_url: str | None, api_token: str | None) -> None:
    """Persist connection settings to ``~/.hindsight/cline.json`` (stable across updates)."""
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


def run_install(
    api_url: str | None = None,
    api_token: str | None = None,
    project_dir: Path | None = None,
    global_install: bool = False,
) -> None:
    """Install the hook scripts into Cline and record connection settings."""
    hooks_dir = get_hooks_dir((project_dir or Path(".")).resolve(), global_install)

    print("Installing Hindsight memory for Cline...")
    print(f"  Hooks dir : {hooks_dir}")
    print(f"  API URL   : {api_url or '(set later in ~/.hindsight/cline.json)'}")
    print()

    install_hooks(hooks_dir)
    write_user_config(api_url, api_token)

    print()
    print("Done. Final step — enable hooks in Cline:")
    print("  Settings → Features → Hooks (toggle on)")
    print()
    print("Note: Cline hooks run on macOS and Linux only.")


def run_uninstall(project_dir: Path | None = None, global_install: bool = False) -> None:
    """Remove the deployed hook scripts, ``lib/`` and ``settings.json``."""
    hooks_dir = get_hooks_dir((project_dir or Path(".")).resolve(), global_install)

    removed = False
    for name in [*HOOK_FILES, "settings.json"]:
        target = hooks_dir / name
        if target.exists():
            target.unlink()
            removed = True
    lib_dir = hooks_dir / "lib"
    if lib_dir.exists():
        shutil.rmtree(lib_dir)
        removed = True

    if removed:
        print(f"Removed Hindsight hooks from {hooks_dir}")
        # Drop the hooks dir if we left it empty.
        try:
            hooks_dir.rmdir()
        except OSError:
            pass
    else:
        print(f"No Hindsight hooks found in {hooks_dir} — nothing to remove")

    print()
    print("User config at ~/.hindsight/cline.json was preserved.")
