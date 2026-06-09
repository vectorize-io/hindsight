"""Install logic for the Cursor CLI Hindsight integration.

Reproduces the installed layout the hook scripts expect at runtime:

    ~/.cursor/hooks/cursor-cli/
        scripts/            — the hook scripts + their ``lib/`` package
        settings.json       — default config (version stamped at install time)
        hooks.json          — rendered with absolute script paths
    ~/.cursor/hooks.json    — Cursor's hook registry (Hindsight block merged in)
    ~/.hindsight/cursor-cli.json — user config (seeded empty, never overwritten)

The hook payload (``scripts/``, ``settings.json``, ``hooks.json``) ships as
package data under ``hindsight_cursor_cli/hooks`` and is read via
``importlib.resources`` so it resolves whether installed as a wheel or run from
a source checkout.
"""

import json
import shutil
import sys
from importlib import metadata, resources
from importlib.resources.abc import Traversable
from pathlib import Path

PACKAGE = "hindsight_cursor_cli"
HOOKS_DIRNAME = "cursor-cli"
SCRIPTS_PLACEHOLDER = "__SCRIPTS_DIR__"
# Marker used to identify Hindsight's own hook entries when merging/stripping
# the shared ~/.cursor/hooks.json. Every command path contains this segment.
HOOK_MARKER = "hooks/cursor-cli"


def _payload_root() -> Traversable:
    """The packaged hook payload (``scripts/``, ``settings.json``, ``hooks.json``)."""
    return resources.files(PACKAGE).joinpath("hooks")


def _package_version() -> str:
    """Installed package version, stamped into the deployed settings.json."""
    try:
        return metadata.version(PACKAGE)
    except metadata.PackageNotFoundError:
        return "0.0.0"


def get_cursor_dir() -> Path:
    return Path.home() / ".cursor"


def get_install_dir() -> Path:
    """Where the hook payload is deployed (``~/.cursor/hooks/cursor-cli``)."""
    return get_cursor_dir() / "hooks" / HOOKS_DIRNAME


def _copy_scripts(install_dir: Path) -> Path:
    """Copy the packaged ``scripts/`` tree into the install dir. Returns its path."""
    scripts_dst = install_dir / "scripts"
    if scripts_dst.exists():
        shutil.rmtree(scripts_dst)
    scripts_dst.mkdir(parents=True, exist_ok=True)
    # importlib.resources.as_file materialises the packaged dir on disk (a no-op
    # copy for a real filesystem install, a real extraction for a zipped wheel).
    with resources.as_file(_payload_root().joinpath("scripts")) as scripts_src:
        for item in Path(scripts_src).iterdir():
            dest = scripts_dst / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
    return scripts_dst


def write_settings(install_dir: Path) -> None:
    """Write the default settings.json, stamping the installed package version."""
    settings = json.loads(_payload_root().joinpath("settings.json").read_text())
    settings = {"version": _package_version(), **settings}
    (install_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")


def render_hooks_block(scripts_dir: Path) -> dict:
    """Load the packaged hooks.json template, substituting the scripts path."""
    template = _payload_root().joinpath("hooks.json").read_text()
    rendered = template.replace(SCRIPTS_PLACEHOLDER, str(scripts_dir))
    return json.loads(rendered)


def _is_hindsight_entry(definition: dict) -> bool:
    return HOOK_MARKER in json.dumps(definition)


def merge_hooks(cursor_dir: Path, hooks_block: dict) -> Path:
    """Merge Hindsight's hooks into ``~/.cursor/hooks.json``, preserving others.

    Idempotent: any pre-existing Hindsight entries are replaced, not duplicated.
    Returns the path to the registry file.
    """
    hooks_json = cursor_dir / "hooks.json"

    existing: dict = {}
    if hooks_json.exists():
        try:
            existing = json.loads(hooks_json.read_text())
        except (OSError, ValueError):
            existing = {}
    existing.setdefault("version", hooks_block.get("version", 1))
    existing.setdefault("hooks", {})

    for event, definitions in hooks_block.get("hooks", {}).items():
        bucket = [d for d in existing["hooks"].get(event, []) if not _is_hindsight_entry(d)]
        bucket.extend(definitions)
        existing["hooks"][event] = bucket

    cursor_dir.mkdir(parents=True, exist_ok=True)
    hooks_json.write_text(json.dumps(existing, indent=2) + "\n")
    return hooks_json


def seed_user_config(api_url: str | None, api_token: str | None) -> Path:
    """Seed ``~/.hindsight/cursor-cli.json`` if absent. Never overwrites."""
    user_config = Path.home() / ".hindsight" / "cursor-cli.json"
    if user_config.exists():
        print(f"User config already exists at {user_config} — leaving it alone")
        return user_config
    user_config.parent.mkdir(parents=True, exist_ok=True)
    user_config.write_text(
        json.dumps(
            {"hindsightApiUrl": api_url or "", "hindsightApiToken": api_token},
            indent=2,
        )
        + "\n"
    )
    print(f"Seeded user config: {user_config}")
    return user_config


def run_install(api_url: str | None = None, api_token: str | None = None) -> None:
    """Install the hook scripts and register them with Cursor CLI."""
    cursor_dir = get_cursor_dir()
    install_dir = get_install_dir()

    print("Installing Hindsight memory for Cursor CLI...")
    print(f"  Install dir : {install_dir}")
    if api_url:
        print(f"  API URL     : {api_url}")
    print()

    install_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir = _copy_scripts(install_dir)
    write_settings(install_dir)

    hooks_block = render_hooks_block(scripts_dir)
    # Keep a rendered copy beside the scripts for reference / debugging.
    (install_dir / "hooks.json").write_text(json.dumps(hooks_block, indent=2) + "\n")
    registry = merge_hooks(cursor_dir, hooks_block)
    print(f"Hooks registered: {registry}")

    seed_user_config(api_url, api_token)

    print()
    print("Done. Restart Cursor CLI to load the new hooks.")
    print("Logs (with debug=true): tail -F ~/.hindsight/cursor-cli/state/*.log")


def run_uninstall() -> None:
    """Remove the hook scripts and strip Hindsight's entries from the registry."""
    install_dir = get_install_dir()
    hooks_json = get_cursor_dir() / "hooks.json"

    if install_dir.exists():
        shutil.rmtree(install_dir)
        print(f"Removed {install_dir}")
    else:
        print(f"{install_dir} does not exist — nothing to remove")

    if hooks_json.exists():
        try:
            data = json.loads(hooks_json.read_text())
        except (OSError, ValueError):
            data = None
        if data is not None:
            hooks = data.get("hooks", {})
            for event, definitions in list(hooks.items()):
                kept = [d for d in definitions if not _is_hindsight_entry(d)]
                if kept:
                    hooks[event] = kept
                else:
                    del hooks[event]
            data["hooks"] = hooks
            hooks_json.write_text(json.dumps(data, indent=2) + "\n")
            print(f"Stripped Hindsight entries from {hooks_json}")
    else:
        print(f"{hooks_json} does not exist — nothing to strip")

    print()
    print("Done. Restart Cursor CLI to unload the hooks.")
    print("User config at ~/.hindsight/cursor-cli.json was preserved.")


if __name__ == "__main__":
    sys.exit(run_install())
