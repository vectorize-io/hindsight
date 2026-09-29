"""Install logic for the ZCode Hindsight integration.

ZCode (Z.ai's GLM desktop coding agent, zcode.z.ai) reads configuration hooks
from its CLI config file at ``~/.zcode/cli/config.json`` — never the user's real
Claude Code config at ``~/.claude/settings.json``.

Reproduces the installed layout the hook scripts expect at runtime:

    ~/.zcode/hooks/hindsight/
        scripts/               — the hook scripts + their ``lib/`` package
        settings.json          — default config (version stamped at install time)
        hooks.json             — rendered with absolute script paths (reference copy)
    ~/.zcode/cli/config.json   — ZCode CLI config; Hindsight's "hooks" block merged
                                 in, preserving any other keys/hooks already present
    ~/.hindsight/zcode.json    — user config (seeded empty, never overwritten)

ZCode's native hook schema lives under the top-level ``"hooks"`` key:

    {"hooks": {"enabled": true, "maxOutputBytes": 32768,
               "events": {"<Event>": [{"hooks": [{"type": "process",
                                                  "command": "python3",
                                                  "args": ["/abs/script.py"],
                                                  "timeoutMs": 12000}]}]}}}

Config hooks are disabled by default, so ``hooks.enabled`` is set to true. Only
SessionStart, UserPromptSubmit, and Stop are wired (ZCode has no SessionEnd) —
retain rides the Stop event.

The hook payload (``scripts/``, ``settings.json``, ``hooks.json``) ships as
package data under ``hindsight_zcode/hooks`` and is read via
``importlib.resources`` so it resolves whether installed as a wheel or run from
a source checkout.
"""

import json
import shutil
import sys
from importlib import metadata, resources
from importlib.resources.abc import Traversable
from pathlib import Path

PACKAGE = "hindsight_zcode"
HOOKS_DIRNAME = "hindsight"
SCRIPTS_PLACEHOLDER = "__SCRIPTS_DIR__"
# ZCode caps hook stdout at maxOutputBytes; anything larger is dropped.
MAX_OUTPUT_BYTES = 32768
# Marker used to identify Hindsight's own hook entries when merging/stripping
# the shared ~/.zcode/cli/config.json. Every script path contains this segment
# (the deploy dir is ~/.zcode/hooks/hindsight), so it never matches a foreign
# hook.
HOOK_MARKER = "hooks/hindsight"


def _payload_root() -> Traversable:
    """The packaged hook payload (``scripts/``, ``settings.json``, ``hooks.json``)."""
    return resources.files(PACKAGE).joinpath("hooks")


def _package_version() -> str:
    """Installed package version, stamped into the deployed settings.json."""
    try:
        return metadata.version(PACKAGE)
    except metadata.PackageNotFoundError:
        return "0.0.0"


def get_zcode_dir() -> Path:
    return Path.home() / ".zcode"


def get_config_path() -> Path:
    """The ZCode CLI config that carries the hooks block (``~/.zcode/cli/config.json``)."""
    return get_zcode_dir() / "cli" / "config.json"


def get_install_dir() -> Path:
    """Where the hook payload is deployed (``~/.zcode/hooks/hindsight``)."""
    return get_zcode_dir() / "hooks" / HOOKS_DIRNAME


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


def render_hooks_events(scripts_dir: Path) -> dict:
    """Load the packaged hooks.json template, substituting the scripts path.

    Returns ZCode's native ``events`` map:
        {"<EventName>": [ {"hooks": [ {"type": "process",
                                       "command": "python3",
                                       "args": ["/abs/script.py"],
                                       "timeoutMs": 12000} ]} ]}
    """
    template = _payload_root().joinpath("hooks.json").read_text()
    rendered = template.replace(SCRIPTS_PLACEHOLDER, str(scripts_dir))
    return json.loads(rendered)


def _is_hindsight_entry(definition: dict) -> bool:
    return HOOK_MARKER in json.dumps(definition)


def merge_hooks(config_path: Path, events: dict) -> Path:
    """Merge Hindsight's hooks into ``~/.zcode/cli/config.json``, preserving others.

    ZCode reads its native hook schema under the top-level ``"hooks"`` key:
    ``{"enabled": true, "maxOutputBytes": N, "events": {...}}``. Config hooks are
    disabled by default, so ``enabled`` is forced true. Any other keys in
    config.json — and any non-Hindsight event entries — are preserved untouched.

    Idempotent: any pre-existing Hindsight entries are replaced, not duplicated.
    Returns the path to the config file.
    """
    existing: dict = {}
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text())
        except (OSError, ValueError):
            existing = {}

    hooks = existing.get("hooks")
    if not isinstance(hooks, dict):
        hooks = {}
    hooks["enabled"] = True
    hooks.setdefault("maxOutputBytes", MAX_OUTPUT_BYTES)
    existing_events = hooks.get("events")
    if not isinstance(existing_events, dict):
        existing_events = {}

    for event, definitions in events.items():
        bucket = [d for d in existing_events.get(event, []) if not _is_hindsight_entry(d)]
        bucket.extend(definitions)
        existing_events[event] = bucket

    hooks["events"] = existing_events
    existing["hooks"] = hooks

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(existing, indent=2) + "\n")
    return config_path


def seed_user_config(api_url: str | None, api_token: str | None) -> Path:
    """Seed ``~/.hindsight/zcode.json`` if absent. Never overwrites."""
    user_config = Path.home() / ".hindsight" / "zcode.json"
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
    """Install the hook scripts and register them with ZCode."""
    install_dir = get_install_dir()

    print("Installing Hindsight memory for ZCode...")
    print(f"  Install dir : {install_dir}")
    if api_url:
        print(f"  API URL     : {api_url}")
    print()

    install_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir = _copy_scripts(install_dir)
    write_settings(install_dir)

    events = render_hooks_events(scripts_dir)
    # Keep a rendered copy beside the scripts for reference / debugging.
    (install_dir / "hooks.json").write_text(json.dumps(events, indent=2) + "\n")
    config_path = merge_hooks(get_config_path(), events)
    print(f"Hooks registered: {config_path}")

    seed_user_config(api_url, api_token)

    print()
    print("Done. Restart ZCode to load the new hooks.")
    print("Logs (with debug=true): tail -F ~/.hindsight/zcode/state/*.log")


def run_uninstall() -> None:
    """Remove the hook scripts and strip Hindsight's entries from ZCode config."""
    install_dir = get_install_dir()
    config_json = get_config_path()

    if install_dir.exists():
        shutil.rmtree(install_dir)
        print(f"Removed {install_dir}")
    else:
        print(f"{install_dir} does not exist — nothing to remove")

    if config_json.exists():
        try:
            data = json.loads(config_json.read_text())
        except (OSError, ValueError):
            data = None
        if isinstance(data, dict) and isinstance(data.get("hooks"), dict):
            hooks = data["hooks"]
            events = hooks.get("events")
            if isinstance(events, dict):
                for event, definitions in list(events.items()):
                    kept = [d for d in definitions if not _is_hindsight_entry(d)]
                    if kept:
                        events[event] = kept
                    else:
                        del events[event]
                hooks["events"] = events
            config_json.write_text(json.dumps(data, indent=2) + "\n")
            print(f"Stripped Hindsight entries from {config_json}")
    else:
        print(f"{config_json} does not exist — nothing to strip")

    print()
    print("Done. Restart ZCode to unload the hooks.")
    print("User config at ~/.hindsight/zcode.json was preserved.")


if __name__ == "__main__":
    sys.exit(run_install())
