"""Install logic for the GitHub Copilot CLI Hindsight integration.

Reproduces the installed layout the hook scripts expect at runtime:

    ~/.copilot/hindsight-copilot-cli/
        scripts/            — the hook scripts + their ``lib/`` package
        settings.json       — default config (version stamped at install time)
    ~/.copilot/hooks/hindsight-copilot-cli.json   — Copilot CLI hook registration (user scope)
    .github/hooks/hindsight-copilot-cli.json      — Copilot CLI hook registration (repo scope)
    ~/.hindsight/copilot-cli.json     — user config (seeded empty, never overwritten)

Unlike Cursor CLI (a single shared ``~/.cursor/hooks.json`` requiring
merge/strip logic to coexist with other tools), Copilot CLI loads **every**
``*.json`` file from its hooks directory and combines them — so we can drop
a **standalone** ``hindsight-copilot-cli.json`` file with no merge/strip logic at all.
``uninstall`` simply deletes our own files.

The hook payload (``scripts/``, ``settings.json``, ``hooks.json`` template)
ships as package data under ``hindsight_copilot_cli/hooks`` and is read via
``importlib.resources`` so it resolves whether installed as a wheel or run
from a source checkout.
"""

import json
import shutil
import sys
from importlib import metadata, resources
from importlib.resources.abc import Traversable
from pathlib import Path

PACKAGE = "hindsight_copilot_cli"
INSTALL_DIRNAME = "hindsight-copilot-cli"
HOOKS_FILENAME = "hindsight-copilot-cli.json"
SCRIPTS_PLACEHOLDER = "__SCRIPTS_DIR__"


def _payload_root() -> Traversable:
    """The packaged hook payload (``scripts/``, ``settings.json``, ``hooks.json``)."""
    return resources.files(PACKAGE).joinpath("hooks")


def _package_version() -> str:
    """Installed package version, stamped into the deployed settings.json."""
    try:
        return metadata.version(PACKAGE)
    except metadata.PackageNotFoundError:
        return "0.0.0"


def get_copilot_home() -> Path:
    """``$COPILOT_HOME`` if set, else ``~/.copilot`` (matches Copilot CLI's own lookup)."""
    import os

    home = os.environ.get("COPILOT_HOME")
    return Path(home) if home else Path.home() / ".copilot"


def get_install_dir() -> Path:
    """Where the hook payload is deployed (``~/.copilot/hindsight-copilot-cli``)."""
    return get_copilot_home() / INSTALL_DIRNAME


def get_user_hooks_registry() -> Path:
    """User-level hook registration file (``~/.copilot/hooks/hindsight-copilot-cli.json``)."""
    return get_copilot_home() / "hooks" / HOOKS_FILENAME


def get_repo_hooks_registry(repo_root: Path | None = None) -> Path:
    """Repo-level hook registration file (``.github/hooks/hindsight-copilot-cli.json``)."""
    root = repo_root or Path.cwd()
    return root / ".github" / "hooks" / HOOKS_FILENAME


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


def _write_registry(registry_path: Path, hooks_block: dict) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(hooks_block, indent=2) + "\n")


def seed_user_config(api_url: str | None, api_token: str | None) -> Path:
    """Seed ``~/.hindsight/copilot-cli.json`` if absent. Never overwrites."""
    user_config = Path.home() / ".hindsight" / "copilot-cli.json"
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


def run_install(
    api_url: str | None = None,
    api_token: str | None = None,
    scope: str = "user",
) -> None:
    """Install the hook scripts and register them with Copilot CLI.

    `scope="user"` (default) registers under ``~/.copilot/hooks/`` — active
    for every Copilot CLI session on this machine. `scope="repo"` registers
    under ``.github/hooks/`` in the current directory instead, for teams
    that want to check in shared hooks; the hook *scripts* are still
    installed once per-machine under `~/.copilot`, only the registration
    pointer is committed to the repo. In repo scope, never bake a token
    into the committed file — rely on the `HINDSIGHT_API_TOKEN` env var at
    runtime instead.
    """
    if scope not in ("user", "repo"):
        raise ValueError(f"Invalid scope: {scope!r} (expected 'user' or 'repo')")

    install_dir = get_install_dir()

    print(f"Installing Hindsight memory for GitHub Copilot CLI ({scope} scope)...")
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

    if scope == "repo":
        registry = get_repo_hooks_registry()
        _write_registry(registry, hooks_block)
        print(f"Hooks registered: {registry}")
        print(
            "Repo-scope hooks reference an absolute, per-machine script path — "
            "each teammate must run this installer locally too. Never commit "
            "hindsightApiToken; set HINDSIGHT_API_TOKEN in each environment instead."
        )
    else:
        registry = get_user_hooks_registry()
        _write_registry(registry, hooks_block)
        print(f"Hooks registered: {registry}")
        seed_user_config(api_url, api_token)

    print()
    print("Done. Restart Copilot CLI to load the new hooks.")
    print("Logs (with debug=true): tail -F ~/.hindsight/copilot-cli/state/*.log")


def run_uninstall(scope: str = "user") -> None:
    """Remove the hook scripts and registration file for the given scope."""
    if scope not in ("user", "repo"):
        raise ValueError(f"Invalid scope: {scope!r} (expected 'user' or 'repo')")

    install_dir = get_install_dir()
    registry = get_user_hooks_registry() if scope == "user" else get_repo_hooks_registry()

    if registry.exists():
        registry.unlink()
        print(f"Removed {registry}")
    else:
        print(f"{registry} does not exist — nothing to remove")

    # Only remove the shared script install when uninstalling the last
    # remaining scope's registration would leave nothing referencing it.
    other_registry = get_repo_hooks_registry() if scope == "user" else get_user_hooks_registry()
    if install_dir.exists() and not other_registry.exists():
        shutil.rmtree(install_dir)
        print(f"Removed {install_dir}")

    print()
    print("Done. Restart Copilot CLI to unload the hooks.")
    print("User config at ~/.hindsight/copilot-cli.json was preserved.")


if __name__ == "__main__":
    sys.exit(run_install())
