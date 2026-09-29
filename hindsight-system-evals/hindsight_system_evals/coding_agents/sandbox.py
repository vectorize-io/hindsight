"""A throwaway HOME with the coding-agents plugin installed in it.

Everything the plugin and Claude Code touch is keyed off ``os.homedir()`` /
``$HOME`` — the config file, the staged runtime, the logs, the hooks in
``~/.claude/settings.json``, the MCP registration in ``~/.claude.json``. So one
temp HOME isolates the whole thing, and a run can never write into the
developer's real memory, logs or agent config.

Two consequences worth knowing:

* **Credentials have to be carried in.** With a fresh HOME the CLI reports "Not
  logged in" even though the subscription token lives in the macOS Keychain, so
  the token is copied into ``<home>/.claude/.credentials.json`` (owner-only).
  That makes this macOS-only for now; a CI runner would set ANTHROPIC_API_KEY
  instead and that path is not built yet.
* **The plugin is installed from the checkout, not from npm.** The point of the
  harness is to measure a prompt change before it is released, so
  ``dist/installer.js`` of the working tree is what gets staged.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PLUGIN_DIR = REPO_ROOT / "hindsight-integrations" / "coding-agents"

#: The macOS Keychain entry Claude Code keeps its OAuth token in.
KEYCHAIN_SERVICE = "Claude Code-credentials"

#: Copied from the developer's ~/.claude.json so the CLI starts as an onboarded
#: install. Everything heavy or leaky is dropped: `projects`/`history` carry the
#: developer's own transcripts, and `mcpServers` would bring their real Hindsight
#: registration — pointing this run at their production bank.
_CLAUDE_JSON_DROP = ("projects", "history", "tipsHistory", "cachedChangelog", "mcpServers")


@dataclass
class Sandbox:
    root: Path
    home: Path
    workdir: Path
    bank_id: str
    config_path: Path

    @property
    def mcp_config(self) -> Path:
        """A Hindsight-only MCP config, passed with ``--strict-mcp-config``.

        Without it the run inherits the developer's account connectors (Gmail,
        Drive, Claude Docs) — dozens of tools competing for the model's attention
        in a measurement about whether it reaches for ONE tool, and a number that
        moves whenever their connector set changes.
        """
        return self.root / "mcp.json"

    @property
    def usage_file(self) -> Path:
        return self.home / ".hindsight" / "coding-agents-logs" / "usage.jsonl"

    @property
    def plugin_log(self) -> Path:
        return self.home / ".hindsight" / "coding-agents-logs" / "plugin.log"

    def env(self) -> dict[str, str]:
        """The environment every `claude` run gets.

        Built from nothing rather than inherited: this process is itself often a
        Claude Code session, and its CLAUDE_CODE_* variables make the child think
        it is a nested invocation.
        """
        return {
            "PATH": os.environ["PATH"],
            "HOME": str(self.home),
            "TERM": "dumb",
            "LANG": os.environ.get("LANG", "en_US.UTF-8"),
        }


def refresh_credentials(sandbox: Sandbox) -> None:
    """Re-copy the host's token into the sandbox.

    Called before every session, not once at build: the host rotates this token,
    and a run holding the copy it took at startup dies part-way through with
    `401 OAuth access token has been revoked` — which cost a replication run its
    last two sessions. Re-reading it costs one Keychain call per session.
    """
    _copy_credentials(sandbox.home)


def _copy_credentials(home: Path) -> None:
    if not sys.platform.startswith("darwin"):
        raise RuntimeError(
            "The sandbox carries the subscription token out of the macOS Keychain, so it only runs "
            "on macOS today. On another OS, teach build_sandbox to pass ANTHROPIC_API_KEY through."
        )
    token = subprocess.run(
        ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"],
        capture_output=True,
        text=True,
    )
    if token.returncode != 0 or not token.stdout.strip():
        raise RuntimeError(
            f"Could not read the Claude Code credentials from the Keychain ({KEYCHAIN_SERVICE}). "
            "Log in with `claude` once, and approve the Keychain prompt if one appears."
        )
    path = home / ".claude" / ".credentials.json"
    path.write_text(token.stdout.strip(), encoding="utf-8")
    path.chmod(0o600)


def _copy_claude_json(home: Path) -> None:
    source = Path.home() / ".claude.json"
    if not source.exists():
        return
    data = json.loads(source.read_text(encoding="utf-8"))
    kept = {k: v for k, v in data.items() if k not in _CLAUDE_JSON_DROP}
    kept["projects"] = {}
    (home / ".claude.json").write_text(json.dumps(kept), encoding="utf-8")


def build_plugin(plugin_dir: Path = PLUGIN_DIR) -> None:
    """`npm run build` in the checkout — what makes a prompt edit measurable."""
    subprocess.run(["npm", "run", "build"], cwd=plugin_dir, check=True, capture_output=True, text=True)


def build_sandbox(
    root: Path,
    *,
    api_url: str,
    bank_id: str,
    workdir: Path,
    plugin_dir: Path = PLUGIN_DIR,
    config_overrides: dict | None = None,
) -> Sandbox:
    """Create the temp HOME and install the plugin's claude-code wiring into it."""
    home = root / "home"
    (home / ".claude").mkdir(parents=True, exist_ok=True)
    _copy_credentials(home)
    _copy_claude_json(home)

    installer = plugin_dir / "dist" / "installer.js"
    if not installer.exists():
        raise RuntimeError(f"{installer} is missing — run `npm run build` in {plugin_dir} first (or pass --build).")

    env = {"PATH": os.environ["PATH"], "HOME": str(home), "TERM": "dumb"}
    # Non-interactive because stdin is a pipe: the installer never prompts without a TTY.
    result = subprocess.run(
        [
            "node",
            str(installer),
            "install",
            "claude-code",
            "--server",
            "self-hosted",
            "--api-url",
            api_url,
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        raise RuntimeError(f"plugin install failed:\n{result.stdout}\n{result.stderr}")

    config_path = home / ".hindsight" / "coding-agent.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        {
            # Pinned rather than derived from the fixture repo's name: every run
            # gets its own bank, so one run's memory cannot raise the next one's
            # search rate and make a prompt change look like it worked.
            "bankId": bank_id,
            "dynamicBankId": False,
            # A published release replacing the runtime mid-run would measure a
            # different plugin than the one under test.
            "autoUpdate": False,
            "logLevel": "debug",
        }
    )
    config.update(config_overrides or {})
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    workdir.mkdir(parents=True, exist_ok=True)
    sandbox = Sandbox(root=root, home=home, workdir=workdir, bank_id=bank_id, config_path=config_path)
    _write_mcp_config(sandbox)
    return sandbox


def _write_mcp_config(sandbox: Sandbox) -> None:
    """Copy the registration the installer just wrote into a standalone config.

    Copied rather than composed by hand so the measured run launches the MCP
    server exactly as an installed machine does — same command, same env — while
    `--strict-mcp-config` keeps everything else out.
    """
    registrations = json.loads((sandbox.home / ".claude.json").read_text(encoding="utf-8")).get("mcpServers", {})
    if "hindsight" not in registrations:
        raise RuntimeError(
            "The installer did not register the hindsight MCP server in the sandbox's ~/.claude.json. "
            "Without it the agent has no search tool and the run would measure nothing."
        )
    sandbox.mcp_config.write_text(
        json.dumps({"mcpServers": {"hindsight": registrations["hindsight"]}}), encoding="utf-8"
    )


def destroy_sandbox(sandbox: Sandbox) -> None:
    shutil.rmtree(sandbox.root, ignore_errors=True)
