"""A real Hermes Agent with the Hindsight plugin from this checkout, in a throwaway home.

Hermes keeps everything under ``HERMES_HOME``: its config, sessions, built-in
memory, plugins, and the background gateway that actually runs the agent. So one
home isolates a run from the developer's own Hermes, banks and memory.

**Every setting is the default**, Hermes' and the plugin's, because the evals grade
what a user gets. Two exceptions, both about the machine rather than the behaviour:

* ``terminal.backend: docker``. Hermes' default tools include a terminal and file
  tools on the host. Asked a coding question in an empty directory, it went looking
  through the developer's real ``~/dev`` for ten minutes (read-only, but the file
  contents went to the model, and ``--yolo`` would not have stopped a write). In
  Docker the same tools run in Hermes' own sandbox image and see none of it.
* The connection: ``mode: local_external`` to the eval's server, and a bank per test.

How the home behaves:

* **It is cached, not created per run.** A new ``HERMES_HOME`` makes Hermes build
  its own environment first (~7 minutes, ~420 MB, measured on Hermes 2026.9.24).
  The environment lives in ``installs/``; everything else is wiped on every
  ``build_sandbox``, so no session, memory or config carries over. Runs therefore
  share one home and must not overlap.
* **The agent runs in a gateway, not in the ``hermes`` process.** ``hermes chat -q``
  starts (or reuses) a gateway that exits on idle; the plugin, its retain writer
  and its recall thread live there. ``build_sandbox`` and the tests stop it, so the
  next sandbox starts with the freshly written plugin and config.
* **Hermes allows one gateway per OS user and attaches every home to it**, through
  a lock in ``$HERMES_GATEWAY_LOCK_DIR``. Left at its default, a run would attach to
  the developer's own gateway — their config, their memory. The sandbox gets its
  own lock dir, so it starts a gateway of its own and nothing else attaches to it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import psutil
from pydantic import BaseModel, ConfigDict, ValidationError

from hindsight_system_evals.coding_agents.sandbox import REPO_ROOT

PLUGIN_DIR = REPO_ROOT / "hindsight-integrations" / "hermes"
#: Where the developer's Hermes keeps its provider credentials (OAuth pools, API keys).
USER_HERMES_HOME = Path.home() / ".hermes"

HOME_DIR = Path(
    os.getenv("HINDSIGHT_EVAL_HERMES_HOME", str(Path.home() / ".cache" / "hindsight-system-evals" / "hermes-home"))
)
#: The model Hermes itself answers with — not the Hindsight server's model.
PROVIDER = os.getenv("HINDSIGHT_EVAL_HERMES_PROVIDER", "gemini")
MODEL = os.getenv("HINDSIGHT_EVAL_HERMES_MODEL", "gemini-3-flash-preview")
#: Provider keys Hermes reads from its environment, passed through when set. OAuth logins
#: (Codex, Copilot, Nous) need none: they come with the copied ``auth.json``.
_PROVIDER_KEYS = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
    "DEEPSEEK_API_KEY",
)
#: Plugin settings for this run, as a JSON object merged over the defaults — how an arm is
#: measured, e.g. ``{"recall_sync": true}``. Unset, the plugin runs on its defaults.
PLUGIN_OVERRIDES: dict = json.loads(os.getenv("HINDSIGHT_EVAL_HERMES_PLUGIN_CONFIG") or "{}")
TURN_TIMEOUT_SECONDS = float(os.getenv("HINDSIGHT_EVAL_HERMES_TURN_TIMEOUT", "900"))

#: Kept between runs: the built environment. Everything else in the home is wiped.
_KEEP = {"installs"}


@dataclass
class ToolCall:
    name: str
    input: str  # the arguments as compact JSON, for the transcript


@dataclass
class Turn:
    prompt: str
    reply: str
    calls: list[ToolCall] = field(default_factory=list)
    error: str = ""
    session_id: str = ""

    @property
    def tools(self) -> list[str]:
        return [call.name for call in self.calls]


@dataclass
class HermesSandbox:
    home: Path
    workdir: Path

    def env(self) -> dict[str, str]:
        return _env(self.home)


class _StreamEvent(BaseModel):
    """One line of ``hermes chat --format stream-json``; only the fields read here."""

    model_config = ConfigDict(extra="ignore")

    type: str
    name: str | None = None
    text: str | None = None
    error: str | None = None
    session_id: str | None = None
    input: dict | None = None  # a tool's arguments: whatever that tool takes


def _env(home: Path) -> dict[str, str]:
    """Built from nothing: this process is often itself an agent session, and its
    HINDSIGHT_* variables are fallbacks the plugin would read."""
    return {
        "PATH": os.environ["PATH"],
        "HOME": str(Path.home()),
        "HERMES_HOME": str(home),
        "HERMES_GATEWAY_LOCK_DIR": str(home.parent / f"{home.name}-gateway-locks"),
        "TERM": "dumb",
        "LANG": os.environ.get("LANG", "en_US.UTF-8"),
    } | {key: os.environ[key] for key in _PROVIDER_KEYS if os.environ.get(key)}


def missing_prerequisite() -> str:
    """Why the Hermes evals cannot run here, or "" when they can."""
    if shutil.which("hermes") is None:
        return "hermes is not on PATH"
    docker = subprocess.run(["docker", "info"], capture_output=True) if shutil.which("docker") else None
    if docker is None or docker.returncode != 0:
        return "Docker is not running; Hermes' terminal and file tools run in it (see hindsight_system_evals/hermes.py)"
    return ""


def build_sandbox(*, api_url: str, api_key: str | None, bank_id: str) -> HermesSandbox:
    """A fresh Hermes profile on the cached home, with the plugin from the checkout.

    Called again with the same bank, it is a user opening Hermes on a new machine:
    no sessions, no built-in memory, only what the bank holds.
    """
    HOME_DIR.mkdir(parents=True, exist_ok=True)
    home = HOME_DIR.resolve()  # the gateway records the resolved path; stop_gateway compares it
    stop_gateway(home)
    for child in home.iterdir():
        if child.name not in _KEEP:
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()

    auth = USER_HERMES_HOME / "auth.json"
    if auth.exists():
        shutil.copy2(auth, home / "auth.json")
        (home / "auth.json").chmod(0o600)

    (home / "config.yaml").write_text(
        json.dumps(  # JSON is YAML, and avoids a yaml dependency for four keys
            {
                "model": {"default": MODEL, "provider": PROVIDER},
                "memory": {"provider": "hindsight"},
                "plugins": {"enabled": ["hindsight"]},
                "terminal": {"backend": "docker"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # A copy, not a symlink: the gateway imports it, and the checkout's tests and caches stay out.
    shutil.copytree(
        PLUGIN_DIR,
        home / "plugins" / "hindsight",
        ignore=shutil.ignore_patterns("tests", "__pycache__", ".venv", "uv.lock", "docs"),
    )
    plugin_config: dict = {"mode": "local_external", "api_url": api_url, "bank_id": bank_id}
    if api_key:
        plugin_config["api_key"] = api_key
    plugin_config |= PLUGIN_OVERRIDES
    (home / "hindsight").mkdir()
    (home / "hindsight" / "config.json").write_text(json.dumps(plugin_config, indent=2), encoding="utf-8")

    # An empty working directory, so no AGENTS.md or repository leaks into the conversation.
    workdir = home / "workdir"
    workdir.mkdir()
    return HermesSandbox(home=home, workdir=workdir)


def run_session(sandbox: HermesSandbox, prompts: list[str], *, log_path: Path) -> list[Turn]:
    """One Hermes session: the first prompt opens it, each later one resumes it."""
    turns: list[Turn] = []
    session_id = ""
    with log_path.open("a", encoding="utf-8") as log:
        for prompt in prompts:
            argv = ["hermes", "chat", "-q", prompt, "--format", "stream-json", "--yolo"]
            if session_id:
                argv += ["--resume", session_id]
            try:
                result = subprocess.run(
                    argv,
                    cwd=sandbox.workdir,
                    env=sandbox.env(),
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=TURN_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired as exc:
                log.write(
                    f"### {' '.join(argv[:3])} … TIMED OUT\n{_text(exc.stdout)}\n--- stderr\n{_text(exc.stderr)}\n"
                )
                turns.append(Turn(prompt=prompt, reply="", error=f"no reply in {TURN_TIMEOUT_SECONDS:.0f}s"))
                break
            log.write(f"### {' '.join(argv[:3])} …\n{result.stdout}\n--- stderr\n{result.stderr}\n")
            turn = _parse_turn(prompt, result.stdout)
            if not turn.error and result.returncode != 0:
                turn.error = f"hermes exited {result.returncode}: {result.stderr.strip()[-500:]}"
            turns.append(turn)
            if turn.error:
                break
            session_id = turn.session_id or session_id
    return turns


def _text(output: str | bytes | None) -> str:
    return output.decode(errors="replace") if isinstance(output, bytes) else output or ""


def _parse_turn(prompt: str, stdout: str) -> Turn:
    turn = Turn(prompt=prompt, reply="")
    for line in stdout.splitlines():
        try:
            event = _StreamEvent.model_validate_json(line)
        except ValidationError:
            continue  # Hermes prints a few plain diagnostic lines on stdout too
        if event.type == "tool_use":
            turn.calls.append(ToolCall(event.name or "unknown", json.dumps(event.input or {}, ensure_ascii=False)))
        elif event.type == "result":
            turn.reply = event.text or ""
            turn.error = event.error or ""
            turn.session_id = event.session_id or ""
    if not turn.reply and not turn.error:
        turn.error = "no result event in Hermes' output"
    return turn


def builtin_memory(sandbox: HermesSandbox) -> str:
    """What Hermes wrote to its own memory files (``MEMORY.md``, ``USER.md``), or ""."""
    return "\n".join(
        text
        for f in sorted((sandbox.home / "memories").glob("*.md"))
        if (text := f.read_text(encoding="utf-8").strip())
    )


def keep_logs(sandbox: HermesSandbox, into: Path) -> None:
    """Copy Hermes' own logs out before the next profile wipes them."""
    if (sandbox.home / "logs").is_dir():
        shutil.copytree(sandbox.home / "logs", into, dirs_exist_ok=True)


def stop_gateway(home: Path) -> None:
    """Stop the gateway this home started; it holds the plugin as it was loaded.

    By PID, checked against the process's own HERMES_HOME, rather than
    ``hermes gateway stop``: that command acts on the host's gateway, which can be
    the developer's.
    """
    try:
        pid = int(json.loads((home / "gateway.pid").read_text())["pid"])
        proc = psutil.Process(pid)
        if proc.environ().get("HERMES_HOME") == str(home):
            proc.terminate()
            proc.wait(timeout=30)
    except (OSError, ValueError, KeyError, TypeError, psutil.Error):
        pass
    _remove_containers(home)


def _remove_containers(home: Path) -> None:
    """Remove the sandbox containers this home's gateway left running.

    A stopped gateway does not reap its Docker sandbox: one was found still up after a run.
    Matched by a mount under this home, so a container of the developer's own Hermes is never touched.
    """
    if shutil.which("docker") is None:
        return
    ids = subprocess.run(
        ["docker", "ps", "-q", "--filter", "label=hermes-agent=1"], capture_output=True, text=True
    ).stdout.split()
    for container in ids:
        mounts = subprocess.run(
            ["docker", "inspect", container, "--format", "{{range .Mounts}}{{.Source}}\n{{end}}"],
            capture_output=True,
            text=True,
        ).stdout.split()
        if any(Path(source).is_relative_to(home) for source in mounts):
            subprocess.run(["docker", "rm", "-f", container], capture_output=True)
