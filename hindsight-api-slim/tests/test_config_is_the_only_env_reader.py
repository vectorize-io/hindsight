"""Every ``HINDSIGHT_API_*`` value is parsed in config.py and nowhere else.

Two parsers for one variable is how the engine and the config drift apart: the
factory that built the default LLM provider had grown its own copies of the
provider defaulting, the Gemini tier gating and the cache-affinity default, each
carrying a comment asking the next reader not to let them disagree. This test
removes the need for those comments.

A module that reads ``os.environ`` for a ``HINDSIGHT_API_*`` name either reads it
off :class:`HindsightConfig` instead, or appears in ``EXEMPT`` below with the
reason it cannot.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

import hindsight_api

PACKAGE_ROOT = Path(hindsight_api.__file__).parent
CONFIG_PATH = PACKAGE_ROOT / "config.py"

#: Modules allowed to touch the environment directly, and why.
EXEMPT: dict[str, str] = {
    # Alembic loads migrations in its own process, against a database that may predate
    # this build. Reaching into the application config from a migration inverts that.
    "alembic/env.py": "runs under standalone Alembic, before/without the app config",
    "alembic/_owned.py": "same; and is read at call time so a late setting still applies",
    # Seeds the environment the config then parses — a write, not a read.
    "mcp_local.py": "sets a default before the config is first built",
    # The guard runs to decide whether this interpreter may run at all.
    "_thread_limits.py": "sets thread-library caps before any config exists",
    # Extensions declare open-ended `HINDSIGHT_API_<NAME>_*` settings that are handed to
    # the extension as a dict; they cannot be enumerated as static config fields.
    "extensions/loader.py": "dynamic per-extension config namespace",
    "engine/storage/__init__.py": "dynamic HINDSIGHT_API_FILE_STORAGE_* config namespace",
}


@dataclass(frozen=True)
class EnvRead:
    """One resolvable ``HINDSIGHT_API_*`` read, and where it is."""

    lineno: int
    name: str


def _env_reads(tree: ast.AST, module_env_names: dict[str, str], config_env_names: dict[str, str]) -> list[EnvRead]:
    """Every ``os.environ``/``os.getenv`` read of a resolvable HINDSIGHT_API_* name."""
    found: list[EnvRead] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in ("getenv", "get", "pop"):
            continue
        target = node.func.value
        is_os_env = (isinstance(target, ast.Name) and target.id == "os") or (
            isinstance(target, ast.Attribute) and target.attr == "environ"
        )
        if not is_os_env or not node.args:
            continue
        arg = node.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            name = arg.value
        elif isinstance(arg, ast.Name):
            name = module_env_names.get(arg.id) or config_env_names.get(arg.id, "")
        else:
            continue
        if name.startswith("HINDSIGHT_API_"):
            found.append(EnvRead(lineno=node.lineno, name=name))
    return found


def _env_constants(tree: ast.AST) -> dict[str, str]:
    names: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id.startswith("ENV_"):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                continue
            if isinstance(value, str):
                names[target.id] = value
    return names


@pytest.fixture(scope="module")
def config_env_names() -> dict[str, str]:
    return _env_constants(ast.parse(CONFIG_PATH.read_text()))


def _modules() -> list[Path]:
    return sorted(p for p in PACKAGE_ROOT.rglob("*.py") if p != CONFIG_PATH)


def test_no_module_outside_config_reads_a_hindsight_env_var(config_env_names):
    offenders: list[str] = []
    for path in _modules():
        rel = path.relative_to(PACKAGE_ROOT).as_posix()
        if rel in EXEMPT or rel.startswith("alembic/versions/"):
            continue
        tree = ast.parse(path.read_text())
        for read in _env_reads(tree, _env_constants(tree), config_env_names):
            offenders.append(f"{rel}:{read.lineno} reads {read.name}")

    assert not offenders, (
        "These modules parse a HINDSIGHT_API_* variable themselves instead of reading the "
        "resolved HindsightConfig:\n  " + "\n  ".join(offenders) + "\n\n"
        "Add a field to HindsightConfig and read it, or — if the value is genuinely needed "
        "before a config can exist — add the module to EXEMPT with the reason."
    )


def test_every_exemption_still_applies():
    """An exemption that no longer touches the environment is stale; drop it.

    Deliberately looser than the check above: these modules are exempt precisely
    because they use forms that check cannot resolve — ``setdefault`` with a
    computed name, a scan over ``os.environ.items()`` for a dynamic prefix. Any
    remaining mention of ``os.environ``/``os.getenv`` keeps the exemption honest.
    """
    stale = []
    for rel in EXEMPT:
        path = PACKAGE_ROOT / rel
        if not path.exists():
            stale.append(f"{rel} (file is gone)")
            continue
        source = path.read_text()
        if "os.environ" not in source and "os.getenv" not in source:
            stale.append(f"{rel} (no longer touches the environment)")
    assert not stale, "Stale entries in EXEMPT:\n  " + "\n  ".join(stale)
