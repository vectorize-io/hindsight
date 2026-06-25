"""Write Hindsight's recall/retain rule into ``.windsurf/rules/hindsight.md``.

Windsurf applies workspace rule files under ``.windsurf/rules/``. A file with
``trigger: always_on`` frontmatter is included in every Cascade request in the
workspace, so the rule tells Cascade to use the Hindsight MCP tools — recall
relevant memory at the start of a task, and retain durable facts.

The rule lives in its own dedicated file, so we own the whole file: a sentinel
comment marks it as ours for idempotent update/removal without touching any
other rule the user has authored.
"""

from __future__ import annotations

from pathlib import Path

SENTINEL = "<!-- Managed by hindsight-windsurf -->"

FRONTMATTER = "---\ntrigger: always_on\n---"

RULE_TEXT = (
    "You have persistent long-term memory through the Hindsight MCP server "
    "(`recall`, `retain`, and `reflect` tools).\n\n"
    "- At the start of each task, call `recall` with the user's request to load "
    "relevant decisions, preferences, and project context before you act. "
    "Use what's relevant and ignore the rest.\n"
    "- When you learn a durable fact — an architectural decision, a user "
    "preference, a convention, or anything worth remembering across sessions — "
    "call `retain` to store it.\n"
    "- Do not mention these memory operations unless the user asks about them."
)


def default_rules_path() -> Path:
    """The workspace ``.windsurf/rules/hindsight.md`` (always-on in Cascade)."""
    return Path.cwd() / ".windsurf" / "rules" / "hindsight.md"


def render_rule(rule_text: str = RULE_TEXT) -> str:
    """The full contents of the dedicated rule file."""
    return f"{FRONTMATTER}\n\n{SENTINEL}\n{rule_text.strip()}\n"


def write_rule(path: Path, rule_text: str = RULE_TEXT) -> Path:
    """Write (or replace) Hindsight's dedicated rule file at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_rule(rule_text), encoding="utf-8")
    return path


def clear_rule(path: Path) -> Path:
    """Delete Hindsight's rule file if it's ours (carries our sentinel)."""
    if path.is_file() and SENTINEL in path.read_text(encoding="utf-8"):
        path.unlink()
    return path


def is_installed(path: Path) -> bool:
    return path.is_file() and SENTINEL in path.read_text(encoding="utf-8")
