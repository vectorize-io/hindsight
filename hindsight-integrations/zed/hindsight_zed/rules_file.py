"""Manage Hindsight's recall/retain rule inside Zed's global instructions file.

Zed includes a global instructions file (``~/.config/zed/AGENTS.md`` on macOS and
Linux) in **every** agent conversation. We write a static rule there telling the
agent to use the Hindsight MCP tools — recall relevant memory at the start of a
task, and retain durable facts as it learns them.

The rule lives inside a fenced ``<!-- HINDSIGHT:BEGIN -->`` … ``<!-- HINDSIGHT:END -->``
block so we can update or remove it without touching the user's own rules in the
same file.
"""

from __future__ import annotations

from pathlib import Path

BEGIN_MARKER = "<!-- HINDSIGHT:BEGIN -->"
END_MARKER = "<!-- HINDSIGHT:END -->"

# The recall/retain instruction injected into Zed's global rules.
RULE_TEXT = (
    "You have persistent long-term memory through the Hindsight MCP server "
    "(`recall`, `retain`, and `reflect` tools).\n\n"
    "- At the start of each task, call `recall` with the user's request to load "
    "relevant decisions, preferences, and project context before you answer. "
    "Use what's relevant and ignore the rest.\n"
    "- When you learn a durable fact — an architectural decision, a user "
    "preference, a convention, or anything worth remembering across sessions — "
    "call `retain` to store it.\n"
    "- Do not mention these memory operations unless the user asks about them."
)


def default_rules_path() -> Path:
    """Zed's global instructions file (``~/.config/zed/AGENTS.md``)."""
    return Path.home() / ".config" / "zed" / "AGENTS.md"


def _strip_block(text: str) -> str:
    """Remove an existing HINDSIGHT block (and its surrounding blank lines)."""
    start = text.find(BEGIN_MARKER)
    if start == -1:
        return text
    end = text.find(END_MARKER, start)
    if end == -1:
        # Malformed (begin without end) — drop from the marker onward.
        return text[:start].rstrip() + "\n"
    end += len(END_MARKER)
    before = text[:start].rstrip()
    after = text[end:].lstrip()
    if before and after:
        return f"{before}\n\n{after}"
    return (before or after).rstrip() + ("\n" if (before or after) else "")


def render_block(rule_text: str = RULE_TEXT) -> str:
    """Render the fenced HINDSIGHT rule block (no trailing newline)."""
    return f"{BEGIN_MARKER}\n{rule_text.strip()}\n{END_MARKER}"


def write_rule(path: Path, rule_text: str = RULE_TEXT) -> Path:
    """Write/replace Hindsight's rule block in the instructions file at ``path``.

    Preserves any user-authored content and only rewrites our fenced block,
    placing it at the top so the memory rule leads the instructions.
    """
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    base = _strip_block(existing).rstrip()
    block = render_block(rule_text)
    new_text = f"{block}\n\n{base}\n" if base else f"{block}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_text, encoding="utf-8")
    return path


def clear_rule(path: Path) -> Path:
    """Remove Hindsight's rule block from the instructions file, if present.

    Leaves the rest of the file intact. If removing the block empties a file that
    held nothing else, the file is deleted.
    """
    if not path.is_file():
        return path
    existing = path.read_text(encoding="utf-8")
    if BEGIN_MARKER not in existing:
        return path
    stripped = _strip_block(existing).strip()
    if not stripped:
        path.unlink()
        return path
    path.write_text(stripped + "\n", encoding="utf-8")
    return path


def is_installed(path: Path) -> bool:
    """Whether our rule block is present in the instructions file at ``path``."""
    return path.is_file() and BEGIN_MARKER in path.read_text(encoding="utf-8")
