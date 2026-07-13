"""Write Hindsight's global-memory rule into Devin's ``global_rules.md``.

Devin Desktop applies a single ``~/.codeium/windsurf/memories/global_rules.md``
across **every** workspace, always on (cap: 6,000 characters). We add a small
managed block there naming the user's cross-project (global) bank, so their
preferences/coding-style memory is active even in repos that never ran ``init``.

Unlike the per-project rule file (which we own entirely), this file is shared
with the user's own global rules — so we only manage a fenced
``<!-- HINDSIGHT:BEGIN -->`` … ``<!-- HINDSIGHT:END -->`` block and never touch
the rest.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

BEGIN_MARKER = "<!-- HINDSIGHT:BEGIN -->"
END_MARKER = "<!-- HINDSIGHT:END -->"

# Devin's documented cap for global_rules.md.
GLOBAL_RULES_CAP = 6000


def default_global_rules_path() -> Path:
    """Devin's global rules file (``~/.codeium/windsurf/memories/global_rules.md``)."""
    return Path.home() / ".codeium" / "windsurf" / "memories" / "global_rules.md"


def render_block(global_bank: str) -> str:
    """The fenced managed block naming the user's global memory bank."""
    body = (
        "You have persistent cross-project long-term memory in the Hindsight MCP "
        f'server\'s `{global_bank}` bank (bank_id: "{global_bank}"). At the start '
        "of a task, `recall` from it for the user's preferences and coding style; "
        "`retain` durable user-level facts (preferences, style, identity) to it. "
        "A specific project may define its own additional project bank in that "
        "project's rules. Briefly mention when you use memory."
    )
    return f"{BEGIN_MARKER}\n{body}\n{END_MARKER}"


def _strip_block(text: str) -> str:
    """Remove an existing HINDSIGHT block (and its surrounding blank lines)."""
    start = text.find(BEGIN_MARKER)
    if start == -1:
        return text
    end = text.find(END_MARKER, start)
    if end == -1:
        return text[:start].rstrip() + "\n"
    end += len(END_MARKER)
    before = text[:start].rstrip()
    after = text[end:].lstrip()
    if before and after:
        return f"{before}\n\n{after}"
    return (before or after).rstrip() + ("\n" if (before or after) else "")


@dataclass
class GlobalRuleResult:
    """Outcome of editing ``global_rules.md``.

    ``action`` is ``created``/``updated``/``unchanged``. ``over_cap`` is the new
    total length when it exceeds :data:`GLOBAL_RULES_CAP` (Devin truncates past
    it), else ``None``.
    """

    action: str
    path: Path
    over_cap: Optional[int] = None


def write_global_rule(path: Path, global_bank: str) -> GlobalRuleResult:
    """Write/replace our managed block at the top of ``global_rules.md``.

    Preserves any user-authored content below our block. Reports ``over_cap`` if
    the resulting file exceeds Devin's 6,000-char limit (we still write — the
    user chooses what to trim — but we warn).
    """
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    had_block = BEGIN_MARKER in existing
    base = _strip_block(existing).rstrip()
    block = render_block(global_bank)
    new_text = f"{block}\n\n{base}\n" if base else f"{block}\n"

    if had_block and existing == new_text:
        action = "unchanged"
    elif path.is_file():
        action = "updated"
    else:
        action = "created"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_text, encoding="utf-8")
    over = len(new_text) if len(new_text) > GLOBAL_RULES_CAP else None
    return GlobalRuleResult(action, path, over_cap=over)


def clear_global_rule(path: Path) -> Path:
    """Remove our managed block; delete the file if nothing else remains."""
    if not path.is_file():
        return path
    existing = path.read_text(encoding="utf-8")
    if BEGIN_MARKER not in existing:
        return path
    stripped = _strip_block(existing).strip()
    if stripped:
        path.write_text(stripped + "\n", encoding="utf-8")
    else:
        path.unlink()
    return path


def is_installed(path: Path) -> bool:
    """Whether our managed block is present in ``global_rules.md``."""
    return path.is_file() and BEGIN_MARKER in path.read_text(encoding="utf-8")
