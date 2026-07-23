"""Write Hindsight's global-memory rule into Cascade's ``global_rules.md``.

Cascade applies a single ``~/.codeium/windsurf/memories/global_rules.md`` across
**every** workspace, always on (cap: 6,000 characters). We add a small managed
block there naming the user's cross-project (global) bank, so their
preferences/coding-style memory is active even in repos that never ran ``init``.

This is Cascade-specific; the equivalent for the Devin Local agent is an
``AGENTS.md`` file (see :mod:`hindsight_devin_desktop.devin_local`). The block
mechanics are shared via :mod:`hindsight_devin_desktop.managed_block`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import managed_block
from .managed_block import BEGIN_MARKER, END_MARKER  # re-exported for callers/tests
from .rules import global_rule_body

# Devin's documented cap for global_rules.md.
GLOBAL_RULES_CAP = 6000

__all__ = [
    "BEGIN_MARKER",
    "END_MARKER",
    "GLOBAL_RULES_CAP",
    "GlobalRuleResult",
    "clear_global_rule",
    "default_global_rules_path",
    "is_installed",
    "render_block",
    "write_global_rule",
]


def default_global_rules_path() -> Path:
    """Cascade's global rules file (``~/.codeium/windsurf/memories/global_rules.md``)."""
    return Path.home() / ".codeium" / "windsurf" / "memories" / "global_rules.md"


def render_block(global_bank: str) -> str:
    """The fenced managed block naming the user's global memory bank."""
    return managed_block.render_block(global_rule_body(global_bank))


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
    action, size = managed_block.upsert(path, global_rule_body(global_bank))
    return GlobalRuleResult(action, path, over_cap=size if size > GLOBAL_RULES_CAP else None)


def clear_global_rule(path: Path) -> Path:
    """Remove our managed block; delete the file if nothing else remains."""
    managed_block.clear(path)
    return path


def is_installed(path: Path) -> bool:
    """Whether our managed block is present in ``global_rules.md``."""
    return managed_block.has(path)
