"""A fenced, sentinel-marked block we own inside a file we share with the user.

Several targets are Markdown files the user may also author (Cascade's
``global_rules.md``, Devin Local's ``AGENTS.md``). We only manage a fenced
``<!-- HINDSIGHT:BEGIN -->`` … ``<!-- HINDSIGHT:END -->`` block at the top and
never touch the rest, so update/removal is idempotent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

BEGIN_MARKER = "<!-- HINDSIGHT:BEGIN -->"
END_MARKER = "<!-- HINDSIGHT:END -->"


def render_block(body: str) -> str:
    """Wrap ``body`` in our fenced markers (no trailing newline)."""
    return f"{BEGIN_MARKER}\n{body.strip()}\n{END_MARKER}"


def strip_block(text: str) -> str:
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


def upsert(path: Path, body: str) -> Tuple[str, int]:
    """Write/replace our managed block at the top of ``path``.

    Preserves any user content below it. Returns ``(action, size)`` where
    ``action`` is ``created``/``updated``/``unchanged`` and ``size`` is the new
    file length (for callers that enforce a cap).
    """
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    had_block = BEGIN_MARKER in existing
    base = strip_block(existing).rstrip()
    block = render_block(body)
    new_text = f"{block}\n\n{base}\n" if base else f"{block}\n"

    if had_block and existing == new_text:
        action = "unchanged"
    elif path.is_file():
        action = "updated"
    else:
        action = "created"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_text, encoding="utf-8")
    return action, len(new_text)


def clear(path: Path) -> None:
    """Remove our managed block; delete the file if nothing else remains."""
    if not path.is_file():
        return
    existing = path.read_text(encoding="utf-8")
    if BEGIN_MARKER not in existing:
        return
    stripped = strip_block(existing).strip()
    if stripped:
        path.write_text(stripped + "\n", encoding="utf-8")
    else:
        path.unlink()


def has(path: Path) -> bool:
    """Whether our managed block is present in ``path``."""
    return path.is_file() and BEGIN_MARKER in path.read_text(encoding="utf-8")
