"""Manage Hindsight's recalled-memory block inside a project's Zed instructions file.

Zed always includes a project's instructions file in every agent conversation,
but it reads **only the first matching file** from a fixed priority list. So we
must not blindly create ``.rules`` — if a project already has an ``AGENTS.md``
(or ``CLAUDE.md``, ``.cursorrules``, …) that the user maintains, creating a
higher-priority ``.rules`` would silently suppress it.

Instead we find the instruction file Zed will *actually* read (the first that
exists), and write our memories into a fenced ``<!-- HINDSIGHT:BEGIN -->`` …
``<!-- HINDSIGHT:END -->`` block inside it, leaving the rest untouched. Only if
the project has no instruction file at all do we create ``.rules``.

Priority order verified against zed.dev/docs/ai/instructions.
"""

from pathlib import Path
from typing import Optional

# Zed's project instruction files, highest priority first. Zed uses the first
# one that exists; ours must target that same file so the block is actually read.
INSTRUCTION_FILES = (
    ".rules",
    ".cursorrules",
    ".windsurfrules",
    ".clinerules",
    ".github/copilot-instructions.md",
    "AGENT.md",
    "AGENTS.md",
    "CLAUDE.md",
    "GEMINI.md",
)

# Fallback file created when a project has no instruction file yet.
DEFAULT_INSTRUCTION_FILE = ".rules"

BEGIN_MARKER = "<!-- HINDSIGHT:BEGIN -->"
END_MARKER = "<!-- HINDSIGHT:END -->"


def resolve_instruction_file(project: Path) -> Path:
    """Return the instruction file Zed will read for *project*.

    The first existing file in priority order; if none exists, the path where we
    should create our own (``.rules``). The returned path may not exist yet.
    """
    for name in INSTRUCTION_FILES:
        candidate = project / name
        if candidate.is_file():
            return candidate
    return project / DEFAULT_INSTRUCTION_FILE


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


def render_block(memory_text: str) -> str:
    """Render the fenced HINDSIGHT block for *memory_text* (no trailing newline)."""
    body = memory_text.strip()
    return f"{BEGIN_MARKER}\n{body}\n{END_MARKER}"


def write_memory_block(project: Path, memory_text: str, *, preamble: Optional[str] = None) -> Path:
    """Write/replace Hindsight's memory block in *project*'s instruction file.

    Preserves any user-authored content in the file and only rewrites our fenced
    block. Returns the path written. An empty ``memory_text`` removes the block
    (see :func:`clear_memory_block`) so stale memory never lingers.
    """
    if not memory_text.strip():
        return clear_memory_block(project)

    target = resolve_instruction_file(project)
    existing = target.read_text(encoding="utf-8") if target.is_file() else ""
    base = _strip_block(existing).rstrip()

    block_body = memory_text.strip()
    if preamble:
        block_body = f"{preamble.strip()}\n\n{block_body}"
    block = render_block(block_body)

    # Our block goes at the top so memories lead the instructions, with the
    # user's existing content following.
    if base:
        new_text = f"{block}\n\n{base}\n"
    else:
        new_text = f"{block}\n"

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(new_text, encoding="utf-8")
    return target


def clear_memory_block(project: Path) -> Path:
    """Remove Hindsight's block from *project*'s instruction file, if present.

    Leaves the rest of the file intact. If removing the block empties a file we
    created (``.rules`` with nothing but our block), the file is deleted.
    """
    target = resolve_instruction_file(project)
    if not target.is_file():
        return target
    existing = target.read_text(encoding="utf-8")
    if BEGIN_MARKER not in existing:
        return target
    stripped = _strip_block(existing).strip()
    if not stripped and target.name == DEFAULT_INSTRUCTION_FILE:
        # The file held only our block and we created it — remove it entirely.
        target.unlink()
        return target
    target.write_text((stripped + "\n") if stripped else "", encoding="utf-8")
    return target
