"""Deterministic structural metrics for a rendered document.

Everything here is computed from the markdown alone, with no LLM, so the same
numbers can be produced for any build of the server. This is the half of the
benchmark that answers "did the document survive being edited" — the other half
(is it any *good*) needs a judge and lives in ``judge.py``.

The collapsed-table detector is the one from issue #3361: a separator cell that
shares a physical line with other cells means the table was welded together.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

_SEPARATOR_CELL_RX = re.compile(r"\|\s*:?-{2,}:?\s*\|")
_PURE_SEPARATOR_RX = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")
_TABLE_LINE_RX = re.compile(r"^\s*\|.*\|?\s*$")
_FENCE_RX = re.compile(r"^\s*(```|~~~)")
_INDENTED_LIST_RX = re.compile(r"^\s+[-*+]\s+\S")
_HEADING_RX = re.compile(r"^(#{1,6})\s+\S")
_HARD_BREAK_RX = re.compile(r"\S {2,}$")


class DocumentShape(BaseModel):
    """What a document is made of, and whether it is intact."""

    bytes: int
    lines: int
    headings: list[str] = Field(default_factory=list)
    table_rows: int = 0
    table_blocks: int = 0
    indented_list_lines: int = 0
    hard_break_lines: int = 0
    fenced_blocks: int = 0
    blockquote_lines: int = 0
    collapsed_tables: int = Field(
        default=0,
        description="Physical lines carrying a table separator plus other cells — a welded table (#3361).",
    )
    unbalanced_fence: bool = False


def describe(markdown: str) -> DocumentShape:
    """Measure a document's structure without interpreting its meaning."""
    lines = markdown.splitlines()
    shape = DocumentShape(bytes=len(markdown), lines=len(lines))

    in_fence = False
    prev_was_table = False
    for line in lines:
        if _FENCE_RX.match(line):
            if in_fence:
                shape.fenced_blocks += 1
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        heading = _HEADING_RX.match(line)
        if heading:
            shape.headings.append(line.strip())

        is_table_line = bool(line.strip().startswith("|") and _TABLE_LINE_RX.match(line))
        if is_table_line:
            shape.table_rows += 1
            if not prev_was_table:
                shape.table_blocks += 1
        prev_was_table = is_table_line

        if _SEPARATOR_CELL_RX.search(line) and not _PURE_SEPARATOR_RX.match(line):
            shape.collapsed_tables += 1
        if _INDENTED_LIST_RX.match(line):
            shape.indented_list_lines += 1
        if _HARD_BREAK_RX.search(line):
            shape.hard_break_lines += 1
        if line.lstrip().startswith(">"):
            shape.blockquote_lines += 1

    shape.unbalanced_fence = in_fence
    return shape


class ShapeDelta(BaseModel):
    """What the structure lost (or gained) between two versions of a document."""

    collapsed_tables_introduced: int = 0
    table_rows_lost: int = 0
    indented_list_lines_lost: int = 0
    hard_break_lines_lost: int = 0
    fenced_blocks_lost: int = 0
    blockquote_lines_lost: int = 0
    headings_lost: list[str] = Field(default_factory=list)
    became_unbalanced: bool = False

    @property
    def damaged(self) -> bool:
        """Any structural loss at all. Content changes are not damage; losing a table is."""
        return bool(
            self.collapsed_tables_introduced
            or self.table_rows_lost
            or self.indented_list_lines_lost
            or self.hard_break_lines_lost
            or self.fenced_blocks_lost
            or self.blockquote_lines_lost
            or self.became_unbalanced
        )


def compare_shape(before: DocumentShape, after: DocumentShape) -> ShapeDelta:
    """Structural loss from ``before`` to ``after``.

    Only losses are reported as damage. A refresh legitimately adds rows, list
    items and sections; nothing legitimately welds a table onto one line or
    silently drops a code fence.
    """
    return ShapeDelta(
        collapsed_tables_introduced=max(0, after.collapsed_tables - before.collapsed_tables),
        table_rows_lost=max(0, before.table_rows - after.table_rows),
        indented_list_lines_lost=max(0, before.indented_list_lines - after.indented_list_lines),
        hard_break_lines_lost=max(0, before.hard_break_lines - after.hard_break_lines),
        fenced_blocks_lost=max(0, before.fenced_blocks - after.fenced_blocks),
        blockquote_lines_lost=max(0, before.blockquote_lines - after.blockquote_lines),
        headings_lost=[h for h in before.headings if h not in after.headings],
        became_unbalanced=after.unbalanced_fence and not before.unbalanced_fence,
    )


def damage_in_untouched_sections(before: str, after: str, touched_headings: set[str]) -> ShapeDelta:
    """Structural loss restricted to the sections no operation named.

    Whole-document loss is not evidence of a bug: a refresh that deliberately
    rewrites a section may legitimately turn its nested list into prose, and
    counting that as damage would score an editorial decision as corruption. A
    section nobody named is different — nothing there was supposed to change at
    all, so any structure missing from it was destroyed by the machinery rather
    than by the model.
    """
    before_sections = _sections(before)
    after_sections = _sections(after)
    untouched_before = "\n\n".join(body for heading, body in before_sections.items() if heading not in touched_headings)
    untouched_after = "\n\n".join(
        body
        for heading, body in after_sections.items()
        if heading not in touched_headings and heading in before_sections
    )
    return compare_shape(describe(untouched_before), describe(untouched_after))


def untouched_sections_drifted(before: str, after: str, touched_headings: set[str]) -> list[str]:
    """Headings whose body changed although no operation named their section.

    Section identity is the heading text, which is what a reader sees and what
    survives a rename-free edit. A section the model never named must come
    through byte-identical; anything else is drift the architecture is supposed
    to make impossible.
    """
    before_sections = _sections(before)
    after_sections = _sections(after)
    drifted: list[str] = []
    for heading, body in before_sections.items():
        if heading in touched_headings or heading not in after_sections:
            continue
        if after_sections[heading] != body:
            drifted.append(heading)
    return drifted


def _sections(markdown: str) -> dict[str, str]:
    """Split a document into ``heading -> body``, ignoring headings inside fences."""
    sections: dict[str, str] = {}
    current = ""
    body: list[str] = []
    in_fence = False
    for line in markdown.splitlines():
        if _FENCE_RX.match(line):
            in_fence = not in_fence
            body.append(line)
            continue
        if not in_fence and _HEADING_RX.match(line):
            sections[current] = "\n".join(body).strip()
            current = line.strip()
            body = []
            continue
        body.append(line)
    sections[current] = "\n".join(body).strip()
    return sections
