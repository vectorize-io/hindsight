"""Structured representation of a mental model document.

Why this exists
---------------
Storing mental models as raw markdown forces every refresh to round-trip prose
through an LLM, which then drifts on stylistic details (numbered vs bulleted
lists, casing, separator lines, paraphrasing) even when instructed to preserve
content byte-for-byte. The intrinsic mechanism of an LLM is to *generate* the
next token from a gestalt of the input — not to copy tokens verbatim — so any
"preserve unchanged content" instruction is fundamentally a soft constraint.

The fix is to give the LLM no opportunity to drift on unchanged content. We
keep an authoritative structured representation of the document; the markdown
shown to users is a deterministic render of that structure. Delta refreshes
emit *operations* against the structure (see ``delta_ops.py``); sections and
blocks not mentioned by any operation are physically untouched.

Schema (v1)
-----------
A document is an ordered list of ``Section``s.  Each section has:
- ``id``    : stable slug derived from ``heading`` (used as the operation
              target across refreshes; surviving renames is a separate
              concern handled by an explicit ``rename`` op).
- ``heading``: the markdown heading text (without the ``#`` prefix).
- ``level`` : 1 (``#``) … 6 (``######``).  Default 2.
- ``blocks``: ordered list of typed blocks — paragraph, bullet_list,
              ordered_list, code.

The schema is intentionally narrow: it covers what real mental-model documents
actually contain (the kind a coding agent writes for itself or a user writes as
a "skill" doc).  Tables, images, and raw HTML are out of scope until needed.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

# Blocks ---------------------------------------------------------------------


class ParagraphBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["paragraph"] = "paragraph"
    text: str


class BulletListBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["bullet_list"] = "bullet_list"
    items: list[str] = Field(default_factory=list)


class OrderedListBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["ordered_list"] = "ordered_list"
    items: list[str] = Field(default_factory=list)


class CodeBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["code"] = "code"
    language: str = ""
    text: str


Block = Annotated[
    Union[ParagraphBlock, BulletListBlock, OrderedListBlock, CodeBlock],
    Field(discriminator="type"),
]


# Section / Document ---------------------------------------------------------


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    heading: str
    level: int = Field(default=2, ge=1, le=6)
    blocks: list[Block] = Field(default_factory=list)


class StructuredDocument(BaseModel):
    """Top-level structured representation of a mental model."""

    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    sections: list[Section] = Field(default_factory=list)

    def section_by_id(self, section_id: str) -> Section | None:
        for s in self.sections:
            if s.id == section_id:
                return s
        return None

    def section_index(self, section_id: str) -> int | None:
        for i, s in enumerate(self.sections):
            if s.id == section_id:
                return i
        return None


# Slug helpers ---------------------------------------------------------------

_SLUG_RX = re.compile(r"[^a-z0-9]+")


def slugify_heading(heading: str) -> str:
    """Stable, deterministic slug from a heading.

    "Stop Conditions" -> "stop-conditions"
    "Inputs and Context" -> "inputs-and-context"
    """
    slug = _SLUG_RX.sub("-", heading.strip().lower()).strip("-")
    return slug or "section"


def make_unique_id(base: str, existing: set[str]) -> str:
    """Disambiguate by appending -2, -3, … if the slug is already in use."""
    if base not in existing:
        return base
    i = 2
    while f"{base}-{i}" in existing:
        i += 1
    return f"{base}-{i}"


# Renderer -------------------------------------------------------------------


def render_block(block: Block) -> str:
    """Render a single block to markdown. No trailing newline."""
    if isinstance(block, ParagraphBlock):
        return block.text.rstrip()
    if isinstance(block, BulletListBlock):
        return "\n".join(f"- {item.rstrip()}" for item in block.items)
    if isinstance(block, OrderedListBlock):
        return "\n".join(f"{i + 1}. {item.rstrip()}" for i, item in enumerate(block.items))
    if isinstance(block, CodeBlock):
        fence_lang = block.language or ""
        return f"```{fence_lang}\n{block.text}\n```"
    raise TypeError(f"Unknown block type: {type(block)!r}")


def render_section(section: Section) -> str:
    """Render a section: heading + blank line + blocks separated by blank lines."""
    parts = ["#" * section.level + " " + section.heading.strip()]
    for block in section.blocks:
        parts.append("")  # blank line before each block
        parts.append(render_block(block))
    return "\n".join(parts)


def render_document(doc: StructuredDocument) -> str:
    """Render the whole document. Sections separated by a single blank line.

    The output is byte-stable: same structured input always produces the same
    markdown, modulo the inherent ordering of sections/blocks/items.
    """
    if not doc.sections:
        return ""
    return "\n\n".join(render_section(s) for s in doc.sections) + "\n"


# Parser ---------------------------------------------------------------------
#
# The parser is intentionally lenient: it accepts the markdown produced by
# our own renderer (round-trip-safe) and the markdown an LLM tends to produce
# for mental-model documents.  It is *not* a general CommonMark parser — it
# does not need to be.  When it cannot classify a block it falls back to a
# paragraph so that no content is silently dropped.

_HEADING_RX = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_BULLET_RX = re.compile(r"^\s*[-*+]\s+(.*)$")
_ORDERED_RX = re.compile(r"^\s*\d+[.)]\s+(.*)$")
_FENCE_RX = re.compile(r"^```([A-Za-z0-9_+-]*)\s*$")


def _strip_separators(lines: list[str]) -> list[str]:
    """Drop horizontal-rule lines (`---`, `***`) used as section separators.

    Our renderer never emits these, but LLM output frequently includes them
    between sections; treating them as blank lines avoids parsing them as
    paragraphs.
    """
    return ["" if re.fullmatch(r"\s*([-*_])\1{2,}\s*", line) else line for line in lines]


def _split_blocks(lines: list[str]) -> list[list[str]]:
    """Group consecutive non-blank lines into block chunks."""
    chunks: list[list[str]] = []
    current: list[str] = []
    in_fence = False
    for line in lines:
        if _FENCE_RX.match(line):
            current.append(line)
            in_fence = not in_fence
            continue
        if in_fence:
            current.append(line)
            continue
        if line.strip() == "":
            if current:
                chunks.append(current)
                current = []
        else:
            current.append(line)
    if current:
        chunks.append(current)
    return chunks


def _parse_block(chunk: list[str]) -> Block:
    """Parse a single non-empty chunk into a block."""
    if chunk and _FENCE_RX.match(chunk[0]):
        m = _FENCE_RX.match(chunk[0])
        lang = m.group(1) if m else ""
        body_lines = chunk[1:]
        if body_lines and _FENCE_RX.match(body_lines[-1]):
            body_lines = body_lines[:-1]
        return CodeBlock(language=lang, text="\n".join(body_lines))

    if all(_BULLET_RX.match(line) for line in chunk):
        items = []
        for line in chunk:
            m = _BULLET_RX.match(line)
            assert m is not None
            items.append(m.group(1).strip())
        return BulletListBlock(items=items)

    if all(_ORDERED_RX.match(line) for line in chunk):
        items = []
        for line in chunk:
            m = _ORDERED_RX.match(line)
            assert m is not None
            items.append(m.group(1).strip())
        return OrderedListBlock(items=items)

    return ParagraphBlock(text=" ".join(line.strip() for line in chunk).strip())


def parse_markdown(markdown: str) -> StructuredDocument:
    """Best-effort parse of a markdown document into the structured schema.

    Sections are introduced by ATX headings (``#``..``######``).  Anything
    before the first heading is wrapped into an implicit "Overview" section
    so we never silently drop user content.  Section IDs are unique slugs of
    their headings.
    """
    raw_lines = (markdown or "").splitlines()
    lines = _strip_separators(raw_lines)

    sections: list[Section] = []
    used_ids: set[str] = set()
    pending: list[str] = []
    current: Section | None = None

    def flush_pending_into(section: Section) -> None:
        if not pending:
            return
        for chunk in _split_blocks(pending):
            section.blocks.append(_parse_block(chunk))
        pending.clear()

    for line in lines:
        m = _HEADING_RX.match(line)
        if m:
            if current is not None:
                flush_pending_into(current)
                sections.append(current)
            elif pending:
                # Content before the first heading: wrap in implicit section.
                base = "overview"
                section_id = make_unique_id(base, used_ids)
                used_ids.add(section_id)
                implicit = Section(id=section_id, heading="Overview", level=2)
                flush_pending_into(implicit)
                sections.append(implicit)
            level = len(m.group(1))
            heading = m.group(2).strip()
            section_id = make_unique_id(slugify_heading(heading), used_ids)
            used_ids.add(section_id)
            current = Section(id=section_id, heading=heading, level=level)
        else:
            pending.append(line)

    if current is not None:
        flush_pending_into(current)
        sections.append(current)
    elif pending:
        base = "overview"
        section_id = make_unique_id(base, used_ids)
        used_ids.add(section_id)
        implicit = Section(id=section_id, heading="Overview", level=2)
        flush_pending_into(implicit)
        sections.append(implicit)

    return StructuredDocument(sections=sections)
