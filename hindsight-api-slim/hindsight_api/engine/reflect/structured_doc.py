"""Structured representation of a mental model document.

Why this exists
---------------
Storing mental models as raw markdown forces every refresh to round-trip prose
through an LLM, which then drifts on stylistic details (numbered vs bulleted
lists, casing, separator lines, paraphrasing) even when instructed to preserve
content byte-for-byte. The intrinsic mechanism of an LLM is to *generate* the
next token from a gestalt of the input — not to copy tokens verbatim — so any
"preserve unchanged content" instruction is fundamentally a soft constraint.

The fix is to give the LLM no opportunity to drift on unchanged content. The
structured document is the *source of truth*; the markdown stored in
``mental_models.content`` is a deterministic render of it. Delta refreshes emit
*operations* against the structure (see ``delta_ops.py``); sections and blocks
not mentioned by any operation are physically untouched.

Schema (v2)
-----------
A document is an ordered list of ``Section``s. Each section has:

- ``id``      : stable slug derived from ``heading``, the target of operations
                across refreshes (renames keep the id, see ``rename_section``).
- ``heading`` : the heading text without the ``#`` prefix. Empty for the
                implicit leading section of a document that opens with prose.
- ``level``   : 1 (``#``) … 6 (``######``). Default 2.
- ``blocks``  : ordered list of ``Block``s.

A ``Block`` is an id plus a **verbatim markdown fragment** — one paragraph, one
list, one table, one fenced code block. The fragment is never interpreted: it
is stored as written and rendered as written.

Why blocks are opaque markdown rather than a typed AST
------------------------------------------------------
Schema v1 modelled each block as one of ``paragraph``/``bullet_list``/
``ordered_list``/``code``/``table`` and *parsed* markdown into that union. Every
construct the union could not express — nested lists, list-item continuation
lines, blockquotes, hard line breaks, horizontal rules, HTML, indented code,
table alignment, a table row missing an outer pipe — collapsed into a paragraph
whose lines were joined with spaces. The collapse was a fixed point (re-parsing
the damaged render reproduced it), so a page could never recover: see #3361,
where stored tables were welded onto one line permanently.

An opaque fragment cannot suffer that class of failure. Nothing parses a table,
so nothing can flatten one. The only structure we recognise is the one delta
operations actually address: where sections start (an ATX heading) and where
blocks start (a blank line). Both are recoverable from the text exactly, which
makes :func:`split_markdown` lossless — see ``tests/test_structured_doc.py``.

Blocks carry an ``id`` (v1 addressed them by integer index). An index is a
number the model has to derive by counting, and an off-by-one lands in range
and silently overwrites an unrelated block (#3273). A wrong id does not
resolve, so it is skipped and reported instead of applied.
"""

from __future__ import annotations

import hashlib
import logging
import re

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# Blocks ---------------------------------------------------------------------


class Block(BaseModel):
    """One markdown fragment — a paragraph, a list, a table, a code fence.

    ``text`` is stored exactly as authored (interior indentation, hard line
    breaks and table pipes included) and rendered exactly as stored.
    """

    model_config = ConfigDict(extra="forbid")
    id: str
    text: str


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    heading: str = ""
    level: int = Field(default=2, ge=1, le=6)
    blocks: list[Block] = Field(default_factory=list)

    def block_by_id(self, block_id: str) -> Block | None:
        for b in self.blocks:
            if b.id == block_id:
                return b
        return None

    def block_index(self, block_id: str) -> int | None:
        for i, b in enumerate(self.blocks):
            if b.id == block_id:
                return i
        return None


class StructuredDocument(BaseModel):
    """Top-level structured representation of a mental model."""

    model_config = ConfigDict(extra="forbid")
    version: int = 2
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

    def block_ids(self) -> set[str]:
        return {b.id for s in self.sections for b in s.blocks}


SCHEMA_VERSION = 2

# The id given to the implicit section holding content that precedes the first
# heading. Operations can target it like any other section.
PREAMBLE_SECTION_ID = "preamble"


# Id helpers -----------------------------------------------------------------

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


def make_block_id(text: str, existing: set[str]) -> str:
    """Derive a short, deterministic, document-unique id for a block.

    Content-derived so that splitting the same markdown twice yields the same
    document (tests and the legacy-import path depend on that), and short
    enough that a model can copy one into an operation without slipping a
    character. Ids are *persisted*: once assigned, a block keeps its id even
    when its text is replaced, so ids never shift under an edit the way an
    index does.
    """
    digest = hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
    return make_unique_id(f"b{digest}", existing)


# Renderer -------------------------------------------------------------------


def normalize_block_text(text: str) -> str:
    """Trim blank lines around a fragment without touching its interior.

    Leading/trailing *blank* lines are separators, not content, and the
    renderer re-inserts them. Everything inside survives byte-for-byte:
    ``rstrip()`` here would eat the two trailing spaces that make a markdown
    hard line break, and ``strip()`` would eat the four-space indent of an
    indented code block.
    """
    lines = text.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def render_section(section: Section) -> str:
    """Render a section: heading (when it has one) then blocks, blank-line separated."""
    parts: list[str] = []
    heading = section.heading.strip()
    if heading:
        parts.append("#" * section.level + " " + heading)
    parts.extend(block.text for block in section.blocks if block.text.strip())
    return "\n\n".join(parts)


def render_document(doc: StructuredDocument) -> str:
    """Render the whole document to markdown.

    The output is byte-stable: the same structured input always produces the
    same markdown, and the render survives a further split/render round trip
    unchanged (``render(split(render(doc))) == render(doc)``).
    """
    rendered = [render_section(s) for s in doc.sections]
    body = "\n\n".join(part for part in rendered if part)
    return body + "\n" if body else ""


# Splitter -------------------------------------------------------------------
#
# This is the ONLY place markdown text is inspected, and it recognises exactly
# two things: ATX headings (section boundaries) and blank lines (block
# boundaries), neither of which is honoured inside a fenced code block. Every
# other byte is carried through untouched. There is deliberately no notion of
# "understanding" a table, a list or a paragraph here — see the module
# docstring for why v1's typed parser had to go.

_HEADING_RX = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_FENCE_RX = re.compile(r"^\s*(```+|~~~+)")


def _split_into_blocks(lines: list[str], existing_ids: set[str]) -> list[Block]:
    """Group lines into blocks on blank lines, ignoring blanks inside fences."""
    blocks: list[Block] = []
    current: list[str] = []
    fence: str | None = None

    def flush() -> None:
        text = normalize_block_text("\n".join(current))
        current.clear()
        if not text.strip():
            return
        block_id = make_block_id(text, existing_ids)
        existing_ids.add(block_id)
        blocks.append(Block(id=block_id, text=text))

    for line in lines:
        fence_match = _FENCE_RX.match(line)
        if fence_match:
            marker = fence_match.group(1)[:3]
            if fence is None:
                fence = marker
            elif marker == fence:
                fence = None
            current.append(line)
            continue
        if fence is None and not line.strip():
            flush()
            continue
        current.append(line)
    flush()
    return blocks


def split_markdown(markdown: str) -> StructuredDocument:
    """Split markdown into sections and blocks, losslessly.

    Content that precedes the first heading becomes a leading section with an
    empty ``heading`` (it renders as bare blocks, so nothing is invented).
    Section ids are unique slugs of their headings.

    "Losslessly" means every non-blank line comes back verbatim, in order.
    Rendering the result normalises only whitespace *between* blocks: runs of
    blank lines collapse to one, and a heading line loses its surrounding
    padding. Nothing is reordered, joined, retyped or dropped.
    """
    lines = (markdown or "").splitlines()

    sections: list[Section] = []
    used_ids: set[str] = set()
    block_ids: set[str] = set()
    pending: list[str] = []
    current: Section | None = None
    fence: str | None = None

    def close(section: Section) -> None:
        section.blocks.extend(_split_into_blocks(pending, block_ids))
        pending.clear()
        sections.append(section)

    for line in lines:
        fence_match = _FENCE_RX.match(line)
        if fence_match:
            marker = fence_match.group(1)[:3]
            if fence is None:
                fence = marker
            elif marker == fence:
                fence = None
            pending.append(line)
            continue

        heading_match = None if fence is not None else _HEADING_RX.match(line)
        if heading_match is None:
            pending.append(line)
            continue

        if current is not None:
            close(current)
        elif any(pending_line.strip() for pending_line in pending):
            preamble_id = make_unique_id(PREAMBLE_SECTION_ID, used_ids)
            used_ids.add(preamble_id)
            close(Section(id=preamble_id, heading="", level=2))
        else:
            pending.clear()

        heading = heading_match.group(2).strip()
        section_id = make_unique_id(slugify_heading(heading), used_ids)
        used_ids.add(section_id)
        current = Section(id=section_id, heading=heading, level=len(heading_match.group(1)))

    if current is not None:
        close(current)
    elif any(pending_line.strip() for pending_line in pending):
        preamble_id = make_unique_id(PREAMBLE_SECTION_ID, used_ids)
        used_ids.add(preamble_id)
        close(Section(id=preamble_id, heading="", level=2))

    return StructuredDocument(sections=sections)


def document_from_sections(payload: dict) -> StructuredDocument:
    """Build a document from the ``done`` tool's emitted sections.

    This is the generation path: the model states the document's structure and
    the markdown it renders to is derived, so no markdown the model wrote is
    ever read back to work out what it meant. Ids are assigned here — the model
    is never asked for one, since a generated document has no prior ids to echo.

    Tolerant by design, because a tool call is still model output: a missing
    heading, an out-of-range level or a non-string block is coerced rather than
    rejected. A block holding several blank-line-separated fragments is split
    into one block each, so the document keeps the granularity delta operations
    address even when the model packs a whole section into one string.

    That tolerance now extends to the wrapper itself. A one-section document is
    the shape a model most often flattens — it emits the section *as* the
    document, ``{"heading": …, "level": …, "blocks": […]}``, with no ``sections``
    array around it. Every fact it was asked for is present and correctly
    structured; only the wrapper is missing. Read literally that payload has no
    sections at all, so it rendered to an empty string and reflect raised
    ReflectNoAnswerError — a whole refresh thrown away, and retried against the
    same prompt, over one absent key (observed from Gemini on the mental-model
    refresh path). Take it as the single section it plainly is.
    """
    if "sections" not in payload and isinstance(payload.get("blocks"), list):
        payload = {"sections": [payload]}

    sections: list[Section] = []
    used_ids: set[str] = set()
    block_ids: set[str] = set()

    for raw_section in payload.get("sections") or []:
        if not isinstance(raw_section, dict):
            continue
        heading = str(raw_section.get("heading") or "").strip().lstrip("#").strip()
        try:
            level = int(raw_section.get("level") or 2)
        except (TypeError, ValueError):
            level = 2
        level = min(6, max(1, level))

        lines: list[str] = []
        for raw_block in raw_section.get("blocks") or []:
            if raw_block is None:
                continue
            text = normalize_block_text(str(raw_block))
            if not text.strip():
                continue
            if lines:
                lines.append("")
            lines.extend(text.split("\n"))

        blocks = _split_into_blocks(lines, block_ids)
        if not heading and not blocks:
            continue
        base_id = slugify_heading(heading) if heading else PREAMBLE_SECTION_ID
        section_id = make_unique_id(base_id, used_ids)
        used_ids.add(section_id)
        sections.append(Section(id=section_id, heading=heading, level=level, blocks=blocks))

    return StructuredDocument(sections=sections)


class CanonicalDocument(BaseModel):
    """A document and the structure it renders from — the pair that must be stored together.

    ``markdown`` and ``structure`` are two views of one document, and every write
    of ``mental_models.content`` has to persist both. Storing markdown alone
    leaves the next delta refresh to reconstruct a structure nobody reviewed;
    storing a structure that does not render to the stored markdown is the
    divergence that let a degraded document reach users one refresh after it was
    created (#3361). Returning them as one value makes it hard to write one and
    forget the other.
    """

    model_config = ConfigDict(extra="forbid")
    markdown: str
    structure: StructuredDocument


def canonical_document(markdown: str) -> CanonicalDocument:
    """Split authored markdown into a structure, and render it back.

    The render is what gets stored, so ``content`` is a view of ``structure``
    from the first byte rather than from the first refresh. The split is
    lossless, so this only normalises whitespace *between* blocks.
    """
    structure = split_markdown(markdown)
    return CanonicalDocument(markdown=render_document(structure), structure=structure)


def structured_document_from_stored(
    stored: dict | None,
    markdown: str,
) -> StructuredDocument:
    """Load the delta baseline: the stored structure, or the markdown it renders to.

    The stored JSON is authoritative when it validates against the current
    schema. Anything else is rebuilt from ``markdown`` by :func:`split_markdown`
    — a lossless import, after which the JSON is the source of truth and the
    markdown is its render.

    Falling back is the *normal* path for a model that has one, not just a
    legacy escape hatch: a mental model created with markdown ``content``, or
    restored from an export, has no structure until its first delta refresh.

    It is also how a schema v1 document is handled. v1 blobs are deliberately
    not upgraded field-by-field — its typed blocks were parsed out of this same
    markdown and lost anything the block union could not express (#3361), so the
    markdown is strictly the better source. Migration ``d1e2f3a4b5c6`` clears
    them, so after an upgrade a non-v2 blob should only appear while old and new
    API versions are running side by side; it is logged rather than passed over
    silently so a persistent one is visible.
    """
    if isinstance(stored, dict) and stored:
        if stored.get("version") != SCHEMA_VERSION:
            logger.info(
                f"[STRUCTURED_DOC] stored structure is schema version {stored.get('version')!r}, "
                f"not {SCHEMA_VERSION}; importing the baseline from the stored markdown instead"
            )
        else:
            try:
                return StructuredDocument.model_validate(stored)
            except Exception as exc:  # noqa: BLE001 — any invalid shape falls back to the markdown
                logger.warning(
                    f"[STRUCTURED_DOC] stored structure failed validation ({exc}); "
                    "rebuilding the baseline from the stored markdown"
                )
    return split_markdown(markdown)
