"""Delta operations for structured mental models.

The LLM's job during a delta refresh is to emit a list of these operations,
each targeting an existing section (by id) or referencing a position relative
to one.  ``apply_operations`` validates and applies each op in turn against a
copy of the document; invalid ops (unknown ``section_id``, out-of-range
``block_index``, malformed payloads) are dropped with a debug-friendly reason.

Sections and blocks not mentioned by any op are physically copied through
unchanged — there is no LLM-mediated re-emission of unchanged text, so prose
drift is structurally impossible.

Why operations and not "output the new structured doc":
- "Output the new doc" still asks the LLM to *generate* every section's
  blocks, including ones it didn't intend to modify, which gives it the same
  opportunity to drift.
- Operations make the no-change case mechanical: zero ops → identical doc.
- Operations are auditable: each refresh produces a log of exactly what
  changed, useful for debugging the LLM's behaviour and explaining diffs.

Failure modes are by design conservative: an operation list that fails to
parse against the Pydantic schema, or an LLM that returns invalid ops, results
in zero changes — the document stays as-is. The structure can only get better
or stay the same per refresh, never get worse.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from .structured_doc import (
    Block,
    Section,
    StructuredDocument,
    make_unique_id,
    slugify_heading,
)

logger = logging.getLogger(__name__)


# Op payloads ---------------------------------------------------------------


class _OpBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AppendBlockOp(_OpBase):
    """Add a new block at the end of an existing section."""

    op: Literal["append_block"] = "append_block"
    section_id: str
    block: Block


class InsertBlockOp(_OpBase):
    """Insert a new block at ``index`` in an existing section.

    ``index`` may equal ``len(section.blocks)`` (append) but not be greater.
    """

    op: Literal["insert_block"] = "insert_block"
    section_id: str
    index: int = Field(ge=0)
    block: Block


class ReplaceBlockOp(_OpBase):
    """Replace the block at ``index`` of an existing section."""

    op: Literal["replace_block"] = "replace_block"
    section_id: str
    index: int = Field(ge=0)
    block: Block


class RemoveBlockOp(_OpBase):
    """Remove the block at ``index`` of an existing section."""

    op: Literal["remove_block"] = "remove_block"
    section_id: str
    index: int = Field(ge=0)


class AddSectionOp(_OpBase):
    """Add a brand-new section.

    ``after_section_id`` is optional; when omitted the new section is appended
    at the end. ``new_id`` is optional; when omitted we slugify the heading
    and disambiguate against existing IDs.
    """

    op: Literal["add_section"] = "add_section"
    heading: str
    level: int = Field(default=2, ge=1, le=6)
    blocks: list[Block] = Field(default_factory=list)
    after_section_id: str | None = None
    new_id: str | None = None


class RemoveSectionOp(_OpBase):
    """Remove an entire section by id."""

    op: Literal["remove_section"] = "remove_section"
    section_id: str


class ReplaceSectionBlocksOp(_OpBase):
    """Replace all blocks of a section in one go.

    Used when most of a section's contents are stale and rebuilding it as a
    unit is clearer than emitting many block-level ops. The section's heading
    and id are preserved.
    """

    op: Literal["replace_section_blocks"] = "replace_section_blocks"
    section_id: str
    blocks: list[Block] = Field(default_factory=list)


class RenameSectionOp(_OpBase):
    """Rename a section's heading. The id is unchanged so future ops still resolve."""

    op: Literal["rename_section"] = "rename_section"
    section_id: str
    new_heading: str


Operation = Annotated[
    Union[
        AppendBlockOp,
        InsertBlockOp,
        ReplaceBlockOp,
        RemoveBlockOp,
        AddSectionOp,
        RemoveSectionOp,
        ReplaceSectionBlocksOp,
        RenameSectionOp,
    ],
    Field(discriminator="op"),
]


class DeltaOperationList(BaseModel):
    """Container for the operations produced by an LLM delta call."""

    model_config = ConfigDict(extra="forbid")
    operations: list[Operation] = Field(default_factory=list)


# Application ---------------------------------------------------------------


class AppliedDelta(BaseModel):
    """Outcome of applying a list of operations to a document."""

    model_config = ConfigDict(extra="forbid")

    document: StructuredDocument
    applied: list[dict[str, Any]] = Field(default_factory=list)
    skipped: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def changed(self) -> bool:
        return len(self.applied) > 0


def _op_summary(op: Operation) -> dict[str, Any]:
    """Compact dict suitable for the audit trail."""
    data = op.model_dump()
    return {k: v for k, v in data.items() if k != "block" and k != "blocks"} | {
        "op": data["op"],
    }


def apply_operations(
    doc: StructuredDocument,
    operations: list[Operation],
) -> AppliedDelta:
    """Apply a list of operations to a document, returning a new document.

    The original document is never mutated. Invalid operations (unknown
    section, out-of-range index, name collision when adding a section) are
    skipped and recorded in ``skipped`` with a ``reason`` string.
    """
    new_doc = doc.model_copy(deep=True)
    applied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    def skip(op: Operation, reason: str) -> None:
        entry = _op_summary(op)
        entry["reason"] = reason
        skipped.append(entry)
        logger.debug(f"[STRUCTURED_DELTA] skipping op {entry}")

    for op in operations:
        if isinstance(op, AppendBlockOp):
            section = new_doc.section_by_id(op.section_id)
            if section is None:
                skip(op, f"unknown section_id: {op.section_id}")
                continue
            section.blocks.append(op.block)
            applied.append(_op_summary(op))
            continue

        if isinstance(op, InsertBlockOp):
            section = new_doc.section_by_id(op.section_id)
            if section is None:
                skip(op, f"unknown section_id: {op.section_id}")
                continue
            if op.index > len(section.blocks):
                skip(
                    op,
                    f"index out of range: {op.index} > {len(section.blocks)}",
                )
                continue
            section.blocks.insert(op.index, op.block)
            applied.append(_op_summary(op))
            continue

        if isinstance(op, ReplaceBlockOp):
            section = new_doc.section_by_id(op.section_id)
            if section is None:
                skip(op, f"unknown section_id: {op.section_id}")
                continue
            if op.index >= len(section.blocks):
                skip(
                    op,
                    f"index out of range: {op.index} >= {len(section.blocks)}",
                )
                continue
            section.blocks[op.index] = op.block
            applied.append(_op_summary(op))
            continue

        if isinstance(op, RemoveBlockOp):
            section = new_doc.section_by_id(op.section_id)
            if section is None:
                skip(op, f"unknown section_id: {op.section_id}")
                continue
            if op.index >= len(section.blocks):
                skip(
                    op,
                    f"index out of range: {op.index} >= {len(section.blocks)}",
                )
                continue
            section.blocks.pop(op.index)
            applied.append(_op_summary(op))
            continue

        if isinstance(op, AddSectionOp):
            existing_ids = {s.id for s in new_doc.sections}
            base_id = op.new_id or slugify_heading(op.heading)
            section_id = make_unique_id(base_id, existing_ids)
            new_section = Section(
                id=section_id,
                heading=op.heading,
                level=op.level,
                blocks=list(op.blocks),
            )
            if op.after_section_id is None:
                new_doc.sections.append(new_section)
            else:
                idx = new_doc.section_index(op.after_section_id)
                if idx is None:
                    skip(op, f"unknown after_section_id: {op.after_section_id}")
                    continue
                new_doc.sections.insert(idx + 1, new_section)
            entry = _op_summary(op)
            entry["assigned_id"] = section_id
            applied.append(entry)
            continue

        if isinstance(op, RemoveSectionOp):
            idx = new_doc.section_index(op.section_id)
            if idx is None:
                skip(op, f"unknown section_id: {op.section_id}")
                continue
            new_doc.sections.pop(idx)
            applied.append(_op_summary(op))
            continue

        if isinstance(op, ReplaceSectionBlocksOp):
            section = new_doc.section_by_id(op.section_id)
            if section is None:
                skip(op, f"unknown section_id: {op.section_id}")
                continue
            section.blocks = list(op.blocks)
            applied.append(_op_summary(op))
            continue

        if isinstance(op, RenameSectionOp):
            section = new_doc.section_by_id(op.section_id)
            if section is None:
                skip(op, f"unknown section_id: {op.section_id}")
                continue
            section.heading = op.new_heading
            applied.append(_op_summary(op))
            continue

        skip(op, f"unhandled op type: {type(op).__name__}")  # pragma: no cover

    return AppliedDelta(document=new_doc, applied=applied, skipped=skipped)
