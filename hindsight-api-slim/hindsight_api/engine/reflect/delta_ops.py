"""Delta operations for structured mental models.

The LLM's job during a delta refresh is to emit a list of these operations,
each targeting an existing section (by id) or a block inside one (by id).
``apply_operations`` validates and applies each op in turn against a copy of
the document; invalid ops (unknown ``section_id``, unknown ``block_id``, a
block that lives in a different section) are dropped with a debug-friendly
reason.

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

Why blocks are addressed by id and not by index (#3273):
An index has to be *counted* by the model, and an off-by-one is still in
range, so it silently overwrites an unrelated block and is recorded as a
success — no length change for a shrink guard to notice. An id is copied, not
derived; a wrong one does not resolve and is skipped and reported. Block ids
travel with the document (see ``structured_doc.Block``), so the model only ever
has to echo back a string it was given.

Failure modes are by design conservative: an operation list that fails to
parse against the Pydantic schema, or an LLM that returns invalid ops, results
in zero changes — the document stays as-is. The structure can only get better
or stay the same per refresh, never get worse.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator

from hindsight_api.engine.llm_wrapper import parse_llm_json

from .structured_doc import (
    Block,
    Section,
    StructuredDocument,
    make_block_id,
    make_unique_id,
    normalize_block_text,
    slugify_heading,
)

logger = logging.getLogger(__name__)


# Op payloads ---------------------------------------------------------------


class _OpBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _coerce_block_texts(value: Any) -> Any:
    """Accept a new block written as ``{"id": ..., "text": ...}`` where a string is expected.

    The document the model is shown gives every existing block an ``id``, and
    half the operations address blocks *by* ``block_id``. A model that has just
    read that structure emits ids for the blocks it creates too; the prompt's
    bare-string example is the only thing saying otherwise, and #3901 is 74
    ``add_section`` ops over 30 hours where that was not enough. Because this
    call is deliberately text-mode (the discriminated-union schema is not
    accepted by every provider — see the call site in ``memory_engine``), the
    prompt is the only lever there is, so the parser absorbs the second spelling
    instead of paying a full reflect to reject it.

    The id is *dropped*, not honoured: ``apply_operations`` mints ids for new
    blocks with ``make_block_id`` against the ids already reserved in this batch,
    and taking the model's would reintroduce exactly the collisions that scheme
    exists to prevent. Nothing else is rewritten — an entry that is neither a
    string nor a ``{"text": ...}`` object is passed through so it still fails
    with its own validation error rather than being quietly discarded.
    """
    if not isinstance(value, list):
        return value
    return [item["text"] if isinstance(item, dict) and isinstance(item.get("text"), str) else item for item in value]


class AppendBlockOp(_OpBase):
    """Add a new block at the end of an existing section."""

    op: Literal["append_block"] = "append_block"
    section_id: str
    text: str


class InsertBlockOp(_OpBase):
    """Insert a new block into an existing section.

    ``after_block_id`` names the block the new one follows; ``null`` puts it
    first. An unknown id is a skip, not an append at a guessed position.
    """

    op: Literal["insert_block"] = "insert_block"
    section_id: str
    after_block_id: str | None = None
    text: str


class ReplaceBlockOp(_OpBase):
    """Replace the text of one block, keeping its position and id."""

    op: Literal["replace_block"] = "replace_block"
    section_id: str
    block_id: str
    text: str


class RemoveBlockOp(_OpBase):
    """Remove one block from a section."""

    op: Literal["remove_block"] = "remove_block"
    section_id: str
    block_id: str


class AddSectionOp(_OpBase):
    """Add a brand-new section.

    ``after_section_id`` is optional; when omitted the new section is appended
    at the end. ``new_id`` is optional; when omitted we slugify the heading
    and disambiguate against existing IDs.
    """

    op: Literal["add_section"] = "add_section"
    heading: str
    level: int = Field(default=2, ge=1, le=6)
    blocks: list[str] = Field(default_factory=list)
    after_section_id: str | None = None
    new_id: str | None = None

    @field_validator("blocks", mode="before")
    @classmethod
    def _accept_id_bearing_blocks(cls, value: Any) -> Any:
        return _coerce_block_texts(value)


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
    blocks: list[str] = Field(default_factory=list)

    @field_validator("blocks", mode="before")
    @classmethod
    def _accept_id_bearing_blocks(cls, value: Any) -> Any:
        return _coerce_block_texts(value)


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

_OPERATION_ADAPTER: TypeAdapter[Operation] = TypeAdapter(Operation)

# Payload fields carrying markdown; kept out of the audit summary so a refresh's
# operation log stays readable (and small enough to store in reflect_response).
_BODY_FIELDS = ("text", "blocks")


def _validate_operations_list(raw_ops: Any) -> tuple[list[Operation], list[dict[str, Any]]]:
    """Validate each operation independently; drop invalid ops instead of failing the batch."""
    if not isinstance(raw_ops, list):
        raise TypeError(f"operations must be a list, got {type(raw_ops)!r}")
    valid: list[Operation] = []
    skipped: list[dict[str, Any]] = []
    for i, item in enumerate(raw_ops):
        try:
            valid.append(_OPERATION_ADAPTER.validate_python(item))
        except ValidationError as exc:
            skipped.append({"index": i, "op": item, "error": exc.errors(include_url=False)})
            logger.warning(
                "[STRUCTURED_DELTA] skipping invalid operation at index %s: %s",
                i,
                exc.errors(include_url=False),
            )
    return valid, skipped


class DeltaOperationList(BaseModel):
    """Container for the operations produced by an LLM delta call."""

    model_config = ConfigDict(extra="forbid")
    operations: list[Operation] = Field(default_factory=list)


class DeltaAllOpsInvalidError(ValueError):
    """Raised when the model emitted operations but none survived validation.

    Distinct from an empty ``operations`` array (a legitimate no-op): here every
    op was malformed, so returning zero valid ops would make the caller apply
    nothing while recording a clean refresh, silently dropping this refresh's
    new facts.

    Raising does *not* buy a full rewrite — the caller catches it, records
    ``delta_ops_failed`` and refuses to write, because the reflect candidate
    covers only memories newer than the last refresh and writing it would drop
    the rest of the document. So the document is preserved and the refresh is
    reported as failed; the facts arrive on a later round. That makes every
    raise here the cost of a discarded reflect call, which is why predictable
    model spellings are absorbed in validation (see ``_coerce_block_texts``)
    rather than left to fail.
    """


def _finalize_operations(valid: list[Operation], skipped: list[dict[str, Any]]) -> DeltaOperationList:
    """Build the result, but refuse a wholesale validation failure as a silent no-op."""
    if skipped and not valid:
        raise DeltaAllOpsInvalidError(f"all {len(skipped)} delta operation(s) failed validation")
    return DeltaOperationList(operations=valid)


def _extract_balanced_json_object(text: str) -> str | None:
    """Return the first top-level ``{...}`` slice, ignoring trailing junk."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_delta_operation_list(raw: Any) -> DeltaOperationList:
    """Parse structured-delta LLM output into a validated operation list."""
    if isinstance(raw, DeltaOperationList):
        return raw
    if isinstance(raw, dict):
        ops_raw = raw.get("operations", [])
        valid, skipped = _validate_operations_list(ops_raw)
        if skipped:
            logger.info(
                "[STRUCTURED_DELTA] parsed %s op(s), skipped %s invalid op(s) from dict payload",
                len(valid),
                len(skipped),
            )
        return _finalize_operations(valid, skipped)

    text = (raw or "").strip()
    if not text:
        return DeltaOperationList()

    candidates: list[str] = [text]
    extracted = _extract_balanced_json_object(text)
    if extracted and extracted != text:
        candidates.append(extracted)

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            payload = parse_llm_json(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(payload, list):
            payload = {"operations": payload}
        if not isinstance(payload, dict) or "operations" not in payload:
            last_error = ValueError("delta payload must be an object with an operations array")
            continue
        try:
            valid, skipped = _validate_operations_list(payload["operations"])
        except TypeError as exc:
            last_error = exc
            continue
        if skipped:
            logger.info(
                "[STRUCTURED_DELTA] parsed %s op(s), skipped %s invalid op(s)",
                len(valid),
                len(skipped),
            )
        return _finalize_operations(valid, skipped)

    if last_error is not None:
        raise last_error
    return DeltaOperationList()


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
    return {k: v for k, v in data.items() if k not in _BODY_FIELDS} | {"op": data["op"]}


# A model that wants one more row in a table sometimes emits the row as its own
# block instead of replacing the table (observed against a real provider in
# ``test_document_survives_many_delta_rounds_intact``). Rendered with a blank
# line before it, a bare row is not a table row — it is a broken fragment. The
# prompt asks for ``replace_block`` in that case; this is the safety net for
# when the model does it anyway.
#
# Note the narrowness: this never re-reads or reclassifies stored content. It
# only decides whether a *new* block should join the one before it, and the join
# is a plain concatenation, so nothing can be reinterpreted or lost.
_TABLE_ROW_RX = re.compile(r"^\s*\|")
_TABLE_SEPARATOR_RX = re.compile(r"^\s*\|?[\s:]*-{2,}[\s:|-]*\|?\s*$")


def _all_table_rows(text: str) -> bool:
    lines = text.splitlines()
    return bool(lines) and all(_TABLE_ROW_RX.match(line) for line in lines)


def _is_table(text: str) -> bool:
    """A table needs a header, a separator row, and pipes on every line."""
    lines = text.splitlines()
    return len(lines) >= 2 and _all_table_rows(text) and any(_TABLE_SEPARATOR_RX.match(line) for line in lines)


def _is_orphan_table_rows(text: str) -> bool:
    """Rows with no separator of their own — only meaningful inside a table."""
    return _all_table_rows(text) and not any(_TABLE_SEPARATOR_RX.match(line) for line in text.splitlines())


def apply_operations(
    doc: StructuredDocument,
    operations: list[Operation],
) -> AppliedDelta:
    """Apply a list of operations to a document, returning a new document.

    The original document is never mutated. Invalid operations (unknown
    section, unknown block, a block that belongs to another section) are
    skipped and recorded in ``skipped`` with a ``reason`` string.
    """
    new_doc = doc.model_copy(deep=True)
    applied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    # Ids handed out during this batch stay reserved even if the block that
    # owned one is later removed, so two ops in the same batch can never be
    # given the same id.
    reserved_ids: set[str] = set(new_doc.block_ids())

    def skip(op: Operation, reason: str) -> None:
        entry = _op_summary(op)
        entry["reason"] = reason
        skipped.append(entry)
        logger.debug(f"[STRUCTURED_DELTA] skipping op {entry}")

    def new_block(text: str) -> Block:
        normalized = normalize_block_text(text)
        block_id = make_block_id(normalized, reserved_ids)
        reserved_ids.add(block_id)
        return Block(id=block_id, text=normalized)

    def resolve_block(op: Operation, section: Section, block_id: str) -> int | None:
        index = section.block_index(block_id)
        if index is not None:
            return index
        # Naming a real block in the wrong section is a targeting mistake, not a
        # licence to edit it: report where it actually lives instead of applying
        # the edit somewhere the model did not ask for.
        owner = next((s.id for s in new_doc.sections if s.block_by_id(block_id) is not None), None)
        if owner is not None:
            skip(op, f"block_id {block_id} belongs to section {owner}, not {section.id}")
        else:
            skip(op, f"unknown block_id: {block_id}")
        return None

    def merge_orphan_rows(op: Operation, section: Section, position: int) -> bool:
        """Fold bare table rows into the table they were meant to extend."""
        if position == 0 or not _is_orphan_table_rows(op.text):
            return False
        target = section.blocks[position - 1]
        if not _is_table(target.text):
            return False
        target.text = f"{target.text}\n{normalize_block_text(op.text)}"
        entry = _op_summary(op)
        entry["merged_into_block_id"] = target.id
        applied.append(entry)
        logger.info(
            "[STRUCTURED_DELTA] merged orphan table row(s) into block %s of section %s",
            target.id,
            section.id,
        )
        return True

    def empty_text(op: Operation, text: str) -> bool:
        if text.strip():
            return False
        skip(op, "block text is empty")
        return True

    for op in operations:
        if isinstance(op, AppendBlockOp):
            section = new_doc.section_by_id(op.section_id)
            if section is None:
                skip(op, f"unknown section_id: {op.section_id}")
                continue
            if empty_text(op, op.text):
                continue
            if not merge_orphan_rows(op, section, len(section.blocks)):
                section.blocks.append(new_block(op.text))
                applied.append(_op_summary(op))
            continue

        if isinstance(op, InsertBlockOp):
            section = new_doc.section_by_id(op.section_id)
            if section is None:
                skip(op, f"unknown section_id: {op.section_id}")
                continue
            if empty_text(op, op.text):
                continue
            if op.after_block_id is None:
                position = 0
            else:
                anchor = resolve_block(op, section, op.after_block_id)
                if anchor is None:
                    continue
                position = anchor + 1
            if not merge_orphan_rows(op, section, position):
                section.blocks.insert(position, new_block(op.text))
                applied.append(_op_summary(op))
            continue

        if isinstance(op, ReplaceBlockOp):
            section = new_doc.section_by_id(op.section_id)
            if section is None:
                skip(op, f"unknown section_id: {op.section_id}")
                continue
            if empty_text(op, op.text):
                continue
            index = resolve_block(op, section, op.block_id)
            if index is None:
                continue
            # The id is positional identity, not a content hash: keeping it means
            # an op list that replaces a block and then edits it again still
            # resolves, and the audit trail follows one block across refreshes.
            section.blocks[index] = Block(id=op.block_id, text=normalize_block_text(op.text))
            applied.append(_op_summary(op))
            continue

        if isinstance(op, RemoveBlockOp):
            section = new_doc.section_by_id(op.section_id)
            if section is None:
                skip(op, f"unknown section_id: {op.section_id}")
                continue
            index = resolve_block(op, section, op.block_id)
            if index is None:
                continue
            section.blocks.pop(index)
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
                blocks=[new_block(text) for text in op.blocks if text.strip()],
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
            section.blocks = [new_block(text) for text in op.blocks if text.strip()]
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
