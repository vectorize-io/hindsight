"""Delta operations for structured mental models.

The LLM's job during a delta refresh is to emit a list of these operations,
each targeting an existing section (by id) or referencing a position relative
to one.  ``apply_operations`` validates and applies each op in turn against a
copy of the document; invalid ops (unknown ``section_id``, out-of-range
``block_index``, a mismatched block ``anchor``, malformed payloads) are
dropped with a debug-friendly reason.

Block-targeting ops (``replace_block``, ``remove_block``, and ``insert_block``
when its index names an existing block) also carry a content ``anchor``: a
verbatim excerpt of the block the LLM believes is at the given index.
``apply_operations`` checks the anchor against the block actually there
before mutating anything. This guards against a wrong-but-in-range index,
which is otherwise indistinguishable from a correct one — the LLM's only way
to know a block's index in the compact JSON it is shown is to count array
elements, and a miscount silently lands the op on the wrong block instead of
failing loudly. See ``serialize_document_for_delta_prompt`` for the
prompt-facing view that annotates each block with its index so the model can
read it instead of counting.

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

import json
import logging
import re
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from hindsight_api.engine.llm_wrapper import parse_llm_json

from .structured_doc import (
    Block,
    BulletListBlock,
    CodeBlock,
    OrderedListBlock,
    ParagraphBlock,
    Section,
    StructuredDocument,
    TableBlock,
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

    ``anchor`` names the block this insert will land before: a verbatim
    excerpt of the block currently at ``index``. Required only when ``index``
    names an existing block (``index < len(section.blocks)``); at the append
    position there is no block to anchor against, so ``anchor`` may be left
    empty there. See ``ReplaceBlockOp`` for the matching rules and why this
    exists.
    """

    op: Literal["insert_block"] = "insert_block"
    section_id: str
    index: int = Field(ge=0)
    anchor: str = ""
    block: Block


class ReplaceBlockOp(_OpBase):
    """Replace the block at ``index`` of an existing section.

    ``anchor`` must be a verbatim excerpt (~50 chars) of the block's own
    content at ``index`` — its ``text`` field, ``items`` joined with a space
    for list blocks, or table ``headers``/``rows`` joined with a space —
    copied from what the model was shown at that index. ``apply_operations``
    compares it (whitespace-normalized) against the block actually there
    before replacing it, and skips the op on a mismatch or a missing anchor
    instead of applying it. This is the guard against a miscounted,
    wrong-but-in-range ``index``: without it, a wrong index is
    indistinguishable from a correct one and silently destroys the wrong
    block's content.
    """

    op: Literal["replace_block"] = "replace_block"
    section_id: str
    index: int = Field(ge=0)
    anchor: str = ""
    block: Block


class RemoveBlockOp(_OpBase):
    """Remove the block at ``index`` of an existing section.

    See ``ReplaceBlockOp`` for the ``anchor`` field's role and matching
    rules — identical here, since removal is equally destructive of the
    wrong block on a miscounted index.
    """

    op: Literal["remove_block"] = "remove_block"
    section_id: str
    index: int = Field(ge=0)
    anchor: str = ""


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

_OPERATION_ADAPTER: TypeAdapter[Operation] = TypeAdapter(Operation)


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
    #: The model's escape hatch: "the evidence I was given is not enough to edit
    #: this document correctly". Only the delta fast path asks for it and only the
    #: fast path reads it — there it means "hand this refresh to the agentic reflect
    #: loop", which can go and retrieve more. Default False, so a model that never
    #: mentions it (every caller before the fast path existed) behaves as it did.
    needs_full_context: bool = False


class DeltaAllOpsInvalidError(ValueError):
    """Raised when the model emitted operations but none survived validation.

    Distinct from an empty ``operations`` array (a legitimate no-op): here every
    op was malformed, so returning zero valid ops would make the caller apply
    nothing and silently drop this refresh's new facts. Raising instead lets the
    caller fall back to a full rewrite, which still integrates the new facts.
    """


def _coerce_needs_full_context(raw: Any) -> bool:
    """Read the escape-hatch flag out of a raw delta payload.

    The delta call is text-mode, not schema-enforced (the discriminated-union JSON
    schema is rejected by some providers — see the caller), so a model can answer
    with the string ``"true"`` where a boolean was asked for. Anything else —
    absent, null, false, a number, prose — reads as False: the flag only ever adds
    a hand-off to the slower agentic loop, so the safe reading is to require an
    explicit yes rather than to guess from a truthy value.
    """
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() == "true"
    return False


def _finalize_operations(
    valid: list[Operation],
    skipped: list[dict[str, Any]],
    *,
    needs_full_context: bool = False,
) -> DeltaOperationList:
    """Build the result, but refuse a wholesale validation failure as a silent no-op."""
    if skipped and not valid:
        raise DeltaAllOpsInvalidError(f"all {len(skipped)} delta operation(s) failed validation")
    return DeltaOperationList(operations=valid, needs_full_context=needs_full_context)


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
    """Parse structured-delta LLM output into a validated operation list.

    ``needs_full_context`` is carried through every branch. It has to be threaded
    explicitly: each branch rebuilds the list from the operations it validated, so
    a flag left on the raw payload would be silently dropped and the fast path
    would run edits the model had just said it could not make safely.
    """
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
        return _finalize_operations(
            valid,
            skipped,
            needs_full_context=_coerce_needs_full_context(raw.get("needs_full_context")),
        )

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
        return _finalize_operations(
            valid,
            skipped,
            needs_full_context=_coerce_needs_full_context(payload.get("needs_full_context")),
        )

    if last_error is not None:
        raise last_error
    return DeltaOperationList()


# Anchor matching -------------------------------------------------------------
#
# Block-targeting ops are validated against the document by *index*, but an
# index alone cannot distinguish a correct one from a miscounted one that
# still happens to be in range. The anchor closes that gap: the LLM quotes a
# short excerpt of the block it believes is at the claimed index, and we
# check that excerpt against the block actually there before mutating it.

#: ~40-60 chars is enough to disambiguate two blocks that happen to share a
#: short common opening (e.g. two bullet items both starting "The system"),
#: while staying cheap for the model to quote and cheap for the prompt to
#: carry every refresh.
_ANCHOR_EXCERPT_CHARS = 50

_WHITESPACE_RX = re.compile(r"\s+")


def _normalize_anchor_text(text: str) -> str:
    """Collapse whitespace runs to a single space and strip.

    Anchors are compared after this normalization so incidental whitespace
    differences (a stray double space, a line-wrapped newline inside the
    model's quoted excerpt) never cause a spurious mismatch. Normalization
    only touches whitespace characters, so any real content difference —
    the case this exists to catch — still causes a mismatch.
    """
    return _WHITESPACE_RX.sub(" ", text).strip()


def _block_content_text(block: Block) -> str:
    """The block's own text content, independent of markdown rendering.

    This mirrors what ``serialize_document_for_delta_prompt`` shows the model
    for each block — the raw ``text`` field, or ``items`` joined with a space
    — rather than ``render_block``'s markdown form (bullet ``- `` prefixes,
    code fences). A model quoting verbatim from what it was shown then
    produces a matching anchor without needing to reconstruct markdown syntax
    it was never shown as such.
    """
    if isinstance(block, (ParagraphBlock, CodeBlock)):
        return block.text
    if isinstance(block, (BulletListBlock, OrderedListBlock)):
        return " ".join(block.items)
    if isinstance(block, TableBlock):
        parts = [" ".join(str(h) for h in block.headers)]
        parts.extend(" ".join(str(c) for c in row) for row in block.rows)
        return " ".join(p for p in parts if p)
    # Newer block types must not fail the refresh. A TypeError here used to
    # abort apply_operations (and the whole refresh) instead of recording an
    # anchor mismatch and skipping just that op. Render best-effort.
    headers = getattr(block, "headers", None)
    rows = getattr(block, "rows", None)
    if headers is not None or rows is not None:
        parts = [" ".join(str(h) for h in (headers or []))]
        parts.extend(" ".join(str(c) for c in row) for row in (rows or []))
        return " ".join(p for p in parts if p)
    text = getattr(block, "text", None)
    if isinstance(text, str):
        return text
    items = getattr(block, "items", None)
    if isinstance(items, list):
        return " ".join(str(i) for i in items)
    return ""


def _anchor_matches(block: Block, anchor: str) -> bool:
    """True if ``anchor`` verbatim-matches (whitespace-normalized) a prefix
    of ``block``'s own content. An empty/whitespace-only anchor never
    matches — see ``_check_block_anchor`` for the "missing anchor" case.
    """
    normalized_anchor = _normalize_anchor_text(anchor)
    if not normalized_anchor:
        return False
    normalized_content = _normalize_anchor_text(_block_content_text(block))
    return normalized_content.startswith(normalized_anchor)


def _check_block_anchor(section: Section, index: int, anchor: str) -> str | None:
    """Validate a block-targeting op's anchor against the block actually at
    ``index`` (the caller must range-check ``index`` first).

    Returns ``None`` when the op may proceed, or a skip reason string when it
    must not. A missing/empty anchor is treated as a mismatch — fail closed —
    so an op from a model (or an older caller) that never adopted the anchor
    contract loses that op rather than risk it landing on the wrong block.
    """
    if not anchor:
        return "missing anchor"
    if not _anchor_matches(section.blocks[index], anchor):
        return f"anchor mismatch at index {index}"
    return None


def serialize_document_for_delta_prompt(doc: StructuredDocument) -> str:
    """Render a document to the JSON the delta-ops prompt shows the model,
    with each block additionally annotated with its own 0-based ``index``
    within its section.

    Without this, the model has to silently count array elements in a
    compact, unannotated JSON dump to know which index a block sits at —
    exactly the failure mode that produces the wrong-but-in-range indices the
    ``anchor`` fields above guard against. Annotating the index removes the
    need to count at all.

    ``index`` is prompt-only. It is not, and must never become, a real field
    on ``Block``: every block subtype declares ``model_config =
    ConfigDict(extra="forbid")``, so nothing that parses this JSON back
    through ``StructuredDocument.model_validate`` can accept it — callers
    parse the model's *operations* against the real document, never this
    view, back into a ``StructuredDocument``.
    """
    sections_out = [
        {
            "id": section.id,
            "heading": section.heading,
            "level": section.level,
            "blocks": [{"index": i, **block.model_dump(mode="json")} for i, block in enumerate(section.blocks)],
        }
        for section in doc.sections
    ]
    return json.dumps({"version": doc.version, "sections": sections_out})


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
    section, out-of-range index, a mismatched or missing block anchor, name
    collision when adding a section) are skipped and recorded in ``skipped``
    with a ``reason`` string.
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
            if op.index < len(section.blocks):
                # `index` names an existing block (the one this insert lands
                # before); at the append position (index == len(blocks))
                # there is no block there to anchor against.
                anchor_reason = _check_block_anchor(section, op.index, op.anchor)
                if anchor_reason is not None:
                    skip(op, anchor_reason)
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
            anchor_reason = _check_block_anchor(section, op.index, op.anchor)
            if anchor_reason is not None:
                skip(op, anchor_reason)
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
            anchor_reason = _check_block_anchor(section, op.index, op.anchor)
            if anchor_reason is not None:
                skip(op, anchor_reason)
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
