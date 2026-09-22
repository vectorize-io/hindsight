"""Delta operations for structured mental models.

The LLM's job during a delta refresh is to emit a list of these operations,
each targeting an existing section (by id) or a block inside one (by id).
There are two layers of validation here, and they answer differently:

- **Shape** (``parse_delta_operation_list``): does the reply match the schema?
  One op that does not refuses the whole reply, and ``request_delta_operations``
  asks the model again with the errors quoted back. A reply written to a shape
  the model invented is not repaired by keeping the parts that happened to fit.
- **Reference** (``apply_operations``): does the op name something real? An
  unknown ``section_id`` or ``block_id``, or a block that lives in a different
  section, is dropped with a debug-friendly reason and the rest still apply —
  the model addressed a document it misread, which the next refresh sees afresh.

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
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator

from hindsight_api.engine.llm_wrapper import ConfiguredLLMProvider, parse_llm_json

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
    # Strict on purpose (#4443). An unknown field is not a harmless typo: the
    # model that invented a key was writing to a shape it made up, so what it
    # meant to say is as likely to be *in* that key as not, and accepting the op
    # without it writes a partial edit while recording a clean refresh. The
    # answer to a reply that does not follow the schema is to ask again with the
    # error attached — see ``request_delta_operations`` — never to guess which
    # half of it was load-bearing.
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


@dataclass(frozen=True)
class RejectedOperation:
    """One operation the schema refused, and why.

    Carried out of validation rather than logged and dropped, because the text of
    the rejection is what the retry sends back to the model: "operation 4 named an
    unknown field ``block_id``" is actionable, "the reply was invalid" is not.
    """

    index: int
    op: Any
    error: str


def _validate_operations_list(raw_ops: Any) -> list[Operation]:
    """Validate every operation, and refuse the batch if any one of them fails.

    Strict by decision (#4443), where this used to drop the bad op and keep the
    rest. A reply that does not match the schema is not a reply with one bad
    line in it — the model was writing to a shape it invented, so an op it got
    wrong is evidence about the ops around it, and applying those while silently
    discarding this one writes a partial edit and reports a clean refresh. The
    caller asks again with the errors attached instead.
    """
    if not isinstance(raw_ops, list):
        raise TypeError(f"operations must be a list, got {type(raw_ops)!r}")
    operations: list[Operation] = []
    rejected: list[RejectedOperation] = []
    for i, item in enumerate(raw_ops):
        try:
            operations.append(_OPERATION_ADAPTER.validate_python(item))
        except ValidationError as exc:
            # One line per pydantic error, in the words the model needs to fix it.
            error = "; ".join(
                f"{'.'.join(str(piece) for piece in e['loc']) or '(operation)'}: {e['msg']}"
                for e in exc.errors(include_url=False)
            )
            rejected.append(RejectedOperation(index=i, op=item, error=error))
    if rejected:
        for rejection in rejected:
            logger.warning("[STRUCTURED_DELTA] rejected operation at index %s: %s", rejection.index, rejection.error)
        raise DeltaOperationsInvalidError(
            f"{len(rejected)} of {len(raw_ops)} delta operation(s) failed validation", rejected
        )
    return operations


class DeltaOperationList(BaseModel):
    """Container for the operations produced by an LLM delta call."""

    model_config = ConfigDict(extra="forbid")
    operations: list[Operation] = Field(default_factory=list)


class DeltaOperationsInvalidError(ValueError):
    """Raised when any operation in the model's reply failed validation.

    Not a partial success: the whole reply is refused, because an op the model
    got wrong says the reply was written to a shape it invented, and keeping the
    survivors would write half an edit and record it as a clean refresh.

    The caller retries the call once with ``rejected`` fed back to the model
    (``request_delta_operations``). If the second reply is bad too, the refresh
    records ``delta_ops_failed`` and refuses to write: the reflect candidate
    covers only memories newer than the last refresh, so writing it would drop
    the rest of the document. The document is preserved, the facts arrive on a
    later round, and the cost is the discarded reflect — which is why the
    predictable model spellings that are *not* schema violations are absorbed in
    validation instead (see ``_coerce_block_texts``).
    """

    def __init__(self, message: str, rejected: list[RejectedOperation]) -> None:
        super().__init__(message)
        self.rejected = rejected


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
        return DeltaOperationList(operations=_validate_operations_list(raw.get("operations", [])))

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
            operations = _validate_operations_list(payload["operations"])
        except TypeError as exc:
            last_error = exc
            continue
        return DeltaOperationList(operations=operations)

    if last_error is not None:
        raise last_error
    return DeltaOperationList()


# Asking the model ----------------------------------------------------------


def _correction_prompt(error: Exception, rejected: list[RejectedOperation]) -> str:
    """The follow-up turn: what was wrong, quoted, and what to send instead."""
    lines = [
        "That reply could not be used. Every operation is validated against the "
        "schema, and the whole reply is refused when any one of them fails — so "
        "nothing you sent has been applied.",
        "",
    ]
    if rejected:
        lines.append("What failed:")
        for rejection in rejected:
            lines.append(f"- operation at index {rejection.index}: {rejection.error}")
            lines.append(f"  you sent: {json.dumps(rejection.op, ensure_ascii=False, default=str)[:600]}")
    else:
        lines.append(f"What failed: {error}")
    lines += [
        "",
        "Send the COMPLETE list again — every operation you still intend, including "
        "the ones that were fine — as a single JSON object with one key, "
        "``operations``. Each operation carries exactly the keys its shape lists and "
        "no others: no key you invented, no key borrowed from a different operation, "
        "no key set to null to stand in for one it does not take. Emit no prose "
        "outside the JSON object.",
    ]
    return "\n".join(lines)


async def request_delta_operations(
    llm: ConfiguredLLMProvider,
    *,
    system_prompt: str,
    user_prompt: str,
    scope: str,
    **call_kwargs: Any,
) -> DeltaOperationList:
    """Ask the model for an operation list, and once more with the errors if it is refused.

    Every delta-shaped call goes through here — the refresh's edit pass and the
    retraction pass both — so a reply that misses the schema gets the same second
    chance either way, and a new caller inherits it by construction.

    The retry is worth making only because it changes the *input*: these calls run
    at a fixed low temperature, and #3421 is the proof that re-sending an identical
    prompt reproduces an identical bad reply, retry after retry. So the follow-up
    turn quotes the operations that were refused and the reason for each, which is
    the one thing the model can act on.

    Exactly one retry. A third ask would repeat the second's input and so its
    output; past that the caller's own failure path takes over — the document is
    preserved and the task retries with a fresh reflect behind it, a better use
    of the next attempt.

    The retry APPENDS to the first request and never rewrites it, so the system
    prompt and the whole document stay a byte-identical prefix and the provider's
    prompt cache still covers them on the second call.
    """
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    first = await llm.call(messages=messages, scope=scope, **call_kwargs)
    try:
        return parse_delta_operation_list(first.content)
    except (ValueError, TypeError) as exc:  # DeltaOperationsInvalidError and JSON errors are ValueErrors
        rejected = exc.rejected if isinstance(exc, DeltaOperationsInvalidError) else []
        logger.warning("[STRUCTURED_DELTA] %s reply refused (%s); asking again with the errors", scope, exc)
        correction = _correction_prompt(exc, rejected)
    retry_messages = [
        *messages,
        {"role": "assistant", "content": first.content or ""},
        {"role": "user", "content": correction},
    ]
    second = await llm.call(messages=retry_messages, scope=scope, **call_kwargs)
    return parse_delta_operation_list(second.content)


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
