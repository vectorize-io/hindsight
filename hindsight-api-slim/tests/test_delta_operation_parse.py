"""Tests for structured-delta LLM JSON parsing."""

from __future__ import annotations

import pytest

from hindsight_api.engine.reflect.delta_ops import (
    AddSectionOp,
    AppendBlockOp,
    DeltaAllOpsInvalidError,
    DeltaOperationList,
    ReplaceSectionBlocksOp,
    apply_operations,
    parse_delta_operation_list,
)
from hindsight_api.engine.reflect.structured_doc import Block, Section, StructuredDocument


def test_parse_delta_operation_list_trailing_brackets():
    """glm-style output with extra ]} after the root object."""
    raw = '{"operations":[{"op":"append_block","section_id":"members","text":"- knip ignore react-dom"}]}]}'
    op_list = parse_delta_operation_list(raw)
    assert len(op_list.operations) == 1
    assert isinstance(op_list.operations[0], AppendBlockOp)


def test_parse_delta_operation_list_backticks_in_path():
    raw = (
        '{"operations":[{"op":"append_block","section_id":"conventions","text":"- hindsight-control-plane/knip.json"}]}'
    )
    op_list = parse_delta_operation_list(raw)
    assert len(op_list.operations) == 1
    op = op_list.operations[0]
    assert isinstance(op, AppendBlockOp)
    assert op.section_id == "conventions"
    assert op.text == "- hindsight-control-plane/knip.json"


def test_parse_delta_operation_list_prose_prefix():
    raw = 'Here is the update:\n{"operations": [{"op": "append_block", "section_id": "x", "text": "ok"}]}\nDone.'
    op_list = parse_delta_operation_list(raw)
    assert len(op_list.operations) == 1


def test_parse_delta_operation_list_multiline_block_text():
    """A table arrives as one string with escaped newlines and must stay multi-line."""
    raw = '{"operations": [{"op": "append_block", "section_id": "s", "text": "| a | b |\\n| --- | --- |\\n| 1 | 2 |"}]}'
    op = parse_delta_operation_list(raw).operations[0]
    assert isinstance(op, AppendBlockOp)
    assert op.text.count("\n") == 2


def test_parse_delta_operation_list_raw_newline_in_block_text():
    """A model that forgets to escape its line breaks still yields a real table (#3361).

    The control-character retry in ``parse_llm_json`` used to replace the raw
    newline with a space, delivering a table already welded onto one line.
    """
    raw = '{"operations": [{"op": "append_block", "section_id": "s", "text": "| a | b |\n| --- | --- |\n| 1 | 2 |"}]}'
    op = parse_delta_operation_list(raw).operations[0]
    assert isinstance(op, AppendBlockOp)
    assert op.text.splitlines() == ["| a | b |", "| --- | --- |", "| 1 | 2 |"]


def test_parse_delta_operation_list_skips_invalid_op_keeps_valid():
    """One bad replace_block (missing block_id) must not discard the whole batch."""
    raw = (
        '{"operations": ['
        '{"op": "append_block", "section_id": "s", "text": "ok"}, '
        '{"op": "replace_block", "section_id": "s", "text": "missing block_id"}, '
        '{"op": "append_block", "section_id": "s", "text": "also ok"}'
        "]}"
    )
    op_list = parse_delta_operation_list(raw)
    assert len(op_list.operations) == 2
    assert all(isinstance(o, AppendBlockOp) for o in op_list.operations)


def test_parse_delta_operation_list_rejects_v1_block_payloads():
    """The v1 typed-block shape is no longer a valid operation."""
    raw = (
        '{"operations": [{"op": "append_block", "section_id": "s", '
        '"block": {"type": "paragraph", "text": "old shape"}}]}'
    )
    with pytest.raises(DeltaAllOpsInvalidError):
        parse_delta_operation_list(raw)


def test_add_section_accepts_id_bearing_blocks():
    """A model that gives its new blocks ids must still land them (#3901).

    Every block in the document it was shown carries an id, so it emits ids for
    the blocks it creates. The id is meaningless to us — ``apply_operations``
    mints its own — but rejecting the op costs a whole refresh.
    """
    raw = (
        '{"operations": [{"op": "add_section", "heading": "Tools", "blocks": ['
        '{"id": "b12a001", "text": "First paragraph."}, '
        '{"id": "b12a002", "text": "Second paragraph."}'
        "]}]}"
    )
    op = parse_delta_operation_list(raw).operations[0]
    assert isinstance(op, AddSectionOp)
    assert op.blocks == ["First paragraph.", "Second paragraph."]


def test_replace_section_blocks_accepts_id_bearing_blocks():
    """The other op carrying ``blocks`` has the same exposure and the same fix."""
    raw = (
        '{"operations": [{"op": "replace_section_blocks", "section_id": "members", '
        '"blocks": [{"id": "b1", "text": "- Only Alice now."}]}]}'
    )
    op = parse_delta_operation_list(raw).operations[0]
    assert isinstance(op, ReplaceSectionBlocksOp)
    assert op.blocks == ["- Only Alice now."]


def test_blocks_coercion_accepts_a_mix_of_both_spellings():
    """One op may carry both shapes; neither spelling disturbs the other."""
    raw = (
        '{"operations": [{"op": "add_section", "heading": "Tools", '
        '"blocks": ["plain string", {"id": "b1", "text": "object form"}]}]}'
    )
    op = parse_delta_operation_list(raw).operations[0]
    assert isinstance(op, AddSectionOp)
    assert op.blocks == ["plain string", "object form"]


def test_blocks_coercion_ignores_a_model_supplied_id():
    """The id is dropped, not honoured: ids for new blocks are minted by the
    applier against the ids already in the document, so accepting the model's
    would reintroduce the collisions that scheme prevents."""
    doc = StructuredDocument(
        sections=[Section(id="members", heading="Members", level=2, blocks=[Block(id="b1", text="- Alice")])]
    )
    raw = '{"operations": [{"op": "add_section", "heading": "Tools", "blocks": [{"id": "b1", "text": "- Linear"}]}]}'
    outcome = apply_operations(doc, parse_delta_operation_list(raw).operations)
    assert len(outcome.applied) == 1
    new_block = outcome.document.section_by_id("tools").blocks[0]
    assert new_block.text == "- Linear"
    assert new_block.id != "b1"


def test_blocks_coercion_leaves_unrecognised_entries_to_fail_validation():
    """An object with no ``text`` is not a block we can read. It must fail with
    its own error rather than be silently dropped from the section."""
    raw = '{"operations": [{"op": "add_section", "heading": "Tools", "blocks": [{"id": "b1", "kind": "paragraph"}]}]}'
    with pytest.raises(DeltaAllOpsInvalidError):
        parse_delta_operation_list(raw)


def test_blocks_coercion_does_not_touch_non_block_text_fields():
    """The coercion is scoped to ``blocks`` lists; ``text`` is untouched, so the
    v1 typed-block payload stays invalid."""
    raw = '{"operations": [{"op": "append_block", "section_id": "s", "text": {"id": "b1", "text": "nope"}}]}'
    with pytest.raises(DeltaAllOpsInvalidError):
        parse_delta_operation_list(raw)


def test_parse_delta_operation_list_empty():
    assert parse_delta_operation_list("").operations == []


def test_parse_delta_operation_list_empty_operations_is_noop():
    """A genuine empty operations array is a valid no-op, not an error."""
    assert parse_delta_operation_list('{"operations": []}').operations == []
    assert parse_delta_operation_list({"operations": []}).operations == []


def test_parse_delta_operation_list_all_invalid_raises():
    """If the model emits ops but every one is malformed, raise so the caller
    refuses the refresh instead of applying zero ops — which would record a
    clean refresh while silently dropping this refresh's new facts."""
    raw = (
        '{"operations": ['
        '{"op": "replace_block", "section_id": "s", "text": "missing block_id a"}, '
        '{"op": "replace_block", "section_id": "s", "text": "missing block_id b"}'
        "]}"
    )
    with pytest.raises(DeltaAllOpsInvalidError):
        parse_delta_operation_list(raw)
    # Same payload shape as a dict must behave identically.
    with pytest.raises(DeltaAllOpsInvalidError):
        parse_delta_operation_list({"operations": [{"op": "replace_block", "section_id": "s", "text": "no block_id"}]})


def test_parse_delta_operation_list_pydantic_instance():
    original = DeltaOperationList(operations=[AppendBlockOp(section_id="s", text="- a")])
    assert parse_delta_operation_list(original) is original


def test_parse_delta_operation_list_top_level_array():
    """A bare array of ops carries the same delta as the dict form (#3820)."""
    op_list = parse_delta_operation_list('[{"op": "append_block", "section_id": "x", "text": "hi"}]')
    assert len(op_list.operations) == 1
    op = op_list.operations[0]
    assert isinstance(op, AppendBlockOp)
    assert op.section_id == "x"
    assert op.text == "hi"


def test_parse_delta_operation_list_top_level_array_skips_invalid_op():
    """Ops in an array are validated one by one, exactly as they are inside the dict form."""
    raw = (
        '[{"op": "append_block", "section_id": "s", "text": "ok"}, '
        '{"op": "replace_block", "section_id": "s", "text": "missing block_id"}]'
    )
    op_list = parse_delta_operation_list(raw)
    assert len(op_list.operations) == 1
    assert isinstance(op_list.operations[0], AppendBlockOp)
    assert op_list.operations[0].text == "ok"


def test_parse_delta_operation_list_top_level_array_all_invalid_raises():
    """An array of v1 typed-block ops is still every-op-invalid, so it must still raise."""
    raw = '[{"op": "append_block", "section_id": "s", "block": {"type": "paragraph", "text": "old shape"}}]'
    with pytest.raises(DeltaAllOpsInvalidError):
        parse_delta_operation_list(raw)
