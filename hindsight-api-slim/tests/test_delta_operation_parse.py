"""Tests for structured-delta LLM JSON parsing."""

from __future__ import annotations

import pytest

from hindsight_api.engine.reflect.delta_ops import (
    AppendBlockOp,
    DeltaAllOpsInvalidError,
    DeltaOperationList,
    parse_delta_operation_list,
)


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


def test_parse_delta_operation_list_empty():
    assert parse_delta_operation_list("").operations == []


def test_parse_delta_operation_list_empty_operations_is_noop():
    """A genuine empty operations array is a valid no-op, not an error."""
    assert parse_delta_operation_list('{"operations": []}').operations == []
    assert parse_delta_operation_list({"operations": []}).operations == []


def test_parse_delta_operation_list_all_invalid_raises():
    """If the model emits ops but every one is malformed, raise so the caller
    falls back to a full rewrite instead of applying zero ops — which would
    silently drop this refresh's new facts."""
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
