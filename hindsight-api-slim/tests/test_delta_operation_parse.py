"""Tests for structured-delta LLM JSON parsing."""

from __future__ import annotations

import pytest

from hindsight_api.engine.reflect.delta_ops import (
    AppendBlockOp,
    DeltaAllOpsInvalidError,
    DeltaOperationList,
    parse_delta_operation_list,
)
from hindsight_api.engine.reflect.structured_doc import BulletListBlock


def test_parse_delta_operation_list_trailing_brackets():
    """glm-style output with extra ]} after the root object."""
    raw = (
        '{"operations":[{"op":"append_block","section_id":"members",'
        '"block":{"type":"bullet_list","items":["knip ignore react-dom"]}}]}]}'
    )
    op_list = parse_delta_operation_list(raw)
    assert len(op_list.operations) == 1
    assert isinstance(op_list.operations[0], AppendBlockOp)


def test_parse_delta_operation_list_backticks_in_path():
    raw = (
        '{"operations":[{"op":"append_block","section_id":"conventions",'
        '"block":{"type":"bullet_list","items":["hindsight-control-plane/knip.json"]}}]}'
    )
    op_list = parse_delta_operation_list(raw)
    assert len(op_list.operations) == 1
    op = op_list.operations[0]
    assert op.section_id == "conventions"
    assert op.block.items == ["hindsight-control-plane/knip.json"]


def test_parse_delta_operation_list_prose_prefix():
    raw = (
        'Here is the update:\n{"operations": [{"op": "append_block", '
        '"section_id": "x", "block": {"type": "paragraph", "text": "ok"}}]}'
        "\nDone."
    )
    op_list = parse_delta_operation_list(raw)
    assert len(op_list.operations) == 1


def test_parse_delta_operation_list_skips_invalid_op_keeps_valid():
    """One bad replace_block (missing index) must not discard the whole batch."""
    raw = (
        '{"operations": ['
        '{"op": "append_block", "section_id": "s", '
        '"block": {"type": "paragraph", "text": "ok"}}, '
        '{"op": "replace_block", "section_id": "s", '
        '"block": {"type": "paragraph", "text": "missing index"}}, '
        '{"op": "append_block", "section_id": "s", '
        '"block": {"type": "paragraph", "text": "also ok"}}'
        "]}"
    )
    op_list = parse_delta_operation_list(raw)
    assert len(op_list.operations) == 2
    assert all(isinstance(o, AppendBlockOp) for o in op_list.operations)


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
        '{"op": "replace_block", "section_id": "s", '
        '"block": {"type": "paragraph", "text": "missing index a"}}, '
        '{"op": "replace_block", "section_id": "s", '
        '"block": {"type": "paragraph", "text": "missing index b"}}'
        "]}"
    )
    with pytest.raises(DeltaAllOpsInvalidError):
        parse_delta_operation_list(raw)
    # Same payload shape as a dict must behave identically.
    with pytest.raises(DeltaAllOpsInvalidError):
        parse_delta_operation_list(
            {
                "operations": [
                    {"op": "replace_block", "section_id": "s", "block": {"type": "paragraph", "text": "no index"}},
                ]
            }
        )


def test_parse_delta_operation_list_pydantic_instance():
    original = DeltaOperationList(
        operations=[
            AppendBlockOp(
                section_id="s",
                block=BulletListBlock(items=["a"]),
            )
        ]
    )
    assert parse_delta_operation_list(original) is original
