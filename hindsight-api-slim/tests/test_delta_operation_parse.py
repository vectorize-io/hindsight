"""Tests for structured-delta LLM JSON parsing."""

from __future__ import annotations

from hindsight_api.engine.reflect.delta_ops import (
    AppendBlockOp,
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


def test_parse_delta_operation_list_empty():
    assert parse_delta_operation_list("").operations == []


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
