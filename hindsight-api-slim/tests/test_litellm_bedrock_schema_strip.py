"""Unit tests for LiteLLMLLM's Bedrock JSON-Schema constraint stripping.

Bedrock's Claude tool-use API rejects `minimum`/`maximum` (and the
`exclusive*` variants) on number/integer schema nodes, but Pydantic's
`Field(ge=..., le=...)` emits them. The provider strips these for the
`bedrock` provider only; for every other provider the schema is sent
through unchanged.
"""

from __future__ import annotations

import copy
from typing import Any

from hindsight_api.engine.providers.litellm_llm import (
    _NUMERIC_CONSTRAINT_KEYS,
    _strip_numeric_constraints,
)


def _has_numeric_constraints(node: Any) -> bool:
    """True if `node` (anywhere recursively) contains any numeric-constraint key."""
    if isinstance(node, dict):
        if any(k in node for k in _NUMERIC_CONSTRAINT_KEYS):
            return True
        return any(_has_numeric_constraints(v) for v in node.values())
    if isinstance(node, list):
        return any(_has_numeric_constraints(v) for v in node)
    return False


def test_strip_numeric_constraints_removes_top_level_keys():
    schema = {
        "type": "integer",
        "minimum": 1,
        "maximum": 5,
        "description": "score",
    }
    _strip_numeric_constraints(schema)
    assert "minimum" not in schema
    assert "maximum" not in schema
    assert schema["type"] == "integer"
    assert schema["description"] == "score"


def test_strip_numeric_constraints_removes_exclusive_variants():
    schema = {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1}
    _strip_numeric_constraints(schema)
    assert "exclusiveMinimum" not in schema
    assert "exclusiveMaximum" not in schema


def test_strip_numeric_constraints_recurses_into_nested_properties():
    schema = {
        "type": "object",
        "properties": {
            "skepticism": {"type": "integer", "minimum": 1, "maximum": 5},
            "literalism": {"type": "integer", "minimum": 1, "maximum": 5},
            "name": {"type": "string"},
        },
        "required": ["skepticism"],
    }
    _strip_numeric_constraints(schema)
    assert not _has_numeric_constraints(schema)
    # untouched siblings preserved
    assert schema["properties"]["skepticism"]["type"] == "integer"
    assert schema["properties"]["name"] == {"type": "string"}
    assert schema["required"] == ["skepticism"]


def test_strip_numeric_constraints_recurses_into_arrays():
    schema = {
        "type": "array",
        "items": {"type": "integer", "minimum": 0, "maximum": 100},
    }
    _strip_numeric_constraints(schema)
    assert not _has_numeric_constraints(schema)
    assert schema["items"]["type"] == "integer"


def test_strip_numeric_constraints_handles_anyof_oneof():
    schema = {
        "anyOf": [
            {"type": "integer", "minimum": 1},
            {"type": "string"},
        ],
    }
    _strip_numeric_constraints(schema)
    assert not _has_numeric_constraints(schema)


def test_strip_numeric_constraints_returns_node_for_chaining():
    schema = {"type": "integer", "minimum": 0}
    result = _strip_numeric_constraints(schema)
    assert result is schema


def test_strip_numeric_constraints_noop_on_clean_schema():
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    }
    before = copy.deepcopy(schema)
    _strip_numeric_constraints(schema)
    assert schema == before


def test_strip_numeric_constraints_noop_on_non_dict_non_list():
    # Should not raise for primitives
    assert _strip_numeric_constraints("string") == "string"
    assert _strip_numeric_constraints(42) == 42
    assert _strip_numeric_constraints(None) is None


def test_strip_numeric_constraints_pydantic_field_constraints():
    """Mirror the Pydantic-emitted shape that triggers the Bedrock rejection."""
    from pydantic import BaseModel, Field

    class Disposition(BaseModel):
        skepticism: int = Field(ge=1, le=5)
        literalism: int = Field(ge=1, le=5)
        empathy: int = Field(ge=1, le=5)

    schema = Disposition.model_json_schema()
    # Pydantic emits the constraints by default
    assert _has_numeric_constraints(schema)
    _strip_numeric_constraints(schema)
    assert not _has_numeric_constraints(schema)
