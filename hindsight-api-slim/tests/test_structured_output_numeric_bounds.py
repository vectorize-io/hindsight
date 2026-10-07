"""A numeric bound on a response model must not reach the provider's schema.

``Field(ge=1, le=6)`` serializes as ``minimum``/``maximum``. Bedrock Converse
validates a structured-output schema against an allowlist and rejects those with
``BedrockException - output_config.format.schema: For 'integer' type, properties
maximum, minimum are not supported``, which failed every mental-model refresh on a
deployment using Bedrock (#5275) — the same class as the ``maxItems`` (#2500) and
``number`` min/max (#1289) cases before it.

The bounds stay on the Pydantic model and still validate the parsed response; they
are only stripped from the schema we send.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, Field, ValidationError

from hindsight_api.engine.reflect.delta_ops import DeltaOperationList
from hindsight_api.engine.reflect.structured_doc import StructuredDocument
from hindsight_api.engine.structured_output import (
    has_tagged_union,
    provider_json_schema,
    strict_json_schema,
)

_BOUND_KEYWORDS = ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf")


class _BoundedChild(BaseModel):
    score: int = Field(default=1, ge=0, le=10)


class _Bounded(BaseModel):
    level: int = Field(default=2, ge=1, le=6)
    ratio: float = Field(default=0.5, gt=0.0, lt=1.0)
    step: int = Field(default=2, multiple_of=2)
    nested: list[_BoundedChild] = Field(default_factory=list)


def _keywords_in(schema: dict) -> list[str]:
    serialized = json.dumps(schema)
    return [keyword for keyword in _BOUND_KEYWORDS if f'"{keyword}"' in serialized]


class TestBoundsAreStripped:
    def test_provider_schema_carries_no_bound(self):
        assert _keywords_in(provider_json_schema(_Bounded)) == []

    def test_strict_schema_carries_no_bound(self):
        assert _keywords_in(strict_json_schema(_Bounded)) == []

    def test_nested_models_are_stripped_too(self):
        """The bound hides in ``$defs``, not in the top-level properties."""
        schema = provider_json_schema(_Bounded)
        assert "_BoundedChild" in json.dumps(schema["$defs"])
        assert _keywords_in(schema) == []

    def test_the_two_real_offenders_are_clean(self):
        """The models the Bedrock refresh actually failed on."""
        assert _keywords_in(provider_json_schema(DeltaOperationList)) == []
        assert _keywords_in(provider_json_schema(StructuredDocument)) == []

    def test_a_property_named_minimum_survives(self):
        """Property names are user data, not schema keywords."""

        class _HasMinimumField(BaseModel):
            minimum: int = 0

        assert "minimum" in provider_json_schema(_HasMinimumField)["properties"]

    def test_deliberate_grammar_keywords_are_kept(self):
        """``maxItems`` and ``pattern`` are emitted on purpose, under their own flags."""

        class _Constrained(BaseModel):
            items: list[str] = Field(default_factory=list, max_length=3)
            stamp: str = Field(default="", pattern=r"^\d{4}$")

        serialized = json.dumps(provider_json_schema(_Constrained))
        assert "maxItems" in serialized
        assert "pattern" in serialized


class TestValidationStillApplies:
    def test_model_still_rejects_an_out_of_range_value(self):
        """The bound left the schema, not the model: the response is still checked."""
        with pytest.raises(ValidationError):
            _Bounded(level=9)

    def test_stripping_does_not_look_like_a_union_rewrite(self):
        """``has_tagged_union`` routes Gemini; a bound must not flip it."""
        assert has_tagged_union(_Bounded) is False
        assert has_tagged_union(DeltaOperationList) is True
