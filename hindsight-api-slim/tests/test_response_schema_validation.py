"""Validation of user-supplied ``response_schema`` for structured output.

Structured output builds a flat Pydantic model from the schema's top-level
``properties``, so a schema without a usable ``properties`` map either silently
yields an empty ``structured_output`` or fails deep inside the LLM extraction
call. ``validate_response_schema`` enforces the usable-shape contract at the API
boundary; these are fast, deterministic unit tests (no DB, no LLM).
"""

import pytest
from pydantic import ValidationError

from hindsight_api.api.http import MentalModelTrigger, ReflectRequest
from hindsight_api.engine.structured_output import validate_response_schema

_VALID_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary"],
}


class TestValidateResponseSchema:
    def test_valid_schema_passes(self):
        validate_response_schema(_VALID_SCHEMA)  # must not raise

    def test_object_type_optional(self):
        # ``type`` may be omitted; properties alone are enough.
        validate_response_schema({"properties": {"a": {"type": "string"}}})

    @pytest.mark.parametrize(
        "schema, needle",
        [
            (["not", "a", "dict"], "must be a JSON object"),
            ("string", "must be a JSON object"),
            ({"type": "array", "properties": {"a": {"type": "string"}}}, "object schema"),
            ({"type": "object"}, "non-empty 'properties'"),
            ({"properties": {}}, "non-empty 'properties'"),
            ({"properties": {"a": "not-an-object"}}, "property 'a' must be an object"),
            ({"properties": {"a": {"type": "banana"}}}, "unsupported type 'banana'"),
            ({"properties": {"a": {"type": "string"}}, "required": "summary"}, "must be a list"),
            ({"properties": {"a": {"type": "string"}}, "required": ["b"]}, "unknown properties"),
        ],
    )
    def test_invalid_schema_raises(self, schema, needle):
        with pytest.raises(ValueError) as exc:
            validate_response_schema(schema)
        assert needle in str(exc.value)


class TestModelIntegration:
    def test_reflect_request_accepts_valid_schema(self):
        req = ReflectRequest(query="q", response_schema=_VALID_SCHEMA)
        assert req.response_schema == _VALID_SCHEMA

    def test_reflect_request_rejects_invalid_schema(self):
        with pytest.raises(ValidationError):
            ReflectRequest(query="q", response_schema={"type": "object"})

    def test_reflect_request_allows_none(self):
        assert ReflectRequest(query="q").response_schema is None

    def test_trigger_accepts_valid_schema(self):
        trigger = MentalModelTrigger(response_schema=_VALID_SCHEMA)
        assert trigger.response_schema == _VALID_SCHEMA

    def test_trigger_rejects_invalid_schema(self):
        with pytest.raises(ValidationError):
            MentalModelTrigger(response_schema={"properties": {}})
