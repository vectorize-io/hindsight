"""Canonical JSON Schema serialization for OpenAI strict output."""

import copy
from typing import Any

from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic_core import core_schema


class OpenAIStrictSchemaGenerator(GenerateJsonSchema):
    """Emit the strict JSON Schema subset required by OpenAI-compatible APIs."""

    def model_schema(self, schema: core_schema.ModelSchema) -> JsonSchemaValue:
        json_schema = super().model_schema(schema)
        properties = json_schema.get("properties")
        if type(properties) is dict:
            json_schema["required"] = list(properties)
            json_schema["additionalProperties"] = False
        return json_schema

    def default_schema(self, schema: core_schema.WithDefaultSchema) -> JsonSchemaValue:
        json_schema = super().default_schema(schema)
        if json_schema.get("default", object()) is None:
            json_schema.pop("default")
        return json_schema


def strict_json_schema(response_format: type[BaseModel]) -> dict[str, Any]:
    """Serialize a typed response model directly into OpenAI's strict subset."""
    return response_format.model_json_schema(schema_generator=OpenAIStrictSchemaGenerator)


# --- Bounded schema for retain fact extraction -----------------------------
#
# A strict json_schema alone does not cap *how much* a model may emit: the facts
# array and its nested string/array fields are unbounded, so on a pathological
# prompt a grammar-constrained backend (e.g. vLLM) will happily generate valid
# JSON until it hits max_completion_tokens and then truncates mid-object,
# yielding invalid JSON and a wasted, often multi-minute, call. Live replays of
# historical pathological retain prompts hung/failed with the unbounded schema
# but succeeded 3/3 (valid JSON, finish_reason=stop) once the *same* schema was
# bounded. These caps give the grammar an escape hatch: a finite maximum output
# the model can always complete within the token budget.
#
# The numbers are deliberately generous — real extractions in that replay
# produced at most 5 facts / ~4.3k tokens, so the caps only ever bite runaway
# degenerate output, never legitimate extraction.
RETAIN_FACTS_MAX_ITEMS = 8
"""Upper bound on the top-level ``facts`` array for retain extraction."""

RETAIN_NESTED_ARRAY_MAX_ITEMS = 12
"""Upper bound applied to any other (nested) array without a tighter limit."""

RETAIN_STRING_MAX_LENGTH = 2048
"""Upper bound applied to any string without a tighter ``maxLength``."""


def _tighter_limit(existing: Any, cap: int) -> int:
    """Return the tighter of an existing integer limit and ``cap``.

    Preserves an already-present (presumably intentional, tighter) bound rather
    than loosening it, while still clamping absent or looser limits to ``cap``.
    """
    if isinstance(existing, int) and not isinstance(existing, bool):
        return min(existing, cap)
    return cap


def _bound_schema_node(node: Any) -> None:
    """Recursively clamp unbounded strings/arrays in-place on a copied schema."""
    if isinstance(node, dict):
        node_type = node.get("type")
        # Skip enum/const string nodes: their length is already fully
        # constrained by the allowed values, so a maxLength is noise.
        if node_type == "string" and "enum" not in node and "const" not in node:
            node["maxLength"] = _tighter_limit(node.get("maxLength"), RETAIN_STRING_MAX_LENGTH)
        elif node_type == "array":
            node["maxItems"] = _tighter_limit(node.get("maxItems"), RETAIN_NESTED_ARRAY_MAX_ITEMS)
        for value in node.values():
            _bound_schema_node(value)
    elif isinstance(node, list):
        for item in node:
            _bound_schema_node(item)


def bound_extraction_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a deep-copied retain-extraction schema with output size bounds.

    Adds ``maxItems`` to the top-level ``facts`` array (``RETAIN_FACTS_MAX_ITEMS``)
    and, recursively, ``maxLength``/``maxItems`` to every otherwise-unbounded
    string/array (``RETAIN_STRING_MAX_LENGTH`` / ``RETAIN_NESTED_ARRAY_MAX_ITEMS``).
    Existing tighter limits are preserved. The input is never mutated — Pydantic
    caches and shares the dicts returned by ``model_json_schema``, so we operate
    on a deep copy.
    """
    bounded = copy.deepcopy(schema)

    properties = bounded.get("properties")
    if isinstance(properties, dict):
        facts = properties.get("facts")
        if isinstance(facts, dict) and facts.get("type") == "array":
            facts["maxItems"] = _tighter_limit(facts.get("maxItems"), RETAIN_FACTS_MAX_ITEMS)

    _bound_schema_node(bounded)
    return bounded


# Types the structured-output extractor knows how to build a Pydantic field for
# (see reflect/agent.py::_generate_structured_output → _json_schema_type_to_python).
_RESPONSE_SCHEMA_PROPERTY_TYPES = frozenset({"string", "number", "integer", "boolean", "array", "object"})


def validate_response_schema(schema: Any) -> None:
    """Validate a user-supplied ``response_schema`` is usable for structured output.

    Structured output builds a flat Pydantic model from the schema's top-level
    ``properties`` (reflect and mental-model refresh both feed the schema through
    ``_generate_structured_output``). A schema with no ``properties`` silently
    produces an empty ``structured_output``, and malformed shapes only blow up
    later inside the LLM extraction call — so validate the contract at the API
    boundary and fail loudly with a ``ValueError`` (surfaced as HTTP 422) instead.

    Raises:
        ValueError: when ``schema`` is not an object schema with a non-empty,
            well-formed ``properties`` map.
    """
    if not isinstance(schema, dict):
        raise ValueError("response_schema must be a JSON object")

    schema_type = schema.get("type")
    if schema_type is not None and schema_type != "object":
        raise ValueError('response_schema must be an object schema (its "type" must be "object")')

    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        raise ValueError("response_schema must define a non-empty 'properties' object")

    for name, prop in properties.items():
        if not isinstance(prop, dict):
            raise ValueError(f"response_schema property '{name}' must be an object")
        prop_type = prop.get("type")
        if prop_type is not None and prop_type not in _RESPONSE_SCHEMA_PROPERTY_TYPES:
            raise ValueError(
                f"response_schema property '{name}' has unsupported type '{prop_type}'; "
                f"expected one of {sorted(_RESPONSE_SCHEMA_PROPERTY_TYPES)}"
            )

    required = schema.get("required")
    if required is not None:
        if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
            raise ValueError("response_schema 'required' must be a list of property names")
        unknown = [item for item in required if item not in properties]
        if unknown:
            raise ValueError(f"response_schema 'required' references unknown properties: {unknown}")
