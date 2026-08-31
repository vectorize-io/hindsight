"""Canonical JSON Schema serialization for OpenAI strict output."""

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


# Keywords whose value is a map of *names* to subschemas, not a subschema itself.
# Their keys are user-controlled, so a key literally called "$ref" must not be
# mistaken for a reference node while walking the tree.
_SUBSCHEMA_NAME_MAPS = frozenset({"$defs", "definitions", "properties", "patternProperties"})


def _strip_ref_siblings(node: Any) -> Any:
    """Drop every sibling keyword of a ``$ref``.

    Pydantic emits ``{"$ref": ..., "description": ...}`` for a field that both
    references a nested model and carries its own ``Field(description=...)``
    (JSON Schema 2020-12 allows that; the dynamic ``labels`` field built for
    entity-label banks hits it). OpenAI-compatible strict mode rejects it with
    ``$ref cannot have keywords {'description'}``, so a reference node keeps
    nothing but the reference.
    """
    if isinstance(node, dict):
        if "$ref" in node:
            return {"$ref": node["$ref"]}
        return {
            key: (
                {name: _strip_ref_siblings(sub) for name, sub in value.items()}
                if key in _SUBSCHEMA_NAME_MAPS and isinstance(value, dict)
                else _strip_ref_siblings(value)
            )
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [_strip_ref_siblings(item) for item in node]
    return node


def strict_json_schema(response_format: type[BaseModel]) -> dict[str, Any]:
    """Serialize a typed response model directly into OpenAI's strict subset."""
    schema = response_format.model_json_schema(schema_generator=OpenAIStrictSchemaGenerator)
    return _strip_ref_siblings(schema)


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
