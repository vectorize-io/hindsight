"""Canonical JSON Schema serialization for OpenAI strict output."""

from functools import lru_cache
from typing import Any

from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic_core import core_schema


class UnionSafeSchemaGenerator(GenerateJsonSchema):
    """Render a discriminated union as ``anyOf`` instead of ``oneOf`` + ``discriminator``.

    Pydantic serializes a ``Field(discriminator=...)`` union as ``oneOf`` with a
    ``discriminator`` block. No provider we target accepts that: OpenAI's strict
    subset takes ``anyOf`` and rejects ``oneOf``, and the Gemini SDK rejects both
    keys outright (``Extra inputs are not permitted`` on ``Schema.oneOf`` /
    ``Schema.discriminator``) before a request is ever sent. That is why the
    mental-model delta call could not use structured output at all and hand-parsed
    its JSON instead — see ``parse_delta_operation_list``.

    ``anyOf`` costs nothing here: each variant carries a ``Literal`` ``op`` field,
    so the variants remain mutually exclusive and the discriminator block was only
    ever a routing hint. Models with no tagged union serialize byte-identically to
    the default generator (pinned by
    ``test_union_safe_schema_is_identical_for_models_without_unions``), so this is
    safe to use wherever a schema is serialized for a provider.
    """

    def tagged_union_schema(self, schema: core_schema.TaggedUnionSchema) -> JsonSchemaValue:
        json_schema = super().tagged_union_schema(schema)
        variants = json_schema.pop("oneOf", None)
        if variants is not None:
            json_schema["anyOf"] = variants
        json_schema.pop("discriminator", None)
        return json_schema


class OpenAIStrictSchemaGenerator(UnionSafeSchemaGenerator):
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


def provider_json_schema(response_format: type[BaseModel]) -> dict[str, Any]:
    """Serialize a response model for a provider that is not using the strict subset.

    Same output as ``model_json_schema()`` for every model without a discriminated
    union; for one that has a union, the union is rendered as ``anyOf`` so the
    schema is transportable (see ``UnionSafeSchemaGenerator``).
    """
    return response_format.model_json_schema(schema_generator=UnionSafeSchemaGenerator)


@lru_cache(maxsize=None)
def has_tagged_union(response_format: type[BaseModel]) -> bool:
    """Whether serializing this model rewrites a tagged union.

    Asked per LLM call by providers that hand the *model* to their SDK rather than
    a schema dict (Gemini), so they can keep that native path for every model it
    already handles and only fall back to a serialized schema for the unions it
    cannot accept. Cached because the answer is a property of the class.
    """
    return provider_json_schema(response_format) != response_format.model_json_schema()


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
