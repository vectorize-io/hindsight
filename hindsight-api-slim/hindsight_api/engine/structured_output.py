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


def strict_json_schema(response_format: type[BaseModel]) -> dict[str, Any]:
    """Serialize a typed response model directly into OpenAI's strict subset."""
    return response_format.model_json_schema(schema_generator=OpenAIStrictSchemaGenerator)
