from unittest.mock import MagicMock

from pydantic import BaseModel, Field

from hindsight_api.engine.retain.fact_extraction import (
    FactExtractionResponse,
    _build_extraction_prompt_and_schema,
)
from hindsight_api.engine.structured_output import strict_json_schema


class StrictSchemaChild(BaseModel):
    name: str


class StrictSchemaResponse(BaseModel):
    child: StrictSchemaChild
    note: str | None = None
    children: list[StrictSchemaChild] = Field(default_factory=list)


def test_strict_json_schema_closes_every_object_and_requires_every_property() -> None:
    schema = strict_json_schema(StrictSchemaResponse)

    assert schema["additionalProperties"] is False
    assert schema["required"] == ["child", "note", "children"]
    assert "default" not in schema["properties"]["note"]
    assert schema["$defs"]["StrictSchemaChild"]["additionalProperties"] is False
    assert schema["$defs"]["StrictSchemaChild"]["required"] == ["name"]


def test_fact_extraction_schema_is_accepted_by_openai_strict_output_contract() -> None:
    schema = strict_json_schema(FactExtractionResponse)
    extracted_fact = schema["$defs"]["ExtractedFact"]

    assert schema["additionalProperties"] is False
    assert schema["required"] == ["facts"]
    assert extracted_fact["additionalProperties"] is False
    assert extracted_fact["required"] == list(extracted_fact["properties"])


class StrictSchemaDescribedRef(BaseModel):
    child: StrictSchemaChild = Field(description="A described nested model.")


def _iter_ref_nodes(node: object):
    """Yield every dict in ``node`` that carries a ``$ref``."""
    if isinstance(node, dict):
        if "$ref" in node:
            yield node
        for value in node.values():
            yield from _iter_ref_nodes(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_ref_nodes(item)


def test_strict_json_schema_leaves_no_sibling_keywords_next_to_a_ref() -> None:
    # Pydantic emits {"$ref": ..., "description": ...} for a described field that
    # points at a nested model; OpenAI strict mode rejects it with
    # "$ref cannot have keywords {'description'}".
    schema = strict_json_schema(StrictSchemaDescribedRef)

    assert schema["properties"]["child"] == {"$ref": "#/$defs/StrictSchemaChild"}
    assert [set(ref) for ref in _iter_ref_nodes(schema)] == [{"$ref"}]


def test_entity_labels_extraction_schema_has_no_ref_siblings() -> None:
    config = MagicMock()
    config.entity_labels = [
        {"key": "knowledge", "type": "multi-values", "values": [{"value": "python"}, {"value": "rust"}]}
    ]
    config.entities_allow_free_form = True
    config.retain_extraction_mode = "concise"
    config.retain_extract_causal_links = False
    config.retain_mission = None
    config.retain_custom_instructions = None
    config.llm_output_language = None
    config.llm_supports_string_pattern = False

    _, response_schema = _build_extraction_prompt_and_schema(config)
    schema = strict_json_schema(response_schema)

    labels_property = schema["$defs"]["LabelsFact"]["properties"]["labels"]
    assert labels_property == {"$ref": "#/$defs/Labels"}
    refs = list(_iter_ref_nodes(schema))
    assert refs, "expected the labels schema to use $ref"
    assert all(set(ref) == {"$ref"} for ref in refs)
