from pydantic import BaseModel, Field

from hindsight_api.engine.retain.fact_extraction import FactExtractionResponse
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
