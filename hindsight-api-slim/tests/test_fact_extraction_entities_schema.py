"""Regression tests for the `entities` extraction contract (issue #2749).

The prompt's few-shot examples teach a flat string array, so the LLM-facing
schema must ask for strings too. Entities stay optional: an omitted field is
coerced to [], so requiring it would buy nothing.
"""

from unittest.mock import MagicMock

import pytest

from hindsight_api.engine.retain.fact_extraction import (
    ExtractedFact,
    ExtractedFactNoCausal,
    ExtractedFactVerbose,
    VerbatimExtractedFact,
    _build_extraction_prompt_and_schema,
)

LLM_FACT_MODELS = (ExtractedFact, ExtractedFactVerbose, ExtractedFactNoCausal, VerbatimExtractedFact)


def _baseline_config() -> MagicMock:
    config = MagicMock()
    config.entity_labels = None
    config.entities_allow_free_form = True
    config.retain_extraction_mode = "concise"
    config.retain_extract_causal_links = False
    config.retain_mission = None
    config.retain_custom_instructions = None
    config.llm_output_language = None
    return config


@pytest.mark.parametrize("model", LLM_FACT_MODELS)
def test_entities_is_an_optional_array_of_strings(model):
    schema = model.model_json_schema()

    assert schema["properties"]["entities"] == {
        "description": schema["properties"]["entities"]["description"],
        "items": {"type": "string"},
        "title": "Entities",
        "type": "array",
    }
    # Deliberately NOT in `required` — an omitted field is coerced to [] anyway,
    # and forcing it would make strict-schema providers reject otherwise-fine facts.
    assert "entities" not in schema["required"]


@pytest.mark.parametrize("model", LLM_FACT_MODELS)
def test_entities_defaults_to_empty_list_and_accepts_legacy_objects(model):
    fields = {name: "x" for name, f in model.model_fields.items() if f.is_required()}
    fields["fact_type"] = "world"

    assert model.model_validate({**fields, "entities": []}).entities == []
    # Older prompts taught {"text": ...} objects; keep accepting them.
    assert model.model_validate({**fields, "entities": [{"text": "Alice"}, "Bob"]}).entities == ["Alice", "Bob"]
    assert model.model_validate({**fields, "entities": None}).entities == []


def test_concise_prompt_demands_plain_string_entities():
    prompt, _ = _build_extraction_prompt_and_schema(_baseline_config())

    assert 'ALWAYS return "entities" as an array of plain strings' in prompt
    assert 'entities=["Alice", "Kubernetes", "CKA"]' in prompt


def test_labels_only_mode_keeps_entities_as_strings():
    config = _baseline_config()
    config.entities_allow_free_form = False
    config.entity_labels = [{"key": "topic", "type": "text"}]

    _, schema = _build_extraction_prompt_and_schema(config)

    entities_schema = schema.model_fields["facts"].annotation.__args__[0].model_json_schema()["properties"]["entities"]
    assert entities_schema["items"] == {"type": "string"}
