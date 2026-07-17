"""Regression coverage for the fact-extraction entity output contract."""

import json
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from hindsight_api import LLMConfig
from hindsight_api.config import _get_raw_config
from hindsight_api.engine.retain.fact_extraction import (
    ExtractedFact,
    ExtractedFactNoCausal,
    ExtractedFactVerbose,
    VerbatimExtractedFact,
    _build_extraction_prompt_and_schema,
    extract_facts_from_text,
)
from tests.llm_judge import assert_meets_criteria


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


@pytest.mark.parametrize(
    "model",
    [ExtractedFact, ExtractedFactVerbose, ExtractedFactNoCausal, VerbatimExtractedFact],
)
def test_entities_are_required_object_arrays_with_empty_default(model):
    schema = model.model_json_schema()

    assert "entities" in schema["required"]
    assert schema["properties"]["entities"]["type"] == "array"

    values = {"when": "N/A", "where": "N/A", "who": "N/A", "fact_type": "world"}
    if model is not VerbatimExtractedFact:
        values.update({"what": "Alice works at Acme", "why": "N/A"})
    assert model(**values).entities == []


def test_concise_prompt_teaches_object_shaped_entities():
    prompt, _ = _build_extraction_prompt_and_schema(_baseline_config())

    assert 'entities=[{"text": "Alice"}, {"text": "Kubernetes"}, {"text": "CKA"}]' in prompt
    assert 'entities=["Alice", "Kubernetes", "CKA"]' not in prompt
    assert 'array of objects with a "text" field, never as an array of strings' in prompt


@pytest.mark.parametrize("free_form_entities, entities_required", [(True, True), (False, False)])
def test_labels_schema_matches_free_form_entity_setting(free_form_entities, entities_required):
    config = _baseline_config()
    config.entity_labels = [{"key": "topic", "values": [{"value": "math"}]}]
    config.entities_allow_free_form = free_form_entities

    _, schema = _build_extraction_prompt_and_schema(config)
    fact_schema = schema.model_json_schema()["$defs"]["LabelsFact"]

    assert ("entities" in fact_schema["required"]) is entities_required
    assert "labels" in fact_schema["required"]


@pytest.mark.hs_llm_core
@pytest.mark.asyncio
async def test_fact_extraction_returns_relevant_entities():
    facts, _, _ = await extract_facts_from_text(
        text="Alice leads the Kubernetes platform team at Acme Corporation in Berlin.",
        event_date=datetime(2026, 7, 16),
        llm_config=LLMConfig.from_env(),
        agent_name="assistant",
        context="employee profile",
        config=_get_raw_config(),
    )

    entity_summary = json.dumps([entity.text for fact in facts for entity in (fact.entities or [])], ensure_ascii=False)
    await assert_meets_criteria(
        response=entity_summary,
        criteria=(
            "The extracted entity list is non-empty and contains the named person Alice plus relevant named entities "
            "from the source, especially Acme Corporation and Kubernetes or Berlin."
        ),
        context="Source: Alice leads the Kubernetes platform team at Acme Corporation in Berlin.",
    )
