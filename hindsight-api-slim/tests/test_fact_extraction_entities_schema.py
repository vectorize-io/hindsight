"""Regression tests for issue #2749: auto-extracted entities were silently dropped.

The four extraction models declared ``entities`` as an optional field that was
absent from the schema's ``required`` array. Under a strict JSON schema, a
literal model resolves the conflict (optional-typed-as-objects, while the
few-shot examples showed strings) by omitting the field entirely — 0 entities,
no error, biting precisely the most schema-conformant models.

The fix makes ``entities`` required with an empty-list default and teaches the
prompts the object form. These tests assert the schema shape (the regression
that would have caught the bug in CI) and the prompt instruction, without
needing an LLM.
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

ALL_EXTRACTION_MODELS = [
    ExtractedFact,
    ExtractedFactVerbose,
    ExtractedFactNoCausal,
    VerbatimExtractedFact,
]


def _entity_def(schema: dict) -> dict:
    """Resolve the entities-item $ref to its object definition."""
    entities = schema["properties"]["entities"]
    assert entities["type"] == "array", "entities must be an array"
    ref = entities["items"]["$ref"].split("/")[-1]
    return schema["$defs"][ref]


@pytest.mark.parametrize("model", ALL_EXTRACTION_MODELS)
def test_entities_is_required_array_of_objects(model):
    """Each extraction model must list entities as required, typed as an array
    of objects with a ``text`` field — never bare strings. This is the schema
    regression behind #2749: an optional field is silently omitted by literal
    models, so all entities are dropped."""
    schema = model.model_json_schema()

    assert "entities" in schema["required"], (
        f"{model.__name__}: 'entities' must be in the schema's required array"
    )

    entity_def = _entity_def(schema)
    assert entity_def.get("type") == "object"
    assert "text" in entity_def.get("properties", {}), (
        f"{model.__name__}: entities items must be objects with a 'text' field"
    )


@pytest.mark.parametrize("model", ALL_EXTRACTION_MODELS)
def test_entities_defaults_to_empty_list_when_omitted(model):
    """Making entities required-in-schema must not break lenient parsing: a
    response that omits entities must still parse, defaulting to an empty list
    (not None)."""
    fields = {"when": "N/A", "where": "N/A", "who": "N/A", "fact_type": "world"}
    if model is not VerbatimExtractedFact:
        fields.update({"what": "something happened", "why": "N/A"})

    instance = model(**fields)

    assert instance.entities == []


def _config(mode: str) -> MagicMock:
    config = MagicMock()
    config.entity_labels = None
    config.entities_allow_free_form = True
    config.retain_extraction_mode = mode
    config.retain_extract_causal_links = False
    config.retain_mission = None
    config.retain_custom_instructions = None
    config.llm_output_language = None
    return config


@pytest.mark.parametrize("mode", ["concise", "verbose", "verbatim", "custom"])
def test_prompt_teaches_entities_object_form(mode):
    """Every extraction prompt must instruct that entities are an array of
    objects, so non-strict / no-schema paths stop emitting bare strings."""
    prompt, _ = _build_extraction_prompt_and_schema(_config(mode))

    assert "array of objects" in prompt
    assert 'entities=["' not in prompt, "prompt must not show bare-string entity examples"


def test_concise_examples_use_object_form():
    """The concise few-shot examples must demonstrate the object form."""
    prompt, _ = _build_extraction_prompt_and_schema(_config("concise"))

    assert '{"text": "user"}' in prompt
    assert '{"text": "Alice"}' in prompt
