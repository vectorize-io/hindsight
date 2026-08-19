"""Retain fact-extraction schemas are bounded before the strict json_schema call.

A strict json_schema caps *shape* but not *size*: the ``facts`` array and its
nested string/array fields are unbounded, so a grammar-constrained backend can
generate valid JSON until it hits ``max_completion_tokens`` and then truncate
mid-object into invalid JSON. Live replays of historical pathological retain
prompts hung/failed with the unbounded schema but succeeded once the same schema
was bounded, so retain applies explicit output-size caps.

These tests pin the helper contract (RED before the helper existed) and prove it
is wired into *both* the interactive and batch retain paths — and only there.
"""

from copy import deepcopy
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from hindsight_api.engine.retain.fact_extraction import (
    FactExtractionResponse,
    FactExtractionResponseNoCausal,
    FactExtractionResponseVerbose,
    _build_extraction_prompt_and_schema,
    _build_request_body,
)
from hindsight_api.engine.structured_output import (
    RETAIN_FACTS_MAX_ITEMS,
    RETAIN_NESTED_ARRAY_MAX_ITEMS,
    RETAIN_STRING_MAX_LENGTH,
    bound_extraction_schema,
    strict_json_schema,
)


def _iter_nodes(node):
    """Yield every dict subschema in a JSON schema tree."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _iter_nodes(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_nodes(item)


def _string_nodes(schema):
    return [n for n in _iter_nodes(schema) if n.get("type") == "string" and "enum" not in n and "const" not in n]


def _array_nodes(schema):
    return [n for n in _iter_nodes(schema) if n.get("type") == "array"]


# --- 1. facts array gets maxItems 8 ----------------------------------------


def test_facts_array_gets_max_items_bound():
    schema = strict_json_schema(FactExtractionResponseVerbose)
    assert "maxItems" not in schema["properties"]["facts"]  # RED baseline

    bounded = bound_extraction_schema(schema)
    assert bounded["properties"]["facts"]["maxItems"] == RETAIN_FACTS_MAX_ITEMS
    assert RETAIN_FACTS_MAX_ITEMS == 8


# --- 2. unbounded strings get maxLength 2048 -------------------------------


def test_unbounded_strings_get_max_length():
    schema = strict_json_schema(FactExtractionResponseVerbose)
    bounded = bound_extraction_schema(schema)

    string_nodes = _string_nodes(bounded)
    assert string_nodes, "expected string fields in the fact schema"
    assert all(node["maxLength"] == RETAIN_STRING_MAX_LENGTH for node in string_nodes)
    assert RETAIN_STRING_MAX_LENGTH == 2048


# --- 3. unbounded nested arrays get maxItems 12 ----------------------------


def test_unbounded_nested_arrays_get_max_items():
    schema = strict_json_schema(FactExtractionResponseVerbose)
    bounded = bound_extraction_schema(schema)

    # entities (list[str]) is a nested array without a tighter bound.
    entities = bounded["$defs"]["ExtractedFactVerbose"]["properties"]["entities"]
    assert entities["type"] == "array"
    assert entities["maxItems"] == RETAIN_NESTED_ARRAY_MAX_ITEMS
    assert RETAIN_NESTED_ARRAY_MAX_ITEMS == 12

    # Every array except the top-level facts array is capped at the nested limit.
    for node in _array_nodes(bounded):
        assert node["maxItems"] <= RETAIN_NESTED_ARRAY_MAX_ITEMS


# --- 4. tighter existing bounds are preserved ------------------------------


def test_existing_tighter_bounds_are_preserved():
    schema = {
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "maxItems": 3,  # tighter than RETAIN_FACTS_MAX_ITEMS
                "items": {
                    "type": "object",
                    "properties": {
                        "short": {"type": "string", "maxLength": 16},  # tighter
                        "tags": {"type": "array", "maxItems": 2, "items": {"type": "string"}},  # tighter
                    },
                },
            },
        },
    }

    bounded = bound_extraction_schema(schema)
    props = bounded["properties"]["facts"]["items"]["properties"]

    assert bounded["properties"]["facts"]["maxItems"] == 3
    assert props["short"]["maxLength"] == 16
    assert props["tags"]["maxItems"] == 2
    # Looser/absent bounds are still clamped.
    assert props["tags"]["items"]["maxLength"] == RETAIN_STRING_MAX_LENGTH


def test_looser_existing_bounds_are_tightened():
    schema = {
        "type": "object",
        "properties": {
            "facts": {"type": "array", "maxItems": 999, "items": {"type": "string", "maxLength": 100_000}},
        },
    }

    bounded = bound_extraction_schema(schema)
    assert bounded["properties"]["facts"]["maxItems"] == RETAIN_FACTS_MAX_ITEMS
    assert bounded["properties"]["facts"]["items"]["maxLength"] == RETAIN_STRING_MAX_LENGTH


# --- 5. original input schema is not mutated -------------------------------


def test_input_schema_is_not_mutated():
    schema = strict_json_schema(FactExtractionResponseVerbose)
    snapshot = deepcopy(schema)

    bound_extraction_schema(schema)

    assert schema == snapshot, "bound_extraction_schema must not mutate its input"
    # And the shared Pydantic model schema stays unbounded.
    assert "maxItems" not in FactExtractionResponseVerbose.model_json_schema()["properties"]["facts"]


# --- 6. interactive AND batch paths use the bounded schema ------------------


def _baseline_config() -> MagicMock:
    config = MagicMock()
    config.entity_labels = None
    config.entities_allow_free_form = True
    config.retain_extraction_mode = "concise"
    config.retain_extract_causal_links = True
    config.retain_mission = None
    config.retain_custom_instructions = None
    config.llm_output_language = None
    config.llm_temperature_retain = 0.0
    config.retain_max_completion_tokens = 8192
    config.llm_strict_schema = True
    config.llm_strict_schema_retain = True
    return config


def _mock_llm_config() -> MagicMock:
    llm_config = MagicMock()
    llm_config.model = "test-model"
    llm_config.provider = "openai_compatible"
    llm_config._provider_impl.openai_service_tier = None
    return llm_config


def test_interactive_response_schema_is_bounded():
    _, response_schema = _build_extraction_prompt_and_schema(_baseline_config())

    strict = strict_json_schema(response_schema)
    assert strict["properties"]["facts"]["maxItems"] == RETAIN_FACTS_MAX_ITEMS
    plain = response_schema.model_json_schema()
    assert plain["properties"]["facts"]["maxItems"] == RETAIN_FACTS_MAX_ITEMS


def test_batch_request_body_uses_bounded_schema():
    config = _baseline_config()
    _, response_schema = _build_extraction_prompt_and_schema(config)

    body = _build_request_body(_mock_llm_config(), config, "system prompt", "user message", response_schema)
    schema = body["response_format"]["json_schema"]["schema"]

    assert schema["properties"]["facts"]["maxItems"] == RETAIN_FACTS_MAX_ITEMS
    assert all(node["maxLength"] == RETAIN_STRING_MAX_LENGTH for node in _string_nodes(schema))


@pytest.mark.parametrize(
    "mode,causal",
    [
        ("concise", True),
        ("verbose", True),
        ("concise", False),
        ("custom", True),
    ],
)
def test_all_extraction_modes_produce_bounded_facts(mode, causal):
    config = _baseline_config()
    config.retain_extraction_mode = mode
    config.retain_extract_causal_links = causal
    if mode == "custom":
        config.retain_custom_instructions = "Extract everything relevant."

    _, response_schema = _build_extraction_prompt_and_schema(config)
    schema = strict_json_schema(response_schema)
    assert schema["properties"]["facts"]["maxItems"] == RETAIN_FACTS_MAX_ITEMS


def test_dynamic_labels_schema_is_bounded():
    config = _baseline_config()
    config.entities_allow_free_form = False
    config.entity_labels = [{"key": "topic", "type": "text"}]

    _, response_schema = _build_extraction_prompt_and_schema(config)
    schema = strict_json_schema(response_schema)
    assert schema["properties"]["facts"]["maxItems"] == RETAIN_FACTS_MAX_ITEMS


# --- 7. unrelated structured-output schemas are unaffected ------------------


class _UnrelatedResponse(BaseModel):
    summary: str = Field(description="A free-form summary")
    tags: list[str] = Field(default_factory=list)


def test_unrelated_strict_schema_is_not_bounded():
    schema = strict_json_schema(_UnrelatedResponse)
    for node in _iter_nodes(schema):
        assert "maxLength" not in node
        assert "maxItems" not in node


def test_base_retain_models_are_not_mutated_globally():
    # Bounding is applied per-request on a copy; the shared model schemas must
    # remain unbounded so non-retain consumers see the original contract.
    for model in (FactExtractionResponse, FactExtractionResponseVerbose, FactExtractionResponseNoCausal):
        assert "maxItems" not in model.model_json_schema()["properties"]["facts"]
