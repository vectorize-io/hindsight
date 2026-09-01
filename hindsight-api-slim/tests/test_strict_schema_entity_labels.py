"""Retain with entity labels under HINDSIGHT_API_LLM_STRICT_SCHEMA (#3904).

The dynamic ``LabelsFact`` model gives its ``labels`` field both a nested model
and a ``Field(description=...)``, which pydantic serializes as
``{"$ref": ..., "description": ...}``. That is legal JSON Schema 2020-12 but
illegal in OpenAI's strict subset, which rejects the request outright with
``$ref cannot have keywords {'description'}`` — so retain failed before
inference for every labelled bank running with strict schema on.

Marked ``hs_llm_mat``, not ``hs_llm_core``: the core job runs
vertexai/gemini-2.5-flash-lite, and the Gemini provider grammar-enforces its own
native ``response_schema`` without ever calling ``strict_json_schema()``. A core
run would pass no matter what the serializer emits — which is precisely why the
existing entity-label LLM tests never caught this. The matrix job includes
openai/gpt-4.1-nano and litellmrouter, whose validators are the ones that reject
the sibling keyword.
"""

import uuid

import pytest

from hindsight_api.config import ENV_LLM_STRICT_SCHEMA_RETAIN, clear_config_cache

pytestmark = pytest.mark.hs_llm_mat


@pytest.fixture
def strict_retain_schema(monkeypatch):
    """Force retain onto the strict-schema path (static config, so env + cache reset)."""
    monkeypatch.setenv(ENV_LLM_STRICT_SCHEMA_RETAIN, "true")
    clear_config_cache()
    yield
    monkeypatch.undo()
    clear_config_cache()


async def test_retain_with_entity_labels_under_strict_schema(
    # strict_retain_schema comes FIRST on purpose: ConfigResolver snapshots
    # _get_raw_config() in its constructor, so the env has to be set before
    # memory_real_llm builds the engine, or retain silently runs non-strict.
    strict_retain_schema,
    memory_real_llm,
    request_context,
):
    """A labelled bank retains successfully when the provider enforces the schema.

    The assertion is that the call completes and extracts facts: an invalid
    strict schema is rejected by the provider before inference, so the failure
    mode this covers is an HTTP 400 (surfaced as "Fact extraction failed"), not
    a wrong label. Label assignment quality is covered by the hs_llm_core tests
    in test_entity_labels.py.
    """
    memory = memory_real_llm
    # Guard against the test going vacuous: without this, a fixture-ordering or
    # config-caching regression would leave retain on the non-strict path, where
    # the schema is never serialized into the strict subset and nothing is checked.
    assert memory._config_resolver._global_config.llm_strict_schema_retain is True

    bank_id = f"test-labels-strict-{uuid.uuid4().hex[:8]}"
    try:
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
        await memory._config_resolver.update_bank_config(
            bank_id=bank_id,
            updates={
                "entity_labels": [
                    {
                        "key": "engagement",
                        "description": "Student engagement level during the session",
                        "values": [
                            {"value": "active", "description": "Student is actively participating"},
                            {"value": "passive", "description": "Student is listening but not participating"},
                        ],
                    }
                ]
            },
            context=request_context,
        )

        unit_ids = await memory.retain_async(
            bank_id=bank_id,
            content=(
                "During today's tutoring session, Maria asked many questions, "
                "participated in every exercise, and solved the problems independently."
            ),
            request_context=request_context,
        )

        assert len(unit_ids) > 0, "Retain under a strict schema should still extract facts"
    finally:
        await memory.delete_bank(bank_id=bank_id, request_context=request_context)
