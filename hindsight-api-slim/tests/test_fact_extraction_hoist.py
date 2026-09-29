"""Tests for Fact Extraction prompt and schema hoisting optimization."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api.engine.response_models import TokenUsage
from hindsight_api.engine.retain import fact_extraction
from hindsight_api.engine.retain.fact_extraction import (
    ExtractionPrompt,
    Fact,
    FactExtractionResponse,
    OutputTooLongError,
    RetainContent,
    build_chunk_prompt_parts,
    extract_facts_from_contents,
    extract_facts_from_text,
)
from hindsight_api.engine.structured_output import strict_json_schema


def _minimal_config(**overrides):
    cfg = {
        "retain_extraction_mode": "concise",
        "retain_extract_causal_links": True,
        "retain_custom_instructions": None,
        "retain_mission": None,
        "entity_labels": None,
        "entities_allow_free_form": True,
        "llm_output_language": None,
        "llm_supports_string_pattern": False,
        "retain_optional_fact_dimensions": False,
        "retain_chunk_size": 50,
        "retain_structured_chunk_size": None,
        "retain_max_attachments_per_chunk": 8,
        "retain_batch_enabled": False,
    }
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


_SAMPLE_LABELS = {
    "attributes": [
        {"key": "sentiment", "type": "value", "optional": False, "values": [{"value": "pos"}, {"value": "neg"}]},
        {"key": "category", "type": "multi-values", "values": [{"value": "work"}, {"value": "personal"}]},
    ]
}


@pytest.mark.parametrize(
    "config_name,supports_pattern,entity_labels",
    [
        ("default", False, None),
        ("pattern", True, None),
        ("labels", False, _SAMPLE_LABELS),
        ("both", True, _SAMPLE_LABELS),
    ],
)
def test_hoisted_equivalence_across_all_configs(config_name: str, supports_pattern: bool, entity_labels: dict | None):
    """Verify that build_chunk_prompt_parts with prebuilt ExtractionPrompt produces byte-identical
    prompt text and exact same JSON Schema as unhoisted on-demand construction for all 4 configs."""
    cfg = _minimal_config(
        llm_supports_string_pattern=supports_pattern,
        entity_labels=entity_labels,
    )
    sample_chunk = "Alice and Bob visited the Munich office on Tuesday to review the quarterly roadmap."

    # 1. Unhoisted on-demand path (builds prompt & schema internally)
    unhoisted_parts = build_chunk_prompt_parts(
        cfg,
        chunk=sample_chunk,
        chunk_index=0,
        total_chunks=1,
    )

    # 2. Hoisted path (precomputes ExtractionPrompt once and passes it in)
    prebuilt = fact_extraction._build_extraction_prompt_and_schema(cfg)
    assert isinstance(prebuilt, ExtractionPrompt)
    hoisted_parts = build_chunk_prompt_parts(
        cfg,
        chunk=sample_chunk,
        chunk_index=0,
        total_chunks=1,
        extraction_prompt=prebuilt,
    )

    # 3. Exact byte-for-byte system prompt match
    assert hoisted_parts.system_prompt == unhoisted_parts.system_prompt
    # 4. User message match
    assert hoisted_parts.user_message == unhoisted_parts.user_message
    # 5. Exact JSON schema equivalence (via strict_json_schema)
    assert strict_json_schema(hoisted_parts.response_schema) == strict_json_schema(unhoisted_parts.response_schema)


@pytest.mark.asyncio
async def test_extract_facts_from_text_builds_prompt_and_schema_once():
    """Verify that _build_extraction_prompt_and_schema is called only once across multiple chunks."""
    config = _minimal_config(retain_chunk_size=30)
    long_text = (
        "Alice visited Paris in June 2024. "
        "Bob went to Tokyo in July 2024. "
        "Charlie travelled to London in August 2024. "
        "David explored Berlin in September 2024. "
    )

    call_count = 0
    original_builder = fact_extraction._build_extraction_prompt_and_schema

    def counting_builder(cfg):
        nonlocal call_count
        call_count += 1
        return original_builder(cfg)

    passed_prompts: list[ExtractionPrompt | None] = []

    async def mock_auto_split(**kwargs):
        passed_prompts.append(kwargs.get("extraction_prompt"))
        return [Fact(fact="some fact", fact_type="world")], TokenUsage()

    with (
        patch.object(fact_extraction, "_build_extraction_prompt_and_schema", side_effect=counting_builder),
        patch.object(fact_extraction, "_extract_facts_with_auto_split", side_effect=mock_auto_split),
    ):
        facts, chunks_meta, _ = await extract_facts_from_text(
            text=long_text,
            event_date=datetime.now(timezone.utc),
            llm_config=SimpleNamespace(),
            config=config,
        )

    # Must be sliced into multiple chunks
    assert len(chunks_meta) >= 3, f"Expected >= 3 chunks, got {len(chunks_meta)}"
    # Schema & prompt must only have been built ONCE
    assert call_count == 1, f"Expected _build_extraction_prompt_and_schema called 1 time, got {call_count}"
    # Every chunk task must have received the identical ExtractionPrompt object
    assert len(passed_prompts) == len(chunks_meta)
    assert all(p is not None and p is passed_prompts[0] for p in passed_prompts)
    assert isinstance(passed_prompts[0], ExtractionPrompt)


@pytest.mark.asyncio
async def test_extract_facts_from_contents_builds_prompt_and_schema_once():
    """Verify that extract_facts_from_contents builds schema once for all documents."""
    config = _minimal_config(retain_chunk_size=1000)
    contents = [
        RetainContent(content="Doc 1 content", event_date=datetime.now(timezone.utc)),
        RetainContent(content="Doc 2 content", event_date=datetime.now(timezone.utc)),
        RetainContent(content="Doc 3 content", event_date=datetime.now(timezone.utc)),
    ]

    call_count = 0
    original_builder = fact_extraction._build_extraction_prompt_and_schema

    def counting_builder(cfg):
        nonlocal call_count
        call_count += 1
        return original_builder(cfg)

    extracted_fact = Fact(fact="Test fact", fact_type="world")
    mock_extract = AsyncMock(return_value=([extracted_fact], [("chunk", 1)], TokenUsage()))

    with (
        patch.object(fact_extraction, "_build_extraction_prompt_and_schema", side_effect=counting_builder),
        patch.object(fact_extraction, "extract_facts_from_text", mock_extract),
    ):
        result = await extract_facts_from_contents(
            contents=contents,
            llm_config=SimpleNamespace(),
            config=config,
        )

    assert len(result.facts) == 3
    assert call_count == 1, f"Expected 1 call to _build_extraction_prompt_and_schema, got {call_count}"
    assert mock_extract.call_count == 3
    # Verify all calls to extract_facts_from_text received the precomputed ExtractionPrompt
    first_call_prompt = mock_extract.call_args_list[0].kwargs["extraction_prompt"]
    assert first_call_prompt is not None
    assert isinstance(first_call_prompt, ExtractionPrompt)
    for call in mock_extract.call_args_list:
        assert call.kwargs["extraction_prompt"] is first_call_prompt


def test_build_chunk_prompt_parts_reuses_passed_extraction_prompt():
    """Verify build_chunk_prompt_parts reuses extraction_prompt if provided."""
    config = _minimal_config()
    custom_prompt_obj = ExtractionPrompt(system_prompt="CUSTOM SYSTEM PROMPT", response_schema=FactExtractionResponse)

    with patch.object(fact_extraction, "_build_extraction_prompt_and_schema") as mock_builder:
        parts = build_chunk_prompt_parts(
            config,
            chunk="Sample chunk text",
            extraction_prompt=custom_prompt_obj,
        )
        assert mock_builder.call_count == 0
        assert parts.system_prompt == custom_prompt_obj.system_prompt
        assert parts.response_schema is custom_prompt_obj.response_schema

    # When omitted, it calls _build_extraction_prompt_and_schema
    with patch.object(
        fact_extraction,
        "_build_extraction_prompt_and_schema",
        return_value=ExtractionPrompt(system_prompt="built prompt", response_schema=FactExtractionResponse),
    ) as mock_builder:
        parts = build_chunk_prompt_parts(
            config,
            chunk="Sample chunk text",
        )
        assert mock_builder.call_count == 1
        assert parts.system_prompt == "built prompt"
        assert parts.response_schema is FactExtractionResponse


@pytest.mark.asyncio
async def test_auto_split_preserves_extraction_prompt():
    """Verify that recursive splitting on OutputTooLongError forwards extraction_prompt."""
    config = _minimal_config()
    dummy_prompt_obj = ExtractionPrompt(system_prompt="prebuilt prompt", response_schema=FactExtractionResponse)

    split_calls = []
    first_attempt = True

    async def mock_extract_chunk(**kwargs):
        nonlocal first_attempt
        if first_attempt:
            first_attempt = False
            raise OutputTooLongError("Output too long")
        split_calls.append(kwargs)
        return [{"what": "extracted"}], TokenUsage()

    with (
        patch.object(fact_extraction, "_extract_facts_from_chunk", side_effect=mock_extract_chunk),
        patch.object(fact_extraction, "_split_chunk_for_output_retry", return_value=("part 1", "part 2")),
    ):
        facts, usage = await fact_extraction._extract_facts_with_auto_split(
            chunk="long chunk needing split",
            chunk_index=0,
            total_chunks=1,
            event_date=None,
            context="",
            llm_config=SimpleNamespace(),
            config=config,
            extraction_prompt=dummy_prompt_obj,
        )

    assert len(split_calls) == 2
    for c in split_calls:
        assert c.get("extraction_prompt") is dummy_prompt_obj
