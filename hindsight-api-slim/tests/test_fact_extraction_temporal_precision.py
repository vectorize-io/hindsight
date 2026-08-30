"""Focused Retain tests for occurrence-precision propagation."""

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.response_models import TokenUsage
from hindsight_api.engine.retain.fact_extraction import (
    ExtractedFact,
    ExtractedFactNoCausal,
    ExtractedFactVerbose,
    Fact,
    RetainContent,
    VerbatimExtractedFact,
    _build_extraction_prompt_and_schema,
    _extract_facts_from_chunk,
    _parse_datetime,
    _resolve_temporal_fact_fields,
    extract_facts_from_contents,
)
from hindsight_api.engine.structured_output import strict_json_schema
from hindsight_api.engine.temporal_precision import OCCURRENCE_PRECISION_METADATA_KEY

UTC = timezone.utc


@pytest.mark.parametrize(
    "schema_model",
    [ExtractedFact, ExtractedFactVerbose, ExtractedFactNoCausal, VerbatimExtractedFact],
)
def test_every_extraction_schema_exposes_optional_occurrence_precision(schema_model):
    field = schema_model.model_fields["occurred_precision"]

    assert field.default is None


@pytest.mark.parametrize(
    "schema_model",
    [ExtractedFact, ExtractedFactVerbose, ExtractedFactNoCausal, VerbatimExtractedFact],
)
def test_every_strict_extraction_schema_requires_nullable_occurrence_precision(schema_model):
    schema = strict_json_schema(schema_model)
    precision_schema = schema["properties"]["occurred_precision"]

    assert "occurred_precision" in schema["required"]
    assert "default" not in precision_schema
    assert any(option.get("type") == "null" for option in precision_schema["anyOf"])
    assert any(
        set(option.get("enum", [])) == {"instant", "day", "month", "year", "range", "unknown"}
        for option in precision_schema["anyOf"]
    )


@pytest.mark.parametrize("extraction_mode", ["concise", "verbose", "verbatim"])
def test_every_llm_extraction_prompt_preserves_coarse_precision(extraction_mode):
    config = MagicMock()
    config.entity_labels = None
    config.entities_allow_free_form = True
    config.retain_extraction_mode = extraction_mode
    config.retain_extract_causal_links = False
    config.retain_mission = None
    config.retain_custom_instructions = None
    config.llm_output_language = None

    prompt, _schema = _build_extraction_prompt_and_schema(config)

    assert "occurred_precision" in prompt
    assert "NEVER invent a missing month" in prompt
    assert "Always include the day name" not in prompt


def test_lenient_temporal_normalization_infers_coarse_when_without_new_field():
    temporal = _resolve_temporal_fact_fields(
        fact_kind="event",
        occurred_start="2026-01-01",
        occurred_end="2026-01-01",
        occurred_precision=None,
        when="2026",
        combined_text="Summit talk | When: 2026",
        event_date=datetime(2026, 8, 30, tzinfo=UTC),
    )

    assert temporal.occurred_start == "2026-01-01"
    assert temporal.occurred_end == "2026-01-01"
    assert temporal.occurred_precision == "year"


def test_relative_date_fallback_preserves_day_precision_instead_of_midnight_instant():
    temporal = _resolve_temporal_fact_fields(
        fact_kind="event",
        occurred_start=None,
        occurred_end=None,
        occurred_precision=None,
        when="yesterday",
        combined_text="The summit happened yesterday",
        event_date=datetime(2026, 8, 30, 12, 0, tzinfo=UTC),
    )

    assert temporal.occurred_start == "2026-08-29T00:00:00+00:00"
    assert temporal.occurred_end == temporal.occurred_start
    assert temporal.occurred_precision == "day"


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("2026年", datetime(2026, 1, 1)),
        ("2026年8月", datetime(2026, 8, 1)),
        ("August 2026", datetime(2026, 8, 1)),
        ("not a date", None),
    ],
)
def test_datetime_parser_materializes_recognized_coarse_values(raw_value, expected):
    assert _parse_datetime(raw_value) == expected


@pytest.mark.asyncio
async def test_streaming_parser_normalizes_multilingual_temporal_matrix():
    raw_facts = [
        {
            "what": "English numeric year",
            "when": "2026",
            "fact_kind": "event",
            "fact_type": "world",
            "occurred_start": "2026-01-01",
            "occurred_end": "2026-01-01",
        },
        {
            "what": "中文年份",
            "when": "2025年",
            "fact_kind": "event",
            "fact_type": "world",
            "occurred_start": "2025年",
        },
        {
            "what": "English month",
            "when": "February 2024",
            "fact_kind": "event",
            "fact_type": "world",
            "occurred_start": "February 2024",
        },
        {
            "what": "中文月份",
            "when": "2024年2月",
            "fact_kind": "event",
            "fact_type": "world",
            "occurred_start": "2024年2月",
        },
        {
            "what": "Exact day",
            "when": "August 30, 2026",
            "fact_kind": "event",
            "fact_type": "world",
            "occurred_start": "2026-08-30",
        },
        {
            "what": "Exact instant",
            "when": "August 30, 2026 at 12:34",
            "fact_kind": "event",
            "fact_type": "world",
            "occurred_start": "2026-08-30T12:34:00+08:00",
        },
        {
            "what": "Genuine range",
            "when": "December 31, 2025 through January 2, 2026",
            "fact_kind": "event",
            "fact_type": "world",
            "occurred_start": "2025-12-31",
            "occurred_end": "2026-01-02",
        },
        {
            "what": "Conversation state",
            "when": "2026",
            "fact_kind": "conversation",
            "fact_type": "world",
            "occurred_start": "2026",
            "occurred_precision": "year",
        },
    ]
    llm_config = SimpleNamespace(
        _provider_impl=None,
        model="test-model",
        provider="test",
        call=AsyncMock(return_value=({"facts": raw_facts}, TokenUsage())),
    )
    config = SimpleNamespace(
        retain_extraction_mode="concise",
        retain_extract_causal_links=False,
        retain_mission=None,
        retain_llm_max_retries=0,
        llm_max_retries=0,
        retain_llm_initial_backoff=None,
        llm_initial_backoff=0.0,
        retain_llm_max_backoff=None,
        llm_max_backoff=0.0,
        llm_temperature_retain=None,
        llm_strict_schema_retain=True,
        retain_max_completion_tokens=None,
        entity_labels=None,
        entities_allow_free_form=True,
    )

    with patch(
        "hindsight_api.engine.retain.fact_extraction._build_extraction_prompt_and_schema",
        return_value=("system prompt", MagicMock()),
    ):
        facts, _usage = await _extract_facts_from_chunk(
            chunk="temporal matrix",
            chunk_index=0,
            total_chunks=1,
            event_date=datetime(2026, 8, 30, tzinfo=UTC),
            context="test",
            llm_config=cast(Any, llm_config),
            config=config,
            agent_name="test-agent",
        )

    assert [fact.occurred_precision for fact in facts] == [
        "year",
        "year",
        "month",
        "month",
        "day",
        "instant",
        "range",
        "unknown",
    ]
    assert facts[1].occurred_end == "2025年"
    assert facts[2].occurred_end == "February 2024"
    assert facts[3].occurred_end == "2024年2月"
    assert facts[7].occurred_start is None
    assert facts[7].occurred_end is None


@pytest.mark.asyncio
async def test_streaming_extraction_copies_metadata_and_engine_precision_wins():
    caller_metadata = {"source": "test", OCCURRENCE_PRECISION_METADATA_KEY: "day"}
    content = RetainContent(
        content="用户在2026年杭州开源峰会分享了时间感知记忆。",
        event_date=datetime(2026, 8, 30, tzinfo=UTC),
        metadata=caller_metadata,
    )
    llm_fact = Fact(
        fact="用户在杭州开源峰会分享了时间感知记忆 | When: 2026 | Involving: 用户",
        fact_type="world",
        occurred_start="2026",
        occurred_end="2026",
        occurred_precision="year",
    )
    config = SimpleNamespace(
        retain_batch_enabled=False,
        retain_extraction_mode="concise",
        entity_labels=None,
    )

    with (
        patch(
            "hindsight_api.engine.retain.fact_extraction.extract_facts_from_text",
            AsyncMock(return_value=([llm_fact], [(content.content, 1)], TokenUsage())),
        ),
        patch("hindsight_api.engine.retain.fact_extraction._add_temporal_offsets"),
    ):
        facts, _chunks, _usage = await extract_facts_from_contents(
            contents=[content],
            llm_config=None,
            agent_name="test-agent",
            config=config,
        )

    assert len(facts) == 1
    assert facts[0].occurred_start == datetime(2026, 1, 1)
    assert facts[0].metadata == {
        "source": "test",
        OCCURRENCE_PRECISION_METADATA_KEY: "year",
    }
    assert facts[0].metadata is not content.metadata
    assert content.metadata == caller_metadata
    assert caller_metadata == {"source": "test", OCCURRENCE_PRECISION_METADATA_KEY: "day"}


def test_chunks_mode_does_not_add_unknown_precision_metadata():
    from hindsight_api.engine.retain.fact_extraction import _extract_facts_chunks

    content = RetainContent(
        content="raw chunk",
        metadata={"source": "test", OCCURRENCE_PRECISION_METADATA_KEY: "year"},
    )
    config = SimpleNamespace(retain_chunk_size=4000, retain_structured_chunk_size=None)

    facts, _chunks, _usage = _extract_facts_chunks([content], config)

    assert facts[0].metadata == {"source": "test"}


@pytest.mark.asyncio
async def test_chunks_mode_bypasses_batch_and_strips_reserved_caller_precision():
    content = RetainContent(
        content="raw chunk",
        metadata={"source": "test", OCCURRENCE_PRECISION_METADATA_KEY: "year"},
    )
    config = SimpleNamespace(
        retain_extraction_mode="chunks",
        retain_batch_enabled=True,
        retain_chunk_size=4000,
        retain_structured_chunk_size=None,
    )
    llm_config = MagicMock()

    facts, _chunks, _usage = await extract_facts_from_contents(
        contents=[content],
        llm_config=llm_config,
        agent_name="test-agent",
        config=config,
    )

    assert facts[0].metadata == {"source": "test"}
    assert facts[0].occurred_start is None
    assert facts[0].mentioned_at == content.event_date
    assert not llm_config.mock_calls
