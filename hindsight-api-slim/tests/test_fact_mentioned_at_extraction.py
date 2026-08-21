"""Regression tests for per-fact source timestamp propagation (issue #2550)."""

import dataclasses
import json
from datetime import datetime, timedelta, timezone
from types import NoneType, SimpleNamespace
from typing import cast, get_args
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from hindsight_api.config import HindsightConfig, _get_raw_config
from hindsight_api.engine.llm_wrapper import LLMProvider
from hindsight_api.engine.providers.mock_llm import MockLLM
from hindsight_api.engine.response_models import TokenUsage
from hindsight_api.engine.retain.fact_extraction import (
    Fact,
    _build_extraction_prompt_and_schema,
    _resolve_mentioned_at,
    _source_instants_from_text,
    extract_facts_from_contents,
    extract_facts_from_contents_batch_api,
)
from hindsight_api.engine.retain.orchestrator import _process_extracted_facts
from hindsight_api.engine.retain.types import ExtractedFact, ProcessedFact, RetainContent
from hindsight_api.engine.structured_output import strict_json_schema

EVENT_DATE = datetime(2026, 7, 1, 8, 0, tzinfo=timezone.utc)
FIRST_SOURCE_TIME = datetime(2026, 7, 5, 9, 15, tzinfo=timezone.utc)
SECOND_SOURCE_TIME = datetime(2026, 7, 10, 18, 30, tzinfo=timezone(timedelta(hours=2)))
OCCURRED_TIME = datetime(2022, 4, 3, 12, 0, tzinfo=timezone.utc)


def _config(*, mode: str = "concise", batch: bool = False, causal: bool = False) -> HindsightConfig:
    return dataclasses.replace(
        _get_raw_config(),
        retain_batch_enabled=batch,
        retain_batch_poll_interval_seconds=0,
        retain_chunk_size=4000,
        retain_extraction_mode=mode,
        retain_extract_causal_links=causal,
        retain_llm_max_retries=0,
        retain_mission=None,
        llm_temperature_retain=0.1,
        llm_strict_schema_retain=True,
    )


def _raw_facts() -> list[dict[str, str | None]]:
    return [
        {
            "what": "The user's preferred database is PostgreSQL",
            "fact_type": "world",
            "fact_kind": "conversation",
            "mentioned_at": FIRST_SOURCE_TIME.isoformat(),
        },
        {
            "what": "The user moved to Berlin in 2022",
            "fact_type": "world",
            "fact_kind": "event",
            "mentioned_at": SECOND_SOURCE_TIME.isoformat(),
            "occurred_start": OCCURRED_TIME.isoformat(),
            "occurred_end": OCCURRED_TIME.isoformat(),
        },
        {
            "what": "The user prefers pnpm",
            "fact_type": "world",
            "fact_kind": "conversation",
        },
    ]


def _streaming_llm(raw_facts: list[dict[str, str | None]]) -> MagicMock:
    llm = MagicMock(spec=LLMProvider)
    llm.provider = "mock"
    llm._provider_impl = SimpleNamespace(supports_prompt_caching=lambda: False)
    llm.call = AsyncMock(return_value=({"facts": raw_facts}, TokenUsage()))
    return llm


def _batch_llm(raw_facts: list[dict[str, str | None]]) -> MagicMock:
    provider = SimpleNamespace(
        provider="groq",
        model="test-model",
        supports_batch_api=AsyncMock(return_value=True),
        submit_batch=AsyncMock(return_value={"batch_id": "batch-mentioned-at"}),
        get_batch_status=AsyncMock(
            return_value={
                "status": "completed",
                "request_counts": {"total": 1, "completed": 1, "failed": 0},
            }
        ),
        retrieve_batch_results=AsyncMock(
            return_value=[
                {
                    "custom_id": "chunk_0",
                    "response": {
                        "body": {
                            "choices": [{"message": {"content": json.dumps({"facts": raw_facts})}}],
                            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                        }
                    },
                }
            ]
        ),
    )
    llm = MagicMock(spec=LLMProvider)
    llm.provider = "groq"
    llm.model = "test-model"
    llm._provider_impl = provider

    async def _batch_provider_impl(*, account_key: str | None = None):
        return provider

    llm.batch_provider_impl = _batch_provider_impl
    return llm


def _content() -> RetainContent:
    return RetainContent(
        content="\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-07-05T09:15:00Z",
                        "content": "The user's preferred database is PostgreSQL",
                    }
                ),
                json.dumps(
                    {
                        "created_at": "2026-07-10T16:30:00Z",
                        "content": "The user moved to Berlin in 2022",
                    }
                ),
            ]
        ),
        event_date=EVENT_DATE,
        context="JSONL session",
    )


@pytest.mark.parametrize(
    ("mode", "causal", "dynamic_labels"),
    [
        ("concise", True, False),
        ("concise", False, False),
        ("verbose", True, False),
        ("concise", False, True),
    ],
)
def test_semantic_extraction_schemas_request_nullable_mentioned_at(
    mode: str, causal: bool, dynamic_labels: bool
) -> None:
    """Semantic extraction modes expose the per-fact source-time contract."""
    config = _config(mode=mode, causal=causal)
    if dynamic_labels:
        config = dataclasses.replace(
            config,
            entity_labels=[{"key": "topic", "type": "value", "values": [{"value": "database"}]}],
        )
    prompt, response_model = _build_extraction_prompt_and_schema(config)
    fact_model = get_args(cast(type[BaseModel], response_model).model_fields["facts"].annotation)[0]
    mentioned_at_field = fact_model.model_fields["mentioned_at"]

    assert set(get_args(mentioned_at_field.annotation)) == {str, NoneType}
    assert "source message or record" in prompt
    assert "not when the described event happened" in prompt
    assert "explicit timezone" in prompt
    assert "Prefer the extended form" in prompt

    schema = strict_json_schema(response_model)
    fact_ref = schema["properties"]["facts"]["items"]["$ref"].rsplit("/", 1)[-1]
    fact_schema = schema["$defs"][fact_ref]
    assert "mentioned_at" in fact_schema["properties"]
    assert "mentioned_at" in fact_schema["required"]
    if dynamic_labels:
        assert "labels" in fact_model.model_fields
        assert "labels" in fact_schema["required"]


def test_verbatim_schema_keeps_one_entry_per_chunk_without_mentioned_at() -> None:
    """Verbatim cannot attribute one raw multi-message chunk to one source time."""
    prompt, response_model = _build_extraction_prompt_and_schema(_config(mode="verbatim"))
    fact_model = get_args(cast(type[BaseModel], response_model).model_fields["facts"].annotation)[0]

    assert "mentioned_at" not in fact_model.model_fields
    assert "SOURCE TIMESTAMP" not in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("batch", [False, True], ids=["streaming", "batch"])
async def test_semantic_paths_resolve_times_and_preserve_fallback_ordering(batch: bool) -> None:
    """Both extraction paths keep source times exact and offset only fallbacks."""
    extractor = extract_facts_from_contents_batch_api if batch else extract_facts_from_contents
    facts, _chunks, _usage = await extractor(
        contents=[_content()],
        llm_config=_batch_llm(_raw_facts()) if batch else _streaming_llm(_raw_facts()),
        agent_name="test-agent",
        config=_config(batch=batch),
    )

    assert [fact.mentioned_at for fact in facts] == [
        FIRST_SOURCE_TIME,
        SECOND_SOURCE_TIME,
        EVENT_DATE + timedelta(milliseconds=20),
    ]
    assert facts[0].occurred_start is None
    assert facts[1].occurred_start == OCCURRED_TIME + timedelta(milliseconds=10)
    assert facts[1].occurred_end == OCCURRED_TIME + timedelta(milliseconds=10)


@pytest.mark.asyncio
async def test_source_timestamp_does_not_cross_chunk_boundaries() -> None:
    """A timestamp visible only in another chunk must fall back to the item time."""
    first_line = json.dumps({"timestamp": FIRST_SOURCE_TIME.isoformat(), "content": "Prefers PostgreSQL"})
    second_line = json.dumps({"timestamp": SECOND_SOURCE_TIME.isoformat(), "content": "Prefers Svelte"})
    llm = _streaming_llm([])
    llm.call = AsyncMock(
        side_effect=[
            (
                {
                    "facts": [
                        {
                            "what": "The user prefers PostgreSQL",
                            "fact_type": "world",
                            "fact_kind": "conversation",
                            "mentioned_at": SECOND_SOURCE_TIME.isoformat(),
                        }
                    ]
                },
                TokenUsage(),
            ),
            (
                {
                    "facts": [
                        {
                            "what": "The user prefers Svelte",
                            "fact_type": "world",
                            "fact_kind": "conversation",
                            "mentioned_at": FIRST_SOURCE_TIME.isoformat(),
                        }
                    ]
                },
                TokenUsage(),
            ),
        ]
    )
    config = dataclasses.replace(_config(), retain_chunk_size=1, retain_structured_chunk_size=1000)

    facts, chunks, _usage = await extract_facts_from_contents(
        contents=[RetainContent(content=f"{first_line}\n{second_line}", event_date=EVENT_DATE)],
        llm_config=llm,
        agent_name="test-agent",
        config=config,
    )

    assert len(chunks) == 2
    assert [fact.mentioned_at for fact in facts] == [EVENT_DATE, EVENT_DATE + timedelta(milliseconds=10)]
    assert all(fact.mentioned_at_from_source is False for fact in facts)


@pytest.mark.asyncio
@pytest.mark.parametrize("batch", [False, True], ids=["streaming", "batch"])
async def test_verbatim_paths_use_item_fallback(batch: bool) -> None:
    """Both paths use the item fallback; streaming also preserves one raw chunk fact."""
    extractor = extract_facts_from_contents_batch_api if batch else extract_facts_from_contents
    content = _content()
    # The Batch fixture supplies a test-only `what` field solely to reach the
    # current post-processing path; it is not a valid verbatim-output guarantee.
    facts, _chunks, _usage = await extractor(
        contents=[content],
        llm_config=_batch_llm(_raw_facts()[:1]) if batch else _streaming_llm(_raw_facts()[:1]),
        agent_name="test-agent",
        config=_config(mode="verbatim", batch=batch),
    )

    assert facts
    assert all(fact.mentioned_at == EVENT_DATE for fact in facts)
    assert all(fact.mentioned_at_from_source is False for fact in facts)
    if not batch:
        assert len(facts) == 1
        assert facts[0].fact_text == content.content


@pytest.mark.parametrize(
    ("candidate", "source_text", "event_date", "expected", "from_source"),
    [
        ("2026-07-05T09:15:00+00:00", "2026-07-05T09:15:00Z", EVENT_DATE, FIRST_SOURCE_TIME, True),
        ("20260705T091500Z", "20260705T091500Z", EVENT_DATE, FIRST_SOURCE_TIME, True),
        ("2026-07-05 09:15:00Z", "2026-07-05 09:15:00Z", EVENT_DATE, FIRST_SOURCE_TIME, True),
        ("2026-07-05t09:15:00z", "2026-07-05t09:15:00z", EVENT_DATE, FIRST_SOURCE_TIME, True),
        ("2026-07-05T09:15:00+0000", "2026-07-05T09:15:00+0000", EVENT_DATE, FIRST_SOURCE_TIME, True),
        ("2026-07-11T09:00:00Z", _content().content, EVENT_DATE, EVENT_DATE, False),
        ("not-a-timestamp", "not-a-timestamp", EVENT_DATE, EVENT_DATE, False),
        ("2026-07-11T09:00:00", "2026-07-11T09:00:00", EVENT_DATE, EVENT_DATE, False),
        ("20260705T091500Z", "120260705T091500Z", EVENT_DATE, EVENT_DATE, False),
        ("20260705T091500Z", "event_20260705T091500Z_record", EVENT_DATE, EVENT_DATE, False),
        (None, "no timestamp", None, None, False),
    ],
    ids=[
        "timezone-equivalent-source",
        "basic",
        "space-separator",
        "lowercase",
        "basic-offset",
        "hallucinated-valid",
        "invalid",
        "timezone-less",
        "embedded-in-longer-number",
        "embedded-in-identifier",
        "no-source-or-fallback",
    ],
)
def test_source_timestamp_resolution_contract(
    candidate: str | None,
    source_text: str,
    event_date: datetime | None,
    expected: datetime | None,
    from_source: bool,
) -> None:
    resolution = _resolve_mentioned_at(candidate, event_date, _source_instants_from_text(source_text))

    assert resolution.value == expected
    assert resolution.from_source is from_source


def test_provenance_is_not_serialized_or_carried_into_processed_fact() -> None:
    fact = Fact(
        fact="The user prefers PostgreSQL",
        fact_type="world",
        mentioned_at=FIRST_SOURCE_TIME.isoformat(),
        mentioned_at_from_source=True,
    )
    assert "mentioned_at_from_source" not in fact.model_dump()
    extracted = ExtractedFact(
        fact_text="The user prefers PostgreSQL",
        fact_type="world",
        mentioned_at=FIRST_SOURCE_TIME,
        mentioned_at_from_source=True,
    )
    # Exercise the actual retain mapper that feeds consolidation/storage.
    mapped = _process_extracted_facts([extracted], [[0.0]])
    assert isinstance(mapped.processed_facts[0], ProcessedFact)
    assert mapped.processed_facts[0].mentioned_at == FIRST_SOURCE_TIME
    assert "mentioned_at_from_source" not in vars(mapped.processed_facts[0])


@pytest.mark.asyncio
async def test_chunks_mode_keeps_item_level_fallback_ordering() -> None:
    """The no-LLM chunks mode preserves its existing event-date fallback behavior."""
    facts, _chunks, _usage = await extract_facts_from_contents(
        contents=[
            RetainContent(content="first raw chunk", event_date=EVENT_DATE),
            RetainContent(content="second raw chunk", event_date=EVENT_DATE),
        ],
        llm_config=MagicMock(),
        agent_name="test-agent",
        config=_config(mode="chunks"),
    )

    assert [fact.mentioned_at for fact in facts] == [EVENT_DATE, EVENT_DATE + timedelta(milliseconds=10)]


@pytest.mark.asyncio
async def test_source_mentioned_at_is_readable_after_retain(memory, request_context) -> None:
    """The public MemoryEngine read API exposes the persisted source timestamp."""
    bank_id = "test_source_mentioned_at_public_read"
    document_id = "source-mentioned-at-document"
    provider = memory._retain_llm_config._provider_impl
    assert isinstance(provider, MockLLM)
    provider.set_mock_response(
        {
            "facts": [
                {
                    "what": "The user prefers PostgreSQL",
                    "fact_type": "world",
                    "fact_kind": "conversation",
                    "mentioned_at": FIRST_SOURCE_TIME.isoformat(),
                }
            ]
        }
    )

    try:
        unit_ids = await memory.retain_async(
            bank_id=bank_id,
            content=json.dumps(
                {
                    "timestamp": FIRST_SOURCE_TIME.isoformat(),
                    "content": "The user prefers PostgreSQL",
                }
            ),
            event_date=EVENT_DATE,
            document_id=document_id,
            request_context=request_context,
        )
        assert len(unit_ids) == 1

        page = await memory.list_memory_units(
            bank_id,
            fact_type="world",
            document_id=document_id,
            request_context=request_context,
        )
        assert len(page["items"]) == 1
        assert datetime.fromisoformat(page["items"][0]["mentioned_at"]) == FIRST_SOURCE_TIME
        assert "mentioned_at_from_source" not in page["items"][0]["metadata"]
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
