"""
Unit tests for fact extraction retry logic.

Tests the fix for the TypeError when LLM returns invalid JSON across all retries.
Previously, `raise last_error` would raise None (TypeError) because last_error was
only set in the BadRequestError handler, not when the LLM returned non-dict JSON.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_config(llm_max_retries: int = 3, retain_llm_max_retries: int | None = None):
    """Build a minimal HindsightConfig for fact extraction tests."""
    from hindsight_api.config import HindsightConfig

    cfg = MagicMock(spec=HindsightConfig)
    cfg.retain_llm_max_retries = retain_llm_max_retries
    cfg.llm_max_retries = llm_max_retries
    cfg.retain_llm_initial_backoff = None
    cfg.llm_initial_backoff = 0.0
    cfg.retain_llm_max_backoff = None
    cfg.llm_max_backoff = 0.0
    cfg.retain_max_completion_tokens = 8192
    cfg.retain_chunk_size = 3000
    cfg.retain_extraction_mode = "concise"
    cfg.retain_extract_causal_links = False
    cfg.retain_mission = None
    return cfg


def _make_llm_config(mock_response):
    """Build a mock LLMProvider that returns the given response."""
    from hindsight_api.engine.llm_wrapper import LLMProvider

    llm = MagicMock(spec=LLMProvider)
    llm.provider = "mock"
    token_usage = MagicMock()
    token_usage.__add__ = lambda self, other: self
    llm.call = AsyncMock(return_value=(mock_response, token_usage))
    return llm


@pytest.mark.asyncio
async def test_non_dict_json_all_retries_returns_empty():
    """
    When LLM returns non-dict JSON on every attempt, extraction should return []
    without raising TypeError ('exceptions must derive from BaseException').

    This was the bug: the loop ran range(2) times (hardcoded), but comparisons
    used config.llm_max_retries (default 10). On the last loop iteration (attempt=1),
    `attempt < 10 - 1` was True, so the code called `continue`, the loop
    exhausted, and `raise last_error` raised None → TypeError.
    """
    from hindsight_api.engine.retain.fact_extraction import _extract_facts_from_chunk

    # llm_max_retries=3 ensures the bug triggers with the old code (3 != 2 hardcoded)
    config = _make_config(llm_max_retries=3, retain_llm_max_retries=None)

    # Mock: always returns a list (non-dict), which is invalid
    llm_config = _make_llm_config(mock_response=[{"invalid": "response"}])

    with patch(
        "hindsight_api.engine.retain.fact_extraction._build_extraction_prompt_and_schema",
        return_value=("system prompt", MagicMock()),
    ):
        facts, usage = await _extract_facts_from_chunk(
            chunk="Alice visited Paris in 2023.",
            chunk_index=0,
            total_chunks=1,
            event_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
            context="travel notes",
            llm_config=llm_config,
            config=config,
            agent_name="test-agent",
        )

    assert facts == []


@pytest.mark.asyncio
async def test_non_dict_json_with_default_max_retries_returns_empty():
    """
    Same scenario with the default llm_max_retries=10 (matching real default config).
    The old code ran range(2) but checked against 10, always continuing until
    the loop exhausted, then raised None → TypeError.
    """
    from hindsight_api.engine.retain.fact_extraction import _extract_facts_from_chunk

    config = _make_config(llm_max_retries=10, retain_llm_max_retries=None)
    llm_config = _make_llm_config(mock_response="not a dict at all")

    with patch(
        "hindsight_api.engine.retain.fact_extraction._build_extraction_prompt_and_schema",
        return_value=("system prompt", MagicMock()),
    ):
        facts, usage = await _extract_facts_from_chunk(
            chunk="Some text.",
            chunk_index=0,
            total_chunks=1,
            event_date=datetime(2023, 6, 1, tzinfo=timezone.utc),
            context="",
            llm_config=llm_config,
            config=config,
            agent_name="agent",
        )

    assert facts == []


@pytest.mark.asyncio
async def test_retain_llm_max_retries_overrides_global():
    """
    When retain_llm_max_retries is set, it should be used for the loop range
    and all comparisons (no shadowing bug).
    """
    from hindsight_api.engine.retain.fact_extraction import _extract_facts_from_chunk

    # retain_llm_max_retries=5 should override llm_max_retries=10
    config = _make_config(llm_max_retries=10, retain_llm_max_retries=5)
    llm_config = _make_llm_config(mock_response=42)  # non-dict: integer

    with patch(
        "hindsight_api.engine.retain.fact_extraction._build_extraction_prompt_and_schema",
        return_value=("system prompt", MagicMock()),
    ):
        facts, usage = await _extract_facts_from_chunk(
            chunk="Bob likes Python.",
            chunk_index=0,
            total_chunks=1,
            event_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            context="",
            llm_config=llm_config,
            config=config,
            agent_name="agent",
        )

    assert facts == []
    # Verify it retried exactly retain_llm_max_retries times
    assert llm_config.call.call_count == 5


@pytest.mark.asyncio
async def test_retain_extraction_owns_structured_retry_budget():
    """
    Retain fact extraction has its own validation/retry loop. It should not pass
    the same retry budget down to the provider, otherwise structured-output
    validation failures are multiplied across nested retry layers.
    """
    from hindsight_api.engine.retain.fact_extraction import _extract_facts_from_chunk

    config = _make_config(llm_max_retries=10, retain_llm_max_retries=3)
    llm_config = _make_llm_config(mock_response=[{"invalid": "response"}])

    with patch(
        "hindsight_api.engine.retain.fact_extraction._build_extraction_prompt_and_schema",
        return_value=("system prompt", MagicMock()),
    ):
        facts, usage = await _extract_facts_from_chunk(
            chunk="Alice visited Paris in 2023.",
            chunk_index=0,
            total_chunks=1,
            event_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
            context="travel notes",
            llm_config=llm_config,
            config=config,
            agent_name="test-agent",
        )

    assert facts == []
    assert llm_config.call.call_count == 3
    assert {call.kwargs["max_retries"] for call in llm_config.call.await_args_list} == {0}


@pytest.mark.asyncio
async def test_chunk_retry_wrapper_respects_retain_retry_budget():
    """
    Chunk-level retries should use the retain LLM retry budget instead of a
    hardcoded retry count. Otherwise a user who lowers retain_llm_max_retries
    still gets repeated structured extraction attempts for each chunk.
    """
    from hindsight_api.engine.retain.fact_extraction import extract_facts_from_text

    config = _make_config(llm_max_retries=10, retain_llm_max_retries=1)
    llm_config = _make_llm_config(mock_response={"facts": []})

    with (
        patch(
            "hindsight_api.engine.retain.fact_extraction._extract_facts_with_auto_split",
            new_callable=AsyncMock,
            side_effect=RuntimeError("schema validation failed"),
        ) as extract_chunk,
        patch("hindsight_api.engine.retain.fact_extraction.asyncio.sleep", new_callable=AsyncMock),
    ):
        with pytest.raises(RuntimeError, match="Fact extraction failed"):
            await extract_facts_from_text(
                text="Alice visited Paris in 2023.",
                event_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
                llm_config=llm_config,
                agent_name="test-agent",
                config=config,
                context="travel notes",
            )

    assert extract_chunk.await_count == 1


@pytest.mark.asyncio
async def test_chunk_retry_wrapper_falls_back_to_global_retry_budget():
    """When retain_llm_max_retries is unset, chunk retries use llm_max_retries."""
    from hindsight_api.engine.retain.fact_extraction import extract_facts_from_text

    config = _make_config(llm_max_retries=2, retain_llm_max_retries=None)
    llm_config = _make_llm_config(mock_response={"facts": []})

    with (
        patch(
            "hindsight_api.engine.retain.fact_extraction._extract_facts_with_auto_split",
            new_callable=AsyncMock,
            side_effect=RuntimeError("upstream unavailable"),
        ) as extract_chunk,
        patch("hindsight_api.engine.retain.fact_extraction.asyncio.sleep", new_callable=AsyncMock),
    ):
        with pytest.raises(RuntimeError, match="Fact extraction failed"):
            await extract_facts_from_text(
                text="Alice visited Paris in 2023.",
                event_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
                llm_config=llm_config,
                agent_name="test-agent",
                config=config,
                context="travel notes",
            )

    assert extract_chunk.await_count == 2


@pytest.mark.asyncio
async def test_zero_retain_retry_budget_still_allows_one_attempt():
    """A zero retry budget should mean no retries, not zero total attempts."""
    from hindsight_api.engine.retain.fact_extraction import _extract_facts_from_chunk

    config = _make_config(llm_max_retries=10, retain_llm_max_retries=0)
    llm_config = _make_llm_config(mock_response={
        "facts": [
            {
                "what": "Alice visited Paris",
                "when": "2023",
                "who": "Alice",
                "why": "vacation",
            }
        ]
    })

    with patch(
        "hindsight_api.engine.retain.fact_extraction._build_extraction_prompt_and_schema",
        return_value=("system prompt", MagicMock()),
    ):
        facts, usage = await _extract_facts_from_chunk(
            chunk="Alice visited Paris in 2023.",
            chunk_index=0,
            total_chunks=1,
            event_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
            context="travel notes",
            llm_config=llm_config,
            config=config,
            agent_name="test-agent",
        )

    assert len(facts) == 1
    assert llm_config.call.call_count == 1
    assert llm_config.call.await_args.kwargs["max_retries"] == 0


@pytest.mark.asyncio
async def test_multi_chunk_retry_budget_is_bounded_per_chunk():
    """Total failed chunk attempts are bounded by chunk_count * retry budget."""
    from hindsight_api.engine.retain.fact_extraction import extract_facts_from_text

    config = _make_config(llm_max_retries=10, retain_llm_max_retries=2)
    llm_config = _make_llm_config(mock_response={"facts": []})

    with (
        patch("hindsight_api.engine.retain.fact_extraction.chunk_text", return_value=["chunk one", "chunk two"]),
        patch(
            "hindsight_api.engine.retain.fact_extraction._extract_facts_with_auto_split",
            new_callable=AsyncMock,
            side_effect=RuntimeError("schema validation failed"),
        ) as extract_chunk,
        patch("hindsight_api.engine.retain.fact_extraction.asyncio.sleep", new_callable=AsyncMock),
    ):
        with pytest.raises(RuntimeError, match="2/2 chunks failed after 2 attempts each"):
            await extract_facts_from_text(
                text="long text split into two chunks",
                event_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
                llm_config=llm_config,
                agent_name="test-agent",
                config=config,
                context="travel notes",
            )

    assert extract_chunk.await_count == 4


@pytest.mark.asyncio
async def test_chunk_retry_still_recovers_from_transient_failure():
    """The retry cap is bounded, but a transient chunk failure can still recover."""
    from hindsight_api.engine.response_models import TokenUsage
    from hindsight_api.engine.retain.fact_extraction import extract_facts_from_text

    config = _make_config(llm_max_retries=10, retain_llm_max_retries=2)
    llm_config = _make_llm_config(mock_response={"facts": []})

    with (
        patch(
            "hindsight_api.engine.retain.fact_extraction._extract_facts_with_auto_split",
            new_callable=AsyncMock,
            side_effect=[
                RuntimeError("temporary provider failure"),
                ([], TokenUsage()),
            ],
        ) as extract_chunk,
        patch("hindsight_api.engine.retain.fact_extraction.asyncio.sleep", new_callable=AsyncMock),
    ):
        facts, chunks, usage = await extract_facts_from_text(
            text="Alice visited Paris in 2023.",
            event_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
            llm_config=llm_config,
            agent_name="test-agent",
            config=config,
            context="travel notes",
        )

    assert facts == []
    assert chunks == [("Alice visited Paris in 2023.", 0)]
    assert extract_chunk.await_count == 2


@pytest.mark.asyncio
async def test_none_event_date_with_empty_facts_no_crash():
    """
    When event_date is None and the LLM returns an empty facts list,
    the debug log should not crash with AttributeError on .isoformat().

    Regression test for https://github.com/vectorize-io/hindsight/issues/874
    """
    from hindsight_api.engine.retain.fact_extraction import _extract_facts_from_chunk

    config = _make_config(llm_max_retries=1)

    # LLM returns a valid dict but with no facts — triggers the debug log path
    llm_config = _make_llm_config(mock_response={"facts": []})

    with patch(
        "hindsight_api.engine.retain.fact_extraction._build_extraction_prompt_and_schema",
        return_value=("system prompt", MagicMock()),
    ):
        facts, usage = await _extract_facts_from_chunk(
            chunk="A plain text document with no timestamp.",
            chunk_index=0,
            total_chunks=1,
            event_date=None,
            context="",
            llm_config=llm_config,
            config=config,
            agent_name="test-agent",
        )

    assert facts == []


@pytest.mark.asyncio
async def test_none_event_date_with_valid_facts_no_crash():
    """
    When event_date is None but the LLM returns valid facts,
    extraction should succeed without errors.
    """
    from hindsight_api.engine.retain.fact_extraction import _extract_facts_from_chunk

    config = _make_config(llm_max_retries=1)

    llm_config = _make_llm_config(mock_response={
        "facts": [
            {
                "what": "Alice visited Paris",
                "when": "2023",
                "who": "Alice",
                "why": "vacation",
            }
        ]
    })

    with patch(
        "hindsight_api.engine.retain.fact_extraction._build_extraction_prompt_and_schema",
        return_value=("system prompt", MagicMock()),
    ):
        facts, usage = await _extract_facts_from_chunk(
            chunk="Alice visited Paris in 2023.",
            chunk_index=0,
            total_chunks=1,
            event_date=None,
            context="",
            llm_config=llm_config,
            config=config,
            agent_name="test-agent",
        )

    assert len(facts) == 1
    assert "Alice visited Paris" in facts[0].fact
