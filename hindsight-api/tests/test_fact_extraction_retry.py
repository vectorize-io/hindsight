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
