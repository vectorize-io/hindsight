"""
Unit tests for fact extraction retry logic.

When the LLM returns non-dict JSON across all retries, extraction must raise a
RuntimeError (issue #1833 — never silently return [] and let the retain commit
the document with 0 facts). This also guards the original TypeError bug: the
raise must be a real exception, not `raise None` ('exceptions must derive from
BaseException'), which happened when last_error was only set in the
BadRequestError handler and not for non-dict JSON responses.
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def test_output_retry_split_preserves_conversation_array_boundaries():
    """OutputTooLong retry splitting must keep conversation chunks valid JSON arrays."""
    from hindsight_api.engine.retain.fact_extraction import _split_chunk_for_output_retry

    turns = [
        {"role": "user", "content": "alpha"},
        {"role": "assistant", "content": "bravo"},
        {"role": "user", "content": "charlie"},
        {"role": "assistant", "content": "delta"},
    ]

    split = _split_chunk_for_output_retry(json.dumps(turns))

    assert split is not None
    first, second = split
    assert json.loads(first) == turns[:2]
    assert json.loads(second) == turns[2:]


def test_output_retry_split_divides_single_oversized_turn_content():
    """A lone oversized conversation turn is split inside content and rewrapped."""
    from hindsight_api.engine.retain.fact_extraction import _split_chunk_for_output_retry

    turn = {"role": "user", "content": "abcdefghijklmnopqrstuvwxyz", "name": "casey"}

    split = _split_chunk_for_output_retry(json.dumps([turn]))

    assert split is not None
    first, second = split
    first_turn = json.loads(first)[0]
    second_turn = json.loads(second)[0]
    assert first_turn["role"] == "user"
    assert second_turn["role"] == "user"
    assert first_turn["name"] == "casey"
    assert second_turn["name"] == "casey"
    assert first_turn["content"] + second_turn["content"] == turn["content"]


def test_output_retry_split_returns_none_when_no_progress_possible():
    """Pathological tiny chunks should be dropped instead of recursively retried."""
    from hindsight_api.engine.retain.fact_extraction import _split_chunk_for_output_retry

    assert _split_chunk_for_output_retry("x") is None
    assert _split_chunk_for_output_retry(json.dumps([{"role": "user", "content": ""}])) is None


@pytest.mark.asyncio
async def test_output_too_long_drops_unsplittable_subchunk_without_recursing():
    """If a chunk cannot be reduced further, auto-split exits gracefully."""
    from hindsight_api.engine.llm_wrapper import OutputTooLongError
    from hindsight_api.engine.retain.fact_extraction import _extract_facts_with_auto_split

    config = _make_config(llm_max_retries=1)
    llm_config = _make_llm_config(mock_response={})

    with patch(
        "hindsight_api.engine.retain.fact_extraction._extract_facts_from_chunk",
        side_effect=OutputTooLongError("too long"),
    ) as extract:
        facts, usage = await _extract_facts_with_auto_split(
            chunk="x",
            chunk_index=0,
            total_chunks=1,
            event_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
            context="",
            llm_config=llm_config,
            config=config,
            agent_name="agent",
        )

    assert facts == []
    assert extract.call_count == 1


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
    cfg.llm_temperature_retain = 0.1
    cfg.llm_strict_schema_retain = False
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
async def test_non_dict_json_all_retries_raises():
    """
    When LLM returns non-dict JSON on every attempt, extraction must RAISE after
    exhausting retries — never silently return [] (which would let the retain
    commit the document with 0 facts; see issue #1833).

    Regression guard for the original TypeError too: the raise must be a real
    RuntimeError, not `raise None` ('exceptions must derive from BaseException').
    """
    from hindsight_api.engine.retain.fact_extraction import _extract_facts_from_chunk

    config = _make_config(llm_max_retries=3, retain_llm_max_retries=None)

    # Mock: always returns a list containing a non-dict item, which is invalid.
    llm_config = _make_llm_config(mock_response=["invalid response"])

    with patch(
        "hindsight_api.engine.retain.fact_extraction._build_extraction_prompt_and_schema",
        return_value=("system prompt", MagicMock()),
    ):
        with pytest.raises(RuntimeError, match="non-dict JSON"):
            await _extract_facts_from_chunk(
                chunk="Alice visited Paris in 2023.",
                chunk_index=0,
                total_chunks=1,
                event_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
                context="travel notes",
                llm_config=llm_config,
                config=config,
                agent_name="test-agent",
            )

    assert llm_config.call.call_count == 3


@pytest.mark.asyncio
async def test_top_level_fact_list_is_accepted_without_retry():
    """
    Some lax-JSON models return the facts array directly instead of wrapping it
    in {"facts": [...]}. A top-level list of dict-shaped facts is recoverable
    and should not burn retries.
    """
    from hindsight_api.engine.retain.fact_extraction import _extract_facts_from_chunk

    config = _make_config(llm_max_retries=3, retain_llm_max_retries=None)
    llm_config = _make_llm_config(
        mock_response=[
            {
                "what": "Alice visited Paris",
                "when": "2023",
                "where": "Paris",
                "who": "Alice",
                "why": "vacation",
                "fact_type": "world",
                "fact_kind": "conversation",
            }
        ]
    )

    with patch(
        "hindsight_api.engine.retain.fact_extraction._build_extraction_prompt_and_schema",
        return_value=("system prompt", MagicMock()),
    ):
        facts, _usage = await _extract_facts_from_chunk(
            chunk="Alice visited Paris in 2023.",
            chunk_index=0,
            total_chunks=1,
            event_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
            context="travel notes",
            llm_config=llm_config,
            config=config,
            agent_name="test-agent",
        )

    assert llm_config.call.call_count == 1
    assert len(facts) == 1
    assert "Alice visited Paris" in facts[0].fact


@pytest.mark.asyncio
async def test_non_dict_json_with_default_max_retries_raises():
    """
    Same scenario with the default llm_max_retries=10 (matching real default config):
    must raise after exhausting all retries rather than returning [].
    """
    from hindsight_api.engine.retain.fact_extraction import _extract_facts_from_chunk

    config = _make_config(llm_max_retries=10, retain_llm_max_retries=None)
    llm_config = _make_llm_config(mock_response="not a dict at all")

    with patch(
        "hindsight_api.engine.retain.fact_extraction._build_extraction_prompt_and_schema",
        return_value=("system prompt", MagicMock()),
    ):
        with pytest.raises(RuntimeError, match="non-dict JSON"):
            await _extract_facts_from_chunk(
                chunk="Some text.",
                chunk_index=0,
                total_chunks=1,
                event_date=datetime(2023, 6, 1, tzinfo=timezone.utc),
                context="",
                llm_config=llm_config,
                config=config,
                agent_name="agent",
            )

    assert llm_config.call.call_count == 10


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
        with pytest.raises(RuntimeError, match="non-dict JSON"):
            await _extract_facts_from_chunk(
                chunk="Bob likes Python.",
                chunk_index=0,
                total_chunks=1,
                event_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                context="",
                llm_config=llm_config,
                config=config,
                agent_name="agent",
            )

    # Verify it retried exactly retain_llm_max_retries times
    assert llm_config.call.call_count == 5


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

    llm_config = _make_llm_config(
        mock_response={
            "facts": [
                {
                    "what": "Alice visited Paris",
                    "when": "2023",
                    "who": "Alice",
                    "why": "vacation",
                }
            ]
        }
    )

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


def _make_batch_temp_config(temperature):
    """Minimal config for _build_request_body temperature tests."""
    from hindsight_api.config import HindsightConfig

    cfg = MagicMock(spec=HindsightConfig)
    cfg.llm_temperature_retain = temperature
    cfg.retain_max_completion_tokens = None
    cfg.llm_strict_schema = False
    return cfg


def _make_batch_llm_config():
    """Minimal LLMProvider mock for _build_request_body (non-openai skips service_tier)."""
    from hindsight_api.engine.llm_wrapper import LLMProvider

    llm = MagicMock(spec=LLMProvider)
    llm.model = "gpt-test"
    llm.provider = "mock"
    return llm


def test_build_request_body_forwards_configured_temperature():
    """Batch retain path must send the configured retain temperature."""
    from hindsight_api.engine.retain.fact_extraction import _build_request_body

    body = _build_request_body(_make_batch_llm_config(), _make_batch_temp_config(0.7), "sys", "user", dict)
    assert body["temperature"] == 0.7


def test_build_request_body_omits_temperature_when_none():
    """HINDSIGHT_API_LLM_TEMPERATURE=none must drop temperature from the batch
    request body too (Azure GPT-5.5 rejects explicit temperatures). Follow-up to
    #2469, which only de-hardcoded the streaming path and left the batch
    _build_request_body hardcoding temperature=0.1."""
    from hindsight_api.engine.retain.fact_extraction import _build_request_body

    body = _build_request_body(_make_batch_llm_config(), _make_batch_temp_config(None), "sys", "user", dict)
    assert "temperature" not in body
