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

    # Budget 3 => 3 retries after the initial request.
    assert llm_config.call.call_count == 4


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

    assert llm_config.call.call_count == 11


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

    # Verify it retried exactly retain_llm_max_retries times after the initial request
    assert llm_config.call.call_count == 6


@pytest.mark.asyncio
async def test_zero_retry_budget_performs_single_chunk_extraction_call():
    """
    Direct _extract_facts_from_chunk with a retry budget of 0 (issue #2731):
    the outer loop must still run once, and the RAW budget (0) must reach the
    provider so it stays the single owner of transport retries.
    """
    from hindsight_api.engine.retain.fact_extraction import _extract_facts_from_chunk

    config = _make_config(llm_max_retries=3, retain_llm_max_retries=0)
    llm_config = _make_llm_config(
        mock_response={"facts": [{"what": "Alice visited Paris", "when": "2023", "who": "Alice", "why": "vacation"}]}
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
            context="",
            llm_config=llm_config,
            config=config,
            agent_name="test-agent",
        )

    assert llm_config.call.call_count == 1
    assert llm_config.call.call_args.kwargs["max_retries"] == 0
    assert len(facts) == 1


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


# --- Retry budget semantics (issue #2731) -----------------------------------
#
# A retry BUDGET of N means N retries *after* the initial request — the meaning
# every provider already implements (`for attempt in range(max_retries + 1)`) and
# the meaning the OpenAI SDK documents for `max_retries=0` ("disable retries",
# i.e. one request). The tests below drive the *public* extract_facts_from_text
# entry point with the real HindsightConfig an operator gets from
# HINDSIGHT_API_RETAIN_LLM_MAX_RETRIES, so they exercise the whole
# env -> config -> chunk -> auto-split -> extraction -> provider chain rather
# than a private helper.

_VALID_EXTRACTION_RESPONSE = {
    "facts": [{"what": "Alice visited Paris", "when": "2023", "who": "Alice", "why": "vacation"}]
}


def _make_recording_llm(mock_response):
    """LLMProvider double returning ``mock_response`` and recording call kwargs."""
    from hindsight_api.engine.llm_wrapper import LLMProvider
    from hindsight_api.engine.response_models import TokenUsage

    llm = MagicMock(spec=LLMProvider)
    llm.provider = "mock"
    llm.call = AsyncMock(return_value=(mock_response, TokenUsage()))
    return llm


@pytest.fixture
def retain_config(monkeypatch):
    """Factory for the real config an operator gets from the retry-budget env var.

    Mirrors the reporter's setup: HINDSIGHT_API_RETAIN_LLM_MAX_RETRIES is the only
    knob they touch. The global budget is pinned so the "unset" row provably
    exercises the fallback.
    """
    from hindsight_api.config import _get_raw_config, clear_config_cache

    def _build(retain_budget: str | None):
        monkeypatch.setenv("HINDSIGHT_API_LLM_MAX_RETRIES", "3")
        if retain_budget is None:
            monkeypatch.delenv("HINDSIGHT_API_RETAIN_LLM_MAX_RETRIES", raising=False)
        else:
            monkeypatch.setenv("HINDSIGHT_API_RETAIN_LLM_MAX_RETRIES", retain_budget)
        clear_config_cache()
        return _get_raw_config()

    yield _build
    clear_config_cache()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "retain_budget, expected_forwarded_retries",
    [
        # The reported repro: a gateway owns transport retries, so the operator
        # sets the budget to 0. This used to perform ZERO extraction requests and
        # raise "Fact extraction failed after 0 attempts".
        pytest.param("0", 0, id="zero_budget_gateway_owns_retries"),
        pytest.param("1", 1, id="budget_one"),
        pytest.param("3", 3, id="budget_three"),
        pytest.param(None, 3, id="unset_falls_back_to_global"),
    ],
)
async def test_retry_budget_always_performs_initial_extraction_request(
    retain_config, retain_budget, expected_forwarded_retries
):
    """Any retry budget — including 0 — must still perform the initial request."""
    from hindsight_api.engine.retain.fact_extraction import extract_facts_from_text

    config = retain_config(retain_budget)
    llm = _make_recording_llm(_VALID_EXTRACTION_RESPONSE)

    facts, _chunks, _usage = await extract_facts_from_text(
        text="Alice visited Paris in 2023.",
        event_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
        llm_config=llm,
        agent_name="test-agent",
        config=config,
        context="",
    )

    assert llm.call.call_count == 1
    assert len(facts) == 1
    assert "Alice visited Paris" in facts[0].fact
    # The RAW budget reaches the provider — not the outer attempt count — so the
    # provider stays the single retry owner (0 => gateway owns transport retries).
    assert llm.call.call_args.kwargs["max_retries"] == expected_forwarded_retries


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "retain_budget, expected_calls",
    [
        # Budget 0 means "one request, no retry" — one attempt, then a real
        # content error. Never zero attempts.
        pytest.param("0", 1, id="zero_budget_one_attempt"),
        pytest.param("1", 2, id="budget_one_retries_once"),
        pytest.param("3", 4, id="budget_three_retries_thrice"),
    ],
)
async def test_malformed_response_still_attempts_then_fails_loudly(retain_config, retain_budget, expected_calls):
    """A malformed response must fail on content, never on a skipped request.

    Guards the #1833 contract (raise, never silently return []) while proving the
    zero budget spends its one attempt before failing.
    """
    from hindsight_api.engine.retain.fact_extraction import extract_facts_from_text

    config = retain_config(retain_budget)
    llm = _make_recording_llm(["not a dict"])

    with pytest.raises(RuntimeError, match="non-dict JSON") as exc:
        await extract_facts_from_text(
            text="Alice visited Paris in 2023.",
            event_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
            llm_config=llm,
            agent_name="test-agent",
            config=config,
            context="",
        )

    assert llm.call.call_count == expected_calls
    assert "after 0 attempts" not in str(exc.value)
