"""
Unit tests for fact extraction retry logic.

When the LLM returns non-dict JSON across all retries, extraction must raise a
RuntimeError (issue #1833 — never silently return [] and let the retain commit
the document with 0 facts). This also guards the original TypeError bug: the
raise must be a real exception, not `raise None` ('exceptions must derive from
BaseException'), which happened when last_error was only set in the
BadRequestError handler and not for non-dict JSON responses.
"""

import dataclasses
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def test_output_retry_split_preserves_conversation_array_boundaries():
    """OutputTooLong retry splitting must keep conversation chunks valid JSON arrays."""
    from hindsight_api.engine.retain.fact_extraction import _split_chunk_for_output_retry

    # Contents are padded so the array clears the minimum-split floor and the
    # boundary-preserving branch (not the drop-when-too-small guard) is exercised.
    turns = [
        {"role": "user", "content": "alpha " * 40},
        {"role": "assistant", "content": "bravo " * 40},
        {"role": "user", "content": "charlie " * 40},
        {"role": "assistant", "content": "delta " * 40},
    ]

    split = _split_chunk_for_output_retry(json.dumps(turns))

    assert split is not None
    first, second = split
    assert json.loads(first) == turns[:2]
    assert json.loads(second) == turns[2:]


def test_output_retry_split_divides_single_oversized_turn_content():
    """A lone oversized conversation turn is split inside content and rewrapped."""
    from hindsight_api.engine.retain.fact_extraction import _split_chunk_for_output_retry

    # Content must exceed the minimum-split floor for the single-turn branch to
    # divide it rather than drop the sub-chunk outright.
    turn = {"role": "user", "content": "abcdefghijklmnopqrstuvwxyz" * 40, "name": "casey"}

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


def test_output_too_long_error_is_a_single_class_across_modules():
    """Regression for #3172.

    ``OutputTooLongError`` must be one class everywhere: the providers raise the
    ``llm_interface`` definition, and ``fact_extraction`` / ``multi_llm`` catch
    the name they import from ``llm_wrapper``. If ``llm_wrapper`` shadows it with
    a second definition, ``except OutputTooLongError`` silently stops matching
    what the providers raise and the #2579 auto-split becomes dead code.
    """
    from hindsight_api.engine import llm_interface, llm_wrapper, multi_llm
    from hindsight_api.engine.providers import litellm_llm, openai_compatible_llm
    from hindsight_api.engine.retain import fact_extraction

    canonical = llm_interface.OutputTooLongError
    assert llm_wrapper.OutputTooLongError is canonical
    assert fact_extraction.OutputTooLongError is canonical
    assert multi_llm.OutputTooLongError is canonical
    assert litellm_llm.OutputTooLongError is canonical
    assert openai_compatible_llm.OutputTooLongError is canonical


def test_output_retry_split_drops_subchunk_below_minimum_floor():
    """A chunk at/under the minimum-split floor is dropped, not recursively halved.

    Without this floor a chunk that overflows the output cap at *every* size
    (degenerate/looping model output) would recurse until it is a single
    character, burning thousands of extraction calls (#3172).
    """
    from hindsight_api.engine.retain.fact_extraction import (
        _MIN_SPLIT_CHUNK_CHARS,
        _split_chunk_for_output_retry,
    )

    assert _split_chunk_for_output_retry("a" * _MIN_SPLIT_CHUNK_CHARS) is None
    # Just over the floor still splits.
    assert _split_chunk_for_output_retry("a. " * ((_MIN_SPLIT_CHUNK_CHARS // 3) + 5)) is not None


@pytest.mark.asyncio
async def test_output_too_long_fails_unsplittable_subchunk_without_recursing():
    """If a chunk cannot be reduced further, auto-split fails without dropping it."""
    from hindsight_api.engine.llm_wrapper import OutputTooLongError
    from hindsight_api.engine.retain.fact_extraction import _extract_facts_with_auto_split

    config = _make_config(llm_max_retries=1)
    llm_config = _make_llm_config(mock_response={})

    with patch(
        "hindsight_api.engine.retain.fact_extraction._extract_facts_from_chunk",
        side_effect=OutputTooLongError("too long"),
    ) as extract:
        with pytest.raises(RuntimeError, match="refusing to drop this sub-chunk"):
            await _extract_facts_with_auto_split(
                chunk="x",
                chunk_index=0,
                total_chunks=1,
                event_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
                context="",
                llm_config=llm_config,
                config=config,
                agent_name="agent",
            )

    assert extract.call_count == 1


@pytest.mark.asyncio
async def test_output_too_long_propagates_when_one_half_cannot_be_split():
    """A failing half must fail the chunk, not return the other half's facts."""
    from hindsight_api.engine.llm_wrapper import OutputTooLongError
    from hindsight_api.engine.retain.fact_extraction import TokenUsage, _extract_facts_with_auto_split

    long_turn = {"role": "user", "content": "alpha " * 300}
    tiny_turn = {"role": "assistant", "content": "b"}
    chunk = json.dumps([long_turn, tiny_turn])

    async def _extract(*, chunk: str, **_kwargs):
        # Only the half holding solely the user turn succeeds. Anything still
        # carrying the assistant turn overflows, so the whole chunk splits first
        # and the tiny assistant half then fails unsplittably — below the split
        # floor, so it cannot be reduced further. Keyed on content, not call
        # order, so gather scheduling cannot flip the test.
        if "assistant" not in chunk:
            return [{"fact": "kept"}], TokenUsage()
        raise OutputTooLongError("too long")

    config = _make_config(llm_max_retries=0)
    llm_config = _make_llm_config(mock_response={})

    with patch(
        "hindsight_api.engine.retain.fact_extraction._extract_facts_from_chunk",
        side_effect=_extract,
    ):
        # Returning the successful half's facts here would report the retain
        # complete with the assistant turn silently missing, which is the
        # failure mode this whole path exists to prevent.
        with pytest.raises(RuntimeError, match="refusing to drop this sub-chunk"):
            await _extract_facts_with_auto_split(
                chunk=chunk,
                chunk_index=0,
                total_chunks=1,
                event_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
                context="",
                llm_config=llm_config,
                config=config,
                agent_name="agent",
            )


def _make_config(llm_max_retries: int = 3, retain_llm_max_retries: int | None = None):
    """Build a minimal HindsightConfig for fact extraction tests."""
    from hindsight_api.config import _get_raw_config

    return dataclasses.replace(
        _get_raw_config(),
        retain_llm_max_retries=retain_llm_max_retries,
        llm_max_retries=llm_max_retries,
        retain_llm_initial_backoff=None,
        llm_initial_backoff=0.0,
        retain_llm_max_backoff=None,
        llm_max_backoff=0.0,
        retain_max_completion_tokens=8192,
        retain_extraction_mode="concise",
        retain_extract_causal_links=False,
        retain_mission=None,
        llm_temperature_retain=0.1,
        llm_strict_schema_retain=False,
    )


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
@pytest.mark.parametrize(
    ("fact_fields", "expected_count", "expected_text"),
    [
        ({"text": "Alice visited Paris"}, 1, "Alice visited Paris"),
        ({"what": "Alice visited Paris"}, 1, "Alice visited Paris"),
        ({}, 0, None),
    ],
)
async def test_fact_text_alias_is_recovered_without_accepting_empty_facts(fact_fields, expected_count, expected_text):
    """Recover schema-drifted ``text`` facts while still skipping empty facts."""
    from hindsight_api.engine.retain.fact_extraction import _extract_facts_from_chunk

    config = _make_config(llm_max_retries=0, retain_llm_max_retries=None)
    llm_config = _make_llm_config(
        mock_response=[
            {
                **fact_fields,
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
            chunk="Alice visited Paris.",
            chunk_index=0,
            total_chunks=1,
            event_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
            context="travel notes",
            llm_config=llm_config,
            config=config,
            agent_name="test-agent",
        )

    assert len(facts) == expected_count
    if expected_text:
        assert expected_text in facts[0].fact


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
    cfg.llm_strict_schema_retain = False
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


def test_build_request_body_uses_retain_strict_schema_flag_for_schema_and_request():
    """The batch retain path must use one resolved strict-schema flag consistently."""
    from hindsight_api.engine.retain.fact_extraction import _build_request_body

    config = _make_batch_temp_config(None)
    config.llm_strict_schema = False
    config.llm_strict_schema_retain = True
    response_schema = MagicMock()
    response_schema.model_json_schema.return_value = {"schema": "non-strict"}

    with patch(
        "hindsight_api.engine.retain.fact_extraction.strict_json_schema",
        return_value={"schema": "strict"},
    ) as strict_schema:
        body = _build_request_body(_make_batch_llm_config(), config, "sys", "user", response_schema)

    strict_schema.assert_called_once_with(response_schema)
    assert body["response_format"]["json_schema"]["schema"] == {"schema": "strict"}
    assert body["response_format"]["json_schema"]["strict"] is True


def test_build_request_body_retain_strict_false_overrides_global_true():
    """The retain opt-out must disable strict schema in the batch request body."""
    from hindsight_api.engine.retain.fact_extraction import _build_request_body

    config = _make_batch_temp_config(None)
    config.llm_strict_schema = True
    config.llm_strict_schema_retain = False
    response_schema = MagicMock()
    response_schema.model_json_schema.return_value = {"schema": "non-strict"}

    with patch("hindsight_api.engine.retain.fact_extraction.strict_json_schema") as strict_schema:
        body = _build_request_body(_make_batch_llm_config(), config, "sys", "user", response_schema)

    strict_schema.assert_not_called()
    assert body["response_format"]["json_schema"]["schema"] == {"schema": "non-strict"}
    assert body["response_format"]["json_schema"]["strict"] is False


@pytest.mark.asyncio
async def test_fact_saturation_boundary_allows_a_response_below_the_bound():
    """A response below the saturation boundary remains a normal success."""
    from hindsight_api.engine.retain.fact_extraction import _extract_facts_from_chunk, resolve_facts_saturation_limit

    config = _make_config(llm_max_retries=0)
    fact_count = resolve_facts_saturation_limit(config) - 1
    llm_config = _make_llm_config(mock_response={"facts": [{"what": f"fact {index}"} for index in range(fact_count)]})

    with patch(
        "hindsight_api.engine.retain.fact_extraction._build_extraction_prompt_and_schema",
        return_value=("system prompt", MagicMock()),
    ):
        facts, _usage = await _extract_facts_from_chunk(
            chunk="A normal chunk.",
            chunk_index=0,
            total_chunks=1,
            event_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
            context="",
            llm_config=llm_config,
            config=config,
            agent_name="agent",
        )

    assert len(facts) == fact_count
    assert llm_config.call.call_count == 1


@pytest.mark.asyncio
async def test_fact_saturation_boundary_raises_output_too_long():
    """A response at the saturation boundary must enter the split path."""
    from hindsight_api.engine.llm_wrapper import OutputTooLongError
    from hindsight_api.engine.retain.fact_extraction import _extract_facts_from_chunk, resolve_facts_saturation_limit

    config = _make_config(llm_max_retries=0)
    saturation_limit = resolve_facts_saturation_limit(config)
    llm_config = _make_llm_config(
        mock_response={"facts": [{"what": f"fact {index}"} for index in range(saturation_limit)]}
    )

    with patch(
        "hindsight_api.engine.retain.fact_extraction._build_extraction_prompt_and_schema",
        return_value=("system prompt", MagicMock()),
    ):
        with pytest.raises(OutputTooLongError, match="saturation limit"):
            await _extract_facts_from_chunk(
                chunk="A dense chunk.",
                chunk_index=0,
                total_chunks=1,
                event_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
                context="",
                llm_config=llm_config,
                config=config,
                agent_name="agent",
            )

    assert llm_config.call.call_count == 1


@pytest.mark.asyncio
async def test_fact_saturation_uses_auto_split_and_returns_subchunk_facts():
    """Saturation must split the chunk and return facts from both sub-chunks."""
    from hindsight_api.engine.response_models import TokenUsage
    from hindsight_api.engine.retain.fact_extraction import (
        _extract_facts_with_auto_split,
        resolve_facts_saturation_limit,
    )

    config = _make_config(llm_max_retries=0)
    llm_config = _make_llm_config(mock_response=None)
    llm_config.call = AsyncMock(
        side_effect=[
            (
                {
                    "facts": [
                        {"what": f"saturated fact {index}"} for index in range(resolve_facts_saturation_limit(config))
                    ]
                },
                TokenUsage(),
            ),
            ({"facts": [{"what": "left fact"}]}, TokenUsage()),
            ({"facts": [{"what": "right fact"}]}, TokenUsage()),
        ]
    )
    chunk = "A sentence that needs extraction. " * 40

    with patch(
        "hindsight_api.engine.retain.fact_extraction._build_extraction_prompt_and_schema",
        return_value=("system prompt", MagicMock()),
    ):
        facts, _usage = await _extract_facts_with_auto_split(
            chunk=chunk,
            chunk_index=0,
            total_chunks=1,
            event_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
            context="",
            llm_config=llm_config,
            config=config,
            agent_name="agent",
        )

    assert [fact.fact for fact in facts] == ["left fact", "right fact"]
    assert llm_config.call.call_count == 3


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


def test_facts_saturation_limit_stays_inside_the_calibration_window():
    """The bound must clear real density and still land under the token ceiling.

    Both edges have bitten. An earlier revision halved the budget and produced 24
    at a 4096-token ceiling, below the 29 facts seen in the field, so honest input
    was split for nothing. Going the other way and exceeding the token-limited
    count makes the schema bound unreachable, leaving the array to run to
    finish_reason=length — the case the bound exists to prevent.
    """
    from hindsight_api.engine.retain.fact_extraction import (
        _APPROX_OUTPUT_TOKENS_PER_FACT,
        resolve_facts_saturation_limit,
    )

    OBSERVED_MAX_FACTS = 29

    for budget in (4096, 8192, 16384, 65536):
        config = MagicMock()
        config.retain_max_completion_tokens = budget
        limit = resolve_facts_saturation_limit(config)
        assert limit > OBSERVED_MAX_FACTS, f"budget {budget} would split honest input at {limit} facts"
        assert limit <= budget // _APPROX_OUTPUT_TOKENS_PER_FACT, f"budget {budget} advertises an unreachable bound"


def test_facts_saturation_limit_never_exceeds_a_small_budget():
    """A budget too small for the floor gets the token-limited count, not the floor."""
    from hindsight_api.engine.retain.fact_extraction import (
        _APPROX_OUTPUT_TOKENS_PER_FACT,
        RETAIN_FACTS_SATURATION_FLOOR,
        resolve_facts_saturation_limit,
    )

    config = MagicMock()
    config.retain_max_completion_tokens = 1024
    limit = resolve_facts_saturation_limit(config)

    assert limit < RETAIN_FACTS_SATURATION_FLOOR
    assert limit == 1024 // _APPROX_OUTPUT_TOKENS_PER_FACT


def test_facts_saturation_limit_scales_with_output_budget():
    """The saturation bound tracks the configured output ceiling."""
    from hindsight_api.engine.retain.fact_extraction import (
        RETAIN_FACTS_SATURATION_FLOOR,
        resolve_facts_saturation_limit,
    )

    small = MagicMock()
    small.retain_max_completion_tokens = 4096
    large = MagicMock()
    large.retain_max_completion_tokens = 65536

    assert resolve_facts_saturation_limit(small) < resolve_facts_saturation_limit(large)
    assert resolve_facts_saturation_limit(small) >= RETAIN_FACTS_SATURATION_FLOOR

