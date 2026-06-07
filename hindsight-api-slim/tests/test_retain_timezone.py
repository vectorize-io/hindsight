"""Regression tests for retain timezone handling."""

import os
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from hindsight_api import LLMConfig
from hindsight_api.api.http import RetainRequest
from hindsight_api.config import DEFAULT_LLM_PROVIDER, ENV_LLM_API_KEY, ENV_LLM_PROVIDER, _get_raw_config
from hindsight_api.engine.llm_wrapper import requires_api_key
from hindsight_api.engine.retain.fact_extraction import _build_user_message, extract_facts_from_text
from hindsight_api.engine.retain.orchestrator import _build_contents, parse_datetime_flexible
from tests.llm_judge import assert_meets_criteria


def test_parse_datetime_flexible_converts_aware_values_to_utc() -> None:
    """Offset-aware retain timestamps must be converted, not relabelled, as UTC."""
    expected = datetime(2026, 6, 5, 12, 36, tzinfo=UTC)

    assert parse_datetime_flexible("2026-06-05T20:36:00+08:00") == expected
    assert parse_datetime_flexible(datetime(2026, 6, 5, 20, 36, tzinfo=ZoneInfo("Asia/Shanghai"))) == expected


def test_retain_request_accepts_and_validates_client_timezone() -> None:
    request = RetainRequest.model_validate(
        {"items": [{"content": "Alice joined the call at 8:36 PM.", "client_timezone": " Asia/Shanghai "}]}
    )

    assert request.items[0].client_timezone == "Asia/Shanghai"

    with pytest.raises(ValidationError, match="Invalid client_timezone"):
        RetainRequest.model_validate(
            {"items": [{"content": "Alice joined the call at 8:36 PM.", "client_timezone": "Mars/Phobos"}]}
        )


def test_build_contents_carries_client_timezone_and_normalizes_event_date() -> None:
    contents = _build_contents(
        [
            {
                "content": "Alice joined the call at 8:36 PM.",
                "event_date": "2026-06-05T20:36:00+08:00",
                "client_timezone": "Asia/Shanghai",
            }
        ],
        document_tags=None,
    )

    assert contents[0].event_date == datetime(2026, 6, 5, 12, 36, tzinfo=UTC)
    assert contents[0].client_timezone == "Asia/Shanghai"

    with pytest.raises(ValueError, match="Invalid client_timezone"):
        _build_contents([{"content": "Alice joined the call.", "client_timezone": "Mars/Phobos"}], None)


def test_build_user_message_uses_client_timezone_for_event_date() -> None:
    msg = _build_user_message(
        chunk="Alice joined the call at 8:36 PM.",
        chunk_index=0,
        total_chunks=1,
        event_date=datetime(2026, 6, 5, 12, 36, tzinfo=UTC),
        context="support chat",
        client_timezone="Asia/Shanghai",
    )

    assert "Event Date: Friday, June 05, 2026 (2026-06-05T20:36:00+08:00" in msg
    assert "UTC instant 2026-06-05T12:36:00+00:00" in msg
    assert "Client Timezone: Asia/Shanghai (UTC+08:00)" in msg
    assert "use this local timezone" in msg


@pytest.mark.asyncio
@pytest.mark.hs_llm_core
@pytest.mark.flaky(reruns=2, reruns_delay=2)
async def test_extract_facts_uses_client_timezone_for_relative_dates() -> None:
    """Real extraction should resolve relative dates against the client's local day."""
    provider = os.getenv(ENV_LLM_PROVIDER, DEFAULT_LLM_PROVIDER)
    if not os.getenv(ENV_LLM_API_KEY) and requires_api_key(provider):
        pytest.skip(f"{ENV_LLM_API_KEY} is required for real LLM extraction")

    facts, _, _ = await extract_facts_from_text(
        text="Today I paid the January rent after breakfast.",
        event_date=datetime(2025, 12, 31, 16, 30, tzinfo=UTC),
        context="personal journal entry written just after midnight in Shanghai",
        llm_config=LLMConfig.from_env(),
        agent_name="TestUser",
        config=_get_raw_config(),
        client_timezone="Asia/Shanghai",
    )

    assert facts, "Should extract at least one fact"
    extracted = "\n".join(
        f"fact={fact.fact}; occurred_start={fact.occurred_start}; occurred_end={fact.occurred_end}"
        for fact in facts
    )

    await assert_meets_criteria(
        response=extracted,
        criteria=(
            "The extraction treats 'today' as January 1, 2026 in Asia/Shanghai for the rent payment. "
            "It must not resolve the event as December 31, 2025 just because the storage/reference "
            "instant is in UTC."
        ),
        context=(
            "The event instant is 2025-12-31T16:30:00+00:00, which is "
            "2026-01-01T00:30:00+08:00 in Asia/Shanghai. The input says: "
            "'Today I paid the January rent after breakfast.'"
        ),
        msg=f"Client timezone should drive relative-date extraction. Facts: {extracted}",
    )
