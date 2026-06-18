"""Provider quota reset windows defer worker retries instead of failing retains."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import APIStatusError

from hindsight_api.engine.llm_interface import ProviderRateLimitResetError
from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM


def _llm() -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        provider="zai",
        model="glm-5-turbo",
        api_key="test",
        base_url="https://example.com/v1",
    )


def _usage_limit_error(reset_at: str) -> APIStatusError:
    body = {
        "code": "1308",
        "message": f"Usage limit reached for 5 hour. Your limit will reset at {reset_at}",
    }
    response = MagicMock()
    response.status_code = 429
    response.text = '{"code": "1308", "message": "usage limit"}'
    response.headers = {}
    return APIStatusError("rate limited", response=response, body=body)


def _short_retry_after_error() -> APIStatusError:
    response = MagicMock()
    response.status_code = 429
    response.text = '{"code": "rate_limit", "message": "retry shortly"}'
    response.headers = {"retry-after": "1"}
    return APIStatusError("rate limited", response=response, body={"message": "retry shortly"})


@pytest.mark.asyncio
async def test_usage_limit_429_with_reset_defers_without_inner_retry() -> None:
    llm = _llm()
    reset_at = (datetime.now(UTC) + timedelta(hours=5)).replace(microsecond=0)
    create = AsyncMock(side_effect=_usage_limit_error(reset_at.isoformat().replace("+00:00", "Z")))
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    with patch(
        "hindsight_api.engine.providers.openai_compatible_llm.asyncio.sleep",
        new_callable=AsyncMock,
    ) as sleep:
        with pytest.raises(ProviderRateLimitResetError) as exc_info:
            await llm.call(
                messages=[{"role": "user", "content": "x"}],
                scope="retain_extract_facts",
                max_retries=2,
            )

    assert create.await_count == 1
    sleep.assert_not_awaited()
    assert abs((exc_info.value.retry_at - reset_at).total_seconds()) < 1
    assert "Provider quota exhausted" in str(exc_info.value)


@pytest.mark.asyncio
async def test_short_retry_after_429_uses_normal_retry_loop() -> None:
    llm = _llm()
    create = AsyncMock(side_effect=_short_retry_after_error())
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    with patch(
        "hindsight_api.engine.providers.openai_compatible_llm.asyncio.sleep",
        new_callable=AsyncMock,
    ) as sleep:
        with pytest.raises(APIStatusError):
            await llm.call(
                messages=[{"role": "user", "content": "x"}],
                scope="retain_extract_facts",
                max_retries=2,
                max_backoff=60,
            )

    assert create.await_count == 3
    assert sleep.await_count == 2


@pytest.mark.asyncio
async def test_extract_facts_from_text_preserves_provider_quota_reset(monkeypatch) -> None:
    from hindsight_api.engine.retain import fact_extraction

    retry_at = (datetime.now(UTC) + timedelta(hours=2)).replace(microsecond=0)

    async def quota_limited_chunk(**_: object) -> None:
        raise ProviderRateLimitResetError(retry_at=retry_at, message="quota resets later")

    monkeypatch.setattr(fact_extraction, "_extract_facts_with_auto_split", quota_limited_chunk)

    with pytest.raises(ProviderRateLimitResetError) as exc_info:
        await fact_extraction.extract_facts_from_text(
            text="Alice moved to Berlin.",
            event_date=None,
            llm_config=object(),
            agent_name="TestAgent",
            config=SimpleNamespace(retain_chunk_size=1000, retain_structured_chunk_size=None),
        )

    assert exc_info.value.retry_at == retry_at
    assert "Fact extraction deferred by provider quota" in str(exc_info.value)
