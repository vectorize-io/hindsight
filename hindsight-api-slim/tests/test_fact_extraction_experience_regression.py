from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.config import HindsightConfig, _get_raw_config
from hindsight_api.engine.response_models import TokenUsage
from hindsight_api.engine.retain.fact_extraction import (
    Fact,
    RetainContent,
    extract_facts_from_contents,
    extract_facts_from_contents_batch_api,
)


@pytest.mark.asyncio
async def test_extract_facts_from_contents_preserves_experience_fact_type():
    """
    The extractor already normalizes raw assistant facts to "experience".
    The final ExtractedFactType conversion must preserve that normalized type.
    """
    contents = [
        RetainContent(
            content="I fixed the failing tests after discovering they mocked the wrong interface.",
            event_date=datetime(2026, 4, 1, tzinfo=timezone.utc),
            context="assistant work log",
        )
    ]
    extracted_fact = Fact(
        fact="Assistant fixed the failing tests after discovering they mocked the wrong interface.",
        fact_type="experience",
    )

    with patch(
        "hindsight_api.engine.retain.fact_extraction.extract_facts_from_text",
        new=AsyncMock(return_value=([extracted_fact], [(contents[0].content, 1)], TokenUsage())),
    ):
        facts, chunks, usage = await extract_facts_from_contents(
            contents=contents,
            llm_config=None,
            agent_name="TestAgent",
            config=_get_raw_config(),
        )

    assert len(facts) == 1
    assert len(chunks) == 1
    assert usage.total_tokens == 0
    assert facts[0].fact_type == "experience"


@pytest.mark.asyncio
async def test_extract_facts_from_contents_batch_api_preserves_experience_fact_type():
    """
    Batch extraction normalizes raw "assistant" facts before conversion.
    The batch conversion layer must not remap the normalized type to "world".
    """
    llm_config = MagicMock()
    llm_config.provider = "openai"
    llm_config.model = "gpt-4.1-mini"
    llm_config._provider_impl = AsyncMock()
    llm_config._provider_impl.submit_batch = AsyncMock(
        return_value={
            "batch_id": "batch_experience_123",
            "status": "validating",
            "request_counts": {"total": 1, "completed": 0, "failed": 0},
        }
    )
    llm_config._provider_impl.get_batch_status = AsyncMock(
        return_value={"status": "completed", "request_counts": {"total": 1, "completed": 1, "failed": 0}}
    )
    llm_config._provider_impl.retrieve_batch_results = AsyncMock(
        return_value=[
            {
                "custom_id": "chunk_0",
                "response": {
                    "body": {
                        "choices": [
                            {
                                "message": {
                                    "content": (
                                        '{"facts":[{"what":"Assistant fixed the failing tests after discovering they '
                                        'mocked the wrong interface","when":"2026-04-01","where":"N/A",'
                                        '"who":"assistant","why":"Tests were failing due to incorrect mocking",'
                                        '"fact_type":"assistant","fact_kind":"event","occurred_start":'
                                        '"2026-04-01T00:00:00+00:00","occurred_end":"2026-04-01T00:00:00+00:00"}]}'
                                    )
                                }
                            }
                        ],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    }
                },
            }
        ]
    )

    facts, chunks, usage = await extract_facts_from_contents_batch_api(
        contents=[
            RetainContent(
                content="I fixed the failing tests after discovering they mocked the wrong interface.",
                event_date=datetime(2026, 4, 1, tzinfo=timezone.utc),
                context="assistant work log",
            )
        ],
        llm_config=llm_config,
        agent_name="TestAgent",
        config=HindsightConfig.from_env(),
        pool=None,
        operation_id=None,
        schema=None,
    )

    assert len(facts) == 1
    assert len(chunks) == 1
    assert usage.total_tokens == 15
    assert facts[0].fact_type == "experience"
