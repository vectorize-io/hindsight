"""
Tests for the 'none' LLM provider mode.

Verifies that when HINDSIGHT_API_LLM_PROVIDER=none:
- Retain defaults to chunks mode (no LLM calls)
- Reflect returns 400
- Mental model refresh returns 400
- Consolidation is skipped
- NoneLLM.call() raises LLMNotAvailableError
"""

import os
from datetime import datetime, timezone

import httpx
import pytest
import pytest_asyncio

from hindsight_api import LLMConfig, LocalSTEmbeddings, MemoryEngine, RequestContext
from hindsight_api.api import create_app
from hindsight_api.engine.cross_encoder import LocalSTCrossEncoder
from hindsight_api.engine.memory_engine import Budget
from hindsight_api.engine.providers.none_llm import LLMNotAvailableError, NoneLLM
from hindsight_api.engine.query_analyzer import DateparserQueryAnalyzer
from hindsight_api.engine.task_backend import SyncTaskBackend


@pytest.fixture(scope="function")
def request_context():
    return RequestContext()


@pytest_asyncio.fixture(scope="function")
async def none_memory(pg0_db_url, embeddings, cross_encoder, query_analyzer):
    """MemoryEngine with provider=none."""
    mem = MemoryEngine(
        db_url=pg0_db_url,
        memory_llm_provider="none",
        memory_llm_api_key=None,
        memory_llm_model="none",
        embeddings=embeddings,
        cross_encoder=cross_encoder,
        query_analyzer=query_analyzer,
        pool_min_size=1,
        pool_max_size=5,
        run_migrations=False,
        task_backend=SyncTaskBackend(),
    )
    await mem.initialize()
    yield mem
    try:
        if mem._pool and not mem._pool._closing:
            await mem.close()
    except Exception:
        pass


@pytest_asyncio.fixture
async def none_api_client(none_memory):
    """HTTP test client backed by a none-provider MemoryEngine."""
    app = create_app(none_memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# -- Unit tests for NoneLLM ---------------------------------------------------


@pytest.mark.asyncio
async def test_none_llm_call_raises():
    """NoneLLM.call() should raise LLMNotAvailableError."""
    llm = NoneLLM(provider="none", api_key="", base_url="", model="none")
    with pytest.raises(LLMNotAvailableError):
        await llm.call(messages=[{"role": "user", "content": "hello"}])


@pytest.mark.asyncio
async def test_none_llm_call_with_tools_raises():
    """NoneLLM.call_with_tools() should raise LLMNotAvailableError."""
    llm = NoneLLM(provider="none", api_key="", base_url="", model="none")
    with pytest.raises(LLMNotAvailableError):
        await llm.call_with_tools(
            messages=[{"role": "user", "content": "hello"}],
            tools=[{"type": "function", "function": {"name": "test", "parameters": {}}}],
        )


@pytest.mark.asyncio
async def test_none_llm_verify_connection_succeeds():
    """NoneLLM.verify_connection() should be a no-op."""
    llm = NoneLLM(provider="none", api_key="", base_url="", model="none")
    await llm.verify_connection()  # Should not raise


# -- Config validation tests ---------------------------------------------------


def test_config_forces_chunks_mode():
    """When provider is 'none', config.validate() forces retain_extraction_mode='chunks'."""
    from hindsight_api.config import HindsightConfig

    config = HindsightConfig.from_env()
    # Override to none for test
    config.llm_provider = "none"
    config.retain_extraction_mode = "facts"
    config.enable_observations = True
    config.validate()

    assert config.retain_extraction_mode == "chunks"
    assert config.enable_observations is False


# -- Integration tests (require database) -------------------------------------


@pytest.mark.asyncio
async def test_retain_works_with_none_provider(none_memory, request_context):
    """Retain should work with provider=none, storing chunks without LLM calls."""
    bank_id = f"test_none_retain_{datetime.now(timezone.utc).timestamp()}"

    unit_ids = await none_memory.retain_async(
        bank_id=bank_id,
        content="Alice is a software engineer. She works at TechCorp and loves Python.",
        context="team info",
        request_context=request_context,
    )

    assert len(unit_ids) > 0, "Should store chunks even without an LLM"


@pytest.mark.asyncio
async def test_recall_works_with_none_provider(none_memory, request_context):
    """Recall should work with provider=none (uses embeddings, not LLM)."""
    bank_id = f"test_none_recall_{datetime.now(timezone.utc).timestamp()}"

    await none_memory.retain_async(
        bank_id=bank_id,
        content="Alice is a software engineer at TechCorp.",
        context="team info",
        request_context=request_context,
    )

    result = await none_memory.recall_async(
        bank_id=bank_id,
        query="Who is Alice?",
        budget=Budget.LOW,
        request_context=request_context,
    )

    assert len(result.results) > 0, "Should find results via semantic search"


@pytest.mark.asyncio
async def test_reflect_raises_with_none_provider(none_memory, request_context):
    """Reflect should raise LLMNotAvailableError with provider=none."""
    bank_id = f"test_none_reflect_{datetime.now(timezone.utc).timestamp()}"

    with pytest.raises(LLMNotAvailableError):
        await none_memory.reflect_async(
            bank_id=bank_id,
            query="What do you know?",
            request_context=request_context,
        )


@pytest.mark.asyncio
async def test_consolidation_skipped_with_none_provider(none_memory, request_context):
    """Consolidation handler should skip when provider=none."""
    result = await none_memory._handle_consolidation({"bank_id": "test_bank"})
    assert result["skipped"] is True
    assert result["memories_processed"] == 0


@pytest.mark.asyncio
async def test_mental_model_refresh_raises_with_none_provider(none_memory, request_context):
    """Mental model refresh should raise LLMNotAvailableError with provider=none."""
    bank_id = f"test_none_mm_{datetime.now(timezone.utc).timestamp()}"

    with pytest.raises(LLMNotAvailableError):
        await none_memory.submit_async_refresh_mental_model(
            bank_id=bank_id,
            mental_model_id="fake-id",
            request_context=request_context,
        )


# -- HTTP API tests -----------------------------------------------------------


@pytest.mark.asyncio
async def test_http_reflect_returns_400(none_api_client):
    """Reflect endpoint should return 400 when LLM provider is none."""
    bank_id = f"test_none_http_{datetime.now(timezone.utc).timestamp()}"

    response = await none_api_client.post(
        f"/v1/default/banks/{bank_id}/reflect",
        json={"query": "What do you know?"},
    )
    assert response.status_code == 400
    assert "none" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_http_retain_works(none_api_client):
    """Retain endpoint should work with provider=none (chunks mode)."""
    bank_id = f"test_none_http_retain_{datetime.now(timezone.utc).timestamp()}"

    response = await none_api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={"items": [{"content": "Hello world", "context": "test"}]},
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_http_recall_works(none_api_client):
    """Recall endpoint should work with provider=none."""
    bank_id = f"test_none_http_recall_{datetime.now(timezone.utc).timestamp()}"

    # Retain first
    await none_api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={"items": [{"content": "Alice is an engineer.", "context": "test"}]},
    )

    # Recall
    response = await none_api_client.post(
        f"/v1/default/banks/{bank_id}/memories/recall",
        json={"query": "Alice"},
    )
    assert response.status_code == 200
