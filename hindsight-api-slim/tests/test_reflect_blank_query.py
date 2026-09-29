"""Reflect rejects a blank query instead of burning LLM calls on it (#4416)."""

import httpx
import pytest
import pytest_asyncio
from hindsight_api.api import create_app


@pytest_asyncio.fixture
async def api_client(memory):
    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["", "   ", "\n\t "])
async def test_blank_query_is_rejected(api_client, memory, query, monkeypatch):
    """422 before the agent loop, with no reflect call at all."""

    async def _fail(*args, **kwargs):
        raise AssertionError("reflect_async must not run for a blank query")

    monkeypatch.setattr(memory, "reflect_async", _fail)

    response = await api_client.post(
        "/v1/default/banks/test_blank_query/reflect",
        json={"query": query},
    )

    assert response.status_code == 422
    assert "query" in response.text


@pytest.mark.asyncio
async def test_blank_query_with_context_is_allowed(api_client, memory, monkeypatch):
    """The deprecated context field still carries the question on its own."""
    seen: list[str] = []

    async def _capture(*args, **kwargs):
        seen.append(kwargs["query"])
        raise RuntimeError("stop here")

    monkeypatch.setattr(memory, "reflect_async", _capture)

    response = await api_client.post(
        "/v1/default/banks/test_blank_query/reflect",
        json={"query": "", "context": "What is the deadline?"},
    )

    assert response.status_code == 500
    assert seen and "What is the deadline?" in seen[0]


@pytest.mark.asyncio
async def test_engine_rejects_blank_query(memory):
    """Internal callers (MCP, mental-model refresh) get the same guard."""
    with pytest.raises(ValueError, match="non-empty query"):
        await memory.reflect_async(
            bank_id="test_blank_query",
            query="  ",
            request_context=None,
        )
