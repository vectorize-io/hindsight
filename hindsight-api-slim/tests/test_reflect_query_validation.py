"""Reject blank reflect input before it can consume provider calls (#4416)."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from hindsight_api import MemoryEngine, RequestContext
from hindsight_api.api.http import create_app
from hindsight_api.engine.response_models import ReflectResult


@pytest.mark.parametrize("query", ["", "   ", "\t\r\n", "\u00a0\u2003"])
def test_http_rejects_blank_query_before_calling_engine(query: str) -> None:
    memory = MagicMock(spec=MemoryEngine)
    memory.audit_logger = None
    memory.reflect_async = AsyncMock(return_value=ReflectResult(text="answer", based_on={}))
    client = TestClient(create_app(memory, initialize_memory=False))

    response = client.post("/v1/default/banks/test/reflect", json={"query": query})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "query"]
    memory.reflect_async.assert_not_called()


@pytest.mark.parametrize("query", ["Where does Alice live?", "  Where does Alice live?\n", "你好？", "?"])
def test_http_preserves_nonblank_query(query: str) -> None:
    memory = MagicMock(spec=MemoryEngine)
    memory.audit_logger = None
    memory.reflect_async = AsyncMock(return_value=ReflectResult(text="answer", based_on={}))
    client = TestClient(create_app(memory, initialize_memory=False))

    response = client.post("/v1/default/banks/test/reflect", json={"query": query})

    assert response.status_code == 200, response.text
    assert response.json()["text"] == "answer"
    assert memory.reflect_async.call_args.kwargs["query"] == query


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["", "   ", "\t\n", "\u00a0\u2003", "\ud800"])
async def test_engine_rejects_blank_query_before_config_or_io(query: str) -> None:
    # No configured dependencies: validation must precede database/provider work,
    # including text that becomes empty after ingress sanitization.
    memory = MemoryEngine.__new__(MemoryEngine)

    with pytest.raises(ValueError, match="query must not be empty or whitespace-only"):
        await memory.reflect_async(bank_id="test", query=query, request_context=RequestContext())
