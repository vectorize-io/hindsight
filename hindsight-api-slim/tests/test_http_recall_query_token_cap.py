"""The HTTP recall limit must honor the same zero-means-unlimited contract as the engine."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from hindsight_api.api import create_app
from hindsight_api.config import get_config
from hindsight_api.engine.response_models import RecallResult
from hindsight_api.engine.token_encoding import count_tokens


@pytest.mark.parametrize(
    ("limit", "query", "expected_status"),
    [
        pytest.param("0", "hello", 200, id="disabled-short-query"),
        pytest.param("0", "hello " * 600, 200, id="disabled-over-default-limit"),
        pytest.param(None, "hello", 200, id="default-short-query"),
        pytest.param(None, "hello " * 600, 400, id="default-long-query"),
        pytest.param("2", "hello", 200, id="below-positive-limit"),
        pytest.param("1", "hello", 200, id="at-positive-limit"),
        pytest.param("1", "hello world", 400, id="over-positive-limit"),
    ],
)
async def test_http_recall_query_token_cap(
    monkeypatch: pytest.MonkeyPatch, limit: str | None, query: str, expected_status: int
) -> None:
    if limit is None:
        monkeypatch.delenv("HINDSIGHT_API_RECALL_MAX_QUERY_TOKENS", raising=False)
    else:
        monkeypatch.setenv("HINDSIGHT_API_RECALL_MAX_QUERY_TOKENS", limit)
    memory = MagicMock()
    memory._operation_validator = None
    memory.audit_logger = None
    memory.recall_async = AsyncMock(return_value=RecallResult(results=[]))
    app = create_app(memory, initialize_memory=False)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/default/banks/query-cap-test/memories/recall",
            json={"query": query},
        )

    assert response.status_code == expected_status, response.text
    if expected_status == 200:
        memory.recall_async.assert_awaited_once()
        assert memory.recall_async.call_args.kwargs["query"] == query
        assert response.json()["results"] == []
    else:
        memory.recall_async.assert_not_awaited()
        max_query_tokens = get_config().recall_max_query_tokens
        assert response.json()["detail"] == (
            f"Query too long: {count_tokens(query)} tokens exceeds maximum of {max_query_tokens}. "
            "Please shorten your query."
        )
