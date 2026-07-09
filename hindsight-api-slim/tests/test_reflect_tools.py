"""Regression tests for reflect tool helpers."""

import re
import uuid

import pytest

from hindsight_api.engine.reflect.agent import _execute_tool, _summarize_input
from hindsight_api.engine.reflect.tools import _document_metadata_from_retain_params, tool_expand


class _FakeReflectConnection:
    """Tiny asyncpg-like connection for tool_expand query behavior."""

    def __init__(self, bank_id: str, memory_id: uuid.UUID, document_id: str, chunk_id: str | None) -> None:
        self.bank_id = bank_id
        self.memory_id = memory_id
        self.document_id = document_id
        self.chunk_id = chunk_id

    async def fetch(self, query: str, *args):
        normalized_query = re.sub(r"\s+", " ", query).strip()

        if "FROM public.memory_units" in normalized_query:
            return [
                {
                    "id": self.memory_id,
                    "text": "The user prefers test-first bug fixes.",
                    "chunk_id": self.chunk_id,
                    "document_id": self.document_id,
                    "fact_type": "experience",
                    "context": "preference",
                }
            ]

        if "FROM public.chunks" in normalized_query:
            if self.chunk_id is None:
                return []
            return [
                {
                    "chunk_id": self.chunk_id,
                    "chunk_text": "The user prefers test-first bug fixes.",
                    "chunk_index": 0,
                    "document_id": self.document_id,
                }
            ]

        if "FROM public.documents" in normalized_query:
            select_clause = normalized_query.split(" FROM ", 1)[0]
            assert " metadata," not in f" {select_clause},", (
                "tool_expand must not query documents.metadata; that column was removed and "
                "document metadata now lives in retain_params.metadata"
            )
            return [
                {
                    "id": self.document_id,
                    "original_text": "The user prefers test-first bug fixes.",
                    "retain_params": {"metadata": {"source": "regression-test"}},
                }
            ]

        raise AssertionError(f"Unexpected query: {normalized_query}")


@pytest.mark.asyncio
async def test_tool_expand_document_depth_reads_metadata_from_retain_params() -> None:
    """Document expansion must work after documents.metadata has been dropped."""
    bank_id = "test-reflect-expand-retain-params-metadata"
    memory_id = uuid.uuid4()
    document_id = "doc-reflect-expand"
    chunk_id = "chunk-reflect-expand"
    conn = _FakeReflectConnection(bank_id, memory_id, document_id, chunk_id)

    result = await tool_expand(
        conn=conn,
        bank_id=bank_id,
        memory_ids=[str(memory_id)],
        depth="document",
    )

    assert result["count"] == 1
    document = result["results"][0]["document"]
    assert document["metadata"] == {"source": "regression-test"}
    assert document["retain_params"] == {"metadata": {"source": "regression-test"}}


@pytest.mark.asyncio
async def test_tool_expand_document_depth_without_chunk_reads_metadata_from_retain_params() -> None:
    """Direct document expansion follows the same metadata source contract."""
    bank_id = "test-reflect-expand-direct-retain-params-metadata"
    memory_id = uuid.uuid4()
    document_id = "doc-reflect-expand-direct"
    conn = _FakeReflectConnection(bank_id, memory_id, document_id, chunk_id=None)

    result = await tool_expand(
        conn=conn,
        bank_id=bank_id,
        memory_ids=[str(memory_id)],
        depth="document",
    )

    assert result["count"] == 1
    document = result["results"][0]["document"]
    assert document["metadata"] == {"source": "regression-test"}
    assert document["retain_params"] == {"metadata": {"source": "regression-test"}}


def test_document_metadata_from_retain_params_accepts_json_strings() -> None:
    """asyncpg JSONB codecs may return retain_params as a dict or JSON string."""
    retain_params = '{"metadata": {"source": "json-string"}}'

    assert _document_metadata_from_retain_params(retain_params) == {"source": "json-string"}


@pytest.mark.parametrize(
    "retain_params",
    [None, [], "not json", {"metadata": ["not", "a", "dict"]}],
)
def test_document_metadata_from_retain_params_ignores_invalid_values(retain_params) -> None:
    """Malformed retain_params should not break reflect expansion."""
    assert _document_metadata_from_retain_params(retain_params) is None


async def _unexpected_tool_call(*_args):
    raise AssertionError("unexpected tool callback")


@pytest.mark.asyncio
async def test_execute_tool_treats_string_none_max_tokens_as_default() -> None:
    """Some providers emit JSON null as the string "None" in tool calls."""
    captured: dict[str, object] = {}

    async def search_observations(query: str, max_tokens: int) -> dict[str, object]:
        captured["query"] = query
        captured["max_tokens"] = max_tokens
        return {"observations": []}

    result = await _execute_tool(
        "search_observations",
        {"query": "deployment failures", "max_tokens": "None"},
        _unexpected_tool_call,
        search_observations,
        _unexpected_tool_call,
        _unexpected_tool_call,
    )

    assert result == {"observations": []}
    assert captured == {"query": "deployment failures", "max_tokens": 5000}


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_limit", ["bogus", float("inf")])
async def test_execute_tool_returns_error_for_invalid_integer_limit(bad_limit) -> None:
    """Malformed integer limits should be a tool error, not an exception."""
    result = await _execute_tool(
        "search_observations",
        {"query": "deployment failures", "max_tokens": bad_limit},
        _unexpected_tool_call,
        _unexpected_tool_call,
        _unexpected_tool_call,
        _unexpected_tool_call,
    )

    assert result == {"error": "max_tokens must be an integer or null-like value"}


@pytest.mark.asyncio
async def test_execute_tool_preserves_search_mental_models_max_results_values() -> None:
    """Only token limits have minimums; max_results keeps the prior pass-through behavior."""
    captured: dict[str, object] = {}

    async def search_mental_models(query: str, max_results: int) -> dict[str, object]:
        captured["query"] = query
        captured["max_results"] = max_results
        return {"mental_models": []}

    result = await _execute_tool(
        "search_mental_models",
        {"query": "deployment failures", "max_results": -1},
        search_mental_models,
        _unexpected_tool_call,
        _unexpected_tool_call,
        _unexpected_tool_call,
    )

    assert result == {"mental_models": []}
    assert captured == {"query": "deployment failures", "max_results": -1}


@pytest.mark.asyncio
async def test_execute_tool_preserves_falsey_values_as_default_sentinel() -> None:
    """The old `or default` behavior treated falsey limit values as omitted."""
    captured: dict[str, object] = {}

    async def search_mental_models(query: str, max_results: int) -> dict[str, object]:
        captured["mental_model_max_results"] = max_results
        return {"mental_models": []}

    async def search_observations(query: str, max_tokens: int) -> dict[str, object]:
        captured["observation_max_tokens"] = max_tokens
        return {"observations": []}

    async def recall(query: str, max_tokens: int, max_chunk_tokens: int) -> dict[str, object]:
        captured["recall_max_tokens"] = max_tokens
        captured["recall_max_chunk_tokens"] = max_chunk_tokens
        return {"memories": []}

    await _execute_tool(
        "search_mental_models",
        {"query": "deployment failures", "max_results": 0},
        search_mental_models,
        search_observations,
        recall,
        _unexpected_tool_call,
    )
    await _execute_tool(
        "search_observations",
        {"query": "deployment failures", "max_tokens": 0},
        search_mental_models,
        search_observations,
        recall,
        _unexpected_tool_call,
    )
    await _execute_tool(
        "recall",
        {"query": "deployment failures", "max_tokens": 0, "max_chunk_tokens": 0},
        search_mental_models,
        search_observations,
        recall,
        _unexpected_tool_call,
    )
    await _execute_tool(
        "search_observations",
        {"query": "deployment failures", "max_tokens": False},
        search_mental_models,
        search_observations,
        recall,
        _unexpected_tool_call,
    )
    await _execute_tool(
        "search_observations",
        {"query": "deployment failures", "max_tokens": []},
        search_mental_models,
        search_observations,
        recall,
        _unexpected_tool_call,
    )
    await _execute_tool(
        "search_observations",
        {"query": "deployment failures", "max_tokens": {}},
        search_mental_models,
        search_observations,
        recall,
        _unexpected_tool_call,
    )

    assert captured == {
        "mental_model_max_results": 5,
        "observation_max_tokens": 5000,
        "recall_max_tokens": 2048,
        "recall_max_chunk_tokens": 1000,
    }


@pytest.mark.asyncio
async def test_execute_tool_treats_null_like_recall_limits_as_defaults() -> None:
    """Null-like string limits should not crash reflect tool execution."""
    captured: dict[str, object] = {}

    async def recall(query: str, max_tokens: int, max_chunk_tokens: int) -> dict[str, object]:
        captured["query"] = query
        captured["max_tokens"] = max_tokens
        captured["max_chunk_tokens"] = max_chunk_tokens
        return {"memories": []}

    result = await _execute_tool(
        "recall",
        {"query": "incident notes", "max_tokens": "null", "max_chunk_tokens": ""},
        _unexpected_tool_call,
        _unexpected_tool_call,
        recall,
        _unexpected_tool_call,
    )

    assert result == {"memories": []}
    assert captured == {"query": "incident notes", "max_tokens": 2048, "max_chunk_tokens": 1000}


def test_summarize_input_never_raises_for_invalid_tool_limit_strings() -> None:
    """Trace logging must not turn a recoverable tool error into HTTP 500."""
    assert (
        _summarize_input(
            "search_observations",
            {"query": "deployment failures", "max_tokens": "None"},
        )
        == "(query='deployment failures', max_tokens=5000)"
    )

    assert (
        _summarize_input(
            "recall",
            {"query": "deployment failures", "max_tokens": "bogus", "max_chunk_tokens": "null"},
        )
        == "(query='deployment failures', max_tokens=invalid:'bogus', max_chunk_tokens=1000)"
    )

    assert (
        _summarize_input(
            "search_observations",
            {"query": "deployment failures", "max_tokens": float("inf")},
        )
        == "(query='deployment failures', max_tokens=invalid:inf)"
    )

    assert (
        _summarize_input(
            "search_observations",
            {"query": None, "max_tokens": "None"},
        )
        == "(query='', max_tokens=5000)"
    )
