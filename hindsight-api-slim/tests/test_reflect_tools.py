"""Regression tests for reflect tool helpers."""

import json
import re
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api.engine.reflect.agent import _execute_tool, _summarize_input
from hindsight_api.engine.reflect.models import SlimMemoryFact
from hindsight_api.engine.reflect.tokenization import count_cl100k_tokens
from hindsight_api.engine.reflect.tools import (
    _document_metadata_from_retain_params,
    tool_expand,
    tool_recall,
    tool_search_mental_models,
    tool_search_observations,
)
from hindsight_api.engine.response_models import ChunkInfo, MemoryFact, RecallResult
from hindsight_api.models import RequestContext


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


class _FakeMultiMemoryConnection:
    """Serves memory_units rows for whichever ids the batch query asks for."""

    def __init__(self, texts: dict[uuid.UUID, str]) -> None:
        self.texts = texts

    async def fetch(self, query: str, *args):
        normalized_query = re.sub(r"\s+", " ", query).strip()

        if "FROM public.memory_units" in normalized_query:
            requested = args[0]
            return [
                {
                    "id": memory_id,
                    "text": self.texts[memory_id],
                    "chunk_id": None,
                    "document_id": None,
                    "fact_type": "experience",
                    "context": "preference",
                }
                for memory_id in requested
                if memory_id in self.texts
            ]

        if "FROM public.chunks" in normalized_query or "FROM public.documents" in normalized_query:
            return []

        raise AssertionError(f"Unexpected query: {normalized_query}")


@pytest.mark.asyncio
@pytest.mark.parametrize("tags", [None, []], ids=["omitted", "empty"])
async def test_tool_search_mental_models_exact_empty_scope_selects_global_models(
    tags: list[str] | None,
) -> None:
    """An exact empty scope must exclude tagged mental models."""
    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=[])

    result = await tool_search_mental_models(
        memory_engine=MagicMock(),
        conn=conn,
        bank_id="test-global-mental-model-scope",
        query="global summary",
        query_embedding=[0.1, 0.2],
        tags=tags,
        tags_match="exact",
    )

    fetch_call = conn.fetch.await_args
    assert fetch_call is not None
    query, *args = fetch_call.args
    assert "(tags IS NULL OR tags = '{}')" in re.sub(r"\s+", " ", query).strip()
    assert args == ["test-global-mental-model-scope", "[0.1, 0.2]", 5]
    assert result["mental_models"] == []


@pytest.mark.asyncio
async def test_tool_expand_pairs_each_memory_id_with_its_own_memory() -> None:
    """An invalid memory_id must not shift the memories returned for the ids after it."""
    first = uuid.uuid4()
    second = uuid.uuid4()
    texts = {first: "The user prefers dark mode.", second: "Dogs are loyal animals."}
    conn = _FakeMultiMemoryConnection(texts)

    result = await tool_expand(
        conn=conn,
        bank_id="test-reflect-expand-pairing",
        memory_ids=["not-a-uuid", str(first), str(second)],
        depth="chunk",
    )

    assert result["count"] == 3
    invalid, first_item, second_item = result["results"]

    assert invalid["memory_id"] == "not-a-uuid"
    assert "Invalid memory_id format" in invalid["error"]

    assert first_item["memory_id"] == str(first)
    assert first_item["memory"]["id"] == str(first)
    assert first_item["memory"]["text"] == texts[first]

    assert second_item["memory_id"] == str(second)
    assert second_item["memory"]["id"] == str(second)
    assert second_item["memory"]["text"] == texts[second]


@pytest.mark.asyncio
async def test_tool_expand_reports_a_trailing_invalid_memory_id() -> None:
    """Every requested memory_id gets an entry, including an invalid one listed last."""
    memory_id = uuid.uuid4()
    conn = _FakeMultiMemoryConnection({memory_id: "The user prefers dark mode."})

    result = await tool_expand(
        conn=conn,
        bank_id="test-reflect-expand-trailing-invalid",
        memory_ids=[str(memory_id), "bad"],
        depth="chunk",
    )

    assert result["count"] == 2
    assert [item["memory_id"] for item in result["results"]] == [str(memory_id), "bad"]
    assert "Invalid memory_id format" in result["results"][1]["error"]


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


class TestToolResultLlmBudget:
    """Reflect tool results are budgeted by what their LLM-facing rendering
    actually costs.

    recall_async budgets result selection by fact-text tokens only, but for
    short facts the serialized rendering the model sees is dominated by ids and
    timestamps, so a text-token budget under-counts by an order of magnitude
    and a single tool call can consume most of the reflect context window.
    """

    @staticmethod
    def _short_fact(i: int, fact_type: str = "world") -> MemoryFact:
        return MemoryFact(
            id=str(uuid.uuid4()),
            text=f"User prefers option {i}.",
            fact_type=fact_type,
            entities=["User"],
            context="preference chat",
            occurred_start="2026-07-01T00:00:00+00:00",
            occurred_end="2026-07-01T00:00:00+00:00",
            mentioned_at="2026-07-02T00:00:00+00:00",
            document_id=str(uuid.uuid4()),
            metadata={"source": "chat"},
            chunk_id=f"chunk-{i}",
            tags=["pref"],
        )

    @staticmethod
    def _llm_rendering_tokens(items: list[dict]) -> int:
        rendered = json.dumps([SlimMemoryFact.project(m) for m in items], default=str, ensure_ascii=False)
        return count_cl100k_tokens(rendered)

    @pytest.mark.asyncio
    async def test_recall_results_fit_llm_rendering_budget(self):
        """With hundreds of short facts, tool_recall returns only as many as fit
        max_tokens when measured on the slim rendering the model actually sees."""
        facts = [self._short_fact(i) for i in range(300)]
        engine = MagicMock()
        engine.recall_async = AsyncMock(return_value=RecallResult(results=facts, source_facts={}))

        out = await tool_recall(engine, "bank-1", "prefs", RequestContext(), max_tokens=1000)

        assert len(out["memories"]) >= 1
        assert len(out["memories"]) < 300, "no trimming happened: every short fact was returned"
        assert self._llm_rendering_tokens(out["memories"]) <= 1000

    @pytest.mark.asyncio
    async def test_recall_chunks_pruned_to_kept_memories(self):
        """When trimming cuts memories, the chunks map keeps only entries the
        surviving memories reference — dropped memories' chunks would otherwise
        leak the very tokens the trim removed."""
        facts = [self._short_fact(i) for i in range(300)]
        chunks = {f"chunk-{i}": ChunkInfo(chunk_text=f"chunk text {i}", chunk_index=i) for i in range(300)}
        engine = MagicMock()
        engine.recall_async = AsyncMock(return_value=RecallResult(results=facts, source_facts={}, chunks=chunks))

        out = await tool_recall(engine, "bank-1", "prefs", RequestContext(), max_tokens=1000)

        kept_chunk_ids = {m["chunk_id"] for m in out["memories"]}
        assert set(out["chunks"]) == kept_chunk_ids
        assert 0 < len(out["chunks"]) < 300

    @pytest.mark.asyncio
    async def test_search_observations_fit_budget_with_pruned_source_facts_and_count(self):
        """Observation results are budget-fitted the same way; count reports what
        is actually returned, and source_facts keeps only facts the surviving
        observations reference."""
        observations = []
        source_facts = {}
        for i in range(300):
            obs = self._short_fact(i, fact_type="observation")
            obs.source_fact_ids = [f"src-{i}"]
            observations.append(obs)
            source_facts[f"src-{i}"] = MemoryFact(id=f"src-{i}", text=f"source fact {i}", fact_type="world")
        engine = MagicMock()
        engine.recall_async = AsyncMock(
            return_value=RecallResult(results=observations, source_facts=source_facts)
        )

        out = await tool_search_observations(
            engine, "bank-1", "prefs", RequestContext(), max_tokens=1000, source_facts_max_tokens=0
        )

        assert 1 <= len(out["observations"]) < 300
        assert self._llm_rendering_tokens(out["observations"]) <= 1000
        assert out["count"] == len(out["observations"])
        kept_source_ids = {sid for o in out["observations"] for sid in o.get("source_fact_ids", [])}
        assert set(out["source_facts"]) == kept_source_ids
