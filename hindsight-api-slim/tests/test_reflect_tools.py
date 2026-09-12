"""Regression tests for reflect tool helpers."""

import re
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

# The mental-model tool cases assert the SQL predicate the tool builds
# (`(tags IS NULL OR tags = '{}')`) off a mocked connection. A store that owns those rows is never
# handed that connection and builds no such query, so the assertion has no subject.
pytestmark = pytest.mark.memory_backend_incompatible

from hindsight_api.engine.reflect.agent import (
    ReflectToolTokenLimits,
    _execute_tool,
    _summarize_input,
    _resolve_tool_arg_ceiling,
)
from hindsight_api.engine.reflect.tools import (
    _document_metadata_from_retain_params,
    tool_expand,
    tool_search_mental_models,
)


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
        # AsyncMock, not MagicMock: the tool awaits the engine's batched staleness
        # check for whatever rows come back, including none.
        memory_engine=AsyncMock(),
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
# Seeds memories through a mocked connection, which a store-owned bank never reads — the
# memories read routes to the store and finds nothing. The store-backed equivalent is
# tests/test_reflect_expand_store_reads.py, which goes through a real retain.
@pytest.mark.memory_backend_incompatible
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
# Seeds memories through a mocked connection, which a store-owned bank never reads — the
# memories read routes to the store and finds nothing. The store-backed equivalent is
# tests/test_reflect_expand_store_reads.py, which goes through a real retain.
@pytest.mark.memory_backend_incompatible
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
# Seeds memories through a mocked connection, which a store-owned bank never reads — the
# memories read routes to the store and finds nothing. The store-backed equivalent is
# tests/test_reflect_expand_store_reads.py, which goes through a real retain.
@pytest.mark.memory_backend_incompatible
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


#: The limits a default deployment resolves to. They used to be constants inlined in
#: ``_execute_tool``; they now travel from bank/env config (#4239), so the tests state
#: them explicitly rather than relying on the call site to know them.
_LIMITS = ReflectToolTokenLimits(
    recall_max_tokens=2048,
    recall_chunk_max_tokens=1000,
    observations_max_tokens=5000,
)

#: The ceiling a call gets with the context budget wide open — the fixed cap.
_CEILING = _resolve_tool_arg_ceiling(None, 1)


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
        _LIMITS,
        _CEILING,
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
        _LIMITS,
        _CEILING,
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
        _LIMITS,
        _CEILING,
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
        _LIMITS,
        _CEILING,
    )
    await _execute_tool(
        "search_observations",
        {"query": "deployment failures", "max_tokens": 0},
        search_mental_models,
        search_observations,
        recall,
        _unexpected_tool_call,
        _LIMITS,
        _CEILING,
    )
    await _execute_tool(
        "recall",
        {"query": "deployment failures", "max_tokens": 0, "max_chunk_tokens": 0},
        search_mental_models,
        search_observations,
        recall,
        _unexpected_tool_call,
        _LIMITS,
        _CEILING,
    )
    await _execute_tool(
        "search_observations",
        {"query": "deployment failures", "max_tokens": False},
        search_mental_models,
        search_observations,
        recall,
        _unexpected_tool_call,
        _LIMITS,
        _CEILING,
    )
    await _execute_tool(
        "search_observations",
        {"query": "deployment failures", "max_tokens": []},
        search_mental_models,
        search_observations,
        recall,
        _unexpected_tool_call,
        _LIMITS,
        _CEILING,
    )
    await _execute_tool(
        "search_observations",
        {"query": "deployment failures", "max_tokens": {}},
        search_mental_models,
        search_observations,
        recall,
        _unexpected_tool_call,
        _LIMITS,
        _CEILING,
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
        _LIMITS,
        _CEILING,
    )

    assert result == {"memories": []}
    assert captured == {"query": "incident notes", "max_tokens": 2048, "max_chunk_tokens": 1000}


def test_summarize_input_never_raises_for_invalid_tool_limit_strings() -> None:
    """Trace logging must not turn a recoverable tool error into HTTP 500."""
    assert (
        _summarize_input(
            "search_observations",
            {"query": "deployment failures", "max_tokens": "None"},
            _LIMITS,
            _CEILING,
        )
        == "(query='deployment failures', max_tokens=5000)"
    )

    assert (
        _summarize_input(
            "recall",
            {"query": "deployment failures", "max_tokens": "bogus", "max_chunk_tokens": "null"},
            _LIMITS,
            _CEILING,
        )
        == "(query='deployment failures', max_tokens=invalid:'bogus', max_chunk_tokens=1000)"
    )

    assert (
        _summarize_input(
            "search_observations",
            {"query": "deployment failures", "max_tokens": float("inf")},
            _LIMITS,
            _CEILING,
        )
        == "(query='deployment failures', max_tokens=invalid:inf)"
    )

    assert (
        _summarize_input(
            "search_observations",
            {"query": None, "max_tokens": "None"},
            _LIMITS,
            _CEILING,
        )
        == "(query='', max_tokens=5000)"
    )


# ---------------------------------------------------------------------------
# Token-limit resolution on the agent path (#4239)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_tool_defaults_come_from_configured_limits() -> None:
    """A bank/env-configured recall budget must reach the agent's recall tool.

    The regression: ``_execute_tool`` parsed the token arguments with constants
    (2048 / 1000 / 5000) and passed them positionally, so the closure defaults bound
    from ``recall_max_tokens`` / ``recall_chunks_max_tokens`` -- and the per-mental-model
    ``trigger.recall_max_tokens`` override -- were never reached. Configuring them
    changed nothing on any reflect.
    """
    limits = ReflectToolTokenLimits(
        recall_max_tokens=6000,
        recall_chunk_max_tokens=2500,
        observations_max_tokens=7000,
    )
    captured: dict[str, object] = {}

    async def recall(query: str, max_tokens: int, max_chunk_tokens: int) -> dict[str, object]:
        captured["recall_max_tokens"] = max_tokens
        captured["recall_max_chunk_tokens"] = max_chunk_tokens
        return {"memories": []}

    async def search_observations(query: str, max_tokens: int) -> dict[str, object]:
        captured["observation_max_tokens"] = max_tokens
        return {"observations": []}

    await _execute_tool(
        "recall",
        {"query": "incident notes"},
        _unexpected_tool_call,
        search_observations,
        recall,
        _unexpected_tool_call,
        limits,
        _CEILING,
    )
    await _execute_tool(
        "search_observations",
        {"query": "incident notes"},
        _unexpected_tool_call,
        search_observations,
        recall,
        _unexpected_tool_call,
        limits,
        _CEILING,
    )

    assert captured == {
        "recall_max_tokens": 6000,
        "recall_max_chunk_tokens": 2500,
        "observation_max_tokens": 7000,
    }


@pytest.mark.asyncio
async def test_execute_tool_caps_model_requested_token_arguments() -> None:
    """The ceiling bounds what the model asks for, mirroring the existing floor.

    The tool schemas tell the model to "use higher values for broader searches" and
    nothing bounded the ask, so one call could pull in more than the whole reflect
    context budget.
    """
    limits = ReflectToolTokenLimits(
        recall_max_tokens=2048,
        recall_chunk_max_tokens=1000,
        observations_max_tokens=5000,
    )
    captured: dict[str, object] = {}

    async def recall(query: str, max_tokens: int, max_chunk_tokens: int) -> dict[str, object]:
        captured["max_tokens"] = max_tokens
        captured["max_chunk_tokens"] = max_chunk_tokens
        return {"memories": []}

    await _execute_tool(
        "recall",
        {"query": "everything", "max_tokens": 200000, "max_chunk_tokens": 50000},
        _unexpected_tool_call,
        _unexpected_tool_call,
        recall,
        _unexpected_tool_call,
        limits,
        _CEILING,
    )

    assert captured == {"max_tokens": _CEILING, "max_chunk_tokens": _CEILING}


@pytest.mark.asyncio
async def test_execute_tool_ceiling_never_overrides_the_floor() -> None:
    """A ceiling squeezed below the floor still yields a usable request."""
    limits = ReflectToolTokenLimits(
        recall_max_tokens=2048,
        recall_chunk_max_tokens=1000,
        observations_max_tokens=5000,
    )
    captured: dict[str, object] = {}

    async def search_observations(query: str, max_tokens: int) -> dict[str, object]:
        captured["max_tokens"] = max_tokens
        return {"observations": []}

    await _execute_tool(
        "search_observations",
        {"query": "everything", "max_tokens": 50000},
        _unexpected_tool_call,
        search_observations,
        _unexpected_tool_call,
        _unexpected_tool_call,
        limits,
        200,  # remaining context budget nearly exhausted
    )

    assert captured == {"max_tokens": 1000}


def test_configured_default_above_the_ceiling_is_honoured() -> None:
    """The ceiling caps the model, not the operator.

    A deployment that deliberately sets ``recall_max_tokens`` above the fixed cap gets
    it; the cap exists to stop the *model* reaching for a broader search than the
    context budget can hold.
    """
    limits = ReflectToolTokenLimits(
        recall_max_tokens=40000,
        recall_chunk_max_tokens=1000,
        observations_max_tokens=5000,
    )
    assert (
        _summarize_input("recall", {"query": "q"}, limits, _CEILING)
        == "(query='q', max_tokens=40000, max_chunk_tokens=1000)"
    )


@pytest.mark.parametrize(
    "remaining,calls,expected",
    [
        (None, 3, 16000),  # no budget known -> the fixed cap only
        (90000, 1, 16000),  # plenty left -> the fixed cap still binds
        (30000, 3, 10000),  # shared across the batch in flight
        (12000, 5, 2400),  # tight budget tightens every call
        (900, 2, 1000),  # never below the floor
        (-5000, 2, 1000),  # already over budget -> floor, not a negative ask
    ],
)
def test_ceiling_shares_the_remaining_context_budget(remaining, calls, expected) -> None:
    """A single iteration's parallel calls must not be able to ask for more than fits.

    The loop checks the accumulated context only between iterations, so before this
    an iteration could overshoot by its whole combined payload and force the slow
    split-synthesis path.
    """
    assert _resolve_tool_arg_ceiling(remaining, calls) == expected
