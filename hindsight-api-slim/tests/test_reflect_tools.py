"""Regression tests for reflect tool helpers."""

import re
import uuid

import pytest

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
