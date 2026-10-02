"""Regression tests for portable async-operation document-id persistence."""

import json
import uuid
from contextlib import asynccontextmanager

import pytest

from hindsight_api.engine.retain.orchestrator import (
    _persist_facts_committed_checkpoint,
    _persist_operation_document_id,
)


class FakeOracleConnection:
    backend_type = "oracle"

    def __init__(self, metadata):
        self.metadata = metadata
        self.committed = False
        self.queries = []

    @asynccontextmanager
    async def transaction(self):
        yield self
        self.committed = True

    def parse_json(self, value):
        return value

    async def fetchrow(self, query, operation_id):
        self.queries.append((query, operation_id))
        return {"result_metadata": self.metadata}

    async def execute(self, query, metadata, operation_id):
        self.queries.append((query, metadata, operation_id))
        self.metadata = json.loads(metadata)


@pytest.mark.asyncio
async def test_oracle_persists_document_id_without_postgres_json_operators():
    operation_id = str(uuid.uuid4())
    conn = FakeOracleConnection({"attempt": 2, "document_ids": ["existing"]})

    await _persist_operation_document_id(conn, "async_operations", operation_id, "generated")

    assert conn.committed
    assert conn.metadata == {"attempt": 2, "document_ids": ["existing", "generated"]}
    assert "jsonb_set" not in conn.queries[-1][0]
    assert "FOR UPDATE" in conn.queries[0][0]


@pytest.mark.asyncio
async def test_oracle_document_id_persistence_is_idempotent():
    operation_id = str(uuid.uuid4())
    conn = FakeOracleConnection({"document_ids": ["generated"]})

    await _persist_operation_document_id(conn, "async_operations", operation_id, "generated")

    assert conn.metadata == {"document_ids": ["generated"]}


@pytest.mark.asyncio
async def test_oracle_persists_streaming_checkpoint_without_postgres_json_operators():
    operation_id = str(uuid.uuid4())
    conn = FakeOracleConnection({"attempt": 1, "facts_committed_document_ids": ["doc-a"]})

    await _persist_facts_committed_checkpoint(conn, "async_operations", operation_id, "doc-b", 7)

    assert conn.committed
    assert conn.metadata == {
        "attempt": 1,
        "facts_committed": True,
        "unit_ids_count": 7,
        "facts_committed_document_ids": ["doc-a", "doc-b"],
    }
    assert "jsonb_set" not in conn.queries[-1][0]
    assert "FOR UPDATE" in conn.queries[0][0]


@pytest.mark.asyncio
async def test_oracle_streaming_checkpoint_is_idempotent():
    operation_id = str(uuid.uuid4())
    conn = FakeOracleConnection({"facts_committed_document_ids": ["doc-a"]})

    await _persist_facts_committed_checkpoint(conn, "async_operations", operation_id, "doc-a", 3)

    assert conn.metadata["facts_committed_document_ids"] == ["doc-a"]
    assert conn.metadata["unit_ids_count"] == 3
