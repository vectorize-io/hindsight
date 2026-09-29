"""Regression tests for portable async-operation document-id persistence."""

import json
import uuid
from contextlib import asynccontextmanager

import pytest

from hindsight_api.engine.retain.orchestrator import _persist_operation_document_id


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
