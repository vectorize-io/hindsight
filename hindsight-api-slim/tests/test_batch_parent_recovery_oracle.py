"""Native Oracle candidate discovery against a migrated isolated test database."""

import json
import uuid

import pytest

from hindsight_api.engine.db import create_database_backend

pytestmark = pytest.mark.oracle


@pytest.mark.asyncio
@pytest.mark.parametrize("linguistic", [False, True], ids=["binary", "linguistic"])
@pytest.mark.parametrize(
    "relation,child_status,eligible",
    [
        ("canonical", "pending", False),
        ("canonical", "processing", False),
        ("canonical", "cancelled", False),
        ("canonical", "completed", True),
        ("canonical", "failed", True),
        ("uppercase", "pending", True),
        ("array", "pending", True),
        ("root-array", "pending", True),
        ("null", "pending", True),
        ("number", "pending", True),
        ("object", "pending", True),
        ("missing", "pending", True),
    ],
)
async def test_native_candidate_relation(oracle_db_url, relation, child_status, eligible, linguistic):
    backend = create_database_backend("oracle")
    await backend.initialize(oracle_db_url, min_size=1, max_size=2)
    bank_id = f"test-recovery-{uuid.uuid4().hex[:8]}"
    parent_id = uuid.UUID("a" + uuid.uuid4().hex[1:])
    parent_text = str(parent_id)
    values = {
        "canonical": parent_text,
        "uppercase": parent_text.upper(),
        "array": [parent_text],
        "null": None,
        "number": 123,
        "object": {"id": parent_text},
    }
    if relation == "root-array":
        metadata = [{"parent_operation_id": parent_text}]
    else:
        metadata = {} if relation == "missing" else {"parent_operation_id": values[relation]}
    try:
        async with backend.acquire() as conn:
            nls_comp = "LINGUISTIC" if linguistic else "BINARY"
            nls_sort = "BINARY_CI" if linguistic else "BINARY"
            await conn.execute(f"ALTER SESSION SET NLS_COMP={nls_comp}")
            await conn.execute(f"ALTER SESSION SET NLS_SORT={nls_sort}")
            await conn.execute("INSERT INTO banks (bank_id, name) VALUES ($1, $2)", bank_id, bank_id)
            await conn.execute(
                """INSERT INTO async_operations (operation_id, bank_id, operation_type, status)
                   VALUES ($1, $2, 'batch_retain', 'pending')""",
                parent_id,
                bank_id,
            )
            await conn.execute(
                """INSERT INTO async_operations
                       (operation_id, bank_id, operation_type, status, task_payload, result_metadata)
                   VALUES ($1, $2, 'retain', $3, $4, $5)""",
                uuid.uuid4(),
                bank_id,
                child_status,
                "{}",
                json.dumps(metadata),
            )
            candidates = await backend.ops.fetch_reconcilable_batch_parents(conn, "async_operations")
            assert any(row["operation_id"] == parent_id for row in candidates) == eligible
    finally:
        async with backend.acquire() as conn:
            await conn.execute("DELETE FROM async_operations WHERE bank_id = $1", bank_id)
            await conn.execute("DELETE FROM banks WHERE bank_id = $1", bank_id)
        await backend.shutdown()
