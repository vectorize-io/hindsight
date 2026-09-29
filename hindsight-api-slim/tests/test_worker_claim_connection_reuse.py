"""One pooled connection per poll cycle (#3499).

Every ``acquire()`` pays the pool's ``setup=`` callback (the session GUCs) and
every release pays asyncpg's ``RESET ALL`` / ``UNLISTEN *`` / ``CLOSE ALL``.
Behind a transaction-mode pooler each of those is its own server-side
transaction, so acquiring once per *schema* multiplied the ceremony by the
number of active tenants: ~12 statements per schema-visit for 2 useful queries.

These tests pin the acquire count to one per ``claim_batch()`` regardless of how
many schemas have work, and assert the per-schema claims still run in their own
transactions (so ``FOR UPDATE SKIP LOCKED`` semantics are unchanged).
"""

import json
import uuid
from contextlib import asynccontextmanager

import pytest

from hindsight_api.extensions.tenant import Tenant, TenantExtension
from hindsight_api.worker import WorkerPoller


class _StubConnection:
    """Records every statement and transaction opened on it."""

    def __init__(self, recorder: "_StubBackend") -> None:
        self._recorder = recorder

    @asynccontextmanager
    async def transaction(self):
        self._recorder.transactions += 1
        yield self

    async def fetchval(self, query: str, *args, **kwargs):
        self._recorder.statements.append(query)
        # Per-schema EXISTS probe: every schema has work.
        return True

    async def fetch(self, query: str, *args, **kwargs):
        self._recorder.statements.append(query)
        return []


class _StubOps:
    def __init__(self, recorder: "_StubBackend") -> None:
        self._recorder = recorder

    async def claim_tasks(self, conn, table, worker_id, reserved_limits, shared_limit, **kwargs):
        from hindsight_api.engine.db.ops import ClaimedOperations

        self._recorder.claimed_tables.append(table)
        self._recorder.claim_conns.append(conn)
        rows = [
            {
                "operation_id": uuid.uuid4(),
                "operation_type": "test",
                "task_payload": json.dumps({"type": "test_task"}),
                "retry_count": 0,
                # Retain-fold only applies to 'retain' rows with a key; keep this
                # stub out of that path.
                "serialization_key": None,
                "bank_id": "test-bank",
            }
        ]
        # The bank rotation's cursor rides back with the rows (#3861); this stub
        # claims for one bank, so it reports that bank as where it got to.
        return ClaimedOperations(rows=rows, next_bank_cursor="test-bank")


class _StubBackend:
    """Minimal DatabaseBackend surface the poller's claim path touches.

    ``backend_type`` is deliberately not "postgresql" so ``OptionalRoutines``
    short-circuits to the per-schema EXISTS fallback without a pg_proc probe.
    """

    backend_type = "stub"

    def __init__(self) -> None:
        self.acquires = 0
        self.transactions = 0
        self.statements: list[str] = []
        self.claimed_tables: list[str] = []
        self.claim_conns: list[object] = []
        self.ops = _StubOps(self)

    @asynccontextmanager
    async def acquire(self):
        self.acquires += 1
        yield _StubConnection(self)


class _StaticTenants(TenantExtension):
    def __init__(self, schemas: list[str]) -> None:
        self._schemas = schemas

    async def authenticate(self, context):
        raise NotImplementedError("Not used in this test")

    async def list_tenants(self) -> list[Tenant]:
        return [Tenant(schema=s) for s in self._schemas]


def _poller(backend: _StubBackend, schemas: list[str]) -> WorkerPoller:
    # One slot per schema: the fairness pass claims 1 per schema and exhausts
    # capacity exactly, so the capacity backfill pass is a no-op and the visit
    # count is deterministic.
    return WorkerPoller(
        backend=backend,
        worker_id="test-worker",
        executor=lambda task: None,
        tenant_extension=_StaticTenants(schemas),
        max_slots=len(schemas),
        slot_reservations={},
    )


@pytest.mark.asyncio
async def test_claim_batch_acquires_one_connection_for_the_whole_cycle():
    backend = _StubBackend()
    schemas = [f"tenant_{i}" for i in range(8)]

    claimed = await _poller(backend, schemas).claim_batch()

    # Every schema had work and got claimed, on a single pooled connection.
    assert backend.acquires == 1, f"expected 1 acquire per poll cycle, got {backend.acquires}"
    assert len(claimed) == len(schemas)
    assert {t.split(".")[0].strip('"') for t in backend.claimed_tables} == set(schemas)


@pytest.mark.asyncio
async def test_acquire_count_does_not_grow_with_active_schemas():
    few = _StubBackend()
    many = _StubBackend()

    await _poller(few, ["tenant_a"]).claim_batch()
    await _poller(many, [f"tenant_{i}" for i in range(20)]).claim_batch()

    assert few.acquires == many.acquires == 1
    assert len(many.claimed_tables) == 20


@pytest.mark.asyncio
async def test_each_schema_claim_still_runs_in_its_own_transaction():
    backend = _StubBackend()
    schemas = [f"tenant_{i}" for i in range(4)]

    await _poller(backend, schemas).claim_batch()

    # Sharing the connection must not merge the claims into one long
    # transaction: row locks are released as each schema's claim commits.
    assert backend.transactions == len(backend.claimed_tables) == len(schemas)
    # All claims ran on the cycle's single connection.
    assert len({id(c) for c in backend.claim_conns}) == 1
