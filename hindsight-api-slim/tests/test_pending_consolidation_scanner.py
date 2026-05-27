"""
Regression tests for the pending-consolidation scanner.

The scanner is a periodic backstop that re-queues consolidation for banks with
``memory_units`` rows where ``consolidated_at IS NULL`` and the event-driven
trigger chain (retain success / deletion / manual API call) hasn't already
queued one. It exists so a single failure mode in the trigger chain
(embeddings-provider outage, manual data import that bypasses retain, etc.)
doesn't leave memory_units unconsolidated indefinitely.

These tests verify:

1. Scanner enabled + bank has unconsolidated rows → calls submit_async_consolidation for the bank
2. Scanner reports dedupe correctly when submit returns ``deduplicated=True``
3. Scanner enabled + bank has only consolidated rows → does not call submit
4. Scanner disabled (interval=0) → returns immediately without scanning

Tests mock ``MemoryEngine.submit_async_consolidation`` so we exercise the
scanner's SQL-driven discovery and dispatch logic without entangling them with
the consolidator pipeline (which runs inline under ``SyncTaskBackend`` and
would make assertions about pending-op state non-deterministic).
"""

import asyncio
import uuid

import pytest
import pytest_asyncio

from hindsight_api.extensions.builtin.tenant import DefaultTenantExtension
from hindsight_api.worker.pending_consolidation_scanner import PendingConsolidationScanner

pytestmark = pytest.mark.xdist_group("worker_tests")


async def _ensure_bank(pool, bank_id: str) -> None:
    await pool.execute(
        "INSERT INTO banks (bank_id, name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        bank_id,
        bank_id,
    )


async def _insert_unconsolidated_memory(pool, bank_id: str, fact_type: str = "experience") -> uuid.UUID:
    """Insert a memory_unit row with consolidated_at = NULL."""
    unit_id = uuid.uuid4()
    await pool.execute(
        """
        INSERT INTO memory_units (id, bank_id, text, fact_type, created_at, updated_at)
        VALUES ($1, $2, $3, $4, NOW(), NOW())
        """,
        unit_id,
        bank_id,
        f"test fact {unit_id}",
        fact_type,
    )
    return unit_id


async def _insert_consolidated_memory(pool, bank_id: str, fact_type: str = "experience") -> uuid.UUID:
    """Insert a memory_unit row with consolidated_at set (i.e. already done)."""
    unit_id = uuid.uuid4()
    await pool.execute(
        """
        INSERT INTO memory_units (id, bank_id, text, fact_type, consolidated_at, created_at, updated_at)
        VALUES ($1, $2, $3, $4, NOW(), NOW(), NOW())
        """,
        unit_id,
        bank_id,
        f"test fact {unit_id}",
        fact_type,
    )
    return unit_id


async def _cleanup_bank(pool, bank_id: str) -> None:
    await pool.execute("DELETE FROM async_operations WHERE bank_id = $1", bank_id)
    await pool.execute("DELETE FROM memory_units WHERE bank_id = $1", bank_id)
    await pool.execute("DELETE FROM banks WHERE bank_id = $1", bank_id)


class _SubmitRecorder:
    """Drop-in async replacement for ``MemoryEngine.submit_async_consolidation``
    that records the calls instead of running the consolidator.

    ``deduplicated`` controls the return value so tests can drive the
    scanner's branch that distinguishes "new op queued" from "dedupe hit".
    """

    def __init__(self, *, deduplicated: bool = False) -> None:
        self.deduplicated = deduplicated
        self.calls: list[str] = []

    async def __call__(self, *, bank_id: str, request_context) -> dict:
        self.calls.append(bank_id)
        return {"operation_id": str(uuid.uuid4()), "deduplicated": self.deduplicated}


@pytest_asyncio.fixture
async def scanner(memory):
    """Build a scanner bound to the test's memory engine and default tenant."""
    tenant_extension = DefaultTenantExtension(config={})
    return PendingConsolidationScanner(
        memory=memory,
        tenant_extension=tenant_extension,
        interval_seconds=300,  # value irrelevant for these unit-style tests; we call _scan_once directly
    )


@pytest.mark.asyncio
async def test_scanner_queues_when_no_pending_op(memory, scanner, monkeypatch):
    """A bank with unconsolidated rows triggers exactly one submit call."""
    bank_id = f"test-scanner-{uuid.uuid4().hex[:8]}"
    pool = await memory._get_pool()
    await _ensure_bank(pool, bank_id)
    try:
        await _insert_unconsolidated_memory(pool, bank_id)
        recorder = _SubmitRecorder(deduplicated=False)
        monkeypatch.setattr(memory, "submit_async_consolidation", recorder)

        queued = await scanner._scan_once()

        assert bank_id in recorder.calls, (
            f"scanner did not call submit_async_consolidation for bank {bank_id}; calls={recorder.calls}"
        )
        assert queued >= 1, "scanner should report at least one bank queued"
    finally:
        await _cleanup_bank(pool, bank_id)


@pytest.mark.asyncio
async def test_scanner_reports_dedupe_when_submit_dedupes(memory, scanner, monkeypatch):
    """When submit returns ``deduplicated=True``, the scanner must not count it as queued."""
    bank_id = f"test-scanner-{uuid.uuid4().hex[:8]}"
    pool = await memory._get_pool()
    await _ensure_bank(pool, bank_id)
    try:
        await _insert_unconsolidated_memory(pool, bank_id)
        recorder = _SubmitRecorder(deduplicated=True)
        monkeypatch.setattr(memory, "submit_async_consolidation", recorder)

        queued = await scanner._scan_once()

        assert bank_id in recorder.calls, "scanner must still attempt submit even when dedupe will hit"
        assert queued == 0, f"scanner must not count dedupe hits as queued banks; got queued={queued}"
    finally:
        await _cleanup_bank(pool, bank_id)


@pytest.mark.asyncio
async def test_scanner_skips_fully_consolidated_bank(memory, scanner, monkeypatch):
    """A bank with only consolidated rows must not trigger any submit call."""
    bank_id = f"test-scanner-{uuid.uuid4().hex[:8]}"
    pool = await memory._get_pool()
    await _ensure_bank(pool, bank_id)
    try:
        await _insert_consolidated_memory(pool, bank_id)
        recorder = _SubmitRecorder()
        monkeypatch.setattr(memory, "submit_async_consolidation", recorder)

        await scanner._scan_once()

        assert bank_id not in recorder.calls, (
            f"scanner must not submit for banks whose memory_units are all consolidated; calls={recorder.calls}"
        )
    finally:
        await _cleanup_bank(pool, bank_id)


@pytest.mark.asyncio
async def test_scanner_disabled_short_circuits(memory):
    """When interval_seconds <= 0, ``run()`` must return immediately without scanning."""
    tenant_extension = DefaultTenantExtension(config={})
    disabled_scanner = PendingConsolidationScanner(
        memory=memory,
        tenant_extension=tenant_extension,
        interval_seconds=0,
    )
    assert disabled_scanner.enabled is False

    # run() should return immediately (well under a second) without doing any work.
    # Use a tight timeout to guard against the loop accidentally starting.
    await asyncio.wait_for(disabled_scanner.run(), timeout=2.0)
