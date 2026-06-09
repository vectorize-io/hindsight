"""Tests for consolidation failure handling: adaptive batch splitting, consolidation_failed_at,
and the recovery API.

These tests use a mock LLM to simulate LLM failures deterministically, without making real
API calls. All tests insert memories directly into the database to bypass retain's LLM calls
and focus exclusively on the consolidation code paths.
"""

import uuid
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from hindsight_api.engine.consolidation.consolidator import run_consolidation_job
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.providers.mock_llm import MockLLM
from hindsight_api.engine.task_backend import SyncTaskBackend


@pytest_asyncio.fixture(scope="function")
async def memory_no_llm_verify(pg0_db_url, embeddings, cross_encoder, query_analyzer):
    """MemoryEngine with mock LLM.

    Migrations are already applied by the session-scoped pg0_db_url fixture, so
    run_migrations=False avoids advisory-lock serialization overhead per test.
    """
    mem = MemoryEngine(
        db_url=pg0_db_url,
        memory_llm_provider="mock",
        memory_llm_api_key="",
        memory_llm_model="mock",
        embeddings=embeddings,
        cross_encoder=cross_encoder,
        query_analyzer=query_analyzer,
        pool_min_size=1,
        pool_max_size=5,
        run_migrations=False,
        task_backend=SyncTaskBackend(),
        skip_llm_verification=True,
    )
    await mem.initialize()
    yield mem
    try:
        if mem._pool and not mem._pool._closing:
            await mem.close()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def enable_observations():
    """Enable observations for all tests in this module."""
    from hindsight_api.config import _get_raw_config

    config = _get_raw_config()
    original = config.enable_observations
    config.enable_observations = True
    yield
    config.enable_observations = original


def _make_failing_mock_llm(*, fail_first_n: int = 999) -> MockLLM:
    """Return a MockLLM that raises ValueError for the first `fail_first_n` consolidation calls."""
    mock_llm = MockLLM(provider="mock", api_key="", base_url="", model="mock-model")
    call_count = 0

    def callback(messages, scope):
        nonlocal call_count
        if scope == "consolidation":
            call_count += 1
            if call_count <= fail_first_n:
                raise ValueError(f"Simulated LLM failure (call {call_count})")
        # Return empty response — no creates/updates/deletes
        from hindsight_api.engine.consolidation.consolidator import _ConsolidationBatchResponse

        return _ConsolidationBatchResponse()

    mock_llm.set_response_callback(callback)
    return mock_llm


def _make_always_success_mock_llm() -> MockLLM:
    """Return a MockLLM that always succeeds with an empty consolidation response."""
    mock_llm = MockLLM(provider="mock", api_key="", base_url="", model="mock-model")

    def callback(messages, scope):
        from hindsight_api.engine.consolidation.consolidator import _ConsolidationBatchResponse

        return _ConsolidationBatchResponse()

    mock_llm.set_response_callback(callback)
    return mock_llm


def _inject_mock_llm(memory: MemoryEngine, mock_llm: MockLLM) -> None:
    """Replace memory._consolidation_llm_config with a wrapper that returns mock_llm from with_config."""
    wrapper = MagicMock()
    wrapper.with_config.return_value = mock_llm
    memory._consolidation_llm_config = wrapper


async def _insert_memories(conn, bank_id: str, texts: list[str]) -> list[uuid.UUID]:
    """Insert experience memories directly, bypassing LLM-based retain."""
    ids = []
    for text in texts:
        mem_id = uuid.uuid4()
        await conn.execute(
            """
            INSERT INTO memory_units (id, bank_id, text, fact_type, created_at)
            VALUES ($1, $2, $3, 'experience', now())
            """,
            mem_id,
            bank_id,
            text,
        )
        ids.append(mem_id)
    return ids


class TestAdaptiveBatchSplitting:
    """Verify that a failing batch is halved and retried until batch_size=1 succeeds."""

    @pytest.mark.asyncio
    async def test_splitting_recovers_all_memories(self, memory_no_llm_verify: MemoryEngine, request_context):
        """When a batch of 2 fails, both are retried individually and succeed."""
        bank_id = f"test-split-recovery-{uuid.uuid4().hex[:8]}"
        await memory_no_llm_verify.get_bank_profile(bank_id=bank_id, request_context=request_context)

        async with memory_no_llm_verify._pool.acquire() as conn:
            mem_ids = await _insert_memories(
                conn,
                bank_id,
                [
                    "Alice runs marathons every spring.",
                    "Alice trained for six months for her last race.",
                ],
            )

        # Exhaust all 3 retries for batch=2 (calls 1-3 fail), then each batch=1 succeeds (calls 4-5)
        mock_llm = _make_failing_mock_llm(fail_first_n=3)
        _inject_mock_llm(memory_no_llm_verify, mock_llm)

        result = await run_consolidation_job(
            memory_engine=memory_no_llm_verify,
            bank_id=bank_id,
            request_context=request_context,
        )

        assert result["status"] == "completed"
        assert result["memories_processed"] == 2
        assert result["memories_failed"] == 0

        # Both memories must have consolidated_at set and consolidation_failed_at NULL
        async with memory_no_llm_verify._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, consolidated_at, consolidation_failed_at
                FROM memory_units
                WHERE bank_id = $1 AND fact_type = 'experience'
                """,
                bank_id,
            )
        assert len(rows) == 2
        for row in rows:
            assert row["consolidated_at"] is not None, f"Memory {row['id']} should have consolidated_at set"
            assert row["consolidation_failed_at"] is None, (
                f"Memory {row['id']} should NOT have consolidation_failed_at set"
            )

        # LLM called 5 times: 3 retries failed (batch=2) + 1 succeeded (batch=1) + 1 succeeded (batch=1)
        consolidation_calls = [c for c in mock_llm.get_mock_calls() if c["scope"] == "consolidation"]
        assert len(consolidation_calls) == 5

        await memory_no_llm_verify.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_splitting_with_larger_batch(self, memory_no_llm_verify: MemoryEngine, request_context):
        """A batch of 4 that always fails at size>1 resolves to 4 individual calls."""
        bank_id = f"test-split-large-{uuid.uuid4().hex[:8]}"
        await memory_no_llm_verify.get_bank_profile(bank_id=bank_id, request_context=request_context)

        async with memory_no_llm_verify._pool.acquire() as conn:
            await _insert_memories(
                conn,
                bank_id,
                [
                    "Bob plays chess competitively.",
                    "Bob won a regional chess tournament.",
                    "Bob practices tactics every morning.",
                    "Bob coaches youth chess on weekends.",
                ],
            )

        # Exhaust all 3 retries for batch=4 (calls 1-3 fail), then both batch=2 halves succeed
        # (calls 4-5). This verifies that halving once is sufficient when batch=2 works.
        mock_llm = _make_failing_mock_llm(fail_first_n=3)
        _inject_mock_llm(memory_no_llm_verify, mock_llm)

        result = await run_consolidation_job(
            memory_engine=memory_no_llm_verify,
            bank_id=bank_id,
            request_context=request_context,
        )

        assert result["memories_processed"] == 4
        assert result["memories_failed"] == 0

        async with memory_no_llm_verify._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT consolidated_at, consolidation_failed_at FROM memory_units "
                "WHERE bank_id = $1 AND fact_type = 'experience'",
                bank_id,
            )
        assert all(r["consolidated_at"] is not None for r in rows)
        assert all(r["consolidation_failed_at"] is None for r in rows)

        await memory_no_llm_verify.delete_bank(bank_id, request_context=request_context)


class TestConsolidationFailedAt:
    """Verify that consolidation_failed_at is set — and consolidated_at is NOT — when all retries fail."""

    @pytest.mark.asyncio
    async def test_single_memory_permanent_failure(self, memory_no_llm_verify: MemoryEngine, request_context):
        """A single memory that exhausts all LLM retries gets consolidation_failed_at, not consolidated_at."""
        bank_id = f"test-perm-fail-{uuid.uuid4().hex[:8]}"
        await memory_no_llm_verify.get_bank_profile(bank_id=bank_id, request_context=request_context)

        async with memory_no_llm_verify._pool.acquire() as conn:
            (mem_id,) = await _insert_memories(conn, bank_id, ["Carol enjoys painting watercolors."])

        # Always fail
        mock_llm = _make_failing_mock_llm(fail_first_n=999)
        _inject_mock_llm(memory_no_llm_verify, mock_llm)

        result = await run_consolidation_job(
            memory_engine=memory_no_llm_verify,
            bank_id=bank_id,
            request_context=request_context,
        )

        assert result["memories_failed"] == 1
        assert result["memories_processed"] == 1

        async with memory_no_llm_verify._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT consolidated_at, consolidation_failed_at FROM memory_units WHERE id = $1",
                mem_id,
            )

        assert row["consolidated_at"] is None, "consolidated_at must NOT be set for a permanently failed memory"
        assert row["consolidation_failed_at"] is not None, "consolidation_failed_at must be set"

        await memory_no_llm_verify.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_failed_memory_excluded_from_next_run(self, memory_no_llm_verify: MemoryEngine, request_context):
        """A memory marked consolidation_failed_at is not re-processed on the next consolidation run."""
        bank_id = f"test-excluded-{uuid.uuid4().hex[:8]}"
        await memory_no_llm_verify.get_bank_profile(bank_id=bank_id, request_context=request_context)

        async with memory_no_llm_verify._pool.acquire() as conn:
            (mem_id,) = await _insert_memories(conn, bank_id, ["Dave collects vinyl records."])
            # Manually stamp consolidation_failed_at to simulate a prior failed run
            await conn.execute(
                "UPDATE memory_units SET consolidation_failed_at = NOW() WHERE id = $1",
                mem_id,
            )

        # Even with a healthy LLM, the memory should be skipped
        mock_llm = _make_always_success_mock_llm()
        _inject_mock_llm(memory_no_llm_verify, mock_llm)

        result = await run_consolidation_job(
            memory_engine=memory_no_llm_verify,
            bank_id=bank_id,
            request_context=request_context,
        )

        # No unconsolidated memories to pick up (consolidation_failed_at ≠ NULL, consolidated_at = NULL
        # but the SELECT filters on consolidated_at IS NULL AND fact_type IN ('experience','world'))
        assert result["status"] in ("no_new_memories", "completed")
        if result["status"] == "completed":
            assert result["memories_processed"] == 0

        # Memory still has consolidation_failed_at set and consolidated_at NULL
        async with memory_no_llm_verify._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT consolidated_at, consolidation_failed_at FROM memory_units WHERE id = $1",
                mem_id,
            )
        assert row["consolidated_at"] is None
        assert row["consolidation_failed_at"] is not None

        await memory_no_llm_verify.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_partial_batch_failure(self, memory_no_llm_verify: MemoryEngine, request_context):
        """In a batch of 2, if only the first individual retry fails, the second still succeeds."""
        bank_id = f"test-partial-fail-{uuid.uuid4().hex[:8]}"
        await memory_no_llm_verify.get_bank_profile(bank_id=bank_id, request_context=request_context)

        async with memory_no_llm_verify._pool.acquire() as conn:
            mem_ids = await _insert_memories(
                conn,
                bank_id,
                [
                    "Eve speaks three languages fluently.",
                    "Eve learned Japanese in two years.",
                ],
            )

        # Exhaust 3 retries for batch=2 (calls 1-3), exhaust 3 retries for first batch=1 (calls 4-6),
        # second batch=1 succeeds (call 7)
        mock_llm = _make_failing_mock_llm(fail_first_n=6)
        _inject_mock_llm(memory_no_llm_verify, mock_llm)

        result = await run_consolidation_job(
            memory_engine=memory_no_llm_verify,
            bank_id=bank_id,
            request_context=request_context,
        )

        assert result["memories_processed"] == 2
        assert result["memories_failed"] == 1

        async with memory_no_llm_verify._pool.acquire() as conn:
            rows = {
                str(r["id"]): r
                for r in await conn.fetch(
                    "SELECT id, consolidated_at, consolidation_failed_at FROM memory_units "
                    "WHERE bank_id = $1 AND fact_type = 'experience'",
                    bank_id,
                )
            }

        # One should have failed, one should have succeeded
        failed = [r for r in rows.values() if r["consolidation_failed_at"] is not None]
        succeeded = [r for r in rows.values() if r["consolidated_at"] is not None]
        assert len(failed) == 1
        assert len(succeeded) == 1
        # They must be different memories
        assert str(failed[0]["id"]) != str(succeeded[0]["id"])

        await memory_no_llm_verify.delete_bank(bank_id, request_context=request_context)


class TestRecoverConsolidation:
    """Verify the retry_failed_consolidation() method and the /consolidation/recover endpoint."""

    @pytest.mark.asyncio
    async def test_recover_resets_failed_memories(self, memory_no_llm_verify: MemoryEngine, request_context):
        """retry_failed_consolidation resets consolidation_failed_at and consolidated_at."""
        bank_id = f"test-recover-reset-{uuid.uuid4().hex[:8]}"
        await memory_no_llm_verify.get_bank_profile(bank_id=bank_id, request_context=request_context)

        async with memory_no_llm_verify._pool.acquire() as conn:
            ids = await _insert_memories(
                conn,
                bank_id,
                [
                    "Frank is a competitive cyclist.",
                    "Frank completed the Tour de France route.",
                ],
            )
            # Mark both as failed
            for mem_id in ids:
                await conn.execute(
                    "UPDATE memory_units SET consolidation_failed_at = NOW() WHERE id = $1",
                    mem_id,
                )

        result = await memory_no_llm_verify.retry_failed_consolidation(bank_id, request_context=request_context)

        assert result["retried_count"] == 2

        async with memory_no_llm_verify._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT consolidated_at, consolidation_failed_at FROM memory_units "
                "WHERE bank_id = $1 AND fact_type = 'experience'",
                bank_id,
            )
        assert all(r["consolidation_failed_at"] is None for r in rows), "consolidation_failed_at must be cleared"
        assert all(r["consolidated_at"] is None for r in rows), "consolidated_at must also be cleared"

        await memory_no_llm_verify.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_recover_returns_zero_when_none_failed(self, memory_no_llm_verify: MemoryEngine, request_context):
        """retry_failed_consolidation returns 0 when no memories have failed."""
        bank_id = f"test-recover-zero-{uuid.uuid4().hex[:8]}"
        await memory_no_llm_verify.get_bank_profile(bank_id=bank_id, request_context=request_context)

        result = await memory_no_llm_verify.retry_failed_consolidation(bank_id, request_context=request_context)

        assert result["retried_count"] == 0

        await memory_no_llm_verify.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_recover_then_consolidate_succeeds(self, memory_no_llm_verify: MemoryEngine, request_context):
        """After recovery, the memory is picked up by the next consolidation run."""
        bank_id = f"test-recover-consolidate-{uuid.uuid4().hex[:8]}"
        await memory_no_llm_verify.get_bank_profile(bank_id=bank_id, request_context=request_context)

        async with memory_no_llm_verify._pool.acquire() as conn:
            (mem_id,) = await _insert_memories(conn, bank_id, ["Grace is an expert rock climber."])
            await conn.execute("UPDATE memory_units SET consolidation_failed_at = NOW() WHERE id = $1", mem_id)

        # Recover
        recover_result = await memory_no_llm_verify.retry_failed_consolidation(bank_id, request_context=request_context)
        assert recover_result["retried_count"] == 1

        # Now consolidate with a healthy LLM
        mock_llm = _make_always_success_mock_llm()
        _inject_mock_llm(memory_no_llm_verify, mock_llm)

        run_result = await run_consolidation_job(
            memory_engine=memory_no_llm_verify,
            bank_id=bank_id,
            request_context=request_context,
        )

        assert run_result["memories_processed"] == 1
        assert run_result["memories_failed"] == 0

        async with memory_no_llm_verify._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT consolidated_at, consolidation_failed_at FROM memory_units WHERE id = $1",
                mem_id,
            )
        assert row["consolidated_at"] is not None, "Memory should be consolidated after recovery"
        assert row["consolidation_failed_at"] is None

        await memory_no_llm_verify.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_recover_endpoint_via_http(self, memory_no_llm_verify: MemoryEngine, request_context):
        """The POST /consolidation/recover endpoint returns the correct retried_count."""
        import httpx

        from hindsight_api.api.http import create_app

        bank_id = f"test-recover-http-{uuid.uuid4().hex[:8]}"
        await memory_no_llm_verify.get_bank_profile(bank_id=bank_id, request_context=request_context)

        async with memory_no_llm_verify._pool.acquire() as conn:
            ids = await _insert_memories(
                conn,
                bank_id,
                ["Henry is a professional chef.", "Henry trained at Le Cordon Bleu."],
            )
            for mem_id in ids:
                await conn.execute("UPDATE memory_units SET consolidation_failed_at = NOW() WHERE id = $1", mem_id)

        app = create_app(memory_no_llm_verify, initialize_memory=False)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(f"/v1/default/banks/{bank_id}/consolidation/recover")

        assert response.status_code == 200
        body = response.json()
        assert body["retried_count"] == 2

        await memory_no_llm_verify.delete_bank(bank_id, request_context=request_context)
