"""The streaming retain pipeline must not outlive the call that started it (#3002).

`_streaming_retain_batch` runs an LLM producer and a DB consumer concurrently.
A plain `asyncio.gather` propagates the consumer's exception immediately but
leaves the producer — and every extraction task it fanned out — running. Those
tasks then park forever on `chunk_queue.put()` into a queue nobody drains: they
pin their chunk payloads for the life of the process and still spend LLM permits
and provider tokens on an operation that already failed.

The test drives the real pipeline with a DB error injected into the consumer
(standing in for the deadlock victim / lock timeout that triggered this in
production) while extractions are still in flight.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from hindsight_api.engine.retain import orchestrator
from hindsight_api.engine.response_models import TokenUsage


class _ExplodingPool:
    """Backend-shaped pool whose acquire() raises, failing the consumer's batch.

    `_wraps_backend` puts `acquire_with_retry` on its backend path; RuntimeError
    is not in the retryable set, so it propagates on the first attempt the way a
    deadlock victim's error would.
    """

    _wraps_backend = True
    ops = None

    def acquire(self):
        raise RuntimeError("deadlock detected")


@pytest.mark.asyncio
# Injects the consumer failure through an exploding Postgres pool. The store-owned write path is
# PG-free and never acquires from it, so the failure never fires, the deliberately-hanging
# extraction is never cancelled, and the test times out at 600s rather than failing.
@pytest.mark.memory_backend_incompatible
def _mock_config() -> MagicMock:
    """A stand-in config that models the fields the streaming pipeline actually reads."""
    config = MagicMock()
    config.retain_memory_budget_mb = 128
    return config


async def test_consumer_failure_cancels_in_flight_extractions(monkeypatch):
    hanging_started = asyncio.Event()
    cancelled = 0
    calls = 0

    async def fake_extract_and_embed(*_args, **_kwargs):
        """First chunk returns instantly (so the consumer runs and fails); the
        rest hang the way a call queued behind a saturated LLM semaphore does."""
        nonlocal calls, cancelled
        calls += 1
        if calls == 1:
            return [], [], [], TokenUsage()
        hanging_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled += 1
            raise
        return [], [], [], TokenUsage()

    monkeypatch.setattr(orchestrator, "_extract_and_embed", fake_extract_and_embed)

    chunks = ["chunk one", "chunk two", "chunk three", "chunk four"]
    before = {t for t in asyncio.all_tasks()}

    with pytest.raises(RuntimeError, match="deadlock detected"):
        await orchestrator._streaming_retain_batch(
            pool=_ExplodingPool(),
            embeddings_model=MagicMock(),
            llm_config=MagicMock(),
            entity_resolver=MagicMock(),
            format_date_fn=lambda d: str(d),
            bank_id="bank-cancel",
            contents_dicts=[{"content": "\n".join(chunks)}],
            contents=[],
            # A real number for the memory budget: `_streaming_retain_batch` builds a
            # RetainMemoryBudget from it, and a bare MagicMock attribute makes every
            # comparison inside it truthy — the producer then waits for room that is never
            # reported, which is a hang rather than the cancellation this test is about.
            config=_mock_config(),
            document_id="doc-cancel",
            is_first_batch=True,
            fact_type_override=None,
            document_tags=None,
            log_buffer=[],
            start_time=0.0,
            all_pre_chunks=list(chunks),
            chunk_to_content=[0] * len(chunks),
            # One chunk per consumer batch, so the consumer reaches the DB (and
            # fails) while the remaining extractions are still in flight.
            chunk_batch_size=1,
        )

    assert hanging_started.is_set(), "test never reached the state it means to cover"

    # Let the cancellations land.
    for _ in range(5):
        await asyncio.sleep(0)

    leaked = [t for t in asyncio.all_tasks() if t not in before and not t.done()]
    assert not leaked, f"pipeline tasks outlived the failed operation: {leaked}"
    assert cancelled >= 1, "in-flight extractions were not cancelled"
