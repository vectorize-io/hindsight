"""Retain stops before the LLM pass when the embeddings backend is known to be down (#5999).

A dead embeddings endpoint used to be discovered only after fact extraction had been
paid for, on every retry. The engine now keeps an in-memory note of transient embedding
failures and checks it before extraction: free while healthy, one tiny probe while the
note is set, and a deferral (not a failure) when the probe fails.
"""

import uuid
import weakref
from datetime import UTC, datetime

import pytest

from hindsight_api.models import RequestContext
from hindsight_api.engine.retain import embedding_utils, fact_extraction
from hindsight_api.engine.retain.embedding_utils import (
    EmbeddingsUnavailableError,
    ensure_embeddings_reachable,
    generate_embeddings_batch,
)
from hindsight_api.worker.exceptions import DeferOperation, RetryTaskAt


class _FakeBackend:
    dimension = 3

    def __init__(self) -> None:
        self.calls = 0
        self.error: Exception | None = None

    async def encode_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.error is not None:
            raise RuntimeError(f"TEI embedding request failed: {self.error}") from self.error
        return [[0.1, 0.2, 0.3] for _ in texts]

    async def encode_query(self, texts: list[str]) -> list[list[float]]:
        return await self.encode_documents(texts)


@pytest.fixture(autouse=True)
def _fresh_health(monkeypatch):
    # Session-scoped backends are shared across tests: never leak a "down" note.
    monkeypatch.setattr(embedding_utils, "_down_backends", weakref.WeakSet())


async def _fail_once(backend: _FakeBackend) -> None:
    backend.error = ConnectionRefusedError(111, "Connection refused")
    with pytest.raises(Exception, match="Failed to generate batch embeddings"):
        await generate_embeddings_batch(backend, ["hello"])


@pytest.mark.asyncio
async def test_healthy_backend_costs_no_extra_calls():
    backend = _FakeBackend()
    await ensure_embeddings_reachable(backend)
    assert backend.calls == 0


@pytest.mark.asyncio
async def test_after_a_transient_failure_the_next_retain_probes_and_stops_while_down():
    backend = _FakeBackend()
    await _fail_once(backend)
    calls = backend.calls

    with pytest.raises(EmbeddingsUnavailableError):
        await ensure_embeddings_reachable(backend)
    assert backend.calls == calls + 1  # one probe, no trust window


@pytest.mark.asyncio
async def test_a_recovered_backend_is_never_blocked():
    backend = _FakeBackend()
    await _fail_once(backend)
    backend.error = None  # back right away: a trust window would wrongly block here

    await ensure_embeddings_reachable(backend)  # one probe, succeeds
    calls = backend.calls
    await ensure_embeddings_reachable(backend)  # note cleared: free again
    assert backend.calls == calls


@pytest.mark.asyncio
async def test_a_real_success_clears_the_note():
    backend = _FakeBackend()
    await _fail_once(backend)
    backend.error = None
    await generate_embeddings_batch(backend, ["hello"])  # e.g. a recall
    calls = backend.calls
    await ensure_embeddings_reachable(backend)
    assert backend.calls == calls


@pytest.mark.asyncio
async def test_non_transient_failure_is_not_an_outage():
    backend = _FakeBackend()
    backend.error = ValueError("invalid api key")
    with pytest.raises(Exception):
        await generate_embeddings_batch(backend, ["hello"])
    await ensure_embeddings_reachable(backend)  # no note, does not block


@pytest.mark.asyncio
async def test_retain_skips_extraction_when_embeddings_down(memory, request_context, monkeypatch):
    async def extraction_must_not_run(*_args, **_kwargs):
        raise AssertionError("LLM extraction ran although embeddings are down")

    async def backend_down(_texts: list[str]) -> list[list[float]]:
        raise ConnectionRefusedError(111, "Connection refused")

    monkeypatch.setattr(fact_extraction, "extract_facts_from_contents", extraction_must_not_run)
    monkeypatch.setattr(memory.embeddings, "encode_documents", backend_down)
    embedding_utils._down_backends.add(memory.embeddings)

    with pytest.raises(EmbeddingsUnavailableError):
        await memory.retain_batch_async(
            bank_id=f"test-emb-preflight-{datetime.now(UTC).timestamp()}",
            contents=[{"content": "Alice works at Acme."}],
            request_context=request_context,
        )


@pytest.mark.asyncio
async def test_worker_defers_retain_instead_of_failing(memory, monkeypatch):
    async def embeddings_down_retain(_task_dict: object) -> None:
        raise EmbeddingsUnavailableError("embeddings backend is unreachable")

    monkeypatch.setattr(memory, "_handle_batch_retain", embeddings_down_retain)
    # A retain job only runs for a bank that exists: one for a missing bank is skipped (#6344).
    bank_id = f"test-emb-preflight-defer-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id, request_context=RequestContext(internal=True))

    with pytest.raises(DeferOperation) as exc_info:
        await memory.execute_task({"type": "batch_retain", "bank_id": bank_id, "contents": [{"content": "x"}]})
    assert not isinstance(exc_info.value, RetryTaskAt)
    assert exc_info.value.exec_date > datetime.now(UTC)
