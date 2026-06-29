"""Deterministic unit tests for the pre-embed guards in the consolidation executors.

Both findings are exercised against the production ``_wraps_backend`` acquisition path
(not a raw-pool sentinel):

* zero-length embeddings are rejected BEFORE any write reaches the connection;
* when every source memory is already gone, the create/update executors short-circuit
  BEFORE running the (slow) embedder, so a no-op consolidation never embeds and a
  failing embedder never raises where it used to skip cleanly.
"""

import types
import uuid
from contextlib import asynccontextmanager

import pytest

from hindsight_api.engine.consolidation import consolidator


class _ZeroLengthEmbeddings:
    dimension = 384

    def encode_documents(self, texts):
        assert texts == ["Consolidated observation text."]
        return [[]]


class _FakeMemoryEngine:
    embeddings = _ZeroLengthEmbeddings()


class _WriteForbiddenConn:
    """Backend connection allowing only the pre-embed liveness probe.

    The preflight (``_any_live_source_memory``) runs ``fetchval``; the write path
    (``transaction``/``fetchrow``/``execute``/``executemany``) must never be reached,
    because the zero-length embedding is rejected first.
    """

    async def fetchval(self, *args, **kwargs):
        return 1  # a live source exists -> proceed to embedding

    async def fetch(self, *args, **kwargs):
        return [{"id": 1}]

    def transaction(self):
        raise AssertionError("write transaction entered before the zero-length embedding was rejected")

    async def fetchrow(self, *args, **kwargs):
        raise AssertionError("INSERT reached before the zero-length embedding was rejected")

    async def execute(self, *args, **kwargs):
        raise AssertionError("write reached before the zero-length embedding was rejected")

    async def executemany(self, *args, **kwargs):
        raise AssertionError("write reached before the zero-length embedding was rejected")


class _NoLiveConn:
    """Backend connection whose liveness probe reports no live source.

    Correct code short-circuits at the preflight (``fetchval`` -> None) and never embeds
    or writes; every write method fails hard as a backstop.
    """

    async def fetchval(self, *args, **kwargs):
        return None  # no live source -> skip before embedding

    def transaction(self):
        raise AssertionError("write transaction entered after all sources were dead")

    async def fetchrow(self, *args, **kwargs):
        raise AssertionError("write reached after all sources were dead")

    async def execute(self, *args, **kwargs):
        raise AssertionError("write reached after all sources were dead")


class _Backend:
    """Backend-shaped stand-in matching acquire_with_retry's ``_wraps_backend`` path."""

    _wraps_backend = True

    def __init__(self, conn):
        self._conn = conn

    @asynccontextmanager
    async def acquire(self):
        yield self._conn


def _forbid_embedder(monkeypatch):
    """Make any call to the embedder fail hard, proving it is never reached."""

    async def _embedder_must_not_run(*args, **kwargs):
        raise AssertionError("embedder must not run when all source memories are dead")

    monkeypatch.setattr(
        "hindsight_api.engine.retain.embedding_utils.generate_embeddings_batch",
        _embedder_must_not_run,
    )


@pytest.mark.asyncio
async def test_create_observation_rejects_zero_length_embedding_before_insert():
    # Preflight passes (fetchval -> live); the real embedder then yields a zero-length vector,
    # which must be rejected before any write. _WriteForbiddenConn fails hard if a write runs.
    with pytest.raises(RuntimeError, match="embedding 0 has dimension 0; expected 384"):
        await consolidator._create_observation_directly(
            pool=_Backend(_WriteForbiddenConn()),
            memory_engine=_FakeMemoryEngine(),
            bank_id="test-bank",
            source_memory_ids=[uuid.uuid4()],
            observation_text="Consolidated observation text.",
        )


@pytest.mark.asyncio
async def test_create_observation_skips_before_embedding_when_all_sources_dead(monkeypatch):
    # All sources gone -> the create must short-circuit at the preflight, BEFORE the embedder
    # (patched to explode if reached). No write may run.
    _forbid_embedder(monkeypatch)
    result = await consolidator._create_observation_directly(
        pool=_Backend(_NoLiveConn()),
        memory_engine=_FakeMemoryEngine(),
        bank_id="test-bank",
        source_memory_ids=[uuid.uuid4()],
        observation_text="Consolidated observation text.",
    )
    assert result == {"action": "skipped", "reason": "sources_deleted"}


@pytest.mark.asyncio
async def test_update_action_skips_before_embedding_when_all_sources_dead(monkeypatch):
    # Same preflight contract on the UPDATE path: all sources gone -> skip (return None) BEFORE
    # the embedder runs. The existing real-DB update test asserts only post-state, so it would
    # pass even with the embed-first ordering this finding fixed; this pins the embed-skip directly.
    _forbid_embedder(monkeypatch)
    obs_id = str(uuid.uuid4())
    result = await consolidator._execute_update_action(
        pool=_Backend(_NoLiveConn()),
        memory_engine=_FakeMemoryEngine(),
        bank_id="test-bank",
        source_memory_ids=[uuid.uuid4(), uuid.uuid4()],
        observation_id=obs_id,
        new_text="This update must not land.",
        observations=[types.SimpleNamespace(id=obs_id)],
    )
    assert result is None


@pytest.mark.asyncio
async def test_update_action_rejects_zero_length_embedding_before_write():
    # Preflight passes (a live source exists -> fetchval=1); the real embedder then yields a
    # zero-length vector, which must be rejected before any write. _WriteForbiddenConn fails
    # hard if the write transaction is reached.
    obs_id = str(uuid.uuid4())
    with pytest.raises(RuntimeError, match="embedding 0 has dimension 0; expected 384"):
        await consolidator._execute_update_action(
            pool=_Backend(_WriteForbiddenConn()),
            memory_engine=_FakeMemoryEngine(),
            bank_id="test-bank",
            source_memory_ids=[uuid.uuid4()],
            observation_id=obs_id,
            new_text="Consolidated observation text.",
            observations=[
                types.SimpleNamespace(
                    id=obs_id,
                    text="prior observation text",
                    tags=[],
                    occurred_start=None,
                    occurred_end=None,
                    mentioned_at=None,
                    source_fact_ids=[],
                )
            ],
        )
