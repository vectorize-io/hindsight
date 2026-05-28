import uuid

import pytest

from hindsight_api.engine.consolidation import consolidator


class _ZeroLengthEmbeddings:
    dimension = 384

    def encode_documents(self, texts):
        assert texts == ["Consolidated observation text."]
        return [[]]


class _FakeMemoryEngine:
    embeddings = _ZeroLengthEmbeddings()


class _FailingConn:
    async def fetchrow(self, *args, **kwargs):
        raise AssertionError("zero-length embedding should be rejected before database insert")


@pytest.mark.asyncio
async def test_create_observation_rejects_zero_length_embedding_before_insert(monkeypatch):
    source_id = uuid.uuid4()

    async def fake_filter_live_source_memories(conn, bank_id, source_memory_ids):
        return source_memory_ids

    monkeypatch.setattr(consolidator, "_filter_live_source_memories", fake_filter_live_source_memories)

    with pytest.raises(RuntimeError, match="embedding 0 has dimension 0; expected 384"):
        await consolidator._create_observation_directly(
            conn=_FailingConn(),
            memory_engine=_FakeMemoryEngine(),
            bank_id="test-bank",
            source_memory_ids=[source_id],
            observation_text="Consolidated observation text.",
        )
