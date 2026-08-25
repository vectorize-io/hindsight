"""Retain embeds a document's chunks in batches, not one request per chunk (#3784).

In ``chunks`` extraction mode there is no LLM call to overlap, so the single-text
embedding request the streaming producer used to issue per chunk *was* the entire
per-chunk cost — ingest sat at a flat ~21 chunks/s regardless of document size. These
tests assert on the shape of what reaches the embeddings backend, which is the thing
that actually changed; wall-clock would be a flaky proxy for it.
"""

import os
import uuid

import pytest

from hindsight_api.config import clear_config_cache
from hindsight_api.models import RequestContext

pytestmark = pytest.mark.asyncio


class CountingEmbeddings:
    """Wraps the session embeddings backend and records the shape of every encode()."""

    # What the coalescer sizes its batches from.
    batch_size = 100

    def __init__(self, inner) -> None:
        self._inner = inner
        self.document_calls: list[int] = []

    @property
    def provider_name(self) -> str:
        return "counting"

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    async def initialize(self) -> None:
        await self._inner.initialize()

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self._inner.encode(texts)

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(len(texts))
        return self._inner.encode_documents(texts)

    def encode_query(self, texts: list[str]) -> list[list[float]]:
        return self._inner.encode_query(texts)


@pytest.fixture
def chunks_mode():
    """Run retain in ``chunks`` extraction mode, where extraction is a no-op."""
    saved = os.environ.get("HINDSIGHT_API_RETAIN_EXTRACTION_MODE")
    os.environ["HINDSIGHT_API_RETAIN_EXTRACTION_MODE"] = "chunks"
    clear_config_cache()
    yield
    if saved is None:
        os.environ.pop("HINDSIGHT_API_RETAIN_EXTRACTION_MODE", None)
    else:
        os.environ["HINDSIGHT_API_RETAIN_EXTRACTION_MODE"] = saved
    clear_config_cache()


async def _prepare_bank(memory, bank_id: str) -> None:
    """Keep the counters on the retain path only.

    Auto-consolidation runs inline under the tests' synchronous task backend and
    embeds each observation on its own, which has nothing to do with what the retain
    producer sends.
    """
    await memory.update_bank_config(
        bank_id,
        {"enable_auto_consolidation": False, "enable_observations": False},
        request_context=RequestContext(),
    )


def _document(sentences: int) -> str:
    """A document whose sentences are all distinct, so no chunk is deduped away.

    Long enough to span several chunks at the default ``retain_chunk_size``.
    """
    return " ".join(
        f"Sentence number {index} describes topic {index} in unmistakable detail and at "
        f"considerable length, so that the text around position {index} is unlike the text "
        f"anywhere else in this document and hashes to a chunk of its own."
        for index in range(sentences)
    )


async def test_chunks_mode_embeds_a_document_in_batched_requests(chunks_mode, memory):
    """One document's chunks reach the backend together, not one request each."""
    bank_id = f"test-3784-batched-{uuid.uuid4().hex[:8]}"
    await _prepare_bank(memory, bank_id)
    counting = CountingEmbeddings(memory.embeddings)
    memory.embeddings = counting

    unit_ids = await memory.retain_async(
        bank_id,
        _document(160),
        document_id="doc-3784",
        request_context=RequestContext(),
    )

    assert len(unit_ids) > 5, "test needs a genuinely multi-chunk document"
    assert counting.document_calls, "no document embeddings were generated"
    # The pre-fix path issued exactly one request per chunk, every one a single text.
    assert len(counting.document_calls) < len(unit_ids)
    assert max(counting.document_calls) > 1


async def test_every_chunk_still_becomes_its_own_memory(chunks_mode, memory):
    """Batching must not merge, drop, or reorder chunks."""
    bank_id = f"test-3784-alignment-{uuid.uuid4().hex[:8]}"
    await _prepare_bank(memory, bank_id)
    counting = CountingEmbeddings(memory.embeddings)
    memory.embeddings = counting

    document = _document(120)
    unit_ids = await memory.retain_async(
        bank_id,
        document,
        document_id="doc-3784-alignment",
        request_context=RequestContext(),
    )

    assert sum(counting.document_calls) == len(unit_ids)

    listed = await memory.list_memory_units(
        bank_id,
        limit=len(unit_ids) + 10,
        request_context=RequestContext(),
    )
    assert listed["total"] == len(unit_ids)
    # chunks mode stores the raw chunk text, so every unit must be document text.
    for unit in listed["items"]:
        assert unit["text"] in document
