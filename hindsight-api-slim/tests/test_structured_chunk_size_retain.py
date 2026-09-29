"""Regression test for https://github.com/vectorize-io/hindsight/issues/2301

Raising ``HINDSIGHT_API_RETAIN_STRUCTURED_CHUNK_SIZE`` above
``HINDSIGHT_API_RETAIN_CHUNK_SIZE`` and retaining a JSONL/conversation document
whose line/turn overflows the chunk size used to crash with::

    asyncpg.exceptions.CardinalityViolationError:
        ON CONFLICT DO UPDATE command cannot affect row a second time

The streaming retain pipeline pre-chunks each document once (one ``chunk_index``
per piece) and then re-chunks every piece during extraction. With the structured
cap above the chunk size, a pre-chunk could legitimately exceed the re-chunk
budget, so it re-split into several sub-chunks that all inherited the one
``chunk_index`` — colliding on ``chunk_id = {bank}_{doc}_{index}`` in a single
upsert batch. ``chunk_text`` is now idempotent, so the re-chunk is a no-op.
"""

from datetime import datetime, timezone

import pytest

from hindsight_api.config import clear_config_cache


def _ts() -> float:
    return datetime.now(timezone.utc).timestamp()


@pytest.fixture(autouse=True)
def _structured_chunk_env(monkeypatch):
    # Structured cap ABOVE the chunk size — the configuration that triggers #2301.
    # Default-scale sizes (matching the issue) so the document yields only a
    # handful of chunks: a smaller chunk size explodes the embedding/link work and
    # destabilises the shared session fixture under xdist.
    monkeypatch.setenv("HINDSIGHT_API_RETAIN_CHUNK_SIZE", "3000")
    monkeypatch.setenv("HINDSIGHT_API_RETAIN_STRUCTURED_CHUNK_SIZE", "4500")
    monkeypatch.setenv("HINDSIGHT_API_ENABLE_AUTO_CONSOLIDATION", "false")
    monkeypatch.setenv("HINDSIGHT_API_ENABLE_OBSERVATIONS", "false")
    clear_config_cache()
    yield
    clear_config_cache()


@pytest.mark.asyncio
async def test_jsonl_line_over_chunk_size_retains_without_collision(memory, request_context):
    """A JSONL document with a line longer than the chunk size retains cleanly
    when the structured cap is raised above it (issue #2301)."""
    import json

    bank_id = f"test_2301_{_ts()}"
    document_id = "doc-2301"

    try:
        body = "\n".join(
            [
                json.dumps({"role": "user", "content": "short opening line"}),
                # Between chunk size (3000) and structured cap (4500): kept whole by
                # the producer, would re-split on re-chunk without the fix.
                json.dumps({"role": "assistant", "content": "k" * 3800}),
                # Past even the structured cap: fragmented as text.
                json.dumps({"role": "assistant", "content": "m" * 9000}),
                json.dumps({"role": "user", "content": "short closing line"}),
            ]
        )

        # The bug raised CardinalityViolationError here.
        await memory.retain_async(
            bank_id=bank_id,
            content=body,
            context="jsonl with oversized line",
            document_id=document_id,
            request_context=request_context,
        )

        chunks = await memory.list_document_chunks(bank_id, document_id, limit=10000, request_context=request_context)
        indices = sorted(c["chunk_index"] for c in chunks["items"])
        # No collisions: chunk_index values are unique.
        assert len(indices) == len(set(indices)), f"duplicate chunk_index values: {indices}"
        assert indices == list(range(len(indices))), f"chunk_index sequence not contiguous: {indices}"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
