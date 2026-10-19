"""A retain whose chunk extraction failed must not replace the document (#5589).

The streaming pipeline extracts chunks concurrently and commits them in batches. A
chunk that failed (an embeddings outage, an LLM error) was only re-raised after the
consumer had already written the batch at the end-of-stream sentinel — and that
write runs document tracking: it deletes the document's previous memories and
overwrites its text. A re-retain during an outage left documents with zero memories.
"""

from datetime import datetime, timezone

import pytest

from hindsight_api.engine.response_models import TokenUsage
from hindsight_api.engine.retain import orchestrator


@pytest.mark.asyncio
async def test_failed_reretain_keeps_previous_memories(memory, request_context, monkeypatch):
    bank_id = f"test_failed_reretain_{datetime.now(timezone.utc).timestamp()}"
    document_id = "doc-5589"
    try:
        await memory.retain_async(
            bank_id=bank_id,
            content="Alice works at Google. Bob works at Microsoft.",
            document_id=document_id,
            request_context=request_context,
        )
        before = await memory.get_document(document_id, bank_id, request_context=request_context)
        assert before["memory_unit_count"] > 0

        # The fact-bearing chunk fails (its embeddings call raised); the other chunk
        # extracted nothing and so never reached the embeddings backend.
        async def flaky_extract_and_embed(contents, *_args, **_kwargs):
            if contents[0].content.startswith("FAIL"):
                raise RuntimeError("embeddings quota exceeded")
            return orchestrator._EmbeddedExtraction([], [], [], TokenUsage())

        monkeypatch.setattr(orchestrator, "_extract_and_embed", flaky_extract_and_embed)
        # One retain item per chunk keeps the chunk boundaries out of the test.
        with pytest.raises(RuntimeError, match="embeddings quota exceeded"):
            await memory.retain_batch_async(
                bank_id=bank_id,
                contents=[
                    {"content": "FAIL Carol works at Apple.", "document_id": document_id},
                    {"content": "nothing worth remembering", "document_id": document_id},
                ],
                document_id=document_id,
                request_context=request_context,
            )

        after = await memory.get_document(document_id, bank_id, request_context=request_context)
        assert after["memory_unit_count"] == before["memory_unit_count"], (
            "the failed retain deleted the document's previous memories"
        )
        assert after["original_text"] == before["original_text"]
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
