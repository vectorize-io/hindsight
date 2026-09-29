"""`expand` must return the chunk and document whichever store holds them.

The memories read in ``tool_expand`` is routed to the memories store, but the chunk and document
reads were plain SQL against the ``chunks`` / ``documents`` tables. A store that owns the document
store leaves those empty by construction, so expand answered without the chunk or the document it
was asked for — on real data, with no error to notice.

The existing coverage in ``test_reflect_tools.py`` cannot catch that: it seeds memories through a
mocked connection, which a store-owned bank never reads. So this goes through a real retain and a
real connection, and asserts on the content that came back rather than on how it was fetched.
"""

from datetime import datetime, timezone

import pytest

from hindsight_api.engine.reflect.tools import tool_expand


def _ts() -> float:
    return datetime.now(timezone.utc).timestamp()


@pytest.mark.asyncio
async def test_expand_returns_chunk_and_document_for_whichever_store_holds_them(memory, request_context):
    bank_id = f"test_expand_store_{_ts()}"
    document_id = "expand-doc-1"
    metadata = {"source": "expand-store-test"}
    body = "\n".join(
        f"[role: user] turn {i}: alpha bravo charlie delta echo foxtrot golf hotel india juliet" for i in range(12)
    )

    try:
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": body, "document_id": document_id, "metadata": metadata}],
            request_context=request_context,
        )

        listing = await memory.list_memory_units(bank_id, document_id=document_id, request_context=request_context)
        memory_ids = [row["id"] for row in listing["items"]][:3]
        assert memory_ids, "retain produced no memories, so the expand assertions would be vacuous"

        async with memory._backend.acquire() as conn:
            result = await tool_expand(conn, bank_id, memory_ids, "document")

        assert result["count"] == len(memory_ids)
        for item in result["results"]:
            # The memory itself — already routed to the store, asserted so a regression there is
            # not mistaken for one in the chunk/document reads below.
            assert "memory" in item, f"expand returned no memory for {item.get('memory_id')}: {item}"
            assert item["memory"]["text"]

            # The document behind it. `full_text` is what makes this more than a key check: an
            # empty record would still carry the key.
            assert "document" in item, f"expand returned no document for {item.get('memory_id')}: {item}"
            assert item["document"]["full_text"], "the document came back without its text"
            assert item["document"]["metadata"] == metadata

        # At least one memory should have carried a chunk, and that chunk's text must come back —
        # otherwise depth="chunk" is silently degraded to "memory".
        with_chunks = [i for i in result["results"] if i.get("chunk")]
        assert with_chunks, "no result carried a chunk, so the chunk read is not being exercised"
        assert all(i["chunk"]["text"] for i in with_chunks), "a chunk came back without its text"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
