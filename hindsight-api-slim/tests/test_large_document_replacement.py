"""
Reproduction for https://github.com/vectorize-io/hindsight/issues/1838

Replacing an existing document by retaining new content with the same
``document_id`` can leave the stored document body partial/truncated when
``retain_batch_async`` auto-splits the submitted content into multiple
sub-batches. Each non-first sub-batch was overwriting
``documents.original_text`` with its own slice, so the persisted body ended
up being one slice of the input, not the full body.

These tests trigger the auto-split path by lowering
``HINDSIGHT_API_RETAIN_BATCH_TOKENS`` to a small value, then assert that the
stored ``original_text`` exactly matches the submitted replacement body.
"""

from datetime import datetime, timezone
from typing import Any

import pytest

from hindsight_api.config import clear_config_cache
from hindsight_api.engine.memory_engine import _split_contents_into_sub_batches
from hindsight_api.engine.response_models import TokenUsage
from hindsight_api.engine.retain.types import ChunkMetadata, ExtractedFact, RetainContent


def _ts() -> float:
    return datetime.now(timezone.utc).timestamp()


@pytest.fixture(autouse=True)
def _fast_split_env(monkeypatch):
    """Make the splitter trigger on small content and skip consolidation work.

    Auto-consolidation runs synchronously after each retain in tests; it
    extracts/recalls/embeds across all observations and dominates wall time
    for these tests, which only care about how the splitter persists the
    document body. Disabling it brings the suite back to single-digit
    seconds.
    """
    monkeypatch.setenv("HINDSIGHT_API_RETAIN_BATCH_TOKENS", "100")
    monkeypatch.setenv("HINDSIGHT_API_ENABLE_AUTO_CONSOLIDATION", "false")
    monkeypatch.setenv("HINDSIGHT_API_ENABLE_OBSERVATIONS", "false")
    clear_config_cache()
    yield
    clear_config_cache()


def _make_replacement_body() -> str:
    """Build a multi-line body comfortably above the 100-token splitter
    threshold so ``_split_contents_into_sub_batches`` chunks it into more
    than one sub-batch.
    """
    lines = [
        f"[role: user] turn {i}: alpha bravo charlie delta echo foxtrot golf hotel india juliet" for i in range(20)
    ]
    return "\n".join(lines)


@pytest.mark.asyncio
async def test_large_same_id_replacement_preserves_full_body(memory, request_context):
    """
    RED test for issue #1838.

    Retain a small initial document, then replace it with a larger body
    under the same ``document_id``. The replacement is sized to trip the
    ``retain_batch_tokens`` threshold so ``retain_batch_async`` splits it
    into multiple sub-batches. After retain returns, the stored
    ``original_text`` must exactly equal the submitted replacement body.
    """
    bank_id = f"test_large_replace_{_ts()}"
    document_id = "claude-code-transcript-1838"

    try:
        initial_body = "[role: user] turn 0: hello\n[role: assistant] turn 0: hi"
        await memory.retain_async(
            bank_id=bank_id,
            content=initial_body,
            context="initial retro",
            document_id=document_id,
            request_context=request_context,
        )

        doc_initial = await memory.get_document(document_id, bank_id, request_context=request_context)
        assert doc_initial is not None
        assert doc_initial["original_text"] == initial_body

        replacement_body = _make_replacement_body()

        await memory.retain_async(
            bank_id=bank_id,
            content=replacement_body,
            context="regenerated retro",
            document_id=document_id,
            request_context=request_context,
        )

        doc_replaced = await memory.get_document(document_id, bank_id, request_context=request_context)
        assert doc_replaced is not None

        stored = doc_replaced["original_text"]
        assert len(stored) == len(replacement_body), (
            f"stored body length {len(stored)} != submitted length "
            f"{len(replacement_body)} — partial replacement persisted"
        )
        assert stored == replacement_body, "stored original_text does not exactly match the submitted replacement body"

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_repeated_large_same_id_replacement_is_idempotent(memory, request_context):
    """
    Retrying the same large replacement (per issue #1838 acceptance criteria)
    must converge to the exact submitted body, not a suffix/prefix subset.
    """
    bank_id = f"test_large_replace_retry_{_ts()}"
    document_id = "claude-code-transcript-1838-retry"

    try:
        await memory.retain_async(
            bank_id=bank_id,
            content="[role: user] turn 0: seed",
            context="seed",
            document_id=document_id,
            request_context=request_context,
        )

        replacement_body = _make_replacement_body()

        for attempt in range(3):
            await memory.retain_async(
                bank_id=bank_id,
                content=replacement_body,
                context=f"regenerated retro attempt {attempt}",
                document_id=document_id,
                request_context=request_context,
            )

            doc = await memory.get_document(document_id, bank_id, request_context=request_context)
            assert doc is not None, f"attempt {attempt}: document missing after retain"
            assert doc["original_text"] == replacement_body, (
                f"attempt {attempt}: stored body diverged from submitted body "
                f"(stored {len(doc['original_text'])} chars, "
                f"submitted {len(replacement_body)} chars)"
            )

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_append_after_zero_fact_header_slice_skips_unchanged_history(
    memory,
    request_context,
    monkeypatch,
):
    """An append-only oversized replacement must not replay historical slices.

    Markdown transcripts can split into a small header-only first sub-batch.
    If that header extracts zero facts, Hindsight stores no chunk at index 0.
    A later append must still recognize the complete stored document as a
    stable prefix and leave historical chunks untouched.
    """
    from hindsight_api.engine.retain import fact_extraction

    bank_id = f"test_large_append_zero_fact_header_{_ts()}"
    document_id = "chat-session-zero-fact-header"
    header = "# Stable Chat Session\nSession id: session-1\nCreated at: 2026-07-09T00:00:00Z"
    history = "\n".join(
        f"[role: user] turn {turn}: "
        f"{'UNCHANGED_HEAD' if turn == 0 else 'UNCHANGED_MIDDLE' if turn == 30 else 'historical'} "
        "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
        for turn in range(60)
    )
    initial_body = f"{header}\n\n{history}"
    tail_marker = "NEW_APPEND_ONLY_TAIL"
    appended_body = f"{initial_body}\n[role: user] turn 60: {tail_marker} alpha bravo charlie delta"

    split = _split_contents_into_sub_batches(
        [{"content": initial_body, "document_id": document_id}],
        100,
    )
    assert len(split.sub_batches) > 3
    assert "[role:" not in split.sub_batches[0][0]["content"]

    extracted_contents: list[str] = []

    async def _record_extraction(
        contents: list[RetainContent],
        *args: Any,
        **kwargs: Any,
    ) -> tuple[list[ExtractedFact], list[ChunkMetadata], TokenUsage]:
        extracted_contents.extend(item.content for item in contents)
        facts: list[ExtractedFact] = []
        chunks: list[ChunkMetadata] = []
        for index, item in enumerate(contents):
            is_header_only = "[role:" not in item.content
            chunks.append(
                ChunkMetadata(
                    chunk_text=item.content,
                    fact_count=0 if is_header_only else 1,
                    content_index=index,
                    chunk_index=index,
                )
            )
            if not is_header_only:
                facts.append(
                    ExtractedFact(
                        fact_text=f"Synthetic extracted fact {index}",
                        fact_type="world",
                        content_index=index,
                        chunk_index=index,
                        context=item.context,
                        tags=item.tags,
                    )
                )
        return facts, chunks, TokenUsage()

    monkeypatch.setattr(
        fact_extraction,
        "extract_facts_from_contents",
        _record_extraction,
    )
    monkeypatch.setattr(
        memory.embeddings,
        "encode_documents",
        lambda texts: [[0.0] * memory.embeddings.dimension for _ in texts],
    )

    try:
        await memory.retain_async(
            bank_id=bank_id,
            content=initial_body,
            context="session transcript",
            document_id=document_id,
            request_context=request_context,
        )

        baseline_extractions = len(extracted_contents)
        assert baseline_extractions > 3
        extracted_contents.clear()

        await memory.retain_async(
            bank_id=bank_id,
            content=appended_body,
            context="session transcript",
            document_id=document_id,
            request_context=request_context,
        )

        extracted_text = "\n".join(extracted_contents)
        assert tail_marker in extracted_text
        assert len(extracted_contents) <= 2, "append-only replacement replayed more than two unchanged/edge slices"
        assert len(extracted_contents) < baseline_extractions / 2
        assert "UNCHANGED_HEAD" not in extracted_text
        assert "UNCHANGED_MIDDLE" not in extracted_text

        document = await memory.get_document(
            document_id,
            bank_id,
            request_context=request_context,
        )
        assert document is not None
        assert document["original_text"] == appended_body
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
