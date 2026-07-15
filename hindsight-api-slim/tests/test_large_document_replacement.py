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

from hindsight_api.config import DEFAULT_RETAIN_CHUNK_SIZE, clear_config_cache, get_config
from hindsight_api.engine.response_models import TokenUsage
from hindsight_api.engine.retain.types import ChunkMetadata, ExtractedFact, RetainContent
from hindsight_api.engine.token_encoding import count_tokens


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
async def test_oversized_changed_tail_does_not_reextract_unchanged_history(memory, request_context, monkeypatch):
    """A changed tail above the batch threshold must retain against the full document."""
    from hindsight_api.engine.retain import fact_extraction, orchestrator

    bank_id = f"test_large_delta_tail_{_ts()}"
    document_id = "claude-code-transcript-delta-tail"
    history_markers = {i: f"UNCHANGED_HISTORY_SENTINEL_{i}" for i in (0, 25, 50, 75)}
    tail = "NEW_TAIL_SENTINEL"
    history = "\n".join(
        f"[role: user] turn {i}: {history_markers.get(i, 'historical')} "
        "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november"
        for i in range(120)
    )

    def _set_batch_tokens(value: int) -> None:
        monkeypatch.setenv("HINDSIGHT_API_RETAIN_BATCH_TOKENS", str(value))
        clear_config_cache()
        memory._config_resolver._global_config.retain_batch_tokens = value

    try:
        _set_batch_tokens(10000)
        assert count_tokens(history) <= get_config().retain_batch_tokens
        await memory.retain_async(
            bank_id=bank_id,
            content=history,
            context="session transcript",
            document_id=document_id,
            request_context=request_context,
        )

        extracted_contents: list[str] = []
        outbox_calls = 0

        async def _record_outbox(_conn: Any) -> None:
            nonlocal outbox_calls
            outbox_calls += 1

        async def _record_extraction(contents: list[RetainContent], *args: Any, **kwargs: Any) -> Any:
            extracted_contents.extend(item.content for item in contents)
            facts = [
                ExtractedFact(
                    fact_text=f"Synthetic extracted fact {index}",
                    fact_type="world",
                    content_index=index,
                    chunk_index=index,
                    context=item.context,
                    tags=item.tags,
                )
                for index, item in enumerate(contents)
            ]
            chunks = [
                ChunkMetadata(
                    chunk_text=item.content,
                    fact_count=1,
                    content_index=index,
                    chunk_index=index,
                )
                for index, item in enumerate(contents)
            ]
            return facts, chunks, TokenUsage()

        monkeypatch.setattr(fact_extraction, "extract_facts_from_contents", _record_extraction)
        monkeypatch.setattr(
            memory.embeddings,
            "encode_documents",
            lambda texts: [[0.0] * memory.embeddings.dimension for _ in texts],
        )

        _set_batch_tokens(2000)
        replacement = f"{history}\n[role: user] turn 120: {tail}"
        assert count_tokens(replacement) > get_config().retain_batch_tokens

        result = await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {
                    "content": replacement,
                    "context": "session transcript",
                    "document_id": document_id,
                }
            ],
            request_context=request_context,
            outbox_callback=_record_outbox,
        )

        extracted_text = "\n".join(extracted_contents)
        assert len(result) == 1, "Delta results must stay aligned with the single submitted document"
        assert outbox_calls == 1, "successful delta retain must commit the outbox exactly once"
        assert tail in extracted_text, "the changed tail must be extracted"
        assert not any(marker in extracted_text for marker in history_markers.values()), (
            "unchanged historical chunks were sent through extraction again"
        )

        extracted_contents.clear()
        appended_turns = "\n".join(
            f"[role: user] appended turn {i}: MULTI_CHUNK_TAIL_SENTINEL_{i} "
            "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november"
            for i in range(30)
        )
        multi_chunk_append = f"{replacement}\n{appended_turns}"
        chunk_size = DEFAULT_RETAIN_CHUNK_SIZE
        previous_chunks = fact_extraction.chunk_text(replacement, chunk_size)
        appended_chunks = fact_extraction.chunk_text(multi_chunk_append, chunk_size)
        changed_indices = [
            i for i, chunk in enumerate(appended_chunks) if i >= len(previous_chunks) or chunk != previous_chunks[i]
        ]
        context_tokens = count_tokens("session transcript")
        changed_tokens = sum(count_tokens(appended_chunks[i]) + context_tokens for i in changed_indices)
        assert count_tokens(multi_chunk_append) > get_config().retain_batch_tokens
        assert len(changed_indices) > 1, "the regression must cross a logical chunk boundary"
        assert changed_tokens <= get_config().retain_batch_tokens

        original_insert = orchestrator._insert_facts_and_links

        async def _synthetic_insert(*args: Any, outbox_callback: Any = None, **kwargs: Any) -> list[list[str]]:
            if outbox_callback is not None:
                await outbox_callback(args[0])
            return [[f"delta-unit-{index}"] for index in range(len(args[3]))]

        monkeypatch.setattr(orchestrator, "_insert_facts_and_links", _synthetic_insert)
        outbox_calls = 0

        multi_chunk_result = await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {
                    "content": multi_chunk_append,
                    "context": "session transcript",
                    "document_id": document_id,
                }
            ],
            request_context=request_context,
            outbox_callback=_record_outbox,
        )

        multi_chunk_extracted = "\n".join(extracted_contents)
        assert multi_chunk_result == [[f"delta-unit-{i}" for i in range(len(changed_indices))]], (
            "multi-chunk Delta unit IDs must collapse into the submitted document's result slot"
        )
        assert outbox_calls == 1, "multi-chunk delta retain must commit the outbox exactly once"
        assert len(extracted_contents) > 1, "the test must extract more than one changed logical chunk"
        assert "MULTI_CHUNK_TAIL_SENTINEL_0" in multi_chunk_extracted
        assert "MULTI_CHUNK_TAIL_SENTINEL_29" in multi_chunk_extracted
        assert not any(marker in multi_chunk_extracted for marker in history_markers.values()), (
            "a bounded multi-chunk append re-extracted unchanged history"
        )

        monkeypatch.setattr(orchestrator, "_insert_facts_and_links", original_insert)
        extracted_contents.clear()
        chunk_limited_turns = "\n".join(
            f"[role: user] chunk-limited turn {i}: CHUNK_LIMIT_TAIL_SENTINEL_{i} "
            "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november"
            for i in range(30)
        )
        chunk_limited_append = f"{multi_chunk_append}\n{chunk_limited_turns}"
        chunk_limited_chunks = fact_extraction.chunk_text(chunk_limited_append, chunk_size)
        chunk_limited_indices = [
            i
            for i, chunk in enumerate(chunk_limited_chunks)
            if i >= len(appended_chunks) or chunk != appended_chunks[i]
        ]
        chunk_limited_tokens = sum(
            count_tokens(chunk_limited_chunks[i]) + context_tokens for i in chunk_limited_indices
        )
        assert len(chunk_limited_indices) > 1
        assert chunk_limited_tokens <= get_config().retain_batch_tokens

        original_internal = memory._retain_batch_async_internal
        internal_delta_modes: list[bool] = []

        async def _record_internal_modes(*args: Any, **kwargs: Any) -> Any:
            internal_delta_modes.append(kwargs.get("delta_only", False))
            return await original_internal(*args, **kwargs)

        monkeypatch.setattr(memory, "_retain_batch_async_internal", _record_internal_modes)
        memory._config_resolver._global_config.retain_chunk_batch_size = 1
        outbox_calls = 0
        chunk_limited_result = await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {
                    "content": chunk_limited_append,
                    "context": "session transcript",
                    "document_id": document_id,
                }
            ],
            request_context=request_context,
            outbox_callback=_record_outbox,
        )
        assert len(chunk_limited_result) == 1
        assert internal_delta_modes[0] is True and False in internal_delta_modes, (
            "a changed delta above retain_chunk_batch_size must use bounded fallback"
        )
        assert outbox_calls == 1, "chunk-count fallback must commit the outbox exactly once"
        monkeypatch.setattr(memory, "_retain_batch_async_internal", original_internal)
        memory._config_resolver._global_config.retain_chunk_batch_size = 100

        multi_chunk_replacement = chunk_limited_append
        for turn, marker in history_markers.items():
            multi_chunk_replacement = multi_chunk_replacement.replace(marker, f"CHANGED_HISTORY_{turn}")
        rewritten_chunks = fact_extraction.chunk_text(multi_chunk_replacement, chunk_size)
        rewritten_indices = [
            i
            for i, chunk in enumerate(rewritten_chunks)
            if i >= len(chunk_limited_chunks) or chunk != chunk_limited_chunks[i]
        ]
        rewritten_tokens = sum(count_tokens(rewritten_chunks[i]) + context_tokens for i in rewritten_indices)
        assert rewritten_tokens > get_config().retain_batch_tokens, "the rewrite must exercise bounded fallback"
        internal_delta_modes.clear()
        monkeypatch.setattr(memory, "_retain_batch_async_internal", _record_internal_modes)
        outbox_calls = 0
        fallback_result = await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {
                    "content": multi_chunk_replacement,
                    "context": "session transcript",
                    "document_id": document_id,
                }
            ],
            request_context=request_context,
            outbox_callback=_record_outbox,
        )
        assert len(fallback_result) == 1, "bounded fallback must preserve one result slot per submitted document"
        assert internal_delta_modes[0] is True and False in internal_delta_modes, (
            "a changed delta above retain_batch_tokens must use bounded fallback"
        )
        assert outbox_calls == 1, "token-count fallback must commit the outbox exactly once"
        stored = await memory.get_document(document_id, bank_id, request_context=request_context)
        assert stored is not None
        assert stored["original_text"] == multi_chunk_replacement
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
        clear_config_cache()
