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

import pytest

from hindsight_api.config import clear_config_cache


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
        f"[role: user] turn {i}: alpha bravo charlie delta echo "
        f"foxtrot golf hotel india juliet"
        for i in range(20)
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

        doc_initial = await memory.get_document(
            document_id, bank_id, request_context=request_context
        )
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

        doc_replaced = await memory.get_document(
            document_id, bank_id, request_context=request_context
        )
        assert doc_replaced is not None

        stored = doc_replaced["original_text"]
        assert len(stored) == len(replacement_body), (
            f"stored body length {len(stored)} != submitted length "
            f"{len(replacement_body)} — partial replacement persisted"
        )
        assert stored == replacement_body, (
            "stored original_text does not exactly match the submitted "
            "replacement body"
        )

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

            doc = await memory.get_document(
                document_id, bank_id, request_context=request_context
            )
            assert doc is not None, f"attempt {attempt}: document missing after retain"
            assert doc["original_text"] == replacement_body, (
                f"attempt {attempt}: stored body diverged from submitted body "
                f"(stored {len(doc['original_text'])} chars, "
                f"submitted {len(replacement_body)} chars)"
            )

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
