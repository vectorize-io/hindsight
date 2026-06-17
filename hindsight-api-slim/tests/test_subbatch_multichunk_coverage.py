"""Regression: sub-batch slices that each span MULTIPLE extraction chunks must
keep full chunk coverage on BOTH the sync (inline) and async (submitted) retain
paths.

Two distinct bugs hid behind the same symptom — ingesting a large plain-text
document dropped most of its body (and any fact past the first slice). Both only
trigger when an oversized single item is split into sequential sub-batches whose
*slices each re-chunk into several extraction chunks* (the default config: batch
tokens 10k → ~30k-char slices, re-chunked at 3k → ~10 chunks/slice):

1. chunk_index offset (sync + async). retain_batch_async advanced the per-document
   chunk_index cursor by re-chunking ``item["content"]`` AFTER the orchestrator
   had consumed (popped) it — ``chunk_text("")`` returns ``[""]`` (count 1), so
   the cursor moved by 1 per sub-batch instead of by the real chunk count. Later
   slices restarted ~1 slot in, colliding ``chunk_id = {bank}_{doc}_{index}`` and
   overwriting earlier chunks via upsert.

2. whole-document recovery skip (async only). All sub-batches of one submitted
   operation share one ``operation_id``; the first slice stamps the document into
   ``result_metadata.facts_committed_document_ids``. The crash-recovery fast-path
   then saw every later slice's document already "committed" and skipped
   extraction entirely, so only the first slice survived.

The existing #1888 coverage tests use ``RETAIN_BATCH_TOKENS=100`` (a ~300-char
budget, under the chunk size) so every slice collapses to ONE chunk — which masks
both bugs (offset-by-1 happens to equal the real count, and a 1-chunk doc isn't
re-sliced). These tests size the body so each slice fans out to ~6 chunks, with
globally-unique tokens so no chunk-hash dedup hides a dropped slice, and assert
full coverage + contiguous indices + a needle planted in a late slice.
"""

from datetime import datetime, timezone

import pytest

from hindsight_api.config import clear_config_cache

# Planted in a late paragraph so it lands in a late sub-batch slice — the first
# thing either bug drops (mirrors the field-reported "165 commits" fact that
# vanished on the async path). A single no-space token so it can't straddle a
# chunk boundary (a multi-word phrase can split across two chunks at this test's
# small 500-char chunk size and read as "dropped" when it wasn't).
NEEDLE = "NEEDLE_165_COMMITS_MERGED_INTO_THE_MAIN_BRANCH"


def _ts() -> float:
    return datetime.now(timezone.utc).timestamp()


@pytest.fixture(autouse=True)
def _multichunk_split_env(monkeypatch):
    # Small extraction chunks (500 chars) with a batch-token budget whose char
    # budget (700 * 3 = 2100) spans several chunks, so each oversized sub-batch
    # slice fans out to ~6 extraction chunks. Skip consolidation/observations to
    # keep the test fast and deterministic.
    monkeypatch.setenv("HINDSIGHT_API_RETAIN_CHUNK_SIZE", "500")
    monkeypatch.setenv("HINDSIGHT_API_RETAIN_BATCH_TOKENS", "700")
    monkeypatch.setenv("HINDSIGHT_API_ENABLE_AUTO_CONSOLIDATION", "false")
    monkeypatch.setenv("HINDSIGHT_API_ENABLE_OBSERVATIONS", "false")
    clear_config_cache()
    yield
    clear_config_cache()


def _make_body(paragraphs: int = 24, needle_at: int = 20) -> str:
    """Plain-text transcript whose every token is unique across the whole body,
    so no two extraction chunks can hash-collide (a real content-hash collision
    would legitimately dedup and mask a dropped slice). The needle sits in a late
    paragraph."""
    lines = []
    for i in range(paragraphs):
        toks = " ".join(f"w{i:03d}t{j:03d}" for j in range(60))
        if i == needle_at:
            lines.append(f"[Turn {i}] Assistant: {NEEDLE} fact {toks}")
        else:
            lines.append(f"[Turn {i}] Assistant: progress {i}: {toks}")
    return "\n\n".join(lines)


async def _chunk_coverage(memory, bank_id, document_id, request_context):
    doc = await memory.get_document(document_id, bank_id, request_context=request_context)
    assert doc is not None
    original_len = len(doc["original_text"])
    chunks = await memory.list_document_chunks(bank_id, document_id, limit=10000, request_context=request_context)
    items = chunks["items"]
    sum_chunk_text = sum(len(c["chunk_text"]) for c in items)
    indices = sorted(c["chunk_index"] for c in items)
    needle_present = any(NEEDLE in c["chunk_text"] for c in items)
    return original_len, sum_chunk_text, indices, needle_present


def _assert_full_coverage(label, original_len, sum_chunk_text, indices, needle_present):
    # Sanity: the body must actually fan out to many chunks across several
    # multi-chunk slices, or the test wouldn't exercise the bug at all.
    assert len(indices) >= 16, f"{label}: only {len(indices)} chunks — body too small to exercise multi-chunk slices"
    assert sum_chunk_text >= original_len * 0.9, (
        f"{label}: chunks cover only {sum_chunk_text}/{original_len} chars "
        f"(~{100 * sum_chunk_text // original_len}%) — a sub-batch slice was overwritten or skipped"
    )
    assert indices == list(range(len(indices))), (
        f"{label}: chunk_index sequence is not contiguous: {indices} — sub-batch slices collided on chunk_id"
    )
    assert needle_present, f"{label}: the late-slice needle fact was dropped (offset collision or recovery skip)"


@pytest.mark.asyncio
async def test_sync_inline_multichunk_subbatch_coverage(memory, request_context):
    """Sync inline path (retain_batch_async): an oversized doc whose slices each
    span several extraction chunks must keep full coverage (offset bug)."""
    bank_id = f"test_multichunk_sync_{_ts()}"
    document_id = "doc-multichunk-sync"
    try:
        body = _make_body()
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": body, "context": "big doc", "document_id": document_id}],
            request_context=request_context,
        )
        cov = await _chunk_coverage(memory, bank_id, document_id, request_context)
        _assert_full_coverage("sync", *cov)
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_async_submit_multichunk_subbatch_coverage(memory, request_context):
    """Async submit path (submit_async_retain → child op → worker): the same
    oversized doc must keep full coverage too. Exercises both the offset bug and
    the shared-operation_id whole-document recovery skip."""
    import asyncio

    bank_id = f"test_multichunk_async_{_ts()}"
    document_id = "doc-multichunk-async"
    try:
        body = _make_body()
        result = await memory.submit_async_retain(
            bank_id=bank_id,
            contents=[{"content": body, "context": "big doc", "document_id": document_id}],
            request_context=request_context,
        )
        operation_id = result["operation_id"]

        # SyncTaskBackend (test backend) drains children inline; wait for the
        # parent to reach a terminal state before reading chunks.
        status = None
        for _ in range(600):
            status = await memory.get_operation_status(
                bank_id=bank_id, operation_id=operation_id, request_context=request_context
            )
            if status["status"] in ("completed", "failed"):
                break
            await asyncio.sleep(0.1)
        assert status is not None and status["status"] == "completed", (
            f"async retain did not complete: {status['status'] if status else 'no status'}"
        )

        cov = await _chunk_coverage(memory, bank_id, document_id, request_context)
        _assert_full_coverage("async", *cov)
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
