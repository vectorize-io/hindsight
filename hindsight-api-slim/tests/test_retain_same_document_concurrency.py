"""
Concurrency regression tests for retains that target the SAME document.

These guard two properties of the retain pipeline under concurrent same-document
upserts:

1. Identical content is never re-extracted. Ten concurrent retains of content
   that already matches the stored document must make ZERO LLM extraction calls
   (they take the delta metadata-only path).

2. Concurrent same-document retains never deadlock or crash, regardless of
   whether the new content overlaps the stored chunks partially (delta path) or
   not at all (streaming fallback).

We instrument the real extraction entry point
(`fact_extraction.extract_facts_from_contents`) to count LLM extraction calls,
and classify each request's outcome.

The local sentence-transformers model segfaults under heavy in-process async
concurrency (a torch limitation, not a product issue — in production embeddings
are served out-of-process). Since these tests only care about extraction counts
and DB-level concurrency behavior, we swap in a deterministic, torch-free stub
embeddings.
"""

import asyncio
import hashlib
import logging
import struct
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from hindsight_api.engine.embeddings import Embeddings
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.retain import fact_extraction
from hindsight_api.engine.task_backend import SyncTaskBackend

logger = logging.getLogger(__name__)

_DIM = 384


def _ts():
    return datetime.now(timezone.utc).timestamp()


class _StubEmbeddings(Embeddings):
    """Deterministic, torch-free embeddings. Same text -> same vector."""

    @property
    def provider_name(self) -> str:
        return "stub"

    @property
    def dimension(self) -> int:
        return _DIM

    async def initialize(self) -> None:
        return None

    def encode(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            seed = hashlib.sha256(t.encode("utf-8")).digest()
            vals: list[float] = []
            i = 0
            while len(vals) < _DIM:
                chunk = hashlib.sha256(seed + struct.pack("<I", i)).digest()
                for j in range(0, len(chunk), 4):
                    if len(vals) >= _DIM:
                        break
                    (u,) = struct.unpack("<I", chunk[j : j + 4])
                    vals.append((u % 100000) / 100000.0)
                i += 1
            out.append(vals)
        return out


@pytest_asyncio.fixture(scope="function")
async def memory_stub_emb(pg0_db_url, cross_encoder, query_analyzer):
    mem = MemoryEngine(
        db_url=pg0_db_url,
        memory_llm_provider="mock",
        memory_llm_api_key="",
        memory_llm_model="mock",
        embeddings=_StubEmbeddings(),
        cross_encoder=cross_encoder,
        query_analyzer=query_analyzer,
        pool_min_size=1,
        pool_max_size=20,
        run_migrations=False,
        task_backend=SyncTaskBackend(),
    )
    await mem.initialize()
    # These tests exercise the retain WRITE path under same-document concurrency
    # (extraction skipping + the streaming deadlock guard). Auto-consolidation
    # runs after every retain and dominates wall-clock (100 memories x many LLM
    # batches per retain) while being irrelevant to what we assert — disable it so
    # the tests stay fast and don't time out on CI. The resolver captures the
    # global config at init, so this only affects this engine instance.
    mem._config_resolver._global_config.enable_auto_consolidation = False
    yield mem
    # Let any fire-and-forget post-processing settle before closing the pool.
    await asyncio.sleep(2)
    try:
        if mem._pool and not mem._pool._closing:
            await mem.close()
    except Exception:
        pass


class _ExtractionCounter:
    """Wraps fact_extraction.extract_facts_from_contents to count real calls."""

    def __init__(self):
        self.calls = 0
        self.total_contents = 0
        self._orig = fact_extraction.extract_facts_from_contents
        self._lock = asyncio.Lock()

    def install(self):
        counter = self

        async def _wrapped(contents, *args, **kwargs):
            async with counter._lock:
                counter.calls += 1
                counter.total_contents += len(contents)
            return await counter._orig(contents, *args, **kwargs)

        fact_extraction.extract_facts_from_contents = _wrapped

    def uninstall(self):
        fact_extraction.extract_facts_from_contents = self._orig


# Multi-chunk bodies. Default retain_chunk_size is 3000 chars; each segment is
# ~3500 chars so the document spans several chunks and the delta path can find
# SOME unchanged chunks (required for delta to apply at all).
def _segment(label: str, n: int) -> str:
    return (f"[{label}] " + " ".join(f"Detail {label}-{n}-{k} about the topic." for k in range(120))).ljust(3500, ".")


_SEGMENTS_BASE = [_segment("base", i) for i in range(5)]
_BODY_BASE = "\n\n".join(_SEGMENTS_BASE)

# Partial overlap: identical to base except the LAST segment is changed.
_SEGMENTS_PARTIAL = _SEGMENTS_BASE[:-1] + [_segment("changed", 99)]
_BODY_PARTIAL = "\n\n".join(_SEGMENTS_PARTIAL)

# Fully different: no chunk overlap with base -> delta bails to streaming.
_BODY_DIFFERENT = "\n\n".join(_segment("other", i) for i in range(5))


async def _concurrent_retains(memory, bank_id, document_id, body, n, request_context, stagger: float = 0.0):
    counter = _ExtractionCounter()
    counter.install()

    async def _one(idx: int):
        try:
            await memory.retain_async(
                bank_id=bank_id,
                content=body,
                context="ctx",
                document_id=document_id,
                request_context=request_context,
            )
            return "ok"
        except Exception as e:  # noqa: BLE001 — we want to classify failures
            logger.warning("retain %d raised: %r", idx, e)
            return type(e).__name__

    async def _one_staggered(idx: int, delay: float):
        await asyncio.sleep(idx * delay)
        return await _one(idx)

    try:
        if stagger:
            outcomes = await asyncio.gather(*[_one_staggered(i, stagger) for i in range(n)])
        else:
            outcomes = await asyncio.gather(*[_one(i) for i in range(n)])
    finally:
        counter.uninstall()

    summary: dict[str, int] = {}
    for o in outcomes:
        summary[o] = summary.get(o, 0) + 1
    return counter, summary


@pytest.mark.asyncio
async def test_concurrent_identical_retains_skip_extraction(memory_stub_emb, request_context):
    """10 concurrent retains of content identical to the stored document must make
    ZERO LLM extraction calls (delta metadata-only path) and all succeed."""
    bank_id = f"concurrent_identical_{_ts()}"
    await memory_stub_emb.retain_async(
        bank_id=bank_id, content=_BODY_BASE, context="ctx", document_id="doc", request_context=request_context
    )
    counter, summary = await _concurrent_retains(memory_stub_emb, bank_id, "doc", _BODY_BASE, 10, request_context)
    logger.warning(
        "identical-to-stored: extraction_calls=%d contents=%d outcomes=%s",
        counter.calls,
        counter.total_contents,
        summary,
    )
    assert summary == {"ok": 10}, f"all retains should succeed, got {summary}"
    assert counter.calls == 0, f"identical content must not be re-extracted, got {counter.calls} extraction calls"


@pytest.mark.asyncio
async def test_concurrent_partial_overlap_retains_no_crash(memory_stub_emb, request_context):
    """10 concurrent retains with partial chunk overlap (the delta hash-mismatch
    race). Must complete without crashing or deadlocking."""
    bank_id = f"concurrent_partial_{_ts()}"
    await memory_stub_emb.retain_async(
        bank_id=bank_id, content=_BODY_BASE, context="ctx", document_id="doc", request_context=request_context
    )
    counter, summary = await _concurrent_retains(memory_stub_emb, bank_id, "doc", _BODY_PARTIAL, 10, request_context)
    logger.warning(
        "partial-overlap race: extraction_calls=%d contents=%d outcomes=%s",
        counter.calls,
        counter.total_contents,
        summary,
    )
    assert summary == {"ok": 10}, f"all retains should succeed without deadlock/crash, got {summary}"


@pytest.mark.asyncio
async def test_staggered_partial_overlap_retains_avoid_redundant_extraction(memory_stub_emb, request_context):
    """Staggered concurrent retains with partial overlap: once the first writer
    commits, later writers must observe the new state and skip extraction. The
    pre-extraction freshness recheck (and the delta no-change path) should keep
    total extraction calls well below the request count — i.e. we do NOT re-run
    the LLM once per racing request."""
    bank_id = f"staggered_partial_{_ts()}"
    await memory_stub_emb.retain_async(
        bank_id=bank_id, content=_BODY_BASE, context="ctx", document_id="doc", request_context=request_context
    )
    n = 10
    counter, summary = await _concurrent_retains(
        memory_stub_emb, bank_id, "doc", _BODY_PARTIAL, n, request_context, stagger=0.25
    )
    logger.warning(
        "staggered partial-overlap: extraction_calls=%d contents=%d outcomes=%s",
        counter.calls,
        counter.total_contents,
        summary,
    )
    assert summary == {"ok": n}, f"all retains should succeed, got {summary}"
    # The invariant we guard: staggered same-document retains do NOT re-extract
    # once per request — once a writer commits, the freshness recheck / delta
    # no-change path lets later writers skip the LLM, so the total stays strictly
    # below one extraction per request (the un-converged worst case is `n`).
    # We assert `< n` rather than a tight constant because the exact number of
    # converged requests is timing-dependent and varies with CI speed (observed:
    # 1 locally, 4 on CI); a regression that re-extracts per request would hit n.
    assert counter.calls < n, f"staggered retains should avoid per-request extraction, got {counter.calls}/{n}"


@pytest.mark.asyncio
async def test_concurrent_fully_different_retains_no_deadlock(memory_stub_emb, request_context):
    """10 concurrent retains with fully-different content (no chunk overlap) force
    the streaming fallback. Regression guard for the same-document deadlock."""
    bank_id = f"concurrent_different_{_ts()}"
    await memory_stub_emb.retain_async(
        bank_id=bank_id, content=_BODY_BASE, context="ctx", document_id="doc", request_context=request_context
    )
    counter, summary = await _concurrent_retains(memory_stub_emb, bank_id, "doc", _BODY_DIFFERENT, 10, request_context)
    logger.warning(
        "fully-different streaming: extraction_calls=%d contents=%d outcomes=%s",
        counter.calls,
        counter.total_contents,
        summary,
    )
    assert summary == {"ok": 10}, f"all retains should succeed without deadlock, got {summary}"
