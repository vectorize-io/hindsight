"""An oversized re-retain must not leave the previous version's facts behind.

Companion to ``test_delta_oversized_replacement.py``, which pins what the split
path re-*extracts*. This file pins what it *stores*.

A replacement body over ``retain_batch_tokens`` is cut into sub-batches before
any retain logic runs, and only sub-batch 1 carries ``is_first_batch=True`` —
the flag that makes ``handle_document_tracking`` cascade-delete the outgoing
version. Sub-batch 1 takes the delta path (it sees only its own slice); every
later sub-batch runs in *recovery* mode, because the document row already
carries the new content hash, and recovery deliberately does NOT delete: it
calls ``upsert_document_metadata`` and skips the chunks whose hash is already
committed.

A CHANGED chunk hashes differently, so it is (correctly) re-extracted — but no
one deleted the facts the previous version left on that same ``chunk_id``, so
both generations end up stored. Re-editing the same section again piles another
generation on top.
"""

from collections import Counter
from datetime import datetime, timezone

import pytest

from hindsight_api.config import clear_config_cache
from hindsight_api.engine.memory_engine import count_tokens
from tests.test_delta_oversized_replacement import _ExtractionSpy

# Each section is just under retain_chunk_size, so the chunker emits a stable
# chunk per section and only the edited section's chunk changes.
_SECTION_REPEATS = 117
_BASE_SECTIONS = 10
_EDITED_SECTION = 9  # late in the document: several sub-batches in
_OVERSIZED_BATCH_TOKENS = 300
_REVISIONS = 3


def _section(idx: int, revision: int = 0, *, head_edit: bool = False) -> str:
    marker = f"MARKER{idx:02d}"
    head = f"Section {idx:02d} {marker} REVISED." if head_edit else f"Section {idx:02d} {marker}."
    body = head + " " + f"{marker} filler word here. " * _SECTION_REPEATS
    return body + f"Tail revision {revision:02d} word."


def _body(revision: int) -> str:
    """The same document every time, except section ``_EDITED_SECTION``'s tail."""
    return "\n\n".join(_section(i, revision if i == _EDITED_SECTION else 0) for i in range(_BASE_SECTIONS))


@pytest.fixture(autouse=True)
def _fast_retain_env(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_ENABLE_AUTO_CONSOLIDATION", "false")
    monkeypatch.setenv("HINDSIGHT_API_ENABLE_OBSERVATIONS", "false")
    clear_config_cache()
    yield
    clear_config_cache()


async def _retain_revisions(memory, request_context, bank_id: str, document_id: str) -> list[Counter]:
    """Retain the document once per revision; return each pass's facts-per-chunk histogram."""
    histograms: list[Counter] = []
    for revision in range(_REVISIONS):
        await memory.retain_async(
            bank_id=bank_id,
            content=_body(revision),
            context="notes",
            document_id=document_id,
            request_context=request_context,
        )
        units = await memory.list_memory_units(
            bank_id, document_id=document_id, limit=2000, request_context=request_context
        )
        histograms.append(Counter(str(u.get("chunk_id", "")).rsplit("_", 1)[-1] for u in units["items"]))
    return histograms


@pytest.mark.asyncio
async def test_replacement_within_budget_replaces_the_changed_chunks_facts(memory, request_context, monkeypatch):
    """Control: inside the transport budget, re-editing one section replaces that
    chunk's facts rather than adding a second copy."""
    bank_id = f"test_stale_control_{datetime.now(timezone.utc).timestamp()}"
    monkeypatch.setenv("HINDSIGHT_API_RETAIN_BATCH_TOKENS", "100000")  # no splitting
    clear_config_cache()
    try:
        histograms = await _retain_revisions(memory, request_context, bank_id, "doc-stale-control")
        first = histograms[0]
        for pass_no, hist in enumerate(histograms):
            assert sum(hist.values()) == sum(first.values()), (
                f"pass {pass_no}: total facts moved from {sum(first.values())} to {sum(hist.values())}"
            )
            assert max(hist.values()) == max(first.values()), f"pass {pass_no}: a chunk gained facts — {hist}"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_oversized_replacement_does_not_accumulate_stale_facts(memory, request_context, monkeypatch):
    """Repro: over the transport budget, the edited section's chunk keeps every
    previous version's facts as well as the new ones."""
    bank_id = f"test_stale_oversized_{datetime.now(timezone.utc).timestamp()}"
    monkeypatch.setenv("HINDSIGHT_API_RETAIN_BATCH_TOKENS", str(_OVERSIZED_BATCH_TOKENS))
    clear_config_cache()
    try:
        histograms = await _retain_revisions(memory, request_context, bank_id, "doc-stale-oversized")
        first = histograms[0]
        per_chunk = max(first.values())
        for pass_no, hist in enumerate(histograms):
            worst_chunk, worst_count = hist.most_common(1)[0]
            assert worst_count == per_chunk, (
                f"after re-retain #{pass_no}, chunk {worst_chunk} holds {worst_count} facts where a "
                f"chunk holds {per_chunk} — the previous version's facts for the edited chunk were "
                f"never deleted (totals per pass: {[sum(h.values()) for h in histograms]})"
            )
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_oversized_replacement_extracts_a_late_edit(memory, request_context, monkeypatch):
    """The skip is keyed on the chunk hash, so an edit landing in a LATE sub-batch
    still reaches extraction — recovery only skips chunks that are byte-identical
    to an already-committed one."""
    bank_id = f"test_stale_late_edit_{datetime.now(timezone.utc).timestamp()}"
    document_id = "doc-stale-late-edit"
    try:
        await memory.retain_async(
            bank_id=bank_id,
            content=_body(0),
            context="notes",
            document_id=document_id,
            request_context=request_context,
        )
        monkeypatch.setenv("HINDSIGHT_API_RETAIN_BATCH_TOKENS", str(_OVERSIZED_BATCH_TOKENS))
        clear_config_cache()
        spy = _ExtractionSpy()
        spy.install(monkeypatch)
        await memory.retain_async(
            bank_id=bank_id,
            content=_body(1),
            context="notes",
            document_id=document_id,
            request_context=request_context,
        )
        assert "Tail revision 01" in "\n".join(spy.texts), (
            "the edited tail of a late sub-batch never reached fact extraction"
        )
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_oversized_replacement_drops_the_removed_tail(memory, request_context, monkeypatch):
    """A shrinking oversized replacement must not leave the dropped sections behind.

    Only the retain's first sub-batch cascade-deletes the outgoing version, and it resolves
    through delta retain, which touches nothing outside its own slice — so the chunks past the
    new document's end were never deleted and their facts stayed recallable.
    """
    bank_id = f"test_stale_shrink_{datetime.now(timezone.utc).timestamp()}"
    document_id = "doc-stale-shrink"
    monkeypatch.setenv("HINDSIGHT_API_RETAIN_BATCH_TOKENS", str(_OVERSIZED_BATCH_TOKENS))
    clear_config_cache()
    try:
        await memory.retain_async(
            bank_id=bank_id,
            content="\n\n".join(_section(i) for i in range(_BASE_SECTIONS)),
            context="notes",
            document_id=document_id,
            request_context=request_context,
        )
        kept = 6
        await memory.retain_async(
            bank_id=bank_id,
            content="\n\n".join(_section(i) for i in range(kept)),
            context="notes",
            document_id=document_id,
            request_context=request_context,
        )
        units = await memory.list_memory_units(
            bank_id, document_id=document_id, limit=2000, request_context=request_context
        )
        stored = "\n".join(str(u) for u in units["items"])
        survivors = [f"MARKER{i:02d}" for i in range(kept, _BASE_SECTIONS) if f"MARKER{i:02d}" in stored]
        assert survivors == [], f"sections {survivors} were removed from the document but their facts are still stored"
        assert f"MARKER{kept - 1:02d}" in stored, "the sections that survived the edit lost their facts"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_oversized_first_chunk_edit_keeps_the_rest_of_the_document(memory, request_context, monkeypatch):
    """Editing the FIRST chunk must not cost a full re-extraction.

    Delta retain runs on sub-batch 1 only and sees only that slice, so every stored chunk after
    it looked "removed": tombstoning them left the following sub-batches nothing to skip and the
    whole document went back through the LLM.
    """
    bank_id = f"test_stale_head_{datetime.now(timezone.utc).timestamp()}"
    document_id = "doc-stale-head"
    v1 = "\n\n".join(_section(i) for i in range(_BASE_SECTIONS))
    try:
        await memory.retain_async(
            bank_id=bank_id,
            content=v1,
            context="notes",
            document_id=document_id,
            request_context=request_context,
        )
        monkeypatch.setenv("HINDSIGHT_API_RETAIN_BATCH_TOKENS", str(_OVERSIZED_BATCH_TOKENS))
        clear_config_cache()
        spy = _ExtractionSpy()
        spy.install(monkeypatch)
        await memory.retain_async(
            bank_id=bank_id,
            content="\n\n".join(_section(i, head_edit=(i == 0)) for i in range(_BASE_SECTIONS)),
            context="notes",
            document_id=document_id,
            request_context=request_context,
        )
        re_extracted = [f"MARKER{i:02d}" for i in range(1, _BASE_SECTIONS) if f"MARKER{i:02d}" in "\n".join(spy.texts)]
        assert re_extracted == [], (
            f"editing the first chunk re-extracted {re_extracted} — {sum(count_tokens(t) for t in spy.texts):,} "
            f"tokens for a one-chunk edit of a {count_tokens(v1):,}-token document"
        )
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
