"""A re-retain that only REMOVES content must drop the removed chunks.

Delta retain classifies the new body against the stored chunks as
``unchanged / changed / new / removed``. Deleting the tail of a document and
leaving the rest byte-identical produces ``changed=0 new=0 removed=N``: there is
nothing to extract, but there is something to delete.

``_try_delta_retain`` used to short-circuit to the metadata-only path as soon as
the *extraction* list came back empty, which threw the non-empty ``removed`` set
away. The document row was updated to the shrunken body while its chunks and
their facts stayed exactly where they were, so the removed sections remained
recallable as part of a document that no longer contained them — and the state
was stable, because the stored ``content_hash`` now matched the new body, so
re-submitting it was a no-op that never revisited the leftovers.

The same edit combined with *any* other change was always correct (the write path
deletes ``removed`` alongside the changed chunks it rewrites), which is what kept
this invisible: it takes a pure deletion to hit.
"""

from datetime import datetime, timezone

import pytest

from hindsight_api.config import clear_config_cache
from hindsight_api.engine.retain import fact_extraction

# One native chunk per section: each is just under retain_chunk_size and sections
# are blank-line separated, so dropping a trailing section leaves every surviving
# chunk byte-identical at the same chunk_index.
_SECTION_REPEATS = 117
_BASE_SECTIONS = 10
_KEPT_SECTIONS = 6
# Low enough that the replacement is sliced into several sequential retain_batch
# calls, so the same deletion is exercised through the split transport path too.
_OVERSIZED_BATCH_TOKENS = 300


def _section(idx: int) -> str:
    marker = f"MARKER{idx:02d}"
    return f"Section {idx:02d} {marker}. " + f"{marker} filler word here. " * _SECTION_REPEATS


def _body(section_indices) -> str:
    return "\n\n".join(_section(i) for i in section_indices)


@pytest.fixture(autouse=True)
def _fast_retain_env(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_ENABLE_AUTO_CONSOLIDATION", "false")
    monkeypatch.setenv("HINDSIGHT_API_ENABLE_OBSERVATIONS", "false")
    clear_config_cache()
    yield
    clear_config_cache()


class _ExtractionSpy:
    """Records the content fed to LLM fact extraction on each call."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def install(self, monkeypatch) -> None:
        original = fact_extraction.extract_facts_from_contents

        async def _spy(contents, *args, **kwargs):
            self.texts.extend(c.content for c in contents)
            return await original(contents, *args, **kwargs)

        monkeypatch.setattr(fact_extraction, "extract_facts_from_contents", _spy)


async def _stored_markers(memory, request_context, bank_id: str, document_id: str) -> list[str]:
    units = await memory.list_memory_units(
        bank_id, document_id=document_id, limit=4000, request_context=request_context
    )
    stored = "\n".join(str(u) for u in units["items"])
    return [f"MARKER{i:02d}" for i in range(_BASE_SECTIONS) if f"MARKER{i:02d}" in stored]


async def _retain(memory, request_context, bank_id: str, document_id: str, section_indices) -> None:
    await memory.retain_async(
        bank_id=bank_id,
        content=_body(section_indices),
        context="notes",
        document_id=document_id,
        request_context=request_context,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "batch_tokens"),
    [("within_budget", 100_000), ("oversized", _OVERSIZED_BATCH_TOKENS)],
)
async def test_pure_deletion_drops_the_removed_chunks(memory, request_context, monkeypatch, case, batch_tokens):
    """Drop the last four sections, change nothing else — their facts must go."""
    bank_id = f"test_removed_tail_{case}_{datetime.now(timezone.utc).timestamp()}"
    document_id = "doc-removed-tail"
    monkeypatch.setenv("HINDSIGHT_API_RETAIN_BATCH_TOKENS", str(batch_tokens))
    clear_config_cache()
    try:
        await _retain(memory, request_context, bank_id, document_id, range(_BASE_SECTIONS))
        assert await _stored_markers(memory, request_context, bank_id, document_id) == [
            f"MARKER{i:02d}" for i in range(_BASE_SECTIONS)
        ]

        await _retain(memory, request_context, bank_id, document_id, range(_KEPT_SECTIONS))

        markers = await _stored_markers(memory, request_context, bank_id, document_id)
        assert markers == [f"MARKER{i:02d}" for i in range(_KEPT_SECTIONS)], (
            "the sections the replacement dropped are still stored (or the surviving ones were lost)"
        )
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_pure_deletion_only_deletes(memory, request_context, monkeypatch):
    """The surviving chunks are byte-identical, so none of them is re-extracted.

    Guards the other direction of the fix: letting the write path run for a
    removal-only delta must not turn it into a full re-ingest.
    """
    bank_id = f"test_removed_no_reextract_{datetime.now(timezone.utc).timestamp()}"
    document_id = "doc-removed-no-reextract"
    monkeypatch.setenv("HINDSIGHT_API_RETAIN_BATCH_TOKENS", "100000")
    clear_config_cache()
    try:
        await _retain(memory, request_context, bank_id, document_id, range(_BASE_SECTIONS))
        spy = _ExtractionSpy()
        spy.install(monkeypatch)
        await _retain(memory, request_context, bank_id, document_id, range(_KEPT_SECTIONS))
        assert spy.texts == [], f"a removal-only re-retain sent {len(spy.texts)} chunk(s) back through fact extraction"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_unchanged_re_retain_still_takes_the_metadata_only_path(memory, request_context, monkeypatch):
    """Nothing changed at all — still no extraction, and every fact survives."""
    bank_id = f"test_removed_noop_{datetime.now(timezone.utc).timestamp()}"
    document_id = "doc-removed-noop"
    monkeypatch.setenv("HINDSIGHT_API_RETAIN_BATCH_TOKENS", "100000")
    clear_config_cache()
    try:
        await _retain(memory, request_context, bank_id, document_id, range(_BASE_SECTIONS))
        before = await _stored_markers(memory, request_context, bank_id, document_id)
        spy = _ExtractionSpy()
        spy.install(monkeypatch)
        await _retain(memory, request_context, bank_id, document_id, range(_BASE_SECTIONS))
        assert spy.texts == [], "an unchanged re-retain re-extracted content"
        assert await _stored_markers(memory, request_context, bank_id, document_id) == before
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_middle_deletion_still_drops_the_removed_section(memory, request_context, monkeypatch):
    """Control: deleting a section from the MIDDLE shifts every later chunk, so
    delta has changed chunks to extract and always deleted ``removed`` correctly.
    Pinned so the removal-only fix is not mistaken for the whole behaviour."""
    bank_id = f"test_removed_middle_{datetime.now(timezone.utc).timestamp()}"
    document_id = "doc-removed-middle"
    monkeypatch.setenv("HINDSIGHT_API_RETAIN_BATCH_TOKENS", "100000")
    clear_config_cache()
    kept = [i for i in range(_BASE_SECTIONS) if i != 3]
    try:
        await _retain(memory, request_context, bank_id, document_id, range(_BASE_SECTIONS))
        await _retain(memory, request_context, bank_id, document_id, kept)
        assert await _stored_markers(memory, request_context, bank_id, document_id) == [f"MARKER{i:02d}" for i in kept]
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
