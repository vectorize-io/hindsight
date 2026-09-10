"""A document written BEFORE the #4244 chunk-id fix keeps working after it.

``chunk_ids.build_chunk_id`` escapes the separator inside the bank and document id, so
for a bank or document whose id contains ``_`` the id it produces differs from the plain
``{bank_id}_{document_id}_{index}`` join that retain used to write. Nothing rewrites the
rows that join produced -- there is no migration -- so a real database holds legacy ids
indefinitely, and both reading them and re-retaining over them have to stay correct.

The setup here rewrites a freshly retained document's chunk ids back to the legacy shape,
which is the only way to get pre-fix rows out of post-fix code. The FK from
``memory_units.chunk_id`` has no ``ON UPDATE CASCADE``, hence insert-repoint-delete rather
than a plain ``UPDATE``.
"""

from datetime import datetime, timezone

import pytest

from hindsight_api.config import clear_config_cache
from hindsight_api.engine.chunk_ids import build_chunk_id

# Blank-line separated sections, each repeating its own marker. The edit below swaps one
# section's marker for a same-length one, so the body's length -- and therefore the chunk
# boundaries and the chunk count -- are identical before and after: a delta retain whose
# only difference is the text of the chunks covering that one section.
_SECTION_REPEATS = 117
_SECTIONS = 4
_EDITED = 2
_ORIGINAL_MARKER = f"MARKER{_EDITED:02d}"
_EDITED_MARKER = "SWAPPED2"
assert len(_ORIGINAL_MARKER) == len(_EDITED_MARKER)

# The ids that make this test worth writing: both carry the separator, so the legacy join
# and the escaped one disagree.
_BANK_PREFIX = "legacy_chunk_ids"
_DOCUMENT_ID = "doc_with_underscores"


def _section(idx: int, marker: str) -> str:
    return f"Section {idx:02d} {marker}. " + f"{marker} filler word here. " * _SECTION_REPEATS


def _body(edited: bool = False) -> str:
    return "\n\n".join(
        _section(i, _EDITED_MARKER if (edited and i == _EDITED) else f"MARKER{i:02d}") for i in range(_SECTIONS)
    )


def _legacy_chunk_id(bank_id: str, document_id: str, index: int) -> str:
    """The id retain wrote before #4244 -- a plain join, with nothing escaped."""
    return f"{bank_id}_{document_id}_{index}"


@pytest.fixture(autouse=True)
def _fast_retain_env(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_ENABLE_AUTO_CONSOLIDATION", "false")
    monkeypatch.setenv("HINDSIGHT_API_ENABLE_OBSERVATIONS", "false")
    monkeypatch.setenv("HINDSIGHT_API_RETAIN_BATCH_TOKENS", "100000")
    clear_config_cache()
    yield
    clear_config_cache()


async def _rewrite_to_legacy_ids(memory, bank_id: str, document_id: str) -> dict[int, str]:
    """Age the document's chunk rows back to the pre-#4244 id shape.

    Returns the legacy id per chunk_index. Direct SQL because this forges a state the
    public API cannot produce any more: writing a legacy id is exactly what the fix removed.
    """
    backend = await memory._get_backend()
    legacy_by_index: dict[int, str] = {}
    async with backend.acquire() as conn:
        rows = await conn.fetch(
            "SELECT chunk_id, chunk_index FROM chunks WHERE bank_id = $1 AND document_id = $2",
            bank_id,
            document_id,
        )
        for row in rows:
            legacy = _legacy_chunk_id(bank_id, document_id, row["chunk_index"])
            legacy_by_index[row["chunk_index"]] = legacy
            assert legacy != row["chunk_id"], "the ids under test must differ from the escaped ones"
            await conn.execute(
                """
                INSERT INTO chunks (chunk_id, document_id, bank_id, chunk_text, chunk_index, content_hash, created_at)
                SELECT $1, document_id, bank_id, chunk_text, chunk_index, content_hash, created_at
                FROM chunks WHERE chunk_id = $2
                """,
                legacy,
                row["chunk_id"],
            )
            await conn.execute(
                "UPDATE memory_units SET chunk_id = $1 WHERE bank_id = $2 AND chunk_id = $3",
                legacy,
                bank_id,
                row["chunk_id"],
            )
            await conn.execute("DELETE FROM chunks WHERE chunk_id = $1", row["chunk_id"])
    return legacy_by_index


async def _stored_markers(memory, request_context, bank_id: str, document_id: str) -> set[str]:
    units = await memory.list_memory_units(
        bank_id, document_id=document_id, limit=4000, request_context=request_context
    )
    stored = "\n".join(str(u) for u in units["items"])
    candidates = [f"MARKER{i:02d}" for i in range(_SECTIONS)] + [_EDITED_MARKER]
    return {marker for marker in candidates if marker in stored}


async def _retain(memory, request_context, bank_id: str, *, edited: bool) -> None:
    await memory.retain_async(
        bank_id=bank_id,
        content=_body(edited),
        context="notes",
        document_id=_DOCUMENT_ID,
        request_context=request_context,
    )


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_legacy_document_is_still_readable(memory, request_context):
    """A document whose chunks carry pre-#4244 ids reads back through the public API."""
    bank_id = f"{_BANK_PREFIX}_read_{datetime.now(timezone.utc).timestamp()}"
    try:
        await _retain(memory, request_context, bank_id, edited=False)
        legacy_by_index = await _rewrite_to_legacy_ids(memory, bank_id, _DOCUMENT_ID)
        assert legacy_by_index, "the retain stored no chunks"

        listed = await memory.list_document_chunks(bank_id, _DOCUMENT_ID, limit=100, request_context=request_context)
        assert {item["chunk_id"] for item in listed["items"]} == set(legacy_by_index.values())

        # The addressed chunk route resolves a legacy id, and answers for the right bank.
        for index, legacy in legacy_by_index.items():
            chunk = await memory.get_chunk(legacy, request_context=request_context)
            assert chunk is not None, f"legacy chunk id {legacy} no longer resolves"
            assert chunk["bank_id"] == bank_id
            assert chunk["document_id"] == _DOCUMENT_ID
            assert chunk["chunk_index"] == index

        assert await _stored_markers(memory, request_context, bank_id, _DOCUMENT_ID) == {
            f"MARKER{i:02d}" for i in range(_SECTIONS)
        }
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_updating_a_chunk_of_a_legacy_document_replaces_it(memory, request_context):
    """Editing one section of a legacy document swaps those chunks and leaves the rest alone.

    The delta path deletes the chunks it replaces by the ids it read from the database, so
    an edited chunk's legacy row goes and its escaped replacement takes its place -- one row
    per chunk_index, with no orphan left behind still carrying the old facts.
    """
    bank_id = f"{_BANK_PREFIX}_update_{datetime.now(timezone.utc).timestamp()}"
    try:
        await _retain(memory, request_context, bank_id, edited=False)
        legacy_by_index = await _rewrite_to_legacy_ids(memory, bank_id, _DOCUMENT_ID)

        await _retain(memory, request_context, bank_id, edited=True)

        listed = await memory.list_document_chunks(bank_id, _DOCUMENT_ID, limit=100, request_context=request_context)
        by_index = {item["chunk_index"]: item for item in listed["items"]}
        assert sorted(by_index) == sorted(legacy_by_index), (
            f"the edit should leave one row per chunk_index, got {[item['chunk_id'] for item in listed['items']]}"
        )

        rewritten = {index for index, item in by_index.items() if _EDITED_MARKER in item["chunk_text"]}
        assert rewritten, "no chunk carries the edited text"
        for index, item in by_index.items():
            if index in rewritten:
                assert item["chunk_id"] == build_chunk_id(bank_id, _DOCUMENT_ID, index), (
                    "a rewritten chunk should carry the escaped id"
                )
            else:
                assert item["chunk_id"] == legacy_by_index[index], (
                    "an untouched chunk must keep the legacy id it was stored under"
                )
            # Whichever id a chunk ended up with, it still addresses this bank's row.
            chunk = await memory.get_chunk(item["chunk_id"], request_context=request_context)
            assert chunk is not None and chunk["bank_id"] == bank_id

        # The replaced chunks' facts went with them; every other section survived.
        assert await _stored_markers(memory, request_context, bank_id, _DOCUMENT_ID) == {
            f"MARKER{i:02d}" for i in range(_SECTIONS) if i != _EDITED
        } | {_EDITED_MARKER}
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
