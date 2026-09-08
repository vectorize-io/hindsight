"""An append that has to be SPLIT truncates the document to its last slice (#3989).

``update_mode="append"`` reads the stored body, prepends it to the incoming tail
and re-retains the whole thing, so delta can classify the pre-existing chunks as
unchanged and re-extract only what is new. When that combined body exceeds
``retain_batch_tokens`` the splitter cuts it into sequential sub-batch slices —
and what ends up in ``documents.original_text`` is only the LAST slice. Every
earlier turn is dropped from the stored document even though the caller only ever
added content.

The facts survive that first append, because the earlier chunks are still
committed. They do not survive the NEXT one: it prepends the truncated body,
diffs it against the stored chunks, classifies the chunks whose text is no longer
in the body as removed, and tombstones their facts. So the loss lands one
generation behind the truncation, and the document's fact count goes UP and then
DOWN while its content only ever grew.

That is the signature reported in #3989: a session transcript grew 212,335 ->
267,311 characters while its extracted facts fell 197 -> 131. It was attributed
to the coding-agent's client-side replace fallback; this is the server losing the
facts on the ordinary append path, with no error and nothing in the response to
say anything was dropped.

Not reproduced at the shipped default of 10,000 tokens with these sizes — the
exact trigger boundary is not pinned here. 4,000 tokens (~16KB slices) is an
entirely plausible configured value and loses 40% of the document's facts.
"""

import json
from datetime import datetime, timezone

import pytest

from hindsight_api.config import clear_config_cache

# Enough filler that a turn is a chunk's worth of text on its own, so the combined
# body spans many chunks and has to be split at the budgets below.
_TURN_REPEATS = 90
_DOCUMENT_ID = "conversation:sess-oversized-append"
_INITIAL_TURNS = 16
_TOTAL_TURNS = 40


@pytest.fixture(scope="session")
def db_url():
    """A database of this module's own.

    The workspace `.env` names the live dev instance and `conftest.py` re-reads it
    with override=True, so an exported URL isolates nothing — shadowing the
    fixture is what does.
    """
    return "pg0://hs-review:5629"


@pytest.fixture(autouse=True)
def _fast_retain_env(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_ENABLE_AUTO_CONSOLIDATION", "false")
    monkeypatch.setenv("HINDSIGHT_API_ENABLE_OBSERVATIONS", "false")
    clear_config_cache()
    yield
    clear_config_cache()


def _turn(idx: int) -> str:
    """One JSONL transcript line, the shape the coding-agent write-back sends."""
    marker = f"MARKER{idx:02d}"
    return json.dumps(
        {
            "role": "user" if idx % 2 == 0 else "assistant",
            "content": f"Turn {idx:02d} {marker}. " + f"{marker} discussion of the change. " * _TURN_REPEATS,
        }
    )


def _body(turn_indices) -> str:
    """JSONL — one turn per line, joined the way the append path joins its items."""
    return "\n".join(_turn(i) for i in turn_indices)


def _markers(text: str) -> list[str]:
    return [f"MARKER{i:02d}" for i in range(_TOTAL_TURNS) if f"MARKER{i:02d}" in text]


async def _stored_body_markers(memory, request_context, bank_id: str) -> list[str]:
    doc = await memory.get_document(_DOCUMENT_ID, bank_id, request_context=request_context)
    return _markers(doc["original_text"])


async def _fact_markers(memory, request_context, bank_id: str) -> list[str]:
    units = await memory.list_memory_units(
        bank_id, document_id=_DOCUMENT_ID, limit=4000, request_context=request_context
    )
    return _markers("\n".join(str(u) for u in units["items"]))


async def _append(memory, request_context, bank_id: str, turn_indices) -> None:
    await memory.retain_batch_async(
        bank_id=bank_id,
        contents=[
            {
                "content": _body(turn_indices),
                "context": "coding agent session",
                "document_id": _DOCUMENT_ID,
                "update_mode": "append",
            }
        ],
        request_context=request_context,
    )


@pytest.mark.asyncio
async def test_an_append_whose_every_chunk_changes_still_extracts_its_new_turns(memory, request_context, monkeypatch):
    """A short document's append leaves NO chunk unchanged — and must still be extracted.

    That is the one shape which reaches delta's oversized-replacement safety valve, whose test is
    "does the complete body strictly append the stored one". An append satisfies that by
    construction now that its reported body is the complete one, so feeding it in would take the
    metadata-only path: historical chunks preserved, and the new turns never extracted at all.
    """
    bank_id = f"test_append_all_changed_{datetime.now(timezone.utc).timestamp()}"
    monkeypatch.setenv("HINDSIGHT_API_RETAIN_BATCH_TOKENS", "300")
    clear_config_cache()
    try:
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": _turn(0), "context": "coding agent session", "document_id": _DOCUMENT_ID}],
            request_context=request_context,
        )
        await _append(memory, request_context, bank_id, [1])
        assert await _fact_markers(memory, request_context, bank_id) == ["MARKER00", "MARKER01"], (
            "the appended turn was never extracted"
        )
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_a_split_append_stores_the_same_body_as_an_unsplit_one(memory, request_context, monkeypatch):
    """Splitting is a transport detail: it must not change what the document ends up holding.

    Pins the invariant the fix rests on — the body a slice reports is the body the retain
    produces — rather than a marker list a future chunker change could legitimately move.
    """
    bodies = {}
    for label, batch_tokens in (("unsplit", 1_000_000), ("split", 2_000)):
        bank_id = f"test_split_parity_{label}_{datetime.now(timezone.utc).timestamp()}"
        monkeypatch.setenv("HINDSIGHT_API_RETAIN_BATCH_TOKENS", str(batch_tokens))
        clear_config_cache()
        try:
            await memory.retain_batch_async(
                bank_id=bank_id,
                contents=[
                    {
                        "content": _body(range(_INITIAL_TURNS)),
                        "context": "coding agent session",
                        "document_id": _DOCUMENT_ID,
                    }
                ],
                request_context=request_context,
            )
            await _append(memory, request_context, bank_id, range(_INITIAL_TURNS, 28))
            await _append(memory, request_context, bank_id, range(28, _TOTAL_TURNS))
            doc = await memory.get_document(_DOCUMENT_ID, bank_id, request_context=request_context)
            bodies[label] = doc["original_text"]
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)
    assert bodies["split"] == bodies["unsplit"]


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_tokens", [4_000, 2_000, 1_000])
async def test_split_append_keeps_the_whole_document(memory, request_context, monkeypatch, batch_tokens):
    """Appending only ever adds content, so nothing already stored may be lost."""
    bank_id = f"test_split_append_{batch_tokens}_{datetime.now(timezone.utc).timestamp()}"
    monkeypatch.setenv("HINDSIGHT_API_RETAIN_BATCH_TOKENS", str(batch_tokens))
    clear_config_cache()
    expected = [f"MARKER{i:02d}" for i in range(_TOTAL_TURNS)]
    try:
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {
                    "content": _body(range(_INITIAL_TURNS)),
                    "context": "coding agent session",
                    "document_id": _DOCUMENT_ID,
                }
            ],
            request_context=request_context,
        )
        assert await _fact_markers(memory, request_context, bank_id) == expected[:_INITIAL_TURNS], (
            "the initial retain did not extract every turn, so the test would assert nothing"
        )

        await _append(memory, request_context, bank_id, range(_INITIAL_TURNS, 28))
        await _append(memory, request_context, bank_id, range(28, _TOTAL_TURNS))

        assert await _stored_body_markers(memory, request_context, bank_id) == expected, (
            "the stored document body lost turns that only ever had content appended to them"
        )
        assert await _fact_markers(memory, request_context, bank_id) == expected, (
            "facts extracted from earlier turns were tombstoned by a later append"
        )
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
