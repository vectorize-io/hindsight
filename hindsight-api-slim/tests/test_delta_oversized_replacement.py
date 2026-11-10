"""
Regression tests for https://github.com/vectorize-io/hindsight/issues/3282

When a replacement body for an existing ``document_id`` exceeds
``retain_batch_tokens``, ``retain_batch_async`` slices it into sub-batches
*before* the orchestrator gets a chance to classify the whole replacement
against the stored chunks. Delta retain only runs on the first sub-batch (and
only sees that slice), so a one-section edit plus an appended tail re-extracts
the entire unchanged history instead of just the changed/new native chunks.

The control test pins the behaviour with a transport budget large enough to
keep the replacement in one piece (delta works there today); the repro test
lowers only ``retain_batch_tokens`` and asserts the same document edit still
skips the unchanged chunks.
"""

import asyncio
import re
from datetime import datetime, timezone

import pytest

from hindsight_api.config import clear_config_cache
from hindsight_api.engine.memory_engine import count_tokens
from hindsight_api.engine.retain import fact_extraction, orchestrator
from tests.retain_result_capture import RetainResultCapture

# Sections are separated by blank lines and each is just under
# retain_chunk_size (3000 chars), so the chunker emits exactly one native chunk
# per section. That keeps the chunk boundaries stable across versions: the edit
# below is length-preserving, so only the edited section's chunk changes.
_SECTION_REPEATS = 117  # ~2.5 KB per section
_BASE_SECTIONS = 10
_APPENDED_SECTIONS = 3

# Transport budget for the replacement retain. The splitter slices an oversized
# item at ``3 * retain_batch_tokens`` chars, so this is deliberately below the
# native chunk size: the slices cut across native chunk boundaries.
_OVERSIZED_BATCH_TOKENS = 300


def _ts() -> float:
    return datetime.now(timezone.utc).timestamp()


def _section(idx: int, *, edited: bool = False) -> str:
    marker = f"MARKER{idx:02d}"
    payload = "bbbbb" if edited else "aaaaa"
    return f"Section {idx:02d} {marker}. Payload {payload}. " + f"{marker} filler word here. " * _SECTION_REPEATS


def _body(*, edited_idx: int | None = None, appended: int = 0) -> str:
    sections = [_section(i, edited=(i == edited_idx)) for i in range(_BASE_SECTIONS)]
    sections += [_section(100 + j) for j in range(appended)]
    return "\n\n".join(sections)


def _unchanged_markers(edited_idx: int) -> list[str]:
    return [f"MARKER{i:02d}" for i in range(_BASE_SECTIONS) if i != edited_idx]


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

    def markers_seen(self, markers: list[str]) -> list[str]:
        blob = "\n".join(self.texts)
        return [m for m in markers if m in blob]

    @property
    def extracted_tokens(self) -> int:
        return sum(count_tokens(t) for t in self.texts)


@pytest.fixture(autouse=True)
def _fast_retain_env(monkeypatch):
    # Keep the tests focused on the retain path.
    monkeypatch.setenv("HINDSIGHT_API_ENABLE_AUTO_CONSOLIDATION", "false")
    monkeypatch.setenv("HINDSIGHT_API_ENABLE_OBSERVATIONS", "false")
    clear_config_cache()
    yield
    clear_config_cache()


async def _retain_v1_then_v2(
    memory,
    request_context,
    bank_id: str,
    document_id: str,
    *,
    replacement_batch_tokens: int | None,
    monkeypatch,
    spy: _ExtractionSpy,
    edited_idx: int,
) -> str:
    """Retain the base body, then the edited+appended replacement.

    ``replacement_batch_tokens`` is applied only to the second retain, matching
    the issue's reproduction steps. Returns the replacement body.
    """
    v1 = _body()
    await memory.retain_async(
        bank_id=bank_id,
        content=v1,
        context="notes",
        document_id=document_id,
        request_context=request_context,
    )

    v2 = _body(edited_idx=edited_idx, appended=_APPENDED_SECTIONS)

    if replacement_batch_tokens is not None:
        monkeypatch.setenv("HINDSIGHT_API_RETAIN_BATCH_TOKENS", str(replacement_batch_tokens))
        clear_config_cache()

    # Only spy on the replacement — the first retain legitimately extracts everything.
    spy.install(monkeypatch)

    await memory.retain_async(
        bank_id=bank_id,
        content=v2,
        context="notes",
        document_id=document_id,
        request_context=request_context,
    )
    return v2


@pytest.mark.asyncio
async def test_replacement_within_batch_budget_skips_unchanged_chunks(memory, request_context, monkeypatch):
    """Control: with the whole replacement inside the transport budget, delta
    retain re-extracts only the edited section and the appended tail."""
    bank_id = f"test_3282_control_{_ts()}"
    document_id = "doc-3282-control"
    edited_idx = 1
    spy = _ExtractionSpy()

    try:
        await _retain_v1_then_v2(
            memory,
            request_context,
            bank_id,
            document_id,
            replacement_batch_tokens=100_000,  # whole replacement fits — no splitting
            monkeypatch=monkeypatch,
            spy=spy,
            edited_idx=edited_idx,
        )

        re_extracted = spy.markers_seen(_unchanged_markers(edited_idx))
        assert re_extracted == [], f"unchanged sections were re-extracted: {re_extracted}"
        assert spy.markers_seen([f"MARKER{edited_idx:02d}", "MARKER100"]), (
            "the edited section and the appended tail should have been extracted"
        )
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_oversized_replacement_still_skips_unchanged_chunks(memory, request_context, monkeypatch):
    """Repro for #3282: the same edit must not re-extract unchanged history just
    because the complete replacement exceeds ``retain_batch_tokens``.

    The budget is set below both the full replacement and the aggregate token
    count of the changed/new chunks, so the transport split cannot be satisfied
    by the delta chunk list either — delta classification must still run against
    the complete body first.
    """
    bank_id = f"test_3282_oversized_{_ts()}"
    document_id = "doc-3282-oversized"
    edited_idx = 1
    spy = _ExtractionSpy()
    batch_tokens = _OVERSIZED_BATCH_TOKENS

    try:
        v2 = await _retain_v1_then_v2(
            memory,
            request_context,
            bank_id,
            document_id,
            replacement_batch_tokens=batch_tokens,
            monkeypatch=monkeypatch,
            spy=spy,
            edited_idx=edited_idx,
        )

        # Sanity-check the premise of the repro: the replacement really is over
        # the transport budget, and so is the changed/new chunk aggregate.
        assert count_tokens(v2) > batch_tokens
        changed_chunk_tokens = count_tokens(_section(edited_idx, edited=True)) + sum(
            count_tokens(_section(100 + j)) for j in range(_APPENDED_SECTIONS)
        )
        assert changed_chunk_tokens > batch_tokens

        re_extracted = spy.markers_seen(_unchanged_markers(edited_idx))
        assert re_extracted == [], (
            f"unchanged sections {re_extracted} were re-extracted: the oversized "
            f"replacement bypassed delta retain (issue #3282). Extraction saw "
            f"{spy.extracted_tokens:,} tokens; only the changed/new chunks "
            f"(~{changed_chunk_tokens:,} tokens) should have reached it."
        )
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_oversized_replacement_edit_in_slice_one_keeps_the_rest(memory, request_context, monkeypatch):
    """The edit lands in slice 1, which is what makes the delta diff destructive if it is taken
    against the slice instead of the whole body.

    With ``edited_idx=1`` (the test above) slice 1 has no changed chunk, so the retain routes to
    ``_delta_metadata_only`` before anything is tombstoned and the diff's scope never matters. Put a
    changed chunk in slice 1 and the diff runs for real: taken against the slice alone, every chunk
    of the document that is merely absent from it classifies as REMOVED — measured on a 20-chunk
    document, ``unchanged=1 changed=0 new=0 removed=19`` — and those memories are deleted.

    Asserted on the surviving unit ids rather than on extraction counts: the previous test already
    covers "was it re-extracted", and a document can be fully re-extracted while still ending up
    with the right memories. What must hold here is that the unchanged sections' memories are the
    SAME ones, not replacements.
    """
    bank_id = f"test_3282_slice1_{_ts()}"
    document_id = "doc-3282-slice1"
    edited_idx = 0
    spy = _ExtractionSpy()

    try:
        v2 = await _retain_v1_then_v2(
            memory,
            request_context,
            bank_id,
            document_id,
            replacement_batch_tokens=_OVERSIZED_BATCH_TOKENS,
            monkeypatch=monkeypatch,
            spy=spy,
            edited_idx=edited_idx,
        )
        assert count_tokens(v2) > _OVERSIZED_BATCH_TOKENS

        after = await memory.list_memory_units(
            bank_id, document_id=document_id, limit=10000, request_context=request_context
        )
        texts = " ".join(row["text"] or "" for row in after["items"])
        missing = [m for m in _unchanged_markers(edited_idx) if m not in texts]
        assert not missing, (
            f"sections {missing} lost their memories: the delta diff for the slice holding the edit "
            f"classified every chunk absent from that slice as removed and tombstoned it"
        )
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_oversized_replacement_screens_document_body_once(memory, request_context, monkeypatch):
    """Companion to #3282: the split fallback must not re-run Memory Defense over
    the whole document for every sub-batch.

    Each sub-batch carries ``full_document_body`` (the COMPLETE body, so
    ``documents.original_text`` isn't clobbered with a slice), and the retain
    path redaction-scans that override before persisting it. With N sub-batches
    the full body is scanned N times even though only one of them wins the
    document row.
    """
    bank_id = f"test_3282_defense_{_ts()}"
    document_id = "doc-3282-defense"
    edited_idx = 1
    spy = _ExtractionSpy()

    full_body_scans: list[int] = []
    original_redaction = orchestrator.apply_redaction

    def _counting_redaction(text: str, *args, **kwargs):
        full_body_scans.append(len(text))
        return original_redaction(text, *args, **kwargs)

    try:
        await memory.update_bank_config(
            bank_id,
            {"memory_defense": {"enabled": True, "rules": [{"on": "sensitive_data", "action": "redact"}]}},
            request_context=request_context,
        )
        monkeypatch.setattr(orchestrator, "apply_redaction", _counting_redaction)

        v2 = await _retain_v1_then_v2(
            memory,
            request_context,
            bank_id,
            document_id,
            replacement_batch_tokens=_OVERSIZED_BATCH_TOKENS,
            monkeypatch=monkeypatch,
            spy=spy,
            edited_idx=edited_idx,
        )

        replacement_scans = [n for n in full_body_scans if n == len(v2)]
        assert len(replacement_scans) <= 1, (
            f"the complete {len(v2):,}-char body was Memory Defense scanned "
            f"{len(replacement_scans)} times — once per fallback sub-batch (issue #3282)"
        )
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# processed_content_tokens for oversized re-retains
# ---------------------------------------------------------------------------
#
# The tests above prove an oversized re-retain skips unchanged chunks. These prove it SAYS so:
# RetainResult.processed_content_tokens is what a metering extension charges by, and None means
# "no dedup signal, charge the full submission". The later sub-batches of an oversized item used
# to return None unconditionally, so every re-retain of a large, mostly unchanged document was
# reported as a full one however little extraction it actually did.


@pytest.mark.asyncio
async def test_oversized_identical_reretain_reports_nothing_processed(memory, request_context, monkeypatch):
    """Re-submitting an oversized document unchanged extracts nothing, and must report 0 — not
    None, which would charge the whole document again."""
    bank_id = f"test_oversized_pct_identical_{_ts()}"
    document_id = "doc-oversized-pct-identical"
    capture = RetainResultCapture()
    memory._operation_validator = capture
    body = _body()
    monkeypatch.setenv("HINDSIGHT_API_RETAIN_BATCH_TOKENS", str(_OVERSIZED_BATCH_TOKENS))
    clear_config_cache()
    assert count_tokens(body) > _OVERSIZED_BATCH_TOKENS, "premise: the body must be split into sub-batches"

    try:
        for _ in range(2):
            await memory.retain_async(
                bank_id=bank_id,
                content=body,
                context="notes",
                document_id=document_id,
                request_context=request_context,
            )
        assert len(capture.results) == 2
        assert capture.results[0].processed_content_tokens is None, "a first retain is a full retain"
        assert capture.results[1].processed_content_tokens == 0, (
            f"an unchanged oversized re-retain reported processed_content_tokens="
            f"{capture.results[1].processed_content_tokens!r}; None charges the full "
            f"{count_tokens(body):,}-token document again"
        )
    finally:
        memory._operation_validator = None
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_oversized_edited_reretain_reports_only_what_was_extracted(memory, request_context, monkeypatch):
    """An oversized re-retain with one edited section and an appended tail reports the tokens that
    went to extraction: far less than the submission, and no less than the text extraction saw."""
    bank_id = f"test_oversized_pct_edited_{_ts()}"
    document_id = "doc-oversized-pct-edited"
    edited_idx = 1
    spy = _ExtractionSpy()
    capture = RetainResultCapture()
    memory._operation_validator = capture

    try:
        v2 = await _retain_v1_then_v2(
            memory,
            request_context,
            bank_id,
            document_id,
            replacement_batch_tokens=_OVERSIZED_BATCH_TOKENS,
            monkeypatch=monkeypatch,
            spy=spy,
            edited_idx=edited_idx,
        )
        assert len(capture.results) == 2
        processed = capture.results[1].processed_content_tokens
        submitted = count_tokens(v2) + count_tokens("notes")

        assert processed is not None, (
            f"an oversized re-retain that extracted {spy.extracted_tokens:,} of {submitted:,} tokens "
            f"reported no dedup signal, so it is charged as a full retain"
        )
        assert processed >= spy.extracted_tokens, "it must not under-report what was extracted"
        assert processed < submitted / 2, (
            f"reported {processed:,} of {submitted:,} submitted tokens for a one-section edit + tail"
        )
    finally:
        memory._operation_validator = None
        await memory.delete_bank(bank_id, request_context=request_context)


# Issue #5485: a sliced retain that crashed part-way is retried with the SAME body. Slice 1 used to
# take the delta path, which diffs the complete body and extracts every chunk the crash left
# unwritten in one step that commits only at the end — hundreds of calls queued at once, and all of
# them lost if the retry is interrupted too. The retry must resume slice by slice instead.


class _CrashAfter:
    """Extraction spy that fails the retain after ``n`` calls and records each call's item count."""

    _original = staticmethod(fact_extraction.extract_facts_from_contents)

    def __init__(self, n: int | None) -> None:
        self.n = n
        self.items_per_call: list[int] = []
        self.texts: list[str] = []

    def install(self, monkeypatch) -> None:
        original = self._original

        async def _spy(contents, *args, **kwargs):
            if self.n is not None and len(self.items_per_call) >= self.n:
                raise RuntimeError("simulated worker crash")
            self.items_per_call.append(len(contents))
            self.texts.extend(c.content for c in contents)
            return await original(contents, *args, **kwargs)

        monkeypatch.setattr(fact_extraction, "extract_facts_from_contents", _spy)


@pytest.mark.asyncio
async def test_recovering_partially_written_oversized_document_resumes_slice_by_slice(
    memory, request_context, monkeypatch
):
    bank_id = f"test_5485_recovery_{_ts()}"
    document_id = "doc-5485"
    body = _body()
    monkeypatch.setenv("HINDSIGHT_API_RETAIN_BATCH_TOKENS", str(_OVERSIZED_BATCH_TOKENS))
    clear_config_cache()

    try:
        first = _CrashAfter(3)
        first.install(monkeypatch)
        with pytest.raises(RuntimeError, match="simulated worker crash"):
            await memory.retain_async(
                bank_id=bank_id, content=body, context="notes", document_id=document_id, request_context=request_context
            )
        committed = [m for m in (f"MARKER{i:02d}" for i in range(_BASE_SECTIONS)) if m in "\n".join(first.texts)]
        assert 0 < len(committed) < _BASE_SECTIONS, "premise: the first run must commit only part of the document"

        retry = _CrashAfter(None)
        retry.install(monkeypatch)
        await memory.retain_async(
            bank_id=bank_id, content=body, context="notes", document_id=document_id, request_context=request_context
        )

        assert max(retry.items_per_call) == 1, (
            f"the retry extracted {max(retry.items_per_call)} chunks in one step (calls: {retry.items_per_call}); "
            f"it must resume one slice at a time, committing as it goes"
        )
        seen = "\n".join(retry.texts)
        missing = [f"MARKER{i:02d}" for i in range(_BASE_SECTIONS) if f"MARKER{i:02d}" not in committed]
        assert all(m in seen for m in missing), "every chunk the crash left unwritten must be extracted"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


# Issue #6394: same as #5485, but the follow-up is a small APPEND, not a retry of the same body. The
# append's body hash differs from the stored one, so the #5485 gate does not apply, and the delta
# extracted the append plus every chunk the crash left unwritten in one step that commits only at
# the end. A timeout there lost all of it, every time.
@pytest.mark.asyncio
async def test_append_to_partially_written_document_saves_batch_by_batch(memory, request_context, monkeypatch):
    bank_id = f"test_6394_append_{_ts()}"
    document_id = "doc-6394"
    body = _body()
    monkeypatch.setenv("HINDSIGHT_API_RETAIN_BATCH_TOKENS", str(_OVERSIZED_BATCH_TOKENS))
    clear_config_cache()

    try:
        first = _CrashAfter(3)
        first.install(monkeypatch)
        with pytest.raises(RuntimeError, match="simulated worker crash"):
            await memory.retain_async(
                bank_id=bank_id, content=body, context="notes", document_id=document_id, request_context=request_context
            )
        committed = [m for m in (f"MARKER{i:02d}" for i in range(_BASE_SECTIONS)) if m in "\n".join(first.texts)]
        assert 0 < len(committed) < _BASE_SECTIONS, "premise: the first run must commit only part of the document"
        monkeypatch.delenv("HINDSIGHT_API_RETAIN_BATCH_TOKENS")
        # One chunk per commit, so "saved as it goes" is visible on a 10-chunk document.
        monkeypatch.setattr(memory._config_resolver._global_config, "retain_chunk_batch_size", 1)
        clear_config_cache()

        # The append is interrupted too, after two extraction calls: what it extracted must stay.
        append = _CrashAfter(2)
        append.install(monkeypatch)
        with pytest.raises(RuntimeError, match="simulated worker crash"):
            await memory.retain_batch_async(
                bank_id=bank_id,
                contents=[{"content": "\n\n" + _section(100), "context": "notes", "update_mode": "append"}],
                document_id=document_id,
                request_context=request_context,
            )
        assert max(append.items_per_call) == 1, (
            f"the append extracted {max(append.items_per_call)} chunks in one step "
            f"(calls: {append.items_per_call}); it must go one slice at a time, committing as it goes"
        )

        # Retrying the append resumes: it re-extracts nothing the interrupted attempt already saved.
        later = _CrashAfter(None)
        later.install(monkeypatch)
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": "\n\n" + _section(100), "context": "notes", "update_mode": "append"}],
            document_id=document_id,
            request_context=request_context,
        )
        redone = [t[:20] for t in later.texts if any(t == s for s in append.texts)]
        assert not redone, f"the later append redid work the interrupted one had saved: {redone}"
        seen = "\n".join(first.texts + append.texts + later.texts)
        expected = [f"MARKER{i:02d}" for i in range(_BASE_SECTIONS)] + ["MARKER100"]
        assert all(m in seen for m in expected), "every chunk of the document must end up extracted"
        doc = await memory.get_document(document_id, bank_id, request_context=request_context)
        assert doc["original_text"].count("MARKER100.") == 1, "the retried append must land exactly once"
        assert max(later.items_per_call) == 1
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


async def _assert_document_consistent(memory, bank_id: str, document_id: str) -> str:
    """The stored chunks are exactly the stored body's chunk plan, and every fact sits on the chunk
    it was extracted from. Returns the stored body."""
    from hindsight_api.engine.retain import chunk_storage

    # Raw SQL on purpose: the defect this checks is a fact left on a chunk it was not extracted
    # from, and no engine read reports a fact's chunk (`list_memory_units` has no chunk_id) or a
    # chunk's content hash.
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        body = await conn.fetchval(
            "SELECT original_text FROM documents WHERE bank_id = $1 AND id = $2", bank_id, document_id
        )
        chunks = await conn.fetch(
            "SELECT chunk_id, chunk_index, chunk_text, content_hash FROM chunks "
            "WHERE bank_id = $1 AND document_id = $2 ORDER BY chunk_index",
            bank_id,
            document_id,
        )
        units = await conn.fetch(
            "SELECT text, chunk_id FROM memory_units WHERE bank_id = $1 AND document_id = $2 AND fact_type = 'world'",
            bank_id,
            document_id,
        )
    plan = fact_extraction.chunk_text(body, max_chars=3000)
    assert [c["content_hash"] for c in chunks] == [chunk_storage.compute_chunk_hash(t) for t in plan], (
        "the stored chunks must be exactly the stored body's chunk plan"
    )
    text_by_chunk = {c["chunk_id"]: c["chunk_text"] for c in chunks}
    for unit in units:
        marker = re.search(r"MARKER\d+", unit["text"])
        if marker is None:  # the mock LLM also echoes prompt sentences, which name no section
            continue
        assert marker.group(0) in text_by_chunk[unit["chunk_id"]], (
            f"fact {unit['text'][:40]!r} sits on a chunk it was not extracted from"
        )
    return body


def _sections(*, edited: frozenset[int] = frozenset()) -> str:
    return "\n\n".join(_section(i, edited=i in edited) for i in range(_BASE_SECTIONS))


# #6394, the edit case (#5485 option a): a large edit to a stored document is saved batch by batch,
# so an interrupted edit keeps what it extracted and its retry extracts only the rest. The document
# keeps its old body until the last batch commits.
@pytest.mark.asyncio
async def test_interrupted_large_edit_keeps_its_saved_batches(memory, request_context, monkeypatch):
    bank_id = f"test_6394_edit_{_ts()}"
    document_id = "doc-6394-edit"
    edited = _sections(edited=frozenset({2, 5, 8}))
    try:
        await memory.retain_async(
            bank_id=bank_id,
            content=_sections(),
            context="notes",
            document_id=document_id,
            request_context=request_context,
        )
        monkeypatch.setattr(memory._config_resolver._global_config, "retain_chunk_batch_size", 1)

        first = _CrashAfter(2)
        first.install(monkeypatch)
        with pytest.raises(RuntimeError, match="simulated worker crash"):
            await memory.retain_async(
                bank_id=bank_id,
                content=edited,
                context="notes",
                document_id=document_id,
                request_context=request_context,
            )
        assert first.items_per_call == [1, 1], f"one chunk per batch, calls: {first.items_per_call}"
        doc = await memory.get_document(document_id, bank_id, request_context=request_context)
        assert doc["original_text"] == _sections(), "the document must keep its old body until the last batch"

        retry = _CrashAfter(None)
        retry.install(monkeypatch)
        await memory.retain_async(
            bank_id=bank_id, content=edited, context="notes", document_id=document_id, request_context=request_context
        )
        assert retry.items_per_call == [1] and "MARKER08" in retry.texts[0], (
            f"the retry must extract only the chunk the interrupted edit did not save, got {retry.texts!r}"
        )
        assert await _assert_document_consistent(memory, bank_id, document_id) == edited
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


# A batched delta leaves the document's hash at its base until its last batch, so the document-hash
# check alone cannot tell a second writer that the first is part-way through. Without the chunk-level
# check the second append upserted its chunk over the first's and both versions' facts stayed on it.
@pytest.mark.asyncio
async def test_append_landing_between_batches_of_another_keeps_document_consistent(
    memory, request_context, monkeypatch
):
    bank_id = f"test_6394_race_{_ts()}"
    document_id = "doc-6394-race"
    try:
        await memory.retain_async(
            bank_id=bank_id,
            content=_sections(),
            context="notes",
            document_id=document_id,
            request_context=request_context,
        )
        monkeypatch.setattr(memory._config_resolver._global_config, "retain_chunk_batch_size", 1)
        original = fact_extraction.extract_facts_from_contents
        b_planned, a_committed, b_done = asyncio.Event(), asyncio.Event(), asyncio.Event()
        a_calls = [0]

        # Force the one order the document-hash check cannot see: B plans against the base, A then
        # commits its first batch (leaving the hash at the base), and only then does B write.
        async def _interleave(contents, *args, **kwargs):
            text = "\n".join(c.content for c in contents)
            if "MARKER200" in text and not b_planned.is_set():
                b_planned.set()
                await a_committed.wait()
            elif "MARKER10" in text:
                a_calls[0] += 1
                if a_calls[0] == 1:
                    await b_planned.wait()
                elif a_calls[0] == 2:  # A's first batch is committed
                    a_committed.set()
                    await b_done.wait()
            return await original(contents, *args, **kwargs)

        monkeypatch.setattr(fact_extraction, "extract_facts_from_contents", _interleave)
        tail = "\n\n".join(_section(100 + j) for j in range(_APPENDED_SECTIONS))

        async def _append(content: str) -> None:
            await memory.retain_batch_async(
                bank_id=bank_id,
                contents=[{"content": content, "context": "notes", "update_mode": "append"}],
                document_id=document_id,
                request_context=request_context,
            )

        async def _b() -> None:
            try:
                await _append("\n\n" + _section(200))
            finally:
                b_done.set()

        await asyncio.wait_for(asyncio.gather(_append("\n\n" + tail), _b()), timeout=300)

        body = await _assert_document_consistent(memory, bank_id, document_id)
        assert body.startswith(_sections()), "both appends must extend the stored document, not replace it"
        for marker in ["MARKER200.", "MARKER100.", "MARKER101.", "MARKER102."]:
            assert body.count(marker) == 1, f"{marker} must be in the document exactly once"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
