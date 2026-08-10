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

from datetime import datetime, timezone

import pytest

from hindsight_api.config import clear_config_cache
from hindsight_api.engine.memory_engine import count_tokens
from hindsight_api.engine.retain import fact_extraction, orchestrator

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
async def test_oversized_replacement_screens_document_body_once(memory, request_context, monkeypatch):
    """Companion to #3282: the split fallback must not re-run Memory Defense over
    the whole document for every sub-batch.

    Each sub-batch carries ``document_body_override`` (the COMPLETE body, so
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
