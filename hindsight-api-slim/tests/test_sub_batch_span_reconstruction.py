"""A run of native chunks is reconstructed from its original span, not a guessed rejoin.

``_pack_native_chunks`` groups consecutive chunks into runs worth ``tokens_per_batch``, and the
text for a run then has to be rebuilt so the retain can be issued once for the whole run.

``_rejoin_native_chunks`` rebuilds it by guessing which separator sat between the chunks — a merged
JSON array, ``"\\n\\n".join``, ``"\\n".join`` — and accepts a candidate only if it re-chunks back to
exactly the chunks it was given. When none of them reproduces the split it returns ``None`` and the
caller falls back to one sub-batch per chunk. That fallback is correct but expensive: each
sub-batch is a separate retain carrying a largely fixed cost, so a run of many chunks becomes many
retains instead of one.

The documents that trigger it are ordinary — any body whose chunk boundaries do not all sit on the
separator the rejoin guesses. A link list separated by single newlines, long enough to spill past
the chunk limit and followed by a short paragraph, is enough: the short trailing chunks merge once
``"\\n\\n".join`` has rewritten the separators, so the rejoin cannot reproduce the original split.

``_span_of_native_chunks`` avoids the guess: the chunks came from the source in order, so the text
that produced them is the slice from the first chunk's start to the last chunk's end, carrying
whatever separators the document actually used.
"""

from hindsight_api.engine.memory_engine import (
    _iter_raw_sub_batches,
    _pack_native_chunks,
    _rejoin_native_chunks,
    _span_of_native_chunks,
)
from hindsight_api.engine.retain.fact_extraction import chunk_text

CHUNK_SIZE = 1500
TOKENS_PER_BATCH = 10_000
# Small enough that the repeated fixture exceeds it, which is what reaches the packing branch.
PACKED_TOKENS = 800


def _link(i: int) -> str:
    return f"- [Service {i} Documentation](https://docs.example.com/compute/docs/instances/guide-{i})"


def _document_with_mixed_separators() -> str:
    """A body whose chunk boundaries do not all fall on the same separator.

    Eighteen link lines joined by single newlines fill the first chunk and spill; the nineteenth
    arrives after a blank line; a short turn follows. The chunker splits that into three, the last
    two short enough to merge once rejoined.
    """
    return "\n".join(_link(i) for i in range(18)) + "\n\n" + _link(99) + "\n" + "[Turn 322] User: " + "word322 " * 4


def test_the_rejoin_cannot_reproduce_this_split():
    """The precondition: this body is one the separator guessing genuinely fails on.

    Asserted so the test below is known to exercise the fallback path rather than passing
    vacuously on a body the rejoin would have handled anyway.
    """
    body = _document_with_mixed_separators()
    chunks = chunk_text(body, CHUNK_SIZE, structured_chunk_size=None)
    assert len(chunks) > 1, "fixture must span several chunks to be meaningful"

    rejoined = chunk_text("\n\n".join(chunks), CHUNK_SIZE, structured_chunk_size=None)
    assert rejoined != chunks, (
        "fixture no longer triggers the fallback: the guessed join now reproduces the split, "
        "so this file would pass without testing anything"
    )
    assert _rejoin_native_chunks(chunks, CHUNK_SIZE, None) is None


def test_a_run_survives_as_one_sub_batch_via_its_original_span():
    """A packed run stays ONE sub-batch, where the rejoin would have fragmented it."""
    body = _document_with_mixed_separators()
    chunks = chunk_text(body, CHUNK_SIZE, structured_chunk_size=None)
    runs = list(_pack_native_chunks(chunks, TOKENS_PER_BATCH))

    # The body is far under the token budget, so packing must offer it as a single run.
    assert len(runs) == 1, f"expected one run under a {TOKENS_PER_BATCH}-token budget, got {len(runs)}"

    span, cursor = _span_of_native_chunks(body, runs[0], 0)
    assert span is not None, "every chunk came from the body, so the span must be locatable"
    assert cursor > 0

    # The property that makes the span safe to use: it re-chunks to the run it came from, so the
    # sub-batch's chunk_index accounting is the same as the splitter's.
    assert chunk_text(span, CHUNK_SIZE, structured_chunk_size=None) == runs[0]


def test_the_span_is_the_source_text_verbatim():
    """The reconstruction carries the document's own separators, which is the whole point."""
    body = _document_with_mixed_separators()
    chunks = chunk_text(body, CHUNK_SIZE, structured_chunk_size=None)

    span, _ = _span_of_native_chunks(body, chunks, 0)
    assert span is not None
    assert span in body, "the span must be a slice of the source, not a reconstruction of it"


def test_the_cursor_only_moves_forward_across_runs():
    """One cursor serves every run, so the document is scanned once rather than per run."""
    body = _document_with_mixed_separators() * 3
    chunks = chunk_text(body, CHUNK_SIZE, structured_chunk_size=None)
    runs = list(_pack_native_chunks(chunks, 400))
    assert len(runs) > 1, "fixture must produce several runs to exercise the cursor"

    cursor = 0
    seen = [cursor]
    for run in runs:
        span, cursor = _span_of_native_chunks(body, run, cursor)
        assert span is not None
        seen.append(cursor)
    assert seen == sorted(seen), f"cursor moved backwards: {seen}"


def test_an_unlocatable_chunk_falls_back_rather_than_guessing_wrong():
    """A chunk that is not in the source yields None, so the caller uses the old path."""
    body = _document_with_mixed_separators()
    span, cursor = _span_of_native_chunks(body, ["text that is not in the document"], 0)
    assert span is None
    assert cursor == 0, "a failed lookup must not advance the cursor"


def test_the_splitter_packs_runs_instead_of_fragmenting_per_chunk():
    """The end-to-end property: a run the rejoin cannot rebuild is still retained ONCE.

    The tests above pin the helper; this one pins what the helper is for, through
    `_iter_raw_sub_batches` — the path retain actually takes.

    The body has to EXCEED `tokens_per_batch` to reach the packing branch at all (a body under the
    budget is emitted whole and the rejoin never runs), so the fixture is repeated and the budget
    lowered until it does. Without the span reconstruction every one of these runs falls back to
    one sub-batch per chunk.
    """
    body = "\n\n".join(_document_with_mixed_separators() for _ in range(4))
    chunks = chunk_text(body, CHUNK_SIZE, structured_chunk_size=None)
    runs = list(_pack_native_chunks(chunks, PACKED_TOKENS))

    # Preconditions, so this cannot pass vacuously: several runs, each holding more than one chunk,
    # and each one the guessed rejoin genuinely fails on.
    assert len(runs) > 1
    assert all(len(r) > 1 for r in runs), "every run must be multi-chunk for the packing to matter"
    assert all(_rejoin_native_chunks(r, CHUNK_SIZE, None) is None for r in runs), (
        "fixture must exercise the fallback the span reconstruction replaces"
    )

    subs = list(
        _iter_raw_sub_batches(
            [{"content": body}],
            PACKED_TOKENS,
            chunk_size=CHUNK_SIZE,
            structured_chunk_size=None,
        )
    )

    assert len(subs) == len(runs), (
        f"expected one sub-batch per packed run ({len(runs)}), got {len(subs)} — the span "
        f"reconstruction fell back to one sub-batch per chunk ({len(chunks)} chunks)"
    )
    # The count the splitter reports is what the caller advances chunk_index by, so it has to match
    # the chunks each sub-batch's text really produces.
    assert [s.chunk_count for s in subs] == [len(r) for r in runs]
    assert sum(s.chunk_count for s in subs) == len(chunks)
