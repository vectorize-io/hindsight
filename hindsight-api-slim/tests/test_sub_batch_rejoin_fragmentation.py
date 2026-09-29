"""A run of native chunks that cannot be rejoined degrades the whole retain to one
sub-batch per chunk.

``_pack_native_chunks`` groups consecutive chunks into runs worth ``tokens_per_batch``,
and ``_rejoin_native_chunks`` then rebuilds the text for a run — but only accepts a join
that re-chunks back to exactly the chunks it was given. It tries three candidates: a
merged JSON array, ``"\\n\\n".join`` and ``"\\n".join``. When none reproduces the split,
it returns ``None`` and the caller falls back to one sub-batch per chunk.

That fallback is correct but expensive: each sub-batch is a separate retain with its own
fixed cost, so a run of ~33 chunks becomes ~33 retains instead of one. Measured against a
deployment, the fixed cost is ~1.1s per retain regardless of size, so the fallback is
worth roughly 30x on wall time for the affected part of a document.

The documents that trigger it are ordinary: any body whose chunk boundaries do not all
sit on the separator the rejoin guesses. A markdown link list separated by single
newlines, long enough to spill past the chunk limit and followed by a short paragraph,
is enough — the two short trailing chunks fit inside one chunk once ``"\\n\\n".join`` has
rewritten the separators, so the rejoin cannot reproduce the original three.
"""

import pytest

from hindsight_api.engine.memory_engine import _pack_native_chunks, _rejoin_native_chunks
from hindsight_api.engine.retain.fact_extraction import chunk_text

CHUNK_SIZE = 1500
TOKENS_PER_BATCH = 10_000


def _link(i: int) -> str:
    return f"- [Service {i} Documentation](https://docs.example.com/compute/docs/instances/guide-{i})"


def _document_with_mixed_separators() -> str:
    """A body whose chunk boundaries do not all fall on the same separator.

    Eighteen link lines joined by single newlines fill the first chunk and spill; the
    nineteenth arrives after a blank line; a short turn follows. The chunker splits that
    into three, the last two short enough to merge once rejoined.
    """
    return "\n".join(_link(i) for i in range(18)) + "\n\n" + _link(99) + "\n" + "[Turn 322] User: " + "word322 " * 4


@pytest.mark.xfail(
    strict=True,
    reason=(
        'Known: `"\\n\\n".join` rewrites separators the body did not use, so re-chunking '
        "the joined text does not reproduce the split. This is the condition that makes "
        "`_rejoin_native_chunks` give up."
    ),
)
def test_chunking_is_idempotent_under_rejoin():
    """Re-chunking a run's rejoined text must reproduce that run."""
    body = _document_with_mixed_separators()
    chunks = chunk_text(body, CHUNK_SIZE, structured_chunk_size=None)
    assert len(chunks) > 1, "fixture must span several chunks to be meaningful"

    rejoined = chunk_text("\n\n".join(chunks), CHUNK_SIZE, structured_chunk_size=None)
    assert rejoined == chunks, (
        f"rejoining changed the split: {[len(c) for c in chunks]} became "
        f"{[len(c) for c in rejoined]}. The short trailing chunks merge because "
        f'"\\n\\n".join rewrote separators the original body did not use.'
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known: a run whose chunks do not rejoin falls back to one sub-batch per chunk. "
        "The fallback is correct, but it multiplies the number of retains — and each "
        "retain carries a fixed cost that dwarfs the work in it."
    ),
)
def test_run_of_chunks_can_be_rejoined_into_one_sub_batch():
    """A packed run should survive as ONE sub-batch, not fragment into one per chunk."""
    body = _document_with_mixed_separators()
    chunks = chunk_text(body, CHUNK_SIZE, structured_chunk_size=None)
    runs = list(_pack_native_chunks(chunks, TOKENS_PER_BATCH))

    # The body is far under the token budget, so packing must offer it as a single run.
    assert len(runs) == 1, f"expected one run under a {TOKENS_PER_BATCH}-token budget, got {len(runs)}"

    rejoined = _rejoin_native_chunks(runs[0], CHUNK_SIZE, None)
    assert rejoined is not None, (
        f"no faithful join for a run of {len(runs[0])} chunks, so retain will issue "
        f"{len(runs[0])} sub-batches instead of 1"
    )
