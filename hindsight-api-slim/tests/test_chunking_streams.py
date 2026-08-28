"""Chunking is streamed, and streaming did not move a single boundary (issue #3756).

Retain used to materialise every chunk of a document before extracting from any of
them, which made its memory scale with the document rather than with a bounded working
set. ``iter_chunks`` yields the chunks instead, and the plain-text splitter underneath
it is a lazy re-implementation of the ``RecursiveCharacterTextSplitter`` configuration
retain had been using.

Chunk boundaries are load-bearing: they are content-hashed for delta retain, and a
chunk's index becomes its ``chunk_id``. A re-implementation that shifted them would make
every stored chunk of every existing document look changed, silently. So the tests here
are mostly differential — they pin the new splitter against the langchain one it
replaced, on inputs chosen to reach every branch of it.
"""

import json
import tracemalloc

from langchain_text_splitters import RecursiveCharacterTextSplitter

from hindsight_api.engine.retain.fact_extraction import (
    _RECURSIVE_TEXT_SEPARATORS,
    _iter_recursive_splits,
    chunk_text,
    iter_chunks,
)


def _langchain_split(text: str, max_chars: int) -> list[str]:
    """The exact splitter call the plain-text chunking path made before #3756.

    Kept as the reference implementation the streaming splitter is diffed against, which is
    the only remaining use of langchain-text-splitters in this repo.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
        separators=_RECURSIVE_TEXT_SEPARATORS,
    )
    return splitter.split_text(text)


# Inputs chosen to reach each separator in _RECURSIVE_TEXT_SEPARATORS in turn, plus the
# degenerate cases: no separator at all (falls through to per-character), a single
# oversized token, and text whose pieces individually exceed the budget.
_PLAIN_TEXT_CASES = {
    "paragraphs": "Alpha para one.\n\nBeta para two.\n\nGamma para three.\n\nDelta para four.",
    "lines": "Alpha line one\nBeta line two\nGamma line three\nDelta line four\nEpsilon five",
    "sentences": "Alpha sentence one. Beta sentence two. Gamma sentence three. Delta four.",
    "exclamations": "Alpha one! Beta two! Gamma three! Delta four! Epsilon five! Zeta six!",
    "questions": "Alpha one? Beta two? Gamma three? Delta four? Epsilon five? Zeta six?",
    "semicolons": "alpha one; beta two; gamma three; delta four; epsilon five; zeta six",
    "commas": "alpha one, beta two, gamma three, delta four, epsilon five, zeta six",
    "words": "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi",
    "no_separators": "x" * 500,
    "one_giant_word": "short. " + "y" * 400 + ". tail.",
    "mixed": "Para one.\n\nLine A\nLine B. Sentence C! Question D? Clause E; item F, word G",
    "leading_trailing_whitespace": "   \n\n  Alpha one.  Beta two.  Gamma three.   \n\n   ",
    "repeated_separators": "Alpha.  \n\n\n\nBeta.  \n\n\n\nGamma.  \n\n\n\nDelta.",
    "unicode": "Ålpha sätz ett. Béta sætning to. Gämma 文章三. Delta предложение четыре.",
    "empty_pieces": "..... ..... ..... .....",
}


def test_streaming_splitter_matches_langchain_across_separator_tiers():
    """Every separator tier and degenerate case splits exactly as langchain did."""
    for name, text in _PLAIN_TEXT_CASES.items():
        for max_chars in (10, 17, 40, 100, 1000):
            streamed = list(_iter_recursive_splits(text, max_chars, _RECURSIVE_TEXT_SEPARATORS))
            assert streamed == _langchain_split(text, max_chars), f"{name} at max_chars={max_chars}"


def test_streaming_splitter_matches_langchain_on_generated_prose():
    """A larger, more varied body — the shape a real document arrives in."""
    paragraphs = []
    for index in range(60):
        paragraphs.append(
            f"Section {index}: the quick brown fox jumped over {index} lazy dogs; "
            f"it was raining, and nobody minded! Did they? Probably not.\n"
            f"A second line for section {index} with no terminal punctuation"
        )
    text = "\n\n".join(paragraphs)

    for max_chars in (50, 137, 500, 1500):
        assert list(_iter_recursive_splits(text, max_chars, _RECURSIVE_TEXT_SEPARATORS)) == _langchain_split(
            text, max_chars
        )


def test_iter_chunks_matches_chunk_text_for_every_input_shape():
    """The streamed and materialised forms agree — including the structured paths."""
    conversation = json.dumps(
        [{"role": "user" if i % 2 == 0 else "assistant", "content": f"Turn {i} content here."} for i in range(40)]
    )
    jsonl = "\n".join(json.dumps({"event": i, "detail": f"payload number {i} with words"}) for i in range(40))
    single_object = json.dumps({"role": "user", "content": "A single object " * 40})

    cases = {
        "conversation": conversation,
        "jsonl": jsonl,
        "jsonl_with_blank_lines": jsonl.replace("\n", "\n\n", 5),
        "jsonl_crlf": jsonl.replace("\n", "\r\n"),
        "single_object": single_object,
        "not_jsonl": '{"a": 1}\nplain text line that is not json\n{"b": 2}',
        **_PLAIN_TEXT_CASES,
    }
    for name, text in cases.items():
        for max_chars in (60, 200, 1000):
            materialised = chunk_text(text, max_chars)
            assert list(iter_chunks(text, max_chars)) == materialised, name
            assert materialised, f"{name} produced no chunks"


def test_iter_chunks_honours_structured_chunk_size():
    """The structured limit reaches the streamed path the same way it reached the list one."""
    jsonl = "\n".join(json.dumps({"event": i, "detail": "x" * 300}) for i in range(10))
    for structured in (100, 400, 5000):
        assert list(iter_chunks(jsonl, 200, structured_chunk_size=structured)) == chunk_text(
            jsonl, 200, structured_chunk_size=structured
        )


def test_iter_chunks_is_lazy():
    """Pulling one chunk does not chunk the rest of the document.

    The whole point of the generator: retain's producer takes chunks one at a time, so a
    body large enough to matter must not be fully split to hand back its first chunk.
    """
    text = "Sentence number one. " * 20_000
    chunks = iter_chunks(text, 100)
    first = next(chunks)
    assert first.startswith("Sentence number one.")
    # A materialising implementation would already hold all ~4000 chunks; a lazy one has
    # produced exactly the one that was asked for. `gi_frame` is non-None only while the
    # generator is suspended mid-body rather than exhausted.
    assert chunks.gi_frame is not None
    assert len(list(chunks)) > 1000


def test_streaming_allocates_a_bounded_amount_whatever_the_document_size():
    """Streaming means bounded, and only an allocation counter can prove it.

    This is the test that would have caught the defect it was written for: the first
    streaming implementation collected every under-budget piece into a list before packing
    any of them — mirroring how the eager splitter builds ``good_splits`` — so for prose,
    which has no over-budget piece to interrupt the run, it buffered the entire document
    and streamed nothing. Every functional test still passed, and peak RSS could not see it
    (the allocator reuses arenas, so the cost lands on whichever caller runs first).

    ``tracemalloc`` sees it: the buffer showed as ~1.8x the document, and a bound shows as
    flat. Asserted against a 4x size ratio so only genuine proportionality can fail it.
    """
    small = "Sentence number one. " * 20_000
    large = "Sentence number one. " * 80_000

    def _consume(text: str) -> float:
        tracemalloc.start()
        try:
            count = sum(1 for _ in iter_chunks(text, 1500))
            assert count > 100
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        return peak / 1024 / 1024

    small_cost = _consume(small)
    large_cost = _consume(large)

    assert large_cost <= max(small_cost, 1.0) * 1.5, f"small={small_cost:.2f} MB, large={large_cost:.2f} MB"


def test_chunk_text_idempotent_on_streamed_chunks():
    """Re-chunking any produced chunk yields it unchanged (issue #2301's invariant)."""
    text = "\n\n".join(f"Para {i}. Sentence two here; clause three, word four!" for i in range(40))
    for chunk in iter_chunks(text, 120):
        assert chunk_text(chunk, 120) == [chunk]
