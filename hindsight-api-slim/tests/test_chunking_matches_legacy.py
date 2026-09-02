"""The new chunker produces byte-identical output to the one it replaced (issue #3756).

Retain's plain-text chunking was ``RecursiveCharacterTextSplitter.split_text`` called on a
whole document. #3756 replaced it with a lazy re-implementation so retain's memory would
stop scaling with the document, and dropped langchain from the runtime dependencies.

Chunk boundaries are not cosmetic. Delta retain matches stored chunks by content hash, and
a chunk's ``chunk_id`` is ``{bank}_{doc}_{index}``. A splitter that moved a boundary — by a
character, on one input shape — would make every stored chunk of every existing document
look changed on its next retain: silent re-extraction, re-embedding, and re-consolidation
of history that did not change. Nothing else in the system would report a problem.

So this file does not test the new chunker's behaviour in the abstract. It runs the OLD
implementation, copied verbatim below and still calling langchain, against the new one over
a corpus of document shapes and chunk sizes, and asserts the two agree exactly. That is the
only property that matters, and it is the reason ``langchain-text-splitters`` is still a
test dependency after #3756 removed it as a runtime one.

The corpus is deliberately awkward: the shapes retain actually receives (prose, transcripts,
JSONL logs, JSON conversations, markdown, source code) plus the ones that break splitters
(no separator at all, only separators, a single token larger than the budget, CJK, mixed
line endings, whitespace runs). Seeded generation adds volume without making failures
irreproducible.
"""

import json
import random

import pytest

from hindsight_api.engine.retain.fact_extraction import (
    _RECURSIVE_TEXT_SEPARATORS,
    chunk_text,
    iter_chunks,
)

# ---------------------------------------------------------------------------
# The implementation being replaced, verbatim from before #3756.
#
# Kept as a copy rather than imported from history so this test keeps working as the real
# code moves on. It must NOT be refactored to share anything with the live implementation:
# the moment the two share a code path, the comparison proves nothing.
# ---------------------------------------------------------------------------


def _legacy_split_oversized_unit(text: str, max_chars: int) -> list[str]:
    """Pre-#3756: the plain-text splitter, straight from langchain."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
        separators=_RECURSIVE_TEXT_SEPARATORS,
    )
    return splitter.split_text(text)


def _legacy_chunk_text(text: str, max_chars: int, structured_chunk_size: int | None = None) -> list[str]:
    """
    Split text into chunks, preserving conversation structure when possible.

    For JSON conversation arrays (user/assistant turns) and JSONL (newline-delimited
    JSON objects), splits at turn/line boundaries so no object is split across chunks.
    A single turn/line that overflows ``max_chars`` is kept whole only up to
    ``structured_chunk_size``. When unset, that limit defaults to ``max_chars``.
    For plain text, uses sentence-aware splitting.

    The result is idempotent: re-chunking any chunk this returns yields that chunk
    unchanged. The streaming retain pipeline pre-chunks each document once and then
    re-chunks every piece during extraction; if a piece re-split, its sub-chunks
    would inherit one chunk_index and collide on ``chunk_id`` (issue #2301).

    Args:
        text: Input text to chunk (plain text, JSON conversation, or JSONL)
        max_chars: Target maximum characters per chunk
        structured_chunk_size: Maximum characters for a single JSONL line or
            conversation turn to keep whole. Defaults to ``max_chars``.

    Returns:
        List of text chunks, roughly under max_chars
    """
    # If text is small enough, return as-is
    if len(text) <= max_chars:
        return [text]

    structured_limit = structured_chunk_size if structured_chunk_size is not None else max_chars

    # Try to parse as JSON conversation array
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        parsed = None

    if isinstance(parsed, list) and all(isinstance(turn, dict) for turn in parsed):
        # This looks like a conversation - chunk at turn boundaries
        return _legacy_chunk_conversation(parsed, max_chars, structured_limit)

    if isinstance(parsed, dict):
        # A single JSON object — e.g. one JSONL line handed back to the extractor
        # after the producer already pre-chunked it. It is one structured unit:
        # keep it whole up to the structured limit, else split it as text within
        # the chunk budget. Without this, a lone object (one line, so _legacy_chunk_jsonl
        # declines) would fall through to plain-text splitting and re-split a chunk
        # the producer deliberately kept whole — breaking idempotency (issue #2301).
        if len(text) <= structured_limit:
            return [text]
        return _legacy_split_oversized_unit(text, max_chars)

    # Try to parse as JSONL (newline-delimited JSON objects, e.g. session logs)
    jsonl_chunks = _legacy_chunk_jsonl(text, max_chars, structured_limit)
    if jsonl_chunks is not None:
        return jsonl_chunks

    # Fall back to sentence-aware text splitting
    return _legacy_split_oversized_unit(text, max_chars)


def _legacy_chunk_conversation(turns: list[dict], max_chars: int, structured_limit: int) -> list[str]:
    """
    Chunk a conversation array at turn boundaries, preserving complete turns.

    Args:
        turns: List of conversation turn dicts (with 'role' and 'content' keys)
        max_chars: Maximum characters per chunk
        structured_limit: Maximum characters for a single turn to keep whole

    Returns:
        List of JSON-serialized chunks, each containing complete turns
    """

    chunks = []
    current_chunk = []
    current_size = 2  # Account for "[]"

    def _flush() -> None:
        nonlocal current_chunk, current_size
        if current_chunk:
            chunks.append(json.dumps(current_chunk, ensure_ascii=False))
            current_chunk = []
            current_size = 2  # Reset to "[]"

    for turn in turns:
        # Estimate size of this turn when serialized (with comma separator)
        turn_json = json.dumps(turn, ensure_ascii=False)
        turn_unit_size = len(turn_json)
        turn_size = turn_unit_size + 1  # +1 for comma

        # A turn too large to keep whole even alone: flush, then split it as
        # text. Fragment within min(structured_limit, max_chars) so no fragment
        # exceeds the chunk budget — otherwise a downstream re-chunk would split
        # it again and collide on chunk_id (issue #2301).
        if turn_unit_size > structured_limit:
            _flush()
            chunks.extend(_legacy_split_oversized_unit(turn_json, min(structured_limit, max_chars)))
            continue

        # If adding this turn would exceed limit and we have turns, save current chunk
        if current_size + turn_size > max_chars and current_chunk:
            _flush()

        # Add turn to current chunk
        current_chunk.append(turn)
        current_size += turn_size

    # Add final chunk if non-empty
    _flush()

    return chunks if chunks else [json.dumps(turns, ensure_ascii=False)]


def _legacy_chunk_jsonl(text: str, max_chars: int, structured_limit: int) -> list[str] | None:
    """Chunk newline-delimited JSON (JSONL) at line boundaries.

    Detects JSONL — two or more non-empty lines, each a complete JSON object —
    and packs whole lines into chunks so no line is split across chunks (multiple
    short lines may share a chunk). A line that overflows ``max_chars`` is kept
    whole only up to ``structured_limit``. Returns ``None`` if the input is not
    JSONL, so the caller falls back to plain-text splitting.

    Args:
        text: Input text to inspect/chunk.
        max_chars: Maximum characters per chunk.
        structured_limit: Maximum characters for a single JSONL line to
            keep whole.

    Returns:
        List of JSONL chunks (lines joined by newline), or ``None`` if not JSONL.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return None

    for line in lines:
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(obj, dict):
            return None

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_size = 0

    def _flush() -> None:
        nonlocal current_chunk, current_size
        if current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_size = 0

    for line in lines:
        line_unit_size = len(line)
        line_size = len(line) + 1  # +1 for the joining newline

        # A line too large to keep whole even alone: flush, then split it as
        # text. Fragment within min(structured_limit, max_chars) so no fragment
        # exceeds the chunk budget — otherwise a downstream re-chunk would split
        # it again and collide on chunk_id (issue #2301).
        if line_unit_size > structured_limit:
            _flush()
            chunks.extend(_legacy_split_oversized_unit(line, min(structured_limit, max_chars)))
            continue

        # If adding this line would exceed the limit and we have lines, flush.
        # A line up to structured_limit is kept whole (a bounded overflow).
        if current_size + line_size > max_chars and current_chunk:
            _flush()

        current_chunk.append(line)
        current_size += line_size

    _flush()

    return chunks


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

_SEED = 20260824


def _prose(paragraphs: int, *, seed: int) -> str:
    """Prose with varied sentence lengths and paragraph breaks."""
    rng = random.Random(seed)
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi".split()
    out = []
    for _ in range(paragraphs):
        sentences = []
        for _ in range(rng.randint(1, 6)):
            length = rng.randint(3, 25)
            sentence = " ".join(rng.choice(words) for _ in range(length))
            sentences.append(sentence + rng.choice([". ", "! ", "? ", "; ", ", "]))
        out.append("".join(sentences).strip())
    return "\n\n".join(out)


def _transcript(turns: int, *, seed: int) -> str:
    """A chat transcript as a JSON array — the conversation path."""
    rng = random.Random(seed)
    roles = ["user", "assistant"]
    return json.dumps(
        [
            {
                "role": roles[i % 2],
                "content": " ".join(f"word{rng.randint(0, 999)}" for _ in range(rng.randint(2, 60))),
            }
            for i in range(turns)
        ],
        ensure_ascii=False,
    )


def _jsonl(lines: int, *, seed: int) -> str:
    """A newline-delimited JSON log — the JSONL path."""
    rng = random.Random(seed)
    return "\n".join(
        json.dumps(
            {
                "ts": f"2026-08-24T09:{i % 60:02d}:00Z",
                "level": rng.choice(["INFO", "WARN", "ERROR"]),
                "msg": " ".join(f"tok{rng.randint(0, 99)}" for _ in range(rng.randint(1, 40))),
            }
        )
        for i in range(lines)
    )


def _markdown(sections: int, *, seed: int) -> str:
    rng = random.Random(seed)
    out = []
    for i in range(sections):
        out.append(f"## Section {i}\n")
        out.append(_prose(rng.randint(1, 3), seed=seed + i))
        out.append("\n- bullet one\n- bullet two\n- bullet three\n")
        out.append("```python\ndef f(x):\n    return x + 1\n```\n")
    return "\n".join(out)


def _source_code(seed: int) -> str:
    rng = random.Random(seed)
    return "\n".join(
        f"def function_{i}(argument_{i}):\n"
        f"    # comment {rng.randint(0, 999)}\n"
        f"    value = argument_{i} * {rng.randint(2, 9)}\n"
        f"    return value\n"
        for i in range(60)
    )


# Fixed shapes, each chosen because it reaches a branch or historically broke a splitter.
_FIXED_CORPUS = {
    "empty_ish": "x",
    "single_sentence": "Ada shipped the parser on Tuesday.",
    "no_separator_at_all": "x" * 4000,
    "only_separators": "\n\n" * 500,
    "whitespace_runs": "Alpha.   \n\n\n\n   Beta.  \t\t  Gamma.\n\n\n",
    "one_token_over_budget": "short. " + "y" * 3000 + ". tail.",
    "repeated_punctuation": "..... ..... ..... ..... " * 40,
    "cjk": "これは日本語のテキストです。これも日本語です。もっと長い文章を書きます。" * 60,
    "mixed_scripts": "Ålpha sätz ett. Béta sætning to. Gämma 文章三. Delta предложение четыре. " * 40,
    "crlf_line_endings": "Line one.\r\nLine two.\r\nLine three.\r\n" * 80,
    "trailing_whitespace_only": "Alpha. Beta. Gamma.   \n\n   ",
    "leading_whitespace_only": "   \n\n   Alpha. Beta. Gamma.",
    "single_json_object": json.dumps({"role": "user", "content": "a single object " * 120}),
    "json_array_of_scalars": json.dumps(list(range(500))),
    "not_quite_jsonl": '{"a": 1}\nplain text line that is not json\n{"b": 2}',
    "jsonl_with_blank_lines": '{"a": 1}\n\n\n{"b": 2}\n\n{"c": 3}',
    "jsonl_one_huge_line": '{"a": 1}\n' + json.dumps({"big": "z" * 6000}) + '\n{"c": 3}',
    "conversation_one_huge_turn": json.dumps(
        [{"role": "user", "content": "q" * 8000}, {"role": "assistant", "content": "short"}]
    ),
}


def _exact_boundary(max_chars: int, pieces: int = 12) -> str:
    """Text whose separator-delimited pieces are exactly ``max_chars`` long.

    The splitter compares a piece against the budget with ``<``, so a piece of exactly
    ``max_chars`` takes the over-budget branch and gets recursed into rather than packed.
    Natural text never lands on that boundary, so it is constructed here to exercise that
    branch on both implementations at once. A piece is the separator plus the text that
    follows it, hence the ``- 2`` for ``". "``.

    Note this does NOT distinguish ``<`` from ``<=``: an exactly-budget piece re-packs to
    itself whichever branch it takes, so that particular mutation is equivalent rather than
    uncaught. What the suite does catch is any change that moves a boundary — mutating the
    packer's budget by one character fails 41 of these tests.
    """
    return "a" * max_chars + (". " + "a" * (max_chars - 2)) * pieces


# Generated shapes, seeded so a failure is reproducible from the test name alone.
_GENERATED_CORPUS = {
    "prose_small": _prose(4, seed=_SEED),
    "prose_large": _prose(120, seed=_SEED + 1),
    "transcript_small": _transcript(6, seed=_SEED + 2),
    "transcript_large": _transcript(200, seed=_SEED + 3),
    "jsonl_small": _jsonl(5, seed=_SEED + 4),
    "jsonl_large": _jsonl(400, seed=_SEED + 5),
    "markdown": _markdown(12, seed=_SEED + 6),
    "source_code": _source_code(_SEED + 7),
    # One per tested chunk size, so each is exercised at the size it was built for.
    **{f"exact_boundary_{size}": _exact_boundary(size) for size in (10, 37, 100, 256, 1500)},
}

CORPUS = {**_FIXED_CORPUS, **_GENERATED_CORPUS}

# Sizes spanning "smaller than one sentence" to "larger than the whole document", including
# the production default (3000) and the size #3756 was reported at (1500).
CHUNK_SIZES = (10, 37, 100, 256, 1500, 3000, 100_000)


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_chunking_matches_the_implementation_it_replaced(name: str):
    """Byte-for-byte agreement with the pre-#3756 chunker, at every chunk size."""
    document = CORPUS[name]
    for max_chars in CHUNK_SIZES:
        expected = _legacy_chunk_text(document, max_chars)
        actual = chunk_text(document, max_chars)
        assert actual == expected, f"{name} at max_chars={max_chars}"


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_chunking_matches_with_a_structured_chunk_size(name: str):
    """Same, with the structured limit set — the JSONL and conversation paths.

    ``structured_chunk_size`` decides how large a single JSONL line or conversation turn may
    be before it is cut as plain text, so it selects between two different branches. Both
    have to agree, not just the default one.
    """
    document = CORPUS[name]
    for max_chars in (100, 1500, 3000):
        for structured in (50, max_chars, max_chars * 4):
            expected = _legacy_chunk_text(document, max_chars, structured_chunk_size=structured)
            actual = chunk_text(document, max_chars, structured_chunk_size=structured)
            assert actual == expected, f"{name} at max_chars={max_chars}, structured={structured}"


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_streaming_and_materialised_forms_agree(name: str):
    """``iter_chunks`` yields exactly what ``chunk_text`` returns.

    Retain reads the streaming form and everything else reads the list form; a divergence
    would mean a document chunked one way on ingest and another on re-chunk.
    """
    document = CORPUS[name]
    for max_chars in CHUNK_SIZES:
        assert list(iter_chunks(document, max_chars)) == chunk_text(document, max_chars), name


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_every_chunk_re_chunks_to_itself(name: str):
    """Idempotency, over the whole corpus (issue #2301).

    The streaming pipeline pre-chunks a document and then re-chunks each piece during
    extraction. A piece that split again would give its sub-chunks one ``chunk_index`` and
    collide on ``chunk_id``.
    """
    document = CORPUS[name]
    for max_chars in (100, 1500, 3000):
        for chunk in chunk_text(document, max_chars):
            assert chunk_text(chunk, max_chars) == [chunk], f"{name} at max_chars={max_chars}"


def test_randomised_documents_agree():
    """A seeded sweep over random shapes, to catch what a hand-written corpus misses.

    Each document is assembled from randomly chosen fragments — separators, punctuation,
    unicode, JSON, whitespace — so the combinations are ones nobody thought to write down.
    The seed is fixed, so a failure names the exact document that broke.
    """
    rng = random.Random(_SEED)
    fragments = [
        "Alpha sentence here. ",
        "Beta! ",
        "Gamma? ",
        "delta; ",
        "epsilon, ",
        "zeta ",
        "\n",
        "\n\n",
        "\r\n",
        "   ",
        "\t",
        "文章です。",
        "x" * 50,
        "y" * 200,
        '{"k": "v"}',
        "...",
    ]

    for case in range(200):
        document = "".join(rng.choice(fragments) for _ in range(rng.randint(1, 120)))
        max_chars = rng.choice([10, 37, 100, 256, 1500])
        structured = rng.choice([None, 50, max_chars, max_chars * 3])
        expected = _legacy_chunk_text(document, max_chars, structured_chunk_size=structured)
        actual = chunk_text(document, max_chars, structured_chunk_size=structured)
        assert actual == expected, (
            f"case {case} (seed {_SEED}) at max_chars={max_chars}, structured={structured}: {document!r}"
        )


def test_the_legacy_reference_really_is_the_old_implementation():
    """Guard the guard: the reference must still be langchain, not a copy of the new code.

    If someone "simplifies" the reference into a call to the live splitter, every test in
    this file passes vacuously forever. Asserting it produces langchain's output for an
    input whose expected split is written out here keeps that from going unnoticed.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text = "Alpha sentence one. Beta sentence two. Gamma sentence three. Delta four."
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=40,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
        separators=_RECURSIVE_TEXT_SEPARATORS,
    )

    assert _legacy_chunk_text(text, 40) == splitter.split_text(text)
    # Written out because it is easy to assume wrong: the splitter runs with
    # keep_separator=True in its default "start" sense, so ". " leads the NEXT chunk rather
    # than closing the previous one.
    assert _legacy_chunk_text(text, 40) == [
        "Alpha sentence one. Beta sentence two",
        ". Gamma sentence three. Delta four.",
    ]
