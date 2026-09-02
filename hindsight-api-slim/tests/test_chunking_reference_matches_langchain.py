"""Re-verify the vendored chunking oracle against the real langchain splitter.

``tests/chunking_reference.py`` is a transcription of langchain's
``RecursiveCharacterTextSplitter``, kept so the differential chunking tests kept their
oracle when ``langchain-text-splitters`` left the dependency tree (it pulled in
``langchain-core`` -> ``langsmith`` -> ``orjson``).

A transcription is only as good as its last check against the original, so that check
lives here rather than in a commit message. **This test skips by default** — langchain is
deliberately not installed. To run it:

    uv run --with langchain-text-splitters pytest tests/test_chunking_reference_matches_langchain.py

Do that after touching ``chunking_reference.py``, or when bumping the langchain version
the transcription claims to mirror. It matched byte-for-byte over 20,079 comparisons
against ``langchain-text-splitters`` 1.1.2 when the dependency was removed.
"""

import random

import pytest

from hindsight_api.engine.retain.fact_extraction import _RECURSIVE_TEXT_SEPARATORS
from tests.chunking_reference import recursive_split
from tests.test_chunking_streams import _PLAIN_TEXT_CASES

# Skips the whole module when langchain is absent, which is the normal state.
langchain_text_splitters = pytest.importorskip(
    "langchain_text_splitters",
    reason="langchain is not a dependency; install it explicitly to re-verify the oracle",
)


def _langchain_split(text: str, max_chars: int) -> list[str]:
    splitter = langchain_text_splitters.RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
        separators=_RECURSIVE_TEXT_SEPARATORS,
    )
    return splitter.split_text(text)


def test_reference_matches_langchain_on_the_separator_tier_corpus():
    """Every case the streaming tests use, at every size they use."""
    for name, text in _PLAIN_TEXT_CASES.items():
        for max_chars in (10, 17, 40, 100, 1000):
            assert recursive_split(text, max_chars, _RECURSIVE_TEXT_SEPARATORS) == _langchain_split(text, max_chars), (
                f"{name} at max_chars={max_chars}"
            )


def test_reference_matches_langchain_on_fuzzed_documents():
    """Randomly assembled documents, over the fragment shapes the legacy corpus uses.

    Seeded so a failure is reproducible; the fragments deliberately include the cases that
    break splitters — empty strings, whitespace runs, CRLF, CJK, and tokens larger than any
    chunk size tried.
    """
    fragments = [
        "Alpha sentence one. ",
        "Beta two! ",
        "Gamma three? ",
        "delta; ",
        "eps, ",
        "\n\n",
        "\n",
        " ",
        "",
        "文章です。",
        "x" * 50,
        "y" * 200,
        '{"k": "v"}',
        "...",
        "\r\n",
        "\t",
        "   ",
        "Ålpha ",
        "a" * 300,
    ]
    rng = random.Random(20260902)

    for case in range(2000):
        document = "".join(rng.choice(fragments) for _ in range(rng.randint(1, 60)))
        max_chars = rng.choice([1, 2, 5, 10, 37, 100, 256, 1500])
        assert recursive_split(document, max_chars, _RECURSIVE_TEXT_SEPARATORS) == _langchain_split(
            document, max_chars
        ), f"case {case} at max_chars={max_chars}: {document!r}"
