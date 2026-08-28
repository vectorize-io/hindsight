"""Sizing a body costs O(1), not O(body), and answers exactly (issue #3756).

Counting a retain submission used to be the single largest allocation in the front
half of retain. Under tiktoken ``count_tokens`` was ``len(encode(text))``, which
builds a Python list with one boxed int per token: on a 45 MB submission that is
11.6M ints — measured at +401 MB peak, a quarter of which the allocator never gave
back — to produce one integer that then gets compared against a batch budget. That
is what OOM'd a worker before a single fact existed.

#3756 worked around it with a ``count_tokens_windowed`` that encoded a megabyte at
a time and paid for it in accuracy: a fixed character cut can split a token, so the
answer could over-count by one token per window boundary.

The tokenizer swap removed the reason for the workaround. ``quicktok.count()``
returns an ``int`` without ever building the list, so the plain ``count_tokens`` is
now both allocation-free and exact, and the windowed variant is gone. These tests
pin the property that made #3756 worth fixing — the cost does not track the input —
now against the function every caller actually uses.
"""

import tracemalloc

import pytest

from hindsight_api.engine.token_encoding import count_tokens, get_token_encoding

_SENTENCE = "The quick brown fox jumps over the lazy dog near the river bank in 2023. "


def _allocated_mb(fn) -> float:
    """Peak Python bytes ``fn`` allocates, in MB.

    ``tracemalloc``, not RSS. RSS cannot attribute an allocation to the code that made it:
    the allocator maps arenas on first touch and reuses them silently, so the same call
    reads as +400 MB or as +0 MB depending only on what ran before it. That noise is what
    made #3756's original diagnosis wrong, and an RSS-based test of it flaky.
    """
    tracemalloc.start()
    try:
        fn()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / 1024 / 1024


@pytest.mark.parametrize(
    "text",
    [
        "",
        "one",
        "A short sentence.",
        _SENTENCE * 100,
        "日本語のテキストもトークン化されます。" * 200,
        "\n\n".join(f"Paragraph {i} with some words in it." for i in range(500)),
    ],
)
def test_counting_is_exact(text: str):
    """No approximation left to tolerate.

    The windowed counter traded accuracy for memory; this one trades nothing, so the
    count must equal what encoding the text produces, on every input.
    """
    assert count_tokens(text) == len(get_token_encoding().encode(text))


def test_allocation_does_not_track_the_input():
    """The property #3756 was about: cost is set by the call, not by the body.

    An 8x larger body must not allocate 8x more. The old ``len(encode(text))`` did
    exactly that — one boxed int per token, so a 45 MB body cost ~385 MB.
    """
    small = _SENTENCE * 15_000
    large = _SENTENCE * 120_000

    count_tokens(small)  # load the vocabulary outside the measurement

    small_cost = _allocated_mb(lambda: count_tokens(small))
    large_cost = _allocated_mb(lambda: count_tokens(large))

    assert large_cost <= small_cost + 1.0, f"small={small_cost:.3f} MB, large={large_cost:.3f} MB"


def test_counting_a_large_body_allocates_almost_nothing():
    """Stated in absolute terms, because the ratio alone would pass at any scale.

    ~8.7 MB of text, ~2M tokens. The list this used to build would be ~80 MB; the
    count-only API builds nothing, so anything above a megabyte here means a list
    has crept back onto the path.
    """
    text = _SENTENCE * 120_000

    count_tokens("warm")  # load the vocabulary outside the measurement

    assert _allocated_mb(lambda: count_tokens(text)) < 1.0
