"""Sizing a body costs O(window), not O(body), and still answers correctly (issue #3756).

``count_tokens`` builds a Python list with one boxed int per token. On a 45 MB retain
submission that is 11.6M ints — measured at +401 MB peak, a quarter of which the allocator
never gave back — to produce a single integer that then gets compared against a batch
budget. ``count_tokens_windowed`` encodes the same text a megabyte at a time instead.

The trade is one token of accuracy per window boundary, in the direction of over-counting.
These tests pin both halves: that the answer stays usable for the comparisons every caller
makes, and that the cost stops tracking the input.
"""

import tracemalloc

import pytest

from hindsight_api.engine.token_encoding import (
    _COUNT_WINDOW_CHARS,
    count_tokens,
    count_tokens_windowed,
)

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
        "A short sentence that fits comfortably inside a single window.",
        _SENTENCE * 100,
        "日本語のテキストもトークン化されます。" * 200,
        "\n\n".join(f"Paragraph {i} with some words in it." for i in range(500)),
    ],
)
def test_matches_the_exact_count_for_anything_inside_one_window(text: str):
    """Below the window size there are no boundaries, so the two must agree exactly."""
    assert len(text) <= _COUNT_WINDOW_CHARS
    assert count_tokens_windowed(text) == count_tokens(text)


def test_stays_within_one_token_per_boundary_across_many_windows():
    """Cutting mid-token can only split one token into two, never lose one.

    The documented error bound, asserted rather than asserted-in-prose: at most one extra
    token per window, and never fewer than the true count.
    """
    text = _SENTENCE * 200_000
    assert len(text) > 5 * _COUNT_WINDOW_CHARS, "the input must actually span several windows"

    exact = count_tokens(text)
    windowed = count_tokens_windowed(text)
    boundaries = (len(text) - 1) // _COUNT_WINDOW_CHARS

    assert windowed >= exact, "windowing must never under-count — a budget check would let oversized work through"
    assert windowed - exact <= boundaries


def test_the_error_is_negligible_against_a_batch_budget():
    """Every caller compares this against a token budget, so the error must not be visible.

    Stated as a ratio because that is the property the callers depend on: the count is
    accurate to far better than any budget's granularity.
    """
    text = _SENTENCE * 200_000

    exact = count_tokens(text)
    windowed = count_tokens_windowed(text)

    assert abs(windowed - exact) / exact < 0.0001


def test_allocation_does_not_track_the_input():
    """The point of the change: cost is set by the window, not by the body.

    An 8x larger body must not allocate 8x more. ``count_tokens`` did exactly that — it
    builds one boxed int per token, so a 45 MB body cost ~385 MB — which is what made a
    large retain OOM a worker before any fact existed.
    """
    small = _SENTENCE * 15_000
    large = _SENTENCE * 120_000
    assert len(large) > 5 * _COUNT_WINDOW_CHARS, "the large input must span several windows"

    count_tokens_windowed(small)  # load the encoding table outside the measurement

    small_cost = _allocated_mb(lambda: count_tokens_windowed(small))
    large_cost = _allocated_mb(lambda: count_tokens_windowed(large))

    # Bounded by the window, so the two are the same size regardless of the 8x input ratio.
    assert large_cost <= small_cost * 1.5, f"small={small_cost:.1f} MB, large={large_cost:.1f} MB"


def test_windowing_allocates_far_less_than_counting_the_whole_body():
    """The exact-count path is still there and still costs what it always did.

    Asserted as a ratio against the same input, so it holds whatever the machine: the
    windowed count must be a small fraction of what encoding the body whole allocates.
    """
    text = _SENTENCE * 120_000

    exact_cost = _allocated_mb(lambda: count_tokens(text))
    windowed_cost = _allocated_mb(lambda: count_tokens_windowed(text))

    assert windowed_cost * 5 < exact_cost, f"windowed={windowed_cost:.1f} MB, exact={exact_cost:.1f} MB"
