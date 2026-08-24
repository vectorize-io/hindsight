"""Shared tiktoken encoding used for token counting and chunking.

Hindsight uses tiktoken purely to *count* and *chunk* arbitrary user content — never
to feed a model that relies on tiktoken's special-token vocabulary. With tiktoken's
default ``disallowed_special="all"``, any content that merely *mentions* a special-token
literal (e.g. ``<|endoftext|>``) makes ``encode()`` raise, which surfaces as an HTTP 500
on retain/recall (see issue #1883).

``_SafeEncoding`` disables that check so such literals are counted as ordinary text. Token
counts are unaffected; this only stops the encoder from rejecting valid input. Every token
call site in the engine routes through ``get_token_encoding()``, so the fix is global.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache

import tiktoken


class _SafeEncoding:
    """Wraps a tiktoken ``Encoding`` so ``encode()`` never raises on special-token literals."""

    def __init__(self, encoding: tiktoken.Encoding) -> None:
        self._encoding = encoding

    def encode(self, text: str, **kwargs) -> list[int]:
        # Count special-token literals as ordinary text instead of rejecting them.
        kwargs.setdefault("disallowed_special", ())
        return self._encoding.encode(text, **kwargs)

    def decode(self, tokens: list[int]) -> str:
        return self._encoding.decode(tokens)


@lru_cache(maxsize=1)
def get_token_encoding() -> _SafeEncoding:
    """Cached cl100k_base encoding (GPT-4/3.5) wrapped to tolerate special-token literals.

    tiktoken downloads the encoding on first lookup; keeping it lazy means importing
    ``hindsight_api`` does not require network access.
    """
    return _SafeEncoding(tiktoken.get_encoding("cl100k_base"))


def count_tokens(text: str) -> int:
    """Count cl100k_base tokens in ``text`` (tolerant of special-token literals).

    Allocates a token list proportional to ``text`` and throws it away — fine for a
    fact, a query, or one chunk, and NOT fine for a whole document. Sizing a retain
    submission goes through :func:`count_tokens_windowed` instead; see there for why
    (#3756).
    """
    return len(get_token_encoding().encode(text))


# How much text one ``encode()`` call is allowed to see when counting a body that may be
# arbitrarily large. tiktoken returns a Python ``list[int]`` — ~40 bytes per token once the
# ids are boxed as PyLongs — so the transient cost of a count is set by this, not by the
# input. 1 MiB holds ~250k tokens ≈ 10 MB of list, small enough to disappear against a
# worker's baseline and large enough that the per-call overhead stays irrelevant.
_COUNT_WINDOW_CHARS = 1024 * 1024


def _iter_window_token_counts(text: str, window_chars: int = _COUNT_WINDOW_CHARS) -> Iterator[int]:
    """Yield the token count of each ``window_chars``-sized slice of ``text``, in order.

    Each window's token list is released before the next is encoded, so peak memory is
    O(window) rather than O(text).
    """
    encoding = get_token_encoding()
    for start in range(0, len(text), window_chars):
        yield len(encoding.encode(text[start : start + window_chars]))


def count_tokens_windowed(text: str) -> int:
    """Approximate cl100k_base token count of ``text`` at O(window) peak memory.

    Encoding a whole document to measure it is the single largest allocation in the
    retain front half: a 45 MB body produced an 11.6M-element token list (+472 MB peak,
    a third of which the allocator never returned) purely to compute one integer. This
    counts the same text a megabyte at a time instead — measured at +35 MB peak and ~5x
    faster on that body (#3756).

    **Approximate, deliberately.** Cutting the text at a fixed character offset can split
    one token into two, so the result may exceed the true count by at most one token per
    window boundary — on a 45 MB body that is ~45 tokens out of 11.6M (0.0003%). Every
    caller compares against a batch-size budget or logs the number, and none of them can
    observe an error that small. Use :func:`count_tokens` where the count must be exact
    (truncating to a provider's hard input limit, for instance), and accept that it costs
    memory proportional to its input.
    """
    if not text:
        return 0
    return sum(_iter_window_token_counts(text))


@dataclass(frozen=True)
class TokenTruncation:
    """Result of :func:`truncate_to_tokens`."""

    text: str  # the (possibly truncated) text
    original_tokens: int  # token count of the input before any truncation


def truncate_to_tokens(text: str, max_tokens: int) -> TokenTruncation:
    """Truncate ``text`` to at most ``max_tokens`` cl100k_base tokens.

    tiktoken is an approximation of any given provider's tokenizer, so set
    ``max_tokens`` with a little headroom below the model's real limit.

    ``original_tokens`` is the input's token count (so the caller can report how
    much was dropped) whether or not truncation occurred.
    """
    enc = get_token_encoding()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return TokenTruncation(text=text, original_tokens=len(tokens))
    return TokenTruncation(text=enc.decode(tokens[:max_tokens]), original_tokens=len(tokens))
