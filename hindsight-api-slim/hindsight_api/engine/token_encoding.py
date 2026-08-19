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
    """Count cl100k_base tokens in ``text`` (tolerant of special-token literals)."""
    return len(get_token_encoding().encode(text))


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
