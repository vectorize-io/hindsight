"""Shared tokenizer used for token counting and chunking.

Hindsight tokenizes purely to *count* and *chunk* arbitrary user content — never to
feed a model that relies on the tokenizer's special-token vocabulary. That is what
makes the two choices below safe.

**Why quicktok and not tiktoken.** Counting is on the hot path of both retain and
recall: recall counts once per candidate fact, per candidate chunk, per source fact
and per reranker document, and retain counts whole documents. Measured on this
repo's own text (``hindsight-dev/benchmarks/micro/token_counting.py``), quicktok
counts 4-12x faster than tiktoken on cl100k_base and 8-19x faster on o200k_base,
and its ids are byte-identical — the benchmark asserts that before it times
anything. Two properties matter as much as the speed:

* ``count()`` returns an ``int`` without building a Python list of ids. Counting an
  80k-token document cost 2.8 MB of transient allocation under tiktoken and costs
  ~0 here, because there was never a list. (tiktoken has no count-only API;
  ``encode_to_numpy`` is the closest it comes.)
* the vocabularies ship *inside the wheel*, so nothing is downloaded on first use.
  Air-gapped deployments no longer need the encoding pre-baked into the image.

quicktok is a small project, so the risk is maintenance, not correctness. It is
contained deliberately: this module is the only place that imports it, every token
call site in the engine routes through ``get_token_encoding()`` or
``count_tokens()``, and its sole dependency (numpy) is one Hindsight already has.
Replacing it means rewriting this file and nothing else.

**Why o200k_base by default.** ``HINDSIGHT_API_TOKENIZER_ENCODING`` selects the
vocabulary; it defaults to ``o200k_base``, which is what current OpenAI models
tokenize with. On English and code it counts within a fraction of a percent of
cl100k_base, but on non-Latin scripts and emoji it is far closer to what a modern
model actually charges — ``🧠 … naïve café 東京 مرحبا`` is 19 tokens under
cl100k_base and 13 under o200k_base. Since these counts drive budgets that stand in
for a model's context window, the closer vocabulary is the more honest one. Set the
variable to ``cl100k_base`` to restore the previous counts exactly.

**Special-token literals.** Both tokenizers default to ``disallowed_special="all"``,
which makes ``encode()`` *raise* on content that merely mentions a literal such as
``<|endoftext|>`` — surfacing as an HTTP 500 on retain/recall (issue #1883).
``_SafeEncoding`` disables that check so such literals are counted as ordinary text.
``count()`` never applies the check in the first place.
"""

from dataclasses import dataclass
from functools import lru_cache

import quicktok

# Vocabularies quicktok ships in its wheel. Used only to make a misconfigured
# HINDSIGHT_API_TOKENIZER_ENCODING fail with a list of what would have worked,
# instead of a bare RuntimeError from the extension module mid-request.
BUNDLED_ENCODINGS = ("o200k_base", "cl100k_base", "o200k_harmony", "llama3", "qwen3")


class _SafeEncoding:
    """Wraps a quicktok ``Tokenizer`` so ``encode()`` never raises on special-token literals."""

    def __init__(self, tokenizer: "quicktok.Tokenizer") -> None:
        self._tokenizer = tokenizer

    @property
    def name(self) -> str:
        """The encoding's name, e.g. ``o200k_base``."""
        return self._tokenizer.name

    def count(self, text: str) -> int:
        """Token count, without materialising the ids.

        Prefer this over ``len(encode(text))`` wherever only the count is wanted —
        it is the whole reason this module is on quicktok.
        """
        return self._tokenizer.count(text)

    def encode(self, text: str, **kwargs) -> list[int]:
        # Count special-token literals as ordinary text instead of rejecting them.
        kwargs.setdefault("disallowed_special", ())
        return self._tokenizer.encode(text, **kwargs)

    def decode(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens)


@lru_cache(maxsize=1)
def get_token_encoding() -> _SafeEncoding:
    """The configured encoding, wrapped to tolerate special-token literals.

    Cached: the tokenizer is immutable and loading one parses a multi-megabyte
    vocabulary. Because the encoding name is read here, changing
    ``HINDSIGHT_API_TOKENIZER_ENCODING`` after the first token count has no effect
    until the cache is cleared — which is what tests do.
    """
    from ..config import get_config

    name = get_config().tokenizer_encoding
    try:
        return _SafeEncoding(quicktok.get_encoding(name))
    except Exception as err:
        raise ValueError(
            f"Unknown tokenizer encoding {name!r} (HINDSIGHT_API_TOKENIZER_ENCODING). "
            f"Available: {', '.join(BUNDLED_ENCODINGS)}."
        ) from err


def count_tokens(text: str) -> int:
    """Count tokens in ``text`` under the configured encoding.

    Tolerant of special-token literals, and never builds a list of ids — so this is
    safe on a whole document, not just a fact or a chunk.

    It did not used to be. Under tiktoken this was ``len(encode(text))``, whose
    ``list[int]`` costs ~40 bytes per token: counting a 45 MB retain body built an
    11.6M-element list purely to produce one integer (+472 MB peak, a third of which
    the allocator never gave back). #3756 worked around that with a
    ``count_tokens_windowed`` that encoded a megabyte at a time and accepted an
    approximate answer, since a fixed character cut can split a token. ``count()``
    removes the reason for both: it allocates nothing and it is exact.
    """
    return get_token_encoding().count(text)


@dataclass(frozen=True)
class TokenTruncation:
    """Result of :func:`truncate_to_tokens`."""

    text: str  # the (possibly truncated) text
    original_tokens: int  # token count of the input before any truncation


def truncate_to_tokens(text: str, max_tokens: int) -> TokenTruncation:
    """Truncate ``text`` to at most ``max_tokens`` tokens.

    The configured encoding is an approximation of any given provider's tokenizer,
    so set ``max_tokens`` with a little headroom below the model's real limit.

    ``original_tokens`` is the input's token count (so the caller can report how
    much was dropped) whether or not truncation occurred.
    """
    enc = get_token_encoding()
    # Count first: the common case is "fits", and that costs no id list at all.
    original_tokens = enc.count(text)
    if original_tokens <= max_tokens:
        return TokenTruncation(text=text, original_tokens=original_tokens)
    return TokenTruncation(text=enc.decode(enc.encode(text)[:max_tokens]), original_tokens=original_tokens)
