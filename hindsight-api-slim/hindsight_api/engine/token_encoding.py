"""Shared tokenizer used for token counting and chunking.

Hindsight tokenizes purely to *count* and *chunk* arbitrary user content — never to
feed a model that relies on the tokenizer's special-token vocabulary. That is what
makes the two choices below safe, and it is why this module exposes exactly two
operations: :func:`count_tokens` and :func:`truncate_to_tokens` (plus
:func:`truncate_many_to_tokens`, the batched form of the latter). Every token budget
in the engine is one or the other.

**Why toktok and not tiktoken.** Counting is on the hot path of both retain and
recall: recall counts once per candidate fact, per candidate chunk, per source fact
and per reranker document, and retain counts whole documents. Measured on this
repo's own text (``hindsight-dev/benchmarks/micro/token_counting.py``), toktok
counts 2-7x faster than tiktoken on cl100k_base and 10-16x faster on o200k_base,
and its ids are byte-identical — the benchmark asserts that before it times
anything. Two properties matter as much as the speed:

* ``count()`` returns an ``int`` without building a Python list of ids. Counting an
  80k-token document cost ~3 MB of transient allocation under tiktoken and costs
  1 KiB here, because there was never a list. (tiktoken has no count-only API;
  ``encode_to_numpy`` is the closest it comes.)
* the vocabularies are compiled *into the extension module*, so nothing is
  downloaded on first use. Air-gapped deployments no longer need the encoding
  pre-baked into the image.

toktok is a small project, so the risk is maintenance, not correctness. It is
contained deliberately: this module is the only place that imports it, and it pulls
in no dependency of its own (numpy is an optional extra Hindsight does not ask for).
Replacing it means rewriting this file and nothing else.

**Why o200k_base by default.** ``HINDSIGHT_API_TOKENIZER_ENCODING`` selects the
vocabulary; it defaults to ``o200k_base``, which is what current OpenAI models
tokenize with. On English and code it counts within a fraction of a percent of
cl100k_base, but on non-Latin scripts and emoji it is far closer to what a modern
model actually charges — ``🧠 … naïve café 東京 مرحبا`` is 19 tokens under
cl100k_base and 13 under o200k_base. Since these counts drive budgets that stand in
for a model's context window, the closer vocabulary is the more honest one. Set the
variable to ``cl100k_base`` to restore the previous counts exactly.

**Special-token literals.** A tiktoken-shaped ``encode()`` defaults to
``disallowed_special="all"``, which makes it *raise* on content that merely mentions
a literal such as ``<|endoftext|>`` — which reached users as an HTTP 500 on
retain/recall (issue #1883). Neither operation here can hit that: both go through
toktok's ``count``/``truncate``, which have no special-token machinery to trigger.
That is the reason to keep using the functions below rather than the tokenizer's own
``encode()`` — that call is the raising one, and #1883 is what it looks like in
production. It is also why the tokenizer itself is not part of this interface.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import toktok

# Vocabularies toktok compiles into its extension module. Used only to make a
# misconfigured HINDSIGHT_API_TOKENIZER_ENCODING fail with a list of what would have
# worked, instead of a bare KeyError from the extension module mid-request.
BUNDLED_ENCODINGS = ("o200k_base", "cl100k_base", "o200k_harmony")


@lru_cache(maxsize=1)
def _load_encoding() -> "toktok._Tokenizer":
    """The tokenizer for ``HINDSIGHT_API_TOKENIZER_ENCODING``.

    Private: the interface is :func:`count_tokens` and :func:`truncate_to_tokens`.
    Handing the raw tokenizer out invites ``encode()``, which is the one spelling
    that raises on special-token literals (#1883). Tests reach for it to name the
    encoding in play and to clear this cache.

    Cached: the tokenizer is immutable and loading one parses a multi-megabyte
    vocabulary. Because the encoding name is read here, changing
    ``HINDSIGHT_API_TOKENIZER_ENCODING`` after the first token count has no effect
    until the cache is cleared — which is what tests do.
    """
    from ..config import get_config

    name = get_config().tokenizer_encoding
    try:
        # toktok's supported API is ``batch_count``/``truncate``, both of which
        # take an encoding *name*. Holding the loaded tokenizer instead lets
        # count_tokens call ``count`` directly — no per-call list to wrap a single
        # string in — and is where the resolved ``.name`` the other two need comes
        # from. ``_encoding`` is the escape hatch toktok's own docstring points at,
        # taken knowingly, and in this module only.
        return toktok._encoding(name)
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
    return _load_encoding().count(text)


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

    The cut lands on a *character* boundary. This used to be
    ``decode(encode(text)[:n])``, which cuts on a *token* boundary — and byte-level
    BPE splits one character across several tokens (``🧠`` is three), so a cut
    could land mid-character and leave a U+FFFD: ``"hello 🧠"`` truncated to two
    tokens decoded to ``"hello \ufffd"``. toktok's ``truncate`` drops the partial
    character instead, giving ``"hello "``, so the result can be one token under
    the budget. Every caller here is a ceiling, so that is free.
    """
    # Negative budgets used to slice a list and silently drop tokens off the end;
    # toktok takes an unsigned count and would raise. Clamp, so a misconfigured cap
    # still degrades to "empty" rather than a 500 mid-request.
    truncated, original_tokens = toktok.truncate(text, max(max_tokens, 0), _load_encoding().name)
    return TokenTruncation(text=truncated, original_tokens=original_tokens)


def truncate_many_to_tokens(texts: Sequence[str], max_tokens: int) -> list[TokenTruncation]:
    """:func:`truncate_to_tokens` over a list, in one call.

    The two callers that truncate a whole list — every reranker document, every
    embedding input — get the cut done in Rust across threads with the GIL
    released, instead of one Python call per text.
    """
    return [
        TokenTruncation(text=truncated, original_tokens=original_tokens)
        for truncated, original_tokens in toktok.batch_truncate(texts, max(max_tokens, 0), _load_encoding().name)
    ]
