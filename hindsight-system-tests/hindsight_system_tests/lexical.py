"""A deterministic stand-in for a neural embedding model.

The stub has to answer ``/v1/embeddings`` with vectors that make semantic search
*rank sensibly* — a test that retains "Alice moved to Berlin" and recalls "where
does Alice live" must get its memory back — without loading a 130MB model or
spending a millisecond doing it.

The trick is to hash words into a fixed-dimension sparse vector and normalize:
cosine similarity between two such vectors then tracks their word overlap. It is
not semantics, but it is a total order that is stable across machines, processes
and runs, which is what a system test actually needs. A story that turns on two
*lexically* unrelated strings retrieving each other — which real embeddings do and
word overlap cannot — will need an explicit override here; none does yet.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

# Matches the server's DEFAULT_EMBEDDING_DIMENSION. The vector columns are
# created at this width, so the stub must answer with exactly this many floats.
EMBEDDING_DIMENSION = 384

# Latin/digit runs are one token each; every CJK codepoint is a token on its own.
# Chinese and Japanese are not space-delimited, so an alphanumeric-only pattern
# silently drops them entirely — the query and the memory both tokenize to
# nothing, land on the same fallback vector, and retrieval becomes meaningless.
# Per-character tokens are roughly what a real CJK analyser does at the unigram
# level and are enough to make overlap track relatedness.
_CJK = (
    r"\u4e00-\u9fff"  # CJK unified ideographs
    r"\u3040-\u309f"  # hiragana
    r"\u30a0-\u30ff"  # katakana
    r"\uac00-\ud7af"  # hangul syllables
)
_TOKEN_RE = re.compile(rf"[a-z0-9]+|[{_CJK}]")


def tokenize(text: str) -> list[str]:
    """Lowercase tokens. Deliberately crude — determinism beats linguistics."""
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class _Slots:
    """Where one token lands in the vector.

    Two positions rather than one so a single hash collision degrades the score
    instead of merging two tokens outright.
    """

    primary: int
    secondary: int


def _slots(token: str, dimension: int) -> _Slots:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    return _Slots(primary=value % dimension, secondary=(value >> 32) % dimension)


def lexical_embedding(text: str, dimension: int = EMBEDDING_DIMENSION) -> list[float]:
    """A unit-length vector whose cosine similarity tracks word overlap."""
    vector = [0.0] * dimension
    for token in tokenize(text):
        slots = _slots(token, dimension)
        vector[slots.primary] += 1.0
        vector[slots.secondary] += 0.5

    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        # Empty or punctuation-only text. A zero vector would divide by zero here
        # and produce NaN distances in pgvector, so park it on a fixed axis that
        # is orthogonal to nothing in particular.
        vector[0] = 1.0
        return vector

    return [component / norm for component in vector]


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def lexical_relevance(query: str, document: str) -> float:
    """Rerank score in [0, 1] from the same lexical model, so ranking stays consistent."""
    score = cosine(lexical_embedding(query), lexical_embedding(document))
    return max(0.0, min(1.0, score))
