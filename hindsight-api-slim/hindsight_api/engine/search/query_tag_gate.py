"""Restrict recall to the tagged identities a query actually names.

Ranking answers "which memories are most related to this query". It cannot answer
"which memory is this query *about*", and it can never answer "none of them" — the
best-scoring memory is still the best-scoring memory on a query the bank has no
answer for. A bank that tags its memories with the names of the things they describe
(``customer:acme``, ``service:auth``, ``name:kubernetes``) already carries what is
needed to answer both: look at which of those names the query mentions.

This module does the lookup. Recall then AND-s the matched tags into the tag filter,
so the restriction is applied in SQL alongside every other tag condition rather than
by re-ranking afterwards, and a query naming nothing known can abstain outright.

Matching is token-based and tolerant of typing slips, because the names users type
are the names they half-remember: ``k8s``, ``mongod``, ``moongoose``. The tolerance
deliberately admits insertions, deletions and transpositions but never substitutions
— at one edit those separate cleanly into "the same word, mistyped" (``eror`` for
``error``, ``typsecript`` for ``typescript``) and "a different word" (``mango`` is
not ``mongo``, ``k9s`` is not ``k8s``). No distance threshold can make that
distinction, since both cases sit at edit distance 1; only the kind of edit can.
"""

from __future__ import annotations

import re
from typing import Literal

TagGateMatch = Literal["exact", "typos"]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, punctuation dropped."""
    return _TOKEN_RE.findall((text or "").lower())


def is_typo_of(candidate: str, target: str) -> bool:
    """True when `candidate` is `target` with one insertion, deletion or transposition.

    Substitutions are excluded on purpose — see the module docstring. Implemented as a
    Damerau-Levenshtein variant with substitution disallowed, short-circuited at a
    distance of one.
    """
    if candidate == target:
        return True
    if abs(len(candidate) - len(target)) > 1:
        return False

    inf = len(candidate) + len(target) + 1
    previous_previous: list[int] = []
    previous = list(range(len(target) + 1))
    for i, c in enumerate(candidate, start=1):
        current = [i]
        for j, t in enumerate(target, start=1):
            if c == t:
                best = previous[j - 1]
            else:
                best = min(previous[j] + 1, current[j - 1] + 1)  # delete / insert only
            if i > 1 and j > 1 and c == target[j - 2] and candidate[i - 2] == t and previous_previous[j - 2] + 1 < best:
                best = previous_previous[j - 2] + 1  # transposition
            current.append(min(best, inf))
        previous_previous, previous = previous, current
    return previous[-1] <= 1


def _phrase_matches(
    phrase_tokens: list[str],
    query_tokens: list[str],
    *,
    match: TagGateMatch,
    min_token_length: int,
) -> bool:
    """Does the query contain this phrase, allowing per-token slips?

    A multi-word name has to appear as a contiguous run: "session backend" must not be
    matched by a query that says "session" in one clause and "backend" in another.
    """
    span = len(phrase_tokens)
    if not span or span > len(query_tokens):
        return False
    for start in range(len(query_tokens) - span + 1):
        window = query_tokens[start : start + span]
        if all(
            _token_matches(p, q, match=match, min_token_length=min_token_length)
            for p, q in zip(phrase_tokens, window, strict=True)
        ):
            return True
    return False


def _token_matches(phrase_token: str, query_token: str, *, match: TagGateMatch, min_token_length: int) -> bool:
    if phrase_token == query_token:
        return True
    if match == "exact":
        return False
    # Short tokens carry too little signal for a one-edit neighbourhood to mean
    # anything: at three characters nearly every identifier is one edit from another.
    if min(len(phrase_token), len(query_token)) < min_token_length:
        return False
    return is_typo_of(query_token, phrase_token)


def match_tags(
    query: str,
    tags: list[str],
    *,
    prefix: str,
    match: TagGateMatch = "typos",
    min_token_length: int = 4,
) -> list[str]:
    """The subset of `tags` whose name the query mentions.

    `tags` is the bank's vocabulary under `prefix`; the prefix is stripped before
    matching so `name:session-backend` is compared as "session backend". Returns the
    full tags, in the order given, so the caller can feed them straight back into a
    tag filter.
    """
    query_tokens = tokenize(query)
    if not query_tokens:
        return []
    matched: list[str] = []
    for tag in tags:
        if not tag.startswith(prefix):
            continue
        phrase_tokens = tokenize(tag[len(prefix) :])
        if _phrase_matches(phrase_tokens, query_tokens, match=match, min_token_length=min_token_length):
            matched.append(tag)
    return matched
