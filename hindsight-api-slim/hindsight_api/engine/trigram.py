"""pg_trgm's trigram similarity, computed in memory.

Shared by entity resolution (the Postgres store's registry) and fuzzy tag matching, so one
notion of "similar name" governs both.
"""

import re

# A pg_trgm "word" is a maximal run of alphanumerics (Unicode letters/digits, underscore excluded);
# everything else (space, punctuation, emoji) is a separator. This is why decoration variants like
# "Wren <emoji>" collapse to the same trigram set.
TRGM_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def trigram_set(text: str) -> set[str]:
    """Trigrams of ``text`` the way PostgreSQL pg_trgm generates them: lowercase, split into words,
    pad each word with two leading + one trailing blank, and take every 3-char window."""
    trigrams: set[str] = set()
    for word in TRGM_WORD.findall(text.lower()):
        padded = f"  {word} "
        for i in range(len(padded) - 2):
            trigrams.add(padded[i : i + 3])
    return trigrams


def trigram_set_similarity(ta: set[str], tb: set[str]) -> float:
    """Jaccard index of two already-computed trigram sets.

    Split out from ``trigram_similarity`` so callers that compare one name against many
    (the candidate scoring loop, the O(N^2) in-batch pass) build each set once instead of
    once per comparison — the loop runs up to ``entity_resolution_max_candidates`` times per
    mention on the retain hot path (GH-3211).
    """
    intersection = len(ta & tb)
    union = len(ta) + len(tb) - intersection
    return intersection / union if union else 0.0


def trigram_similarity(a: str, b: str) -> float:
    """pg_trgm ``similarity(a, b)`` computed in-memory — the Jaccard index of the trigram sets.

    Public because two subsystems share it: entity resolution (``memories/pg/entity_resolver``)
    and fuzzy tag matching (``search.tag_resolution``). One notion of "similar name" for both, so a change to it is a
    deliberate change to both — see ``tests/test_entity_intrabatch_clustering.py``, which pins
    the values against Postgres.

    Verified byte-for-byte against Postgres pg_trgm across emoji / accent / CJK / hyphen /
    apostrophe cases (issue #3107), so the merge cutoff calibrated on pg_trgm transfers exactly.
    Doing it in Python keeps the in-batch dedup off the retain transaction's DB connection and makes
    it backend-agnostic (Postgres, Oracle, and the pg_trgm-absent "full" fallback all behave alike).
    """
    return trigram_set_similarity(trigram_set(a), trigram_set(b))
