"""Recall result deduplication helpers."""

import re
from difflib import SequenceMatcher

from .types import ScoredResult

_RAW_FACT_TYPES = frozenset({"world", "experience"})
_WORD_RE = re.compile(r"\w+")


def normalize_recall_dedup_text(text: str) -> str:
    """Normalize recall text for duplicate comparison."""
    return " ".join(_WORD_RE.findall(text.casefold()))


def recall_text_similarity(left: str, right: str) -> float:
    """Return normalized lexical similarity for two recall result texts."""
    left_normalized = normalize_recall_dedup_text(left)
    right_normalized = normalize_recall_dedup_text(right)
    if not left_normalized or not right_normalized:
        return 0.0
    if left_normalized == right_normalized:
        return 1.0
    return SequenceMatcher(None, left_normalized, right_normalized).ratio()


def collapse_near_duplicate_raw_facts(
    scored_results: list[ScoredResult],
    threshold: float,
) -> list[ScoredResult]:
    """Drop lower-ranked raw facts whose text duplicates an earlier raw fact.

    The input is already sorted by final recall rank. Dedup therefore keeps the
    first/highest-ranked result in each duplicate cluster and lets downstream
    truncation backfill freed slots from later candidates.
    """
    if threshold <= 0.0:
        return list(scored_results)

    retained_results: list[ScoredResult] = []
    retained_world_texts: list[str] = []
    retained_experience_texts: list[str] = []
    retained_world_text_set: set[str] = set()
    retained_experience_text_set: set[str] = set()

    for scored_result in scored_results:
        retrieval = scored_result.retrieval
        if retrieval.fact_type not in _RAW_FACT_TYPES:
            retained_results.append(scored_result)
            continue

        normalized_text = normalize_recall_dedup_text(retrieval.text)
        if not normalized_text:
            retained_results.append(scored_result)
            continue

        retained_texts = retained_world_texts
        retained_text_set = retained_world_text_set
        if retrieval.fact_type == "experience":
            retained_texts = retained_experience_texts
            retained_text_set = retained_experience_text_set

        if threshold >= 1.0:
            is_duplicate = normalized_text in retained_text_set
        else:
            is_duplicate = any(
                retained_text == normalized_text
                or SequenceMatcher(None, retained_text, normalized_text).ratio() >= threshold
                for retained_text in retained_texts
            )
        if is_duplicate:
            continue

        retained_results.append(scored_result)
        retained_texts.append(normalized_text)
        retained_text_set.add(normalized_text)

    return retained_results
