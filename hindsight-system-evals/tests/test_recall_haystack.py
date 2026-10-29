"""The recall pool must sit next to every case without answering one.

A pool memory naming a word a case grades on either passes that case for the
wrong reason or fails it with a second right answer — and either way the eval
stops measuring ranking. Cheap to check here, invisible in a run.
"""

from __future__ import annotations

import datetime

from hindsight_system_evals.recall_capabilities import CASE_BANKS, CASES, _names
from hindsight_system_evals.recall_haystack import BLACKOUTS, POOL, haystack_items


def test_no_pool_memory_names_a_graded_word() -> None:
    clashes = [
        f"{case.id}: {word!r} in {text!r}"
        for case in CASES
        for word in (*case.found, *(w for pair in case.ranked for w in pair))
        for text in (*POOL, *(m.content for m in case.decoys))
        if _names(text, word)
    ]
    assert not clashes, "\n".join(clashes)


def test_no_pool_memory_is_dated_inside_a_window_a_case_asks_about() -> None:
    days = [datetime.date.fromisoformat(str(item["timestamp"])[:10]) for item in haystack_items(len(POOL))]
    assert not [d for d in days if any(lo <= d <= hi for lo, hi in BLACKOUTS)]
    assert len(days) == len(POOL)


def test_every_case_has_its_recorded_bank() -> None:
    """A run imports each case's archive; a missing one fails that case mid-run instead of here."""
    assert not [case.id for case in CASES if not (CASE_BANKS / f"{case.id}.zip").exists()]
