"""Tests for the recall max_tokens selection (issue #3688).

The budget must be spent in rank order, an oversized fact must skip only itself,
a run that matched something must never answer with nothing, and any budget skip
must be reported so a caller can tell a truncated answer from an empty bank.
"""

from hindsight_api.engine.fact_budget import select_facts_within_budget


def _count_words(text: str) -> int:
    """Stand-in tokenizer: one token per word, so budgets read literally."""
    return len(text.split())


def _select(texts: dict[str, str], max_tokens: int):
    return select_facts_within_budget(
        fact_ids_ordered=list(texts),
        text_by_id=texts,
        max_tokens=max_tokens,
        count_tokens=_count_words,
    )


class TestBudgetPacking:
    def test_budget_is_spent_in_rank_order(self):
        """The top-ranked facts survive; the tail is what the budget drops."""
        selection = _select({"rank-1": "one two", "rank-2": "three four", "rank-3": "five six"}, max_tokens=4)

        assert selection.ids == ["rank-1", "rank-2"]
        assert selection.total_tokens == 4
        assert selection.truncated is True

    def test_generous_budget_keeps_everything(self):
        selection = _select({"f1": "a b c", "f2": "d e f"}, max_tokens=100)

        assert selection.ids == ["f1", "f2"]
        assert selection.total_tokens == 6
        assert selection.truncated is False

    def test_oversized_fact_does_not_evict_the_facts_behind_it(self):
        """The regression from #3221, on the fact budget this time: one long fact
        used to `break` the loop and drop every shorter fact ranked below it."""
        texts = {"long": "a b c d e f g h", "short-1": "i j", "short-2": "k l"}

        selection = _select(texts, max_tokens=5)

        assert selection.ids == ["short-1", "short-2"]
        assert selection.total_tokens == 4
        assert selection.truncated is True

    def test_unresolvable_ids_are_skipped_without_flagging_truncation(self):
        """A missing text is a missing row, not a budget skip."""
        selection = select_facts_within_budget(
            fact_ids_ordered=["present", "absent"],
            text_by_id={"present": "a b"},
            max_tokens=100,
            count_tokens=_count_words,
        )

        assert selection.ids == ["present"]
        assert selection.truncated is False

    def test_no_candidates_is_not_truncation(self):
        selection = _select({}, max_tokens=100)

        assert selection.ids == []
        assert selection.total_tokens == 0
        assert selection.truncated is False


class TestFloor:
    def test_a_match_is_never_answered_with_nothing(self):
        """#3688: with every fact individually over budget, recall returned [],
        which reads to an agent as 'this bank knows nothing about that'."""
        texts = {"rank-1": "a b c d e", "rank-2": "f g h i j"}

        selection = _select(texts, max_tokens=2)

        assert selection.ids == ["rank-1"]
        assert selection.truncated is True

    def test_the_floor_reports_the_tokens_it_actually_spent(self):
        """The overshoot is visible rather than silently reported as in-budget."""
        selection = _select({"rank-1": "a b c d e"}, max_tokens=2)

        assert selection.total_tokens == 5

    def test_the_floor_takes_the_top_ranked_resolvable_fact(self):
        selection = select_facts_within_budget(
            fact_ids_ordered=["absent", "rank-1", "rank-2"],
            text_by_id={"rank-1": "a b c", "rank-2": "d e f"},
            max_tokens=1,
            count_tokens=_count_words,
        )

        assert selection.ids == ["rank-1"]

    def test_zero_budget_still_means_no_facts(self):
        """max_tokens=0 is the documented way to ask for chunks only (issue #364),
        so the floor must not smuggle a fact into that answer."""
        selection = _select({"rank-1": "a b c"}, max_tokens=0)

        assert selection.ids == []
        assert selection.total_tokens == 0
        assert selection.truncated is True

    def test_negative_budget_selects_nothing(self):
        selection = _select({"rank-1": "a b c"}, max_tokens=-1)

        assert selection.ids == []
