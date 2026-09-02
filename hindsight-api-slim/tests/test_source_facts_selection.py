"""Tests for the recall source_facts token-budget selection (issue #3221).

The budget must be spent in observation-rank order, an oversized fact must skip
only itself, and any budget skip must be reported so a caller can tell a
truncated map from a dangling reference.
"""

from hindsight_api.engine.source_facts import select_source_facts_within_budget


def _count_words(text: str) -> int:
    """Stand-in tokenizer: one token per word, so budgets read literally."""
    return len(text.split())


def _select(
    *,
    by_obs: dict[str, list[str]],
    texts: dict[str, str],
    max_total_tokens: int = -1,
    max_tokens_per_observation: int = -1,
):
    ordered: list[str] = []
    seen: set[str] = set()
    for sids in by_obs.values():
        for sid in sids:
            if sid not in seen:
                seen.add(sid)
                ordered.append(sid)
    return select_source_facts_within_budget(
        source_ids_ordered=ordered,
        source_fact_ids_by_obs=by_obs,
        text_by_id=texts,
        max_total_tokens=max_total_tokens,
        max_tokens_per_observation=max_tokens_per_observation,
        count_tokens=_count_words,
    )


class TestGlobalBudget:
    def test_budget_is_spent_in_rank_order(self):
        """The top-ranked observation keeps its provenance; the tail loses it."""
        by_obs = {"obs-rank-1": ["s1"], "obs-rank-2": ["s2"], "obs-rank-3": ["s3"]}
        texts = {"s1": "one two", "s2": "three four", "s3": "five six"}

        selection = _select(by_obs=by_obs, texts=texts, max_total_tokens=2)

        assert selection.ids == ["s1"]
        assert selection.truncated is True

    def test_unlimited_budget_keeps_everything(self):
        by_obs = {"obs-1": ["s1", "s2"], "obs-2": ["s3"]}
        texts = {"s1": "a b c", "s2": "d e f", "s3": "g h i"}

        selection = _select(by_obs=by_obs, texts=texts, max_total_tokens=-1)

        assert selection.ids == ["s1", "s2", "s3"]
        assert selection.truncated is False

    def test_oversized_fact_does_not_evict_the_facts_behind_it(self):
        """One long fact skips itself only — shorter facts behind it still fit."""
        by_obs = {"obs-1": ["long"], "obs-2": ["short-a"], "obs-3": ["short-b"]}
        texts = {"long": "w " * 50, "short-a": "a", "short-b": "b"}

        selection = _select(by_obs=by_obs, texts=texts, max_total_tokens=2)

        assert selection.ids == ["short-a", "short-b"]
        assert selection.truncated is True

    def test_shared_source_counted_once(self):
        """A source cited by two observations is selected (and charged) once."""
        by_obs = {"obs-1": ["shared"], "obs-2": ["shared", "s2"]}
        texts = {"shared": "a b", "s2": "c d"}

        selection = _select(by_obs=by_obs, texts=texts, max_total_tokens=4)

        assert selection.ids == ["shared", "s2"]
        assert selection.truncated is False

    def test_unresolvable_id_is_not_truncation(self):
        """A source row that did not resolve is skipped without flagging truncation."""
        by_obs = {"obs-1": ["s1", "missing"]}
        texts = {"s1": "a b"}

        selection = _select(by_obs=by_obs, texts=texts, max_total_tokens=-1)

        assert selection.ids == ["s1"]
        assert selection.truncated is False


class TestPerObservationBudget:
    def test_each_observation_gets_its_own_budget(self):
        by_obs = {"obs-1": ["s1"], "obs-2": ["s2"]}
        texts = {"s1": "a b", "s2": "c d"}

        selection = _select(by_obs=by_obs, texts=texts, max_tokens_per_observation=2)

        assert selection.ids == ["s1", "s2"]
        assert selection.truncated is False

    def test_oversized_fact_does_not_evict_the_facts_behind_it(self):
        by_obs = {"obs-1": ["long", "short"]}
        texts = {"long": "w " * 50, "short": "a"}

        selection = _select(by_obs=by_obs, texts=texts, max_tokens_per_observation=1)

        assert selection.ids == ["short"]
        assert selection.truncated is True

    def test_per_observation_cap_takes_precedence_over_global(self):
        """With a per-observation cap set, the global budget does not apply."""
        by_obs = {"obs-1": ["s1"], "obs-2": ["s2"]}
        texts = {"s1": "a b", "s2": "c d"}

        selection = _select(
            by_obs=by_obs,
            texts=texts,
            max_total_tokens=2,
            max_tokens_per_observation=2,
        )

        assert selection.ids == ["s1", "s2"]
        assert selection.truncated is False

    def test_fact_dropped_by_one_observation_but_kept_by_another_is_not_truncation(self):
        """A source that still lands in the map was not lost, so nothing is flagged."""
        by_obs = {"obs-1": ["filler", "shared"], "obs-2": ["shared"]}
        texts = {"filler": "a", "shared": "b"}

        # obs-1 spends its whole budget on "filler" and skips "shared"; obs-2 has room.
        selection = _select(by_obs=by_obs, texts=texts, max_tokens_per_observation=1)

        assert set(selection.ids) == {"filler", "shared"}
        assert selection.truncated is False

    def test_zero_cap_drops_every_fact_and_reports_truncation(self):
        by_obs = {"obs-1": ["s1"]}
        texts = {"s1": "a"}

        selection = _select(by_obs=by_obs, texts=texts, max_tokens_per_observation=0)

        assert selection.ids == []
        assert selection.truncated is True
