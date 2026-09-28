"""Unit tests for fair group selection in the consolidation fetch (#4823).

When one scope group holds the oldest unconsolidated facts, every round's
``fetch_limit``-oldest fetch contains only that group, so it runs serially
while the other parallel slots sit idle. ``_select_fair_subset`` picks each
round's rows fairly across ``_consolidation_batch_key`` groups instead.

These tests are pure-function (no DB): they pin that
1. a huge oldest group no longer crowds small groups out of a round,
2. within-group oldest-first order is preserved (per-scope serial guarantees
   from #4063/#1604 are unchanged),
3. groups are visited in order of their oldest fact,
4. the per-group share is ``ceil(limit / parallelism)``,
5. edge cases (empty input, zero limit, single group, short round) behave.
"""

from hindsight_api.engine.consolidation.consolidator import (
    _consolidation_batch_key,
    _select_fair_subset,
)


def _mem(i: int, tags: list[str] | None = None) -> dict:
    """Oldest-first candidate row; only the batch key fields matter."""
    return {"id": i, "tags": tags or [], "observation_scopes": None}


class TestSelectFairSubset:
    def test_huge_oldest_group_does_not_starve_small_groups(self):
        # 100 untagged facts (oldest) + 2 tagged facts; round of 8 with
        # parallelism 4 -> per-group share ceil(8/4)=2, so the round holds
        # 2 untagged + 2 tagged instead of 8 untagged.
        candidates = [_mem(i) for i in range(100)] + [_mem(100, ["a"]), _mem(101, ["a"])]
        selected = _select_fair_subset(candidates, limit=8, per_group=2)
        assert len(selected) == 8
        untagged = [m for m in selected if m["tags"] == []]
        tagged = [m for m in selected if m["tags"] == ["a"]]
        assert [m["id"] for m in untagged] == [0, 1, 2, 3, 4, 5]
        assert [m["id"] for m in tagged] == [100, 101]

    def test_oldest_first_fetch_crowds_out_small_group_without_fairness(self):
        # Documents the bug: a plain [:limit] slice of the same input holds
        # only the huge group, leaving parallel slots idle.
        candidates = [_mem(i) for i in range(100)] + [_mem(100, ["a"]), _mem(101, ["a"])]
        assert {m["id"] for m in candidates[:8]} == set(range(8))

    def test_within_group_oldest_first_order_preserved(self):
        candidates = [_mem(i) for i in range(10)] + [_mem(10 + i, ["a"]) for i in range(10)]
        selected = _select_fair_subset(candidates, limit=6, per_group=3)
        untagged = [m["id"] for m in selected if m["tags"] == []]
        tagged = [m["id"] for m in selected if m["tags"] == ["a"]]
        assert untagged == [0, 1, 2]
        assert tagged == [10, 11, 12]

    def test_groups_visited_in_oldest_fact_order(self):
        # Group "b" is newer than the untagged group but older than "a":
        # visit order must be untagged, b, a.
        candidates = (
            [_mem(i) for i in range(4)]
            + [_mem(4 + i, ["b"]) for i in range(4)]
            + [_mem(8 + i, ["a"]) for i in range(4)]
        )
        selected = _select_fair_subset(candidates, limit=3, per_group=1)
        assert [m["id"] for m in selected] == [0, 4, 8]

    def test_per_group_share_is_ceiling_of_limit_over_parallelism(self):
        # limit=8, parallelism=4 -> share 2 per group per pass; two groups of
        # 20 fill the round 4+4 across two passes.
        candidates = [_mem(i) for i in range(20)] + [_mem(20 + i, ["a"]) for i in range(20)]
        selected = _select_fair_subset(candidates, limit=8, per_group=2)
        assert [m["id"] for m in selected if m["tags"] == []] == [0, 1, 2, 3]
        assert [m["id"] for m in selected if m["tags"] == ["a"]] == [20, 21, 22, 23]
        assert len(selected) == 8

    def test_round_robin_across_passes(self):
        # Three groups, limit 6, share 1: two full passes -> 2 per group.
        candidates = (
            [_mem(i) for i in range(5)]
            + [_mem(5 + i, ["a"]) for i in range(5)]
            + [_mem(10 + i, ["b"]) for i in range(5)]
        )
        selected = _select_fair_subset(candidates, limit=6, per_group=1)
        assert len(selected) == 6
        by_key: dict = {}
        for m in selected:
            by_key.setdefault(_consolidation_batch_key(m), []).append(m["id"])
        assert sorted(len(v) for v in by_key.values()) == [2, 2, 2]

    def test_output_keeps_global_oldest_first_order(self):
        candidates = [_mem(i) for i in range(4)] + [_mem(4 + i, ["a"]) for i in range(4)]
        selected = _select_fair_subset(candidates, limit=4, per_group=1)
        assert [m["id"] for m in selected] == sorted(m["id"] for m in selected)

    def test_scope_override_memories_key_by_resolved_scope(self):
        # Memories requesting observation_scopes="shared" batch with each other
        # regardless of native tags — fair selection must use the same key.
        shared_a = {"id": 0, "tags": ["x"], "observation_scopes": [["shared"]]}
        shared_b = {"id": 1, "tags": ["y"], "observation_scopes": [["shared"]]}
        plain = {"id": 2, "tags": [], "observation_scopes": None}
        assert _consolidation_batch_key(shared_a) == _consolidation_batch_key(shared_b)
        assert _consolidation_batch_key(shared_a) != _consolidation_batch_key(plain)
        selected = _select_fair_subset([plain, shared_a, shared_b], limit=2, per_group=1)
        # One from the plain group (id 2) and one from the shared group.
        assert len(selected) == 2
        assert any(m["id"] == 2 for m in selected)
        assert any(m["id"] in (0, 1) for m in selected)

    def test_empty_and_zero_limit(self):
        assert _select_fair_subset([], limit=8, per_group=2) == []
        assert _select_fair_subset([_mem(0)], limit=0, per_group=2) == []

    def test_single_group_returns_oldest_slice(self):
        candidates = [_mem(i) for i in range(10)]
        selected = _select_fair_subset(candidates, limit=4, per_group=2)
        assert [m["id"] for m in selected] == [0, 1, 2, 3]

    def test_short_round_returns_everything(self):
        candidates = [_mem(i) for i in range(3)] + [_mem(3, ["a"])]
        selected = _select_fair_subset(candidates, limit=50, per_group=25)
        assert [m["id"] for m in selected] == [0, 1, 2, 3]

    def test_per_group_floor_is_one(self):
        candidates = [_mem(i) for i in range(4)] + [_mem(4, ["a"])]
        selected = _select_fair_subset(candidates, limit=3, per_group=0)
        assert len(selected) == 3
