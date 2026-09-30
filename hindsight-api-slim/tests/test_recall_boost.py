"""Tests for per-strategy recall boosting (config parsing + boost math)."""

from dataclasses import dataclass

import pytest

from hindsight_api.config import RECALL_BOOST_LEVELS, _parse_strategy_boosts
from hindsight_api.engine.search.recall_boost import (
    apply_post_rerank_boost,
    BOOST_LEVELS,
    additive_strategy_boost,
    boosted_rrf_score,
    stage2_passthrough,
    trim_merged_candidates,
)
from hindsight_api.engine.search.reranking import RerankResult
from hindsight_api.engine.search.types import MergedCandidate, RetrievalResult, ScoredResult


def _candidate(rrf_score: float, source_ranks: dict[str, int], *, id: str = "x") -> MergedCandidate:
    retrieval = RetrievalResult(id=id, text="t", fact_type="world")
    return MergedCandidate(retrieval=retrieval, rrf_score=rrf_score, source_ranks=source_ranks)


def _scored(weight: float, source_ranks: dict[str, int], *, suffix: str = "x"):
    from hindsight_api.engine.search.types import ScoredResult

    return ScoredResult(candidate=_candidate(0.0, source_ranks, id=suffix), weight=weight)


# --- level table integrity ----------------------------------------------------


def test_config_levels_match_boost_table():
    """The user-facing level names in config must match the weights table keys."""
    assert set(RECALL_BOOST_LEVELS) == set(BOOST_LEVELS)


def test_levels_are_monotonic():
    """Higher levels must boost more in both stages, or the names lie."""
    low, medium, high = (BOOST_LEVELS[lvl] for lvl in ("low", "medium", "high"))
    assert low.rank_divisor < medium.rank_divisor < high.rank_divisor
    assert low.additive < medium.additive < high.additive


# --- _parse_strategy_boosts ---------------------------------------------------


def test_parse_empty_is_noop():
    assert _parse_strategy_boosts("") == {}
    assert _parse_strategy_boosts(None) == {}
    assert _parse_strategy_boosts("   ") == {}


def test_parse_single_and_multiple():
    assert _parse_strategy_boosts("graph:high") == {"graph": "high"}
    assert _parse_strategy_boosts("graph:high,semantic:low") == {"graph": "high", "semantic": "low"}


def test_parse_is_case_insensitive_and_strips_whitespace():
    assert _parse_strategy_boosts(" GRAPH : HIGH , BM25:Low ") == {"graph": "high", "bm25": "low"}


def test_parse_skips_unknown_strategy():
    assert _parse_strategy_boosts("graphh:high,graph:low") == {"graph": "low"}


def test_parse_skips_unknown_level():
    # A raw number (the old format) is now an invalid level and skipped.
    assert _parse_strategy_boosts("graph:0.1,semantic:medium") == {"semantic": "medium"}
    assert _parse_strategy_boosts("graph:huge") == {}


def test_parse_bare_strategy_defaults_to_medium():
    # A strategy with no level (or a trailing colon) defaults to medium.
    assert _parse_strategy_boosts("graph") == {"graph": "medium"}
    assert _parse_strategy_boosts("graph:") == {"graph": "medium"}
    assert _parse_strategy_boosts("graph,semantic:high") == {"graph": "medium", "semantic": "high"}


def test_parse_skips_empty_name():
    assert _parse_strategy_boosts(":high") == {}


# --- boosted_rrf_score (pre-rerank, rank-aware) -------------------------------


def test_boosted_rrf_noop_when_no_boosts():
    cand = _candidate(0.5, {"graph_rank": 1})
    assert boosted_rrf_score(cand, {}) == 0.5


def test_boosted_rrf_promotes_the_arm_in_rank_space():
    """The boosted arm contributes as if it had placed rank/divisor."""
    cand = _candidate(0.5, {"graph_rank": 8})
    divisor = BOOST_LEVELS["high"].rank_divisor
    # rank 8 at divisor 8 contributes as rank 1, replacing its rank-8 contribution.
    expected = 0.5 + (1.0 / (60 + 8 / divisor)) - (1.0 / 68)
    assert boosted_rrf_score(cand, {"graph": "high"}, k=60) == pytest.approx(expected)


def test_boosted_rrf_higher_level_boosts_more():
    cand = _candidate(0.5, {"graph_rank": 5})
    low = boosted_rrf_score(cand, {"graph": "low"})
    high = boosted_rrf_score(cand, {"graph": "high"})
    assert high > low > 0.5


def test_boosted_rrf_is_rank_aware():
    """Boosting preserves the boosted arm's internal order.

    Note the boost *delta* is deliberately largest deep in the arm, where the
    reranker cut bites — a rank-1 candidate needs no rescuing. So this asserts
    the invariant that matters, final-score monotonicity, using base scores
    consistent with the ranks (as fusion produces them) rather than a flat stub.
    """
    top = _candidate(1.0 / 61, {"graph_rank": 1})
    deep = _candidate(1.0 / 260, {"graph_rank": 200})
    assert boosted_rrf_score(top, {"graph": "high"}) > boosted_rrf_score(deep, {"graph": "high"})


def test_boost_delta_is_largest_where_the_cut_bites():
    """The rescue is aimed at candidates near the cut, not at the arm's head."""
    top = _candidate(1.0 / 61, {"graph_rank": 1})
    deep = _candidate(1.0 / 260, {"graph_rank": 200})
    top_delta = boosted_rrf_score(top, {"graph": "high"}) - top.rrf_score
    deep_delta = boosted_rrf_score(deep, {"graph": "high"}) - deep.rrf_score
    assert deep_delta > top_delta


def test_boosted_rrf_ignores_non_matching_arm():
    # Candidate only came from semantic; a graph boost must not touch it.
    cand = _candidate(0.5, {"semantic_rank": 3})
    assert boosted_rrf_score(cand, {"graph": "high"}) == 0.5


# --- additive_strategy_boost (post-rerank, rank-decayed) ----------------------


def _bump(level: str, rank: int) -> float:
    weights = BOOST_LEVELS[level]
    return weights.additive * weights.rank_divisor / (weights.rank_divisor + rank - 1)


def test_additive_noop_when_no_boosts():
    assert additive_strategy_boost({"graph_rank": 1}, {}) == 0.0


def test_additive_rank_one_keeps_the_ceiling_and_deep_ranks_decay():
    """Rank 1 keeps the level's full additive. A deep rank does not."""
    assert additive_strategy_boost({"graph_rank": 1}, {"graph": "high"}) == pytest.approx(_bump("high", 1))
    assert _bump("high", 1) == pytest.approx(BOOST_LEVELS["high"].additive)
    deep = additive_strategy_boost({"graph_rank": 200}, {"graph": "high"})
    assert deep == pytest.approx(_bump("high", 200))
    assert deep < 0.05
    # high's decay scale is the reused divisor: half the ceiling at rank 9.
    assert additive_strategy_boost({"graph_rank": 9}, {"graph": "high"}) == pytest.approx(0.25)


def test_additive_decreases_with_rank():
    top = additive_strategy_boost({"graph_rank": 1}, {"graph": "high"})
    deep = additive_strategy_boost({"graph_rank": 200}, {"graph": "high"})
    assert top > deep > 0.0


@pytest.mark.parametrize("rank", [1, 9, 40])
def test_additive_is_monotonic_in_level_at_the_same_rank(rank):
    low = additive_strategy_boost({"graph_rank": rank}, {"graph": "low"})
    medium = additive_strategy_boost({"graph_rank": rank}, {"graph": "medium"})
    high = additive_strategy_boost({"graph_rank": rank}, {"graph": "high"})
    assert low < medium < high


def test_additive_sums_matched_arms():
    ranks = {"graph_rank": 2, "semantic_rank": 5}
    expected = _bump("high", 2) + _bump("low", 5)
    assert additive_strategy_boost(ranks, {"graph": "high", "semantic": "low"}) == pytest.approx(expected)


def test_additive_sum_has_no_combined_cap():
    """Two rank-1 highs add. A per-arm ceiling is not a cap on the total."""
    total = additive_strategy_boost({"graph_rank": 1, "semantic_rank": 1}, {"graph": "high", "semantic": "high"})
    assert total == pytest.approx(1.0)
    assert total > BOOST_LEVELS["high"].additive


def test_additive_ignores_unmatched_arm():
    assert additive_strategy_boost({"semantic_rank": 1}, {"graph": "high"}) == 0.0


def test_deep_rank_bump_still_beats_a_tiny_absolute_score():
    """Boundary, not a recall prediction.

    Twenty semantic rows at 0.01 against 300 zero-weight graph rows: after the
    high bump the top 20 are still all graph, because rank 300 still adds about
    0.013. Absolute adds do not fix cross-encoder calibration.
    """
    semantic = [_scored(0.01, {}, suffix=f"s{i}") for i in range(20)]
    graph = [_scored(0.0, {"graph_rank": rank}, suffix=f"g{rank}") for rank in range(1, 301)]
    rows = semantic + graph
    assert apply_post_rerank_boost(rows, {"graph": "high"}, passthrough=stage2_passthrough("cross_encoder", "tei")) == (
        "stage2=rank_decay"
    )
    rows.sort(key=lambda sr: sr.weight, reverse=True)
    assert all(sr.id.startswith("g") for sr in rows[:20])
    assert rows[19].candidate.source_ranks["graph_rank"] == 20


# --- #3956: the boost must not monopolise the reranker cap --------------------


def _cut(level: str | None, cap: int = 300, arm_depth: int = 1000, k: int = 60) -> list[tuple[str, int]]:
    """Merge a boosted arm and an unboosted arm, sort as recall does, take top ``cap``.

    Mirrors ``memory_engine`` step 4's pre-filter: build the merged pool, sort by
    ``boosted_rrf_score``, slice to the reranker candidate budget. Returns the
    surviving ``(arm, rank)`` pairs.
    """
    boosts = {"graph": level} if level else {}
    pool = [_candidate(1.0 / (k + r), {"graph_rank": r}) for r in range(1, arm_depth + 1)]
    pool += [_candidate(1.0 / (k + s), {"semantic_rank": s}) for s in range(1, arm_depth + 1)]
    pool.sort(key=lambda mc: boosted_rrf_score(mc, boosts, k=k), reverse=True)
    survivors = []
    for mc in pool[:cap]:
        arm, rank = next(iter(mc.source_ranks.items()))
        survivors.append((arm.removesuffix("_rank"), rank))
    return survivors


@pytest.mark.parametrize("level", ["low", "medium", "high"])
def test_boost_never_starves_the_other_arm_at_the_cap(level):
    """Regression for #3956: `graph:high` left zero semantic-only survivors.

    The score-space form multiplied the arm's contribution by a weight larger
    than RRF's whole dynamic range over the cap window, so the sort degenerated
    to "boosted arm first" and the cut kept 300/300 graph candidates.
    """
    semantic = [rank for arm, rank in _cut(level) if arm == "semantic"]
    assert semantic, f"{level} starved the unboosted arm out of the reranker budget"


@pytest.mark.parametrize("level", ["low", "medium", "high"])
def test_boost_never_displaces_the_head_of_the_other_arm(level):
    """No level may push the *top* unboosted hit out of the reranker budget.

    This is the property that makes the boost safe on banks whose merged pool is
    far larger than ``RERANKER_MAX_CANDIDATES``: displacement is proportional to
    rank, so the head of every arm is preserved whatever the pool size.
    """
    assert ("semantic", 1) in _cut(level)


@pytest.mark.parametrize("level", ["low", "medium", "high"])
def test_boost_still_protects_the_arm_from_the_cut(level):
    """The feature must still do its job: reach deeper into the boosted arm."""
    unboosted_depth = max(rank for arm, rank in _cut(None) if arm == "graph")
    boosted_depth = max(rank for arm, rank in _cut(level) if arm == "graph")
    assert boosted_depth > unboosted_depth


def test_higher_levels_protect_the_arm_more_deeply():
    depths = [max(rank for arm, rank in _cut(lvl) if arm == "graph") for lvl in ("low", "medium", "high")]
    assert depths[0] < depths[1] < depths[2]


def test_rank_boost_crossover_is_independent_of_k():
    """`r < divisor * s` must hold whatever RRF constant fusion was run with.

    The score-space form's crossover carried a `w*k` term, which is why the
    damage scaled with k and surprised on real banks; the rank-space form must
    not depend on k at all.
    """
    divisor = BOOST_LEVELS["medium"].rank_divisor
    for k in (10, 60, 200):
        boosted_wins = _candidate(1.0 / (k + 39), {"graph_rank": 39})
        boosted_loses = _candidate(1.0 / (k + 41), {"graph_rank": 41})
        rival = _candidate(1.0 / (k + 10), {"semantic_rank": 10})  # crossover at r = 4*10 = 40
        assert boosted_rrf_score(boosted_wins, {"graph": "medium"}, k=k) > boosted_rrf_score(rival, {}, k=k)
        assert boosted_rrf_score(boosted_loses, {"graph": "medium"}, k=k) < boosted_rrf_score(rival, {}, k=k)
        assert divisor == 4.0


# --- stage 2 call shape: passthrough, cap, min_scores -------------------------


def test_stage2_passthrough_modes_skip_the_bump():
    """Explicit rrf mode and an rrf provider both skip. A real provider does not."""
    row = _scored(0.2, {"graph_rank": 1})
    assert stage2_passthrough("rrf", "tei") is True
    assert stage2_passthrough("cross_encoder", "rrf") is True
    assert stage2_passthrough("cross_encoder", "tei") is False
    assert stage2_passthrough("cross_encoder", None) is False

    skipped = apply_post_rerank_boost([row], {"graph": "high"}, passthrough=stage2_passthrough("rrf", "tei"))
    assert skipped == "stage2=skipped_passthrough"
    assert row.weight == pytest.approx(0.2)

    configured = _scored(0.2, {"graph_rank": 1}, suffix="configured")
    token = apply_post_rerank_boost(
        [configured], {"graph": "high"}, passthrough=stage2_passthrough("cross_encoder", "rrf")
    )
    assert token == "stage2=skipped_passthrough"
    assert configured.weight == pytest.approx(0.2)

    live = _scored(0.2, {"graph_rank": 1}, suffix="live")
    assert apply_post_rerank_boost(
        [live], {"graph": "high"}, passthrough=stage2_passthrough("cross_encoder", "tei")
    ) == ("stage2=rank_decay")
    assert live.weight == pytest.approx(0.2 + BOOST_LEVELS["high"].additive)


def test_failover_to_rrf_is_passthrough():
    """A chain that has degraded reports the active member's name, and stage 2 trusts it."""
    from hindsight_api.engine.cross_encoder import CrossEncoderModel, MultiCrossEncoder

    class _Named(CrossEncoderModel):
        def __init__(self, name: str) -> None:
            self._name = name

        @property
        def provider_name(self) -> str:
            return self._name

        async def initialize(self) -> None:
            return None

    chain = MultiCrossEncoder([_Named("tei"), _Named("rrf")])
    assert chain.provider_name == "tei"
    chain._active = 1
    assert chain.provider_name == "rrf"
    row = _scored(0.2, {"graph_rank": 1}, suffix="failed-over")
    token = apply_post_rerank_boost(
        [row], {"graph": "high"}, passthrough=stage2_passthrough("cross_encoder", chain.provider_name)
    )
    assert token == "stage2=skipped_passthrough"
    assert row.weight == pytest.approx(0.2)


def test_rank_decay_can_fall_below_min_final():
    """Recall's final floor compares weight after stage 2. A smaller bump can drop a row.

    The flat +0.5 would have cleared this floor. The filter itself is
    ``sr.weight >= min_final`` in ``MemoryEngine.recall_async``.
    """
    sr = _scored(0.04, {"graph_rank": 200})
    apply_post_rerank_boost([sr], {"graph": "high"}, passthrough=stage2_passthrough("cross_encoder", "tei"))
    min_final = 0.5
    assert 0.04 + BOOST_LEVELS["high"].additive >= min_final
    assert sr.weight < min_final
    assert not (sr.weight >= min_final)


@dataclass
class _PassthroughFinish:
    scored: list[ScoredResult]
    token: str | None


def _passthrough_finish(pool: list[MergedCandidate], cap: int, boosts: dict[str, str]) -> _PassthroughFinish:
    """The rrf-mode tail of recall: trim, order by raw rrf, combined scoring, stage 2."""
    from datetime import UTC, datetime

    from hindsight_api.engine.search.reranking import apply_combined_scoring
    from hindsight_api.engine.search.types import ScoredResult

    trimmed = trim_merged_candidates(list(pool), cap, boosts)
    ordered = sorted(trimmed.kept, key=lambda mc: mc.rrf_score, reverse=True)
    scored = [ScoredResult(candidate=mc, weight=0.0) for mc in ordered]
    apply_combined_scoring(scored, now=datetime(2026, 1, 1, tzinfo=UTC), is_passthrough_reranker=True)
    token = apply_post_rerank_boost(scored, boosts, passthrough=stage2_passthrough("rrf", "local"))
    scored.sort(key=lambda sr: sr.weight, reverse=True)
    return _PassthroughFinish(scored=scored, token=token)


def _pool() -> list[MergedCandidate]:
    # Raw RRF is 1/(60+rank). Graph rank 3's boosted score beats semantic rank 1;
    # its raw score does not.
    return [
        _candidate(1.0 / 61, {"semantic_rank": 1}, id="sem1"),
        _candidate(1.0 / 62, {"semantic_rank": 2}, id="sem2"),
        _candidate(1.0 / 63, {"graph_rank": 3}, id="g3"),
        _candidate(1.0 / 100, {"graph_rank": 40}, id="g40"),
    ]


def test_passthrough_under_cap_matches_no_boost():
    """Pool within the cap: stage 1 does not run, stage 2 is skipped, so the boost is a no-op."""
    boosted = _passthrough_finish(_pool(), cap=10, boosts={"graph": "high"})
    plain = _passthrough_finish(_pool(), cap=10, boosts={})
    assert boosted.token == "stage2=skipped_passthrough"
    assert plain.token is None
    assert [sr.id for sr in boosted.scored] == [sr.id for sr in plain.scored]
    assert [sr.weight for sr in boosted.scored] == pytest.approx([sr.weight for sr in plain.scored])


def test_passthrough_over_cap_changes_membership_not_rrf_order():
    """Over the cap the boost only chooses who enters. Order stays raw RRF, with no stage-2 add."""
    boosted = _passthrough_finish(_pool(), cap=2, boosts={"graph": "high"})
    plain = _passthrough_finish(_pool(), cap=2, boosts={})
    assert boosted.token == "stage2=skipped_passthrough"
    assert {sr.id for sr in boosted.scored} == {"g3", "sem1"}
    assert {sr.id for sr in plain.scored} == {"sem1", "sem2"}
    # Raw RRF puts sem1 ahead of g3. The boosted rank-space score would not.
    assert [sr.id for sr in boosted.scored] == ["sem1", "g3"]
    graph = next(sr for sr in boosted.scored if sr.id == "g3")
    assert graph.weight == pytest.approx(graph.combined_score)
    assert graph.weight == pytest.approx(0.1)
    assert graph.weight + BOOST_LEVELS["high"].additive != pytest.approx(graph.weight)


def test_empty_boosts_leave_stage2_unlogged():
    row = _scored(0.4, {"graph_rank": 1})
    assert apply_post_rerank_boost([row], {}, passthrough=stage2_passthrough("cross_encoder", "tei")) is None
    assert row.weight == pytest.approx(0.4)


def test_recall_async_wires_stage2_and_keeps_interleave(monkeypatch):
    """recall_async really does call the stage-2 helper with the passthrough decision.

    A cross-encoder recall of one graph hit gains the rank-1 bump. The same hit
    in rrf mode and in interleave mode does not: those paths skip the add.
    """
    import asyncio
    from unittest.mock import AsyncMock

    from hindsight_api.engine import memory_engine
    from hindsight_api.engine.memory_engine import Budget
    from hindsight_api.engine.search.retrieval import MultiFactTypeRetrievalResult, ParallelRetrievalResult
    from hindsight_api.engine.search.types import RetrievalResult, ScoredResult
    from hindsight_api.models import RequestContext

    fact_id = "00000000-0000-0000-0000-00000000000a"

    class _ConfigResolver:
        async def get_bank_config(self, _bank_id: str, _request_context: RequestContext) -> dict[str, object]:
            return {}

    class _Encoder:
        provider_name = "tei"

    class _Reranker:
        def __init__(self) -> None:
            self.cross_encoder = _Encoder()

        async def ensure_initialized(self) -> None:
            return None

        async def rerank(self, _query: str, candidates: list[MergedCandidate]) -> RerankResult:
            return RerankResult(
                results=[
                    ScoredResult(
                        candidate=candidate,
                        cross_encoder_score=0.2,
                        cross_encoder_score_normalized=0.2,
                        weight=0.2,
                    )
                    for candidate in candidates
                ],
                provider_name=self.cross_encoder.provider_name,
            )

    class _Config:
        def __init__(self, base: object, boosts: dict[str, str]) -> None:
            self._base = base
            self._boosts = boosts

        def __getattr__(self, name: str):
            if name == "recall_strategy_boosts":
                return self._boosts
            return getattr(self._base, name)

    async def _recall(monkeypatch, boosts: dict[str, str], reranking: str) -> float:
        from hindsight_api.config import get_config as real_get_config

        engine = memory_engine.MemoryEngine.__new__(memory_engine.MemoryEngine)
        engine._operation_validator = None
        engine._config_resolver = _ConfigResolver()
        engine._search_semaphore = asyncio.Semaphore(1)
        engine._initialized = True
        engine._read_backend = object()
        engine.embeddings = object()
        engine.query_analyzer = object()
        engine._cross_encoder_reranker = _Reranker()
        engine._authenticate_tenant = AsyncMock()
        engine._require_bank_exists = AsyncMock()

        async def generate_embeddings_batch(*_args: object, **_kwargs: object) -> list[list[float]]:
            return [[0.1, 0.2, 0.3]]

        async def retrieve_all_fact_types_parallel(*_args: object, **_kwargs: object) -> MultiFactTypeRetrievalResult:
            retrieval = RetrievalResult(id=fact_id, text="graph hit", fact_type="world")
            return MultiFactTypeRetrievalResult(
                results_by_fact_type={
                    "world": ParallelRetrievalResult(
                        semantic=[],
                        bm25=[],
                        graph=[retrieval],
                        temporal=None,
                        timings={"semantic": 0.0, "bm25": 0.0, "graph": 0.0, "temporal_extraction": 0.0},
                    )
                }
            )

        monkeypatch.setattr(memory_engine.embedding_utils, "generate_embeddings_batch", generate_embeddings_batch)
        monkeypatch.setattr(
            "hindsight_api.engine.search.retrieval.retrieve_all_fact_types_parallel",
            retrieve_all_fact_types_parallel,
        )
        monkeypatch.setattr(memory_engine, "get_config", lambda: _Config(real_get_config(), boosts))

        result = await engine.recall_async(
            bank_id="test-bank",
            query="graph hit",
            budget=Budget.LOW,
            fact_type=["world"],
            request_context=RequestContext(),
            reranking=reranking,  # type: ignore[arg-type]
            _quiet=True,
        )
        assert len(result.results) == 1
        assert result.results[0].scores is not None
        return result.results[0].scores.final

    plain = asyncio.run(_recall(monkeypatch, {}, "cross_encoder"))
    boosted = asyncio.run(_recall(monkeypatch, {"graph": "high"}, "cross_encoder"))
    passthrough = asyncio.run(_recall(monkeypatch, {"graph": "high"}, "rrf"))
    passthrough_plain = asyncio.run(_recall(monkeypatch, {}, "rrf"))
    interleaved = asyncio.run(_recall(monkeypatch, {"graph": "high"}, "interleave"))
    interleaved_plain = asyncio.run(_recall(monkeypatch, {}, "interleave"))

    assert boosted == pytest.approx(plain + BOOST_LEVELS["high"].additive)
    assert passthrough == pytest.approx(passthrough_plain)
    assert interleaved == pytest.approx(interleaved_plain)
