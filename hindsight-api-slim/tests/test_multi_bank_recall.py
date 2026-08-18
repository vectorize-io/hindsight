"""Multi-bank recall: pure merge helpers + MemoryEngine.recall_multi_async orchestrator.

Covers the 2026-08-10 multi-bank plan:
- score-merge order (incl. ties)
- interleave order
- auto-fallback to interleave when CE is not comparable
- token cut on the merged list
- bank_id attribution
- partial-failure metadata
- single-bank equivalence with recall_async
- empty bank member
- ContextVar isolation across parallel sub-calls (via @_bind_bank_id on each task)

No DB / embeddings required — sub-calls are mocked; pure helpers are unit-tested directly.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hindsight_api.cancellation import OperationCancelledError
from hindsight_api.engine.memory_engine import MemoryEngine, _bind_bank_id, get_current_bank_id
from hindsight_api.engine.multi_bank_recall import (
    DEFAULT_PER_BANK_MERGE_CAP,
    FALLBACK_NO_USABLE_RERANKER_SCORES,
    MAX_MULTI_BANK_RECALL_BANKS,
    META_BANKS,
    META_DEDUP,
    META_DEDUP_DROPPED,
    META_DEDUP_V1,
    META_MERGE_APPLIED,
    META_MERGE_FALLBACK_REASON,
    META_MERGE_REQUESTED,
    META_MULTI_BANK,
    META_PER_BANK_CAP,
    MULTI_BANK_PREFER_OBSERVATIONS,
    bank_rank_from_merged,
    build_multi_bank_metadata,
    cap_per_bank_results,
    cross_encoder_eligible,
    cut_to_token_budget,
    dedup_exact_normalized,
    has_usable_reranker_scores,
    interleave_merge,
    merge_cap_dedup_cut,
    normalize_dedup_key,
    score_merge,
    stamp_bank_id,
    union_merge_dicts,
)
from hindsight_api.engine.response_models import (
    ChunkInfo,
    EntityState,
    MemoryFact,
    RecallResult,
    RecallScores,
)
from hindsight_api.extensions.operation_validator import OperationValidationError
from hindsight_api.extensions.tenant import AuthenticationError
from hindsight_api.models import RequestContext

RC = RequestContext(tenant_id="default")


def _fact(
    id: str,
    text: str,
    *,
    reranker: float | None = None,
    final: float | None = None,
    bank_id: str | None = None,
) -> MemoryFact:
    scores = None
    if reranker is not None or final is not None:
        scores = RecallScores(
            final=final if final is not None else (reranker or 0.0),
            reranker=reranker,
        )
    return MemoryFact(id=id, text=text, fact_type="world", scores=scores, bank_id=bank_id)


# --- pure helpers -------------------------------------------------------------


def test_score_merge_orders_by_reranker_descending():
    bank_a = [
        _fact("a1", "low from A", reranker=0.2),
        _fact("a2", "high from A", reranker=0.9),
    ]
    bank_b = [
        _fact("b1", "mid from B", reranker=0.5),
        _fact("b2", "higher from B", reranker=0.8),
    ]
    merged = score_merge([("bank-a", bank_a), ("bank-b", bank_b)])
    assert [f.id for f in merged] == ["a2", "b2", "b1", "a1"]
    assert [f.bank_id for f in merged] == ["bank-a", "bank-b", "bank-b", "bank-a"]


def test_score_merge_ties_break_by_bank_order_then_rank():
    """Equal reranker scores: earlier bank, then earlier within-bank rank wins."""
    bank_a = [
        _fact("a1", "A first", reranker=0.7),
        _fact("a2", "A second", reranker=0.7),
    ]
    bank_b = [
        _fact("b1", "B first", reranker=0.7),
    ]
    merged = score_merge([("bank-a", bank_a), ("bank-b", bank_b)])
    assert [f.id for f in merged] == ["a1", "a2", "b1"]


def test_score_merge_missing_reranker_sorts_last():
    bank_a = [_fact("a1", "no ce", reranker=None, final=0.9)]
    bank_b = [_fact("b1", "has ce", reranker=0.1)]
    merged = score_merge([("bank-a", bank_a), ("bank-b", bank_b)])
    assert [f.id for f in merged] == ["b1", "a1"]


def test_interleave_merge_round_robin_by_rank():
    bank_a = [
        _fact("a1", "A1"),
        _fact("a2", "A2"),
        _fact("a3", "A3"),
    ]
    bank_b = [
        _fact("b1", "B1"),
        _fact("b2", "B2"),
    ]
    merged = interleave_merge([("bank-a", bank_a), ("bank-b", bank_b)])
    assert [f.id for f in merged] == ["a1", "b1", "a2", "b2", "a3"]
    assert all(f.bank_id in ("bank-a", "bank-b") for f in merged)


def test_interleave_merge_empty_bank_member():
    bank_a = [_fact("a1", "only A")]
    bank_b: list[MemoryFact] = []
    merged = interleave_merge([("bank-a", bank_a), ("bank-b", bank_b)])
    assert [f.id for f in merged] == ["a1"]
    assert merged[0].bank_id == "bank-a"


def test_stamp_bank_id_does_not_mutate_original():
    original = _fact("x", "text")
    stamped = stamp_bank_id(original, "bank-z")
    assert stamped.bank_id == "bank-z"
    assert original.bank_id is None


def test_cut_to_token_budget_stops_before_exceeding():
    from hindsight_api.engine.memory_engine import count_tokens

    f1 = _fact("1", "alpha")
    f2 = _fact("2", "beta gamma")
    f3 = _fact("3", "delta")
    budget = count_tokens(f1.text) + count_tokens(f2.text)
    cut = cut_to_token_budget([f1, f2, f3], budget)
    assert [f.id for f in cut] == ["1", "2"]
    assert sum(count_tokens(f.text) for f in cut) <= budget

    # Budget too small for the first fact → empty (stop-before-exceeding).
    under_first = max(0, count_tokens(f1.text) - 1)
    assert cut_to_token_budget([f1, f2], under_first) == []


def test_cut_to_token_budget_zero_is_empty():
    assert cut_to_token_budget([_fact("1", "hello")], 0) == []


def test_cross_encoder_eligible_requires_cross_encoder_request():
    ok, reason = cross_encoder_eligible(
        requested_reranking="rrf",
        bank_enable_reranking=[True, True],
    )
    assert ok is False
    assert reason is not None
    assert "rrf" in reason


def test_cross_encoder_eligible_rejects_disabled_reranking_bank():
    ok, reason = cross_encoder_eligible(
        requested_reranking="cross_encoder",
        bank_enable_reranking=[True, False],
    )
    assert ok is False
    assert reason is not None
    assert "enable_reranking" in reason


def test_cross_encoder_eligible_all_ce():
    ok, reason = cross_encoder_eligible(
        requested_reranking="cross_encoder",
        bank_enable_reranking=[True, True],
    )
    assert ok is True
    assert reason is None


def test_build_multi_bank_metadata_shape():
    meta = build_multi_bank_metadata(
        merge_requested="score",
        merge_applied="interleave",
        merge_fallback_reason="test reason",
        bank_statuses={"a": {"status": "ok", "count": 1}},
    )
    block = meta[META_MULTI_BANK]
    assert block[META_MERGE_REQUESTED] == "score"
    assert block[META_MERGE_APPLIED] == "interleave"
    assert block[META_MERGE_FALLBACK_REASON] == "test reason"
    assert block[META_BANKS]["a"]["status"] == "ok"
    assert block[META_DEDUP] == META_DEDUP_V1
    assert block[META_DEDUP] == "exact_normalized"
    assert block[META_DEDUP_DROPPED] == 0
    assert block[META_PER_BANK_CAP] == DEFAULT_PER_BANK_MERGE_CAP


def test_union_merge_dicts_union_and_collision_by_rank():
    """On collision keep the bank with the better (lower) rank from the merged order."""
    merged = [
        stamp_bank_id(_fact("a1", "top from A", reranker=0.9), "bank-a"),
        stamp_bank_id(_fact("b1", "from B", reranker=0.5), "bank-b"),
    ]
    ranks = bank_rank_from_merged(merged)
    assert ranks == {"bank-a": 0, "bank-b": 1}

    # Distinct keys union
    entities = union_merge_dicts(
        [
            ("bank-a", {"Alice": EntityState(entity_id="e-a", canonical_name="Alice")}),
            ("bank-b", {"Bob": EntityState(entity_id="e-b", canonical_name="Bob")}),
        ],
        bank_rank=ranks,
    )
    assert set(entities) == {"Alice", "Bob"}

    # Collision: same key — bank-a ranks higher so its value wins
    chunks = union_merge_dicts(
        [
            ("bank-a", {"shared": ChunkInfo(chunk_text="from A", chunk_index=0)}),
            ("bank-b", {"shared": ChunkInfo(chunk_text="from B", chunk_index=0)}),
        ],
        bank_rank=ranks,
    )
    assert chunks is not None
    assert chunks["shared"].chunk_text == "from A"

    # None / empty → None
    assert union_merge_dicts([("bank-a", None), ("bank-b", {})], bank_rank=ranks) is None


# --- orchestrator (mocked recall_async) ---------------------------------------


def _harness(
    *,
    bank_results: dict[str, list[MemoryFact] | Exception | RecallResult],
    enable_reranking: dict[str, bool] | None = None,
) -> MemoryEngine:
    """Minimal MemoryEngine shell: real recall_multi_async, mocked sub-calls + config."""
    engine = object.__new__(MemoryEngine)

    async def fake_recall(bank_id: str, query: str, **kwargs) -> RecallResult:
        outcome = bank_results[bank_id]
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, RecallResult):
            return outcome
        return RecallResult(results=list(outcome))

    engine.recall_async = fake_recall  # type: ignore[method-assign]

    async def fake_auth(request_context: RequestContext) -> str:
        return "public"

    engine._authenticate_tenant = fake_auth  # type: ignore[method-assign]

    enable_reranking = enable_reranking or {
        bid: True for bid in bank_results if not isinstance(bank_results[bid], Exception)
    }
    # Failed banks may still need config flags
    for bid in bank_results:
        enable_reranking.setdefault(bid, True)

    async def fake_config(bank_id: str, request_context):
        return {"enable_reranking": enable_reranking.get(bank_id, True)}

    engine._config_resolver = SimpleNamespace(get_bank_config=fake_config)  # type: ignore[attr-defined]
    return engine


@pytest.mark.asyncio
async def test_orchestrator_score_merge_order():
    engine = _harness(
        bank_results={
            "bank-a": [
                _fact("a1", "low", reranker=0.2),
                _fact("a2", "high", reranker=0.95),
            ],
            "bank-b": [
                _fact("b1", "mid", reranker=0.6),
            ],
        }
    )
    result = await MemoryEngine.recall_multi_async(
        engine,
        ["bank-a", "bank-b"],
        "query",
        merge="score",
        request_context=RC,
        max_tokens=10_000,
    )
    assert [f.id for f in result.results] == ["a2", "b1", "a1"]
    assert result.metadata[META_MULTI_BANK][META_MERGE_APPLIED] == "score"
    assert result.metadata[META_MULTI_BANK][META_MERGE_FALLBACK_REASON] is None


@pytest.mark.asyncio
async def test_orchestrator_interleave_order():
    engine = _harness(
        bank_results={
            "bank-a": [_fact("a1", "A1"), _fact("a2", "A2")],
            "bank-b": [_fact("b1", "B1"), _fact("b2", "B2")],
        }
    )
    result = await MemoryEngine.recall_multi_async(
        engine,
        ["bank-a", "bank-b"],
        "query",
        merge="interleave",
        request_context=RC,
        max_tokens=10_000,
    )
    assert [f.id for f in result.results] == ["a1", "b1", "a2", "b2"]
    assert result.metadata[META_MULTI_BANK][META_MERGE_APPLIED] == "interleave"


@pytest.mark.asyncio
async def test_orchestrator_auto_fallback_when_bank_disables_reranking():
    engine = _harness(
        bank_results={
            "bank-a": [_fact("a1", "A1", reranker=0.9), _fact("a2", "A2", reranker=0.1)],
            "bank-b": [_fact("b1", "B1", reranker=None, final=0.5)],
        },
        enable_reranking={"bank-a": True, "bank-b": False},
    )
    result = await MemoryEngine.recall_multi_async(
        engine,
        ["bank-a", "bank-b"],
        "query",
        merge="score",
        request_context=RC,
        max_tokens=10_000,
    )
    mb = result.metadata[META_MULTI_BANK]
    assert mb[META_MERGE_REQUESTED] == "score"
    assert mb[META_MERGE_APPLIED] == "interleave"
    assert mb[META_MERGE_FALLBACK_REASON] is not None
    # Interleave order, not score order (score would put a1 first and maybe a2 before b1).
    assert [f.id for f in result.results] == ["a1", "b1", "a2"]


@pytest.mark.asyncio
async def test_orchestrator_auto_fallback_when_requested_rrf():
    engine = _harness(
        bank_results={
            "bank-a": [_fact("a1", "A1")],
            "bank-b": [_fact("b1", "B1")],
        }
    )
    result = await MemoryEngine.recall_multi_async(
        engine,
        ["bank-a", "bank-b"],
        "query",
        merge="score",
        reranking="rrf",
        request_context=RC,
        max_tokens=10_000,
    )
    mb = result.metadata[META_MULTI_BANK]
    assert mb[META_MERGE_APPLIED] == "interleave"
    assert "rrf" in (mb[META_MERGE_FALLBACK_REASON] or "")


@pytest.mark.asyncio
async def test_orchestrator_token_cut_on_merged_list():
    from hindsight_api.engine.memory_engine import count_tokens

    # Distinct short texts so ordering is stable and budget math is simple.
    a1 = _fact("a1", "alpha alpha", reranker=0.9)
    b1 = _fact("b1", "beta beta beta", reranker=0.8)
    a2 = _fact("a2", "gamma", reranker=0.7)
    engine = _harness(bank_results={"bank-a": [a1, a2], "bank-b": [b1]})

    # Fit a1 + b1 but not a2 after score-merge order a1, b1, a2.
    budget = count_tokens(a1.text) + count_tokens(b1.text)
    result = await MemoryEngine.recall_multi_async(
        engine,
        ["bank-a", "bank-b"],
        "query",
        merge="score",
        request_context=RC,
        max_tokens=budget,
    )
    assert [f.id for f in result.results] == ["a1", "b1"]
    assert sum(count_tokens(f.text) for f in result.results) <= budget


@pytest.mark.asyncio
async def test_orchestrator_bank_id_attribution():
    engine = _harness(
        bank_results={
            "bank-a": [_fact("a1", "from A", reranker=0.5)],
            "bank-b": [_fact("b1", "from B", reranker=0.6)],
        }
    )
    result = await MemoryEngine.recall_multi_async(
        engine,
        ["bank-a", "bank-b"],
        "query",
        request_context=RC,
        max_tokens=10_000,
    )
    by_id = {f.id: f.bank_id for f in result.results}
    assert by_id == {"b1": "bank-b", "a1": "bank-a"}


@pytest.mark.asyncio
async def test_orchestrator_partial_failure_metadata():
    engine = _harness(
        bank_results={
            "bank-a": [_fact("a1", "ok", reranker=0.5)],
            "bank-b": RuntimeError("boom"),
        }
    )
    result = await MemoryEngine.recall_multi_async(
        engine,
        ["bank-a", "bank-b"],
        "query",
        request_context=RC,
        max_tokens=10_000,
    )
    banks = result.metadata[META_MULTI_BANK][META_BANKS]
    assert banks["bank-a"]["status"] == "ok"
    assert banks["bank-a"]["count"] == 1
    assert banks["bank-b"]["status"] == "error"
    # Client-visible text is generic — no exception class or message oracle.
    assert banks["bank-b"]["error"] == "recall failed for this bank"
    assert "RuntimeError" not in banks["bank-b"]["error"]
    assert "boom" not in banks["bank-b"]["error"]
    assert [f.id for f in result.results] == ["a1"]


@pytest.mark.asyncio
async def test_orchestrator_empty_bank_member():
    engine = _harness(
        bank_results={
            "bank-a": [_fact("a1", "only", reranker=0.5)],
            "bank-empty": [],
        }
    )
    result = await MemoryEngine.recall_multi_async(
        engine,
        ["bank-a", "bank-empty"],
        "query",
        request_context=RC,
        max_tokens=10_000,
    )
    banks = result.metadata[META_MULTI_BANK][META_BANKS]
    assert banks["bank-empty"] == {"status": "ok", "count": 0}
    assert [f.id for f in result.results] == ["a1"]
    assert result.results[0].bank_id == "bank-a"


@pytest.mark.asyncio
async def test_orchestrator_single_bank_equivalence():
    """Single-bank multi-recall matches recall_async content order (plus bank_id stamp)."""
    facts = [
        _fact("s1", "first", reranker=0.9),
        _fact("s2", "second", reranker=0.5),
    ]
    engine = _harness(bank_results={"only": facts})

    single = await engine.recall_async("only", "query", request_context=RC, max_tokens=10_000)
    multi = await MemoryEngine.recall_multi_async(
        engine,
        ["only"],
        "query",
        merge="score",
        request_context=RC,
        max_tokens=10_000,
    )
    assert [f.id for f in multi.results] == [f.id for f in single.results]
    assert [f.text for f in multi.results] == [f.text for f in single.results]
    assert all(f.bank_id == "only" for f in multi.results)


@pytest.mark.asyncio
async def test_orchestrator_empty_bank_ids_list():
    engine = _harness(bank_results={})
    result = await MemoryEngine.recall_multi_async(
        engine,
        [],
        "query",
        request_context=RC,
    )
    assert result.results == []
    assert result.metadata[META_MULTI_BANK][META_BANKS] == {}


@pytest.mark.asyncio
async def test_orchestrator_contextvar_isolation_across_parallel_subcalls():
    """Each parallel sub-call sees its own bank_id via @_bind_bank_id (task context)."""
    engine = object.__new__(MemoryEngine)
    observed: dict[str, str | None] = {}
    barrier = asyncio.Barrier(2)

    @_bind_bank_id()
    async def bound_recall(bank_id: str, query: str, **kwargs) -> RecallResult:
        # Wait so both tasks are concurrent before reading ContextVar.
        await barrier.wait()
        observed[bank_id] = get_current_bank_id()
        return RecallResult(results=[_fact(f"{bank_id}-1", bank_id, reranker=0.5)])

    engine.recall_async = bound_recall  # type: ignore[method-assign]

    async def fake_auth(request_context: RequestContext) -> str:
        return "public"

    engine._authenticate_tenant = fake_auth  # type: ignore[method-assign]
    engine._config_resolver = SimpleNamespace(  # type: ignore[attr-defined]
        get_bank_config=AsyncMock(return_value={"enable_reranking": True})
    )

    result = await MemoryEngine.recall_multi_async(
        engine,
        ["bank-x", "bank-y"],
        "query",
        request_context=RC,
        max_tokens=10_000,
    )
    assert observed["bank-x"] == "bank-x"
    assert observed["bank-y"] == "bank-y"
    assert {f.bank_id for f in result.results} == {"bank-x", "bank-y"}


@pytest.mark.asyncio
async def test_orchestrator_invalid_merge_raises():
    engine = _harness(bank_results={"a": []})
    with pytest.raises(ValueError, match="merge must be"):
        await MemoryEngine.recall_multi_async(
            engine,
            ["a"],
            "query",
            merge="nope",  # type: ignore[arg-type]
            request_context=RC,
        )


@pytest.mark.asyncio
async def test_orchestrator_passes_full_max_tokens_to_each_subcall():
    """Each sub-call receives the caller's full max_tokens (cut happens after merge)."""
    seen_max: dict[str, int] = {}

    engine = object.__new__(MemoryEngine)

    async def tracking_recall(bank_id: str, query: str, **kwargs) -> RecallResult:
        seen_max[bank_id] = kwargs.get("max_tokens", -1)
        return RecallResult(results=[_fact(f"{bank_id}-1", "x" * 50, reranker=0.5)])

    engine.recall_async = tracking_recall  # type: ignore[method-assign]

    async def fake_auth(request_context: RequestContext) -> str:
        return "public"

    engine._authenticate_tenant = fake_auth  # type: ignore[method-assign]
    engine._config_resolver = SimpleNamespace(  # type: ignore[attr-defined]
        get_bank_config=AsyncMock(return_value={"enable_reranking": True})
    )

    await MemoryEngine.recall_multi_async(
        engine,
        ["b1", "b2"],
        "query",
        request_context=RC,
        max_tokens=1234,
    )
    assert seen_max == {"b1": 1234, "b2": 1234}


# --- B0: include_* side-dict merge --------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_merges_entities_chunks_source_facts():
    """include_* payloads from successful banks are union-merged into the response."""
    engine = _harness(
        bank_results={
            "bank-a": RecallResult(
                results=[_fact("a1", "A top", reranker=0.9)],
                entities={"Alice": EntityState(entity_id="ea", canonical_name="Alice")},
                chunks={"a_doc_0": ChunkInfo(chunk_text="chunk A", chunk_index=0)},
                source_facts={"sf-a": _fact("sf-a", "source from A")},
            ),
            "bank-b": RecallResult(
                results=[_fact("b1", "B mid", reranker=0.5)],
                entities={"Bob": EntityState(entity_id="eb", canonical_name="Bob")},
                chunks={"b_doc_0": ChunkInfo(chunk_text="chunk B", chunk_index=0)},
                source_facts={"sf-b": _fact("sf-b", "source from B")},
            ),
        }
    )
    result = await MemoryEngine.recall_multi_async(
        engine,
        ["bank-a", "bank-b"],
        "query",
        merge="score",
        include_entities=True,
        include_chunks=True,
        include_source_facts=True,
        request_context=RC,
        max_tokens=10_000,
    )
    assert result.entities is not None and set(result.entities) == {"Alice", "Bob"}
    assert result.chunks is not None and set(result.chunks) == {"a_doc_0", "b_doc_0"}
    assert result.source_facts is not None and set(result.source_facts) == {"sf-a", "sf-b"}


@pytest.mark.asyncio
async def test_orchestrator_side_dict_collision_prefers_higher_ranked_bank():
    """On key collision, keep the bank whose results ranked higher in the merged list."""
    # bank-b has the higher CE score → ranks first after score-merge → wins collisions.
    engine = _harness(
        bank_results={
            "bank-a": RecallResult(
                results=[_fact("a1", "low", reranker=0.2)],
                entities={"Shared": EntityState(entity_id="from-a", canonical_name="Shared")},
                chunks={"shared_key": ChunkInfo(chunk_text="from A", chunk_index=0)},
                source_facts={"sf-shared": _fact("sf-shared", "from A")},
            ),
            "bank-b": RecallResult(
                results=[_fact("b1", "high", reranker=0.95)],
                entities={"Shared": EntityState(entity_id="from-b", canonical_name="Shared")},
                chunks={"shared_key": ChunkInfo(chunk_text="from B", chunk_index=0)},
                source_facts={"sf-shared": _fact("sf-shared", "from B")},
            ),
        }
    )
    result = await MemoryEngine.recall_multi_async(
        engine,
        ["bank-a", "bank-b"],
        "query",
        merge="score",
        request_context=RC,
        max_tokens=10_000,
    )
    assert [f.id for f in result.results] == ["b1", "a1"]
    assert result.entities is not None
    assert result.entities["Shared"].entity_id == "from-b"
    assert result.chunks is not None
    assert result.chunks["shared_key"].chunk_text == "from B"
    assert result.source_facts is not None
    assert result.source_facts["sf-shared"].text == "from B"


@pytest.mark.asyncio
async def test_orchestrator_include_none_when_subcalls_omit_side_dicts():
    engine = _harness(
        bank_results={
            "bank-a": [_fact("a1", "only", reranker=0.5)],
            "bank-b": [_fact("b1", "only", reranker=0.4)],
        }
    )
    result = await MemoryEngine.recall_multi_async(
        engine,
        ["bank-a", "bank-b"],
        "query",
        request_context=RC,
        max_tokens=10_000,
    )
    assert result.entities is None
    assert result.chunks is None
    assert result.source_facts is None


# --- Job C audit fixes --------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_cancellation_propagates_not_soft_fail():
    """OperationCancelledError from any sub-call must re-raise, not enter bank metadata."""
    engine = _harness(
        bank_results={
            "bank-a": [_fact("a1", "ok", reranker=0.5)],
            "bank-b": OperationCancelledError("client disconnected"),
        }
    )
    with pytest.raises(OperationCancelledError, match="client disconnected"):
        await MemoryEngine.recall_multi_async(
            engine,
            ["bank-a", "bank-b"],
            "query",
            request_context=RC,
            max_tokens=10_000,
        )


def test_has_usable_reranker_scores_empty_is_ok():
    assert has_usable_reranker_scores([("a", []), ("b", [])]) is True


def test_has_usable_reranker_scores_all_none_is_false():
    facts = [
        ("a", [_fact("a1", "x", reranker=None, final=0.9)]),
        ("b", [_fact("b1", "y", reranker=None, final=0.8)]),
    ]
    assert has_usable_reranker_scores(facts) is False


def test_has_usable_reranker_scores_any_usable_is_true():
    facts = [
        ("a", [_fact("a1", "x", reranker=None, final=0.9)]),
        ("b", [_fact("b1", "y", reranker=0.3)]),
    ]
    assert has_usable_reranker_scores(facts) is True


@pytest.mark.asyncio
async def test_orchestrator_null_reranker_falls_back_to_interleave_order():
    """All scores.reranker None + merge=score → interleave applied, real interleave order.

    Without the post-gather check, score-merge would stable-sort all -inf and
    concatenate banks (a1,a2,b1,b2) while claiming merge_applied='score'.
    """
    engine = _harness(
        bank_results={
            "bank-a": [
                _fact("a1", "A1", reranker=None, final=0.9),
                _fact("a2", "A2", reranker=None, final=0.8),
            ],
            "bank-b": [
                _fact("b1", "B1", reranker=None, final=0.7),
                _fact("b2", "B2", reranker=None, final=0.6),
            ],
        }
    )
    result = await MemoryEngine.recall_multi_async(
        engine,
        ["bank-a", "bank-b"],
        "query",
        merge="score",
        request_context=RC,
        max_tokens=10_000,
    )
    mb = result.metadata[META_MULTI_BANK]
    assert mb[META_MERGE_REQUESTED] == "score"
    assert mb[META_MERGE_APPLIED] == "interleave"
    assert mb[META_MERGE_FALLBACK_REASON] == FALLBACK_NO_USABLE_RERANKER_SCORES
    # Genuine interleave order — not bank concatenation (a1,a2,b1,b2).
    assert [f.id for f in result.results] == ["a1", "b1", "a2", "b2"]


@pytest.mark.asyncio
async def test_orchestrator_rejects_over_cap_bank_ids():
    too_many = [f"bank-{i}" for i in range(MAX_MULTI_BANK_RECALL_BANKS + 1)]
    engine = _harness(bank_results={bid: [] for bid in too_many})
    with pytest.raises(OperationValidationError) as excinfo:
        await MemoryEngine.recall_multi_async(
            engine,
            too_many,
            "query",
            request_context=RC,
            max_tokens=10_000,
        )
    assert excinfo.value.status_code == 422
    assert str(MAX_MULTI_BANK_RECALL_BANKS) in str(excinfo.value)
    # Must fail before fan-out — no sub-call attempts required when count is known.


# --- Job D auth / metadata fixes ---------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_validation_error_propagates_not_soft_fail():
    """OperationValidationError (auth denial) must re-raise, not soft-fail into metadata."""
    engine = _harness(
        bank_results={
            "bank-a": [_fact("a1", "ok", reranker=0.5)],
            "bank-b": OperationValidationError("forbidden for bank-b", status_code=403),
        }
    )
    with pytest.raises(OperationValidationError) as excinfo:
        await MemoryEngine.recall_multi_async(
            engine,
            ["bank-a", "bank-b"],
            "query",
            request_context=RC,
            max_tokens=10_000,
        )
    assert excinfo.value.status_code == 403
    assert "forbidden for bank-b" in str(excinfo.value)


@pytest.mark.asyncio
async def test_orchestrator_cancel_precedes_validation_error():
    """When both cancel and validation errors are present, cancellation wins."""
    engine = _harness(
        bank_results={
            "bank-a": OperationCancelledError("client disconnected"),
            "bank-b": OperationValidationError("forbidden", status_code=403),
        }
    )
    with pytest.raises(OperationCancelledError, match="client disconnected"):
        await MemoryEngine.recall_multi_async(
            engine,
            ["bank-a", "bank-b"],
            "query",
            request_context=RC,
            max_tokens=10_000,
        )


@pytest.mark.asyncio
async def test_orchestrator_runtime_error_still_soft_fails():
    """Ordinary infrastructure errors remain soft partial failures (regression guard)."""
    engine = _harness(
        bank_results={
            "bank-a": [_fact("a1", "ok", reranker=0.9)],
            "bank-b": RuntimeError("db timeout"),
        }
    )
    result = await MemoryEngine.recall_multi_async(
        engine,
        ["bank-a", "bank-b"],
        "query",
        request_context=RC,
        max_tokens=10_000,
    )
    banks = result.metadata[META_MULTI_BANK][META_BANKS]
    assert banks["bank-a"]["status"] == "ok"
    assert banks["bank-b"]["status"] == "error"
    assert [f.id for f in result.results] == ["a1"]


@pytest.mark.asyncio
async def test_orchestrator_client_metadata_has_no_exception_oracle():
    """Client-visible per-bank error must not leak exception class or repr."""
    engine = _harness(
        bank_results={
            "bank-a": [_fact("a1", "ok", reranker=0.5)],
            "bank-b": RuntimeError("secret internal detail"),
        }
    )
    result = await MemoryEngine.recall_multi_async(
        engine,
        ["bank-a", "bank-b"],
        "query",
        request_context=RC,
        max_tokens=10_000,
    )
    err = result.metadata[META_MULTI_BANK][META_BANKS]["bank-b"]["error"]
    assert err == "recall failed for this bank"
    assert "RuntimeError" not in err
    assert "secret" not in err
    # No exception-style repr leakage in the whole multi_bank block.
    import json

    blob = json.dumps(result.metadata[META_MULTI_BANK])
    assert "RuntimeError" not in blob
    assert "secret internal detail" not in blob


# --- Track-A: cap / exact_normalized dedup / prefer_observations / metadata ---


def test_normalize_dedup_key_casefold_and_whitespace():
    assert normalize_dedup_key("  The Sky\nIs  BLUE  ") == "the sky is blue"
    assert normalize_dedup_key(None) == ""
    assert normalize_dedup_key("already clean") == "already clean"


def test_dedup_exact_normalized_drops_later_duplicate_keeps_first():
    """Higher-ranked (earlier) copy wins; later exact-normalized twin is dropped."""
    facts = [
        _fact("keep", "The sky is blue", reranker=0.9),
        _fact("drop", "  the   SKY is\tBLUE ", reranker=0.1),
        _fact("other", "grass is green", reranker=0.5),
    ]
    result = dedup_exact_normalized(facts)
    assert [f.id for f in result.facts] == ["keep", "other"]
    assert result.dropped == 1


def test_cap_per_bank_results_trims_head_and_never_starves():
    bank_a = [_fact(f"a{i}", f"A{i}", reranker=1.0 - i * 0.01) for i in range(8)]
    bank_b = [_fact("b0", "only B", reranker=0.5)]
    capped = cap_per_bank_results([("bank-a", bank_a), ("bank-b", bank_b)], max_per_bank=3)
    assert [f.id for f in capped[0][1]] == ["a0", "a1", "a2"]
    assert [f.id for f in capped[1][1]] == ["b0"]
    # Clamp to >= 1 so a bank with results is never starved by the cap alone.
    clamped = cap_per_bank_results([("bank-a", bank_a)], max_per_bank=0)
    assert len(clamped[0][1]) == 1


def test_merge_cap_dedup_cut_drops_duplicate_before_token_budget():
    """A later duplicate must not consume token budget that a unique fact needs."""
    from hindsight_api.engine.memory_engine import count_tokens

    keep = _fact("keep", "unique-head", reranker=0.9)
    twin = _fact("twin", "UNIQUE-HEAD", reranker=0.8)
    tail = _fact("tail", "unique-tail", reranker=0.7)
    budget = count_tokens(keep.text) + count_tokens(tail.text)
    pipeline = merge_cap_dedup_cut(
        [("bank-a", [keep]), ("bank-b", [twin, tail])],
        merge="score",
        max_tokens=budget,
    )
    assert [f.id for f in pipeline.facts] == ["keep", "tail"]
    assert pipeline.dropped == 1


@pytest.mark.asyncio
async def test_orchestrator_dedup_drops_exact_normalized_duplicate_across_banks():
    engine = _harness(
        bank_results={
            "bank-a": [_fact("a1", "The sky is blue", reranker=0.95)],
            "bank-b": [_fact("b1", "  the SKY   is blue", reranker=0.40)],
        }
    )
    result = await MemoryEngine.recall_multi_async(
        engine,
        ["bank-a", "bank-b"],
        "query",
        merge="score",
        request_context=RC,
        max_tokens=10_000,
    )
    assert [f.id for f in result.results] == ["a1"]
    mb = result.metadata[META_MULTI_BANK]
    assert mb[META_DEDUP] == "exact_normalized"
    assert mb[META_DEDUP_DROPPED] == 1
    assert mb[META_PER_BANK_CAP] == DEFAULT_PER_BANK_MERGE_CAP
    assert mb[META_BANKS]["bank-a"]["status"] == "ok"
    assert mb[META_BANKS]["bank-b"]["status"] == "ok"
    # banks.count is the pre-dedup per-bank contribution.
    assert mb[META_BANKS]["bank-b"]["count"] == 1


@pytest.mark.asyncio
async def test_orchestrator_per_bank_cap_trims_before_merge():
    bank_a = [_fact(f"a{i}", f"A fact {i}", reranker=0.9 - i * 0.001) for i in range(60)]
    bank_b = [_fact("b0", "B only", reranker=0.5)]
    engine = _harness(bank_results={"bank-a": bank_a, "bank-b": bank_b})
    result = await MemoryEngine.recall_multi_async(
        engine,
        ["bank-a", "bank-b"],
        "query",
        merge="score",
        request_context=RC,
        max_tokens=10_000,
    )
    ids = [f.id for f in result.results]
    assert "a49" in ids
    assert "a50" not in ids
    assert "b0" in ids
    assert result.metadata[META_MULTI_BANK][META_PER_BANK_CAP] == 50
    assert result.metadata[META_MULTI_BANK][META_BANKS]["bank-a"]["count"] == 60


@pytest.mark.asyncio
async def test_orchestrator_prefer_observations_default_true_on_fanout_only():
    """Default True is passed to each recall_async; HTTP/MCP still default False."""
    import inspect

    seen: dict[str, bool] = {}
    engine = object.__new__(MemoryEngine)

    async def tracking_recall(bank_id: str, query: str, **kwargs) -> RecallResult:
        seen[bank_id] = kwargs.get("prefer_observations")
        return RecallResult(results=[_fact(f"{bank_id}-1", "x", reranker=0.5)])

    engine.recall_async = tracking_recall  # type: ignore[method-assign]

    async def fake_auth(request_context: RequestContext) -> str:
        return "public"

    engine._authenticate_tenant = fake_auth  # type: ignore[method-assign]
    engine._config_resolver = SimpleNamespace(  # type: ignore[attr-defined]
        get_bank_config=AsyncMock(return_value={"enable_reranking": True})
    )

    await MemoryEngine.recall_multi_async(
        engine,
        ["b1", "b2"],
        "query",
        request_context=RC,
        max_tokens=10_000,
    )
    assert seen == {"b1": True, "b2": True}
    assert MULTI_BANK_PREFER_OBSERVATIONS is True

    single_default = inspect.signature(MemoryEngine.recall_async).parameters["prefer_observations"].default
    multi_default = inspect.signature(MemoryEngine.recall_multi_async).parameters["prefer_observations"].default
    assert single_default is False
    assert multi_default is True


@pytest.mark.asyncio
async def test_orchestrator_prefer_observations_false_is_passed_through():
    """Caller False must reach fan-out; merge/dedup still run independently."""
    seen: dict[str, bool] = {}
    engine = object.__new__(MemoryEngine)

    async def tracking_recall(bank_id: str, query: str, **kwargs) -> RecallResult:
        seen[bank_id] = kwargs.get("prefer_observations")
        text = "Same sentence in both banks" if bank_id == "bank-a" else "same sentence in both banks"
        return RecallResult(results=[_fact(f"{bank_id}-1", text, reranker=0.9 if bank_id == "bank-a" else 0.4)])

    engine.recall_async = tracking_recall  # type: ignore[method-assign]

    async def fake_auth(request_context: RequestContext) -> str:
        return "public"

    engine._authenticate_tenant = fake_auth  # type: ignore[method-assign]
    engine._config_resolver = SimpleNamespace(  # type: ignore[attr-defined]
        get_bank_config=AsyncMock(return_value={"enable_reranking": True})
    )

    result = await MemoryEngine.recall_multi_async(
        engine,
        ["bank-a", "bank-b"],
        "query",
        prefer_observations=False,
        request_context=RC,
        max_tokens=10_000,
    )
    assert seen == {"bank-a": False, "bank-b": False}
    # Dedup is independent of prefer_observations (fan-out flag only).
    assert [f.id for f in result.results] == ["bank-a-1"]
    assert result.metadata[META_MULTI_BANK][META_DEDUP_DROPPED] == 1


@pytest.mark.asyncio
async def test_orchestrator_metadata_keys_present_on_success():
    engine = _harness(
        bank_results={
            "bank-a": [_fact("a1", "ok", reranker=0.5)],
            "bank-b": [_fact("b1", "also", reranker=0.4)],
        }
    )
    result = await MemoryEngine.recall_multi_async(
        engine,
        ["bank-a", "bank-b"],
        "query",
        request_context=RC,
        max_tokens=10_000,
    )
    mb = result.metadata[META_MULTI_BANK]
    assert set(mb) >= {
        META_MERGE_REQUESTED,
        META_MERGE_APPLIED,
        META_MERGE_FALLBACK_REASON,
        META_BANKS,
        META_DEDUP,
        META_DEDUP_DROPPED,
        META_PER_BANK_CAP,
    }
    assert mb[META_DEDUP] == "exact_normalized"
    assert mb[META_DEDUP_DROPPED] == 0
    assert mb[META_PER_BANK_CAP] == 50
    assert mb[META_BANKS]["bank-a"] == {"status": "ok", "count": 1}
    assert mb[META_BANKS]["bank-b"] == {"status": "ok", "count": 1}


@pytest.mark.asyncio
async def test_orchestrator_source_facts_truncated_or_across_banks():
    """origin/main source_facts_truncated: True if any successful bank reported it."""
    engine = _harness(
        bank_results={
            "bank-a": RecallResult(
                results=[_fact("a1", "A", reranker=0.9)],
                source_facts={"sf-a": _fact("sf-a", "src A")},
                source_facts_truncated=False,
            ),
            "bank-b": RecallResult(
                results=[_fact("b1", "B", reranker=0.4)],
                source_facts={"sf-b": _fact("sf-b", "src B")},
                source_facts_truncated=True,
            ),
        }
    )
    result = await MemoryEngine.recall_multi_async(
        engine,
        ["bank-a", "bank-b"],
        "query",
        request_context=RC,
        max_tokens=10_000,
    )
    assert result.source_facts_truncated is True


# --- AuthenticationError hard-fail (94b831d3; not covered by e0a56d80) ---


@pytest.mark.asyncio
async def test_orchestrator_authentication_error_propagates_not_soft_fail():
    """Tenant AuthenticationError from any sub-call must re-raise, not enter bank metadata.

    Auth is per-request (tenant), not per-bank. e0a56d80 re-raised
    OperationValidationError only; AuthenticationError fell into the generic
    soft-fail (live-measured 2026-08-15: multi POST -> 200).
    """
    engine = _harness(
        bank_results={
            "bank-a": [_fact("a1", "ok", reranker=0.5)],
            "bank-b": AuthenticationError("Invalid API key"),
        }
    )
    with pytest.raises(AuthenticationError, match="Invalid API key"):
        await MemoryEngine.recall_multi_async(
            engine,
            ["bank-a", "bank-b"],
            "query",
            request_context=RC,
            max_tokens=10_000,
        )


@pytest.mark.asyncio
async def test_orchestrator_upfront_tenant_auth_failure_skips_fanout():
    """Unauthenticated multi-recall fails before any per-bank recall_async starts."""
    engine = _harness(
        bank_results={
            "bank-a": [_fact("a1", "ok", reranker=0.5)],
            "bank-b": [_fact("b1", "ok", reranker=0.4)],
        }
    )
    called: list[str] = []
    original = engine.recall_async

    async def tracking_recall(bank_id: str, query: str, **kwargs):
        called.append(bank_id)
        return await original(bank_id, query, **kwargs)

    engine.recall_async = tracking_recall  # type: ignore[method-assign]

    async def reject_auth(request_context: RequestContext) -> str:
        raise AuthenticationError("Invalid API key")

    engine._authenticate_tenant = reject_auth  # type: ignore[method-assign]
    with pytest.raises(AuthenticationError, match="Invalid API key"):
        await MemoryEngine.recall_multi_async(
            engine,
            ["bank-a", "bank-b"],
            "query",
            request_context=RC,
            max_tokens=10_000,
        )
    assert called == []
