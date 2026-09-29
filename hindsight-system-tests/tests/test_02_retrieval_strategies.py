"""Every retrieval arm runs, and fusion can see what each one found.

The docs describe recall as several strategies searched in parallel and then
fused. That claim is only worth making if a caller can tell the arms apart, so
this story asserts the shape through `trace=True` rather than inferring it from
final scores — a single blended number cannot distinguish "both arms agreed"
from "one arm did all the work and the other returned nothing".

What makes this checkable at all is that the stub's embedder is a pure function
of the text: the same query produces the same candidate order every run, so the
per-arm ranks below are facts about the pipeline, not about a model's mood.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

# The three arms that actually issue a retrieval, times the three kinds of thing
# they can retrieve. Temporal is deliberately absent: despite the docs' "four
# search strategies", time enters as a *scoring* component and an optional query
# constraint, not as a fourth arm with its own candidate list — see the
# `temporal` entry in `score_components` below.
EXPECTED_ARMS = {
    ("semantic", "world"),
    ("semantic", "experience"),
    ("semantic", "observation"),
    ("bm25", "world"),
    ("bm25", "experience"),
    ("bm25", "observation"),
    ("graph", "world"),
    ("graph", "experience"),
    ("graph", "observation"),
}


async def test_each_retrieval_arm_runs_and_fusion_records_what_it_found(client, llm, bank_id, settled):
    llm.on_step("extract_facts").returns(
        extracted(
            fact("Alice moved to Berlin in 2021", who="Alice", entities=["Alice", "Berlin"]),
            fact("Alice plays the cello professionally", who="Alice", entities=["Alice", "cello"]),
        )
    )
    llm.on_step("consolidate").returns(consolidation())

    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin in 2021 and works as a cellist.")
    await settled(bank_id)

    response = await client.arecall(bank_id=bank_id, query="Where does Alice live?", trace=True)
    trace = response.trace
    assert trace is not None, "trace=True must return a trace"

    # --- every arm ran --------------------------------------------------------
    # Asserting the whole set, not just membership: an arm that silently stops
    # being dispatched is the failure this catches, and it looks like nothing at
    # all in the results.
    arms = {(arm["method_name"], arm["fact_type"]) for arm in trace["retrieval_results"]}
    assert arms == EXPECTED_ARMS

    # --- two arms independently found the same facts --------------------------
    by_arm = {(a["method_name"], a["fact_type"]): [r["text"] for r in a["results"]] for a in trace["retrieval_results"]}
    assert len(by_arm[("semantic", "world")]) == 2
    assert len(by_arm[("bm25", "world")]) == 2

    # Graph traversal walks entity links *between* facts. Both facts here came
    # from one document and share only the entity "Alice", which is an entry
    # point rather than a path to somewhere new — so the arm correctly returns
    # nothing. A story that retains two documents linked by a shared entity is
    # what exercises the traversal itself (epic item 73).
    assert by_arm[("graph", "world")] == []

    # --- fusion recorded each arm's contribution ------------------------------
    # This is the part a blended score cannot tell you: both arms ranked both
    # facts, and RRF saw both ranks.
    merged = {m["text"]: m["source_ranks"] for m in trace["rrf_merged"]}
    assert merged["Alice moved to Berlin in 2021 | Involving: Alice"] == {"semantic_rank": 1, "bm25_rank": 1}
    assert merged["Alice plays the cello professionally | Involving: Alice"] == {"semantic_rank": 2, "bm25_rank": 2}

    # --- the reranker combined the documented components ----------------------
    top = trace["reranked"][0]
    assert top["rerank_rank"] == 1
    assert set(top["score_components"]) == {
        "cross_encoder_score",
        "cross_encoder_score_normalized",
        "rrf_score",
        "rrf_normalized",
        "temporal",
        "recency",
        "proof_norm",
        "combined_score",
    }
    # The combined score is what surfaces as `scores.final`, so the trace and the
    # response must agree — a trace that reports a different number from the one
    # the caller was ranked by would be worse than no trace.
    assert top["score_components"]["combined_score"] == pytest.approx(response.results[0].scores.final, abs=1e-9)
