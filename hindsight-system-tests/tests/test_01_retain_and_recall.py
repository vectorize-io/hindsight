"""The foundation story: something retained can be recalled.

Everything else in this suite builds on this working. A caller hands Hindsight a
sentence, the extraction model turns it into facts, the worker finishes the
background half of the retain, and a later question that never repeats the
original wording gets those facts back.

The assertions are deliberately total. With the LLM, the embedder and the
reranker all stubbed, a recall is a pure function of the input — so the *whole*
response is pinned, not just the presence of a keyword: the ranking, the four
scores behind it, how a fact is rendered back into text, which envelope fields
stay empty. A test that only checks "Berlin appears somewhere" passes just as
happily when fusion inverts, the reranker stops contributing, or the temporal
fields quietly stop being parsed.

Note what this test does *not* touch: no engine object, no SQL, no internal
module. It talks to a server process over HTTP through the published client, so
it stays true across any refactor that keeps the API's promises — which is
exactly what makes it worth having.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

CONTENT = "Alice moved to Berlin in 2021 and works as a cellist."


async def test_a_retained_memory_comes_back_from_recall(client, llm, bank_id, settled):
    llm.on_step("extract_facts", contains="Berlin").returns(
        extracted(
            fact(
                "Alice moved to Berlin in 2021",
                when="2021",
                where="Berlin",
                who="Alice",
                entities=["Alice", "Berlin"],
            ),
            fact("Alice plays the cello professionally", who="Alice", entities=["Alice", "cello"]),
        )
    )
    # Retain also triggers consolidation in the worker. This story is not about
    # observations, so the model is told to draw none — but it is told explicitly,
    # because a call nobody scripted is a gap in the test, not a detail.
    llm.on_step("consolidate").returns(consolidation())

    await client.aretain(bank_id=bank_id, content=CONTENT)
    await settled(bank_id)

    response = await client.arecall(bank_id=bank_id, query="Where does Alice live?")

    # --- ranking ------------------------------------------------------------
    # Both facts come back, and the one that answers the question ranks first.
    # The query shares "Alice" with both and nothing else with the cello fact, so
    # a working pipeline must separate them; an inverted order here means fusion
    # or reranking regressed.
    assert [result.text for result in response.results] == [
        "Alice moved to Berlin in 2021 | When: 2021 | Involving: Alice",
        "Alice plays the cello professionally | Involving: Alice",
    ]

    berlin, cello = response.results

    # --- how a fact is rendered back ----------------------------------------
    # `what` is not stored verbatim: the populated `when`/`who` fields are appended
    # as labelled clauses, and the ones left at "N/A" are dropped. That rendering is
    # what a caller reads, so it is part of the contract.
    assert berlin.type == "world"
    assert cello.type == "world"
    assert berlin.context == ""
    assert berlin.tags == []
    assert berlin.metadata == {}

    # `when="2021"` is prose for the reader and does not populate the structured
    # temporal fields — those come from `occurred_start`/`occurred_end`, which this
    # retain left unset.
    assert berlin.occurred_start is None
    assert berlin.occurred_end is None

    # --- identity -----------------------------------------------------------
    # One retain is one document, so both facts carry the same document id, and the
    # chunk id is the documented `{bank_id}_{document_id}_{index}` composite — the
    # form that makes chunk ids globally unique and therefore safe to query without
    # a separate bank predicate.
    assert berlin.document_id == cello.document_id
    assert berlin.chunk_id == f"{bank_id}_{berlin.document_id}_0"
    assert berlin.id != cello.id

    # --- scores -------------------------------------------------------------
    # Every retrieval strategy contributed, and each component is reproducible to
    # the bit: the embedder is a pure function of the text (see `lexical.py`), the
    # reranker derives from the same model, and BM25 is deterministic.
    assert berlin.scores.semantic == pytest.approx(0.43465916228227297, abs=1e-12)
    assert berlin.scores.reranker == pytest.approx(0.37416573867739417, abs=1e-12)
    assert berlin.scores.keyword == pytest.approx(0.30000001192092896, abs=1e-12)

    assert cello.scores.semantic == pytest.approx(0.4146140043322447, abs=1e-12)
    assert cello.scores.reranker == pytest.approx(0.3333333333333333, abs=1e-12)
    assert cello.scores.keyword == pytest.approx(0.30000001192092896, abs=1e-12)

    # `final` folds in recency, measured against wall-clock now, so it drifts in the
    # ninth decimal between runs. Loose enough to absorb that, tight enough that a
    # changed weighting fails.
    assert berlin.scores.final == pytest.approx(0.41158231, abs=1e-6)
    assert cello.scores.final == pytest.approx(0.36666666, abs=1e-6)
    assert berlin.scores.final > cello.scores.final

    # --- envelope -----------------------------------------------------------
    # Nothing was asked for beyond the facts, so the optional sections stay unset.
    # Pinning them catches a change that starts populating (and paying for) one of
    # these by default.
    assert response.trace is None
    assert response.entities is None
    assert response.chunks is None
    assert response.source_facts is None
