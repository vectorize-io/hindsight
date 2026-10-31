"""A recall scored by a fallback reranker says so, and its log lines can be joined.

A reranker failover chain keeps recall answering when the primary is down, but the
members score on different scales: a client that thresholds `scores.reranker`, or
anyone measuring recall quality, has to know which member scored a response (#6072).
Two things make that possible, and both only show up across the seam between the
failover and the rest of recall:

- the trace names the member that answered — but only when the server opts in, because
  it reveals how the server's reranker is set up;
- the failover warning carries the same recall id as that recall's start line, so it
  can be tied to one request when several are in flight.

The bank lives in the shared database, so the same memories are recalled once through
the ordinary server (flag off) and once through `failover_server` (dead primary, `rrf`
next, flag on).
"""

from __future__ import annotations

import re

import pytest

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

QUERY = "Where does Alice live?"
BERLIN = "Alice moved to Berlin in 2021 | Involving: Alice"
CELLO = "Alice plays the cello professionally | Involving: Alice"


async def test_a_failed_over_recall_names_its_reranker_and_joins_its_logs(
    client, failover_client, failover_server, llm, bank_id, settled
):
    llm.on_step("extract_facts").returns(
        extracted(
            fact("Alice moved to Berlin in 2021", who="Alice", entities=["Alice", "Berlin"]),
            fact("Alice plays the cello professionally", who="Alice", entities=["Alice", "cello"]),
        )
    )
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin in 2021 and plays the cello.")
    await settled(bank_id)

    # --- off by default -------------------------------------------------------
    ordinary = await client.arecall(bank_id=bank_id, query=QUERY, trace=True)
    assert ordinary.trace["reranker_provider"] is None, "the reranker setup must not leak unless the server opts in"

    # --- the failover answers, and the trace says who -------------------------
    failed_over = await failover_client.arecall(bank_id=bank_id, query=QUERY, trace=True)
    assert sorted(r.text for r in failed_over.results) == sorted([BERLIN, CELLO]), (
        "recall must still answer when the primary reranker is down"
    )
    assert failed_over.trace["reranker_provider"] == "rrf", "the trace names the member that scored the recall"

    # --- the failover warning joins this recall's own log lines ---------------
    logs = failover_server.logs()
    warning = re.search(r"\[RECALL (\S+)\] Reranker member 0 \(siliconflow\) failed", logs)
    assert warning, f"no failover warning with a recall id in the server log:\n{logs[-4000:]}"
    recall_id = warning.group(1)
    assert recall_id.startswith(bank_id[:8])
    assert f"[RECALL {recall_id}] Starting recall for query: {QUERY}" in logs, (
        "the failover warning's id must match the start line of the recall it belongs to"
    )
