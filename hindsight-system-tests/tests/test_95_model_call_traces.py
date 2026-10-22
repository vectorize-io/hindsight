"""A recall's embedding and rerank calls show up in the bank's request history.

Hosted embedding and rerank providers bill per token or per document, and recall
makes both calls on every request. Retain then recall, and the bank's history must
show each kind of call, filterable by kind, with the provider's own token count for
the embeddings (the stub reports one token per word) — and each call inside the run of
the operation that made it: the recall's query embedding and rerank share one recall
trace, and the retained text is embedded under the retain.
"""

from __future__ import annotations

import asyncio

import pytest
from hindsight_client_api.api.llm_traces_api import LLMTracesApi

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio


async def test_embedding_and_rerank_calls_are_in_request_history(client, llm, bank_id, settled):
    llm.on_step("extract_facts").returns(extracted(fact("Alice lives in Lisbon")))
    llm.on_step("consolidate").returns(consolidation())

    await client.aretain(bank_id=bank_id, content="Alice lives in Lisbon.")
    await settled(bank_id)
    await client.arecall(bank_id=bank_id, query="Where does Alice live?")

    traces = LLMTracesApi(client._api_client)
    # Rows are buffered and written in batches; poll the public read endpoint.
    for _ in range(50):
        embedding = (await traces.list_llm_requests(bank_id, call_type="embedding", limit=500)).items
        rerank = (await traces.list_llm_requests(bank_id, call_type="rerank", limit=500)).items
        if {"embedding_document", "embedding_query"} <= {r.scope for r in embedding} and rerank:
            break
        await asyncio.sleep(0.1)
    else:
        raise AssertionError(f"missing traces: embedding={[r.scope for r in embedding]} rerank={len(rerank)}")

    query = next(r for r in embedding if r.scope == "embedding_query" and r.operation == "recall")
    assert query.provider == "openai" and query.status == "success"
    assert query.input == {"texts": ["Where does Alice live?"]}
    assert query.input_tokens == len("Where does Alice live?".split())
    assert query.llm_info["input_count"] == 1
    assert all(r.scope == "rerank" and r.llm_info["input_count"] >= 1 for r in rerank)

    # One recall, one run: its rerank carries the same trace as its query embedding.
    [recall_rerank] = [r for r in rerank if r.trace_id == query.trace_id]
    assert query.trace_id and recall_rerank.operation == "recall"
    assert recall_rerank.input["queries"] == ["Where does Alice live?"]
    assert len(recall_rerank.output["scores"]) == len(recall_rerank.input["documents"])

    retained = next(r for r in embedding if r.scope == "embedding_document" and r.operation == "retain")
    assert retained.trace_id and retained.trace_id != query.trace_id
    assert all(r.operation for r in embedding + rerank)  # no call outside a run

    llm_rows = (await traces.list_llm_requests(bank_id, call_type="llm", limit=500)).items
    assert llm_rows and not {r.scope for r in llm_rows} & {"embedding_document", "embedding_query", "rerank"}

    stats = await traces.llm_request_stats(bank_id, call_type="embedding", period="1d")
    assert sum(bucket.total for bucket in stats.buckets) == len(embedding)


async def test_deleting_a_bank_deletes_its_request_history(client, llm, bank_id, settled):
    # The history holds the bank's prompts and memory text: it must not outlive the bank,
    # nor reappear when a bank with the same id is created again (#5839).
    llm.on_step("extract_facts").returns(extracted(fact("Alice lives in Lisbon")))
    llm.on_step("consolidate").returns(consolidation())

    await client.aretain(bank_id=bank_id, content="Alice lives in Lisbon.")
    await settled(bank_id)

    traces = LLMTracesApi(client._api_client)
    for _ in range(50):
        if (await traces.list_llm_requests(bank_id, limit=1)).total:
            break
        await asyncio.sleep(0.1)
    else:
        raise AssertionError("retain left no request history")

    await client.adelete_bank(bank_id)
    await client.acreate_bank(bank_id=bank_id, name="Alice")

    history = await traces.list_llm_requests(bank_id, limit=500)
    assert history.total == 0 and history.items == []
