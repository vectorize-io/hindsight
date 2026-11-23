"""Knowledge-bank search on a server whose knowledge search has a reranker.

The reranker (any Cohere-style /rerank endpoint) gets the search's best hits in one request, as many as
its token budget holds; a PDF page goes as its picture with its text. Its scores reorder them. Each step
is unit-tested; the story shows that this order is what the client receives, that the page goes with its
picture, that a server without the setting searches the very same bank in the search's own order, and that
a server with ``HINDSIGHT_API_KB_ENABLE_RERANKING=false`` reranks only a search that asks for it.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest

from hindsight_system_tests import KnowledgeBanks, UploadFile

from .test_110_knowledge_banks import GUIDES, _text_pdf

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def knowledge_bank(client, knowledge_reranker_client) -> AsyncIterator[str]:
    """The guides and a one-page PDF, ingested through the main server: the reranking server runs no
    worker, and the two share one database."""
    kb = KnowledgeBanks(client)
    bank_id = f"systest-kb-{uuid.uuid4().hex[:12]}"
    assert (await kb.create(bank_id)).status == 200
    await kb.ingest(bank_id, [{"document_id": d, "text": t} for d, t in GUIDES.items()])
    r = await kb.ingest_files(
        bank_id, [UploadFile("exits.pdf", _text_pdf("Emergency exits are on floor nine"))], [{"document_id": "exits"}]
    )
    assert r.status == 202, r.body
    await kb.wait_until_processed(bank_id, 4)
    yield bank_id
    await kb.delete(bank_id)


async def _order(client, bank_id: str, query: str) -> list[str]:
    r = await KnowledgeBanks(client).search(bank_id, query=query)
    assert r.status == 200, r.body
    return [hit["document_id"] for hit in r.body["results"]]


async def test_the_rerankers_order_reaches_the_client_and_a_page_goes_with_its_picture(
    client, knowledge_reranker_client, knowledge_bank, stubs
):
    query = "how do I rotate an api key"
    # Without the setting, the search's own order: the guide about keys first.
    assert await _order(client, knowledge_bank, query) == ["keys", "outage", "billing", "exits"]
    assert stubs.rerank.assets == []

    # The reranker scores billing and exits highest and keys lowest: billing leads and keys, the search's
    # first, ends last.
    stubs.rerank.rerank_scores = {"Invoices": 0.9, "Emergency exits": 0.8, "outage": 0.3, "Rotate": 0.05}
    assert await _order(knowledge_reranker_client, knowledge_bank, query) == ["billing", "exits", "outage", "keys"]
    # All four in one request, the page with its picture as a JPEG.
    assert stubs.rerank.assets == [{next(iter(stubs.rerank.assets[0])): "jpg"}]


async def test_a_search_can_skip_the_reranker(knowledge_reranker_client, knowledge_bank, stubs):
    query = "how do I rotate an api key"
    stubs.rerank.rerank_scores = {"Invoices": 0.9, "Emergency exits": 0.8, "outage": 0.3, "Rotate": 0.05}
    kb = KnowledgeBanks(knowledge_reranker_client)

    async def order(**flag: bool) -> list[str]:
        r = await kb.search(knowledge_bank, query=query, **flag)
        assert r.status == 200, r.body
        return [hit["document_id"] for hit in r.body["results"]]

    # false: the search's own order, and no rerank request goes out.
    assert await order(enable_reranking=False) == ["keys", "outage", "billing", "exits"]
    assert stubs.rerank.assets == []
    # Unset, as true: the configured reranker's order.
    assert await order() == ["billing", "exits", "outage", "keys"]
    assert await order(enable_reranking=True) == ["billing", "exits", "outage", "keys"]


async def test_a_server_can_make_the_reranker_opt_in(knowledge_reranker_opt_in_client, knowledge_bank, stubs):
    query = "how do I rotate an api key"
    stubs.rerank.rerank_scores = {"Invoices": 0.9, "Emergency exits": 0.8, "outage": 0.3, "Rotate": 0.05}
    kb = KnowledgeBanks(knowledge_reranker_opt_in_client)

    async def order(**flag: bool) -> list[str]:
        r = await kb.search(knowledge_bank, query=query, **flag)
        assert r.status == 200, r.body
        return [hit["document_id"] for hit in r.body["results"]]

    # Unset: the search's own order, and no rerank request goes out.
    assert await order() == ["keys", "outage", "billing", "exits"]
    assert stubs.rerank.assets == []
    # true: the configured reranker's order.
    assert await order(enable_reranking=True) == ["billing", "exits", "outage", "keys"]
    assert len(stubs.rerank.assets) == 1
