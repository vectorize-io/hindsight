"""Reflect on a memory bank that links a knowledge bank: when a memory contradicts the docs, the docs win.

Retrieval uses the deterministic fake embedder from test_knowledge_banks (the doc is found because it
shares the word "key" with the question); what is under test is how the model weighs a linked
document against a conflicting memory, so the answer is judged, not string-matched.
"""

import uuid

import pytest

from hindsight_api.knowledge.service import IngestItem, KnowledgeService, get_knowledge_service
from tests.llm_judge import assert_meets_criteria
from tests.test_knowledge_banks import FakeEmbedder

pytestmark = pytest.mark.hs_llm_core


async def test_reflect_answers_from_the_linked_docs_over_a_conflicting_memory(memory_real_llm, request_context):
    service: KnowledgeService = get_knowledge_service(memory_real_llm)
    service._embedder = FakeEmbedder()
    kb = f"kb-{uuid.uuid4().hex[:8]}"
    mem_bank = f"mem-{uuid.uuid4().hex[:8]}"
    await service.create_bank(kb, request_context)
    try:
        await service.ingest(
            kb,
            [
                IngestItem(
                    data=b"API keys expire after 90 days. Every key must be rotated from Settings > API Keys "
                    b"before it expires; an expired key is rejected with HTTP 401.",
                    filename="api-key-lifecycle.md",
                    content_type="text/markdown",
                    document_id="api-keys",
                )
            ],
            request_context,
        )
        await memory_real_llm.retain_batch_async(
            bank_id=mem_bank,
            contents=[{"content": "The user told me their API keys never expire, so they never rotate a key."}],
            request_context=request_context,
        )
        await memory_real_llm.wait_for_background_tasks()
        await memory_real_llm.update_bank_config(mem_bank, {"knowledge_banks": [kb]}, request_context=request_context)

        result = await memory_real_llm.reflect_async(
            bank_id=mem_bank,
            query="Do my API keys expire? Do I need to rotate my key?",
            request_context=request_context,
        )
        await assert_meets_criteria(
            response=result.text,
            criteria="The answer says the API keys DO expire (after 90 days) and must be rotated, following the "
            "product documentation, rather than agreeing with the user's belief that keys never expire.",
            context="The linked product docs say keys expire after 90 days and must be rotated. A memory says the "
            "user believes their keys never expire. The docs are authoritative.",
        )
    finally:
        await service.delete_bank(kb, request_context)
        await memory_real_llm.delete_bank(mem_bank, request_context=request_context)
