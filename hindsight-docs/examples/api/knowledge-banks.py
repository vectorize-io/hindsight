#!/usr/bin/env python3
"""
Knowledge Banks examples for Hindsight (Python).
Run: python examples/api/knowledge-banks.py
"""
import asyncio
import json
import os
from pathlib import Path

import requests

HINDSIGHT_URL = os.getenv("HINDSIGHT_API_URL", "http://localhost:8888")
SAMPLE_PDF = Path(__file__).parent / "sample.pdf"

# =============================================================================
# Setup (not shown in docs)
# =============================================================================
from hindsight_client import Hindsight
from hindsight_client_api import ApiClient, Configuration
from hindsight_client_api.api import KnowledgeBanksApi
from hindsight_client_api.models import (
    CreateKnowledgeBankRequest,
    KnowledgeBankConfigUpdate,
    KnowledgeIngestRequest,
    KnowledgeIngestTextItem,
    KnowledgeSearchRequest,
)

client = Hindsight(base_url=HINDSIGHT_URL)
client.create_bank(bank_id="support-agent", name="Support Agent")


async def main() -> None:
    api_client = ApiClient(Configuration(host=HINDSIGHT_URL))
    kb = KnowledgeBanksApi(api_client)

    # =========================================================================
    # Doc Examples
    # =========================================================================

    # [docs:create-knowledge-bank]
    await kb.create_knowledge_bank(CreateKnowledgeBankRequest(bank_id="product-manuals"))
    # [/docs:create-knowledge-bank]

    # [docs:ingest-files]
    result = await kb.ingest_knowledge_files(
        bank_id="product-manuals",
        files=[
            ("x200-manual.pdf", SAMPLE_PDF.read_bytes()),
            ("x200-quickstart.md", b"# X200 quick start\n\nHold the power button for two seconds.\n"),
        ],
        request=json.dumps({"files": [
            {"document_id": "x200-manual", "tags": ["x200"]},
            {"document_id": "x200-quickstart", "tags": ["x200", "quickstart"]},
        ]}),
    )
    print(result.document_ids, result.operation_ids)
    # [/docs:ingest-files]

    # [docs:ingest-text]
    await kb.ingest_knowledge_texts(
        "product-manuals",
        KnowledgeIngestRequest(items=[
            KnowledgeIngestTextItem(
                document_id="returns-policy",
                text="# Returns\n\nYou can return an X200 within 30 days...",
                tags=["policy"],
            )
        ]),
    )
    # [/docs:ingest-text]

    # [docs:search]
    found = await kb.search_knowledge_bank(
        "product-manuals",
        KnowledgeSearchRequest(query="How long do I have to return an X200?", max_tokens=2048),
    )
    for hit in found.results:
        print(hit.document_id, hit.text)
    # [/docs:search]

    # [docs:list-tags]
    tags = await kb.list_knowledge_bank_tags("product-manuals", q="x2*")
    for item in tags.items:
        print(item.tag, item.count)
    # [/docs:list-tags]

    # [docs:bank-config]
    await kb.update_knowledge_bank_config(
        "product-manuals",
        KnowledgeBankConfigUpdate(updates={"text_passage_tokens": 256, "disabled_media_types": ["video"]}),
    )
    # [/docs:bank-config]

    # =========================================================================
    # Checks (not shown in docs): the calls above must have worked, not just answered
    # =========================================================================
    assert any(h.document_id == "returns-policy" for h in found.results)
    assert any(t.tag == "policy" for t in (await kb.list_knowledge_bank_tags("product-manuals")).items)
    config = await kb.get_knowledge_bank_config("product-manuals")
    assert config.overrides.text_passage_tokens == 256
    # The file ingest runs in the background: wait for the quick start to be ready.
    for _ in range(120):
        doc = await kb.get_knowledge_document("product-manuals", "x200-quickstart")
        if doc.status in ("ready", "failed"):
            break
        await asyncio.sleep(1)
    assert doc.status == "ready", doc.error

    await api_client.close()


asyncio.run(main())

# [docs:link-memory-bank]
client.update_bank_config("support-agent", knowledge_banks=["product-manuals"])
# [/docs:link-memory-bank]
assert "product-manuals" in client.get_bank_config("support-agent")["config"]["knowledge_banks"]


async def delete() -> None:
    api_client = ApiClient(Configuration(host=HINDSIGHT_URL))
    kb = KnowledgeBanksApi(api_client)
    # [docs:delete-knowledge-bank]
    await kb.delete_knowledge_bank("product-manuals")
    # [/docs:delete-knowledge-bank]
    await api_client.close()


asyncio.run(delete())

# =============================================================================
# Cleanup (not shown in docs)
# =============================================================================
requests.delete(f"{HINDSIGHT_URL}/v1/default/banks/support-agent")

print("knowledge-banks.py: All examples passed")
