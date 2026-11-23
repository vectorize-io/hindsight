#!/usr/bin/env python3
"""
Knowledge Banks examples for Hindsight (Python).
Run: python examples/api/knowledge-banks.py
"""
import os
import time
from pathlib import Path

import requests

HINDSIGHT_URL = os.getenv("HINDSIGHT_API_URL", "http://localhost:8888")
SAMPLE_PDF = Path(__file__).parent / "sample.pdf"

# =============================================================================
# Setup (not shown in docs)
# =============================================================================
from hindsight_client import Hindsight

client = Hindsight(base_url=HINDSIGHT_URL)
client.create_bank(bank_id="support-agent", name="Support Agent")

# =============================================================================
# Doc Examples
# =============================================================================

# [docs:create-knowledge-bank]
client.create_knowledge_bank("product-manuals")
# [/docs:create-knowledge-bank]

# [docs:ingest-files]
result = client.ingest_knowledge_files(
    "product-manuals",
    files=[
        SAMPLE_PDF,  # a path, or a (filename, bytes) pair
        ("x200-quickstart.md", b"# X200 quick start\n\nHold the power button for two seconds.\n"),
    ],
    files_options=[
        {"document_id": "x200-manual", "tags": ["x200"]},
        {"document_id": "x200-quickstart", "tags": ["x200", "quickstart"]},
    ],
)
print(result.document_ids, result.operation_ids)
# [/docs:ingest-files]

# [docs:ingest-text]
client.ingest_knowledge_texts(
    "product-manuals",
    [
        {
            "document_id": "returns-policy",
            "text": "# Returns\n\nYou can return an X200 within 30 days...",
            "tags": ["policy"],
        }
    ],
)
# [/docs:ingest-text]

# [docs:search]
found = client.search_knowledge_bank(
    "product-manuals", query="How long do I have to return an X200?", max_tokens=2048
)
for hit in found.results:
    print(hit.document_id, hit.text)
# [/docs:search]

# [docs:list-tags]
tags = client.list_knowledge_bank_tags("product-manuals", q="x2*")
for item in tags.items:
    print(item.tag, item.count)
# [/docs:list-tags]

# [docs:link-memory-bank]
client.update_bank_config("support-agent", knowledge_banks=["product-manuals"])
# [/docs:link-memory-bank]

# [docs:bank-config]
client.update_knowledge_bank_config(
    "product-manuals", {"text_passage_tokens": 256, "disabled_media_types": ["video"]}
)
# [/docs:bank-config]

# =============================================================================
# Checks (not shown in docs): the calls above must have worked, not just answered
# =============================================================================
assert any(h.document_id == "returns-policy" for h in found.results)
assert any(t.tag == "policy" for t in client.list_knowledge_bank_tags("product-manuals").items)
assert client.get_knowledge_bank_config("product-manuals").overrides.text_passage_tokens == 256
assert "product-manuals" in client.get_bank_config("support-agent")["config"]["knowledge_banks"]
# The file ingest runs in the background: wait for the quick start to be ready.
for _ in range(120):
    doc = client.get_knowledge_document("product-manuals", "x200-quickstart")
    if doc.status in ("ready", "failed"):
        break
    time.sleep(1)
assert doc.status == "ready", doc.error

# [docs:documents]
for d in client.list_knowledge_documents("product-manuals", status="ready").documents:
    print(d.document_id, d.media_type, d.passage_count)
doc = client.get_knowledge_document("product-manuals", "x200-quickstart")
print(doc.status, [p.text for p in doc.passages])
# [/docs:documents]
assert any(d.document_id == "returns-policy" for d in client.list_knowledge_documents("product-manuals").documents)

# [docs:settings]
# Every setting as the bank uses it, then only what the bank changes
config = client.get_knowledge_bank_config("product-manuals")
print(config.config, config.overrides)

# Drop one override, or all of them
client.update_knowledge_bank_config("product-manuals", {"disabled_media_types": None})
client.reset_knowledge_bank_config("product-manuals")
# [/docs:settings]
assert client.get_knowledge_bank_config("product-manuals").overrides.text_passage_tokens is None

# [docs:delete-document]
client.delete_knowledge_document("product-manuals", "returns-policy")
# [/docs:delete-document]
assert all(d.document_id != "returns-policy" for d in client.list_knowledge_documents("product-manuals").documents)

# [docs:delete-knowledge-bank]
client.delete_knowledge_bank("product-manuals")
# [/docs:delete-knowledge-bank]

# =============================================================================
# Cleanup (not shown in docs)
# =============================================================================
requests.delete(f"{HINDSIGHT_URL}/v1/default/banks/support-agent")

print("knowledge-banks.py: All examples passed")
