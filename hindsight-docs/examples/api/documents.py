#!/usr/bin/env python3
"""
Documents API examples for Hindsight.
Run: python examples/api/documents.py
"""
import os
import requests

HINDSIGHT_URL = os.getenv("HINDSIGHT_API_URL", "http://localhost:8888")

# =============================================================================
# Setup (not shown in docs)
# =============================================================================
from hindsight_client import Hindsight

client = Hindsight(base_url=HINDSIGHT_URL)

# =============================================================================
# Doc Examples
# =============================================================================

# [docs:document-retain]
# Retain with document ID
client.retain(
    bank_id="my-bank",
    content="Alice presented the Q4 roadmap...",
    document_id="meeting-2024-03-15"
)

# Batch retain for a document with different sections
client.retain_batch(
    bank_id="my-bank",
    items=[
        {"content": "Item 1: Product launch delayed to Q2", "document_id": "meeting-2024-03-15-section-1"},
        {"content": "Item 2: New hiring targets announced", "document_id": "meeting-2024-03-15-section-2"},
        {"content": "Item 3: Budget approved for ML team", "document_id": "meeting-2024-03-15-section-3"}
    ]
)
# [/docs:document-retain]


# [docs:document-update]
# Original
client.retain(
    bank_id="my-bank",
    content="Project deadline: March 31",
    document_id="project-plan"
)

# Update (deletes old facts, creates new ones)
client.retain(
    bank_id="my-bank",
    content="Project deadline: April 15 (extended)",
    document_id="project-plan"
)
# [/docs:document-update]


# [docs:document-get]
import asyncio
from hindsight_client_api import ApiClient, Configuration
from hindsight_client_api.api import DocumentsApi

async def get_document_example():
    config = Configuration(host="http://localhost:8888")
    api_client = ApiClient(config)
    api = DocumentsApi(api_client)

    # Get document to expand context from recall results
    doc = await api.get_document(
        bank_id="my-bank",
        document_id="meeting-2024-03-15"
    )

    print(f"Document: {doc.id}")
    print(f"Original text: {doc.original_text}")
    print(f"Memory count: {doc.memory_unit_count}")
    print(f"Created: {doc.created_at}")

asyncio.run(get_document_example())
# [/docs:document-get]


# [docs:document-delete]
from hindsight_client_api import ApiClient, Configuration
from hindsight_client_api.api import DocumentsApi

async def delete_document_example():
    config = Configuration(host="http://localhost:8888")
    api_client = ApiClient(config)
    api = DocumentsApi(api_client)

    # Delete document and all its memories
    result = await api.delete_document(
        bank_id="my-bank",
        document_id="meeting-2024-03-15"
    )

    print(f"Deleted {result.memory_units_deleted} memories")

asyncio.run(delete_document_example())
# [/docs:document-delete]


# =============================================================================
# Cleanup (not shown in docs)
# =============================================================================
requests.delete(f"{HINDSIGHT_URL}/v1/default/banks/my-bank")

print("documents.py: All examples passed")
