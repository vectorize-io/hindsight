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


# [docs:document-list]
from hindsight_client_api import ApiClient, Configuration
from hindsight_client_api.api import DocumentsApi

async def list_documents_example():
    config = Configuration(host="http://localhost:8888")
    api_client = ApiClient(config)
    api = DocumentsApi(api_client)

    # List all documents
    result = await api.list_documents(bank_id="my-bank")
    print(f"Total documents: {result.total}")

    # Filter by document ID substring
    result = await api.list_documents(bank_id="my-bank", q="report")

    # Filter by tags — only docs tagged with "team-a" (untagged excluded)
    result = await api.list_documents(
        bank_id="my-bank",
        tags=["team-a"],
        tags_match="any_strict",
    )

    # Combine ID search and tags
    result = await api.list_documents(
        bank_id="my-bank",
        q="meeting",
        tags=["team-a", "team-b"],
        tags_match="all_strict",  # must have both tags
    )

    # Paginate
    result = await api.list_documents(bank_id="my-bank", limit=20, offset=40)
    print(f"Page items: {len(result.items)}")

import asyncio
asyncio.run(list_documents_example())
# [/docs:document-list]


# [docs:document-get]
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


# [docs:document-update-tags]
from hindsight_client_api import ApiClient, Configuration
from hindsight_client_api.api import DocumentsApi
from hindsight_client_api.models import UpdateDocumentTagsRequest

async def update_document_tags_example():
    config = Configuration(host="http://localhost:8888")
    api_client = ApiClient(config)
    api = DocumentsApi(api_client)

    # Fix tags on a document retained with the wrong scope
    result = await api.update_document_tags(
        bank_id="my-bank",
        document_id="meeting-2024-03-15",
        update_document_tags_request=UpdateDocumentTagsRequest(tags=["team-a", "team-b"]),
    )
    print(f"Updated: {result.success}")

    # Remove all tags (make document visible everywhere)
    await api.update_document_tags(
        bank_id="my-bank",
        document_id="meeting-2024-03-15",
        update_document_tags_request=UpdateDocumentTagsRequest(tags=[]),
    )

asyncio.run(update_document_tags_example())
# [/docs:document-update-tags]


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
