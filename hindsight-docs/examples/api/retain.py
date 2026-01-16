#!/usr/bin/env python3
"""
Retain API examples for Hindsight.
Run: python examples/api/retain.py
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

# [docs:retain-basic]
client.retain(
    bank_id="my-bank",
    content="Alice works at Google as a software engineer"
)
# [/docs:retain-basic]


# [docs:retain-with-context]
client.retain(
    bank_id="my-bank",
    content="Alice got promoted to senior engineer",
    context="career update",
    timestamp="2024-03-15T10:00:00Z"
)
# [/docs:retain-with-context]


# [docs:retain-batch]
client.retain_batch(
    bank_id="my-bank",
    items=[
        {"content": "Alice works at Google", "context": "career"},
        {"content": "Bob is a data scientist at Meta", "context": "career"},
        {"content": "Alice and Bob are friends", "context": "relationship"}
    ],
    document_id="conversation_001"
)
# [/docs:retain-batch]


# [docs:retain-async]
# Start async ingestion (returns immediately)
result = client.retain_batch(
    bank_id="my-bank",
    items=[
        {"content": "Large batch item 1"},
        {"content": "Large batch item 2"},
    ],
    document_id="large-doc",
    retain_async=True
)

# Check if it was processed asynchronously
print(result.var_async)  # True
# [/docs:retain-async]


# [docs:retain-with-tags]
# Tag individual items for visibility scoping
client.retain_batch(
    bank_id="my-bank",
    items=[
        {
            "content": "User Alice said she loves the new dashboard",
            "tags": ["user:alice", "feedback"]
        },
        {
            "content": "User Bob reported a bug in the search feature",
            "tags": ["user:bob", "bug-report"]
        }
    ],
    document_id="user_feedback_001"
)
# [/docs:retain-with-tags]


# [docs:retain-with-document-tags]
# Apply tags to all items in a batch
client.retain_batch(
    bank_id="my-bank",
    items=[
        {"content": "Alice mentioned she prefers dark mode"},
        {"content": "Bob asked about keyboard shortcuts"}
    ],
    document_id="support_session_123",
    document_tags=["session:123", "support"]  # Applied to all items
)
# [/docs:retain-with-document-tags]


# =============================================================================
# Cleanup (not shown in docs)
# =============================================================================
requests.delete(f"{HINDSIGHT_URL}/v1/default/banks/my-bank")

print("retain.py: All examples passed")
