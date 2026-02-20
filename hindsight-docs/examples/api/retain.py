#!/usr/bin/env python3
"""
Retain API examples for Hindsight.
Run: python examples/api/retain.py
"""
import os
import requests
from pathlib import Path

HINDSIGHT_URL = os.getenv("HINDSIGHT_API_URL", "http://localhost:8888")
EXAMPLES_DIR = Path(__file__).parent

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


# [docs:retain-conversation]
# Retain an entire conversation as a single document.
# Format each message as "Name (timestamp): text" so the LLM can attribute
# facts to the right person and resolve temporal references across the thread.
conversation = "\n".join([
    "Alice (2024-03-15T09:00:00Z): Hi Bob! Did you end up going to the doctor last week?",
    "Bob (2024-03-15T09:01:00Z): Yes, finally. Turns out I have a mild peanut allergy.",
    "Alice (2024-03-15T09:02:00Z): Oh no! Are you okay?",
    "Bob (2024-03-15T09:03:00Z): Yeah, nothing serious. Just need to carry an antihistamine.",
    "Alice (2024-03-15T09:04:00Z): Good to know. We'll avoid peanuts at the team lunch.",
])

client.retain(
    bank_id="my-bank",
    content=conversation,
    context="team chat",
    timestamp="2024-03-15T09:04:00Z",
    document_id="chat-2024-03-15-alice-bob",
)
# [/docs:retain-conversation]


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
        {"content": "Alice works at Google", "context": "career", "document_id": "conversation_001_msg_1"},
        {"content": "Bob is a data scientist at Meta", "context": "career", "document_id": "conversation_001_msg_2"},
        {"content": "Alice and Bob are friends", "context": "relationship", "document_id": "conversation_001_msg_3"}
    ]
)
# [/docs:retain-batch]


# [docs:retain-async]
# Start async ingestion (returns immediately)
result = client.retain_batch(
    bank_id="my-bank",
    items=[
        {"content": "Large batch item 1", "document_id": "large-doc-1"},
        {"content": "Large batch item 2", "document_id": "large-doc-2"},
    ],
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
            "tags": ["user:alice", "feedback"],
            "document_id": "user_feedback_001"
        },
        {
            "content": "User Bob reported a bug in the search feature",
            "tags": ["user:bob", "bug-report"],
            "document_id": "user_feedback_002"
        }
    ]
)
# [/docs:retain-with-tags]


# [docs:retain-with-document-tags]
# Apply tags to all items in a batch
client.retain_batch(
    bank_id="my-bank",
    items=[
        {"content": "Alice mentioned she prefers dark mode", "document_id": "support_session_123_msg_1"},
        {"content": "Bob asked about keyboard shortcuts", "document_id": "support_session_123_msg_2"}
    ],
    document_tags=["session:123", "support"]  # Applied to all items
)
# [/docs:retain-with-document-tags]


# [docs:retain-files]
# Upload files and retain their contents as memories.
# Supports: PDF, DOCX, PPTX, XLSX, images (OCR), audio (transcription), and text formats.
# Pass file paths — the client reads and uploads them automatically.
result = client.retain_files(
    bank_id="my-bank",
    files=[EXAMPLES_DIR / "sample.pdf"],
    context="quarterly report",
)
print(result.operation_ids)  # Track processing via the operations endpoint
# [/docs:retain-files]


# [docs:retain-files-batch]
# Upload multiple files with per-file metadata (up to 10 files per batch).
# Each file gets its own context, document_id, and tags.
result = client.retain_files(
    bank_id="my-bank",
    files=[
        EXAMPLES_DIR / "sample.pdf",
        EXAMPLES_DIR / "sample.pdf",  # Replace with a second file path
    ],
    files_metadata=[
        {"context": "quarterly report", "document_id": "q1-report", "tags": ["project:alpha"]},
        {"context": "meeting notes", "document_id": "q1-notes", "tags": ["project:alpha"]},
    ],
)
print(result.operation_ids)  # One operation ID per file
# [/docs:retain-files-batch]


# [docs:retain-list-tags]
# List all tags in a bank
response = requests.get(f"{HINDSIGHT_URL}/v1/default/banks/my-bank/tags")
tags = response.json()
for tag in tags["items"]:
    print(f"{tag['tag']}: {tag['count']} memories")

# Search with wildcards (* matches any characters)
response = requests.get(f"{HINDSIGHT_URL}/v1/default/banks/my-bank/tags", params={"q": "user:*"})
user_tags = response.json()
response = requests.get(f"{HINDSIGHT_URL}/v1/default/banks/my-bank/tags", params={"q": "*-admin"})
admin_tags = response.json()
# [/docs:retain-list-tags]


# =============================================================================
# Cleanup (not shown in docs)
# =============================================================================
requests.delete(f"{HINDSIGHT_URL}/v1/default/banks/my-bank")

print("retain.py: All examples passed")
