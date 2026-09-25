#!/usr/bin/env python3
"""
Python SDK page examples for Hindsight (docs/sdks/python.mdx).
Run: python examples/api/python-sdk.py
"""
import os

import requests

HINDSIGHT_URL = os.getenv("HINDSIGHT_API_URL", "http://localhost:8888")

# =============================================================================
# Doc Examples
# =============================================================================

# [docs:quickstart]
from hindsight_client import Hindsight

client = Hindsight(base_url="http://localhost:8888")

# Retain a memory
client.retain(bank_id="python-sdk-bank", content="Alice works at Google")

# Recall memories
results = client.recall(bank_id="python-sdk-bank", query="What does Alice do?")
for r in results.results:
    print(r.text)

# Reflect - generate a contextual answer
answer = client.reflect(bank_id="python-sdk-bank", query="Tell me about Alice")
print(answer.text)
# [/docs:quickstart]


# [docs:client-init]
from hindsight_client import Hindsight

client = Hindsight(
    base_url="http://localhost:8888",  # Hindsight API URL
    timeout=30.0,                       # Request timeout in seconds
    # api_key="your-api-key",          # Optional bearer token
)

# Core operations
client.retain(bank_id="python-sdk-test", content="Hello world")
results = client.recall(bank_id="python-sdk-test", query="Hello")

# Bank, mental model, directive and memory helpers
client.create_bank(bank_id="python-sdk-test", name="Test Bank")
models = client.list_mental_models(bank_id="python-sdk-test")
directives = client.list_directives(bank_id="python-sdk-test")
memories = client.list_memories(bank_id="python-sdk-test")
# [/docs:client-init]


# [docs:get-version]
version = client.get_version()

print(version.api_version)

if not version.features.mcp:
    raise RuntimeError("This server does not expose the MCP endpoint")
# [/docs:get-version]


# [docs:retain]
# Simple
client.retain(
    bank_id="python-sdk-bank",
    content="Alice works at Google as a software engineer",
)

# With options
from datetime import datetime

client.retain(
    bank_id="python-sdk-bank",
    content="Alice got promoted",
    context="career update",
    timestamp=datetime(2024, 1, 15),
    document_id="conversation_001",
    metadata={"source": "slack"},
    retain_async=False,  # Set True for background processing
)
# [/docs:retain]


# [docs:retain-batch]
client.retain_batch(
    bank_id="python-sdk-bank",
    items=[
        {"content": "Alice works at Google", "context": "career"},
        {"content": "Bob is a data scientist", "context": "career"},
    ],
    document_id="conversation_001",
    retain_async=False,  # Set True for background processing
)
# [/docs:retain-batch]


# [docs:recall]
# Simple - returns a RecallResponse
results = client.recall(
    bank_id="python-sdk-bank",
    query="What does Alice do?",
)

for r in results.results:
    print(f"{r.text} (type: {r.type})")

# With options
results = client.recall(
    bank_id="python-sdk-bank",
    query="What does Alice do?",
    types=["world", "observation"],  # Filter by fact type
    max_tokens=4096,
    budget="high",  # low, mid, or high
)
# [/docs:recall]


# [docs:recall-chunks]
# Returns RecallResponse with source chunks
response = client.recall(
    bank_id="python-sdk-bank",
    query="What does Alice do?",
    types=["world", "experience"],
    budget="mid",
    max_tokens=4096,
    include_chunks=True,
    max_chunk_tokens=500
)

print(f"Found {len(response.results)} memories")
for r in response.results:
    print(f"  - {r.text}")
    chunk = (response.chunks or {}).get(r.chunk_id)
    if chunk:
        print(f"    Source: {chunk.text[:100]}...")
# [/docs:recall-chunks]


# [docs:reflect]
answer = client.reflect(
    bank_id="python-sdk-bank",
    query="What should I know about Alice?",
    budget="low",  # low, mid, or high
    context="preparing for a meeting",
)

print(answer.text)  # Generated response
# [/docs:reflect]


# [docs:create-bank]
client.create_bank(
    bank_id="python-sdk-bank",
    name="Assistant",
    mission="You're a helpful AI assistant - keep track of user preferences and conversation history.",
    disposition={
        "skepticism": 3,    # 1-5: trusting to skeptical
        "literalism": 3,    # 1-5: flexible to literal
        "empathy": 3,       # 1-5: detached to empathetic
    },
)
# [/docs:create-bank]


# [docs:list-memories]
client.list_memories(
    bank_id="python-sdk-bank",
    type="world",  # Optional: filter by type
    search_query="Alice",  # Optional: text search
    limit=100,
    offset=0,
)
# [/docs:list-memories]

client.close()


# [docs:async]
import asyncio
from hindsight_client import Hindsight

async def main():
    client = Hindsight(base_url="http://localhost:8888")

    # Async retain
    await client.aretain(bank_id="python-sdk-bank", content="Hello world")

    # Async recall
    results = await client.arecall(bank_id="python-sdk-bank", query="Hello")
    for r in results.results:
        print(r.text)

    # Async reflect
    answer = await client.areflect(bank_id="python-sdk-bank", query="What did I say?")
    print(answer.text)

    await client.aclose()

asyncio.run(main())
# [/docs:async]


# [docs:context-manager]
from hindsight_client import Hindsight

with Hindsight(base_url="http://localhost:8888") as client:
    client.retain(bank_id="python-sdk-bank", content="Hello")
    results = client.recall(bank_id="python-sdk-bank", query="Hello")
# Client automatically closed
# [/docs:context-manager]


# =============================================================================
# Cleanup (not shown in docs)
# =============================================================================
requests.delete(f"{HINDSIGHT_URL}/v1/default/banks/python-sdk-bank")
requests.delete(f"{HINDSIGHT_URL}/v1/default/banks/python-sdk-test")

print("python-sdk.py: All examples passed")
