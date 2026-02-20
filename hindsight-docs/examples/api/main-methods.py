#!/usr/bin/env python3
"""
Main Methods overview examples for Hindsight.
Run: python examples/api/main-methods.py
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
# Doc Examples - Retain Section
# =============================================================================

# [docs:main-retain]
# Store a single fact
client.retain(
    bank_id="my-bank",
    content="Alice joined Google in March 2024 as a Senior ML Engineer"
)

# Store a conversation
conversation = """
User: What did you work on today?
Assistant: I reviewed the new ML pipeline architecture.
User: How did it look?
Assistant: Promising, but needs better error handling.
"""

client.retain(
    bank_id="my-bank",
    content=conversation,
    context="Daily standup conversation"
)

# Batch retain multiple items
client.retain_batch(
    bank_id="my-bank",
    items=[
        {"content": "Bob prefers Python for data science"},
        {"content": "Alice recommends using pytest for testing"},
        {"content": "The team uses GitHub for code reviews"}
    ]
)
# [/docs:main-retain]


# =============================================================================
# Doc Examples - Recall Section
# =============================================================================

# [docs:main-recall]
# Basic search
results = client.recall(
    bank_id="my-bank",
    query="What does Alice do at Google?"
)

for result in results.results:
    print(f"- {result.text}")

# Search with options
results = client.recall(
    bank_id="my-bank",
    query="What happened last spring?",
    budget="high",  # More thorough graph traversal
    max_tokens=8192,  # Return more context
    types=["world"]  # Only world facts
)

# Include source chunks for more context
results = client.recall(
    bank_id="my-bank",
    query="Tell me about Alice",
    include_chunks=True,
    max_chunk_tokens=500
)

# Check chunk details (chunks are on response level, keyed by memory ID)
for result in results.results:
    print(f"Memory: {result.text}")
    if results.chunks and result.id in results.chunks:
        chunk = results.chunks[result.id]
        print(f"  Source: {chunk.text[:100]}...")
# [/docs:main-recall]


# =============================================================================
# Doc Examples - Reflect Section
# =============================================================================

# [docs:main-reflect]
# Basic reflect
response = client.reflect(
    bank_id="my-bank",
    query="Should we adopt TypeScript for our backend?",
    include_facts=True,
)

print(response.text)
print("\nBased on:", len(response.based_on.memories if response.based_on else []), "facts")

# Reflect with options
response = client.reflect(
    bank_id="my-bank",
    query="What are Alice's strengths for the team lead role?",
    budget="high",  # More thorough reasoning
    include_facts=True,
)

# See which facts influenced the response
for fact in (response.based_on.memories if response.based_on else []):
    print(f"- {fact.text}")
# [/docs:main-reflect]


# =============================================================================
# Doc Examples - List Memories Section
# =============================================================================

# [docs:main-list-memories]
# List all memories in a bank
memories = client.list_memories(
    bank_id="my-bank",
    limit=10
)

for memory in memories.items:
    print(f"- [{memory['fact_type']}] {memory['text']}")

# Filter by type
world_facts = client.list_memories(
    bank_id="my-bank",
    type="world",
    limit=5
)

# Search within memories
search_results = client.list_memories(
    bank_id="my-bank",
    search_query="Alice",
    limit=10
)
# [/docs:main-list-memories]


# =============================================================================
# Doc Examples - Async Methods Section
# =============================================================================

# [docs:main-async]
import asyncio

async def async_example():
    # Create a fresh client for async operations
    async_client = Hindsight(base_url=HINDSIGHT_URL)

    # All sync methods have async versions prefixed with 'a'
    await async_client.aretain(bank_id="my-bank", content="Async memory")

    results = await async_client.arecall(bank_id="my-bank", query="Async")
    for r in results:
        print(f"- {r.text}")

    response = await async_client.areflect(bank_id="my-bank", query="What was stored?")
    print(response.text)

asyncio.run(async_example())
# [/docs:main-async]


# =============================================================================
# Cleanup (not shown in docs)
# =============================================================================
requests.delete(f"{HINDSIGHT_URL}/v1/default/banks/my-bank")

print("main-methods.py: All examples passed")
