#!/usr/bin/env python3
"""
Recall API examples for Hindsight.
Run: python examples/api/recall.py
"""
import os
import requests

HINDSIGHT_URL = os.getenv("HINDSIGHT_API_URL", "http://localhost:8888")

# =============================================================================
# Setup (not shown in docs)
# =============================================================================
from hindsight_client import Hindsight

client = Hindsight(base_url=HINDSIGHT_URL)

# Seed some data for recall examples
client.retain(bank_id="my-bank", content="Alice works at Google as a software engineer")
client.retain(bank_id="my-bank", content="Alice loves hiking on weekends")
client.retain(bank_id="my-bank", content="Bob is a data scientist who works with Alice")

# =============================================================================
# Doc Examples
# =============================================================================

# [docs:recall-basic]
response = client.recall(bank_id="my-bank", query="What does Alice do?")
for r in response.results:
    print(f"- {r.text}")
# [/docs:recall-basic]


# [docs:recall-with-options]
response = client.recall(
    bank_id="my-bank",
    query="What does Alice do?",
    types=["world", "experience"],
    budget="high",
    max_tokens=8000,
    trace=True,
)

# Access results
for r in response.results:
    print(f"- {r.text}")
# [/docs:recall-with-options]


# [docs:recall-world-only]
# Only world facts (objective information)
world_facts = client.recall(
    bank_id="my-bank",
    query="Where does Alice work?",
    types=["world"]
)
# [/docs:recall-world-only]


# [docs:recall-experience-only]
# Only experience (conversations and events)
experience = client.recall(
    bank_id="my-bank",
    query="What have I recommended?",
    types=["experience"]
)
# [/docs:recall-experience-only]


# [docs:recall-observations-only]
# Only observations (consolidated knowledge)
observations = client.recall(
    bank_id="my-bank",
    query="What patterns have I learned?",
    types=["observation"]
)
# [/docs:recall-observations-only]


# [docs:recall-with-observations]
# Include observations in recall
results = client.recall(
    bank_id="my-bank",
    query="What programming languages does Alice prefer?",
    types=["world", "experience", "observation"]
)

# Observations only
observations = client.recall(
    bank_id="my-bank",
    query="What patterns have I learned?",
    types=["observation"]
)
# [/docs:recall-with-observations]


# [docs:recall-token-budget]
# Fill up to 4K tokens of context with relevant memories
results = client.recall(bank_id="my-bank", query="What do I know about Alice?", max_tokens=4096)

# Smaller budget for quick lookups
results = client.recall(bank_id="my-bank", query="Alice's email", max_tokens=500)
# [/docs:recall-token-budget]




# [docs:recall-budget-levels]
# Quick lookup
results = client.recall(bank_id="my-bank", query="Alice's email", budget="low")

# Deep exploration
results = client.recall(bank_id="my-bank", query="How are Alice and Bob connected?", budget="high")
# [/docs:recall-budget-levels]


# [docs:recall-with-tags]
# Filter recall to only memories tagged for a specific user
response = client.recall(
    bank_id="my-bank",
    query="What feedback did the user give?",
    tags=["user:alice"],
    tags_match="any"  # OR matching, includes untagged (default)
)
# [/docs:recall-with-tags]


# [docs:recall-tags-strict]
# Strict mode: only return memories that have matching tags (exclude untagged)
response = client.recall(
    bank_id="my-bank",
    query="What did the user say?",
    tags=["user:alice"],
    tags_match="any_strict"  # OR matching, excludes untagged memories
)
# [/docs:recall-tags-strict]


# [docs:recall-tags-all]
# AND matching: require ALL specified tags to be present
response = client.recall(
    bank_id="my-bank",
    query="What bugs were reported?",
    tags=["user:alice", "bug-report"],
    tags_match="all_strict"  # Memory must have BOTH tags
)
# [/docs:recall-tags-all]


# =============================================================================
# Legacy snippets for v0.3 docs (kept for backward compatibility)
# =============================================================================

# [docs:recall-opinions-only]
# Legacy: opinions replaced by observations in v0.4+
# Only retrieve opinions (beliefs and preferences)
opinions = client.recall(
    bank_id="my-bank",
    query="What are my preferences?",
    types=["opinion"]
)
# [/docs:recall-opinions-only]


# [docs:recall-include-entities]
# Legacy: entity summaries replaced by observations in v0.4+
# Include entity summaries in recall results
response = client.recall(
    bank_id="my-bank",
    query="What do I know about Alice?",
    include_entities=True,
    max_entity_tokens=500
)
# [/docs:recall-include-entities]


# =============================================================================
# Cleanup (not shown in docs)
# =============================================================================
requests.delete(f"{HINDSIGHT_URL}/v1/default/banks/my-bank")

print("recall.py: All examples passed")
