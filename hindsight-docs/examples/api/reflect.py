#!/usr/bin/env python3
"""
Reflect API examples for Hindsight.
Run: python examples/api/reflect.py
"""
import os
import requests

HINDSIGHT_URL = os.getenv("HINDSIGHT_API_URL", "http://localhost:8888")

# =============================================================================
# Setup (not shown in docs)
# =============================================================================
from hindsight_client import Hindsight

client = Hindsight(base_url=HINDSIGHT_URL)

# Seed some data for reflect examples
client.retain(bank_id="my-bank", content="Alice works at Google as a software engineer")
client.retain(bank_id="my-bank", content="Alice has been working there for 5 years")
client.retain(bank_id="my-bank", content="Alice recently got promoted to senior engineer")

# =============================================================================
# Doc Examples
# =============================================================================

# [docs:reflect-basic]
client.reflect(bank_id="my-bank", query="What should I know about Alice?")
# [/docs:reflect-basic]


# [docs:reflect-with-params]
response = client.reflect(
    bank_id="my-bank",
    query="What do you think about remote work?",
    budget="mid",
    context="We're considering a hybrid work policy"
)
# [/docs:reflect-with-params]


# [docs:reflect-with-context]
# Context is passed to the LLM to help it understand the situation
response = client.reflect(
    bank_id="my-bank",
    query="What do you think about the proposal?",
    context="We're in a budget review meeting discussing Q4 spending"
)
# [/docs:reflect-with-context]


# [docs:reflect-disposition]
# Create a bank with specific disposition
client.create_bank(
    bank_id="cautious-advisor",
    name="Cautious Advisor",
    background="I am a risk-aware financial advisor",
    disposition={
        "skepticism": 5,   # Very skeptical of claims
        "literalism": 4,   # Focuses on exact requirements
        "empathy": 2       # Prioritizes facts over feelings
    }
)

# Reflect responses will reflect this disposition
response = client.reflect(
    bank_id="cautious-advisor",
    query="Should I invest in crypto?"
)
# Response will likely emphasize risks and caution
# [/docs:reflect-disposition]


# [docs:reflect-sources]
response = client.reflect(bank_id="my-bank", query="Tell me about Alice")

print("Response:", response.text)
print("\nBased on:")
for fact in response.based_on or []:
    print(f"  - [{fact.type}] {fact.text}")
# [/docs:reflect-sources]


# [docs:reflect-with-tags]
# Filter reflection to only consider memories for a specific user
response = client.reflect(
    bank_id="my-bank",
    query="What does this user think about our product?",
    tags=["user:alice"],
    tags_match="any_strict"  # Only use memories tagged for this user
)
# [/docs:reflect-with-tags]


# =============================================================================
# Cleanup (not shown in docs)
# =============================================================================
requests.delete(f"{HINDSIGHT_URL}/v1/default/banks/my-bank")
requests.delete(f"{HINDSIGHT_URL}/v1/default/banks/cautious-advisor")

print("reflect.py: All examples passed")
