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
    query="We're considering a hybrid work policy. What do you think about remote work?",
    budget="mid",
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
    mission="I am a risk-aware financial advisor",
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
# include_facts=True enables the based_on field in the response
response = client.reflect(
    bank_id="my-bank",
    query="Tell me about Alice",
    include_facts=True,
)

print("Response:", response.text)
print("\nBased on:")
for fact in (response.based_on.memories if response.based_on else []):
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


# [docs:reflect-structured-output]
from pydantic import BaseModel

# Define your response structure with Pydantic
class HiringRecommendation(BaseModel):
    recommendation: str
    confidence: str  # "low", "medium", "high"
    key_factors: list[str]
    risks: list[str] = []

response = client.reflect(
    bank_id="hiring-team",
    query="Should we hire Alice for the ML team lead position?",
    response_schema=HiringRecommendation.model_json_schema(),
)

# Parse structured output into Pydantic model
result = HiringRecommendation.model_validate(response.structured_output)
print(f"Recommendation: {result.recommendation}")
print(f"Confidence: {result.confidence}")
print(f"Key factors: {result.key_factors}")
# [/docs:reflect-structured-output]


# =============================================================================
# Cleanup (not shown in docs)
# =============================================================================
requests.delete(f"{HINDSIGHT_URL}/v1/default/banks/my-bank")
requests.delete(f"{HINDSIGHT_URL}/v1/default/banks/cautious-advisor")

print("reflect.py: All examples passed")
