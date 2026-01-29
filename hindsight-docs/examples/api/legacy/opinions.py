#!/usr/bin/env python3
"""
Opinions API examples for Hindsight (deprecated - kept for versioned docs).
This file is preserved for backward compatibility with v0.3 documentation.
Opinions have been replaced by Mental Models in v0.4+.
"""
import os
import requests

from hindsight_client import Hindsight

HINDSIGHT_URL = os.getenv("HINDSIGHT_API_URL", "http://localhost:8888")
client = Hindsight(base_url=HINDSIGHT_URL)

# [docs:opinion-form]
# Opinions are automatically formed when the bank encounters
# claims, preferences, or judgments in retained content
client.retain(
    bank_id="my-bank",
    content="I think Python is excellent for data science because of its libraries"
)

# The bank forms an opinion with confidence based on evidence
# [/docs:opinion-form]


# [docs:opinion-search]
# Search for opinions on a topic
response = client.recall(
    bank_id="my-bank",
    query="What do you think about Python?",
    types=["opinion"]
)

for opinion in response.results:
    print(f"Opinion: {opinion.text}")
    print(f"Confidence: {opinion.confidence}")
# [/docs:opinion-search]


# [docs:recall-opinions-only]
# Only retrieve opinions (beliefs and preferences)
opinions = client.recall(
    bank_id="my-bank",
    query="What are my preferences?",
    types=["opinion"]
)
# [/docs:recall-opinions-only]


# [docs:recall-include-entities]
# Include entity summaries in recall results
response = client.recall(
    bank_id="my-bank",
    query="What do I know about Alice?",
    include_entities=True,
    max_entity_tokens=500
)

# Results include both facts and entity summaries
for result in response.results:
    print(f"- {result.text}")
    if hasattr(result, 'entity_summary'):
        print(f"  Entity: {result.entity_summary}")
# [/docs:recall-include-entities]


# [docs:opinion-disposition]
# Bank disposition affects how opinions are formed
# High skepticism = lower confidence, requires more evidence
# Low skepticism = higher confidence, accepts claims more readily

client.create_bank(
    bank_id="skeptical-bank",
    disposition={"skepticism": 5, "literalism": 3, "empathy": 2}
)

# Same content, different confidence due to disposition
client.retain(bank_id="skeptical-bank", content="Python is the best language")
# [/docs:opinion-disposition]


# [docs:opinion-in-reflect]
# Opinions influence reflect responses
response = client.reflect(
    bank_id="my-bank",
    query="Should I use Python for my data project?"
)

# The response incorporates the bank's opinions with appropriate confidence
print(response.text)
# [/docs:opinion-in-reflect]


# =============================================================================
# Cleanup (not shown in docs)
# =============================================================================
requests.delete(f"{HINDSIGHT_URL}/v1/default/banks/my-bank")
requests.delete(f"{HINDSIGHT_URL}/v1/default/banks/skeptical-bank")

print("opinions.py: All examples passed")
