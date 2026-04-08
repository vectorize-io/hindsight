#!/usr/bin/env python3
"""
Bank Templates API examples for Hindsight.
Run: python examples/api/bank-templates.py
"""
import json
import os

import requests

HINDSIGHT_URL = os.getenv("HINDSIGHT_API_URL", "http://localhost:8888")

# =============================================================================
# Doc Examples
# =============================================================================

# [docs:import-template]
template = {
    "version": "1",
    "bank": {
        "retain_mission": "Extract customer issues, resolutions, and sentiment.",
        "enable_observations": True,
        "observations_mission": "Track recurring customer pain points.",
    },
    "mental_models": [
        {
            "id": "sentiment-overview",
            "name": "Customer Sentiment Overview",
            "source_query": "What is the overall sentiment trend?",
            "trigger": {"refresh_after_consolidation": True},
        }
    ],
    "directives": [
        {
            "name": "Acknowledge frustration",
            "content": "Always acknowledge frustration before offering solutions.",
            "priority": 10,
        }
    ],
}

response = requests.post(
    f"{HINDSIGHT_URL}/v1/default/banks/my-bank/import",
    json=template,
)
result = response.json()
print(f"Config applied: {result['config_applied']}")
print(f"Mental models created: {result['mental_models_created']}")
print(f"Directives created: {result['directives_created']}")
# [/docs:import-template]


# [docs:import-dry-run]
response = requests.post(
    f"{HINDSIGHT_URL}/v1/default/banks/my-bank/import",
    params={"dry_run": "true"},
    json=template,
)
result = response.json()
print(f"Dry run: {result['dry_run']}")
print(f"Would apply config: {result['config_applied']}")
# [/docs:import-dry-run]


# [docs:export-template]
response = requests.get(
    f"{HINDSIGHT_URL}/v1/default/banks/my-bank/export"
)
exported = response.json()
print(json.dumps(exported, indent=2))
# [/docs:export-template]


# [docs:export-reimport]
# Export from source bank
response = requests.get(
    f"{HINDSIGHT_URL}/v1/default/banks/source-bank/export"
)
exported = response.json()

# Import into a new bank
response = requests.post(
    f"{HINDSIGHT_URL}/v1/default/banks/new-bank/import",
    json=exported,
)
# [/docs:export-reimport]


# [docs:get-schema]
response = requests.get(
    f"{HINDSIGHT_URL}/v1/bank-template-schema"
)
schema = response.json()
print(json.dumps(schema, indent=2))
# [/docs:get-schema]
