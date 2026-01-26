#!/usr/bin/env python3
"""
Mental Models API examples for Hindsight.
Run: python examples/api/mental-models.py
"""
import os
import time
import requests

HINDSIGHT_URL = os.getenv("HINDSIGHT_API_URL", "http://localhost:8888")
BANK_ID = "mental-models-demo-bank"

# =============================================================================
# Setup (not shown in docs)
# =============================================================================
from hindsight_client import Hindsight

client = Hindsight(base_url=HINDSIGHT_URL)

# Create bank and seed some data
client.create_bank(bank_id=BANK_ID, name="Mental Models Demo")
client.retain(bank_id=BANK_ID, content="The team prefers async communication via Slack")
client.retain(bank_id=BANK_ID, content="For urgent issues, use the #incidents channel")
client.retain(bank_id=BANK_ID, content="Weekly syncs happen every Monday at 10am")

# Wait for data to be processed
time.sleep(2)

# =============================================================================
# Doc Examples
# =============================================================================

# [docs:create-mental-model]
# Create a mental model (runs reflect in background)
response = requests.post(
    f"{HINDSIGHT_URL}/v1/default/banks/{BANK_ID}/mental-models",
    json={
        "name": "Team Communication Preferences",
        "source_query": "How does the team prefer to communicate?",
        "tags": ["team", "communication"]
    }
)
result = response.json()

# Returns an operation_id - check operations endpoint for completion
print(f"Operation ID: {result['operation_id']}")
# [/docs:create-mental-model]

# Wait for the mental model to be created
time.sleep(5)

# [docs:list-mental-models]
# List all mental models in a bank
response = requests.get(f"{HINDSIGHT_URL}/v1/default/banks/{BANK_ID}/mental-models")
mental_models = response.json()

for mental_model in mental_models["items"]:
    print(f"- {mental_model['name']}: {mental_model['source_query']}")
# [/docs:list-mental-models]

# Get the mental model ID for subsequent examples
mental_model_id = mental_models["items"][0]["id"] if mental_models["items"] else None

if mental_model_id:
    # [docs:get-mental-model]
    # Get a specific mental model
    response = requests.get(
        f"{HINDSIGHT_URL}/v1/default/banks/{BANK_ID}/mental-models/{mental_model_id}"
    )
    mental_model = response.json()

    print(f"Name: {mental_model['name']}")
    print(f"Content: {mental_model['content']}")
    print(f"Last refreshed: {mental_model['last_refreshed_at']}")
    # [/docs:get-mental-model]


    # [docs:refresh-mental-model]
    # Refresh a mental model to update with current knowledge
    response = requests.post(
        f"{HINDSIGHT_URL}/v1/default/banks/{BANK_ID}/mental-models/{mental_model_id}/refresh"
    )
    result = response.json()

    print(f"Refresh operation ID: {result['operation_id']}")
    # [/docs:refresh-mental-model]


    # [docs:update-mental-model]
    # Update a mental model's name
    response = requests.patch(
        f"{HINDSIGHT_URL}/v1/default/banks/{BANK_ID}/mental-models/{mental_model_id}",
        json={"name": "Updated Team Communication Preferences"}
    )
    updated = response.json()

    print(f"Updated name: {updated['name']}")
    # [/docs:update-mental-model]


    # [docs:delete-mental-model]
    # Delete a mental model
    requests.delete(
        f"{HINDSIGHT_URL}/v1/default/banks/{BANK_ID}/mental-models/{mental_model_id}"
    )
    # [/docs:delete-mental-model]


# =============================================================================
# Cleanup (not shown in docs)
# =============================================================================
requests.delete(f"{HINDSIGHT_URL}/v1/default/banks/{BANK_ID}")

print("mental-models.py: All examples passed")
