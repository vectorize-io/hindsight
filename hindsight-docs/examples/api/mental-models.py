#!/usr/bin/env python3
"""
Mental Models API examples for Hindsight.
Run: python examples/api/mental-models.py
"""
import os
import time

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
result = client.create_mental_model(
    bank_id=BANK_ID,
    name="Team Communication Preferences",
    source_query="How does the team prefer to communicate?",
    tags=["team", "communication"]
)

# Returns an operation_id - check operations endpoint for completion
print(f"Operation ID: {result.operation_id}")
# [/docs:create-mental-model]

# Wait for the mental model to be created
time.sleep(5)

# [docs:create-mental-model-with-trigger]
# Create a mental model with automatic refresh enabled
result = client.create_mental_model(
    bank_id=BANK_ID,
    name="Project Status",
    source_query="What is the current project status?",
    trigger={"refresh_after_consolidation": True}
)

# This mental model will automatically refresh when observations are updated
print(f"Operation ID: {result.operation_id}")
# [/docs:create-mental-model-with-trigger]

# Wait for the mental model to be created
time.sleep(5)

# [docs:list-mental-models]
# List all mental models in a bank
mental_models = client.list_mental_models(bank_id=BANK_ID)

for mental_model in mental_models.items:
    print(f"- {mental_model.name}: {mental_model.source_query}")
# [/docs:list-mental-models]

# Get the mental model ID for subsequent examples
mental_model_id = mental_models.items[0].id if mental_models.items else None

if mental_model_id:
    # [docs:get-mental-model]
    # Get a specific mental model
    mental_model = client.get_mental_model(
        bank_id=BANK_ID,
        mental_model_id=mental_model_id
    )

    print(f"Name: {mental_model.name}")
    print(f"Content: {mental_model.content}")
    print(f"Last refreshed: {mental_model.last_refreshed_at}")
    # [/docs:get-mental-model]


    # [docs:refresh-mental-model]
    # Refresh a mental model to update with current knowledge
    result = client.refresh_mental_model(
        bank_id=BANK_ID,
        mental_model_id=mental_model_id
    )

    print(f"Refresh operation ID: {result.operation_id}")
    # [/docs:refresh-mental-model]


    # [docs:update-mental-model]
    # Update a mental model's metadata
    updated = client.update_mental_model(
        bank_id=BANK_ID,
        mental_model_id=mental_model_id,
        name="Updated Team Communication Preferences",
        trigger={"refresh_after_consolidation": True}  # Enable auto-refresh
    )

    print(f"Updated name: {updated.name}")
    # [/docs:update-mental-model]


    # [docs:delete-mental-model]
    # Delete a mental model
    client.delete_mental_model(
        bank_id=BANK_ID,
        mental_model_id=mental_model_id
    )
    # [/docs:delete-mental-model]

    # [docs:get-mental-model-history]
    # Get the change history of a mental model
    history = client.get_mental_model_history(
        bank_id=BANK_ID,
        mental_model_id=mental_model_id
    )

    for entry in history:
        print(f"Changed at: {entry['changed_at']}")
        print(f"Previous content: {entry['previous_content']}")
    # [/docs:get-mental-model-history]


# =============================================================================
# Cleanup (not shown in docs)
# =============================================================================
client.delete_bank(bank_id=BANK_ID)

print("mental-models.py: All examples passed")
