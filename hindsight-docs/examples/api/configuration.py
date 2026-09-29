#!/usr/bin/env python3
"""
Configuration examples for Hindsight (per-call retain strategies).
Run: python examples/api/configuration.py
"""
import os

from hindsight_client import Hindsight

HINDSIGHT_URL = os.getenv("HINDSIGHT_API_URL", "http://localhost:8888")

# =============================================================================
# Setup (not shown in docs)
# =============================================================================
client = Hindsight(base_url=HINDSIGHT_URL)
bank_id = "strategy-demo-bank"
client.create_bank(bank_id=bank_id)
client.update_bank_config(
    bank_id,
    retain_default_strategy="conversations",
    retain_strategies={
        "conversations": {
            "retain_extraction_mode": "concise",
            "retain_chunk_size": 3000,
            "retain_structured_chunk_size": 12000,
        },
        "documents": {
            "retain_extraction_mode": "chunks",
            "retain_chunk_size": 800,
            "entity_labels": None,
            "entities_allow_free_form": False,
        },
    },
)

# =============================================================================
# Doc Examples
# =============================================================================

# [docs:retain-strategy]
# Uses default strategy ("conversations")
client.retain_batch(bank_id, items=[{"content": "Alice joined the team today"}])

# Explicitly use document strategy
client.retain_batch(bank_id, items=[{"content": "...document text...", "strategy": "documents"}])
# [/docs:retain-strategy]

# =============================================================================
# Cleanup (not shown in docs)
# =============================================================================
client.delete_bank(bank_id)
client.close()

print("configuration.py: All examples passed")
