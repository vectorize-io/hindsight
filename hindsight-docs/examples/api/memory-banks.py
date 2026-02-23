#!/usr/bin/env python3
"""
Memory Banks API examples for Hindsight.
Run: python examples/api/memory-banks.py
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
# Doc Examples
# =============================================================================

# [docs:create-bank]
client.create_bank(bank_id="my-bank")
# [/docs:create-bank]


# [docs:bank-with-disposition]
client.create_bank(bank_id="architect-bank")
client.update_bank_config(
    "architect-bank",
    reflect_mission="You're a senior software architect - keep track of system designs, "
            "technology decisions, and architectural patterns. Prefer simplicity over cutting-edge.",
    disposition_skepticism=4,   # Questions new technologies
    disposition_literalism=4,   # Focuses on concrete specs
    disposition_empathy=2,      # Prioritizes technical facts
)
# [/docs:bank-with-disposition]


# [docs:update-bank-config]
client.update_bank_config(
    "my-bank",
    retain_mission="Always include technical decisions, API design choices, and architectural trade-offs. Ignore meeting logistics and social exchanges.",
    retain_extraction_mode="verbose",
    observations_mission="Observations are stable facts about people and projects. Always include preferences, skills, and recurring patterns. Ignore one-off events.",
    disposition_skepticism=4,
    disposition_literalism=4,
    disposition_empathy=2,
)
# [/docs:update-bank-config]


# [docs:get-bank-config]
# Returns resolved config (server defaults merged with bank overrides) and the raw overrides
data = client.get_bank_config("my-bank")
# data["config"]     — full resolved configuration
# data["overrides"]  — only fields overridden at the bank level
# [/docs:get-bank-config]


# [docs:reset-bank-config]
# Remove all bank-level overrides, reverting to server defaults
client.reset_bank_config("my-bank")
# [/docs:reset-bank-config]


# =============================================================================
# Cleanup (not shown in docs)
# =============================================================================
requests.delete(f"{HINDSIGHT_URL}/v1/default/banks/my-bank")
requests.delete(f"{HINDSIGHT_URL}/v1/default/banks/architect-bank")

print("memory-banks.py: All examples passed")
