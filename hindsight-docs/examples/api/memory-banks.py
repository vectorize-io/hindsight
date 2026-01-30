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
client.create_bank(
    bank_id="my-bank",
    name="Research Assistant",
    mission="You're a research assistant specializing in machine learning - keep track of papers, methods, and findings.",
    disposition={
        "skepticism": 4,
        "literalism": 3,
        "empathy": 3
    }
)
# [/docs:create-bank]


# [docs:bank-mission]
client.create_bank(
    bank_id="financial-advisor",
    name="Financial Advisor",
    mission="""You're a conservative financial advisor - keep track of client risk tolerance,
    investment preferences, and market conditions. Prioritize capital preservation over growth."""
)
# [/docs:bank-mission]


# [docs:bank-background]
# Legacy snippet for v0.3 docs (background renamed to mission in v0.4)
client.create_bank(
    bank_id="legacy-bank",
    name="Legacy Example",
    mission="""I'm a personal assistant helping a software engineer. I should track their
    project preferences, coding style, and technology choices."""
)
# [/docs:bank-background]


# [docs:bank-with-disposition]
client.create_bank(
    bank_id="architect-bank",
    mission="You're a senior software architect - keep track of system designs, "
            "technology decisions, and architectural patterns. Prefer simplicity over cutting-edge.",
    disposition={
        "skepticism": 4,   # Questions new technologies
        "literalism": 4,   # Focuses on concrete specs
        "empathy": 2       # Prioritizes technical facts
    }
)
# [/docs:bank-with-disposition]


# [docs:bank-support-agent]
client.create_bank(
    bank_id="support-agent",
    mission="You're a customer support agent - keep track of "
            "customer preferences, past issues, and communication styles."
)
# [/docs:bank-support-agent]


# =============================================================================
# Cleanup (not shown in docs)
# =============================================================================
requests.delete(f"{HINDSIGHT_URL}/v1/default/banks/my-bank")
requests.delete(f"{HINDSIGHT_URL}/v1/default/banks/financial-advisor")
requests.delete(f"{HINDSIGHT_URL}/v1/default/banks/architect-bank")
requests.delete(f"{HINDSIGHT_URL}/v1/default/banks/support-agent")

print("memory-banks.py: All examples passed")
