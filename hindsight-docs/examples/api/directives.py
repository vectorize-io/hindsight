#!/usr/bin/env python3
"""
Directives API examples for Hindsight.
Run: python examples/api/directives.py
"""
import os

HINDSIGHT_URL = os.getenv("HINDSIGHT_API_URL", "http://localhost:8888")
BANK_ID = "directives-example-bank"

# =============================================================================
# Setup (not shown in docs)
# =============================================================================
from hindsight_client import Hindsight

client = Hindsight(base_url=HINDSIGHT_URL)

# Create a test bank
client.create_bank(bank_id=BANK_ID, name="Test Bank")

# =============================================================================
# Doc Examples
# =============================================================================

# [docs:create-directive]
# Create a directive (hard rule for reflect)
directive = client.create_directive(
    bank_id=BANK_ID,
    name="Formal Language",
    content="Always respond in formal English, avoiding slang and colloquialisms."
)

print(f"Created directive: {directive.id}")
# [/docs:create-directive]

directive_id = directive.id

# [docs:list-directives]
# List all directives in a bank
directives = client.list_directives(bank_id=BANK_ID)

for d in directives.items:
    print(f"- {d.name}: {d.content[:50]}...")
# [/docs:list-directives]

# [docs:update-directive]
# Update a directive (e.g., disable without deleting)
updated = client.update_directive(
    bank_id=BANK_ID,
    directive_id=directive_id,
    is_active=False
)

print(f"Directive active: {updated.is_active}")
# [/docs:update-directive]

# [docs:delete-directive]
# Delete a directive
client.delete_directive(
    bank_id=BANK_ID,
    directive_id=directive_id
)
# [/docs:delete-directive]

# =============================================================================
# Cleanup (not shown in docs)
# =============================================================================
client.delete_bank(bank_id=BANK_ID)

print("directives.py: All examples passed")
