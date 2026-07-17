"""TealTiger governance memory integration for Hindsight.

Stores AI agent governance decisions with importance-weighted retention
and enables contextual recall of past decisions.

Usage:
    from hindsight_client import Hindsight
    from hindsight_tealtiger import HindsightGovernanceMemory

    client = Hindsight(base_url="http://localhost:8888")
    memory = HindsightGovernanceMemory(client=client, bank_id="governance")

    # Store a governance decision
    memory.store(decision)

    # Recall similar past decisions for context
    past = memory.recall(agent_id="research-agent", context="tool:web_search")
"""

__version__ = "0.1.0"

from hindsight_tealtiger.memory import HindsightGovernanceMemory

__all__ = ["HindsightGovernanceMemory", "__version__"]
