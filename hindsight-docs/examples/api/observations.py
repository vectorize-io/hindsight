#!/usr/bin/env python3
"""
Observations (consolidation) examples for Hindsight.
Run: python examples/api/observations.py
"""
import asyncio
import os

from hindsight_client import Hindsight
from hindsight_client_api.models.consolidation_request import ConsolidationRequest

HINDSIGHT_URL = os.getenv("HINDSIGHT_API_URL", "http://localhost:8888")
BANK_ID = "observations-demo-bank"


async def main():
    # =========================================================================
    # Setup (not shown in docs)
    # =========================================================================
    client = Hindsight(base_url=HINDSIGHT_URL)
    await client.acreate_bank(bank_id=BANK_ID)
    await client.aretain(bank_id=BANK_ID, content="Alice prefers dark mode.", tags=["user:alice"])
    await client.aretain(bank_id=BANK_ID, content="The engineering team ships on Fridays.", tags=["team:engineering"])

    # =========================================================================
    # Doc Examples
    # =========================================================================

    # [docs:targeted-consolidation]
    # Consolidate only memories tagged with user:alice
    await client.banks.trigger_consolidation(
        bank_id=BANK_ID,
        consolidation_request=ConsolidationRequest(observation_scopes=[["user:alice"]]),
    )

    # Consolidate memories for alice OR the engineering team
    await client.banks.trigger_consolidation(
        bank_id=BANK_ID,
        consolidation_request=ConsolidationRequest(
            observation_scopes=[["user:alice"], ["team:engineering"]]
        ),
    )
    # [/docs:targeted-consolidation]

    # [docs:clear-observations]
    # Clear all observations for a bank
    await client.banks.clear_observations(bank_id=BANK_ID)
    # [/docs:clear-observations]

    # =========================================================================
    # Cleanup (not shown in docs)
    # =========================================================================
    await client.adelete_bank(bank_id=BANK_ID)
    await client.aclose()
    print("observations.py: All examples passed")


asyncio.run(main())
