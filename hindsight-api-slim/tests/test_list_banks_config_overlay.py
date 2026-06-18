"""Regression test: list_banks must apply the same disposition + mission
config overlay that get_bank_profile applies.

Bug (reproduced live against 0.8.1): for a bank whose disposition and
mission were evolved/overridden via bank *config* (the banks.config JSONB:
reflect_mission, disposition_skepticism/literalism/empathy), the single-bank
get path returns the real values while the list path returns the stale legacy
DB-column defaults ({skepticism:3, literalism:3, empathy:3} and "").

Root cause: MemoryEngine.get_bank_profile overlays the resolved bank config
on top of the legacy banks.disposition/banks.mission columns, but
MemoryEngine.list_banks returned bank_utils.list_banks rows straight from
those columns with no overlay. The two endpoints disagreed for the same bank.

This test sets disposition + mission through the config path (so the legacy
columns keep their defaults) and asserts list_banks agrees with
get_bank_profile for that bank.

Runs via: uv run pytest tests/test_list_banks_config_overlay.py -v
"""

from __future__ import annotations

import pytest

from hindsight_api.models import RequestContext


@pytest.mark.asyncio
async def test_list_banks_overlays_config_disposition_and_mission(memory):
    bank_id = "list_banks_config_overlay_bank"
    request_context = RequestContext(api_key=None, api_key_id=None, tenant_id=None, internal=False)

    # Values that differ from the 3/3/3 defaults on every trait, and a
    # clearly non-empty mission, so a stale-default regression is unmissable.
    overrides = {
        "reflect_mission": "I am the shared long-term memory for this regression test.",
        "disposition_skepticism": 4,
        "disposition_literalism": 5,
        "disposition_empathy": 2,
    }

    try:
        # Create the bank. Its legacy banks.disposition/banks.mission columns
        # keep their defaults (3/3/3 and "") — the real values live in config.
        await memory.get_bank_profile(bank_id, request_context=request_context)

        # Set disposition + mission via the *config* path (banks.config JSONB),
        # exactly the path that triggered the live bug.
        await memory._config_resolver.update_bank_config(bank_id, overrides, request_context)

        # Source of truth: the single-bank get path already overlays config.
        profile = await memory.get_bank_profile(bank_id, request_context=request_context)
        assert profile["mission"] == overrides["reflect_mission"]
        assert profile["disposition"] == {"skepticism": 4, "literalism": 5, "empathy": 2}

        # The list path must agree with the get path for this bank.
        banks = await memory.list_banks(request_context=request_context)
        entry = next((b for b in banks if b["bank_id"] == bank_id), None)
        assert entry is not None, f"bank {bank_id!r} not present in list_banks output"

        assert entry["mission"] == profile["mission"], (
            "list_banks returned a different mission than get_bank_profile: "
            f"list={entry['mission']!r} get={profile['mission']!r}"
        )
        assert entry["disposition"] == profile["disposition"], (
            "list_banks returned a different disposition than get_bank_profile: "
            f"list={entry['disposition']!r} get={profile['disposition']!r}"
        )
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
