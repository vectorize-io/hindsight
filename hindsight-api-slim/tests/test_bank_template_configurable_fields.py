"""Verify that BankTemplateConfig exposes every hierarchical field that
_CONFIGURABLE_FIELDS already accepts at the engine layer.

This test guards the fix for the gap described in the upstream PR title
"fix(bank-template): align BankTemplateConfig with _CONFIGURABLE_FIELDS".
Each new field is POSTed through /v1/default/banks/{id}/import and then
read back via the bank-config endpoint; assertion is that the applied
value round-trips through the engine.

Runs via: uv run pytest tests/test_bank_template_configurable_fields.py -v

The api_client fixture (shared with tests/test_bank_templates.py) wraps
create_app(memory, initialize_memory=False) in an httpx.ASGITransport
with base_url http://test — in-process, no network, no tenant extension.
Copy the fixture inline here so the test file does not depend on a
conftest we do not ship in the patch.
"""

from __future__ import annotations

from datetime import datetime

import httpx
import pytest
import pytest_asyncio

from hindsight_api.api import create_app
from hindsight_api.api.http import BankTemplateConfig

# Each tuple is (field_name, applied_value). Values chosen to differ
# visibly from defaults so round-trip bugs surface.
NEW_FIELDS: list[tuple[str, object]] = [
    ("retain_default_strategy", "strategy-a"),
    ("retain_strategies", {"strategy-a": {"mode": "concise", "max_tokens": 512}}),
    ("retain_chunk_batch_size", 7),
    ("mcp_enabled_tools", ["list_banks", "get_bank_profile"]),
    ("consolidation_llm_batch_size", 11),
    ("consolidation_source_facts_max_tokens", 2048),
    ("consolidation_source_facts_max_tokens_per_observation", 256),
    ("max_observations_per_scope", 13),
    ("reflect_source_facts_max_tokens", 4096),
    ("llm_gemini_safety_settings", [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}]),
    ("recall_budget_function", "adaptive"),
    ("recall_budget_fixed_low", 50),
    ("recall_budget_fixed_mid", 250),
    ("recall_budget_fixed_high", 800),
    ("recall_budget_adaptive_low", 0.05),
    ("recall_budget_adaptive_mid", 0.1),
    ("recall_budget_adaptive_high", 0.4),
    ("recall_budget_min", 30),
    ("recall_budget_max", 1500),
]


@pytest_asyncio.fixture
async def api_client(memory):
    """Matches the fixture in tests/test_bank_templates.py — in-process
    ASGI test client, no tenant extension, no auth."""
    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def bank_id():
    return f"tmpl_config_{datetime.now().timestamp()}"


def test_bank_template_config_declares_every_configurable_field():
    """Pydantic-level guard: every field in NEW_FIELDS must be a declared
    attribute of BankTemplateConfig so get_config_updates() picks it up."""
    declared = set(BankTemplateConfig.model_fields.keys())
    missing = [name for name, _ in NEW_FIELDS if name not in declared]
    assert not missing, f"BankTemplateConfig missing fields: {missing}"


@pytest.mark.asyncio
@pytest.mark.parametrize("field_name,applied_value", NEW_FIELDS, ids=[n for n, _ in NEW_FIELDS])
async def test_new_field_round_trips_through_import(
    api_client: httpx.AsyncClient,
    bank_id: str,
    field_name: str,
    applied_value: object,
):
    """POST a minimal manifest with one new field set, then read bank
    config back and assert the value made it through.

    Bank config response shape per upstream's test_import_applies_config:
    top-level keys are resolved hierarchical config; per-bank overrides
    live under config["overrides"][<field>]. Assert on the override slot.
    """
    unique_bank_id = f"{bank_id}_{field_name}"
    manifest = {
        "version": "1",
        "bank": {field_name: applied_value},
    }

    resp = await api_client.post(
        f"/v1/default/banks/{unique_bank_id}/import",
        json=manifest,
    )
    assert resp.status_code == 200, resp.text

    # Read bank config back — field must reflect the applied value
    # under the "overrides" slot, matching upstream's own test shape.
    read = await api_client.get(f"/v1/default/banks/{unique_bank_id}/config")
    assert read.status_code == 200, read.text
    config = read.json()
    overrides = config.get("overrides", {})
    assert overrides.get(field_name) == applied_value, (
        f"round-trip mismatch for {field_name}: "
        f"sent {applied_value!r}, got {overrides.get(field_name)!r} "
        f"(full overrides: {overrides!r})"
    )
