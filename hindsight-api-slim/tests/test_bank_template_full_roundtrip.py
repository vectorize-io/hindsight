"""Every exportable bank field survives a full export -> import round-trip.

The existing coverage leaves a gap in the middle:

* ``test_bank_template_configurable_fields`` sets fields **one at a time** and
  never exports — it proves import writes a field, not that export reads it back.
* ``TestExport::test_export_reimport_roundtrip`` does export then import, but
  with a single config field, and it asserts only the response flags
  (``config_applied is True``) — never that any value survived.

So a field that import accepts but export drops (or that export reshapes into
something import rejects) passes both. This module closes that: it sets *every*
field ``BankTemplateConfig`` declares on one bank, exports it, imports the
exported manifest into a fresh bank, and asserts the second bank's overrides
match the first's.

``BankTemplateConfig.model_fields`` is the exportable surface — the export
endpoint filters bank overrides through exactly that set — so the sample table
below is checked against it. A newly added template field fails
``test_sample_values_cover_every_exportable_field`` until it gets a value here,
which is the point: the round-trip must not silently stop covering it.

Adding a per-bank config field is a multi-step flow, and each step here fails
until the previous one is done — so a half-wired field cannot land quietly:

1. add it to ``_CONFIGURABLE_FIELDS`` → ``test_every_configurable_field_is_exportable``
   fails until it is declared on ``BankTemplateConfig``;
2. declare it there → ``test_sample_values_cover_every_exportable_field`` fails
   until it has a value in ``_SAMPLE_VALUES``;
3. give it a value → the round-trip below actually exercises it end to end;
4. changing ``BankTemplateConfig`` also moves the OpenAPI spec, the generated
   clients and ``bank-template-schema.json``, so CI's ``verify-generated-files``
   fails until those are regenerated.

(``test_bank_config_value_types`` adds a fifth: every configurable field must
have a derivable type contract.) See #3218 for what a half-wired config field
costs in production.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx
import pytest
import pytest_asyncio

from hindsight_api.api import create_app
from hindsight_api.api.http import BankTemplateConfig
from hindsight_api.config import HindsightConfig


# One value per BankTemplateConfig field, each chosen to differ visibly from the
# server default so a value that silently reverts is caught rather than matching
# by luck. Cross-field constraints enforced by validate_bank_template() and the
# config validators are respected:
#   * retain_custom_instructions requires retain_extraction_mode == "custom"
#   * retain_default_strategy names a key in retain_strategies
#   * recall_budget_min <= recall_budget_max
_SAMPLE_VALUES: dict[str, Any] = {
    "reflect_mission": "Answer as a careful archivist.",
    "retain_mission": "Keep only decisions and the reasoning behind them.",
    "retain_extraction_mode": "custom",
    "retain_custom_instructions": "Extract one fact per decision, dated.",
    "retain_chunk_size": 2500,
    "retain_structured_chunk_size": 1800,
    "enable_observations": False,
    "observations_mission": "Observations cover preferences and skills only.",
    "enable_temporal_retrieval": False,
    "enable_graph_retrieval": False,
    "enable_reranking": False,
    "disposition_skepticism": 4,
    "disposition_literalism": 2,
    "disposition_empathy": 5,
    "entity_labels": [
        {
            "key": "team",
            "description": "Owning team",
            "type": "value",
            "optional": False,
            "tag": True,
            "values": [
                {"value": "platform", "description": "Platform team"},
                {"value": "growth", "description": "Growth team"},
            ],
        }
    ],
    "entities_allow_free_form": False,
    "retain_default_strategy": "meetings",
    "retain_strategies": {"meetings": {"retain_chunk_size": 1200, "retain_extraction_mode": "verbose"}},
    "retain_chunk_batch_size": 7,
    "mcp_enabled_tools": ["recall", "retain"],
    "consolidation_llm_batch_size": 11,
    "consolidation_source_facts_max_tokens": 2048,
    "consolidation_source_facts_max_tokens_per_observation": 256,
    "max_observations_per_scope": 13,
    "observation_scope_limits": [{"scope": ["run_*"], "limit": 2}],
    "reflect_source_facts_max_tokens": 4096,
    "llm_gemini_safety_settings": [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}],
    "recall_budget_function": "adaptive",
    "recall_budget_fixed_low": 50,
    "recall_budget_fixed_mid": 250,
    "recall_budget_fixed_high": 800,
    "recall_budget_adaptive_low": 0.05,
    "recall_budget_adaptive_mid": 0.1,
    "recall_budget_adaptive_high": 0.4,
    "recall_budget_min": 30,
    "recall_budget_max": 1500,
    "audit_log_enabled": True,
    "store_document_text": False,
    "enable_auto_consolidation": False,
    "consolidation_max_memories_per_round": 42,
    "consolidation_llm_parallelism": 3,
    "recall_include_chunks": True,
    "recall_max_tokens": 9000,
    "recall_chunks_max_tokens": 4500,
    # Validated against the DefensePolicy schema on write (parse_policy), so this
    # must be a real policy — and carrying a rule means the round-trip covers the
    # nested list, not just the top-level flag.
    "memory_defense": {"enabled": True, "rules": [{"on": "sensitive_data", "action": "redact"}]},
}


@pytest_asyncio.fixture
async def api_client(memory):
    """In-process ASGI client — matches the fixture in tests/test_bank_templates.py."""
    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def bank_id():
    return f"tmpl_roundtrip_{datetime.now().timestamp()}"


async def _read_overrides(api_client: httpx.AsyncClient, bank_id: str) -> dict[str, Any]:
    """Per-bank overrides only — the resolved config would hide a dropped field
    behind the server default and make the round-trip look successful."""
    resp = await api_client.get(f"/v1/default/banks/{bank_id}/config")
    assert resp.status_code == 200, resp.text
    return resp.json()["overrides"]


def test_every_configurable_field_is_exportable():
    """Every per-bank config field must be part of the template engine.

    A field in ``_CONFIGURABLE_FIELDS`` but not on ``BankTemplateConfig`` is
    settable per bank yet invisible to export/import: cloning a bank silently
    drops it, and the clone runs on the server default while looking correctly
    configured. The two sets must match exactly — an intentional exclusion is a
    decision to record here, in an explicit set with a reason, not an omission.
    """
    configurable = HindsightConfig.get_configurable_fields()
    exportable = set(BankTemplateConfig.model_fields)
    assert configurable == exportable, (
        f"bank config and the template engine have drifted:\n"
        f"  configurable but not exportable (add to BankTemplateConfig): "
        f"{sorted(configurable - exportable)}\n"
        f"  exportable but not configurable (remove, or add to _CONFIGURABLE_FIELDS): "
        f"{sorted(exportable - configurable)}"
    )


def test_sample_values_cover_every_exportable_field():
    """A new BankTemplateConfig field must be given a value in _SAMPLE_VALUES.

    Guards the round-trip test below from quietly narrowing: without this, adding
    a template field leaves it untested and nothing says so.
    """
    declared = set(BankTemplateConfig.model_fields)
    sampled = set(_SAMPLE_VALUES)
    assert declared == sampled, (
        f"_SAMPLE_VALUES is out of sync with BankTemplateConfig: "
        f"missing values for {sorted(declared - sampled)}, "
        f"stale entries for {sorted(sampled - declared)}"
    )


@pytest.mark.asyncio
async def test_every_exportable_field_survives_export_then_import(api_client, bank_id):
    """Set every exportable field, export it, and import into a fresh bank."""
    source_id = f"{bank_id}_source"
    clone_id = f"{bank_id}_clone"

    resp = await api_client.post(
        f"/v1/default/banks/{source_id}/import",
        json={"version": "1", "bank": dict(_SAMPLE_VALUES)},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["config_applied"] is True

    # The source bank must actually carry all of them before the round-trip can
    # prove anything — a field the import path drops would otherwise show up as
    # a clean "match" between two equally empty banks.
    source_overrides = await _read_overrides(api_client, source_id)
    missing_after_import = sorted(set(_SAMPLE_VALUES) - set(source_overrides))
    assert not missing_after_import, f"import did not persist: {missing_after_import}"

    export_resp = await api_client.get(f"/v1/default/banks/{source_id}/export")
    assert export_resp.status_code == 200, export_resp.text
    exported = export_resp.json()

    exported_bank = exported.get("bank") or {}
    dropped_by_export = sorted(f for f in _SAMPLE_VALUES if exported_bank.get(f) is None)
    assert not dropped_by_export, f"export dropped fields that were set: {dropped_by_export}"

    import_resp = await api_client.post(f"/v1/default/banks/{clone_id}/import", json=exported)
    assert import_resp.status_code == 200, import_resp.text
    assert import_resp.json()["config_applied"] is True

    # Compare the two banks' overrides rather than the literal input: some fields
    # are normalized on the way in (entity_labels migrates legacy shapes), and the
    # property under test is that a clone ends up configured like its source.
    clone_overrides = await _read_overrides(api_client, clone_id)
    for field in sorted(_SAMPLE_VALUES):
        assert clone_overrides.get(field) == source_overrides.get(field), (
            f"round-trip mismatch for {field}: "
            f"source has {source_overrides.get(field)!r}, clone has {clone_overrides.get(field)!r}"
        )
