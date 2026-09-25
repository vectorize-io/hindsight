"""consolidation_strategies through the bank-config and template endpoints.

Three things that only break in the seam between the editor, the config store and
the template engine:

* a malformed strategy is **rejected on write**, not stored. The config type is a
  plain list, so before this a typo ("scope" for "scopes") saved happily and the
  strategy then never applied, with nothing to show why.
* an **incomplete draft** is still accepted — the control plane saves strategies
  as typed, and consolidation ignores the unusable ones.
* **export/import round-trips** the value, and an export still works for a bank
  that stored a bad value before the field was typed (otherwise one dead field
  would make the whole bank un-exportable).
"""

from datetime import datetime
from typing import Any

import httpx
import pytest
import pytest_asyncio

from hindsight_api.api import create_app
from hindsight_api.api.http import BankTemplateConfig, _bank_template_config_from_overrides

STRATEGIES: list[dict[str, Any]] = [
    {
        "scopes": [
            {"tags": ["company:*"], "tags_match": "exact"},
            {"tags": ["org:*", "shared"]},
        ],
        "observations_mission": "Record only generalized trends.",
        "max_observations_per_scope": 20,
        "consolidation_source_facts_max_tokens": 1024,
    }
]


@pytest_asyncio.fixture
async def api_client(memory):
    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest_asyncio.fixture
async def bank(memory, request_context):
    bank_id = f"tmpl_strategies_{datetime.now().timestamp()}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    yield bank_id
    await memory.delete_bank(bank_id, request_context=request_context)


async def _set_strategies(api_client, bank_id, value):
    return await api_client.patch(
        f"/v1/default/banks/{bank_id}/config", json={"updates": {"consolidation_strategies": value}}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value,expected_in_detail",
    [
        ("not-a-list", "must be a list"),
        (["not-an-object"], "index 0"),
        ([{"scope": [{"tags": ["a"]}]}], "scope"),  # the "scopes" typo
        ([{"scopes": "company:*"}], "scopes"),
        ([{"scopes": [{"tags": "company:*"}]}], "tags"),
        ([{"scopes": [{"tags": ["a"], "tags_match": "sometimes"}]}], "tags_match"),
    ],
)
async def test_a_malformed_strategy_is_rejected_on_write(api_client, bank, value, expected_in_detail):
    resp = await _set_strategies(api_client, bank, value)

    assert resp.status_code == 400, resp.text
    assert expected_in_detail in resp.text
    # And nothing was stored.
    config = await api_client.get(f"/v1/default/banks/{bank}/config")
    assert "consolidation_strategies" not in config.json()["overrides"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "draft",
    [
        [{"scopes": []}],  # a strategy whose rules were all removed
        [{"scopes": [{"tags": []}]}],  # a rule with no tags typed yet
        [{"scopes": [{"tags": ["company:*"]}]}],  # rules, but no setting yet
    ],
)
async def test_an_incomplete_draft_is_accepted(api_client, bank, draft):
    """The editor saves what is on screen; consolidation ignores what it cannot use."""
    resp = await _set_strategies(api_client, bank, draft)

    assert resp.status_code == 200, resp.text
    assert resp.json()["config"]["consolidation_strategies"] == draft


def _without_nulls(value: Any) -> Any:
    """Drop null fields, which the template manifest keeps for every unset field
    (this route opts out of dropping nulls — see ExcludeNoneRoute)."""
    if isinstance(value, dict):
        return {k: _without_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_without_nulls(v) for v in value]
    return value


@pytest.mark.asyncio
async def test_strategies_survive_export_and_import(api_client, bank, memory, request_context):
    assert (await _set_strategies(api_client, bank, STRATEGIES)).status_code == 200

    exported = await api_client.get(f"/v1/default/banks/{bank}/export")
    assert exported.status_code == 200, exported.text
    exported_strategies = exported.json()["bank"]["consolidation_strategies"]
    assert _without_nulls(exported_strategies) == STRATEGIES, "export must not reshape the value"

    target = f"{bank}_clone"
    await memory.ensure_bank_profile(bank_id=target, request_context=request_context)
    try:
        imported = await api_client.post(f"/v1/default/banks/{target}/import", json=exported.json())
        assert imported.status_code == 200, imported.text
        assert imported.json()["config_applied"] is True

        config = await api_client.get(f"/v1/default/banks/{target}/config")
        assert config.json()["overrides"]["consolidation_strategies"] == exported_strategies

        # Exporting the clone gives the same manifest: importing an export and
        # re-exporting it must reach a fixed point, or every round trip through a
        # template would grow the stored value.
        re_exported = await api_client.get(f"/v1/default/banks/{target}/export")
        assert re_exported.json()["bank"]["consolidation_strategies"] == exported_strategies
    finally:
        await memory.delete_bank(target, request_context=request_context)


def test_export_drops_a_stored_value_the_template_model_rejects():
    """A bank can hold a value written before the field was typed. Exporting it
    must not fail the whole bank — the bad field is left out."""
    config = _bank_template_config_from_overrides(
        {
            "observations_mission": "kept",
            # A shape the template model cannot parse at all (the tag rules are a
            # string). An unknown key would not do: only the write path's strict
            # twin rejects those, so the export model reads past them.
            "consolidation_strategies": "written before this field was typed",
        }
    )

    assert config is not None
    assert config.observations_mission == "kept"
    assert config.consolidation_strategies is None


def test_export_of_a_clean_config_keeps_every_field():
    config = _bank_template_config_from_overrides(
        {"observations_mission": "kept", "consolidation_strategies": STRATEGIES}
    )

    assert config is not None
    dumped = config.model_dump(exclude_none=True)["consolidation_strategies"]
    assert dumped == STRATEGIES, "round-trip through the typed model must not reshape the value"
    assert all("tags_match" not in rule for rule in dumped[0]["scopes"][1:]), (
        "a rule that omitted tags_match must not gain one"
    )


def test_unknown_fields_are_still_ignored_by_the_template_model():
    """Overrides carry fields the template does not export (credentials, static
    settings); those are filtered, not treated as invalid."""
    assert _bank_template_config_from_overrides({"llm_api_key": "secret"}) is None
    assert set(BankTemplateConfig.model_fields) >= {"consolidation_strategies"}


@pytest.mark.asyncio
async def test_importing_a_template_with_a_malformed_strategy_is_refused(api_client, bank):
    """The typed template model is the other door into the config. A manifest
    carrying a broken strategy must be refused rather than written, or import
    would be a way around the validation the config endpoint applies."""
    resp = await api_client.post(
        f"/v1/default/banks/{bank}/import",
        json={"version": "1", "bank": {"consolidation_strategies": [{"scopes": "company:*"}]}},
    )

    assert resp.status_code in (400, 422), resp.text
    config = await api_client.get(f"/v1/default/banks/{bank}/config")
    assert "consolidation_strategies" not in config.json()["overrides"]
