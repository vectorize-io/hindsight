"""End-to-end import of the shipped Hermes-tagged bank templates.

Proves every `hermes`-tagged manifest in the Templates Hub catalog actually
imports: creating the bank, applying config, and creating its mental models
and directives — plus idempotent re-apply and additive layering.
"""

import json
from datetime import datetime
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from hindsight_api.api import create_app

_DATA_DIR = Path(__file__).resolve().parents[2] / "hindsight-docs" / "src" / "data"


def _hermes_manifests():
    catalog = json.loads((_DATA_DIR / "templates.json").read_text())
    out = []
    for entry in catalog["templates"]:
        if "hermes" in (entry.get("integrations") or []):
            manifest = json.loads((_DATA_DIR / entry["manifest_file"]).read_text())
            out.append((entry["id"], manifest))
    return out


@pytest_asyncio.fixture
async def api_client(memory):
    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


def _bank(name):
    return f"hermes_tmpl_{name}_{datetime.now().timestamp()}"


@pytest.mark.asyncio
@pytest.mark.parametrize("template_id,manifest", _hermes_manifests(), ids=lambda v: v if isinstance(v, str) else "")
async def test_every_hermes_template_imports_into_a_fresh_bank(api_client, template_id, manifest):
    bank_id = _bank(template_id)
    resp = await api_client.post(f"/v1/default/banks/{bank_id}/import", json=manifest)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["dry_run"] is False

    expected_mm = {m["id"] for m in manifest.get("mental_models", [])}
    expected_dir = {d["name"] for d in manifest.get("directives", [])}
    assert set(data["mental_models_created"]) == expected_mm
    assert set(data["directives_created"]) == expected_dir
    if manifest.get("bank"):
        assert data["config_applied"] is True

    # The bank now exists; export reflects its models + directives.
    exported = (await api_client.get(f"/v1/default/banks/{bank_id}/export")).json()
    assert {m["id"] for m in (exported.get("mental_models") or [])} == expected_mm
    assert {d["name"] for d in (exported.get("directives") or [])} == expected_dir


@pytest.mark.asyncio
async def test_reapply_is_idempotent(api_client):
    _, manifest = next(m for m in _hermes_manifests() if m[0] == "hermes-gateway-bot")
    bank_id = _bank("idem")
    ids = {m["id"] for m in manifest["mental_models"]}

    first = (await api_client.post(f"/v1/default/banks/{bank_id}/import", json=manifest)).json()
    assert set(first["mental_models_created"]) == ids

    second = (await api_client.post(f"/v1/default/banks/{bank_id}/import", json=manifest)).json()
    assert set(second["mental_models_updated"]) == ids
    assert second["mental_models_created"] == []

    # No duplicates: still exactly the same set of models.
    exported = (await api_client.get(f"/v1/default/banks/{bank_id}/export")).json()
    assert {m["id"] for m in (exported.get("mental_models") or [])} == ids


@pytest.mark.asyncio
async def test_layering_a_second_template_is_additive(api_client):
    manifests = dict(_hermes_manifests())
    gateway, research = manifests["hermes-gateway-bot"], manifests["research-assistant"]
    bank_id = _bank("layer")

    await api_client.post(f"/v1/default/banks/{bank_id}/import", json=gateway)
    res = (await api_client.post(f"/v1/default/banks/{bank_id}/import", json=research)).json()

    # research-assistant's models are added on top; gateway's remain.
    assert set(res["mental_models_created"]) == {m["id"] for m in research["mental_models"]}
    exported = (await api_client.get(f"/v1/default/banks/{bank_id}/export")).json()
    got = {m["id"] for m in (exported.get("mental_models") or [])}
    assert {m["id"] for m in gateway["mental_models"]} <= got
    assert {m["id"] for m in research["mental_models"]} <= got

    # config overrides get overwritten by the later template.
    assert exported["bank"]["reflect_mission"] == research["bank"]["reflect_mission"]
