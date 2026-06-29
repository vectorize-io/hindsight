"""HTTP + engine integration tests for the knowledge base (folders + pages).

Pages are seeded directly via the engine (deterministic content, no LLM) so the
tree, OKF projection, move/rename, and cascade-delete behaviour can be asserted
without consolidation.
"""

import urllib.parse
import uuid

import pytest_asyncio

from hindsight_api.engine.memory_engine import MemoryEngine


def _enc(bank_id: str) -> str:
    return urllib.parse.quote(bank_id, safe="")


class _Seed:
    """Holds the ids created by the seed fixture for assertions."""

    def __init__(self, **ids):
        self.__dict__.update(ids)


@pytest_asyncio.fixture
async def kb_bank(memory: MemoryEngine, request_context):
    """A bank with folders, nested folders, and pages."""
    bank_id = f"test-kb-{uuid.uuid4().hex[:8]}"

    runbooks = await memory.create_knowledge_folder(bank_id, "Runbooks", request_context=request_context)
    policies = await memory.create_knowledge_folder(bank_id, "Policies", request_context=request_context)
    sub = await memory.create_knowledge_folder(
        bank_id, "Sub", parent_id=runbooks["id"], request_context=request_context
    )
    orders = await memory.create_knowledge_page(
        bank_id,
        "Orders",
        "What are the order facts?",
        "# Orders\n\nOne row per order.",
        parent_id=runbooks["id"],
        tags=["type:runbook", "sales", "revenue"],
        request_context=request_context,
    )
    billing = await memory.create_knowledge_page(
        bank_id,
        "Billing",
        "What is the billing policy?",
        "# Billing\n\nNet-30.",
        parent_id=policies["id"],
        tags=["type:policy", "revenue"],
        request_context=request_context,
    )
    loose = await memory.create_knowledge_page(
        bank_id,
        "Loose",
        "A root page.",
        "# Loose\n\nNo folder, no tags.",
        tags=[],
        request_context=request_context,
    )

    yield (
        bank_id,
        _Seed(
            runbooks=runbooks["id"],
            policies=policies["id"],
            sub=sub["id"],
            orders=orders["id"],
            billing=billing["id"],
            loose=loose["id"],
            orders_mm=orders["mental_model_id"],
        ),
    )

    await memory.delete_bank(bank_id, request_context=request_context)


class TestTree:
    async def test_nested_tree(self, api_client, kb_bank):
        bank_id, ids = kb_bank
        resp = await api_client.get(f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/tree")
        assert resp.status_code == 200, resp.text
        roots = {r["name"]: r for r in resp.json()["roots"]}
        assert set(roots) == {"Runbooks", "Policies", "Loose"}

        runbooks = roots["Runbooks"]
        assert runbooks["kind"] == "folder"
        child_names = {c["name"] for c in runbooks["children"]}
        assert child_names == {"Sub", "Orders"}

        orders = next(c for c in runbooks["children"] if c["name"] == "Orders")
        assert orders["kind"] == "page"
        # Human-created pages are pinned (not curator-managed).
        assert orders["managed"] is False
        assert "sales" in orders["tags"]
        assert roots["Loose"]["kind"] == "page"


class TestGetPage:
    async def test_okf_document(self, api_client, kb_bank):
        bank_id, ids = kb_bank
        resp = await api_client.get(f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/pages/{ids.orders}")
        assert resp.status_code == 200, resp.text
        page = resp.json()
        assert page["type"] == "runbook"
        assert page["body"].startswith("# Orders")
        assert page["markdown"].startswith("---\n")
        assert 'type: "runbook"' in page["markdown"]

    async def test_missing_page_404(self, api_client, kb_bank):
        bank_id, ids = kb_bank
        resp = await api_client.get(f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/pages/nope")
        assert resp.status_code == 404


class TestCreate:
    async def test_create_folder(self, api_client, kb_bank):
        bank_id, ids = kb_bank
        resp = await api_client.post(
            f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/folders",
            json={"name": "Guides", "parent_id": None},
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["kind"] == "folder"
        assert resp.json()["name"] == "Guides"

    async def test_create_folder_bad_parent(self, api_client, kb_bank):
        bank_id, ids = kb_bank
        # parent that is a page, not a folder → 400
        resp = await api_client.post(
            f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/folders",
            json={"name": "Nope", "parent_id": ids.orders},
        )
        assert resp.status_code == 400

    async def test_create_folder_with_mission_schedules_curation(self, api_client, kb_bank):
        bank_id, ids = kb_bank
        # Creating a folder with a mission triggers a background curation; with the
        # sync task backend it runs inline and must not fail the create.
        resp = await api_client.post(
            f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/folders",
            json={"name": "Auto", "mission": "Collect everything about orders"},
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["mission"] == "Collect everything about orders"


class TestGraphAndExport:
    async def test_graph_shared_tag_edge(self, api_client, kb_bank):
        bank_id, ids = kb_bank
        resp = await api_client.get(f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/graph")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["total_pages"] == 3
        # orders & billing share "revenue"; loose has no tags
        assert data["total_edges"] == 1
        edge = data["edges"][0]["data"]
        assert {edge["source"], edge["target"]} == {ids.orders, ids.billing}
        assert edge["sharedTags"] == ["revenue"]

    async def test_export_bundle_nested_index(self, api_client, kb_bank):
        bank_id, ids = kb_bank
        resp = await api_client.get(f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/export")
        assert resp.status_code == 200, resp.text
        files = {f["path"]: f["content"] for f in resp.json()["files"]}
        assert "index.md" in files
        assert f"{ids.orders}.md" in files
        # index reflects the folder hierarchy
        assert "**Runbooks/**" in files["index.md"]
        assert "One row per order." in files[f"{ids.orders}.md"]


class TestMoveRenameDelete:
    async def test_rename(self, api_client, kb_bank):
        bank_id, ids = kb_bank
        resp = await api_client.patch(
            f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/nodes/{ids.policies}",
            json={"name": "Compliance"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["name"] == "Compliance"

    async def test_move_into_folder(self, api_client, kb_bank):
        bank_id, ids = kb_bank
        # move the Loose root page under Policies
        resp = await api_client.patch(
            f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/nodes/{ids.loose}",
            json={"parent_id": ids.policies},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["parent_id"] == ids.policies

    async def test_move_cycle_rejected(self, api_client, kb_bank):
        bank_id, ids = kb_bank
        # moving Runbooks under its own descendant Sub must fail
        resp = await api_client.patch(
            f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/nodes/{ids.runbooks}",
            json={"parent_id": ids.sub},
        )
        assert resp.status_code == 400

    async def test_delete_folder_cascades(self, api_client, kb_bank, memory, request_context):
        bank_id, ids = kb_bank
        # deleting Runbooks removes Sub + Orders (and Orders' mental model)
        resp = await api_client.delete(f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/nodes/{ids.runbooks}")
        assert resp.status_code == 200, resp.text

        tree = (await api_client.get(f"/v1/default/banks/{_enc(bank_id)}/knowledge-base/tree")).json()
        root_names = {r["name"] for r in tree["roots"]}
        assert "Runbooks" not in root_names
        # the backing mental model is gone too
        mm = await memory.get_mental_model(bank_id, ids.orders_mm, request_context=request_context)
        assert mm is None
