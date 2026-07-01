"""MCP tool surface tests — governance + membership, routed through services."""

import asyncio

from app.connectors.google_drive import service as gd
from app.db import repositories as repo
from app.db.engine import init_models, session_scope
from app.mcp import tools
from tests.test_connector_flow import SEED


def test_allowed_tools_are_the_v01_set():
    assert set(tools.ALLOWED_TOOLS) == {
        "list_workspaces", "list_connected_sources", "list_source_documents", "sync_source",
        "get_source_audit", "list_ingestion_jobs", "search_governed_documents",
    }


def test_tools_route_through_service_layer_not_raw_clients():
    # MCP tools must route through the service/repository layer only — never reach
    # Google Drive, the vector DB, or raw DB engine clients directly. Check the
    # module's actually-bound names (not docstring text).
    ns = vars(tools)
    assert "repo" in ns and "gdrive" in ns  # routes through repositories + service
    assert "build" not in ns  # no googleapiclient.discovery.build
    assert "create_async_engine" not in ns  # no direct engine construction
    imported = {getattr(v, "__name__", "") for v in ns.values()}
    assert not any("googleapiclient" in name or "qdrant" in name for name in imported)


def test_non_member_is_denied():
    async def scenario():
        await init_models()
        async with session_scope() as s:
            owner = await repo.create_user(s, email="owner@x.com")
            outsider = await repo.create_user(s, email="out@x.com")
            ws = await repo.create_workspace(s, name="W", owner_id=owner["id"])
        async with session_scope() as s:
            return await tools.list_source_documents(s, actor_id=outsider["id"],
                                                     workspace_id=ws["id"])

    assert asyncio.run(scenario()) == {"error": "not_workspace_member"}


def test_search_governed_documents_fails_closed():
    async def scenario():
        await init_models()
        async with session_scope() as s:
            owner = await repo.create_user(s, email="owner@x.com")
            ws = await repo.create_workspace(s, name="W", owner_id=owner["id"])
        async with session_scope() as s:
            await gd.connect(s, workspace_id=ws["id"], actor_id=owner["id"], folder_ids=["fld1"],
                             refresh_token="rt", seed=SEED)
        async with session_scope() as s:
            await gd.sync(s, workspace_id=ws["id"], actor_id=owner["id"])
        # Permitted principal (a@x.com has reader on f1)
        async with session_scope() as s:
            permitted = await tools.search_governed_documents(
                s, actor_id=owner["id"], workspace_id=ws["id"], email="a@x.com")
        # Principal with no matching permission anywhere
        async with session_scope() as s:
            blocked = await tools.search_governed_documents(
                s, actor_id=owner["id"], workspace_id=ws["id"], email="nobody@x.com")
        async with session_scope() as s:
            denials = [e for e in await repo.list_audit(s, workspace_id=ws["id"])
                       if e["action"] == "retrieval_permission_denied"]
        return permitted, blocked, denials

    permitted, blocked, denials = asyncio.run(scenario())
    assert [r["external_id"] for r in permitted["results"]] == ["f1"]
    assert permitted["denied_count"] == 1  # f2 has no permissions → denied
    assert blocked["results"] == []  # nobody@x.com permitted on nothing
    assert len(denials) >= 1  # denials are audited
