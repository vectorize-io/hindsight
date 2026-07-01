"""Control-plane API tests via TestClient (dev auth fallback).

The TestClient is used as a context manager so the lifespan runs and creates the
control-plane tables against the per-test SQLite DB.
"""

from fastapi.testclient import TestClient

from app.main import app

SEED = {
    "files": {
        "fld1": [
            {"id": "f1", "name": "Plan", "mimeType": "application/vnd.google-apps.document",
             "owners": [{"emailAddress": "a@x.com"}], "webViewLink": "http://v/f1",
             "parents": ["fld1"]},
        ]
    },
    "permissions": {"f1": [{"id": "p1", "type": "user", "role": "reader",
                            "emailAddress": "a@x.com"}]},
}


def test_full_operator_flow_through_api():
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"

        me = client.get("/me").json()
        assert me["user"]["is_operator"] is True

        ws = client.post("/workspaces", json={"name": "Telecom"})
        assert ws.status_code == 201
        ws_id = ws.json()["id"]

        # Connect Google Drive (seeded fake) through the API — not directly.
        conn = client.post("/connectors/google-drive/connect",
                           json={"workspace_id": ws_id, "folder_ids": ["fld1"],
                                 "account_email": "op@x.com", "seed": SEED})
        assert conn.status_code == 200
        assert conn.json()["status"] == "connected"

        status = client.get("/connectors/google-drive/status", params={"workspace_id": ws_id})
        assert status.json()["status"] == "connected"

        synced = client.post("/connectors/google-drive/sync", json={"workspace_id": ws_id}).json()
        assert synced["discovered"] == 1 and synced["queued"] == 1

        files = client.get("/connectors/google-drive/files", params={"workspace_id": ws_id}).json()
        assert len(files["files"]) == 1
        doc_id = files["files"][0]["id"]

        perms = client.get(f"/connectors/google-drive/files/{doc_id}/permissions",
                          params={"workspace_id": ws_id}).json()
        assert len(perms["permissions"]) == 1

        jobs = client.get("/ingestion-jobs", params={"workspace_id": ws_id}).json()
        assert jobs["jobs"][0]["status"] == "pending"

        processed = client.post("/ingestion/process", json={"workspace_id": ws_id}).json()
        assert processed["results"][0]["status"] == "indexed"

        events = client.get("/audit-events", params={"workspace_id": ws_id}).json()
        actions = {e["action"] for e in events["events"]}
        assert {"google_drive_connected", "file_discovered", "file_indexed"} <= actions

        # operator disables a document
        disabled = client.post(f"/source-documents/{doc_id}/disable",
                              params={"workspace_id": ws_id}).json()
        assert disabled["enabled"] is False

        # agent action request returns a governed decision
        decision = client.post("/agent-activity", json={
            "agent_id": "agent-7", "action": "bulk_sync", "workspace_id": ws_id}).json()
        assert decision["decision"] == "requires_approval"


def test_non_member_forbidden():
    with TestClient(app) as client:
        # Create a workspace owned by someone else, then query as a different user
        # would require a second identity; here we assert the membership guard fires
        # for a random workspace id the dev user is not a member of.
        r = client.get("/source-documents", params={"workspace_id": "00000000-dead-beef"})
        assert r.status_code == 403


def test_governance_policy_check_and_execution_history_are_mounted():
    with TestClient(app) as client:
        policy = client.post("/api/gov/policy-check", json={"content": "hello world", "classification": "public"})
        assert policy.status_code == 200
        assert policy.json()["allowed"] is True

        history = client.get("/api/executions/history")
        assert history.status_code == 200
        assert "executions" in history.json()


def test_api_center_route_catalog_includes_new_backend_contracts():
    with TestClient(app) as client:
        routes = client.get("/api/api-center/routes", params={"service": "central-api"})
        assert routes.status_code == 200
        paths = {route["path"] for route in routes.json()["routes"]}
        assert "/api/gov/policy-check" in paths
        assert "/api/executions/history" in paths
        assert "/api/memory/search/unified" in paths
        assert "/health/dependencies" in paths
