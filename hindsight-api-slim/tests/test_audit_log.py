"""
Tests for the audit log feature.

Tests the audit log list, stats, filtering, and pagination endpoints.
Verifies that audit entries are created for operations when audit logging is enabled.
"""

import asyncio

import httpx
import pytest
import pytest_asyncio

from hindsight_api.api import create_app
from hindsight_api.config import get_config
from tests.conftest import enable_audit_default


@pytest_asyncio.fixture
async def audit_api_client(memory):
    """Create a test client with audit logging enabled deployment-wide."""
    # audit_log_enabled is hierarchical, so the effective value for a
    # bank-scoped action comes from the ConfigResolver, not the logger's own
    # flag. Set both so this fixture models "enabled by default, no per-bank
    # override" — the logger flag covers actions with no bank in scope.
    memory._audit_logger._enabled = True
    memory._audit_logger._allowed_actions = None  # All actions
    enable_audit_default(memory, True)

    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def bank_id():
    """Provide a unique bank ID for audit tests."""
    from datetime import datetime

    return f"audit_test_{datetime.now().timestamp()}"


@pytest.mark.asyncio
async def test_audit_log_list_empty(audit_api_client, bank_id):
    """Test listing audit logs for a bank with no entries returns empty."""
    # Create the bank first
    await audit_api_client.put(
        f"/v1/default/banks/{bank_id}",
        json={"name": "Audit Test Bank"},
    )

    # Small delay for fire-and-forget audit writes
    await asyncio.sleep(0.5)

    response = await audit_api_client.get(f"/v1/default/banks/{bank_id}/audit-logs")
    assert response.status_code == 200
    data = response.json()
    assert data["bank_id"] == bank_id
    assert "total" in data
    assert "items" in data
    assert "limit" in data
    assert "offset" in data
    assert isinstance(data["items"], list)


@pytest.mark.asyncio
async def test_audit_log_created_for_retain(audit_api_client, bank_id):
    """Test that a retain operation creates an audit log entry."""
    # Create bank
    await audit_api_client.put(
        f"/v1/default/banks/{bank_id}",
        json={"name": "Audit Test Bank"},
    )

    # Perform a retain
    response = await audit_api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={
            "items": [{"content": "Alice likes cats", "context": "preferences"}],
        },
    )
    assert response.status_code == 200

    # Wait for fire-and-forget audit writes
    await asyncio.sleep(1.0)

    # List audit logs - should have entries for create_bank and retain
    response = await audit_api_client.get(f"/v1/default/banks/{bank_id}/audit-logs")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 1

    actions = [item["action"] for item in data["items"]]
    assert "retain" in actions, f"Expected 'retain' in audit actions, got: {actions}"


@pytest.mark.asyncio
async def test_audit_log_entry_fields(audit_api_client, bank_id):
    """Test that audit log entries have all expected fields."""
    # Create bank + recall to generate entries
    await audit_api_client.put(
        f"/v1/default/banks/{bank_id}",
        json={"name": "Audit Test Bank"},
    )

    await audit_api_client.post(
        f"/v1/default/banks/{bank_id}/memories/recall",
        json={"query": "test query"},
    )

    await asyncio.sleep(1.0)

    response = await audit_api_client.get(f"/v1/default/banks/{bank_id}/audit-logs")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 1

    # Check the recall entry has all fields
    recall_entries = [item for item in data["items"] if item["action"] == "recall"]
    assert len(recall_entries) >= 1, f"Expected recall entry, got actions: {[i['action'] for i in data['items']]}"

    entry = recall_entries[0]
    assert entry["id"] is not None
    assert entry["action"] == "recall"
    assert entry["transport"] == "http"
    assert entry["bank_id"] == bank_id
    assert entry["started_at"] is not None
    assert entry["ended_at"] is not None
    # Request should contain the recall parameters
    assert entry["request"] is not None
    assert "query" in entry["request"]
    # Response should contain the recall results
    assert entry["response"] is not None


@pytest.mark.asyncio
async def test_audit_log_filter_by_action(audit_api_client, bank_id):
    """Test filtering audit logs by action type."""
    # Create bank and do retain + recall
    await audit_api_client.put(
        f"/v1/default/banks/{bank_id}",
        json={"name": "Audit Test Bank"},
    )
    await audit_api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={"items": [{"content": "test content", "context": "test"}]},
    )
    await audit_api_client.post(
        f"/v1/default/banks/{bank_id}/memories/recall",
        json={"query": "test"},
    )

    await asyncio.sleep(1.0)

    # Filter by retain only
    response = await audit_api_client.get(
        f"/v1/default/banks/{bank_id}/audit-logs",
        params={"action": "retain"},
    )
    assert response.status_code == 200
    data = response.json()
    for item in data["items"]:
        assert item["action"] == "retain"

    # Filter by recall only
    response = await audit_api_client.get(
        f"/v1/default/banks/{bank_id}/audit-logs",
        params={"action": "recall"},
    )
    assert response.status_code == 200
    data = response.json()
    for item in data["items"]:
        assert item["action"] == "recall"


@pytest.mark.asyncio
async def test_audit_log_filter_by_transport(audit_api_client, bank_id):
    """Test filtering audit logs by transport type."""
    await audit_api_client.put(
        f"/v1/default/banks/{bank_id}",
        json={"name": "Audit Test Bank"},
    )

    await asyncio.sleep(0.5)

    # Filter by http transport
    response = await audit_api_client.get(
        f"/v1/default/banks/{bank_id}/audit-logs",
        params={"transport": "http"},
    )
    assert response.status_code == 200
    data = response.json()
    for item in data["items"]:
        assert item["transport"] == "http"

    # Filter by mcp transport - should be empty (no MCP calls in this test)
    response = await audit_api_client.get(
        f"/v1/default/banks/{bank_id}/audit-logs",
        params={"transport": "mcp"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0


@pytest.mark.asyncio
async def test_audit_log_filter_by_date_range(audit_api_client, bank_id):
    """Test filtering audit logs by date range."""
    from datetime import datetime, timedelta, timezone

    await audit_api_client.put(
        f"/v1/default/banks/{bank_id}",
        json={"name": "Audit Test Bank"},
    )

    await asyncio.sleep(0.5)

    now = datetime.now(timezone.utc)

    # Filter with start_date in the past - should include entries
    past = (now - timedelta(hours=1)).isoformat()
    response = await audit_api_client.get(
        f"/v1/default/banks/{bank_id}/audit-logs",
        params={"start_date": past},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 1

    # Filter with start_date in the future - should be empty
    future = (now + timedelta(hours=1)).isoformat()
    response = await audit_api_client.get(
        f"/v1/default/banks/{bank_id}/audit-logs",
        params={"start_date": future},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0


@pytest.mark.asyncio
async def test_audit_log_pagination(audit_api_client, bank_id):
    """Test audit log pagination with limit and offset."""
    await audit_api_client.put(
        f"/v1/default/banks/{bank_id}",
        json={"name": "Audit Test Bank"},
    )

    # Generate multiple audit entries
    for i in range(5):
        await audit_api_client.post(
            f"/v1/default/banks/{bank_id}/memories/recall",
            json={"query": f"test query {i}"},
        )

    await asyncio.sleep(1.5)

    # Get first page
    response = await audit_api_client.get(
        f"/v1/default/banks/{bank_id}/audit-logs",
        params={"limit": 2, "offset": 0},
    )
    assert response.status_code == 200
    page1 = response.json()
    assert len(page1["items"]) == 2
    assert page1["limit"] == 2
    assert page1["offset"] == 0
    assert page1["total"] >= 5  # At least 5 recall + 1 create_bank

    # Get second page
    response = await audit_api_client.get(
        f"/v1/default/banks/{bank_id}/audit-logs",
        params={"limit": 2, "offset": 2},
    )
    assert response.status_code == 200
    page2 = response.json()
    assert len(page2["items"]) == 2
    assert page2["offset"] == 2

    # Entries should be different between pages
    page1_ids = {item["id"] for item in page1["items"]}
    page2_ids = {item["id"] for item in page2["items"]}
    assert page1_ids.isdisjoint(page2_ids), "Pages should not overlap"


@pytest.mark.asyncio
async def test_audit_log_stats(audit_api_client, bank_id):
    """Test the audit log stats endpoint returns correct structure."""
    await audit_api_client.put(
        f"/v1/default/banks/{bank_id}",
        json={"name": "Audit Test Bank"},
    )

    await audit_api_client.post(
        f"/v1/default/banks/{bank_id}/memories/recall",
        json={"query": "stats test"},
    )

    await asyncio.sleep(1.0)

    # Get stats for last 24h
    response = await audit_api_client.get(
        f"/v1/default/banks/{bank_id}/audit-logs/stats",
        params={"period": "1d"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["bank_id"] == bank_id
    assert data["period"] == "1d"
    assert data["trunc"] == "day"
    assert "buckets" in data
    assert isinstance(data["buckets"], list)

    # Should have at least one bucket with our operations
    assert len(data["buckets"]) >= 1
    bucket = data["buckets"][0]
    assert "time" in bucket
    assert "actions" in bucket
    assert "total" in bucket
    assert bucket["total"] >= 1


@pytest.mark.asyncio
async def test_audit_log_stats_filter_by_action(audit_api_client, bank_id):
    """Test stats endpoint filters by action."""
    await audit_api_client.put(
        f"/v1/default/banks/{bank_id}",
        json={"name": "Audit Test Bank"},
    )

    await audit_api_client.post(
        f"/v1/default/banks/{bank_id}/memories/recall",
        json={"query": "test"},
    )

    await asyncio.sleep(1.0)

    # Stats filtered by recall
    response = await audit_api_client.get(
        f"/v1/default/banks/{bank_id}/audit-logs/stats",
        params={"period": "1d", "action": "recall"},
    )
    assert response.status_code == 200
    data = response.json()
    for bucket in data["buckets"]:
        # All actions in buckets should be "recall" only
        for action_name in bucket["actions"]:
            assert action_name == "recall"


@pytest.mark.asyncio
async def test_audit_log_stats_periods(audit_api_client, bank_id):
    """Test stats endpoint supports different periods."""
    await audit_api_client.put(
        f"/v1/default/banks/{bank_id}",
        json={"name": "Audit Test Bank"},
    )

    await asyncio.sleep(0.5)

    for period, expected_trunc in [("1d", "day"), ("7d", "day"), ("30d", "day")]:
        response = await audit_api_client.get(
            f"/v1/default/banks/{bank_id}/audit-logs/stats",
            params={"period": period},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["period"] == period
        assert data["trunc"] == expected_trunc


@pytest.mark.asyncio
async def test_audit_log_disabled(memory):
    """Test that no audit logs are created when audit logging is disabled."""
    # Ensure audit logging is disabled
    memory._audit_logger._enabled = False

    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        from datetime import datetime

        bid = f"audit_disabled_test_{datetime.now().timestamp()}"

        await client.put(f"/v1/default/banks/{bid}", json={"name": "No Audit"})
        await client.post(
            f"/v1/default/banks/{bid}/memories/recall",
            json={"query": "test"},
        )

        await asyncio.sleep(0.5)

        response = await client.get(f"/v1/default/banks/{bid}/audit-logs")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0, "No audit entries should exist when audit logging is disabled"


@pytest.mark.asyncio
async def test_audit_log_action_allowlist(memory):
    """Test that only allowed actions are audited when allowlist is set."""
    memory._audit_logger._enabled = True
    memory._audit_logger._allowed_actions = frozenset({"recall"})  # Only audit recall
    enable_audit_default(memory, True)

    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        from datetime import datetime

        bid = f"audit_allowlist_test_{datetime.now().timestamp()}"

        # create_bank should NOT be audited
        await client.put(f"/v1/default/banks/{bid}", json={"name": "Allowlist Test"})
        # recall should be audited
        await client.post(
            f"/v1/default/banks/{bid}/memories/recall",
            json={"query": "allowlist test"},
        )

        await asyncio.sleep(1.0)

        response = await client.get(f"/v1/default/banks/{bid}/audit-logs")
        assert response.status_code == 200
        data = response.json()
        actions = [item["action"] for item in data["items"]]
        assert "recall" in actions, "recall should be audited"
        assert "create_bank" not in actions, "create_bank should NOT be audited (not in allowlist)"


@pytest.mark.asyncio
async def test_audit_log_ordered_by_most_recent(audit_api_client, bank_id):
    """Test that audit logs are returned ordered by most recent first."""
    await audit_api_client.put(
        f"/v1/default/banks/{bank_id}",
        json={"name": "Order Test Bank"},
    )

    for i in range(3):
        await audit_api_client.post(
            f"/v1/default/banks/{bank_id}/memories/recall",
            json={"query": f"order test {i}"},
        )
        await asyncio.sleep(0.2)  # Small gap between requests

    await asyncio.sleep(1.0)

    response = await audit_api_client.get(f"/v1/default/banks/{bank_id}/audit-logs")
    assert response.status_code == 200
    data = response.json()

    # Check descending order by started_at
    timestamps = [item["started_at"] for item in data["items"] if item["started_at"]]
    assert timestamps == sorted(timestamps, reverse=True), "Audit logs should be ordered most recent first"
