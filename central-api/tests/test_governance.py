"""Tests for governance — Postgres-backed quarantine persistence (GOV-003)."""

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.db.tables import metadata
from app.governance.quarantine import (
    approve_quarantine_item,
    create_quarantine_item,
    get_approval_history,
    get_quarantine_item,
    list_quarantine_items,
    reject_quarantine_item,
)
from app.governance.write_gate import policy_check


@pytest_asyncio.fixture
async def db_session():
    """Create in-memory SQLite session for tests."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
    
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        yield session
    
    await engine.dispose()


@pytest.mark.asyncio
async def test_create_quarantine_restricted(db_session: AsyncSession):
    """Test creating a quarantine item for restricted content."""
    tenant_id = "test-tenant-1"
    content = {"data": "restricted_value"}
    
    item_id = await create_quarantine_item(
        db_session,
        tenant_id=tenant_id,
        content=content,
        classification="restricted",
        created_by="user-1",
        reason="User requested storage of restricted data",
    )
    
    assert item_id is not None
    assert len(item_id) == 36  # UUID format
    
    # Verify item in quarantine
    items = await list_quarantine_items(db_session, tenant_id=tenant_id, status="pending")
    assert len(items) > 0
    assert items[0]["classification"] == "restricted"
    assert items[0]["status"] == "pending"


@pytest.mark.asyncio
async def test_reject_quarantine_item(db_session: AsyncSession):
    """Test rejecting a quarantine item."""
    tenant_id = "test-tenant-1"
    item_id = await create_quarantine_item(
        db_session,
        tenant_id=tenant_id,
        content={"sensitive": "data"},
        classification="private",
        created_by="user-1",
    )
    
    success = await reject_quarantine_item(
        db_session,
        item_id=item_id,
        tenant_id=tenant_id,
        rejector_id="approver-1",
        reason="Content violates policy",
    )
    
    assert success is True
    
    item = await get_quarantine_item(db_session, item_id=item_id, tenant_id=tenant_id)
    assert item["status"] == "rejected"


@pytest.mark.asyncio
async def test_approve_quarantine_item(db_session: AsyncSession):
    """Test approving a quarantine item."""
    tenant_id = "test-tenant-1"
    item_id = await create_quarantine_item(
        db_session,
        tenant_id=tenant_id,
        content={"sensitive": "data"},
        classification="restricted",
        created_by="user-1",
    )
    
    success = await approve_quarantine_item(
        db_session,
        item_id=item_id,
        tenant_id=tenant_id,
        approver_id="approver-1",
        reason="Approved for release",
    )
    
    assert success is True
    
    item = await get_quarantine_item(db_session, item_id=item_id, tenant_id=tenant_id)
    assert item["status"] == "approved"
    assert item["approved_by"] == "approver-1"


@pytest.mark.asyncio
async def test_get_approval_history(db_session: AsyncSession):
    """Test retrieving approval history."""
    tenant_id = "test-tenant-1"
    item_id = await create_quarantine_item(
        db_session,
        tenant_id=tenant_id,
        content={"data": "test"},
        classification="private",
        created_by="user-1",
    )
    
    await approve_quarantine_item(
        db_session,
        item_id=item_id,
        tenant_id=tenant_id,
        approver_id="approver-1",
        reason="Looks good",
    )
    
    history = await get_approval_history(db_session, item_id=item_id, tenant_id=tenant_id)
    assert len(history) > 0
    assert history[0]["decision"] == "approved"
    assert history[0]["approver_id"] == "approver-1"


@pytest.mark.asyncio
async def test_tenant_isolation(db_session: AsyncSession):
    """Test that quarantine items are tenant-isolated."""
    # Create in tenant 1
    item_id = await create_quarantine_item(
        db_session,
        tenant_id="tenant-1",
        content={"data": "secret"},
        classification="private",
    )
    
    # Try to get from tenant 2 — should fail
    item = await get_quarantine_item(db_session, item_id=item_id, tenant_id="tenant-2")
    assert item is None


@pytest.mark.asyncio
async def test_policy_check_secret_rejected(db_session: AsyncSession):
    """Test that secret content is rejected by policy."""
    result = await policy_check(
        db_session,
        content={"password": "secret123"},
        tenant_id="test-tenant",
        actor_id="user-1",
        classification="secret",
    )
    
    assert result["allowed"] is False
    assert result["reason"] == "secret_content_rejected"
    assert result["quarantine_id"] is None


@pytest.mark.asyncio
async def test_policy_check_restricted_quarantined(db_session: AsyncSession):
    """Test that restricted content is quarantined."""
    result = await policy_check(
        db_session,
        content={"data": "restricted"},
        tenant_id="test-tenant",
        actor_id="user-1",
        classification="restricted",
    )
    
    assert result["allowed"] is False
    assert result["reason"] == "awaiting_approval"
    assert result["quarantine_id"] is not None


@pytest.mark.asyncio
async def test_policy_check_public_allowed(db_session: AsyncSession):
    """Test that public content is allowed."""
    result = await policy_check(
        db_session,
        content={"data": "public"},
        tenant_id="test-tenant",
        actor_id="user-1",
        classification="public",
    )
    
    assert result["allowed"] is True
    assert result["reason"] == "approved_by_policy"
    assert result["quarantine_id"] is None


@pytest.mark.asyncio
async def test_list_quarantine_filters(db_session: AsyncSession):
    """Test quarantine list with status filter."""
    tenant_id = "filter-test-tenant"
    
    # Create 2 items
    item1 = await create_quarantine_item(
        db_session,
        tenant_id=tenant_id,
        content={"data": "1"},
        classification="restricted",
    )
    
    item2 = await create_quarantine_item(
        db_session,
        tenant_id=tenant_id,
        content={"data": "2"},
        classification="private",
    )
    
    # Approve one
    await approve_quarantine_item(
        db_session,
        item_id=item1,
        tenant_id=tenant_id,
        approver_id="approver-1",
    )
    
    # List pending
    pending = await list_quarantine_items(db_session, tenant_id=tenant_id, status="pending")
    assert len(pending) >= 1
    assert pending[0]["id"] == item2
    
    # List approved
    approved = await list_quarantine_items(db_session, tenant_id=tenant_id, status="approved")
    assert len(approved) >= 1
    assert approved[0]["id"] == item1


@pytest.mark.asyncio
async def test_get_nonexistent_item(db_session: AsyncSession):
    """Test retrieving nonexistent quarantine item."""
    item = await get_quarantine_item(
        db_session,
        item_id="nonexistent-id",
        tenant_id="test-tenant",
    )
    assert item is None
