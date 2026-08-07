"""Tests for the GitHub role-based OperationValidator."""

from unittest.mock import MagicMock

import pytest

from hindsight_api.extensions.builtin.github_role_validator import GitHubRoleOperationValidator
from hindsight_api.extensions.builtin.github_tenant import (
    ROLE_ADMIN,
    ROLE_MEMBER,
    ROLE_VIEWER,
    encode_tenant_id,
)
from hindsight_api.models import RequestContext


def _ctx(role: str | None):
    tenant_id = encode_tenant_id("1", role) if role else None
    c = MagicMock()
    c.request_context = RequestContext(api_key="x", tenant_id=tenant_id)
    return c


@pytest.fixture
def validator():
    return GitHubRoleOperationValidator({})


class TestRetain:
    @pytest.mark.asyncio
    async def test_admin_allowed(self, validator):
        assert (await validator.validate_retain(_ctx(ROLE_ADMIN))).allowed

    @pytest.mark.asyncio
    async def test_member_allowed(self, validator):
        assert (await validator.validate_retain(_ctx(ROLE_MEMBER))).allowed

    @pytest.mark.asyncio
    async def test_viewer_rejected(self, validator):
        result = await validator.validate_retain(_ctx(ROLE_VIEWER))
        assert not result.allowed

    @pytest.mark.asyncio
    async def test_unknown_rejected(self, validator):
        assert not (await validator.validate_retain(_ctx(None))).allowed


class TestReflect:
    @pytest.mark.asyncio
    async def test_viewer_rejected(self, validator):
        assert not (await validator.validate_reflect(_ctx(ROLE_VIEWER))).allowed

    @pytest.mark.asyncio
    async def test_member_allowed(self, validator):
        assert (await validator.validate_reflect(_ctx(ROLE_MEMBER))).allowed


class TestRecall:
    @pytest.mark.asyncio
    async def test_viewer_allowed(self, validator):
        assert (await validator.validate_recall(_ctx(ROLE_VIEWER))).allowed

    @pytest.mark.asyncio
    async def test_unknown_allowed_to_read(self, validator):
        # Recall is permitted for everyone, including unmapped tenants.
        assert (await validator.validate_recall(_ctx(None))).allowed
