import pytest
from hindsight_agentcore.bank import TurnContext, default_bank_resolver
from hindsight_agentcore.errors import BankResolutionError


class TestTurnContext:
    def _ctx(self, **kwargs):
        defaults = dict(
            runtime_session_id="sess-abc",
            user_id="user-123",
            agent_name="support-agent",
            tenant_id="acme",
            request_id="req-001",
        )
        return TurnContext(**{**defaults, **kwargs})

    def test_as_metadata_includes_required_fields(self):
        ctx = self._ctx()
        meta = ctx.as_metadata()
        assert meta["channel"] == "agentcore-runtime"
        assert meta["runtime_session_id"] == "sess-abc"
        assert meta["user_id"] == "user-123"
        assert meta["agent_name"] == "support-agent"
        assert meta["tenant_id"] == "acme"
        assert meta["request_id"] == "req-001"

    def test_as_metadata_omits_optional_none_fields(self):
        ctx = self._ctx(tenant_id=None, request_id=None)
        meta = ctx.as_metadata()
        assert "tenant_id" not in meta
        assert "request_id" not in meta

    def test_as_tags_includes_user_agent_session(self):
        ctx = self._ctx(tenant_id=None)
        tags = ctx.as_tags()
        assert "user:user-123" in tags
        assert "agent:support-agent" in tags
        assert "session:sess-abc" in tags

    def test_as_tags_includes_tenant_when_present(self):
        ctx = self._ctx()
        tags = ctx.as_tags()
        assert "tenant:acme" in tags
        # tenant should appear first for correct filtering
        assert tags[0] == "tenant:acme"


class TestDefaultBankResolver:
    def _ctx(self, **kwargs):
        defaults = dict(
            runtime_session_id="sess-xyz",
            user_id="user-456",
            agent_name="eng-agent",
        )
        return TurnContext(**{**defaults, **kwargs})

    def test_with_tenant_user_agent(self):
        ctx = self._ctx(tenant_id="acme")
        bank_id = default_bank_resolver(ctx)
        assert bank_id == "tenant:acme:user:user-456:agent:eng-agent"

    def test_without_tenant(self):
        ctx = self._ctx()
        bank_id = default_bank_resolver(ctx)
        assert bank_id == "user:user-456:agent:eng-agent"

    def test_session_id_not_in_bank_id(self):
        """Runtime sessions expire — session ID must never be the bank ID."""
        ctx = self._ctx(tenant_id="acme")
        bank_id = default_bank_resolver(ctx)
        assert "sess-xyz" not in bank_id

    def test_missing_user_id_raises(self):
        ctx = self._ctx(user_id="")
        with pytest.raises(BankResolutionError, match="user_id is required"):
            default_bank_resolver(ctx)

    def test_missing_agent_name_raises(self):
        ctx = self._ctx(agent_name="")
        with pytest.raises(BankResolutionError, match="agent_name is required"):
            default_bank_resolver(ctx)

    def test_custom_resolver(self):
        def my_resolver(ctx: TurnContext) -> str:
            return f"custom:{ctx.user_id}"

        ctx = self._ctx()
        assert my_resolver(ctx) == "custom:user-456"
