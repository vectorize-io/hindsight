"""Regression test for #972: reflect sub-recalls must be marked internal.

When reflect calls search_observations or recall, the sub-recalls must use
``request_context.internal=True`` to avoid double-billing.  The reflect caller
is already billed for the overall operation; sub-recalls are implementation
details that should not generate additional billing events.
"""

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api.engine.reflect.tools import tool_recall, tool_search_observations
from hindsight_api.engine.response_models import RecallResult


@dataclass
class _FakeRequestContext:
    """Dataclass stand-in matching the fields used by ``dataclasses.replace``."""

    api_key: str | None = None
    api_key_id: str | None = None
    tenant_id: str | None = None
    internal: bool = False
    mcp_authenticated: bool = False
    user_initiated: bool = False
    allowed_bank_ids: list[str] | None = None


def _mock_engine():
    engine = MagicMock()
    engine.recall_async = AsyncMock(
        return_value=RecallResult(results=[], source_facts={})
    )
    return engine


class TestReflectInternalBilling:
    """Verify that reflect sub-recalls are marked internal (#972)."""

    @pytest.mark.asyncio
    async def test_search_observations_marks_recall_internal(self):
        engine = _mock_engine()
        ctx = _FakeRequestContext(api_key="k", internal=False)

        await tool_search_observations(engine, "bank-1", "query", ctx)

        engine.recall_async.assert_called_once()
        passed_ctx = engine.recall_async.call_args.kwargs["request_context"]
        assert passed_ctx.internal is True, "sub-recall must be internal"

    @pytest.mark.asyncio
    async def test_search_observations_preserves_original_context(self):
        engine = _mock_engine()
        ctx = _FakeRequestContext(api_key="k", internal=False)

        await tool_search_observations(engine, "bank-1", "query", ctx)

        assert ctx.internal is False, "original context must not be mutated"

    @pytest.mark.asyncio
    async def test_recall_marks_recall_internal(self):
        engine = _mock_engine()
        ctx = _FakeRequestContext(api_key="k", internal=False)

        await tool_recall(engine, "bank-1", "query", ctx)

        engine.recall_async.assert_called_once()
        passed_ctx = engine.recall_async.call_args.kwargs["request_context"]
        assert passed_ctx.internal is True, "sub-recall must be internal"

    @pytest.mark.asyncio
    async def test_recall_preserves_original_context(self):
        engine = _mock_engine()
        ctx = _FakeRequestContext(api_key="k", internal=False)

        await tool_recall(engine, "bank-1", "query", ctx)

        assert ctx.internal is False, "original context must not be mutated"
