"""Reflect sub-recalls must request entity resolution.

Canonical entity names are semantic signal the surface text may lack ("Bob"
in the text vs canonical "Robert Smith"). `recall_async` only populates each
result's `entities` field when `include_entities=True`, and it defaults to
False — so both reflect retrieval tools must pass it explicitly, and the
names must survive the serialization into the tool result the agent reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api.engine.reflect.tools import tool_recall, tool_search_observations
from hindsight_api.engine.response_models import MemoryFact, RecallResult


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


def _fact_with_entities() -> MemoryFact:
    return MemoryFact(
        id="123e4567-e89b-12d3-a456-426614174000",
        text="Bob moved the deploy to 09:00 UTC.",
        fact_type="world",
        entities=["Robert Smith"],
    )


def _mock_engine(results: list[MemoryFact] | None = None):
    engine = MagicMock()
    engine.recall_async = AsyncMock(return_value=RecallResult(results=results or [], source_facts={}))
    return engine


class TestReflectRecallRequestsEntities:
    @pytest.mark.asyncio
    async def test_recall_passes_include_entities(self):
        engine = _mock_engine()

        await tool_recall(engine, "bank-1", "query", _FakeRequestContext())

        assert engine.recall_async.call_args.kwargs["include_entities"] is True

    @pytest.mark.asyncio
    async def test_search_observations_passes_include_entities(self):
        engine = _mock_engine()

        await tool_search_observations(engine, "bank-1", "query", _FakeRequestContext())

        assert engine.recall_async.call_args.kwargs["include_entities"] is True

    @pytest.mark.asyncio
    async def test_entity_names_reach_the_agent(self):
        """End-to-end through the tool's serialization (null-pruning, field
        trimming): the canonical names land in the payload the agent reads."""
        engine = _mock_engine(results=[_fact_with_entities()])

        result = await tool_recall(engine, "bank-1", "query", _FakeRequestContext())

        assert result["memories"][0]["entities"] == ["Robert Smith"]

    @pytest.mark.asyncio
    async def test_observation_entity_names_reach_the_agent(self):
        engine = _mock_engine(results=[_fact_with_entities()])

        result = await tool_search_observations(engine, "bank-1", "query", _FakeRequestContext())

        assert result["observations"][0]["entities"] == ["Robert Smith"]
