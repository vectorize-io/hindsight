"""
Tests for agent management API (profile, disposition, background).
"""
import pytest
import uuid
from hindsight_api import MemoryEngine, RequestContext
from hindsight_api.api import CreateBankRequest, DispositionTraits
from hindsight_api.engine.memory_engine import Budget


def unique_agent_id(prefix: str) -> str:
    """Generate a unique agent ID for testing."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class TestAgentProfile:
    """Tests for agent profile management."""

    @pytest.mark.asyncio
    async def test_get_agent_profile_creates_default(self, memory: MemoryEngine, request_context):
        """Test that getting a profile for a new agent creates default disposition."""
        bank_id = unique_agent_id("test_profile_default")

        profile = await memory.get_bank_profile(bank_id, request_context=request_context)

        assert profile is not None
        assert "disposition" in profile
        assert "background" in profile

        disposition = profile["disposition"]
        assert disposition.skepticism == 3
        assert disposition.literalism == 3
        assert disposition.empathy == 3

        assert profile["background"] == ""

    @pytest.mark.asyncio
    async def test_update_agent_disposition(self, memory: MemoryEngine, request_context):
        """Test updating agent disposition traits."""
        bank_id = unique_agent_id("test_profile_update")

        profile = await memory.get_bank_profile(bank_id, request_context=request_context)
        assert profile["disposition"].skepticism == 3

        new_disposition = {
            "skepticism": 5,
            "literalism": 4,
            "empathy": 2,
        }
        await memory.update_bank_disposition(bank_id, new_disposition, request_context=request_context)

        updated_profile = await memory.get_bank_profile(bank_id, request_context=request_context)
        disposition = updated_profile["disposition"]
        assert disposition.skepticism == new_disposition["skepticism"]
        assert disposition.literalism == new_disposition["literalism"]
        assert disposition.empathy == new_disposition["empathy"]

    @pytest.mark.asyncio
    async def test_list_agents(self, memory: MemoryEngine, request_context):
        """Test listing all agents."""
        agent_id_1 = unique_agent_id("test_list")
        agent_id_2 = unique_agent_id("test_list")
        agent_id_3 = unique_agent_id("test_list")

        await memory.get_bank_profile(agent_id_1, request_context=request_context)
        await memory.get_bank_profile(agent_id_2, request_context=request_context)
        await memory.get_bank_profile(agent_id_3, request_context=request_context)

        agents = await memory.list_banks(request_context=request_context)

        agent_ids = [a["bank_id"] for a in agents]
        assert agent_id_1 in agent_ids
        assert agent_id_2 in agent_ids
        assert agent_id_3 in agent_ids

        for agent in agents:
            assert "bank_id" in agent
            assert "disposition" in agent
            assert "background" in agent
            assert "created_at" in agent
            assert "updated_at" in agent


class TestAgentBackground:
    """Tests for agent background management."""

    @pytest.mark.asyncio
    async def test_merge_agent_background(self, memory: MemoryEngine, request_context):
        """Test merging agent background information."""
        bank_id = unique_agent_id("test_profile_merge")

        profile = await memory.get_bank_profile(bank_id, request_context=request_context)
        assert profile["background"] == ""

        result1 = await memory.merge_bank_background(
            bank_id,
            "I was born in Texas",
            update_disposition=False,
            request_context=request_context,
        )
        assert "Texas" in result1["background"]

        result2 = await memory.merge_bank_background(
            bank_id,
            "I have 10 years of startup experience",
            update_disposition=False,
            request_context=request_context,
        )
        assert "Texas" in result2["background"] or "startup" in result2["background"]

        final_profile = await memory.get_bank_profile(bank_id, request_context=request_context)
        assert final_profile["background"] != ""

    @pytest.mark.asyncio
    async def test_merge_background_handles_conflicts(self, memory: MemoryEngine, request_context):
        """Test that merging background handles conflicts (new overwrites old)."""
        bank_id = unique_agent_id("test_profile_conflict")

        result1 = await memory.merge_bank_background(
            bank_id,
            "I was born in Colorado",
            update_disposition=False,
            request_context=request_context,
        )
        assert "Colorado" in result1["background"]

        result2 = await memory.merge_bank_background(
            bank_id,
            "You were born in Texas",
            update_disposition=False,
            request_context=request_context,
        )
        assert "Texas" in result2["background"]


class TestAgentEndpoint:
    """Tests for agent PUT endpoint logic."""

    @pytest.mark.asyncio
    async def test_put_agent_create(self, memory: MemoryEngine, request_context):
        """Test creating an agent via PUT endpoint."""
        bank_id = unique_agent_id("test_put_create")

        request = CreateBankRequest(
            disposition=DispositionTraits(
                skepticism=4,
                literalism=5,
                empathy=2
            ),
            background="I am a creative software engineer"
        )

        profile = await memory.get_bank_profile(bank_id, request_context=request_context)

        if request.disposition is not None:
            await memory.update_bank_disposition(
                bank_id,
                request.disposition.model_dump(),
                request_context=request_context,
            )

        if request.background is not None:
            pool = await memory._get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE banks
                    SET background = $2,
                        updated_at = NOW()
                    WHERE bank_id = $1
                    """,
                    bank_id,
                    request.background
                )

        final_profile = await memory.get_bank_profile(bank_id, request_context=request_context)

        assert final_profile["disposition"].skepticism == 4
        assert final_profile["disposition"].literalism == 5
        assert final_profile["background"] == "I am a creative software engineer"

    @pytest.mark.asyncio
    async def test_put_agent_partial_update(self, memory: MemoryEngine, request_context):
        """Test updating only background."""
        bank_id = unique_agent_id("test_put_partial")

        request = CreateBankRequest(
            background="I am a data scientist"
        )

        profile = await memory.get_bank_profile(bank_id, request_context=request_context)

        if request.background is not None:
            pool = await memory._get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE banks
                    SET background = $2,
                        updated_at = NOW()
                    WHERE bank_id = $1
                    """,
                    bank_id,
                    request.background
                )

        final_profile = await memory.get_bank_profile(bank_id, request_context=request_context)

        assert final_profile["disposition"].skepticism == 3  # Default
        assert final_profile["background"] == "I am a data scientist"


class TestAgentDispositionIntegration:
    """Tests for disposition integration with other features."""

    @pytest.mark.asyncio
    async def test_think_uses_disposition(self, memory: MemoryEngine, request_context):
        """Test that THINK operation uses agent disposition."""
        bank_id = unique_agent_id("test_think")

        disposition = {
            "skepticism": 5,  # Very skeptical
            "literalism": 4,  # High literalism
            "empathy": 2,     # Low empathy
        }
        await memory.update_bank_disposition(bank_id, disposition, request_context=request_context)

        await memory.merge_bank_background(
            bank_id,
            "I am a creative artist who values innovation over tradition",
            update_disposition=False,
            request_context=request_context,
        )

        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {"content": "Traditional painting techniques have been used for centuries"},
                {"content": "Modern digital art is changing the art world"}
            ],
            request_context=request_context,
        )

        result = await memory.reflect_async(
            bank_id=bank_id,
            query="What do you think about traditional vs modern art?",
            budget=Budget.LOW,
            request_context=request_context,
        )

        assert result.text is not None
        assert len(result.text) > 0
