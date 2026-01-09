"""Tests for the reflect agent and its tools."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api.engine.mental_models.models import MentalModelType
from hindsight_api.engine.reflect.agent import run_reflect_agent
from hindsight_api.engine.reflect.models import (
    MentalModelInput,
    ReflectAction,
    ReflectActionBatch,
    ReflectAgentResult,
)
from hindsight_api.engine.reflect.tools import (
    generate_model_id,
    tool_expand,
    tool_learn,
    tool_lookup,
    tool_recall,
)


class TestGenerateModelId:
    """Test model ID generation."""

    def test_basic_name(self):
        """Test simple name conversion."""
        assert generate_model_id("My Model") == "my-model"

    def test_special_characters(self):
        """Test name with special characters."""
        assert generate_model_id("Alice's Project (2024)") == "alice-s-project-2024"

    def test_truncation(self):
        """Test long name truncation."""
        long_name = "A" * 100
        result = generate_model_id(long_name)
        assert len(result) <= 50

    def test_leading_trailing_hyphens(self):
        """Test that leading/trailing hyphens are stripped."""
        assert generate_model_id("--Test--") == "test"


class TestToolLookup:
    """Test the lookup tool."""

    @pytest.fixture
    def mock_conn(self):
        """Create a mock database connection."""
        conn = AsyncMock()
        return conn

    async def test_list_all_models(self, mock_conn):
        """Test listing all mental models."""
        mock_conn.fetch.return_value = [
            {
                "id": "model-1",
                "type": "entity",
                "subtype": "learned",
                "name": "Model 1",
                "description": "First model",
            },
            {
                "id": "model-2",
                "type": "concept",
                "subtype": "structural",
                "name": "Model 2",
                "description": "Second model",
            },
        ]

        result = await tool_lookup(mock_conn, "test-bank")

        assert result["count"] == 2
        assert len(result["models"]) == 2
        assert result["models"][0]["id"] == "model-1"
        assert result["models"][1]["id"] == "model-2"

    async def test_get_specific_model(self, mock_conn):
        """Test getting a specific mental model."""
        mock_conn.fetchrow.return_value = {
            "id": "model-1",
            "type": "entity",
            "subtype": "learned",
            "name": "Model 1",
            "description": "First model",
            "summary": "Full summary of model 1",
            "entity_id": None,
            "triggers": ["keyword1", "keyword2"],
            "last_updated": MagicMock(isoformat=lambda: "2024-01-01T00:00:00"),
        }

        result = await tool_lookup(mock_conn, "test-bank", "model-1")

        assert result["found"] is True
        assert result["model"]["id"] == "model-1"
        assert result["model"]["summary"] == "Full summary of model 1"
        assert result["model"]["triggers"] == ["keyword1", "keyword2"]

    async def test_model_not_found(self, mock_conn):
        """Test looking up non-existent model."""
        mock_conn.fetchrow.return_value = None

        result = await tool_lookup(mock_conn, "test-bank", "non-existent")

        assert result["found"] is False
        assert result["model_id"] == "non-existent"


class TestToolLearn:
    """Test the learn tool."""

    @pytest.fixture
    def mock_conn(self):
        """Create a mock database connection."""
        conn = AsyncMock()
        return conn

    async def test_create_new_model(self, mock_conn):
        """Test creating a new mental model."""
        mock_conn.fetchrow.return_value = None  # Model doesn't exist

        input_model = MentalModelInput(
            name="Test Model",
            type=MentalModelType.CONCEPT,
            description="A test model",
            summary="Full summary of the test model",
            triggers=["test", "model"],
        )

        result = await tool_learn(mock_conn, "test-bank", input_model)

        assert result["status"] == "created"
        assert result["model_id"] == "test-model"
        assert result["name"] == "Test Model"
        assert result["type"] == "concept"
        mock_conn.execute.assert_called_once()

    async def test_update_existing_model(self, mock_conn):
        """Test updating an existing mental model."""
        mock_conn.fetchrow.return_value = {"id": "test-model"}  # Model exists

        input_model = MentalModelInput(
            name="Test Model",
            type=MentalModelType.CONCEPT,
            description="Updated description",
            summary="Updated summary",
            triggers=["updated", "triggers"],
        )

        result = await tool_learn(mock_conn, "test-bank", input_model)

        assert result["status"] == "updated"
        assert result["model_id"] == "test-model"

    async def test_learn_with_entity_id(self, mock_conn):
        """Test creating model linked to an entity."""
        mock_conn.fetchrow.return_value = None

        entity_uuid = str(uuid.uuid4())
        input_model = MentalModelInput(
            name="Entity Model",
            type=MentalModelType.ENTITY,
            description="Model linked to entity",
            summary="Summary",
            triggers=["entity"],
            entity_id=entity_uuid,
        )

        result = await tool_learn(mock_conn, "test-bank", input_model)

        assert result["status"] == "created"
        # Verify entity_uuid was passed to the execute call
        call_args = mock_conn.execute.call_args
        assert uuid.UUID(entity_uuid) in call_args[0]


class TestToolExpand:
    """Test the expand tool."""

    @pytest.fixture
    def mock_conn(self):
        """Create a mock database connection."""
        conn = AsyncMock()
        return conn

    async def test_invalid_memory_id(self, mock_conn):
        """Test expand with invalid UUID format."""
        result = await tool_expand(mock_conn, "test-bank", "not-a-uuid", "chunk")

        assert "error" in result
        assert "Invalid memory_id format" in result["error"]

    async def test_memory_not_found(self, mock_conn):
        """Test expand with non-existent memory."""
        mock_conn.fetchrow.return_value = None
        memory_id = str(uuid.uuid4())

        result = await tool_expand(mock_conn, "test-bank", memory_id, "chunk")

        assert "error" in result
        assert "Memory not found" in result["error"]

    async def test_expand_to_chunk(self, mock_conn):
        """Test expanding memory to chunk level."""
        memory_id = uuid.uuid4()
        mock_conn.fetchrow.side_effect = [
            # First call: get memory
            {
                "id": memory_id,
                "text": "Memory text",
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "fact_type": "experience",
                "context": "some context",
            },
            # Second call: get chunk
            {
                "chunk_id": "chunk-1",
                "chunk_text": "Full chunk text with more context",
                "chunk_index": 0,
                "document_id": "doc-1",
            },
        ]

        result = await tool_expand(mock_conn, "test-bank", str(memory_id), "chunk")

        assert "memory" in result
        assert result["memory"]["text"] == "Memory text"
        assert "chunk" in result
        assert result["chunk"]["text"] == "Full chunk text with more context"
        assert "document" not in result  # depth=chunk doesn't include document

    async def test_expand_to_document(self, mock_conn):
        """Test expanding memory to document level."""
        memory_id = uuid.uuid4()
        mock_conn.fetchrow.side_effect = [
            # First call: get memory
            {
                "id": memory_id,
                "text": "Memory text",
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "fact_type": "experience",
                "context": None,
            },
            # Second call: get chunk
            {
                "chunk_id": "chunk-1",
                "chunk_text": "Chunk text",
                "chunk_index": 0,
                "document_id": "doc-1",
            },
            # Third call: get document
            {
                "id": "doc-1",
                "original_text": "Full document text here",
                "metadata": {"source": "test"},
                "retain_params": {},
            },
        ]

        result = await tool_expand(mock_conn, "test-bank", str(memory_id), "document")

        assert "memory" in result
        assert "chunk" in result
        assert "document" in result
        assert result["document"]["full_text"] == "Full document text here"


class TestToolRecall:
    """Test the recall tool."""

    async def test_recall_returns_facts(self):
        """Test recall searches and returns facts."""
        mock_engine = AsyncMock()
        mock_result = MagicMock()
        mock_result.results = [
            MagicMock(
                id=uuid.uuid4(),
                text="Fact 1",
                fact_type="experience",
                entities=["Alice"],
                occurred_start="2024-01-01",
            ),
            MagicMock(
                id=uuid.uuid4(),
                text="Fact 2",
                fact_type="world",
                entities=None,
                occurred_start=None,
            ),
        ]
        mock_engine.recall_async.return_value = mock_result

        mock_request_context = MagicMock()

        result = await tool_recall(mock_engine, "test-bank", "test query", mock_request_context)

        assert result["query"] == "test query"
        assert result["count"] == 2
        assert len(result["facts"]) == 2
        assert result["facts"][0]["text"] == "Fact 1"
        assert result["facts"][0]["entities"] == ["Alice"]

        # Verify recall_async was called with correct params
        mock_engine.recall_async.assert_called_once()
        call_kwargs = mock_engine.recall_async.call_args[1]
        assert call_kwargs["bank_id"] == "test-bank"
        assert call_kwargs["query"] == "test query"
        assert call_kwargs["fact_type"] == ["experience", "world"]  # No opinions


class TestReflectAgent:
    """Test the reflect agent loop."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM provider."""
        llm = AsyncMock()
        return llm

    @pytest.fixture
    def bank_profile(self):
        """Create a test bank profile."""
        return {
            "name": "Test Assistant",
            "background": "A helpful test assistant",
            "goal": "Help with testing",
        }

    @pytest.fixture
    def mock_tools(self):
        """Create mock tool callbacks."""
        return {
            "lookup_fn": AsyncMock(return_value={"count": 0, "models": []}),
            "recall_fn": AsyncMock(return_value={"query": "test", "count": 1, "facts": [{"text": "Fact"}]}),
            "learn_fn": AsyncMock(return_value={"status": "created", "model_id": "new-model"}),
            "expand_fn": AsyncMock(return_value={"memory": {"id": "123", "text": "Memory text"}}),
        }

    async def test_agent_done_immediately(self, mock_llm, bank_profile, mock_tools):
        """Test agent that signals done on first iteration."""
        mock_llm.call.return_value = ReflectActionBatch(
            actions=[ReflectAction(tool="done", answer="The answer is 42.")]
        )

        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="What is the answer?",
            bank_profile=bank_profile,
            **mock_tools,
        )

        assert isinstance(result, ReflectAgentResult)
        assert result.text == "The answer is 42."
        assert result.iterations == 1
        assert result.tools_called == 0

    async def test_agent_calls_tools_then_done(self, mock_llm, bank_profile, mock_tools):
        """Test agent that calls tools before completing."""
        # First call: lookup and recall
        # Second call: done
        mock_llm.call.side_effect = [
            ReflectActionBatch(
                actions=[
                    ReflectAction(tool="lookup", reasoning="Check models"),
                    ReflectAction(tool="recall", query="test query", reasoning="Search"),
                ]
            ),
            ReflectActionBatch(actions=[ReflectAction(tool="done", answer="Based on my research, the answer is yes.")]),
        ]

        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="Is testing important?",
            bank_profile=bank_profile,
            **mock_tools,
        )

        assert result.text == "Based on my research, the answer is yes."
        assert result.iterations == 2
        assert result.tools_called == 2
        mock_tools["lookup_fn"].assert_called_once()
        mock_tools["recall_fn"].assert_called_once_with("test query")

    async def test_agent_learns_model(self, mock_llm, bank_profile, mock_tools):
        """Test agent that creates a mental model."""
        mock_llm.call.side_effect = [
            ReflectActionBatch(
                actions=[
                    ReflectAction(
                        tool="learn",
                        mental_model=MentalModelInput(
                            name="New Insight",
                            type=MentalModelType.CONCEPT,
                            description="A new concept",
                            summary="Detailed understanding",
                            triggers=["insight"],
                        ),
                        reasoning="Learned something new",
                    )
                ]
            ),
            ReflectActionBatch(actions=[ReflectAction(tool="done", answer="I've learned something new.")]),
        ]

        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="What can you learn?",
            bank_profile=bank_profile,
            **mock_tools,
        )

        assert result.mental_models_created == ["new-model"]
        mock_tools["learn_fn"].assert_called_once()

    async def test_agent_max_iterations_forces_response(self, mock_llm, bank_profile, mock_tools):
        """Test that max iterations forces a text response."""
        # Return tools indefinitely
        mock_llm.call.side_effect = [
            ReflectActionBatch(actions=[ReflectAction(tool="recall", query="query", reasoning="Search")]),
            ReflectActionBatch(actions=[ReflectAction(tool="recall", query="query2", reasoning="Search")]),
            # On last iteration, LLM should be called without tools, returning text
            "Forced final answer after max iterations.",
        ]

        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="Test question",
            bank_profile=bank_profile,
            max_iterations=3,
            **mock_tools,
        )

        assert result.text == "Forced final answer after max iterations."
        assert result.iterations == 3

    async def test_agent_handles_tool_error(self, mock_llm, bank_profile, mock_tools):
        """Test agent handles tool execution errors gracefully."""
        mock_tools["recall_fn"].side_effect = Exception("Database error")

        mock_llm.call.side_effect = [
            ReflectActionBatch(actions=[ReflectAction(tool="recall", query="query", reasoning="Search")]),
            ReflectActionBatch(actions=[ReflectAction(tool="done", answer="Could not complete due to error.")]),
        ]

        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="Test question",
            bank_profile=bank_profile,
            **mock_tools,
        )

        assert result.text == "Could not complete due to error."
        assert result.tools_called == 1  # Tool was attempted

    async def test_agent_parallel_tool_calls(self, mock_llm, bank_profile, mock_tools):
        """Test agent executes multiple tools in parallel."""
        mock_llm.call.side_effect = [
            ReflectActionBatch(
                actions=[
                    ReflectAction(tool="lookup", reasoning="Check models"),
                    ReflectAction(tool="recall", query="query1", reasoning="Search 1"),
                    ReflectAction(tool="recall", query="query2", reasoning="Search 2"),
                ]
            ),
            ReflectActionBatch(actions=[ReflectAction(tool="done", answer="Done after parallel calls.")]),
        ]

        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="Test question",
            bank_profile=bank_profile,
            **mock_tools,
        )

        assert result.tools_called == 3
        # recall should be called twice
        assert mock_tools["recall_fn"].call_count == 2


@pytest.mark.integration
class TestReflectIntegration:
    """Integration tests for reflect with real database.

    These tests require a running database and LLM provider.
    Skip with: pytest -m "not integration"
    """

    async def test_reflect_creates_learned_mental_model(self, memory, request_context):
        """Test that reflect can create a 'learned' mental model via the agent."""
        bank_id = f"test-reflect-{uuid.uuid4().hex[:8]}"

        # Add some test data
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {"content": "Alice is the team lead and manages the engineering team."},
                {"content": "The team has weekly planning meetings on Monday."},
                {"content": "Alice prefers asynchronous communication via Slack."},
            ],
            request_context=request_context,
        )
        await memory.wait_for_background_tasks()

        # Run reflect - this should use the agentic loop
        result = await memory.reflect_async(
            bank_id=bank_id,
            query="What do you know about Alice and how she manages the team?",
            request_context=request_context,
        )

        assert result.text is not None
        assert len(result.text) > 0

        # Check if any mental models were created (may or may not happen depending on LLM)
        models = await memory.list_mental_models(
            bank_id=bank_id,
            request_context=request_context,
        )

        # If models were created, they should be 'learned' subtype
        for model in models:
            if model.get("subtype") == "learned":
                assert model["type"] in ["entity", "concept", "event"]

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_reflect_excludes_opinions_from_recall(self, memory, request_context):
        """Test that reflect's recall tool doesn't return opinions."""
        bank_id = f"test-reflect-no-opinions-{uuid.uuid4().hex[:8]}"

        # Add test data (note: we can't directly add opinions since opinion
        # extraction was removed, but we can verify recall behavior)
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {"content": "The weather today is sunny and warm."},
            ],
            request_context=request_context,
        )
        await memory.wait_for_background_tasks()

        # Run recall directly to verify it excludes opinions
        recall_result = await memory.recall_async(
            bank_id=bank_id,
            query="weather",
            request_context=request_context,
        )

        # All returned facts should be experience or world, not opinion
        for fact in recall_result.results:
            assert fact.fact_type in ["experience", "world"]
            assert fact.fact_type != "opinion"

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)
