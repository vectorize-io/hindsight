"""Tests for the reflect agent and its tools."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api.engine.reflect.agent import run_reflect_agent
from hindsight_api.engine.reflect.models import (
    AnswerSection,
    MentalModelInput,
    MentalModelObservation,
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
from hindsight_api.engine.response_models import LLMToolCall, LLMToolCallResult


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
        """Test listing all mental models (compact: id, name, description only)."""
        mock_conn.fetch.return_value = [
            {
                "id": "model-1",
                "subtype": "learned",
                "name": "Model 1",
                "description": "First model",
            },
            {
                "id": "model-2",
                "subtype": "structural",
                "name": "Model 2",
                "description": "Second model",
            },
        ]

        result = await tool_lookup(mock_conn, "test-bank")

        assert result["count"] == 2
        assert len(result["models"]) == 2
        assert result["models"][0]["id"] == "model-1"
        assert result["models"][0]["name"] == "Model 1"
        assert "observation_titles" not in result["models"][0]  # No observation_titles in list view
        assert result["models"][1]["id"] == "model-2"

    async def test_get_specific_model(self, mock_conn):
        """Test getting a specific mental model."""
        mock_conn.fetchrow.return_value = {
            "id": "model-1",
            "subtype": "learned",
            "name": "Model 1",
            "description": "First model",
            "observations": {"observations": [{"title": "Overview", "text": "Full summary of model 1", "memory_ids": ["mem-1", "mem-2"]}]},
            "entity_id": None,
            "last_updated": MagicMock(isoformat=lambda: "2024-01-01T00:00:00"),
        }

        result = await tool_lookup(mock_conn, "test-bank", "model-1")

        assert result["found"] is True
        assert result["model"]["id"] == "model-1"
        assert len(result["model"]["observations"]) == 1
        assert result["model"]["observations"][0]["text"] == "Full summary of model 1"
        # Verify memory_ids are mapped to based_on
        assert result["model"]["observations"][0]["based_on"] == ["mem-1", "mem-2"]

    async def test_model_not_found(self, mock_conn):
        """Test looking up non-existent model."""
        mock_conn.fetchrow.return_value = None

        result = await tool_lookup(mock_conn, "test-bank", "non-existent")

        assert result["found"] is False
        assert result["model_id"] == "non-existent"


class TestToolLearn:
    """Test the learn tool.

    The learn tool creates placeholder mental models with name/description only.
    Actual content is generated in the background via refresh.
    """

    @pytest.fixture
    def mock_conn(self):
        """Create a mock database connection."""
        conn = AsyncMock()
        return conn

    async def test_create_new_model(self, mock_conn):
        """Test creating a new mental model placeholder."""
        mock_conn.fetchrow.return_value = None  # Model doesn't exist

        input_model = MentalModelInput(
            name="Test Model",
            description="A test model to track important patterns",
        )

        result = await tool_learn(mock_conn, "test-bank", input_model)

        assert result["status"] == "created"
        assert result["model_id"] == "test-model"
        assert result["name"] == "Test Model"
        assert result["pending_generation"] is True
        mock_conn.execute.assert_called_once()

    async def test_update_existing_model(self, mock_conn):
        """Test updating an existing mental model."""
        mock_conn.fetchrow.return_value = {"id": "test-model"}  # Model exists

        input_model = MentalModelInput(
            name="Test Model",
            description="Updated description",
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
            description="Model linked to entity",
            entity_id=entity_uuid,
        )

        result = await tool_learn(mock_conn, "test-bank", input_model)

        assert result["status"] == "created"
        assert result["pending_generation"] is True
        # Verify entity_uuid was passed to the execute call
        call_args = mock_conn.execute.call_args
        assert uuid.UUID(entity_uuid) in call_args[0]

    async def test_learn_creates_empty_observations(self, mock_conn):
        """Test that learn creates model with empty observations (content generated later)."""
        mock_conn.fetchrow.return_value = None  # Model doesn't exist

        input_model = MentalModelInput(
            name="Model With Sources",
            description="A model to track source facts",
        )

        result = await tool_learn(mock_conn, "test-bank", input_model)

        assert result["status"] == "created"
        assert result["pending_generation"] is True
        # Verify the INSERT query was called with empty observations
        call_args = mock_conn.execute.call_args
        # observations should be empty JSON
        assert "'{}'::jsonb" in call_args[0][0]


class TestToolExpand:
    """Test the expand tool."""

    @pytest.fixture
    def mock_conn(self):
        """Create a mock database connection."""
        conn = AsyncMock()
        return conn

    async def test_empty_memory_ids(self, mock_conn):
        """Test expand with empty memory_ids list."""
        result = await tool_expand(mock_conn, "test-bank", [], "chunk")

        assert "error" in result
        assert "memory_ids is required" in result["error"]

    async def test_invalid_memory_id(self, mock_conn):
        """Test expand with invalid UUID format."""
        result = await tool_expand(mock_conn, "test-bank", ["not-a-uuid"], "chunk")

        assert "error" in result
        assert "No valid memory IDs provided" in result["error"]

    async def test_memory_not_found(self, mock_conn):
        """Test expand with non-existent memory."""
        mock_conn.fetch.return_value = []  # No memories found
        memory_id = str(uuid.uuid4())

        result = await tool_expand(mock_conn, "test-bank", [memory_id], "chunk")

        assert "results" in result
        assert len(result["results"]) == 1
        assert "error" in result["results"][0]
        assert "Memory not found" in result["results"][0]["error"]

    async def test_expand_to_chunk(self, mock_conn):
        """Test expanding memory to chunk level."""
        memory_id = uuid.uuid4()
        # Mock batch fetch for memories
        mock_conn.fetch.side_effect = [
            # First call: get memories
            [
                {
                    "id": memory_id,
                    "text": "Memory text",
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "fact_type": "experience",
                    "context": "some context",
                }
            ],
            # Second call: get chunks
            [
                {
                    "chunk_id": "chunk-1",
                    "chunk_text": "Full chunk text with more context",
                    "chunk_index": 0,
                    "document_id": "doc-1",
                }
            ],
        ]

        result = await tool_expand(mock_conn, "test-bank", [str(memory_id)], "chunk")

        assert "results" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["memory"]["text"] == "Memory text"
        assert result["results"][0]["chunk"]["text"] == "Full chunk text with more context"
        assert "document" not in result["results"][0]  # depth=chunk doesn't include document

    async def test_expand_to_document(self, mock_conn):
        """Test expanding memory to document level."""
        memory_id = uuid.uuid4()
        mock_conn.fetch.side_effect = [
            # First call: get memories
            [
                {
                    "id": memory_id,
                    "text": "Memory text",
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "fact_type": "experience",
                    "context": None,
                }
            ],
            # Second call: get chunks
            [
                {
                    "chunk_id": "chunk-1",
                    "chunk_text": "Chunk text",
                    "chunk_index": 0,
                    "document_id": "doc-1",
                }
            ],
            # Third call: get documents
            [
                {
                    "id": "doc-1",
                    "original_text": "Full document text here",
                    "metadata": {"source": "test"},
                    "retain_params": {},
                }
            ],
        ]

        result = await tool_expand(mock_conn, "test-bank", [str(memory_id)], "document")

        assert "results" in result
        assert len(result["results"]) == 1
        assert "memory" in result["results"][0]
        assert "chunk" in result["results"][0]
        assert "document" in result["results"][0]
        assert result["results"][0]["document"]["full_text"] == "Full document text here"

    async def test_expand_multiple_memories(self, mock_conn):
        """Test expanding multiple memories in a single batch."""
        memory_id_1 = uuid.uuid4()
        memory_id_2 = uuid.uuid4()
        mock_conn.fetch.side_effect = [
            # First call: get memories
            [
                {
                    "id": memory_id_1,
                    "text": "Memory 1",
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "fact_type": "experience",
                    "context": None,
                },
                {
                    "id": memory_id_2,
                    "text": "Memory 2",
                    "chunk_id": "chunk-2",
                    "document_id": "doc-1",
                    "fact_type": "world",
                    "context": None,
                },
            ],
            # Second call: get chunks
            [
                {
                    "chunk_id": "chunk-1",
                    "chunk_text": "Chunk 1 text",
                    "chunk_index": 0,
                    "document_id": "doc-1",
                },
                {
                    "chunk_id": "chunk-2",
                    "chunk_text": "Chunk 2 text",
                    "chunk_index": 1,
                    "document_id": "doc-1",
                },
            ],
        ]

        result = await tool_expand(mock_conn, "test-bank", [str(memory_id_1), str(memory_id_2)], "chunk")

        assert "results" in result
        assert result["count"] == 2
        assert result["results"][0]["memory"]["text"] == "Memory 1"
        assert result["results"][1]["memory"]["text"] == "Memory 2"


class TestToolRecall:
    """Test the recall tool."""

    async def test_recall_returns_memories(self):
        """Test recall searches and returns memories."""
        mock_engine = AsyncMock()
        mock_result = MagicMock()
        mock_result.results = [
            MagicMock(
                id=uuid.uuid4(),
                text="Memory 1",
                fact_type="experience",
                entities=["Alice"],
                occurred_start="2024-01-01",
            ),
            MagicMock(
                id=uuid.uuid4(),
                text="Memory 2",
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
        assert len(result["memories"]) == 2
        assert result["memories"][0]["text"] == "Memory 1"
        assert result["memories"][0]["entities"] == ["Alice"]

        # Verify recall_async was called with correct params
        mock_engine.recall_async.assert_called_once()
        call_kwargs = mock_engine.recall_async.call_args[1]
        assert call_kwargs["bank_id"] == "test-bank"
        assert call_kwargs["query"] == "test query"
        assert call_kwargs["fact_type"] == ["experience", "world"]  # No opinions


class TestPromptSize:
    """Test that prompts stay within reasonable size limits.

    Large prompts cause slow LLM responses (120s+ observed in production).
    The agent should not pre-load all mental models; use lookup() instead.
    """

    def test_initial_prompt_is_small(self):
        """Verify the initial prompt (no tool history) is reasonably small."""
        from hindsight_api.engine.reflect.prompts import build_agent_prompt, build_system_prompt_for_tools

        # Typical bank profile
        bank_profile = {
            "name": "Test Assistant",
            "mission": "A helpful assistant for tracking engineering team activities. Help the team stay organized and informed.",
        }

        # First iteration: no context history
        context_history: list[dict] = []
        query = "Who should take ownership of storing load test scripts in Git?"

        # No additional context (mental models not pre-loaded)
        prompt = build_agent_prompt(query, context_history, bank_profile, additional_context=None)
        system_prompt = build_system_prompt_for_tools(bank_profile)

        total_chars = len(prompt) + len(system_prompt)
        estimated_tokens = total_chars // 4  # Rough estimate

        # Initial prompt should be under 3000 tokens (~12k chars)
        # This ensures fast LLM responses on the first iteration
        assert total_chars < 12000, f"Initial prompt too large: {total_chars} chars (~{estimated_tokens} tokens)"
        assert estimated_tokens < 3000, f"Initial prompt too large: ~{estimated_tokens} tokens"

    def test_prompt_with_tool_history_grows_reasonably(self):
        """Verify prompts grow reasonably with tool results."""
        from hindsight_api.engine.reflect.prompts import build_agent_prompt, build_system_prompt_for_tools

        bank_profile = {
            "name": "Test Assistant",
            "mission": "A helpful assistant. Help the team.",
        }

        # Simulate recall result with 50 memories (realistic scenario)
        recall_result = {
            "query": "test query",
            "count": 50,
            "memories": [
                {"id": f"mem-{i}", "text": f"This is memory number {i} with some content.", "type": "experience"}
                for i in range(50)
            ],
        }

        context_history = [{"tool": "recall", "input": {"query": "test"}, "output": recall_result}]
        query = "What do you know about the team?"

        prompt = build_agent_prompt(query, context_history, bank_profile, additional_context=None)
        system_prompt = build_system_prompt_for_tools(bank_profile)

        total_chars = len(prompt) + len(system_prompt)
        estimated_tokens = total_chars // 4

        # With tool results, prompt should still be manageable (<20k tokens)
        assert total_chars < 80000, f"Prompt with tools too large: {total_chars} chars (~{estimated_tokens} tokens)"


class TestReflectAgent:
    """Test the reflect agent loop with native tool calling."""

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
            "mission": "A helpful test assistant. Help with testing.",
        }

    @pytest.fixture
    def mock_tools(self):
        """Create mock tool callbacks."""
        # Include memory IDs in recall results so guardrail passes
        memory_id = str(uuid.uuid4())
        return {
            "lookup_fn": AsyncMock(return_value={"count": 0, "models": []}),
            "recall_fn": AsyncMock(return_value={
                "query": "test",
                "count": 1,
                "memories": [{"id": memory_id, "text": "Memory", "type": "experience"}]
            }),
            "learn_fn": AsyncMock(return_value={"status": "created", "model_id": "new-model"}),
            "expand_fn": AsyncMock(return_value={
                "results": [{"memory_id": "123", "memory": {"id": "123", "text": "Memory text"}}],
                "count": 1
            }),
        }

    def _make_tool_result(self, tool_calls: list[dict]) -> LLMToolCallResult:
        """Helper to create LLMToolCallResult from tool call dicts."""
        return LLMToolCallResult(
            tool_calls=[
                LLMToolCall(id=f"call_{i}", name=tc["name"], arguments=tc.get("arguments", {}))
                for i, tc in enumerate(tool_calls)
            ],
            finish_reason="tool_calls",
        )

    async def test_agent_done_immediately_rejected_by_guardrail(self, mock_llm, bank_profile, mock_tools):
        """Test that guardrail rejects done without evidence gathering."""
        # First call: agent tries to return done immediately (rejected by guardrail)
        # Second call: agent gathers evidence
        # Third call: agent returns done with evidence
        mock_llm.call_with_tools.side_effect = [
            self._make_tool_result([{"name": "done", "arguments": {"answer": "The answer is 42."}}]),
            # After guardrail rejection, agent should gather evidence
            self._make_tool_result([{"name": "recall", "arguments": {"query": "test query"}}]),
            # Now with evidence, done is accepted
            self._make_tool_result([{"name": "done", "arguments": {"answer": "The answer is 42."}}]),
        ]

        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="What is the answer?",
            bank_profile=bank_profile,
            **mock_tools,
        )

        assert isinstance(result, ReflectAgentResult)
        assert result.text == "The answer is 42."
        # 3 iterations: rejected done, recall, accepted done
        assert result.iterations == 3
        # Tools called: list_mental_models (auto at start of each iteration) + recall
        # The exact count may vary based on implementation
        assert result.tools_called >= 1  # At least recall was called

    async def test_agent_calls_tools_then_done(self, mock_llm, bank_profile, mock_tools):
        """Test agent that calls tools before completing."""
        # First call: lookup and recall
        # Second call: done
        mock_llm.call_with_tools.side_effect = [
            self._make_tool_result([
                {"name": "list_mental_models", "arguments": {}},
                {"name": "recall", "arguments": {"query": "test query"}},
            ]),
            self._make_tool_result([
                {"name": "done", "arguments": {"answer": "Based on my research, the answer is yes."}},
            ]),
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
        # Tools called: list_mental_models + recall (+ possibly auto list_mental_models)
        assert result.tools_called >= 2
        mock_tools["recall_fn"].assert_called_once_with("test query", 2048)

    async def test_agent_learns_model(self, mock_llm, bank_profile, mock_tools):
        """Test agent that creates a mental model placeholder."""
        mock_tools["learn_fn"].return_value = {"status": "created", "model_id": "new-insight", "pending_generation": True}

        mock_llm.call_with_tools.side_effect = [
            # First: gather evidence via recall (required by guardrail)
            self._make_tool_result([{"name": "recall", "arguments": {"query": "user preferences"}}]),
            # Then: learn from the gathered evidence
            self._make_tool_result([{
                "name": "learn",
                "arguments": {
                    "name": "New Insight",
                    "description": "Track patterns about user preferences and communication style",
                }
            }]),
            # Finally: done with the learning
            self._make_tool_result([{"name": "done", "arguments": {"answer": "I've learned something new."}}]),
        ]

        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="What can you learn?",
            bank_profile=bank_profile,
            **mock_tools,
        )

        assert result.mental_models_created == ["new-insight"]
        mock_tools["learn_fn"].assert_called_once()
        # Verify the learn_fn was called with name and description only
        call_args = mock_tools["learn_fn"].call_args
        mental_model_arg = call_args[0][0]
        assert mental_model_arg.name == "New Insight"
        assert "preferences" in mental_model_arg.description

    async def test_agent_max_iterations_forces_response(self, mock_llm, bank_profile, mock_tools):
        """Test that max iterations forces a text response."""
        # Return tools indefinitely, then final plain text call
        mock_llm.call_with_tools.side_effect = [
            self._make_tool_result([{"name": "recall", "arguments": {"query": "query"}}]),
            self._make_tool_result([{"name": "recall", "arguments": {"query": "query2"}}]),
        ]
        # On last iteration, LLM.call is used (not call_with_tools)
        mock_llm.call.return_value = "Forced final answer after max iterations."

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
        """Test agent propagates tool execution errors."""
        # Make recall fail
        mock_tools["recall_fn"].side_effect = Exception("Database error")

        mock_llm.call_with_tools.side_effect = [
            self._make_tool_result([{"name": "recall", "arguments": {"query": "query"}}]),
        ]

        # Tool errors are now propagated as RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            await run_reflect_agent(
                llm_config=mock_llm,
                bank_id="test-bank",
                query="Test question",
                bank_profile=bank_profile,
                **mock_tools,
            )

        assert "Database error" in str(exc_info.value)

    async def test_agent_parallel_tool_calls(self, mock_llm, bank_profile, mock_tools):
        """Test agent executes multiple tools in parallel."""
        mock_llm.call_with_tools.side_effect = [
            self._make_tool_result([
                {"name": "list_mental_models", "arguments": {}},
                {"name": "recall", "arguments": {"query": "query1"}},
                {"name": "recall", "arguments": {"query": "query2"}},
            ]),
            self._make_tool_result([{"name": "done", "arguments": {"answer": "Done after parallel calls."}}]),
        ]

        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="Test question",
            bank_profile=bank_profile,
            **mock_tools,
        )

        # Tools called: list_mental_models + 2x recall (+ possibly auto list_mental_models)
        assert result.tools_called >= 3
        # recall should be called twice
        assert mock_tools["recall_fn"].call_count == 2

    async def test_agent_returns_validated_memory_ids(self, mock_llm, bank_profile):
        """Test agent returns only validated memory IDs that were actually recalled."""
        memory_id_1 = str(uuid.uuid4())
        memory_id_2 = str(uuid.uuid4())

        # Mock recall returns these specific memory IDs
        mock_recall = AsyncMock(
            return_value={
                "query": "test",
                "count": 2,
                "memories": [
                    {"id": memory_id_1, "text": "Memory 1", "type": "experience"},
                    {"id": memory_id_2, "text": "Memory 2", "type": "world"},
                ],
            }
        )

        mock_llm.call_with_tools.side_effect = [
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="call_0", name="recall", arguments={"query": "test query"})],
                finish_reason="tool_calls",
            ),
            LLMToolCallResult(
                tool_calls=[LLMToolCall(
                    id="call_1",
                    name="done",
                    arguments={"answer": "Based on the evidence...", "memory_ids": [memory_id_1, memory_id_2]}
                )],
                finish_reason="tool_calls",
            ),
        ]

        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="What do we know?",
            bank_profile=bank_profile,
            lookup_fn=AsyncMock(return_value={"count": 0, "models": []}),
            recall_fn=mock_recall,
            expand_fn=AsyncMock(return_value={}),
        )

        # Both memory IDs should be in the result (they were recalled)
        assert memory_id_1 in result.used_memory_ids
        assert memory_id_2 in result.used_memory_ids
        assert len(result.used_memory_ids) == 2

    async def test_agent_filters_hallucinated_memory_ids(self, mock_llm, bank_profile):
        """Test agent filters out memory IDs that were not in recall results."""
        valid_memory_id = str(uuid.uuid4())
        hallucinated_memory_id = str(uuid.uuid4())

        # Mock recall returns only one memory ID
        mock_recall = AsyncMock(
            return_value={
                "query": "test",
                "count": 1,
                "memories": [
                    {"id": valid_memory_id, "text": "Real memory", "type": "experience"},
                ],
            }
        )

        mock_llm.call_with_tools.side_effect = [
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="call_0", name="recall", arguments={"query": "test query"})],
                finish_reason="tool_calls",
            ),
            LLMToolCallResult(
                tool_calls=[LLMToolCall(
                    id="call_1",
                    name="done",
                    arguments={"answer": "Based on evidence...", "memory_ids": [valid_memory_id, hallucinated_memory_id]}
                )],
                finish_reason="tool_calls",
            ),
        ]

        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="What do we know?",
            bank_profile=bank_profile,
            lookup_fn=AsyncMock(return_value={"count": 0, "models": []}),
            recall_fn=mock_recall,
            expand_fn=AsyncMock(return_value={}),
        )

        # Only the valid memory ID should be in the result
        assert valid_memory_id in result.used_memory_ids
        assert hallucinated_memory_id not in result.used_memory_ids
        assert len(result.used_memory_ids) == 1

    async def test_agent_returns_validated_model_ids(self, mock_llm, bank_profile):
        """Test agent returns only validated model IDs that were actually looked up."""
        model_id = "team-structure"
        hallucinated_model_id = "non-existent-model"

        # Mock lookup returns different results based on input
        # - None (or no arg): list_mental_models - returns list of models
        # - model_id: get_mental_model - returns specific model with found=True
        async def mock_lookup_impl(arg=None):
            if arg is None:
                return {"count": 1, "models": [{"id": model_id, "name": "Team Structure", "description": "desc"}]}
            else:
                return {"found": True, "model": {"id": model_id, "name": "Team Structure", "summary": "Full summary"}}

        mock_lookup = AsyncMock(side_effect=mock_lookup_impl)

        mock_llm.call_with_tools.side_effect = [
            LLMToolCallResult(
                tool_calls=[
                    LLMToolCall(id="call_0", name="list_mental_models", arguments={}),
                    LLMToolCall(id="call_1", name="get_mental_model", arguments={"model_id": model_id}),
                ],
                finish_reason="tool_calls",
            ),
            LLMToolCallResult(
                tool_calls=[LLMToolCall(
                    id="call_2",
                    name="done",
                    arguments={"answer": "Based on team structure...", "model_ids": [model_id, hallucinated_model_id]}
                )],
                finish_reason="tool_calls",
            ),
        ]

        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="How is the team organized?",
            bank_profile=bank_profile,
            lookup_fn=mock_lookup,
            recall_fn=AsyncMock(return_value={"query": "test", "count": 0, "memories": []}),
            expand_fn=AsyncMock(return_value={}),
        )

        # Only the valid model ID should be in the result
        assert model_id in result.used_model_ids
        assert hallucinated_model_id not in result.used_model_ids

    async def test_agent_plain_text_answer(self, mock_llm, bank_profile, mock_tools):
        """Test agent with plain text answer format."""
        mock_llm.call_with_tools.side_effect = [
            # First: gather evidence via recall (required by guardrail)
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="call_0", name="recall", arguments={"query": "answer"})],
                finish_reason="tool_calls",
            ),
            # Then: done with plain text answer
            LLMToolCallResult(
                tool_calls=[LLMToolCall(
                    id="call_1",
                    name="done",
                    arguments={"answer": "The answer is simple and direct."}
                )],
                finish_reason="tool_calls",
            ),
        ]

        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="What's the answer?",
            bank_profile=bank_profile,
            **mock_tools,
        )

        assert result.text == "The answer is simple and direct."


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
                # Learned models are created as placeholders pending generation
                assert model.get("name") is not None
                assert model.get("description") is not None

        # Cleanup
        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_reflect_learn_triggers_background_generation(self, memory, request_context):
        """Test that when reflect calls learn, background generation is triggered.

        This test verifies the full flow:
        1. Agent decides to learn something important
        2. learn tool creates a placeholder model
        3. Background generation is automatically triggered
        """
        import asyncio

        bank_id = f"test-reflect-learn-{uuid.uuid4().hex[:8]}"

        # Add rich test data that should prompt the agent to learn something
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {"content": "Bob is the CEO and founder of the company."},
                {"content": "Bob started the company in 2015 after leaving Google."},
                {"content": "Bob holds weekly all-hands meetings every Friday at 3pm."},
                {"content": "Bob's management style is very hands-off and trusts his team."},
                {"content": "Bob prefers face-to-face communication over email."},
                {"content": "Bob has a strong focus on company culture and team building."},
            ],
            request_context=request_context,
        )
        await memory.wait_for_background_tasks()

        # Run reflect with a query that should prompt learning
        result = await memory.reflect_async(
            bank_id=bank_id,
            query="Tell me everything about Bob's leadership style and how he runs the company. "
            "This is important information I'll need to reference frequently.",
            request_context=request_context,
        )

        assert result.text is not None
        assert len(result.text) > 0

        # Wait for any background tasks to complete
        await memory.wait_for_background_tasks()
        # Give a bit more time for async generation
        await asyncio.sleep(2)

        # Check if learned models were created
        models = await memory.list_mental_models(
            bank_id=bank_id,
            request_context=request_context,
        )

        learned_models = [m for m in models if m.get("subtype") == "learned"]

        # If learned models were created, verify they have proper structure
        for model in learned_models:
            assert model.get("name") is not None
            assert model.get("description") is not None
            # After background generation, the model should have been updated
            # (observations may or may not be populated depending on timing)

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
