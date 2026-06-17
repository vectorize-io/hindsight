"""Unit tests for Hindsight LlamaIndex memory."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core.llms import ChatMessage, MessageRole
from hindsight_llamaindex import HindsightMemory


def _mock_client():
    """Create a mock Hindsight client."""
    client = MagicMock()
    client.retain = MagicMock()
    client.recall = MagicMock()
    client.create_bank = MagicMock()
    client.aretain = AsyncMock()
    client.arecall = AsyncMock()
    client.acreate_bank = AsyncMock()
    return client


def _mock_recall_response(texts: list[str]):
    response = MagicMock()
    results = []
    for t in texts:
        r = MagicMock()
        r.text = t
        results.append(r)
    response.results = results
    return response


class TestHindsightMemoryCreation:
    def test_from_client(self):
        client = _mock_client()
        memory = HindsightMemory.from_client(
            client=client,
            bank_id="test-bank",
        )
        assert memory.bank_id == "test-bank"
        assert memory.context == "llamaindex"
        assert memory.budget == "mid"

    def test_from_client_with_options(self):
        client = _mock_client()
        memory = HindsightMemory.from_client(
            client=client,
            bank_id="test-bank",
            mission="Track preferences",
            context="my-app",
            budget="high",
            tags=["source:chat"],
        )
        assert memory.bank_id == "test-bank"
        assert memory.context == "my-app"
        assert memory.budget == "high"
        assert memory.tags == ["source:chat"]

    def test_from_url(self):
        with patch("hindsight_llamaindex.memory.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            memory = HindsightMemory.from_url(
                hindsight_api_url="http://localhost:8888",
                bank_id="test-bank",
            )
            assert memory.bank_id == "test-bank"
            mock_cls.assert_called_once_with(base_url="http://localhost:8888", timeout=30.0)

    def test_from_defaults_uses_cloud_default_when_nothing_supplied(self):
        """Parity with create_hindsight_tools: no URL/key → DEFAULT_HINDSIGHT_API_URL.

        Pins the cloud-default constructor that the 2026-06-02 audit asked for
        — HindsightMemory used to require an explicit client or explicit URL,
        leaving callers to wire the cloud-default themselves while the tools
        factory did it for them.
        """
        from hindsight_llamaindex.config import DEFAULT_HINDSIGHT_API_URL

        with patch("hindsight_llamaindex._client.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            memory = HindsightMemory.from_defaults(bank_id="test-bank")
            assert memory.bank_id == "test-bank"
            kwargs = mock_cls.call_args.kwargs
            assert kwargs.get("base_url") == DEFAULT_HINDSIGHT_API_URL

    def test_from_defaults_threads_api_key_into_constructed_client(self):
        """When api_key is provided, resolve_client must include it."""
        with patch("hindsight_llamaindex._client.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            HindsightMemory.from_defaults(
                bank_id="test-bank",
                api_key="hsk_test_key_42",
            )
            kwargs = mock_cls.call_args.kwargs
            assert kwargs.get("api_key") == "hsk_test_key_42"

    def test_from_defaults_explicit_client_wins(self):
        """When client is given, no new Hindsight is constructed."""
        client = _mock_client()
        with patch("hindsight_llamaindex._client.Hindsight") as mock_cls:
            memory = HindsightMemory.from_defaults(bank_id="test-bank", client=client)
            assert memory.bank_id == "test-bank"
            mock_cls.assert_not_called()

    def test_class_name(self):
        assert HindsightMemory.class_name() == "HindsightMemory"


class TestPut:
    def test_put_user_message_retains(self):
        client = _mock_client()
        memory = HindsightMemory.from_client(client=client, bank_id="test")
        msg = ChatMessage(role=MessageRole.USER, content="I like Python")
        memory.put(msg)

        client.retain.assert_called_once()
        kwargs = client.retain.call_args[1]
        assert kwargs["bank_id"] == "test"
        assert kwargs["content"] == "I like Python"
        assert kwargs["context"] == "llamaindex"
        assert kwargs["metadata"]["role"] == "user"

    def test_put_assistant_message_retains(self):
        client = _mock_client()
        memory = HindsightMemory.from_client(client=client, bank_id="test")
        msg = ChatMessage(role=MessageRole.ASSISTANT, content="Noted!")
        memory.put(msg)

        client.retain.assert_called_once()
        kwargs = client.retain.call_args[1]
        assert kwargs["metadata"]["role"] == "assistant"

    def test_put_system_message_skipped(self):
        client = _mock_client()
        memory = HindsightMemory.from_client(client=client, bank_id="test")
        msg = ChatMessage(role=MessageRole.SYSTEM, content="You are helpful")
        memory.put(msg)

        client.retain.assert_not_called()

    def test_put_empty_content_skipped(self):
        client = _mock_client()
        memory = HindsightMemory.from_client(client=client, bank_id="test")
        msg = ChatMessage(role=MessageRole.USER, content="   ")
        memory.put(msg)

        client.retain.assert_not_called()

    def test_put_adds_to_local_history(self):
        client = _mock_client()
        memory = HindsightMemory.from_client(client=client, bank_id="test")
        msg = ChatMessage(role=MessageRole.USER, content="hello")
        memory.put(msg)

        assert len(memory.get_all()) == 1
        assert memory.get_all()[0].content == "hello"

    def test_put_trims_to_limit(self):
        client = _mock_client()
        memory = HindsightMemory.from_client(client=client, bank_id="test", chat_history_limit=3)
        for i in range(5):
            memory.put(ChatMessage(role=MessageRole.USER, content=f"msg-{i}"))

        history = memory.get_all()
        assert len(history) == 3
        assert history[0].content == "msg-2"
        assert history[2].content == "msg-4"

    def test_put_tags_passed(self):
        client = _mock_client()
        memory = HindsightMemory.from_client(client=client, bank_id="test", tags=["source:chat"])
        memory.put(ChatMessage(role=MessageRole.USER, content="hello"))

        kwargs = client.retain.call_args[1]
        assert kwargs["tags"] == ["source:chat"]

    def test_put_retain_failure_is_graceful(self):
        client = _mock_client()
        client.retain.side_effect = RuntimeError("connection refused")
        memory = HindsightMemory.from_client(client=client, bank_id="test")
        # Should not raise
        memory.put(ChatMessage(role=MessageRole.USER, content="hello"))
        # Message still in local history
        assert len(memory.get_all()) == 1

    def test_put_generates_document_id(self):
        client = _mock_client()
        memory = HindsightMemory.from_client(client=client, bank_id="test")
        memory.put(ChatMessage(role=MessageRole.USER, content="hello"))

        kwargs = client.retain.call_args[1]
        doc_id = kwargs["document_id"]
        # Auto-generated format: {session_id}-{uuid_hex_12}
        assert "-" in doc_id
        suffix = doc_id.rsplit("-", 1)[1]
        assert len(suffix) == 12


class TestGet:
    def test_get_without_input_falls_back_to_last_user_message(self):
        """Workflow-based agents (llama_index.core.agent.workflow.ReActAgent
        and friends) call ``memory.aget()`` without passing ``input=``. Without
        a fallback, automatic recall would never fire for those agents. We
        recall using the most recent user message in local history instead.
        """
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["User likes Python"])
        memory = HindsightMemory.from_client(client=client, bank_id="test")
        memory.put(ChatMessage(role=MessageRole.USER, content="what do I like?"))
        memory.put(ChatMessage(role=MessageRole.ASSISTANT, content="ack"))

        messages = memory.get()  # no input — fall back to last user msg
        client.recall.assert_called_once()
        assert client.recall.call_args[1]["query"] == "what do I like?"
        # System (with recalled memories) + 2 history messages
        assert len(messages) == 3
        assert messages[0].role == MessageRole.SYSTEM
        assert "User likes Python" in str(messages[0].content)

    def test_get_without_input_and_empty_history_skips_recall(self):
        client = _mock_client()
        memory = HindsightMemory.from_client(client=client, bank_id="test")
        messages = memory.get()
        assert len(messages) == 0
        client.recall.assert_not_called()

    def test_get_without_input_and_no_user_msg_skips_recall(self):
        """History with only system/assistant messages and no user message
        should not fire recall — there's no semantically meaningful query
        to look up."""
        client = _mock_client()
        memory = HindsightMemory.from_client(client=client, bank_id="test")
        memory.put(ChatMessage(role=MessageRole.ASSISTANT, content="standalone"))
        messages = memory.get()
        assert len(messages) == 1
        client.recall.assert_not_called()

    def test_get_with_input_recalls_memories(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["User likes Python", "User prefers dark mode"])
        memory = HindsightMemory.from_client(client=client, bank_id="test")
        memory.put(ChatMessage(role=MessageRole.USER, content="hello"))

        messages = memory.get(input="What are my preferences?")

        # Should have system message + chat history
        assert len(messages) == 2
        assert messages[0].role == MessageRole.SYSTEM
        assert "User likes Python" in str(messages[0].content)
        assert "User prefers dark mode" in str(messages[0].content)
        assert messages[1].content == "hello"

    def test_get_with_input_no_memories(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response([])
        memory = HindsightMemory.from_client(client=client, bank_id="test")
        memory.put(ChatMessage(role=MessageRole.USER, content="hello"))

        messages = memory.get(input="anything")
        # No system message when no memories found
        assert len(messages) == 1
        assert messages[0].content == "hello"

    def test_get_recall_passes_params(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])
        memory = HindsightMemory.from_client(
            client=client,
            bank_id="test",
            budget="high",
            max_tokens=2048,
            recall_tags=["scope:user"],
            recall_tags_match="all",
        )

        memory.get(input="query")
        kwargs = client.recall.call_args[1]
        assert kwargs["budget"] == "high"
        assert kwargs["max_tokens"] == 2048
        assert kwargs["tags"] == ["scope:user"]
        assert kwargs["tags_match"] == "all"

    def test_get_recall_failure_is_graceful(self):
        client = _mock_client()
        client.recall.side_effect = RuntimeError("timeout")
        memory = HindsightMemory.from_client(client=client, bank_id="test")
        memory.put(ChatMessage(role=MessageRole.USER, content="hello"))

        # Should not raise, returns history without memories
        messages = memory.get(input="query")
        assert len(messages) == 1

    def test_get_custom_system_prompt(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact1"])
        memory = HindsightMemory.from_client(
            client=client,
            bank_id="test",
            system_prompt="MEMORIES: {memories}",
        )

        messages = memory.get(input="query")
        assert str(messages[0].content) == "MEMORIES: - fact1"


class TestSet:
    def test_set_replaces_history(self):
        client = _mock_client()
        memory = HindsightMemory.from_client(client=client, bank_id="test")
        memory.put(ChatMessage(role=MessageRole.USER, content="old"))

        new_messages = [
            ChatMessage(role=MessageRole.USER, content="new1"),
            ChatMessage(role=MessageRole.ASSISTANT, content="new2"),
        ]
        memory.set(new_messages)

        history = memory.get_all()
        assert len(history) == 2
        assert history[0].content == "new1"

    def test_set_retains_only_new_messages(self):
        client = _mock_client()
        memory = HindsightMemory.from_client(client=client, bank_id="test")
        memory.put(ChatMessage(role=MessageRole.USER, content="existing"))
        client.retain.reset_mock()

        # Set with 3 messages (1 existing + 2 new)
        messages = [
            ChatMessage(role=MessageRole.USER, content="existing"),
            ChatMessage(role=MessageRole.USER, content="new1"),
            ChatMessage(role=MessageRole.ASSISTANT, content="new2"),
        ]
        memory.set(messages)

        # Should retain only the 2 new messages
        assert client.retain.call_count == 2


class TestReset:
    def test_reset_clears_local_history(self):
        client = _mock_client()
        memory = HindsightMemory.from_client(client=client, bank_id="test")
        memory.put(ChatMessage(role=MessageRole.USER, content="hello"))
        assert len(memory.get_all()) == 1

        memory.reset()
        assert len(memory.get_all()) == 0


class TestReActStripping:
    """Tests for stripping ReAct reasoning traces from assistant messages."""

    def test_plain_assistant_message_retained(self):
        client = _mock_client()
        memory = HindsightMemory.from_client(client=client, bank_id="test")
        msg = ChatMessage(role=MessageRole.ASSISTANT, content="You use VS Code.")
        memory.put(msg)

        client.retain.assert_called_once()
        assert client.retain.call_args[1]["content"] == "You use VS Code."

    def test_react_with_answer_retains_answer_only(self):
        client = _mock_client()
        memory = HindsightMemory.from_client(client=client, bank_id="test")
        react_content = (
            "Thought: I need to recall the user's IDE preference.\n"
            "Action: recall_memory\n"
            'Action Input: {"query": "IDE preference"}\n'
            "Observation: User prefers VS Code with dark mode.\n"
            "Thought: I now have the answer.\n"
            "Answer: You use VS Code with dark mode."
        )
        msg = ChatMessage(role=MessageRole.ASSISTANT, content=react_content)
        memory.put(msg)

        client.retain.assert_called_once()
        assert client.retain.call_args[1]["content"] == "You use VS Code with dark mode."

    def test_react_without_answer_skips_retain(self):
        client = _mock_client()
        memory = HindsightMemory.from_client(client=client, bank_id="test")
        react_content = (
            "Thought: I need to use a tool to help me answer.\n"
            "Action: retain_memory\n"
            'Action Input: {"content": "User likes Python"}'
        )
        msg = ChatMessage(role=MessageRole.ASSISTANT, content=react_content)
        memory.put(msg)

        client.retain.assert_not_called()

    def test_react_stripping_still_adds_to_local_history(self):
        """Even when retain is skipped, the message is in local history."""
        client = _mock_client()
        memory = HindsightMemory.from_client(client=client, bank_id="test")
        react_content = "Thought: I need a tool.\nAction: some_tool\nAction Input: {}"
        msg = ChatMessage(role=MessageRole.ASSISTANT, content=react_content)
        memory.put(msg)

        # Not retained to Hindsight
        client.retain.assert_not_called()
        # But still in local history
        assert len(memory.get_all()) == 1

    def test_user_message_not_stripped(self):
        """User messages with Thought:/Action: are retained as-is."""
        client = _mock_client()
        memory = HindsightMemory.from_client(client=client, bank_id="test")
        content = "Thought: can you explain this Action: pattern?"
        msg = ChatMessage(role=MessageRole.USER, content=content)
        memory.put(msg)

        client.retain.assert_called_once()
        assert client.retain.call_args[1]["content"] == content

    def test_react_multiple_answers_uses_last(self):
        client = _mock_client()
        memory = HindsightMemory.from_client(client=client, bank_id="test")
        react_content = (
            "Thought: Let me check.\n"
            "Answer: Actually, let me reconsider.\n"
            "Thought: After further review.\n"
            "Answer: The final answer is 42."
        )
        msg = ChatMessage(role=MessageRole.ASSISTANT, content=react_content)
        memory.put(msg)

        client.retain.assert_called_once()
        assert client.retain.call_args[1]["content"] == "The final answer is 42."


class TestBankMission:
    def test_creates_bank_with_mission_on_put(self):
        client = _mock_client()
        memory = HindsightMemory.from_client(client=client, bank_id="test", mission="Track preferences")
        memory.put(ChatMessage(role=MessageRole.USER, content="hello"))

        client.create_bank.assert_called_once_with(
            bank_id="test",
            name="test",
            mission="Track preferences",
        )

    def test_creates_bank_with_mission_on_get(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response([])
        memory = HindsightMemory.from_client(client=client, bank_id="test", mission="Track preferences")
        memory.get(input="query")

        client.create_bank.assert_called_once()

    def test_bank_creation_idempotent(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response([])
        memory = HindsightMemory.from_client(client=client, bank_id="test", mission="mission")
        memory.put(ChatMessage(role=MessageRole.USER, content="hello"))
        memory.get(input="query")

        assert client.create_bank.call_count == 1

    def test_no_bank_creation_without_mission(self):
        client = _mock_client()
        memory = HindsightMemory.from_client(client=client, bank_id="test")
        memory.put(ChatMessage(role=MessageRole.USER, content="hello"))

        client.create_bank.assert_not_called()

    def test_bank_creation_failure_is_graceful(self):
        client = _mock_client()
        client.create_bank.side_effect = RuntimeError("already exists")
        memory = HindsightMemory.from_client(client=client, bank_id="test", mission="mission")
        # Should not raise
        memory.put(ChatMessage(role=MessageRole.USER, content="hello"))
        client.retain.assert_called_once()
