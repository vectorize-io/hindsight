"""Hindsight BaseMemory implementation for LlamaIndex.

Provides automatic memory for LlamaIndex agents:
- ``put()`` retains messages to Hindsight for long-term storage
- ``get()`` recalls relevant memories and prepends them as context
- Chat history is kept in-memory for the current session
"""

import logging
import re
import time
import uuid
from typing import Any, Optional

from hindsight_client import Hindsight
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory.types import BaseMemory

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "Below are relevant memories from previous conversations:\n{memories}\n"
    "Use these memories to provide more personalized and contextual responses."
)

# Patterns for detecting ReAct-style reasoning traces in assistant messages
_REACT_PATTERN = re.compile(r"^(Thought|Action|Action Input|Observation)\s*:", re.MULTILINE)
_ANSWER_PATTERN = re.compile(r"^Answer\s*:\s*", re.MULTILINE)


class HindsightMemory(BaseMemory):
    """Automatic long-term memory for LlamaIndex agents via Hindsight.

    On ``put()``, user and assistant messages are automatically retained
    to Hindsight. On ``get()``, relevant memories are recalled and
    prepended as a system message to enrich the agent's context.

    This follows the same pattern as Mem0's LlamaIndex integration:
    a local chat buffer for the current session, with Hindsight
    providing cross-session long-term memory.

    Args:
        bank_id: Hindsight memory bank to operate on.
        context: Source label for retain operations.
        budget: Recall budget level (low/mid/high).
        max_tokens: Maximum tokens for recall results.
        tags: Tags applied when storing memories.
        recall_tags: Tags to filter when recalling.
        recall_tags_match: Tag matching mode.
        system_prompt: Template for the memory system message.
            Must contain ``{memories}`` placeholder.
        chat_history_limit: Max messages to keep in local buffer.
            Oldest messages are dropped when exceeded.

    Example::

        from hindsight_client import Hindsight
        from hindsight_llamaindex import HindsightMemory

        client = Hindsight(base_url="http://localhost:8888")
        memory = HindsightMemory.from_client(
            client=client,
            bank_id="user-123",
            mission="Track user preferences",
        )

        # Use with any LlamaIndex agent — pass memory to run(), not the constructor
        agent = ReActAgent(tools=[], llm=llm)
        response = await agent.run("Hello!", memory=memory)
    """

    bank_id: str = Field(description="Hindsight memory bank ID")
    context: str = Field(default="llamaindex", description="Source label for retain")
    budget: str = Field(default="mid", description="Recall budget level")
    max_tokens: int = Field(default=4096, description="Max tokens for recall")
    tags: Optional[list[str]] = Field(default=None, description="Tags for retain")
    recall_tags: Optional[list[str]] = Field(default=None, description="Tags to filter recall")
    recall_tags_match: str = Field(default="any", description="Tag matching mode")
    system_prompt: str = Field(default=DEFAULT_SYSTEM_PROMPT, description="Memory system message template")
    chat_history_limit: int = Field(default=100, description="Max messages in local buffer")

    _client: Hindsight = PrivateAttr()
    _chat_history: list[ChatMessage] = PrivateAttr(default_factory=list)
    _session_id: str = PrivateAttr()
    _bank_initialized: bool = PrivateAttr(default=False)
    _mission: Optional[str] = PrivateAttr(default=None)

    def __init__(self, client: Hindsight, mission: Optional[str] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self._client = client
        self._session_id = str(uuid.uuid4())[:8]
        self._mission = mission
        self._chat_history = []
        self._bank_initialized = False

    @classmethod
    def class_name(cls) -> str:
        return "HindsightMemory"

    @classmethod
    def from_defaults(cls, **kwargs: Any) -> "HindsightMemory":
        """Create from defaults. Prefer ``from_client()`` instead."""
        raise NotImplementedError("Use HindsightMemory.from_client() or HindsightMemory.from_url() instead.")

    @classmethod
    def from_client(
        cls,
        client: Hindsight,
        bank_id: str,
        *,
        mission: Optional[str] = None,
        context: str = "llamaindex",
        budget: str = "mid",
        max_tokens: int = 4096,
        tags: Optional[list[str]] = None,
        recall_tags: Optional[list[str]] = None,
        recall_tags_match: str = "any",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        chat_history_limit: int = 100,
    ) -> "HindsightMemory":
        """Create a HindsightMemory with a pre-configured client.

        Args:
            client: Hindsight client instance.
            bank_id: Memory bank ID.
            mission: Bank mission (creates bank on first use if set).
            context: Source label for retain operations.
            budget: Recall budget level.
            max_tokens: Max recall tokens.
            tags: Tags for retain operations.
            recall_tags: Tags to filter recall.
            recall_tags_match: Tag matching mode.
            system_prompt: Memory system message template.
            chat_history_limit: Max local buffer size.
        """
        return cls(
            client=client,
            bank_id=bank_id,
            mission=mission,
            context=context,
            budget=budget,
            max_tokens=max_tokens,
            tags=tags,
            recall_tags=recall_tags,
            recall_tags_match=recall_tags_match,
            system_prompt=system_prompt,
            chat_history_limit=chat_history_limit,
        )

    @classmethod
    def from_url(
        cls,
        hindsight_api_url: str,
        bank_id: str,
        *,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> "HindsightMemory":
        """Create a HindsightMemory from an API URL.

        Args:
            hindsight_api_url: Hindsight API URL.
            bank_id: Memory bank ID.
            api_key: Optional API key.
            **kwargs: Additional arguments passed to ``from_client()``.
        """
        client_kwargs: dict[str, Any] = {"base_url": hindsight_api_url, "timeout": 30.0}
        if api_key:
            client_kwargs["api_key"] = api_key
        client = Hindsight(**client_kwargs)
        return cls.from_client(client=client, bank_id=bank_id, **kwargs)

    def _ensure_bank(self) -> None:
        if self._bank_initialized or not self._mission:
            return
        try:
            self._client.create_bank(
                bank_id=self.bank_id,
                name=self.bank_id,
                mission=self._mission,
            )
        except Exception as e:
            logger.debug(f"Bank creation for {self.bank_id}: {e}")
        self._bank_initialized = True

    async def _aensure_bank(self) -> None:
        if self._bank_initialized or not self._mission:
            return
        try:
            await self._client.acreate_bank(
                bank_id=self.bank_id,
                name=self.bank_id,
                mission=self._mission,
            )
        except Exception as e:
            logger.debug(f"Bank creation for {self.bank_id}: {e}")
        self._bank_initialized = True

    def _generate_document_id(self) -> str:
        return f"{self._session_id}-{uuid.uuid4().hex[:12]}"

    @staticmethod
    def _extract_clean_content(content: str, role: MessageRole) -> str:
        """Extract clean content from a message, stripping ReAct traces.

        For assistant messages containing ReAct reasoning (Thought:/Action:/
        Observation: prefixes), extracts only the final Answer: text.
        Returns empty string if the message is purely reasoning with no answer.

        User messages are returned as-is.
        """
        if role != MessageRole.ASSISTANT:
            return content

        # Check if this looks like ReAct reasoning
        if not _REACT_PATTERN.search(content):
            return content

        # Extract the final Answer: block
        answer_match = list(_ANSWER_PATTERN.finditer(content))
        if answer_match:
            # Use the last Answer: block (final answer after reasoning)
            last_answer = answer_match[-1]
            return content[last_answer.end() :].strip()

        # ReAct traces with no Answer: — skip retention
        return ""

    def _retain_message(self, message: ChatMessage) -> None:
        """Retain a message to Hindsight (sync)."""
        if message.role not in (MessageRole.USER, MessageRole.ASSISTANT):
            return
        content = str(message.content) if message.content else ""
        content = self._extract_clean_content(content, message.role)
        if not content.strip():
            return
        try:
            self._ensure_bank()
            kwargs: dict[str, Any] = {
                "bank_id": self.bank_id,
                "content": content,
                "context": self.context,
                "document_id": self._generate_document_id(),
                "metadata": {"role": message.role.value, "source": "llamaindex"},
            }
            if self.tags:
                kwargs["tags"] = self.tags
            self._client.retain(**kwargs)
        except Exception as e:
            logger.error(f"Failed to retain message: {e}")

    async def _aretain_message(self, message: ChatMessage) -> None:
        """Retain a message to Hindsight (async)."""
        if message.role not in (MessageRole.USER, MessageRole.ASSISTANT):
            return
        content = str(message.content) if message.content else ""
        content = self._extract_clean_content(content, message.role)
        if not content.strip():
            return
        try:
            await self._aensure_bank()
            kwargs: dict[str, Any] = {
                "bank_id": self.bank_id,
                "content": content,
                "context": self.context,
                "document_id": self._generate_document_id(),
                "metadata": {"role": message.role.value, "source": "llamaindex"},
            }
            if self.tags:
                kwargs["tags"] = self.tags
            await self._client.aretain(**kwargs)
        except Exception as e:
            logger.error(f"Failed to retain message: {e}")

    def _recall_memories(self, query: str) -> str:
        """Recall relevant memories (sync)."""
        try:
            self._ensure_bank()
            kwargs: dict[str, Any] = {
                "bank_id": self.bank_id,
                "query": query,
                "budget": self.budget,
                "max_tokens": self.max_tokens,
            }
            if self.recall_tags:
                kwargs["tags"] = self.recall_tags
                kwargs["tags_match"] = self.recall_tags_match
            response = self._client.recall(**kwargs)
            if not response.results:
                return ""
            lines = [r.text for r in response.results]
            return "\n".join(f"- {line}" for line in lines)
        except Exception as e:
            logger.error(f"Failed to recall memories: {e}")
            return ""

    async def _arecall_memories(self, query: str) -> str:
        """Recall relevant memories (async)."""
        try:
            await self._aensure_bank()
            kwargs: dict[str, Any] = {
                "bank_id": self.bank_id,
                "query": query,
                "budget": self.budget,
                "max_tokens": self.max_tokens,
            }
            if self.recall_tags:
                kwargs["tags"] = self.recall_tags
                kwargs["tags_match"] = self.recall_tags_match
            response = await self._client.arecall(**kwargs)
            if not response.results:
                return ""
            lines = [r.text for r in response.results]
            return "\n".join(f"- {line}" for line in lines)
        except Exception as e:
            logger.error(f"Failed to recall memories: {e}")
            return ""

    # -- BaseMemory interface --

    def get(self, input: Optional[str] = None, **kwargs: Any) -> list[ChatMessage]:
        """Get chat history, enriched with recalled Hindsight memories.

        If ``input`` is provided, relevant memories are recalled and
        prepended as a system message.
        """
        messages: list[ChatMessage] = []

        if input:
            memories_text = self._recall_memories(input)
            if memories_text:
                system_content = self.system_prompt.format(memories=memories_text)
                messages.append(ChatMessage(role=MessageRole.SYSTEM, content=system_content))

        messages.extend(self._chat_history)
        return messages

    async def aget(self, input: Optional[str] = None, **kwargs: Any) -> list[ChatMessage]:
        """Async version of get()."""
        messages: list[ChatMessage] = []

        if input:
            memories_text = await self._arecall_memories(input)
            if memories_text:
                system_content = self.system_prompt.format(memories=memories_text)
                messages.append(ChatMessage(role=MessageRole.SYSTEM, content=system_content))

        messages.extend(self._chat_history)
        return messages

    def get_all(self) -> list[ChatMessage]:
        """Get all messages in the local chat buffer."""
        return list(self._chat_history)

    def put(self, message: ChatMessage) -> None:
        """Store a message in local buffer and retain to Hindsight."""
        self._chat_history.append(message)
        # Trim to limit
        if len(self._chat_history) > self.chat_history_limit:
            self._chat_history = self._chat_history[-self.chat_history_limit :]
        self._retain_message(message)

    async def aput(self, message: ChatMessage) -> None:
        """Async version of put()."""
        self._chat_history.append(message)
        if len(self._chat_history) > self.chat_history_limit:
            self._chat_history = self._chat_history[-self.chat_history_limit :]
        await self._aretain_message(message)

    def set(self, messages: list[ChatMessage]) -> None:
        """Set the chat history, retaining new messages to Hindsight."""
        existing_len = len(self._chat_history)
        self._chat_history = list(messages)

        # Retain only new messages (beyond previous length)
        for msg in messages[existing_len:]:
            self._retain_message(msg)

    async def aset(self, messages: list[ChatMessage]) -> None:
        """Async version of set()."""
        existing_len = len(self._chat_history)
        self._chat_history = list(messages)

        for msg in messages[existing_len:]:
            await self._aretain_message(msg)

    def reset(self) -> None:
        """Reset the local chat buffer. Does not clear Hindsight memories."""
        self._chat_history = []
