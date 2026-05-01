"""Haystack tool factory for Hindsight memory operations.

Provides a convenience factory that creates Haystack-compatible ``Tool``
instances backed by Hindsight's retain/recall/reflect APIs, and a
``HindsightToolset`` with optional auto-recall and auto-retain.
"""

import asyncio
import json
import logging
import threading
import uuid
from typing import Any, Optional

from haystack.dataclasses import ChatMessage
from haystack.tools import Tool, Toolset
from hindsight_client import Hindsight

from ._client import resolve_client
from .config import get_config

logger = logging.getLogger(__name__)


# Persistent event loop in a daemon thread for async Hindsight client calls.
# aiohttp binds its session to the event loop that created it, so every call
# must use the *same* loop.  ``asyncio.run()`` creates and closes a fresh loop
# each time, which breaks aiohttp on subsequent calls.  We keep one loop alive
# in a background thread and submit coroutines to it via run_coroutine_threadsafe.
_loop = asyncio.new_event_loop()


def _start_loop() -> None:
    asyncio.set_event_loop(_loop)
    _loop.run_forever()


threading.Thread(target=_start_loop, daemon=True).start()


def _run_sync(coro):  # type: ignore[no-untyped-def]
    """Run an async coroutine synchronously from any context.

    Haystack's ``Tool`` requires sync callables, but the hindsight_client's
    async methods (aretain, arecall, areflect) must be awaited.  This helper
    submits the coroutine to a persistent background event loop and blocks
    until the result is ready.  Works regardless of whether the caller is
    inside a running event loop (e.g., Haystack's agent runtime) or not.
    """
    future = asyncio.run_coroutine_threadsafe(coro, _loop)
    return future.result()


DEFAULT_MEMORY_PROMPT = (
    "Below are relevant memories from previous conversations:\n{memories}\n"
    "Use these memories to provide more personalized and contextual responses."
)


class _HindsightToolBackend:
    """Internal backend that implements Hindsight memory operations.

    Encapsulates client resolution, config fallback, bank management,
    and the retain/recall/reflect logic. Methods are wrapped as Haystack
    ``Tool`` objects by ``create_hindsight_tools()``.
    """

    def __init__(
        self,
        *,
        bank_id: str,
        client: Optional[Hindsight] = None,
        hindsight_api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        budget: Optional[str] = None,
        max_tokens: Optional[int] = None,
        tags: Optional[list[str]] = None,
        recall_tags: Optional[list[str]] = None,
        recall_tags_match: Optional[str] = None,
        # Retain options
        retain_metadata: Optional[dict[str, str]] = None,
        retain_document_id: Optional[str] = None,
        retain_context: Optional[str] = None,
        # Recall options
        recall_types: Optional[list[str]] = None,
        recall_include_entities: bool = False,
        # Reflect options
        reflect_context: Optional[str] = None,
        reflect_max_tokens: Optional[int] = None,
        reflect_response_schema: Optional[dict[str, Any]] = None,
        reflect_tags: Optional[list[str]] = None,
        reflect_tags_match: Optional[str] = None,
        # Bank management
        mission: Optional[str] = None,
    ):
        self._client = resolve_client(client, hindsight_api_url, api_key)
        self._bank_id = bank_id
        self._session_id = str(uuid.uuid4())[:8]
        self._bank_initialized = False

        # Resolve effective values using None-sentinel config fallback
        config = get_config()
        self._tags = tags if tags is not None else (config.tags if config else None)
        self._recall_tags = recall_tags if recall_tags is not None else (config.recall_tags if config else None)
        self._recall_tags_match = (
            recall_tags_match if recall_tags_match is not None else (config.recall_tags_match if config else "any")
        )
        self._budget = budget if budget is not None else (config.budget if config else "mid")
        self._max_tokens = max_tokens if max_tokens is not None else (config.max_tokens if config else 4096)

        # Retain-specific
        self._retain_metadata = retain_metadata
        self._retain_document_id = retain_document_id
        self._retain_context = (
            retain_context if retain_context is not None else (config.context if config else "haystack")
        )

        # Recall-specific
        self._recall_types = recall_types
        self._recall_include_entities = recall_include_entities

        # Reflect-specific
        self._reflect_context = reflect_context
        self._reflect_max_tokens = reflect_max_tokens
        self._reflect_response_schema = reflect_response_schema
        self._reflect_tags = reflect_tags
        self._reflect_tags_match = reflect_tags_match

        # Bank management
        self._mission = mission if mission is not None else (config.mission if config else None)

    def _ensure_bank(self) -> None:
        """Create/update the bank with mission if not already done."""
        if self._bank_initialized or not self._mission:
            return
        try:
            _run_sync(
                self._client.acreate_bank(
                    bank_id=self._bank_id,
                    name=self._bank_id,
                    mission=self._mission,
                )
            )
            self._bank_initialized = True
            logger.debug(f"Created/updated bank: {self._bank_id}")
        except Exception as e:
            err_str = str(e).lower()
            if "already exists" in err_str or "409" in err_str or "conflict" in err_str:
                self._bank_initialized = True
                logger.debug(f"Bank already exists: {self._bank_id}")
            else:
                # Transient error — don't mark as initialized so we retry next time
                logger.warning(f"Bank creation failed for {self._bank_id}: {e}")

    def _generate_document_id(self) -> str:
        """Generate a unique document_id for retain operations."""
        return f"{self._session_id}-{uuid.uuid4().hex[:12]}"

    def _retain_kwargs(self, content: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "bank_id": self._bank_id,
            "content": content,
            "context": self._retain_context,
        }
        if self._tags:
            kwargs["tags"] = self._tags
        if self._retain_metadata:
            kwargs["metadata"] = self._retain_metadata
        # Use explicit document_id if set, otherwise auto-generate
        kwargs["document_id"] = self._retain_document_id or self._generate_document_id()
        return kwargs

    def _recall_kwargs(self, query: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "bank_id": self._bank_id,
            "query": query,
            "budget": self._budget,
            "max_tokens": self._max_tokens,
        }
        if self._recall_tags:
            kwargs["tags"] = self._recall_tags
            kwargs["tags_match"] = self._recall_tags_match
        if self._recall_types:
            kwargs["types"] = self._recall_types
        if self._recall_include_entities:
            kwargs["include_entities"] = True
        return kwargs

    def _reflect_kwargs(self, query: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "bank_id": self._bank_id,
            "query": query,
            "budget": self._budget,
        }
        if self._reflect_context:
            kwargs["context"] = self._reflect_context
        effective_reflect_max = self._reflect_max_tokens or self._max_tokens
        if effective_reflect_max:
            kwargs["max_tokens"] = effective_reflect_max
        if self._reflect_response_schema:
            kwargs["response_schema"] = self._reflect_response_schema
        effective_reflect_tags = self._reflect_tags if self._reflect_tags is not None else self._recall_tags
        effective_reflect_tags_match = self._reflect_tags_match or self._recall_tags_match
        if effective_reflect_tags:
            kwargs["tags"] = effective_reflect_tags
            kwargs["tags_match"] = effective_reflect_tags_match
        return kwargs

    @staticmethod
    def _format_recall(response: Any) -> str:
        if not response.results:
            return "No relevant memories found."
        lines = []
        for i, result in enumerate(response.results, 1):
            lines.append(f"{i}. {result.text}")
        return "\n".join(lines)

    def retain_memory(self, content: str) -> str:
        """Store information to long-term memory for later retrieval.

        Use this to save important facts, user preferences, decisions,
        or any information that should be remembered across conversations.

        Args:
            content: The information to store in memory.
        """
        try:
            self._ensure_bank()
            _run_sync(self._client.aretain(**self._retain_kwargs(content)))
            return "Memory stored successfully."
        except Exception as e:
            logger.error(f"Retain failed: {e}")
            return f"Failed to store memory: {e!s:.200}"

    def recall_memory(self, query: str) -> str:
        """Search long-term memory for relevant information.

        Use this to find previously stored facts, preferences, or context.
        Returns a numbered list of matching memories.

        Args:
            query: What to search for in memory.
        """
        try:
            self._ensure_bank()
            response = _run_sync(self._client.arecall(**self._recall_kwargs(query)))
            return self._format_recall(response)
        except Exception as e:
            logger.error(f"Recall failed: {e}")
            return f"Failed to search memory: {e!s:.200}"

    def reflect_on_memory(self, query: str) -> str:
        """Synthesize a thoughtful answer from long-term memories.

        Use this when you need a coherent summary or reasoned response
        about what you know, rather than raw memory facts.

        Args:
            query: The question to reflect on using stored memories.
        """
        try:
            self._ensure_bank()
            response = _run_sync(self._client.areflect(**self._reflect_kwargs(query)))
            # When response_schema is set, return the structured JSON output
            if self._reflect_response_schema and response.structured_output is not None:
                return json.dumps(response.structured_output)
            return response.text or "No relevant memories found."
        except Exception as e:
            logger.error(f"Reflect failed: {e}")
            return f"Failed to reflect on memory: {e!s:.200}"


# -- Tool definitions mapping operation name -> (method name, description, parameter schema) --
_TOOL_DEFS: dict[str, tuple[str, str, dict[str, Any]]] = {
    "retain_memory": (
        "retain_memory",
        (
            "Store information to long-term memory for later retrieval. "
            "Use this to save important facts, user preferences, decisions, "
            "or any information that should be remembered across conversations."
        ),
        {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The information to store in memory.",
                },
            },
            "required": ["content"],
        },
    ),
    "recall_memory": (
        "recall_memory",
        (
            "Search long-term memory for relevant information. "
            "Use this to find previously stored facts, preferences, or context. "
            "Returns a numbered list of matching memories."
        ),
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for in memory.",
                },
            },
            "required": ["query"],
        },
    ),
    "reflect_on_memory": (
        "reflect_on_memory",
        (
            "Synthesize a thoughtful answer from long-term memories. "
            "Use this when you need a coherent summary or reasoned response "
            "about what you know, rather than raw memory facts."
        ),
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question to reflect on using stored memories.",
                },
            },
            "required": ["query"],
        },
    ),
}


class _HindsightTool(Tool):
    """A Haystack Tool backed by a Hindsight memory operation.

    Overrides ``to_dict()``/``from_dict()`` so that the tool's configuration
    (bank_id, API URL, etc.) is serialized instead of the bound method,
    which Haystack cannot serialize.
    """

    def __init__(
        self,
        *,
        backend: "_HindsightToolBackend",
        tool_name: str,
        backend_kwargs: dict[str, Any],
    ):
        method_name, description, parameters = _TOOL_DEFS[tool_name]
        super().__init__(
            name=tool_name,
            description=description,
            function=getattr(backend, method_name),
            parameters=parameters,
        )
        self._backend = backend
        self._backend_kwargs = backend_kwargs

    def to_dict(self) -> dict[str, Any]:
        """Serialize the tool to a dictionary.

        Stores the backend configuration so the tool can be reconstructed
        via ``from_dict()`` without needing to serialize the bound method.
        """
        data = {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "backend_kwargs": self._backend_kwargs,
        }
        cls = type(self)
        qualified_name = f"{cls.__module__}.{cls.__qualname__}"
        return {"type": qualified_name, "data": data}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "_HindsightTool":
        """Deserialize the tool from a dictionary."""
        inner = data["data"]
        backend_kwargs = inner["backend_kwargs"]
        backend = _HindsightToolBackend(**backend_kwargs)
        return cls(
            backend=backend,
            tool_name=inner["name"],
            backend_kwargs=backend_kwargs,
        )


def _build_backend_kwargs(
    *,
    bank_id: str,
    client: Optional[Hindsight],
    hindsight_api_url: Optional[str],
    api_key: Optional[str],
    budget: Optional[str],
    max_tokens: Optional[int],
    tags: Optional[list[str]],
    recall_tags: Optional[list[str]],
    recall_tags_match: Optional[str],
    retain_metadata: Optional[dict[str, str]],
    retain_document_id: Optional[str],
    retain_context: Optional[str],
    recall_types: Optional[list[str]],
    recall_include_entities: bool,
    reflect_context: Optional[str],
    reflect_max_tokens: Optional[int],
    reflect_response_schema: Optional[dict[str, Any]],
    reflect_tags: Optional[list[str]],
    reflect_tags_match: Optional[str],
    mission: Optional[str],
) -> dict[str, Any]:
    """Build serializable backend kwargs, extracting client connection info."""
    serializable_url = hindsight_api_url
    serializable_key = api_key
    if client is not None and serializable_url is None:
        serializable_url = getattr(client, "_base_url", None) or getattr(client, "base_url", None)
        if serializable_url is not None:
            serializable_url = str(serializable_url)
    if client is not None and serializable_key is None:
        serializable_key = getattr(client, "_api_key", None) or getattr(client, "api_key", None)

    return {
        "bank_id": bank_id,
        "hindsight_api_url": serializable_url,
        "api_key": serializable_key,
        "budget": budget,
        "max_tokens": max_tokens,
        "tags": tags,
        "recall_tags": recall_tags,
        "recall_tags_match": recall_tags_match,
        "retain_metadata": retain_metadata,
        "retain_document_id": retain_document_id,
        "retain_context": retain_context,
        "recall_types": recall_types,
        "recall_include_entities": recall_include_entities,
        "reflect_context": reflect_context,
        "reflect_max_tokens": reflect_max_tokens,
        "reflect_response_schema": reflect_response_schema,
        "reflect_tags": reflect_tags,
        "reflect_tags_match": reflect_tags_match,
        "mission": mission,
    }


def _build_tools(
    backend: _HindsightToolBackend,
    backend_kwargs: dict[str, Any],
    *,
    include_retain: bool = True,
    include_recall: bool = True,
    include_reflect: bool = True,
) -> list[Tool]:
    """Create _HindsightTool instances from a backend."""
    tools: list[Tool] = []
    if include_retain:
        tools.append(_HindsightTool(backend=backend, tool_name="retain_memory", backend_kwargs=backend_kwargs))
    if include_recall:
        tools.append(_HindsightTool(backend=backend, tool_name="recall_memory", backend_kwargs=backend_kwargs))
    if include_reflect:
        tools.append(_HindsightTool(backend=backend, tool_name="reflect_on_memory", backend_kwargs=backend_kwargs))
    return tools


def create_hindsight_tools(
    *,
    bank_id: str,
    client: Optional[Hindsight] = None,
    hindsight_api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    budget: Optional[str] = None,
    max_tokens: Optional[int] = None,
    tags: Optional[list[str]] = None,
    recall_tags: Optional[list[str]] = None,
    recall_tags_match: Optional[str] = None,
    # Retain options
    retain_metadata: Optional[dict[str, str]] = None,
    retain_document_id: Optional[str] = None,
    retain_context: Optional[str] = None,
    # Recall options
    recall_types: Optional[list[str]] = None,
    recall_include_entities: bool = False,
    # Reflect options
    reflect_context: Optional[str] = None,
    reflect_max_tokens: Optional[int] = None,
    reflect_response_schema: Optional[dict[str, Any]] = None,
    reflect_tags: Optional[list[str]] = None,
    reflect_tags_match: Optional[str] = None,
    # Bank management
    mission: Optional[str] = None,
    include_retain: bool = True,
    include_recall: bool = True,
    include_reflect: bool = True,
) -> list[Tool]:
    """Create Hindsight memory tools for a Haystack agent.

    Convenience factory that creates a backend and returns Haystack ``Tool``
    instances ready for use with any Haystack agent. For automatic recall
    and retain behavior, use :class:`HindsightToolset` instead.

    Args:
        bank_id: The Hindsight memory bank to operate on.
        client: Pre-configured Hindsight client (preferred).
        hindsight_api_url: API URL (used if no client provided).
        api_key: API key (used if no client provided).
        budget: Recall/reflect budget level (low/mid/high).
        max_tokens: Maximum tokens for recall results.
        tags: Tags applied when storing memories via retain.
        recall_tags: Tags to filter when searching memories.
        recall_tags_match: Tag matching mode (any/all/any_strict/all_strict).
        retain_metadata: Default metadata dict for retain operations.
        retain_document_id: Default document_id for retain. If None,
            auto-generates per call.
        retain_context: Source label for retain operations.
        recall_types: Fact types to filter (world, experience, opinion, observation).
        recall_include_entities: Include entity information in recall results.
        reflect_context: Additional context for reflect operations.
        reflect_max_tokens: Max tokens for reflect results (defaults to max_tokens).
        reflect_response_schema: JSON schema to constrain reflect output format.
        reflect_tags: Tags to filter memories used in reflect (defaults to recall_tags).
        reflect_tags_match: Tag matching for reflect (defaults to recall_tags_match).
        mission: Bank mission for fact extraction context.
        include_retain: Include the retain (store) tool.
        include_recall: Include the recall (search) tool.
        include_reflect: Include the reflect (synthesize) tool.

    Returns:
        List of Haystack Tool instances.

    Raises:
        HindsightError: If no client or API URL can be resolved during
            tool creation. Tool invocations return error strings instead
            of raising.
    """
    backend_kwargs = _build_backend_kwargs(
        bank_id=bank_id,
        client=client,
        hindsight_api_url=hindsight_api_url,
        api_key=api_key,
        budget=budget,
        max_tokens=max_tokens,
        tags=tags,
        recall_tags=recall_tags,
        recall_tags_match=recall_tags_match,
        retain_metadata=retain_metadata,
        retain_document_id=retain_document_id,
        retain_context=retain_context,
        recall_types=recall_types,
        recall_include_entities=recall_include_entities,
        reflect_context=reflect_context,
        reflect_max_tokens=reflect_max_tokens,
        reflect_response_schema=reflect_response_schema,
        reflect_tags=reflect_tags,
        reflect_tags_match=reflect_tags_match,
        mission=mission,
    )

    backend = _HindsightToolBackend(client=client, **backend_kwargs)

    return _build_tools(
        backend,
        backend_kwargs,
        include_retain=include_retain,
        include_recall=include_recall,
        include_reflect=include_reflect,
    )


class HindsightToolset(Toolset):
    """Haystack ``Toolset`` with optional auto-recall and auto-retain.

    Groups Hindsight memory tools into a single toolset and optionally adds
    automatic memory behavior:

    - **auto_recall**: Before each agent turn, recalls relevant memories and
      prepends them to the system prompt so the agent has context without
      needing to call a tool.
    - **auto_retain**: After each agent turn, retains user and assistant
      messages to Hindsight for long-term storage.

    Use :meth:`run` / :meth:`run_async` for automatic behavior, or pass the
    toolset directly to ``Agent(tools=...)`` for tool-only (explicit) mode.

    Example::

        from hindsight_haystack import HindsightToolset
        from haystack.components.agents import Agent
        from haystack.components.generators.chat import OpenAIChatGenerator

        toolset = HindsightToolset(
            bank_id="user-123",
            client=client,
            mission="Track user preferences",
            auto_recall=True,
            auto_retain=True,
        )

        agent = Agent(
            chat_generator=OpenAIChatGenerator(),
            tools=toolset,
            system_prompt="You are a helpful assistant with long-term memory.",
        )

        # Use toolset.run() for auto-recall/retain behavior
        result = toolset.run(agent, messages=[ChatMessage.from_user("Hi!")])

    Args:
        bank_id: The Hindsight memory bank to operate on.
        client: Pre-configured Hindsight client (preferred).
        hindsight_api_url: API URL (used if no client provided).
        api_key: API key (used if no client provided).
        budget: Recall/reflect budget level (low/mid/high).
        max_tokens: Maximum tokens for recall results.
        tags: Tags applied when storing memories via retain.
        recall_tags: Tags to filter when searching memories.
        recall_tags_match: Tag matching mode.
        retain_metadata: Default metadata for retain operations.
        retain_document_id: Default document_id for retain.
        retain_context: Source label for retain operations.
        recall_types: Fact types to filter on recall.
        recall_include_entities: Include entities in recall results.
        reflect_context: Additional context for reflect.
        reflect_max_tokens: Max tokens for reflect.
        reflect_response_schema: JSON schema for structured reflect output.
        reflect_tags: Tags to filter reflect memories.
        reflect_tags_match: Tag matching for reflect.
        mission: Bank mission for fact extraction context.
        include_retain: Include the retain tool (default True).
        include_recall: Include the recall tool (default True).
        include_reflect: Include the reflect tool (default True).
        auto_recall: Auto-recall memories into system prompt before each turn.
        auto_retain: Auto-retain user/assistant messages after each turn.
        max_recall_results: Maximum number of memories to inject into the
            system prompt during auto-recall (default 10).
        memory_prompt_template: Template for injecting memories into system
            prompt. Must contain ``{memories}`` placeholder.
    """

    def __init__(
        self,
        *,
        bank_id: str,
        client: Optional[Hindsight] = None,
        hindsight_api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        budget: Optional[str] = None,
        max_tokens: Optional[int] = None,
        tags: Optional[list[str]] = None,
        recall_tags: Optional[list[str]] = None,
        recall_tags_match: Optional[str] = None,
        retain_metadata: Optional[dict[str, str]] = None,
        retain_document_id: Optional[str] = None,
        retain_context: Optional[str] = None,
        recall_types: Optional[list[str]] = None,
        recall_include_entities: bool = False,
        reflect_context: Optional[str] = None,
        reflect_max_tokens: Optional[int] = None,
        reflect_response_schema: Optional[dict[str, Any]] = None,
        reflect_tags: Optional[list[str]] = None,
        reflect_tags_match: Optional[str] = None,
        mission: Optional[str] = None,
        include_retain: bool = True,
        include_recall: bool = True,
        include_reflect: bool = True,
        auto_recall: bool = False,
        auto_retain: bool = False,
        max_recall_results: int = 10,
        memory_prompt_template: str = DEFAULT_MEMORY_PROMPT,
    ):
        backend_kwargs = _build_backend_kwargs(
            bank_id=bank_id,
            client=client,
            hindsight_api_url=hindsight_api_url,
            api_key=api_key,
            budget=budget,
            max_tokens=max_tokens,
            tags=tags,
            recall_tags=recall_tags,
            recall_tags_match=recall_tags_match,
            retain_metadata=retain_metadata,
            retain_document_id=retain_document_id,
            retain_context=retain_context,
            recall_types=recall_types,
            recall_include_entities=recall_include_entities,
            reflect_context=reflect_context,
            reflect_max_tokens=reflect_max_tokens,
            reflect_response_schema=reflect_response_schema,
            reflect_tags=reflect_tags,
            reflect_tags_match=reflect_tags_match,
            mission=mission,
        )

        self._backend = _HindsightToolBackend(client=client, **backend_kwargs)
        self._backend_kwargs = backend_kwargs
        self._auto_recall = auto_recall
        self._auto_retain = auto_retain
        self._max_recall_results = max_recall_results
        self._memory_prompt_template = memory_prompt_template
        self._include_retain = include_retain
        self._include_recall = include_recall
        self._include_reflect = include_reflect

        tools = _build_tools(
            self._backend,
            backend_kwargs,
            include_retain=include_retain,
            include_recall=include_recall,
            include_reflect=include_reflect,
        )
        super().__init__(tools=tools)

    def _recall_for_prompt(self, query: str) -> str:
        """Recall memories and format for system prompt injection.

        Caps results at ``max_recall_results`` to prevent unbounded prompt growth.
        """
        try:
            self._backend._ensure_bank()
            response = _run_sync(self._backend._client.arecall(**self._backend._recall_kwargs(query)))
            if not response.results:
                return ""
            lines = []
            for i, r in enumerate(response.results[: self._max_recall_results], 1):
                lines.append(f"{i}. {r.text}")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Auto-recall failed: {e}")
            return ""

    @staticmethod
    def _extract_last_user_text(messages: list[ChatMessage]) -> str:
        """Extract the text of the last user message."""
        for msg in reversed(messages):
            if msg.role.value == "user" and msg.text:
                return msg.text
        return ""

    def _enrich_system_prompt(self, base_prompt: Optional[str], query: str) -> Optional[str]:
        """Recall memories and append to the system prompt."""
        memories = self._recall_for_prompt(query)
        if not memories:
            return base_prompt
        memory_block = self._memory_prompt_template.format(memories=memories)
        if base_prompt:
            return f"{base_prompt}\n\n{memory_block}"
        return memory_block

    def _retain_messages(self, messages: list[ChatMessage]) -> None:
        """Retain user and assistant messages to Hindsight.

        Includes message role in metadata so recalled turns are distinguishable.
        """
        for msg in messages:
            role = msg.role.value
            if role in ("user", "assistant") and msg.text:
                try:
                    self._backend._ensure_bank()
                    kwargs = self._backend._retain_kwargs(msg.text)
                    # Merge role metadata with any configured retain_metadata
                    existing_meta = kwargs.get("metadata") or {}
                    kwargs["metadata"] = {**existing_meta, "role": role, "source": "haystack"}
                    _run_sync(self._backend._client.aretain(**kwargs))
                except Exception as e:
                    logger.error(f"Auto-retain failed for {role} message: {e}")

    def run(
        self,
        agent: Any,
        *,
        messages: list[ChatMessage],
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run an agent with auto-recall and auto-retain.

        Wraps ``agent.run()`` with automatic memory behavior:

        1. **Auto-recall** (if enabled): Recalls memories relevant to the
           user's message and prepends them to the system prompt.
        2. **Agent execution**: Calls ``agent.run()`` with the enriched
           system prompt.
        3. **Auto-retain** (if enabled): Retains the user messages and the
           agent's final response to Hindsight.

        Args:
            agent: A Haystack ``Agent`` instance.
            messages: User messages to process.
            system_prompt: Base system prompt. If None, uses the agent's
                configured ``system_prompt``.
            **kwargs: Additional kwargs passed to ``agent.run()``.

        Returns:
            The result dict from ``agent.run()``.
        """
        base_prompt = system_prompt if system_prompt is not None else getattr(agent, "system_prompt", None)

        # Auto-recall: enrich system prompt with relevant memories
        effective_prompt = base_prompt
        if self._auto_recall:
            user_text = self._extract_last_user_text(messages)
            if user_text:
                effective_prompt = self._enrich_system_prompt(base_prompt, user_text)

        result = agent.run(messages=messages, system_prompt=effective_prompt, **kwargs)

        # Auto-retain: store user messages and agent response
        if self._auto_retain:
            self._retain_messages(messages)
            last_msg = result.get("last_message")
            if last_msg and last_msg.role.value == "assistant" and last_msg.text:
                self._retain_messages([last_msg])

        return result

    async def run_async(
        self,
        agent: Any,
        *,
        messages: list[ChatMessage],
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Async version of :meth:`run`.

        Uses the same persistent event loop bridge internally, so auto-recall
        and auto-retain work identically to the sync version.

        Args:
            agent: A Haystack ``Agent`` instance.
            messages: User messages to process.
            system_prompt: Base system prompt override.
            **kwargs: Additional kwargs passed to ``agent.run_async()``.

        Returns:
            The result dict from ``agent.run_async()``.
        """
        base_prompt = system_prompt if system_prompt is not None else getattr(agent, "system_prompt", None)

        effective_prompt = base_prompt
        if self._auto_recall:
            user_text = self._extract_last_user_text(messages)
            if user_text:
                effective_prompt = self._enrich_system_prompt(base_prompt, user_text)

        result = await agent.run_async(messages=messages, system_prompt=effective_prompt, **kwargs)

        if self._auto_retain:
            self._retain_messages(messages)
            last_msg = result.get("last_message")
            if last_msg and last_msg.role.value == "assistant" and last_msg.text:
                self._retain_messages([last_msg])

        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialize the toolset to a dictionary."""
        cls = type(self)
        qualified_name = f"{cls.__module__}.{cls.__qualname__}"
        return {
            "type": qualified_name,
            "data": {
                "backend_kwargs": self._backend_kwargs,
                "include_retain": self._include_retain,
                "include_recall": self._include_recall,
                "include_reflect": self._include_reflect,
                "auto_recall": self._auto_recall,
                "auto_retain": self._auto_retain,
                "max_recall_results": self._max_recall_results,
                "memory_prompt_template": self._memory_prompt_template,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HindsightToolset":
        """Deserialize the toolset from a dictionary."""
        inner = data["data"]
        backend_kwargs = inner["backend_kwargs"]
        return cls(
            **backend_kwargs,
            include_retain=inner.get("include_retain", True),
            include_recall=inner.get("include_recall", True),
            include_reflect=inner.get("include_reflect", True),
            auto_recall=inner.get("auto_recall", False),
            auto_retain=inner.get("auto_retain", False),
            max_recall_results=inner.get("max_recall_results", 10),
            memory_prompt_template=inner.get("memory_prompt_template", DEFAULT_MEMORY_PROMPT),
        )
