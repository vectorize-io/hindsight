"""Native client wrappers for Hindsight memory integration.

This module provides wrappers for native LLM client SDKs (OpenAI, Anthropic)
that automatically integrate with Hindsight for memory injection and storage.

This is an alternative to the LiteLLM callback approach, providing direct
integration with native client libraries.
"""

import logging
import threading
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

from .config import get_config, get_defaults, is_configured, HindsightConfig


# Background thread support for async retain
_retain_errors: List[Exception] = []
_retain_errors_lock = threading.Lock()

logger = logging.getLogger(__name__)


def _get_client(api_url: str):
    """Create a fresh Hindsight client for the given URL.

    Note: We create a fresh client each time because the hindsight_client
    uses aiohttp internally, and reusing clients across different sync
    calls causes asyncio context issues.
    """
    from hindsight_client import Hindsight
    return Hindsight(base_url=api_url, timeout=30.0)


def _close_client():
    """No-op for compatibility. Clients are now closed after each use."""
    pass


@dataclass
class RecallResult:
    """A single memory recall result."""
    text: str
    fact_type: str
    weight: float
    metadata: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        return self.text


@dataclass
class RecallDebugInfo:
    """Debug information from a recall operation."""
    query: str
    bank_id: str
    budget: str
    max_tokens: int
    fact_types: Optional[List[str]]
    results_count: int
    api_url: str


@dataclass
class RecallResponse:
    """Response from a recall operation, including results and optional debug info."""
    results: List[RecallResult]
    debug: Optional[RecallDebugInfo] = None

    def __iter__(self):
        return iter(self.results)

    def __len__(self):
        return len(self.results)

    def __getitem__(self, key):
        return self.results[key]

    def __bool__(self):
        return bool(self.results)


def recall(
    query: str,
    bank_id: Optional[str] = None,
    fact_types: Optional[List[str]] = None,
    budget: Optional[str] = None,
    max_tokens: Optional[int] = None,
    hindsight_api_url: Optional[str] = None,
) -> RecallResponse:
    """Recall memories from Hindsight.

    This function allows you to manually query memories without making an LLM call.
    Useful for debugging, building custom UIs, or pre-filtering memories.

    Args:
        query: The query string to search memories for
        bank_id: Override the configured bank_id. For multi-user support,
            use different bank_ids per user (e.g., f"user-{user_id}")
        fact_types: Filter by fact types (world, agent, opinion, observation)
        budget: Recall budget level (low, mid, high) - controls how many memories are returned
        max_tokens: Maximum tokens for memory context
        hindsight_api_url: Override the configured API URL

    Returns:
        RecallResponse containing matched memories (iterable like a list).
        When verbose=True in config, includes debug info via .debug attribute.

    Raises:
        RuntimeError: If Hindsight is not configured and no overrides provided

    Example:
        >>> from hindsight_litellm import configure, recall
        >>> configure(bank_id="my-agent", hindsight_api_url="http://localhost:8888")
        >>>
        >>> # Query memories
        >>> memories = recall("what projects am I working on?")
        >>> for m in memories:
        ...     print(f"- [{m.fact_type}] {m.text}")
        - [world] User is building a FastAPI project
        - [opinion] User prefers Python over JavaScript
        >>>
        >>> # With verbose mode, access debug info
        >>> configure(bank_id="my-agent", verbose=True)
        >>> memories = recall("what projects am I working on?")
        >>> if memories.debug:
        ...     print(f"Queried bank: {memories.debug.bank_id}")
    """
    # Get config and defaults, or use overrides
    config = get_config()
    defaults = get_defaults()

    api_url = hindsight_api_url or (config.hindsight_api_url if config else None)
    target_bank_id = bank_id or (defaults.bank_id if defaults else None)
    target_fact_types = fact_types or (defaults.fact_types if defaults else None)
    target_budget = budget or (defaults.budget if defaults else "mid")
    target_max_tokens = max_tokens or (defaults.max_memory_tokens if defaults else 4096)

    if not api_url or not target_bank_id:
        raise RuntimeError(
            "Hindsight not configured. Call configure() or provide bank_id and hindsight_api_url."
        )

    client = None
    try:
        # Create fresh client for this operation
        client = _get_client(api_url)

        # Call recall API
        results = client.recall(
            bank_id=target_bank_id,
            query=query,
            types=target_fact_types,
            budget=target_budget,
            max_tokens=target_max_tokens,
        )

        # Convert to RecallResult objects
        recall_results = []
        if results:
            for r in results:
                if hasattr(r, 'text'):
                    # Object with attributes
                    fact_type = getattr(r, 'type', None) or getattr(r, 'fact_type', 'unknown')
                    recall_results.append(RecallResult(
                        text=r.text,
                        fact_type=fact_type,
                        weight=getattr(r, 'weight', 0.0),
                        metadata=getattr(r, 'metadata', None),
                    ))
                elif isinstance(r, dict):
                    # Dict from API response - API returns 'type' not 'fact_type'
                    fact_type = r.get('type') or r.get('fact_type', 'unknown')
                    recall_results.append(RecallResult(
                        text=r.get('text', str(r)),
                        fact_type=fact_type,
                        weight=r.get('weight', 0.0),
                        metadata=r.get('metadata'),
                    ))

        # Include debug info if verbose
        debug_info = None
        if config and config.verbose:
            debug_info = RecallDebugInfo(
                query=query,
                bank_id=target_bank_id,
                budget=target_budget,
                max_tokens=target_max_tokens,
                fact_types=target_fact_types,
                results_count=len(recall_results),
                api_url=api_url,
            )

        return RecallResponse(results=recall_results, debug=debug_info)

    except ImportError as e:
        raise RuntimeError(f"hindsight-client not installed: {e}")
    except Exception as e:
        if config and config.verbose:
            logger.warning(f"Failed to recall memories: {e}")
        raise
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


async def arecall(
    query: str,
    bank_id: Optional[str] = None,
    fact_types: Optional[List[str]] = None,
    budget: Optional[str] = None,
    max_tokens: Optional[int] = None,
    hindsight_api_url: Optional[str] = None,
) -> RecallResponse:
    """Async version of recall().

    See recall() for full documentation.
    """
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: recall(
            query=query,
            bank_id=bank_id,
            fact_types=fact_types,
            budget=budget,
            max_tokens=max_tokens,
            hindsight_api_url=hindsight_api_url,
        )
    )


@dataclass
class ReflectDebugInfo:
    """Debug information from a reflect operation."""
    query: str
    bank_id: str
    budget: str
    context: Optional[str]
    api_url: str


@dataclass
class ReflectResult:
    """Result from a reflect operation."""
    text: str
    based_on: Optional[Dict[str, List[Any]]] = None
    debug: Optional[ReflectDebugInfo] = None

    def __str__(self) -> str:
        return self.text


def reflect(
    query: str,
    bank_id: Optional[str] = None,
    budget: Optional[str] = None,
    context: Optional[str] = None,
    response_schema: Optional[dict] = None,
    hindsight_api_url: Optional[str] = None,
) -> ReflectResult:
    """Generate a contextual answer based on memories.

    Unlike recall() which returns raw memory facts, reflect() uses an LLM
    to synthesize a coherent answer based on the bank's memories.

    Args:
        query: The question or prompt to answer
        bank_id: Override the configured bank_id. For multi-user support,
            use different bank_ids per user (e.g., f"user-{user_id}")
        budget: Budget level for reflection (low, mid, high)
        context: Additional context to include in the reflection
        response_schema: JSON Schema for structured output
        hindsight_api_url: Override the configured API URL

    Returns:
        ReflectResult with synthesized answer text (or structured_output if schema provided)

    Raises:
        RuntimeError: If Hindsight is not configured and no overrides provided

    Example:
        >>> from hindsight_litellm import configure, reflect
        >>> configure(bank_id="my-agent", hindsight_api_url="http://localhost:8888")
        >>>
        >>> # Get a synthesized answer based on memories
        >>> result = reflect("What projects am I working on?")
        >>> print(result.text)
        Based on our conversations, you're working on a FastAPI project...
    """
    config = get_config()
    defaults = get_defaults()

    api_url = hindsight_api_url or (config.hindsight_api_url if config else None)
    target_bank_id = bank_id or (defaults.bank_id if defaults else None)
    target_budget = budget or (defaults.budget if defaults else "mid")

    if not api_url or not target_bank_id:
        raise RuntimeError(
            "Hindsight not configured. Call configure() or provide bank_id and hindsight_api_url."
        )

    client = None
    try:
        # Create fresh client for this operation
        client = _get_client(api_url)

        # Call reflect API
        reflect_kwargs = {
            "bank_id": target_bank_id,
            "query": query,
            "budget": target_budget,
        }
        if context is not None:
            reflect_kwargs["context"] = context
        if response_schema is not None:
            reflect_kwargs["response_schema"] = response_schema
        result = client.reflect(**reflect_kwargs)

        # Convert to ReflectResult
        text = result.text if hasattr(result, 'text') else str(result)
        based_on = getattr(result, 'based_on', None)

        # Include debug info if verbose
        debug_info = None
        if config and config.verbose:
            debug_info = ReflectDebugInfo(
                query=query,
                bank_id=target_bank_id,
                budget=target_budget,
                context=context,
                api_url=api_url,
            )

        return ReflectResult(text=text, based_on=based_on, debug=debug_info)

    except ImportError as e:
        raise RuntimeError(f"hindsight-client not installed: {e}")
    except Exception as e:
        if config and config.verbose:
            logger.warning(f"Failed to reflect: {e}")
        raise
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


async def areflect(
    query: str,
    bank_id: Optional[str] = None,
    budget: Optional[str] = None,
    context: Optional[str] = None,
    response_schema: Optional[dict] = None,
    hindsight_api_url: Optional[str] = None,
) -> ReflectResult:
    """Async version of reflect().

    See reflect() for full documentation.
    """
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: reflect(
            query=query,
            bank_id=bank_id,
            budget=budget,
            context=context,
            response_schema=response_schema,
            hindsight_api_url=hindsight_api_url,
        )
    )


@dataclass
class RetainDebugInfo:
    """Debug information from a retain operation."""
    content: str
    bank_id: str
    context: Optional[str]
    document_id: Optional[str]
    metadata: Optional[Dict[str, str]]
    api_url: str


@dataclass
class RetainResult:
    """Result from a retain operation."""
    success: bool
    items_count: int = 0
    debug: Optional[RetainDebugInfo] = None

    def __bool__(self) -> bool:
        return self.success


def _retain_sync(
    content: str,
    api_url: str,
    target_bank_id: str,
    context: Optional[str],
    target_document_id: Optional[str],
    metadata: Optional[Dict[str, str]],
    verbose: bool,
) -> RetainResult:
    """Internal synchronous retain implementation."""
    client = None
    try:
        # Create fresh client for this operation
        client = _get_client(api_url)

        # Call retain API
        result = client.retain(
            bank_id=target_bank_id,
            content=content,
            context=context,
            document_id=target_document_id,
            metadata=metadata,
        )

        # Check success
        success = getattr(result, 'success', True)
        items_count = getattr(result, 'items_count', 1)

        # Include debug info if verbose
        debug_info = None
        if verbose:
            logger.info(f"Stored content to Hindsight bank: {target_bank_id}")
            debug_info = RetainDebugInfo(
                content=content,
                bank_id=target_bank_id,
                context=context,
                document_id=target_document_id,
                metadata=metadata,
                api_url=api_url,
            )

        return RetainResult(success=success, items_count=items_count, debug=debug_info)

    except ImportError as e:
        raise RuntimeError(f"hindsight-client not installed: {e}")
    except Exception as e:
        if verbose:
            logger.warning(f"Failed to retain: {e}")
        raise
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


def _retain_background(
    content: str,
    api_url: str,
    target_bank_id: str,
    context: Optional[str],
    target_document_id: Optional[str],
    metadata: Optional[Dict[str, str]],
    verbose: bool,
) -> None:
    """Background thread worker for async retain."""
    global _retain_errors
    try:
        _retain_sync(
            content=content,
            api_url=api_url,
            target_bank_id=target_bank_id,
            context=context,
            target_document_id=target_document_id,
            metadata=metadata,
            verbose=verbose,
        )
    except Exception as e:
        with _retain_errors_lock:
            _retain_errors.append(e)
        logger.warning(f"Background retain failed: {e}")


def get_pending_retain_errors() -> List[Exception]:
    """Get and clear any pending errors from background retain operations.

    When using async retain (sync=False), errors are collected in the background.
    Call this periodically to check for and handle any failures.

    Returns:
        List of exceptions from failed background retain operations.
        The list is cleared after calling this function.

    Example:
        >>> errors = get_pending_retain_errors()
        >>> if errors:
        ...     for e in errors:
        ...         print(f"Retain failed: {e}")
    """
    global _retain_errors
    with _retain_errors_lock:
        errors = _retain_errors.copy()
        _retain_errors.clear()
    return errors


def retain(
    content: str,
    bank_id: Optional[str] = None,
    context: Optional[str] = None,
    document_id: Optional[str] = None,
    metadata: Optional[Dict[str, str]] = None,
    hindsight_api_url: Optional[str] = None,
    sync: bool = False,
) -> RetainResult:
    """Store content to Hindsight memory.

    This function allows you to manually store content to memory without
    making an LLM call. Useful for storing feedback, user preferences,
    or any other information you want the system to remember.

    Args:
        content: The text content to store
        bank_id: Override the configured bank_id. For multi-user support,
            use different bank_ids per user (e.g., f"user-{user_id}")
        context: Context description for the memory (e.g., "customer_feedback")
        document_id: Optional document ID for grouping related memories
        metadata: Optional key-value metadata to attach to the memory
        hindsight_api_url: Override the configured API URL
        sync: If True, block until storage completes. If False (default),
            run in background thread for better performance. Use
            get_pending_retain_errors() to check for async failures.

    Returns:
        RetainResult indicating success. For async mode (sync=False),
        always returns success=True immediately; actual errors are
        collected via get_pending_retain_errors().

    Raises:
        RuntimeError: If Hindsight is not configured and no overrides provided
        Exception: Only raised in sync mode if storage fails

    Example:
        >>> from hindsight_litellm import configure, retain
        >>> configure(bank_id="my-agent", hindsight_api_url="http://localhost:8888")
        >>>
        >>> # Async retain (default) - fast, non-blocking
        >>> retain("User prefers dark mode", context="user_preference")
        >>>
        >>> # Sync retain - blocks until complete
        >>> retain("Critical data", sync=True)
        >>>
        >>> # Check for async errors
        >>> errors = get_pending_retain_errors()
    """
    config = get_config()
    defaults = get_defaults()

    api_url = hindsight_api_url or (config.hindsight_api_url if config else None)
    target_bank_id = bank_id or (defaults.bank_id if defaults else None)
    target_document_id = document_id or (defaults.document_id if defaults else None)
    verbose = config.verbose if config else False

    if not api_url or not target_bank_id:
        raise RuntimeError(
            "Hindsight not configured. Call configure() or provide bank_id and hindsight_api_url."
        )

    if sync:
        # Synchronous mode - block and return result
        return _retain_sync(
            content=content,
            api_url=api_url,
            target_bank_id=target_bank_id,
            context=context,
            target_document_id=target_document_id,
            metadata=metadata,
            verbose=verbose,
        )
    else:
        # Async mode - run in background thread
        thread = threading.Thread(
            target=_retain_background,
            args=(
                content,
                api_url,
                target_bank_id,
                context,
                target_document_id,
                metadata,
                verbose,
            ),
            daemon=True,
        )
        thread.start()
        # Return immediate success - actual errors collected via get_pending_retain_errors()
        return RetainResult(success=True, items_count=0)


async def aretain(
    content: str,
    bank_id: Optional[str] = None,
    context: Optional[str] = None,
    document_id: Optional[str] = None,
    metadata: Optional[Dict[str, str]] = None,
    hindsight_api_url: Optional[str] = None,
) -> RetainResult:
    """Async version of retain().

    See retain() for full documentation.
    """
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: retain(
            content=content,
            bank_id=bank_id,
            context=context,
            document_id=document_id,
            metadata=metadata,
            hindsight_api_url=hindsight_api_url,
        )
    )


class HindsightOpenAI:
    """Wrapper for OpenAI client with Hindsight memory integration.

    This wraps the native OpenAI client to automatically inject memories
    and store conversations.

    Example:
        >>> from openai import OpenAI
        >>> from hindsight_litellm import wrap_openai
        >>>
        >>> client = OpenAI()
        >>> wrapped = wrap_openai(client, bank_id="my-agent")
        >>>
        >>> response = wrapped.chat.completions.create(
        ...     model="gpt-4",
        ...     messages=[{"role": "user", "content": "What do you know about me?"}]
        ... )
    """

    def __init__(
        self,
        client: Any,
        bank_id: str,
        hindsight_api_url: str = "http://localhost:8888",
        session_id: Optional[str] = None,
        store_conversations: bool = True,
        inject_memories: bool = True,
        max_memories: Optional[int] = None,
        budget: str = "mid",
        verbose: bool = False,
    ):
        """Initialize the wrapped OpenAI client.

        Args:
            client: The OpenAI client instance to wrap
            bank_id: Memory bank ID for memory operations. For multi-user support,
                use different bank_ids per user (e.g., f"user-{user_id}")
            hindsight_api_url: URL of the Hindsight API server
            session_id: Session identifier for conversation grouping
            store_conversations: Whether to store conversations
            inject_memories: Whether to inject relevant memories
            max_memories: Maximum number of memories to inject (None = no limit)
            budget: Budget level for memory recall (low, mid, high)
            verbose: Enable verbose logging
        """
        self._client = client
        self._bank_id = bank_id
        self._api_url = hindsight_api_url
        self._session_id = session_id
        self._store_conversations = store_conversations
        self._inject_memories = inject_memories
        self._max_memories = max_memories
        self._budget = budget
        self._verbose = verbose
        self._hindsight_client = None

        # Create wrapped chat.completions interface
        self.chat = _WrappedChat(self)

    def _get_hindsight_client(self):
        """Get or create the Hindsight client."""
        if self._hindsight_client is None:
            from hindsight_client import Hindsight
            self._hindsight_client = Hindsight(
                base_url=self._api_url,
                timeout=30.0,
            )
        return self._hindsight_client

    def _recall_memories(self, query: str) -> str:
        """Recall and format memories for injection."""
        if not self._inject_memories:
            return ""

        try:
            client = self._get_hindsight_client()
            results = client.recall(
                bank_id=self._bank_id,
                query=query,
                budget=self._budget,
                max_tokens=self._max_memories * 200 if self._max_memories else 4096,
            )

            if not results:
                return ""

            results_to_use = results[:self._max_memories] if self._max_memories else results
            memory_lines = []
            for i, r in enumerate(results_to_use, 1):
                text = r.text if hasattr(r, 'text') else str(r)
                fact_type = r.fact_type if hasattr(r, 'fact_type') else 'memory'
                memory_lines.append(f"{i}. [{fact_type.upper()}] {text}")

            if not memory_lines:
                return ""

            return (
                "# Relevant Memories\n"
                "The following information from memory may be relevant:\n\n"
                + "\n".join(memory_lines)
            )

        except Exception as e:
            if self._verbose:
                logger.warning(f"Failed to recall memories: {e}")
            return ""

    def _store_conversation(self, user_input: str, assistant_output: str, model: str):
        """Store the conversation to Hindsight."""
        if not self._store_conversations:
            return

        try:
            client = self._get_hindsight_client()
            conversation_text = f"USER: {user_input}\n\nASSISTANT: {assistant_output}"

            metadata = {
                "source": "openai-wrapper",
                "model": model,
            }
            if self._session_id:
                metadata["session_id"] = self._session_id

            client.retain(
                bank_id=self._bank_id,
                content=conversation_text,
                context=f"conversation:openai:{model}",
                metadata=metadata,
            )

            if self._verbose:
                logger.info(f"Stored conversation to Hindsight")

        except Exception as e:
            if self._verbose:
                logger.warning(f"Failed to store conversation: {e}")

    # Proxy other attributes to the underlying client
    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class _WrappedChat:
    """Wrapped chat interface for OpenAI client."""

    def __init__(self, wrapper: HindsightOpenAI):
        self._wrapper = wrapper
        self.completions = _WrappedCompletions(wrapper)


class _WrappedCompletions:
    """Wrapped completions interface for OpenAI client."""

    def __init__(self, wrapper: HindsightOpenAI):
        self._wrapper = wrapper

    def create(self, **kwargs) -> Any:
        """Create a chat completion with memory integration."""
        messages = list(kwargs.get("messages", []))
        model = kwargs.get("model", "gpt-4")

        # Extract user query
        user_query = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    user_query = content
                    break

        # Inject memories
        if user_query and self._wrapper._inject_memories:
            memory_context = self._wrapper._recall_memories(user_query)
            if memory_context:
                # Find system message and append, or prepend new one
                found_system = False
                for i, msg in enumerate(messages):
                    if msg.get("role") == "system":
                        messages[i] = {
                            **msg,
                            "content": f"{msg.get('content', '')}\n\n{memory_context}"
                        }
                        found_system = True
                        break

                if not found_system:
                    messages.insert(0, {"role": "system", "content": memory_context})

                kwargs["messages"] = messages

        # Make the actual API call
        response = self._wrapper._client.chat.completions.create(**kwargs)

        # Store conversation
        if user_query and self._wrapper._store_conversations:
            if response.choices and response.choices[0].message:
                assistant_output = response.choices[0].message.content or ""
                if assistant_output:
                    self._wrapper._store_conversation(user_query, assistant_output, model)

        return response


class HindsightAnthropic:
    """Wrapper for Anthropic client with Hindsight memory integration.

    This wraps the native Anthropic client to automatically inject memories
    and store conversations.

    Example:
        >>> from anthropic import Anthropic
        >>> from hindsight_litellm import wrap_anthropic
        >>>
        >>> client = Anthropic()
        >>> wrapped = wrap_anthropic(client, bank_id="my-agent")
        >>>
        >>> response = wrapped.messages.create(
        ...     model="claude-3-5-sonnet-20241022",
        ...     max_tokens=1024,
        ...     messages=[{"role": "user", "content": "What do you know about me?"}]
        ... )
    """

    def __init__(
        self,
        client: Any,
        bank_id: str,
        hindsight_api_url: str = "http://localhost:8888",
        session_id: Optional[str] = None,
        store_conversations: bool = True,
        inject_memories: bool = True,
        max_memories: Optional[int] = None,
        budget: str = "mid",
        verbose: bool = False,
    ):
        """Initialize the wrapped Anthropic client.

        Args:
            client: The Anthropic client instance to wrap
            bank_id: Memory bank ID for memory operations. For multi-user support,
                use different bank_ids per user (e.g., f"user-{user_id}")
            hindsight_api_url: URL of the Hindsight API server
            session_id: Session identifier for conversation grouping
            store_conversations: Whether to store conversations
            inject_memories: Whether to inject relevant memories
            max_memories: Maximum number of memories to inject (None = no limit)
            budget: Budget level for memory recall (low, mid, high)
            verbose: Enable verbose logging
        """
        self._client = client
        self._bank_id = bank_id
        self._api_url = hindsight_api_url
        self._session_id = session_id
        self._store_conversations = store_conversations
        self._inject_memories = inject_memories
        self._max_memories = max_memories
        self._budget = budget
        self._verbose = verbose
        self._hindsight_client = None

        # Create wrapped messages interface
        self.messages = _WrappedAnthropicMessages(self)

    def _get_hindsight_client(self):
        """Get or create the Hindsight client."""
        if self._hindsight_client is None:
            from hindsight_client import Hindsight
            self._hindsight_client = Hindsight(
                base_url=self._api_url,
                timeout=30.0,
            )
        return self._hindsight_client

    def _recall_memories(self, query: str) -> str:
        """Recall and format memories for injection."""
        if not self._inject_memories:
            return ""

        try:
            client = self._get_hindsight_client()
            results = client.recall(
                bank_id=self._bank_id,
                query=query,
                budget=self._budget,
                max_tokens=self._max_memories * 200 if self._max_memories else 4096,
            )

            if not results:
                return ""

            results_to_use = results[:self._max_memories] if self._max_memories else results
            memory_lines = []
            for i, r in enumerate(results_to_use, 1):
                text = r.text if hasattr(r, 'text') else str(r)
                fact_type = r.fact_type if hasattr(r, 'fact_type') else 'memory'
                memory_lines.append(f"{i}. [{fact_type.upper()}] {text}")

            if not memory_lines:
                return ""

            return (
                "# Relevant Memories\n"
                "The following information from memory may be relevant:\n\n"
                + "\n".join(memory_lines)
            )

        except Exception as e:
            if self._verbose:
                logger.warning(f"Failed to recall memories: {e}")
            return ""

    def _store_conversation(self, user_input: str, assistant_output: str, model: str):
        """Store the conversation to Hindsight."""
        if not self._store_conversations:
            return

        try:
            client = self._get_hindsight_client()
            conversation_text = f"USER: {user_input}\n\nASSISTANT: {assistant_output}"

            metadata = {
                "source": "anthropic-wrapper",
                "model": model,
            }
            if self._session_id:
                metadata["session_id"] = self._session_id

            client.retain(
                bank_id=self._bank_id,
                content=conversation_text,
                context=f"conversation:anthropic:{model}",
                metadata=metadata,
            )

            if self._verbose:
                logger.info(f"Stored conversation to Hindsight")

        except Exception as e:
            if self._verbose:
                logger.warning(f"Failed to store conversation: {e}")

    # Proxy other attributes to the underlying client
    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class _WrappedAnthropicMessages:
    """Wrapped messages interface for Anthropic client."""

    def __init__(self, wrapper: HindsightAnthropic):
        self._wrapper = wrapper

    def create(self, **kwargs) -> Any:
        """Create a message with memory integration."""
        messages = list(kwargs.get("messages", []))
        model = kwargs.get("model", "claude-3-5-sonnet-20241022")
        system = kwargs.get("system", "")

        # Extract user query
        user_query = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    user_query = content
                    break
                elif isinstance(content, list):
                    # Handle structured content
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            user_query = item.get("text", "")
                            break
                    if user_query:
                        break

        # Inject memories into system prompt
        if user_query and self._wrapper._inject_memories:
            memory_context = self._wrapper._recall_memories(user_query)
            if memory_context:
                if system:
                    kwargs["system"] = f"{system}\n\n{memory_context}"
                else:
                    kwargs["system"] = memory_context

        # Make the actual API call
        response = self._wrapper._client.messages.create(**kwargs)

        # Store conversation
        if user_query and self._wrapper._store_conversations:
            if response.content:
                assistant_output = ""
                for block in response.content:
                    if hasattr(block, 'text'):
                        assistant_output += block.text
                if assistant_output:
                    self._wrapper._store_conversation(user_query, assistant_output, model)

        return response


def wrap_openai(
    client: Any,
    bank_id: str,
    hindsight_api_url: str = "http://localhost:8888",
    session_id: Optional[str] = None,
    store_conversations: bool = True,
    inject_memories: bool = True,
    max_memories: Optional[int] = None,
    budget: str = "mid",
    verbose: bool = False,
) -> HindsightOpenAI:
    """Wrap an OpenAI client with Hindsight memory integration.

    This creates a wrapped client that automatically injects memories
    and stores conversations when making chat completion calls.

    Args:
        client: The OpenAI client instance to wrap
        bank_id: Memory bank ID for memory operations. For multi-user support,
            use different bank_ids per user (e.g., f"user-{user_id}")
        hindsight_api_url: URL of the Hindsight API server
        session_id: Session identifier for conversation grouping
        store_conversations: Whether to store conversations
        inject_memories: Whether to inject relevant memories
        max_memories: Maximum number of memories to inject (None = no limit)
        budget: Budget level for memory recall (low, mid, high)
        verbose: Enable verbose logging

    Returns:
        Wrapped OpenAI client with memory integration

    Example:
        >>> from openai import OpenAI
        >>> from hindsight_litellm import wrap_openai
        >>>
        >>> client = OpenAI()
        >>> wrapped = wrap_openai(
        ...     client,
        ...     bank_id=f"user-{user_id}",  # Multi-user support via separate banks
        ... )
        >>>
        >>> response = wrapped.chat.completions.create(
        ...     model="gpt-4",
        ...     messages=[{"role": "user", "content": "What do you know about me?"}]
        ... )
    """
    return HindsightOpenAI(
        client=client,
        bank_id=bank_id,
        hindsight_api_url=hindsight_api_url,
        session_id=session_id,
        store_conversations=store_conversations,
        inject_memories=inject_memories,
        max_memories=max_memories,
        budget=budget,
        verbose=verbose,
    )


def wrap_anthropic(
    client: Any,
    bank_id: str,
    hindsight_api_url: str = "http://localhost:8888",
    session_id: Optional[str] = None,
    store_conversations: bool = True,
    inject_memories: bool = True,
    max_memories: Optional[int] = None,
    budget: str = "mid",
    verbose: bool = False,
) -> HindsightAnthropic:
    """Wrap an Anthropic client with Hindsight memory integration.

    This creates a wrapped client that automatically injects memories
    and stores conversations when making message calls.

    Args:
        client: The Anthropic client instance to wrap
        bank_id: Memory bank ID for memory operations. For multi-user support,
            use different bank_ids per user (e.g., f"user-{user_id}")
        hindsight_api_url: URL of the Hindsight API server
        session_id: Session identifier for conversation grouping
        store_conversations: Whether to store conversations
        inject_memories: Whether to inject relevant memories
        max_memories: Maximum number of memories to inject (None = no limit)
        budget: Budget level for memory recall (low, mid, high)
        verbose: Enable verbose logging

    Returns:
        Wrapped Anthropic client with memory integration

    Example:
        >>> from anthropic import Anthropic
        >>> from hindsight_litellm import wrap_anthropic
        >>>
        >>> client = Anthropic()
        >>> wrapped = wrap_anthropic(
        ...     client,
        ...     bank_id=f"user-{user_id}",  # Multi-user support via separate banks
        ... )
        >>>
        >>> response = wrapped.messages.create(
        ...     model="claude-3-5-sonnet-20241022",
        ...     max_tokens=1024,
        ...     messages=[{"role": "user", "content": "What do you know about me?"}]
        ... )
    """
    return HindsightAnthropic(
        client=client,
        bank_id=bank_id,
        hindsight_api_url=hindsight_api_url,
        session_id=session_id,
        store_conversations=store_conversations,
        inject_memories=inject_memories,
        max_memories=max_memories,
        budget=budget,
        verbose=verbose,
    )
