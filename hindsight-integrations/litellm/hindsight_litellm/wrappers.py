"""Native client wrappers for Hindsight memory integration.

This module provides wrappers for native LLM client SDKs (OpenAI, Anthropic)
that automatically integrate with Hindsight for memory injection and storage.

This is an alternative to the LiteLLM callback approach, providing direct
integration with native client libraries.
"""

import logging
import os
import threading
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

from .config import (
    DEFAULT_BANK_ID,
    DEFAULT_HINDSIGHT_API_URL,
    HINDSIGHT_API_KEY_ENV,
    HindsightCallSettings,
    _merge_call_settings as _merge_settings,
    get_config,
    get_defaults,
)

# Background thread support for async retain
_retain_errors: List[Exception] = []
_retain_errors_lock = threading.Lock()

logger = logging.getLogger(__name__)


def _get_client(api_url: str, api_key: Optional[str] = None):
    """Create a fresh Hindsight client for the given URL.

    Note: We create a fresh client each time because the hindsight_client
    uses aiohttp internally, and reusing clients across different sync
    calls causes asyncio context issues.
    """
    from hindsight_client import Hindsight

    return Hindsight(base_url=api_url, api_key=api_key, timeout=30.0)


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
        fact_types: Filter by fact types (world, experience, observation)
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
        client = _get_client(api_url, config.api_key if config else None)

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
                if hasattr(r, "text"):
                    # Object with attributes
                    fact_type = getattr(r, "type", None) or getattr(
                        r, "fact_type", "unknown"
                    )
                    recall_results.append(
                        RecallResult(
                            text=r.text,
                            fact_type=fact_type,
                            weight=getattr(r, "weight", 0.0),
                            metadata=getattr(r, "metadata", None),
                        )
                    )
                elif isinstance(r, dict):
                    # Dict from API response - API returns 'type' not 'fact_type'
                    fact_type = r.get("type") or r.get("fact_type", "unknown")
                    recall_results.append(
                        RecallResult(
                            text=r.get("text", str(r)),
                            fact_type=fact_type,
                            weight=r.get("weight", 0.0),
                            metadata=r.get("metadata"),
                        )
                    )

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
        ),
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
        client = _get_client(api_url, config.api_key if config else None)

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
        text = result.text if hasattr(result, "text") else str(result)
        based_on = getattr(result, "based_on", None)

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
        ),
    )


@dataclass
class RetainDebugInfo:
    """Debug information from a retain operation."""

    content: str
    bank_id: str
    context: Optional[str]
    document_id: Optional[str]
    tags: Optional[List[str]]
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
    tags: Optional[List[str]],
    metadata: Optional[Dict[str, str]],
    verbose: bool,
    api_key: Optional[str] = None,
) -> RetainResult:
    """Internal synchronous retain implementation."""
    client = None
    try:
        # Create fresh client for this operation
        client = _get_client(api_url, api_key)

        # Build retain kwargs
        retain_kwargs = {
            "bank_id": target_bank_id,
            "content": content,
            "context": context,
            "document_id": target_document_id,
            "metadata": metadata,
        }
        if tags:
            retain_kwargs["tags"] = tags

        # Call retain API
        result = client.retain(**retain_kwargs)

        # Check success
        success = getattr(result, "success", True)
        items_count = getattr(result, "items_count", 1)

        # Include debug info if verbose
        debug_info = None
        if verbose:
            logger.info(f"Stored content to Hindsight bank: {target_bank_id}")
            debug_info = RetainDebugInfo(
                content=content,
                bank_id=target_bank_id,
                context=context,
                document_id=target_document_id,
                tags=tags,
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
    tags: Optional[List[str]],
    metadata: Optional[Dict[str, str]],
    verbose: bool,
    api_key: Optional[str] = None,
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
            tags=tags,
            metadata=metadata,
            verbose=verbose,
            api_key=api_key,
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
    tags: Optional[List[str]] = None,
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
        tags: Tags for visibility scoping (e.g., ["user:alice", "session:123"])
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
    target_tags = tags or (defaults.tags if defaults else None)
    verbose = config.verbose if config else False

    if not api_url or not target_bank_id:
        raise RuntimeError(
            "Hindsight not configured. Call configure() or provide bank_id and hindsight_api_url."
        )

    api_key = config.api_key if config else None

    if sync:
        # Synchronous mode - block and return result
        return _retain_sync(
            content=content,
            api_url=api_url,
            target_bank_id=target_bank_id,
            context=context,
            target_document_id=target_document_id,
            tags=target_tags,
            metadata=metadata,
            verbose=verbose,
            api_key=api_key,
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
                target_tags,
                metadata,
                verbose,
                api_key,
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
    tags: Optional[List[str]] = None,
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
            tags=tags,
            metadata=metadata,
            hindsight_api_url=hindsight_api_url,
        ),
    )


class _StreamWrapper:
    """Wrapper for OpenAI stream that collects content and stores conversation when done."""

    def __init__(
        self,
        stream: Any,
        user_query: str,
        model: str,
        wrapper: "HindsightOpenAI",
        settings: HindsightCallSettings,
    ):
        self._stream = stream
        self._user_query = user_query
        self._model = model
        self._wrapper = wrapper
        self._settings = settings
        self._collected_content: List[str] = []
        self._finished = False

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._stream)
            # Collect content from the chunk
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    self._collected_content.append(delta.content)
            return chunk
        except StopIteration:
            # Stream exhausted - store conversation if we collected content
            self._store_if_needed()
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._store_if_needed()
        if hasattr(self._stream, "__exit__"):
            return self._stream.__exit__(exc_type, exc_val, exc_tb)

    def _store_if_needed(self):
        """Store the collected conversation if not already done."""
        if self._finished or not self._settings.store_conversations:
            return

        self._finished = True
        if self._collected_content:
            assistant_output = "".join(self._collected_content)
            if assistant_output:
                try:
                    self._wrapper._store_conversation(
                        self._user_query, assistant_output, self._model, self._settings
                    )
                except Exception as e:
                    if self._settings.verbose:
                        logger.warning(f"Failed to store streamed conversation: {e}")

    def close(self):
        """Close the underlying stream if it has a close method."""
        self._store_if_needed()
        if hasattr(self._stream, "close"):
            self._stream.close()

    def __getattr__(self, name: str) -> Any:
        """Proxy other attributes to the underlying stream."""
        return getattr(self._stream, name)


class _AnthropicStreamWrapper:
    """Wrapper for Anthropic stream that collects content and stores conversation when done."""

    def __init__(
        self,
        stream: Any,
        user_query: str,
        model: str,
        wrapper: "HindsightAnthropic",
        settings: HindsightCallSettings,
    ):
        self._stream = stream
        self._user_query = user_query
        self._model = model
        self._wrapper = wrapper
        self._settings = settings
        self._collected_content: List[str] = []
        self._finished = False

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._stream)
            # Collect content from the chunk
            if hasattr(chunk, "type") and chunk.type == "content_block_delta":
                if hasattr(chunk, "delta") and hasattr(chunk.delta, "text"):
                    self._collected_content.append(chunk.delta.text)
            return chunk
        except StopIteration:
            # Stream exhausted - store conversation if we collected content
            self._store_if_needed()
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._store_if_needed()
        if hasattr(self._stream, "__exit__"):
            return self._stream.__exit__(exc_type, exc_val, exc_tb)

    def _store_if_needed(self):
        """Store the collected conversation if not already done."""
        if self._finished or not self._settings.store_conversations:
            return

        self._finished = True
        if self._collected_content:
            assistant_output = "".join(self._collected_content)
            if assistant_output:
                try:
                    self._wrapper._store_conversation(
                        self._user_query, assistant_output, self._model, self._settings
                    )
                except Exception as e:
                    if self._settings.verbose:
                        logger.warning(f"Failed to store streamed conversation: {e}")

    def close(self):
        """Close the underlying stream if it has a close method."""
        self._store_if_needed()
        if hasattr(self._stream, "close"):
            self._stream.close()

    def __getattr__(self, name: str) -> Any:
        """Proxy other attributes to the underlying stream."""
        return getattr(self._stream, name)


class HindsightOpenAI:
    """Wrapper for OpenAI client with Hindsight memory integration.

    This wraps the native OpenAI client to automatically inject memories
    and store conversations. All settings can be overridden per-call using
    hindsight_* kwargs.

    Example:
        >>> from openai import OpenAI
        >>> from hindsight_litellm import wrap_openai
        >>>
        >>> client = OpenAI()
        >>> wrapped = wrap_openai(client, bank_id="my-agent")
        >>>
        >>> # Use default settings
        >>> response = wrapped.chat.completions.create(
        ...     model="gpt-4",
        ...     messages=[{"role": "user", "content": "What do you know about me?"}]
        ... )
        >>>
        >>> # Override settings per-call
        >>> response = wrapped.chat.completions.create(
        ...     model="gpt-4",
        ...     messages=[{"role": "user", "content": "Hello"}],
        ...     hindsight_bank_id="other-user",  # Different bank for this call
        ...     hindsight_budget="high",          # Higher recall budget
        ...     hindsight_inject_memories=False,  # Skip memory injection
        ... )
    """

    def __init__(
        self,
        client: Any,
        hindsight_api_url: str,
        api_key: Optional[str] = None,
        default_settings: Optional[HindsightCallSettings] = None,
        **setting_kwargs,
    ):
        """Initialize the wrapped OpenAI client.

        Args:
            client: The OpenAI client instance to wrap
            hindsight_api_url: URL of the Hindsight API server (required, connection-level)
            api_key: API key for Hindsight authentication (connection-level)
            default_settings: Default settings for all calls (alternative to kwargs)
            **setting_kwargs: Default values for any HindsightCallSettings field
                (e.g., bank_id="my-bank", budget="high", verbose=True)
        """
        self._client = client
        self._api_url = hindsight_api_url
        self._api_key = api_key
        self._hindsight_client = None

        # Build default settings from kwargs or use provided settings
        if default_settings is not None:
            self._default_settings = default_settings
        else:
            # Filter kwargs to only valid HindsightCallSettings fields
            valid_fields = {f.name for f in fields(HindsightCallSettings)}
            settings_kwargs = {
                k: v for k, v in setting_kwargs.items() if k in valid_fields
            }
            self._default_settings = HindsightCallSettings(**settings_kwargs)

        # Create wrapped chat.completions interface
        self.chat = _WrappedChat(self)

    def _get_hindsight_client(self):
        """Get or create the Hindsight client."""
        if self._hindsight_client is None:
            from hindsight_client import Hindsight

            self._hindsight_client = Hindsight(
                base_url=self._api_url,
                api_key=self._api_key,
                timeout=30.0,
            )
        return self._hindsight_client

    def _recall_memories(self, query: str, settings: HindsightCallSettings) -> str:
        """Recall and format memories for injection."""
        if not settings.inject_memories:
            return ""

        if not settings.bank_id:
            if settings.verbose:
                logger.warning("No bank_id configured, skipping memory recall")
            return ""

        try:
            client = self._get_hindsight_client()

            recall_kwargs = {
                "bank_id": settings.bank_id,
                "query": query,
                "budget": settings.budget,
                "max_tokens": settings.max_memory_tokens,
                "trace": settings.trace,
                "include_entities": settings.include_entities,
            }
            if settings.fact_types:
                recall_kwargs["types"] = settings.fact_types
            if settings.recall_tags:
                recall_kwargs["tags"] = settings.recall_tags
                recall_kwargs["tags_match"] = settings.recall_tags_match

            results = client.recall(**recall_kwargs)

            if not results:
                return ""

            results_to_use = (
                results[: settings.max_memories] if settings.max_memories else results
            )
            memory_lines = []
            for i, r in enumerate(results_to_use, 1):
                text = r.text if hasattr(r, "text") else str(r)
                fact_type = getattr(r, "type", None) or getattr(
                    r, "fact_type", "memory"
                )

                # Build memory line with available context
                line_parts = [f"{i}. [{fact_type.upper()}]"]

                # Add temporal context if available
                occurred_start = getattr(r, "occurred_start", None)
                occurred_end = getattr(r, "occurred_end", None)
                mentioned_at = getattr(r, "mentioned_at", None)

                if occurred_start and occurred_end and occurred_start != occurred_end:
                    line_parts.append(f"(occurred: {occurred_start} to {occurred_end})")
                elif occurred_start:
                    line_parts.append(f"(occurred: {occurred_start})")
                elif mentioned_at:
                    line_parts.append(f"(mentioned: {mentioned_at})")

                # Add source context if available
                context = getattr(r, "context", None)
                if context:
                    line_parts.append(f"[source: {context}]")

                # Add the main text
                line_parts.append(text)

                # Add metadata if available
                metadata = getattr(r, "metadata", None)
                if metadata:
                    meta_str = ", ".join(f"{k}={v}" for k, v in metadata.items())
                    line_parts.append(f"({meta_str})")

                memory_lines.append(" ".join(line_parts))

            if not memory_lines:
                return ""

            current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            return (
                f"# Relevant Memories\n"
                f"Current date/time: {current_time}\n"
                f"The following information from memory may be relevant:\n\n"
                + "\n".join(memory_lines)
            )

        except Exception as e:
            if settings.verbose:
                logger.warning(f"Failed to recall memories: {e}")
            return ""

    def _reflect_memories(self, query: str, settings: HindsightCallSettings) -> str:
        """Use reflect API for disposition-aware memory retrieval."""
        if not settings.inject_memories:
            return ""

        if not settings.bank_id:
            if settings.verbose:
                logger.warning("No bank_id configured, skipping reflect")
            return ""

        try:
            client = self._get_hindsight_client()

            reflect_kwargs = {
                "bank_id": settings.bank_id,
                "query": query,
                "budget": settings.budget,
            }
            if settings.reflect_context:
                reflect_kwargs["context"] = settings.reflect_context
            if settings.reflect_response_schema:
                reflect_kwargs["response_schema"] = settings.reflect_response_schema
            if settings.recall_tags:
                reflect_kwargs["tags"] = settings.recall_tags
                reflect_kwargs["tags_match"] = settings.recall_tags_match

            result = client.reflect(**reflect_kwargs)

            if not result:
                return ""

            text = result.text if hasattr(result, "text") else str(result)
            if not text:
                return ""

            current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            return f"# Relevant Context from Memory\nCurrent date/time: {current_time}\n{text}"

        except Exception as e:
            if settings.verbose:
                logger.warning(f"Failed to reflect: {e}")
            return ""

    def _get_document_content(
        self, bank_id: str, document_id: str
    ) -> Optional[str]:
        """Fetch existing document content using the low-level API.

        Returns:
            The document's original_text, or None if not found.
        """
        import asyncio

        from hindsight_client_api.api import documents_api

        try:
            client = self._get_hindsight_client()
            docs_api = documents_api.DocumentsApi(client._api_client)

            async def _fetch():
                try:
                    doc = await docs_api.get_document(bank_id, document_id)
                    return doc.original_text if doc else None
                except Exception as e:
                    if "404" in str(e) or "Not Found" in str(e):
                        return None
                    raise

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            return loop.run_until_complete(_fetch())
        except Exception as e:
            logger.debug(f"Failed to fetch document: {e}")
            return None

    def _store_conversation(
        self,
        user_input: str,
        assistant_output: str,
        model: str,
        settings: HindsightCallSettings,
    ):
        """Store the conversation to Hindsight.

        If session_id (effective_document_id) is set, accumulates the full
        conversation in a single document for better context learning.
        """
        if not settings.store_conversations:
            return

        if not settings.bank_id:
            if settings.verbose:
                logger.warning("No bank_id configured, skipping conversation storage")
            return

        try:
            client = self._get_hindsight_client()
            new_exchange = f"USER: {user_input}\n\nASSISTANT: {assistant_output}"

            # If document_id is set, fetch existing content and append
            conversation_text = new_exchange
            if settings.effective_document_id:
                existing_content = self._get_document_content(
                    settings.bank_id, settings.effective_document_id
                )
                if existing_content:
                    conversation_text = f"{existing_content}\n\n{new_exchange}"
                    if settings.verbose:
                        logger.debug(
                            f"Appending to existing document: {settings.effective_document_id}"
                        )

            metadata = {
                "source": "openai-wrapper",
                "model": model,
            }

            retain_kwargs = {
                "bank_id": settings.bank_id,
                "content": conversation_text,
                "context": f"conversation:openai:{model}",
                "metadata": metadata,
            }
            if settings.effective_document_id:
                retain_kwargs["document_id"] = settings.effective_document_id
            if settings.tags:
                retain_kwargs["tags"] = settings.tags

            client.retain(**retain_kwargs)

            if settings.verbose:
                logger.info(
                    f"Stored conversation to Hindsight bank: {settings.bank_id}"
                )

        except Exception as e:
            if settings.verbose:
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
        """Create a chat completion with memory integration.

        Supports per-call overrides via hindsight_* kwargs:
        - hindsight_bank_id: Override memory bank
        - hindsight_budget: Override recall budget (low/mid/high)
        - hindsight_inject_memories: Override whether to inject memories
        - hindsight_store_conversations: Override whether to store
        - hindsight_query: Custom query for memory recall
        - hindsight_use_reflect: Use reflect API instead of recall
        - And all other HindsightCallSettings fields...
        """
        # Merge default settings with per-call overrides
        settings = _merge_settings(self._wrapper._default_settings, kwargs)

        # Remove hindsight_* kwargs before passing to OpenAI
        openai_kwargs = {
            k: v for k, v in kwargs.items() if not k.startswith("hindsight_")
        }

        messages = list(openai_kwargs.get("messages", []))
        model = openai_kwargs.get("model", "gpt-4")

        # Extract user query (use custom query if provided, else extract from messages)
        user_query = settings.query
        if not user_query:
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    content = msg.get("content")
                    if isinstance(content, str):
                        user_query = content
                        break
                    elif isinstance(content, list):
                        # Handle structured content (e.g., vision messages)
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                user_query = item.get("text", "")
                                break
                        if user_query:
                            break

        # Inject memories
        if user_query and settings.inject_memories:
            # Use reflect or recall based on settings
            if settings.use_reflect:
                memory_context = self._wrapper._reflect_memories(user_query, settings)
            else:
                memory_context = self._wrapper._recall_memories(user_query, settings)

            if memory_context:
                if settings.verbose:
                    logger.debug(f"Injecting memories into prompt:\n{memory_context}")

                # Find system message and append, or prepend new one
                found_system = False
                for i, msg in enumerate(messages):
                    if msg.get("role") == "system":
                        messages[i] = {
                            **msg,
                            "content": f"{msg.get('content', '')}\n\n{memory_context}",
                        }
                        found_system = True
                        break

                if not found_system:
                    messages.insert(0, {"role": "system", "content": memory_context})

                openai_kwargs["messages"] = messages

        # Make the actual API call
        response = self._wrapper._client.chat.completions.create(**openai_kwargs)

        # Handle streaming vs non-streaming responses
        is_streaming = openai_kwargs.get("stream", False)
        if is_streaming:
            # Wrap the stream to collect content and store conversation when done
            if user_query and settings.store_conversations:
                return _StreamWrapper(
                    stream=response,
                    user_query=user_query,
                    model=model,
                    wrapper=self._wrapper,
                    settings=settings,
                )
            else:
                return response
        else:
            # Non-streaming: store conversation immediately
            if user_query and settings.store_conversations:
                if response.choices and response.choices[0].message:
                    assistant_output = response.choices[0].message.content or ""
                    if assistant_output:
                        self._wrapper._store_conversation(
                            user_query, assistant_output, model, settings
                        )
            return response


class HindsightAnthropic:
    """Wrapper for Anthropic client with Hindsight memory integration.

    This wraps the native Anthropic client to automatically inject memories
    and store conversations. All settings can be overridden per-call using
    hindsight_* kwargs.

    Example:
        >>> from anthropic import Anthropic
        >>> from hindsight_litellm import wrap_anthropic
        >>>
        >>> client = Anthropic()
        >>> wrapped = wrap_anthropic(client, bank_id="my-agent")
        >>>
        >>> # Use default settings
        >>> response = wrapped.messages.create(
        ...     model="claude-sonnet-4-20250514",
        ...     max_tokens=1024,
        ...     messages=[{"role": "user", "content": "What do you know about me?"}]
        ... )
        >>>
        >>> # Override settings per-call
        >>> response = wrapped.messages.create(
        ...     model="claude-sonnet-4-20250514",
        ...     max_tokens=1024,
        ...     messages=[{"role": "user", "content": "Hello"}],
        ...     hindsight_bank_id="other-user",  # Different bank
        ...     hindsight_budget="high",          # More memories
        ... )
    """

    def __init__(
        self,
        client: Any,
        hindsight_api_url: str,
        api_key: Optional[str] = None,
        default_settings: Optional[HindsightCallSettings] = None,
        **setting_kwargs,
    ):
        """Initialize the wrapped Anthropic client.

        Args:
            client: The Anthropic client instance to wrap
            hindsight_api_url: URL of the Hindsight API server (required, connection-level)
            api_key: API key for Hindsight authentication (connection-level)
            default_settings: Default settings for all calls (alternative to kwargs)
            **setting_kwargs: Default values for any HindsightCallSettings field
                (e.g., bank_id="my-bank", budget="high", verbose=True)
        """
        self._client = client
        self._api_url = hindsight_api_url
        self._api_key = api_key
        self._hindsight_client = None

        # Build default settings from kwargs or use provided settings
        if default_settings is not None:
            self._default_settings = default_settings
        else:
            # Filter kwargs to only valid HindsightCallSettings fields
            valid_fields = {f.name for f in fields(HindsightCallSettings)}
            settings_kwargs = {
                k: v for k, v in setting_kwargs.items() if k in valid_fields
            }
            self._default_settings = HindsightCallSettings(**settings_kwargs)

        # Create wrapped messages interface
        self.messages = _WrappedAnthropicMessages(self)

    def _get_hindsight_client(self):
        """Get or create the Hindsight client."""
        if self._hindsight_client is None:
            from hindsight_client import Hindsight

            self._hindsight_client = Hindsight(
                base_url=self._api_url,
                api_key=self._api_key,
                timeout=30.0,
            )
        return self._hindsight_client

    def _recall_memories(self, query: str, settings: HindsightCallSettings) -> str:
        """Recall and format memories for injection."""
        if not settings.inject_memories:
            return ""

        if not settings.bank_id:
            if settings.verbose:
                logger.warning("No bank_id configured, skipping memory recall")
            return ""

        try:
            client = self._get_hindsight_client()

            recall_kwargs = {
                "bank_id": settings.bank_id,
                "query": query,
                "budget": settings.budget,
                "max_tokens": settings.max_memory_tokens,
                "trace": settings.trace,
                "include_entities": settings.include_entities,
            }
            if settings.fact_types:
                recall_kwargs["types"] = settings.fact_types
            if settings.recall_tags:
                recall_kwargs["tags"] = settings.recall_tags
                recall_kwargs["tags_match"] = settings.recall_tags_match

            results = client.recall(**recall_kwargs)

            if not results:
                return ""

            results_to_use = (
                results[: settings.max_memories] if settings.max_memories else results
            )
            memory_lines = []
            for i, r in enumerate(results_to_use, 1):
                text = r.text if hasattr(r, "text") else str(r)
                fact_type = getattr(r, "type", None) or getattr(
                    r, "fact_type", "memory"
                )

                # Build memory line with available context
                line_parts = [f"{i}. [{fact_type.upper()}]"]

                # Add temporal context if available
                occurred_start = getattr(r, "occurred_start", None)
                occurred_end = getattr(r, "occurred_end", None)
                mentioned_at = getattr(r, "mentioned_at", None)

                if occurred_start and occurred_end and occurred_start != occurred_end:
                    line_parts.append(f"(occurred: {occurred_start} to {occurred_end})")
                elif occurred_start:
                    line_parts.append(f"(occurred: {occurred_start})")
                elif mentioned_at:
                    line_parts.append(f"(mentioned: {mentioned_at})")

                # Add source context if available
                context = getattr(r, "context", None)
                if context:
                    line_parts.append(f"[source: {context}]")

                # Add the main text
                line_parts.append(text)

                # Add metadata if available
                metadata = getattr(r, "metadata", None)
                if metadata:
                    meta_str = ", ".join(f"{k}={v}" for k, v in metadata.items())
                    line_parts.append(f"({meta_str})")

                memory_lines.append(" ".join(line_parts))

            if not memory_lines:
                return ""

            current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            return (
                f"# Relevant Memories\n"
                f"Current date/time: {current_time}\n"
                f"The following information from memory may be relevant:\n\n"
                + "\n".join(memory_lines)
            )

        except Exception as e:
            if settings.verbose:
                logger.warning(f"Failed to recall memories: {e}")
            return ""

    def _reflect_memories(self, query: str, settings: HindsightCallSettings) -> str:
        """Use reflect API for disposition-aware memory retrieval."""
        if not settings.inject_memories:
            return ""

        if not settings.bank_id:
            if settings.verbose:
                logger.warning("No bank_id configured, skipping reflect")
            return ""

        try:
            client = self._get_hindsight_client()

            reflect_kwargs = {
                "bank_id": settings.bank_id,
                "query": query,
                "budget": settings.budget,
            }
            if settings.reflect_context:
                reflect_kwargs["context"] = settings.reflect_context
            if settings.reflect_response_schema:
                reflect_kwargs["response_schema"] = settings.reflect_response_schema
            if settings.recall_tags:
                reflect_kwargs["tags"] = settings.recall_tags
                reflect_kwargs["tags_match"] = settings.recall_tags_match

            result = client.reflect(**reflect_kwargs)

            if not result:
                return ""

            text = result.text if hasattr(result, "text") else str(result)
            if not text:
                return ""

            current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            return f"# Relevant Context from Memory\nCurrent date/time: {current_time}\n{text}"

        except Exception as e:
            if settings.verbose:
                logger.warning(f"Failed to reflect: {e}")
            return ""

    def _get_document_content(
        self, bank_id: str, document_id: str
    ) -> Optional[str]:
        """Fetch existing document content using the low-level API.

        Returns:
            The document's original_text, or None if not found.
        """
        import asyncio

        from hindsight_client_api.api import documents_api

        try:
            client = self._get_hindsight_client()
            docs_api = documents_api.DocumentsApi(client._api_client)

            async def _fetch():
                try:
                    doc = await docs_api.get_document(bank_id, document_id)
                    return doc.original_text if doc else None
                except Exception as e:
                    if "404" in str(e) or "Not Found" in str(e):
                        return None
                    raise

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            return loop.run_until_complete(_fetch())
        except Exception as e:
            logger.debug(f"Failed to fetch document: {e}")
            return None

    def _store_conversation(
        self,
        user_input: str,
        assistant_output: str,
        model: str,
        settings: HindsightCallSettings,
    ):
        """Store the conversation to Hindsight.

        If session_id (effective_document_id) is set, accumulates the full
        conversation in a single document for better context learning.
        """
        if not settings.store_conversations:
            return

        if not settings.bank_id:
            if settings.verbose:
                logger.warning("No bank_id configured, skipping conversation storage")
            return

        try:
            client = self._get_hindsight_client()
            new_exchange = f"USER: {user_input}\n\nASSISTANT: {assistant_output}"

            # If document_id is set, fetch existing content and append
            conversation_text = new_exchange
            if settings.effective_document_id:
                existing_content = self._get_document_content(
                    settings.bank_id, settings.effective_document_id
                )
                if existing_content:
                    conversation_text = f"{existing_content}\n\n{new_exchange}"
                    if settings.verbose:
                        logger.debug(
                            f"Appending to existing document: {settings.effective_document_id}"
                        )

            metadata = {
                "source": "anthropic-wrapper",
                "model": model,
            }

            retain_kwargs = {
                "bank_id": settings.bank_id,
                "content": conversation_text,
                "context": f"conversation:anthropic:{model}",
                "metadata": metadata,
            }
            if settings.effective_document_id:
                retain_kwargs["document_id"] = settings.effective_document_id
            if settings.tags:
                retain_kwargs["tags"] = settings.tags

            client.retain(**retain_kwargs)

            if settings.verbose:
                logger.info(
                    f"Stored conversation to Hindsight bank: {settings.bank_id}"
                )

        except Exception as e:
            if settings.verbose:
                logger.warning(f"Failed to store conversation: {e}")

    # Proxy other attributes to the underlying client
    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class _WrappedAnthropicMessages:
    """Wrapped messages interface for Anthropic client."""

    def __init__(self, wrapper: HindsightAnthropic):
        self._wrapper = wrapper

    def create(self, **kwargs) -> Any:
        """Create a message with memory integration.

        Supports per-call overrides via hindsight_* kwargs:
        - hindsight_bank_id: Override memory bank
        - hindsight_budget: Override recall budget (low/mid/high)
        - hindsight_inject_memories: Override whether to inject memories
        - hindsight_store_conversations: Override whether to store
        - hindsight_query: Custom query for memory recall
        - hindsight_use_reflect: Use reflect API instead of recall
        - And all other HindsightCallSettings fields...
        """
        # Merge default settings with per-call overrides
        settings = _merge_settings(self._wrapper._default_settings, kwargs)

        # Remove hindsight_* kwargs before passing to Anthropic
        anthropic_kwargs = {
            k: v for k, v in kwargs.items() if not k.startswith("hindsight_")
        }

        messages = list(anthropic_kwargs.get("messages", []))
        model = anthropic_kwargs.get("model", "claude-sonnet-4-20250514")
        system = anthropic_kwargs.get("system", "")

        # Extract user query (use custom query if provided, else extract from messages)
        user_query = settings.query
        if not user_query:
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
        if user_query and settings.inject_memories:
            # Use reflect or recall based on settings
            if settings.use_reflect:
                memory_context = self._wrapper._reflect_memories(user_query, settings)
            else:
                memory_context = self._wrapper._recall_memories(user_query, settings)

            if memory_context:
                if settings.verbose:
                    logger.debug(f"Injecting memories into prompt:\n{memory_context}")

                if system:
                    anthropic_kwargs["system"] = f"{system}\n\n{memory_context}"
                else:
                    anthropic_kwargs["system"] = memory_context

        # Make the actual API call
        response = self._wrapper._client.messages.create(**anthropic_kwargs)

        # Handle streaming vs non-streaming responses
        is_streaming = anthropic_kwargs.get("stream", False)
        if is_streaming:
            # Wrap the stream to collect content and store conversation when done
            if user_query and settings.store_conversations:
                return _AnthropicStreamWrapper(
                    stream=response,
                    user_query=user_query,
                    model=model,
                    wrapper=self._wrapper,
                    settings=settings,
                )
            else:
                return response
        else:
            # Non-streaming: store conversation immediately
            if user_query and settings.store_conversations:
                if response.content:
                    assistant_output = ""
                    for block in response.content:
                        if hasattr(block, "text"):
                            assistant_output += block.text
                    if assistant_output:
                        self._wrapper._store_conversation(
                            user_query, assistant_output, model, settings
                        )
            return response


def wrap_openai(
    client: Any,
    hindsight_api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    mission: Optional[str] = None,
    bank_name: Optional[str] = None,
    **settings_kwargs,
) -> HindsightOpenAI:
    """Wrap an OpenAI client with Hindsight memory integration.

    This creates a wrapped client that automatically injects memories
    and stores conversations when making chat completion calls.

    All settings can be overridden per-call using hindsight_* kwargs.

    Args:
        client: The OpenAI client instance to wrap
        hindsight_api_url: URL of the Hindsight API server
            (default: https://api.hindsight.vectorize.io)
        api_key: API key for Hindsight authentication. If not provided,
            reads from HINDSIGHT_API_KEY environment variable.
        mission: Instructions guiding what Hindsight should learn and remember
            (used for mental model generation). If provided, creates/updates the bank.
        bank_name: Optional display name for the bank.
        **settings_kwargs: Default values for any HindsightCallSettings field:
            - bank_id: Memory bank ID (default: "default")
            - document_id: Document ID for conversation grouping
            - session_id: Session identifier for metadata
            - store_conversations: Whether to store conversations (default: True)
            - inject_memories: Whether to inject memories (default: True)
            - budget: Recall budget level - low/mid/high (default: "mid")
            - fact_types: Filter by fact types (world/experience/observation)
            - max_memories: Max memories to inject (None = no limit)
            - max_memory_tokens: Max tokens for memory context (default: 4096)
            - use_reflect: Use reflect API instead of recall (default: False)
            - reflect_context: Context for reflect reasoning
            - reflect_response_schema: JSON Schema for structured reflect output
            - query: Custom query for memory recall (default: extract from message)
            - verbose: Enable verbose logging (default: False)

    Returns:
        Wrapped OpenAI client with memory integration

    Example:
        >>> from openai import OpenAI
        >>> from hindsight_litellm import wrap_openai
        >>>
        >>> # With mission for mental models
        >>> client = wrap_openai(
        ...     OpenAI(),
        ...     bank_id="my-agent",
        ...     mission="Remember user preferences and past interactions.",
        ... )
        >>>
        >>> # Use default settings
        >>> response = client.chat.completions.create(
        ...     model="gpt-4o-mini",
        ...     messages=[{"role": "user", "content": "What do you know about me?"}]
        ... )
        >>>
        >>> # Override per-call
        >>> response = client.chat.completions.create(
        ...     model="gpt-4o-mini",
        ...     messages=[{"role": "user", "content": "Hello"}],
        ...     hindsight_bank_id="other-user",  # Different bank
        ...     hindsight_budget="high",          # More memories
        ... )
    """
    # Apply connection-level defaults
    resolved_api_url = hindsight_api_url or DEFAULT_HINDSIGHT_API_URL
    resolved_api_key = api_key or os.environ.get(HINDSIGHT_API_KEY_ENV)

    # Apply default bank_id if not provided
    if "bank_id" not in settings_kwargs:
        settings_kwargs["bank_id"] = DEFAULT_BANK_ID

    # Create/update bank if mission or bank_name is provided
    if mission or bank_name:
        try:
            from hindsight_client import Hindsight

            hs_client = Hindsight(base_url=resolved_api_url, api_key=resolved_api_key)
            hs_client.create_bank(
                bank_id=settings_kwargs["bank_id"],
                name=bank_name,
                mission=mission,
            )
            hs_client.close()
        except Exception as e:
            if settings_kwargs.get("verbose"):
                logger.warning(f"Failed to create/update bank: {e}")

    return HindsightOpenAI(
        client=client,
        hindsight_api_url=resolved_api_url,
        api_key=resolved_api_key,
        **settings_kwargs,
    )


def wrap_anthropic(
    client: Any,
    hindsight_api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    mission: Optional[str] = None,
    bank_name: Optional[str] = None,
    **settings_kwargs,
) -> HindsightAnthropic:
    """Wrap an Anthropic client with Hindsight memory integration.

    This creates a wrapped client that automatically injects memories
    and stores conversations when making message calls.

    All settings can be overridden per-call using hindsight_* kwargs.

    Args:
        client: The Anthropic client instance to wrap
        hindsight_api_url: URL of the Hindsight API server
            (default: https://api.hindsight.vectorize.io)
        api_key: API key for Hindsight authentication. If not provided,
            reads from HINDSIGHT_API_KEY environment variable.
        mission: Instructions guiding what Hindsight should learn and remember
            (used for mental model generation). If provided, creates/updates the bank.
        bank_name: Optional display name for the bank.
        **settings_kwargs: Default values for any HindsightCallSettings field:
            - bank_id: Memory bank ID (default: "default")
            - document_id: Document ID for conversation grouping
            - session_id: Session identifier for metadata
            - store_conversations: Whether to store conversations (default: True)
            - inject_memories: Whether to inject memories (default: True)
            - budget: Recall budget level - low/mid/high (default: "mid")
            - fact_types: Filter by fact types (world/experience/observation)
            - max_memories: Max memories to inject (None = no limit)
            - max_memory_tokens: Max tokens for memory context (default: 4096)
            - use_reflect: Use reflect API instead of recall (default: False)
            - reflect_context: Context for reflect reasoning
            - reflect_response_schema: JSON Schema for structured reflect output
            - query: Custom query for memory recall (default: extract from message)
            - verbose: Enable verbose logging (default: False)

    Returns:
        Wrapped Anthropic client with memory integration

    Example:
        >>> from anthropic import Anthropic
        >>> from hindsight_litellm import wrap_anthropic
        >>>
        >>> # With mission for mental models
        >>> client = wrap_anthropic(
        ...     Anthropic(),
        ...     bank_id="my-agent",
        ...     mission="Remember user preferences and past interactions.",
        ... )
        >>>
        >>> # Use default settings
        >>> response = client.messages.create(
        ...     model="claude-sonnet-4-20250514",
        ...     max_tokens=1024,
        ...     messages=[{"role": "user", "content": "What do you know about me?"}]
        ... )
        >>>
        >>> # Override per-call
        >>> response = client.messages.create(
        ...     model="claude-sonnet-4-20250514",
        ...     max_tokens=1024,
        ...     messages=[{"role": "user", "content": "Hello"}],
        ...     hindsight_bank_id="other-user",  # Different bank
        ...     hindsight_budget="high",          # More memories
        ... )
    """
    # Apply connection-level defaults
    resolved_api_url = hindsight_api_url or DEFAULT_HINDSIGHT_API_URL
    resolved_api_key = api_key or os.environ.get(HINDSIGHT_API_KEY_ENV)

    # Apply default bank_id if not provided
    if "bank_id" not in settings_kwargs:
        settings_kwargs["bank_id"] = DEFAULT_BANK_ID

    # Create/update bank if mission or bank_name is provided
    if mission or bank_name:
        try:
            from hindsight_client import Hindsight

            hs_client = Hindsight(base_url=resolved_api_url, api_key=resolved_api_key)
            hs_client.create_bank(
                bank_id=settings_kwargs["bank_id"],
                name=bank_name,
                mission=mission,
            )
            hs_client.close()
        except Exception as e:
            if settings_kwargs.get("verbose"):
                logger.warning(f"Failed to create/update bank: {e}")

    return HindsightAnthropic(
        client=client,
        hindsight_api_url=resolved_api_url,
        api_key=resolved_api_key,
        **settings_kwargs,
    )
