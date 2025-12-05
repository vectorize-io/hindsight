"""Hindsight-LiteLLM: Universal LLM memory integration via LiteLLM.

This package provides automatic memory integration for any LLM provider
supported by LiteLLM (100+ providers including OpenAI, Anthropic, Groq,
Azure, AWS Bedrock, Google Vertex AI, and more).

Features:
- Automatic memory injection before LLM calls
- Automatic conversation storage after LLM calls
- Works with any LiteLLM-supported provider
- Zero code changes to existing LiteLLM usage
- Multi-user support via entity_id
- Session management for conversation threading
- Direct recall API for manual memory queries
- Native client wrappers for OpenAI and Anthropic

Basic usage:
    >>> from hindsight_litellm import configure, enable
    >>>
    >>> # Configure Hindsight integration
    >>> configure(
    ...     hindsight_api_url="http://localhost:8888",
    ...     bank_id="my-agent",
    ...     entity_id="user-123",  # Multi-user support
    ...     store_conversations=True,
    ...     inject_memories=True,
    ... )
    >>>
    >>> # Enable memory integration
    >>> enable()
    >>>
    >>> # Now use LiteLLM as normal - memory integration is automatic
    >>> import litellm
    >>> response = litellm.completion(
    ...     model="gpt-4",
    ...     messages=[{"role": "user", "content": "What did we discuss about AI?"}]
    ... )

Direct recall API:
    >>> from hindsight_litellm import configure, recall
    >>> configure(bank_id="my-agent", hindsight_api_url="http://localhost:8888")
    >>>
    >>> # Query memories directly
    >>> memories = recall("what projects am I working on?", limit=5)
    >>> for m in memories:
    ...     print(f"- [{m.fact_type}] {m.text}")

Native client wrappers:
    >>> from openai import OpenAI
    >>> from hindsight_litellm import wrap_openai
    >>>
    >>> client = OpenAI()
    >>> wrapped = wrap_openai(client, bank_id="my-agent", entity_id="user-123")
    >>>
    >>> response = wrapped.chat.completions.create(
    ...     model="gpt-4",
    ...     messages=[{"role": "user", "content": "Hello!"}]
    ... )

Session management:
    >>> from hindsight_litellm import configure, new_session, set_session
    >>> configure(bank_id="my-agent")
    >>>
    >>> session_id = new_session()  # Start fresh conversation
    >>> set_session("previous-id")  # Resume previous conversation

Works with any LiteLLM-supported provider:
    >>> # OpenAI
    >>> litellm.completion(model="gpt-4", messages=[...])
    >>>
    >>> # Anthropic
    >>> litellm.completion(model="claude-3-opus-20240229", messages=[...])
    >>>
    >>> # Groq
    >>> litellm.completion(model="groq/llama-3.1-70b-versatile", messages=[...])
    >>>
    >>> # Azure OpenAI
    >>> litellm.completion(model="azure/gpt-4", messages=[...])
    >>>
    >>> # AWS Bedrock
    >>> litellm.completion(model="bedrock/anthropic.claude-3", messages=[...])
    >>>
    >>> # Google Vertex AI
    >>> litellm.completion(model="vertex_ai/gemini-pro", messages=[...])

Context manager usage:
    >>> from hindsight_litellm import hindsight_memory
    >>>
    >>> with hindsight_memory(bank_id="my-agent", entity_id="user-123"):
    ...     response = litellm.completion(model="gpt-4", messages=[...])
    >>> # Memory integration automatically disabled after context

Configuration options:
    - hindsight_api_url: URL of your Hindsight API server
    - bank_id: Memory bank ID for memory operations (required)
    - api_key: Optional API key for Hindsight authentication
    - entity_id: User identifier for multi-user memory isolation
    - session_id: Session identifier for conversation grouping
    - store_conversations: Whether to store conversations (default: True)
    - inject_memories: Whether to inject relevant memories (default: True)
    - injection_mode: How to inject memories (system_message or prepend_user)
    - max_memories: Maximum number of memories to inject (default: 10)
    - recall_budget: Budget for memory recall (low, mid, high)
    - excluded_models: List of model patterns to exclude from interception
    - verbose: Enable verbose logging
"""

from contextlib import contextmanager
from typing import Optional, List

import litellm

from .config import (
    configure,
    get_config,
    is_configured,
    reset_config,
    new_session,
    set_session,
    get_session,
    set_entity,
    get_entity,
    HindsightConfig,
    MemoryInjectionMode,
)
from .callbacks import (
    HindsightCallback,
    get_callback,
    cleanup_callback,
)
from .wrappers import (
    recall,
    arecall,
    RecallResult,
    wrap_openai,
    wrap_anthropic,
    HindsightOpenAI,
    HindsightAnthropic,
)


__version__ = "0.1.0"

# Track whether we've registered with LiteLLM
_enabled = False

# Store original functions for restoration
_original_completion = None
_original_acompletion = None


def _inject_memories(messages: List[dict]) -> List[dict]:
    """Inject memories into messages list.

    Returns the modified messages list with memories injected into the system message.
    """
    import logging
    import requests

    if not is_configured():
        return messages

    config = get_config()
    if not config or not config.enabled or not config.inject_memories:
        return messages

    if not messages:
        return messages

    # Extract user query from last user message
    user_query = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                user_query = content
                break

    if not user_query:
        return messages

    try:
        # Build scoped bank_id
        scoped_bank_id = config.bank_id
        if config.entity_id:
            scoped_bank_id = f"{config.bank_id}:{config.entity_id}"

        # Build recall request
        url = f"{config.hindsight_api_url}/v1/default/banks/{scoped_bank_id}/memories/recall"
        request_data = {
            "query": user_query,
            "budget": config.recall_budget or "mid",
            "max_tokens": config.max_memory_tokens or 2000,
        }
        if config.fact_types:
            request_data["types"] = config.fact_types

        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"

        response = requests.post(url, json=request_data, headers=headers, timeout=30)
        response.raise_for_status()
        response_data = response.json()
        results = response_data.get("results", [])

        if not results:
            return messages

        # Format memories
        memory_lines = []
        for i, result in enumerate(results[:config.max_memories], 1):
            text = result.get("text", "")
            fact_type = result.get("type", result.get("fact_type", "world"))
            if text:
                type_label = fact_type.upper() if fact_type else "MEMORY"
                memory_lines.append(f"{i}. [{type_label}] {text}")

        if not memory_lines:
            return messages

        memory_context = (
            "# Relevant Memories\n"
            "The following information from memory may be relevant:\n\n"
            + "\n".join(memory_lines)
        )

        # Inject into messages
        updated_messages = list(messages)

        # Find existing system message or create new one
        found_system = False
        for i, msg in enumerate(updated_messages):
            if msg.get("role") == "system":
                existing_content = msg.get("content", "")
                updated_messages[i] = {
                    **msg,
                    "content": f"{existing_content}\n\n{memory_context}"
                }
                found_system = True
                break

        if not found_system:
            updated_messages.insert(0, {
                "role": "system",
                "content": memory_context
            })

        if config.verbose:
            logger = logging.getLogger("hindsight_litellm")
            logger.info(f"Injected {len(results)} memories into prompt")

        return updated_messages

    except Exception as e:
        if config.verbose:
            logging.getLogger("hindsight_litellm").warning(f"Failed to inject memories: {e}")
        return messages


def _wrapped_completion(*args, **kwargs):
    """Wrapper for litellm.completion that injects memories before the call."""
    # Inject memories into messages
    if "messages" in kwargs:
        kwargs["messages"] = _inject_memories(kwargs["messages"])
    elif args and len(args) > 1:
        # messages might be second positional arg after model
        args = list(args)
        if isinstance(args[1], list):
            args[1] = _inject_memories(args[1])
        args = tuple(args)

    # Call original
    return _original_completion(*args, **kwargs)


async def _wrapped_acompletion(*args, **kwargs):
    """Wrapper for litellm.acompletion that injects memories before the call."""
    # Inject memories into messages
    if "messages" in kwargs:
        kwargs["messages"] = _inject_memories(kwargs["messages"])
    elif args and len(args) > 1:
        args = list(args)
        if isinstance(args[1], list):
            args[1] = _inject_memories(args[1])
        args = tuple(args)

    # Call original
    return await _original_acompletion(*args, **kwargs)


def enable() -> None:
    """Enable Hindsight memory integration with LiteLLM.

    This monkeypatches LiteLLM functions to:
    1. Inject relevant memories into prompts before LLM calls
    2. Store conversations to Hindsight after successful LLM calls

    Must be called after configure() to take effect.

    Example:
        >>> from hindsight_litellm import configure, enable
        >>> configure(bank_id="my-agent", hindsight_api_url="http://localhost:8888")
        >>> enable()
        >>>
        >>> # Now all LiteLLM calls will have memory integration
        >>> import litellm
        >>> response = litellm.completion(model="gpt-4", messages=[...])
    """
    global _enabled, _original_completion, _original_acompletion

    if _enabled:
        return  # Already enabled

    if not is_configured():
        raise RuntimeError(
            "Hindsight not configured. Call configure() before enable()."
        )

    # Store original functions and monkeypatch for memory injection
    _original_completion = litellm.completion
    _original_acompletion = litellm.acompletion
    litellm.completion = _wrapped_completion
    litellm.acompletion = _wrapped_acompletion

    # Get or create the callback instance for storing conversations
    callback = get_callback()

    # Register callback using litellm.callbacks for conversation storage
    if callback not in litellm.callbacks:
        litellm.callbacks.append(callback)

    _enabled = True

    config = get_config()
    if config and config.verbose:
        print(f"Hindsight memory enabled for bank: {config.bank_id}")


def disable() -> None:
    """Disable Hindsight memory integration with LiteLLM.

    This restores the original LiteLLM functions and removes callbacks,
    stopping memory injection and conversation storage.

    Example:
        >>> from hindsight_litellm import disable
        >>> disable()  # Stop memory integration
    """
    global _enabled, _original_completion, _original_acompletion

    if not _enabled:
        return  # Already disabled

    # Restore original functions
    if _original_completion is not None:
        litellm.completion = _original_completion
        _original_completion = None
    if _original_acompletion is not None:
        litellm.acompletion = _original_acompletion
        _original_acompletion = None

    # Remove callback from litellm.callbacks
    callback = get_callback()
    if callback in litellm.callbacks:
        litellm.callbacks.remove(callback)

    _enabled = False

    config = get_config()
    if config and config.verbose:
        print("Hindsight memory disabled")


def is_enabled() -> bool:
    """Check if Hindsight memory integration is currently enabled.

    Returns:
        True if enable() has been called and not subsequently disabled
    """
    return _enabled


def cleanup() -> None:
    """Clean up all Hindsight resources.

    This disables the integration and closes any open connections.
    Call this when shutting down your application.

    Example:
        >>> from hindsight_litellm import cleanup
        >>> cleanup()  # Clean up when done
    """
    disable()
    cleanup_callback()
    reset_config()


# =============================================================================
# Convenience wrappers - use hindsight_litellm.completion() directly
# =============================================================================

def completion(*args, **kwargs):
    """Call LiteLLM completion with Hindsight memory integration.

    This is a convenience wrapper that delegates to litellm.completion().
    Memory injection and storage happen automatically if configured and enabled.

    Args:
        *args: Positional arguments passed to litellm.completion()
        **kwargs: Keyword arguments passed to litellm.completion()

    Returns:
        LiteLLM ModelResponse object

    Example:
        >>> import hindsight_litellm
        >>>
        >>> hindsight_litellm.configure(
        ...     hindsight_api_url="http://localhost:8888",
        ...     bank_id="my-agent",
        ... )
        >>> hindsight_litellm.enable()
        >>>
        >>> # Use directly - no need to import litellm separately
        >>> response = hindsight_litellm.completion(
        ...     model="gpt-4o-mini",
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
    """
    return litellm.completion(*args, **kwargs)


async def acompletion(*args, **kwargs):
    """Call LiteLLM async completion with Hindsight memory integration.

    This is a convenience wrapper that delegates to litellm.acompletion().
    Memory injection and storage happen automatically if configured and enabled.

    Args:
        *args: Positional arguments passed to litellm.acompletion()
        **kwargs: Keyword arguments passed to litellm.acompletion()

    Returns:
        LiteLLM ModelResponse object

    Example:
        >>> import hindsight_litellm
        >>> import asyncio
        >>>
        >>> hindsight_litellm.configure(
        ...     hindsight_api_url="http://localhost:8888",
        ...     bank_id="my-agent",
        ... )
        >>> hindsight_litellm.enable()
        >>>
        >>> async def main():
        ...     response = await hindsight_litellm.acompletion(
        ...         model="gpt-4o-mini",
        ...         messages=[{"role": "user", "content": "Hello!"}]
        ...     )
        ...     return response
        >>>
        >>> asyncio.run(main())
    """
    return await litellm.acompletion(*args, **kwargs)


@contextmanager
def hindsight_memory(
    hindsight_api_url: str = "http://localhost:8888",
    bank_id: Optional[str] = None,
    api_key: Optional[str] = None,
    entity_id: Optional[str] = None,
    session_id: Optional[str] = None,
    store_conversations: bool = True,
    inject_memories: bool = True,
    injection_mode: MemoryInjectionMode = MemoryInjectionMode.SYSTEM_MESSAGE,
    max_memories: int = 10,
    max_memory_tokens: int = 2000,
    recall_budget: str = "mid",
    fact_types: Optional[List[str]] = None,
    document_id: Optional[str] = None,
    excluded_models: Optional[List[str]] = None,
    verbose: bool = False,
):
    """Context manager for temporary Hindsight memory integration.

    Use this to enable memory integration for a specific block of code,
    automatically cleaning up afterwards.

    Args:
        hindsight_api_url: URL of the Hindsight API server
        bank_id: Memory bank ID for memory operations (required)
        api_key: Optional API key for Hindsight authentication
        entity_id: User identifier for multi-user memory isolation
        session_id: Session identifier for conversation grouping
        store_conversations: Whether to store conversations
        inject_memories: Whether to inject relevant memories
        injection_mode: How to inject memories
        max_memories: Maximum number of memories to inject
        max_memory_tokens: Maximum tokens for memory context
        recall_budget: Budget for memory recall (low, mid, high)
        fact_types: List of fact types to filter (world, agent, opinion, observation)
        document_id: Optional document ID for grouping conversations
        excluded_models: List of model patterns to exclude
        verbose: Enable verbose logging

    Example:
        >>> from hindsight_litellm import hindsight_memory
        >>> import litellm
        >>>
        >>> with hindsight_memory(bank_id="my-agent", entity_id="user-123"):
        ...     response = litellm.completion(model="gpt-4", messages=[...])
        >>> # Memory integration automatically disabled after context
    """
    # Save previous state
    was_enabled = is_enabled()
    previous_config = get_config()

    try:
        # Configure and enable
        configure(
            hindsight_api_url=hindsight_api_url,
            bank_id=bank_id,
            api_key=api_key,
            entity_id=entity_id,
            session_id=session_id,
            store_conversations=store_conversations,
            inject_memories=inject_memories,
            injection_mode=injection_mode,
            max_memories=max_memories,
            max_memory_tokens=max_memory_tokens,
            recall_budget=recall_budget,
            fact_types=fact_types,
            document_id=document_id,
            excluded_models=excluded_models,
            verbose=verbose,
        )
        enable()
        yield
    finally:
        # Restore previous state
        disable()
        if previous_config:
            configure(
                hindsight_api_url=previous_config.hindsight_api_url,
                bank_id=previous_config.bank_id,
                api_key=previous_config.api_key,
                entity_id=previous_config.entity_id,
                session_id=previous_config.session_id,
                store_conversations=previous_config.store_conversations,
                inject_memories=previous_config.inject_memories,
                injection_mode=previous_config.injection_mode,
                max_memories=previous_config.max_memories,
                max_memory_tokens=previous_config.max_memory_tokens,
                recall_budget=previous_config.recall_budget,
                fact_types=previous_config.fact_types,
                document_id=previous_config.document_id,
                excluded_models=previous_config.excluded_models,
                verbose=previous_config.verbose,
            )
            if was_enabled:
                enable()
        else:
            reset_config()


__all__ = [
    # Main API
    "configure",
    "enable",
    "disable",
    "is_enabled",
    "cleanup",
    "hindsight_memory",
    # LLM completion wrappers (convenience)
    "completion",
    "acompletion",
    # Session/Entity management
    "new_session",
    "set_session",
    "get_session",
    "set_entity",
    "get_entity",
    # Direct recall API
    "recall",
    "arecall",
    "RecallResult",
    # Native client wrappers
    "wrap_openai",
    "wrap_anthropic",
    "HindsightOpenAI",
    "HindsightAnthropic",
    # Configuration
    "get_config",
    "is_configured",
    "reset_config",
    "HindsightConfig",
    "MemoryInjectionMode",
    # Callback (for advanced usage)
    "HindsightCallback",
    "get_callback",
    "cleanup_callback",
]
