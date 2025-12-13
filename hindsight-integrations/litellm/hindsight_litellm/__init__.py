"""Hindsight-LiteLLM: Universal LLM memory integration via LiteLLM.

This package provides automatic memory integration for any LLM provider
supported by LiteLLM (100+ providers including OpenAI, Anthropic, Groq,
Azure, AWS Bedrock, Google Vertex AI, and more).

Features:
- Automatic memory injection before LLM calls
- Automatic conversation storage after LLM calls
- Works with any LiteLLM-supported provider
- Zero code changes to existing LiteLLM usage
- Multi-user support via separate bank_ids
- Document grouping for conversation threading
- Direct recall API for manual memory queries
- Native client wrappers for OpenAI and Anthropic

Basic usage:
    >>> from hindsight_litellm import configure, enable
    >>>
    >>> # Configure Hindsight integration
    >>> configure(
    ...     hindsight_api_url="http://localhost:8888",
    ...     bank_id="user-123",  # Use separate bank_ids for multi-user support
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
    >>> memories = recall("what projects am I working on?")
    >>> for m in memories:
    ...     print(f"- [{m.fact_type}] {m.text}")

Native client wrappers:
    >>> from openai import OpenAI
    >>> from hindsight_litellm import wrap_openai
    >>>
    >>> client = OpenAI()
    >>> wrapped = wrap_openai(client, bank_id="user-123")
    >>>
    >>> response = wrapped.chat.completions.create(
    ...     model="gpt-4",
    ...     messages=[{"role": "user", "content": "Hello!"}]
    ... )

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
    >>> with hindsight_memory(bank_id="user-123"):
    ...     response = litellm.completion(model="gpt-4", messages=[...])
    >>> # Memory integration automatically disabled after context

Configuration options:
    - hindsight_api_url: URL of your Hindsight API server
    - bank_id: Memory bank ID for memory operations (required). For multi-user
        support, use different bank_ids per user (e.g., f"user-{user_id}")
    - api_key: Optional API key for Hindsight authentication
    - store_conversations: Whether to store conversations (default: True)
    - inject_memories: Whether to inject relevant memories (default: True)
    - injection_mode: How to inject memories (system_message or prepend_user)
    - max_memories: Maximum number of memories to inject (None = unlimited)
    - recall_budget: Budget for memory recall (low, mid, high)
    - excluded_models: List of model patterns to exclude from interception
    - verbose: Enable verbose logging
    - bank_name: Display name for the memory bank
    - background: Instructions that help Hindsight understand what to remember

Background example:
    >>> configure(
    ...     bank_id="routing-agent",
    ...     background="This agent routes customer requests to support channels. "
    ...                "Remember which types of issues should go to which channels.",
    ... )
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, List, Any

import litellm

from .config import (
    configure,
    get_config,
    is_configured,
    reset_config,
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
    RecallResponse,
    RecallDebugInfo,
    reflect,
    areflect,
    ReflectResult,
    ReflectDebugInfo,
    retain,
    aretain,
    RetainResult,
    RetainDebugInfo,
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


@dataclass
class InjectionDebugInfo:
    """Debug information from a memory injection operation.

    This is populated when verbose=True in the config and can be retrieved
    via get_last_injection_debug() after a completion() call.

    Attributes:
        mode: The injection mode used ("reflect" or "recall")
        query: The user query used for memory lookup
        bank_id: The bank ID used
        memory_context: The formatted memory context that was injected
        reflect_text: The raw reflect text (when mode="reflect")
        reflect_facts: The facts used to generate the reflect response (when reflect_include_facts=True)
        recall_results: The raw recall results (when mode="recall")
        results_count: Number of memories/results found
        injected: Whether memories were actually injected into the prompt
        error: Error message if injection failed (None on success)
    """
    mode: str  # "reflect" or "recall"
    query: str
    bank_id: str
    memory_context: str  # The formatted context that was injected
    reflect_text: Optional[str] = None  # Raw reflect response text
    reflect_facts: Optional[List[dict]] = None  # Facts used by reflect (when reflect_include_facts=True)
    recall_results: Optional[List[dict]] = None  # Raw recall results
    results_count: int = 0
    injected: bool = False
    error: Optional[str] = None  # Error message if injection failed


# Store the last injection debug info (populated when verbose=True)
_last_injection_debug: Optional[InjectionDebugInfo] = None


def get_last_injection_debug() -> Optional[InjectionDebugInfo]:
    """Get debug info from the last memory injection operation.

    When verbose=True in the config, this returns information about
    what memories were injected into the last completion() call.

    Returns:
        InjectionDebugInfo if verbose mode captured injection info, None otherwise

    Example:
        >>> from hindsight_litellm import configure, enable, completion, get_last_injection_debug
        >>> configure(bank_id="my-agent", verbose=True, use_reflect=True)
        >>> enable()
        >>> response = completion(model="gpt-4o-mini", messages=[...])
        >>> debug = get_last_injection_debug()
        >>> if debug:
        ...     print(f"Injected {debug.results_count} memories via {debug.mode}")
        ...     print(f"Reflect text: {debug.reflect_text}")
    """
    return _last_injection_debug


def clear_injection_debug() -> None:
    """Clear the stored injection debug info."""
    global _last_injection_debug
    _last_injection_debug = None


def _inject_memories(messages: List[dict]) -> List[dict]:
    """Inject memories into messages list.

    Returns the modified messages list with memories injected into the system message.
    Uses reflect API when config.use_reflect=True, otherwise uses recall API.

    When verbose=True in config, stores debug info retrievable via get_last_injection_debug().
    """
    global _last_injection_debug
    import logging

    # Clear previous debug info
    _last_injection_debug = None

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
        from hindsight_client import Hindsight

        # Use bank_id directly (no entity scoping)
        bank_id = config.bank_id

        # Track debug info
        mode = "reflect" if config.use_reflect else "recall"
        reflect_text = None
        reflect_facts = None
        recall_results = None
        results_count = 0
        memory_context = ""

        # Create client
        client = Hindsight(base_url=config.hindsight_api_url, timeout=30.0)

        # Use reflect API if use_reflect is enabled
        if config.use_reflect:
            # If reflect_include_facts is enabled, use the API directly to include facts
            if config.reflect_include_facts:
                from hindsight_client_api.models import reflect_request, reflect_include_options
                request_obj = reflect_request.ReflectRequest(
                    query=user_query,
                    budget=config.recall_budget or "mid",
                    include=reflect_include_options.ReflectIncludeOptions(facts={}),
                )
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                result = loop.run_until_complete(client._api.reflect(bank_id, request_obj))
                # Extract facts from based_on
                if hasattr(result, 'based_on') and result.based_on:
                    reflect_facts = [
                        {
                            "text": f.text if hasattr(f, 'text') else str(f),
                            "type": getattr(f, 'type', None),
                            "context": getattr(f, 'context', None),
                        }
                        for f in result.based_on
                    ]
            else:
                result = client.reflect(
                    bank_id=bank_id,
                    query=user_query,
                    budget=config.recall_budget or "mid",
                )
            reflect_text = result.text if hasattr(result, 'text') else str(result)

            if not reflect_text:
                # Store debug info for empty result
                if config.verbose:
                    _last_injection_debug = InjectionDebugInfo(
                        mode=mode,
                        query=user_query,
                        bank_id=bank_id,
                        memory_context="",
                        reflect_text="",
                        reflect_facts=reflect_facts,
                        results_count=0,
                        injected=False,
                    )
                return messages

            results_count = 1  # reflect returns a single synthesized response
            memory_context = (
                "# Relevant Context from Memory\n"
                f"{reflect_text}"
            )
        else:
            # Use recall API (original behavior)
            result = client.recall(
                bank_id=bank_id,
                query=user_query,
                budget=config.recall_budget or "mid",
                max_tokens=config.max_memory_tokens or 4096,
                types=config.fact_types,
            )
            # client.recall() returns a list directly, not an object with .results
            if isinstance(result, list):
                results = result
            elif hasattr(result, 'results'):
                results = result.results
            else:
                results = []
            # Convert to dicts for debug info
            recall_results = [
                {
                    "text": r.text if hasattr(r, 'text') else str(r),
                    "type": getattr(r, 'type', 'world'),
                }
                for r in results
            ]

            if not results:
                # Store debug info for empty result
                if config.verbose:
                    _last_injection_debug = InjectionDebugInfo(
                        mode=mode,
                        query=user_query,
                        bank_id=bank_id,
                        memory_context="",
                        recall_results=[],
                        results_count=0,
                        injected=False,
                    )
                return messages

            # Format memories (apply limit if set, otherwise use all)
            results_to_use = results[:config.max_memories] if config.max_memories else results
            memory_lines = []
            for i, r in enumerate(results_to_use, 1):
                text = r.text if hasattr(r, 'text') else str(r)
                fact_type = getattr(r, 'type', 'world')
                if text:
                    type_label = fact_type.upper() if fact_type else "MEMORY"
                    memory_lines.append(f"{i}. [{type_label}] {text}")

            if not memory_lines:
                if config.verbose:
                    _last_injection_debug = InjectionDebugInfo(
                        mode=mode,
                        query=user_query,
                        bank_id=bank_id,
                        memory_context="",
                        recall_results=recall_results,
                        results_count=0,
                        injected=False,
                    )
                return messages

            results_count = len(memory_lines)
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

        # Store debug info when verbose
        if config.verbose:
            _last_injection_debug = InjectionDebugInfo(
                mode=mode,
                query=user_query,
                bank_id=bank_id,
                memory_context=memory_context,
                reflect_text=reflect_text,
                reflect_facts=reflect_facts,
                recall_results=recall_results,
                results_count=results_count,
                injected=True,
            )
            logger = logging.getLogger("hindsight_litellm")
            logger.info(f"Injected memories using {mode} into prompt")

        return updated_messages

    except ImportError as e:
        if config.verbose:
            logging.getLogger("hindsight_litellm").warning(
                f"hindsight_client not installed: {e}. Install with: pip install hindsight-client"
            )
            _last_injection_debug = InjectionDebugInfo(
                mode="reflect" if config.use_reflect else "recall",
                query=user_query or "",
                bank_id=config.bank_id or "",
                memory_context="",
                results_count=0,
                injected=False,
                error=f"hindsight_client not installed: {e}",
            )
        return messages
    except Exception as e:
        # Always set debug info on error when verbose mode is on
        if config.verbose:
            logging.getLogger("hindsight_litellm").warning(f"Failed to inject memories: {e}")
            _last_injection_debug = InjectionDebugInfo(
                mode="reflect" if config.use_reflect else "recall",
                query=user_query or "",
                bank_id=config.bank_id or "",
                memory_context="",
                results_count=0,
                injected=False,
                error=str(e),
            )
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
    store_conversations: bool = True,
    inject_memories: bool = True,
    injection_mode: MemoryInjectionMode = MemoryInjectionMode.SYSTEM_MESSAGE,
    max_memories: Optional[int] = None,
    max_memory_tokens: int = 4096,
    recall_budget: str = "mid",
    fact_types: Optional[List[str]] = None,
    document_id: Optional[str] = None,
    excluded_models: Optional[List[str]] = None,
    verbose: bool = False,
    bank_name: Optional[str] = None,
    background: Optional[str] = None,
):
    """Context manager for temporary Hindsight memory integration.

    Use this to enable memory integration for a specific block of code,
    automatically cleaning up afterwards.

    Args:
        hindsight_api_url: URL of the Hindsight API server
        bank_id: Memory bank ID for memory operations (required). For multi-user
            support, use different bank_ids per user (e.g., f"user-{user_id}")
        api_key: Optional API key for Hindsight authentication
        store_conversations: Whether to store conversations
        inject_memories: Whether to inject relevant memories
        injection_mode: How to inject memories
        max_memories: Maximum number of memories to inject (None = unlimited)
        max_memory_tokens: Maximum tokens for memory context
        recall_budget: Budget for memory recall (low, mid, high)
        fact_types: List of fact types to filter (world, agent, opinion, observation)
        document_id: Optional document ID for grouping conversations
        excluded_models: List of model patterns to exclude
        verbose: Enable verbose logging
        bank_name: Optional display name for the memory bank
        background: Optional background/instructions for memory extraction

    Example:
        >>> from hindsight_litellm import hindsight_memory
        >>> import litellm
        >>>
        >>> with hindsight_memory(bank_id="user-123"):
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
            bank_name=bank_name,
            background=background,
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
                bank_name=previous_config.bank_name,
                background=previous_config.background,
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
    # Direct memory APIs
    "recall",
    "arecall",
    "RecallResult",
    "reflect",
    "areflect",
    "ReflectResult",
    "retain",
    "aretain",
    "RetainResult",
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
    # Injection debug (verbose mode)
    "get_last_injection_debug",
    "clear_injection_debug",
    "InjectionDebugInfo",
    # Callback (for advanced usage)
    "HindsightCallback",
    "get_callback",
    "cleanup_callback",
]
