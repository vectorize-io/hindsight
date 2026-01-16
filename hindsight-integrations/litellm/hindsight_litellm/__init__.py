"""Hindsight-LiteLLM: Universal LLM memory integration via LiteLLM.

This package provides automatic memory integration for any LLM provider
supported by LiteLLM (100+ providers including OpenAI, Anthropic, Groq,
Azure, AWS Bedrock, Google Vertex AI, and more).

Features:
- Automatic memory injection before LLM calls
- Automatic conversation storage after LLM calls
- Works with any LiteLLM-supported provider
- Multi-user support via separate bank_ids
- Per-call overrides via hindsight_* kwargs
- Document grouping for conversation threading
- Direct recall API for manual memory queries
- Native client wrappers for OpenAI and Anthropic
- STRICT ERROR HANDLING: Raises HindsightError on any memory operation failure

Error Handling:
    Unlike LiteLLM's callback system which silently swallows exceptions, this
    integration uses STRICT error handling. If memory injection fails (when
    inject_memories=True) or storage fails (when store_conversations=True),
    a HindsightError will be raised and propagate to your code.

API Structure:
    1. configure() - Static settings (rarely change during session)
       - hindsight_api_url, api_key, verbose
       - injection_mode, excluded_models, store_conversations, inject_memories

    2. set_defaults() - Default values for per-call settings (required: bank_id)
       - bank_id (REQUIRED), document_id, budget, fact_types
       - max_memories, max_memory_tokens, use_reflect, reflect_include_facts
       - include_entities (default True), trace (default False)

    3. Per-call kwargs (hindsight_* prefix) - Override any default per-call
       - hindsight_bank_id, hindsight_document_id, hindsight_budget, etc.
       - hindsight_include_entities, hindsight_trace

    4. set_bank_mission() - Set mission/instructions for a bank (for mental models)
       - Can be called anytime, bank is auto-created if needed
       - set_bank_background() is deprecated, use set_bank_mission() instead

Basic usage:
    >>> import hindsight_litellm
    >>> from hindsight_litellm import HindsightError
    >>>
    >>> # Configure static settings
    >>> hindsight_litellm.configure(
    ...     hindsight_api_url="http://localhost:8888",
    ...     verbose=True,
    ... )
    >>>
    >>> # Set defaults (bank_id is required)
    >>> hindsight_litellm.set_defaults(bank_id="user-123")
    >>>
    >>> # Optionally set bank mission (for mental models)
    >>> hindsight_litellm.set_bank_mission(
    ...     mission="This agent helps with customer support. Remember customer preferences."
    ... )
    >>>
    >>> # Enable memory integration
    >>> hindsight_litellm.enable()
    >>>
    >>> # Use litellm.completion() or hindsight_litellm.completion() - both work
    >>> try:
    ...     response = hindsight_litellm.completion(
    ...         model="gpt-4",
    ...         messages=[{"role": "user", "content": "What did we discuss?"}]
    ...     )
    ... except HindsightError as e:
    ...     print(f"Memory operation failed: {e}")
    >>>
    >>> # Override per-call:
    >>> response = hindsight_litellm.completion(
    ...     model="gpt-4",
    ...     messages=[...],
    ...     hindsight_bank_id="different-bank",  # Override default bank_id
    ...     hindsight_document_id="conv-123",    # Set document_id for this call
    ... )

Direct recall API:
    >>> from hindsight_litellm import configure, set_defaults, recall
    >>> configure(hindsight_api_url="http://localhost:8888")
    >>> set_defaults(bank_id="my-agent")
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
    >>> hindsight_litellm.completion(model="gpt-4", messages=[...])
    >>>
    >>> # Anthropic
    >>> hindsight_litellm.completion(model="claude-3-opus-20240229", messages=[...])
    >>>
    >>> # Groq
    >>> hindsight_litellm.completion(model="groq/llama-3.1-70b-versatile", messages=[...])
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, List, Any
import threading
import logging

import litellm

from .config import (
    configure,
    set_defaults,
    set_bank_mission,
    get_config,
    get_defaults,
    is_configured,
    reset_config,
    set_document_id,
    HindsightConfig,
    HindsightDefaults,
    MemoryInjectionMode,
)
from .callbacks import (
    HindsightCallback,
    HindsightError,
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
    get_pending_retain_errors,
    wrap_openai,
    wrap_anthropic,
    HindsightOpenAI,
    HindsightAnthropic,
    _get_client,
    _close_client,
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


def _inject_memories(messages: List[dict], custom_query: Optional[str] = None, custom_reflect_context: Optional[str] = None) -> List[dict]:
    """Inject memories into messages list.

    Returns the modified messages list with memories injected into the system message.
    Uses reflect API when defaults.use_reflect=True, otherwise uses recall API.

    When verbose=True in config, stores debug info retrievable via get_last_injection_debug().

    Args:
        messages: List of message dicts to inject memories into
        custom_query: Optional custom query to use for memory lookup instead of user message
        custom_reflect_context: Optional context to pass to reflect API (overrides defaults.reflect_context)
    """
    global _last_injection_debug
    import logging

    # Clear previous debug info
    _last_injection_debug = None

    config = get_config()
    defaults = get_defaults()

    if not config or not config.inject_memories:
        return messages

    if not defaults or not defaults.bank_id:
        raise ValueError(
            "No bank_id configured. Either call set_defaults(bank_id=...) "
            "or pass hindsight_bank_id=... to the completion call."
        )

    if not messages:
        return messages

    # hindsight_query is required when inject_memories=True
    if not custom_query:
        raise ValueError(
            "hindsight_query is required when inject_memories=True. "
            "Pass hindsight_query='your query' to specify what to search for in memory. "
            "Example: hindsight_query=recipient_name or hindsight_query='What do I know about Alice?'"
        )

    user_query = custom_query

    # Use bank_id from defaults
    bank_id = defaults.bank_id

    # Track debug info
    mode = "reflect" if defaults.use_reflect else "recall"
    reflect_text = None
    reflect_facts = None
    recall_results = None
    results_count = 0
    memory_context = ""

    # Create fresh client for this operation (closed in finally block)
    client = None
    try:
        client = _get_client(config.hindsight_api_url)

        # Use reflect API if use_reflect is enabled
        if defaults.use_reflect:
            # Build common reflect parameters
            reflect_kwargs = {
                "query": user_query,
                "budget": defaults.budget or "mid",
            }
            # Add context if provided (shapes reasoning but not retrieval)
            # Per-call context overrides default context
            if custom_reflect_context:
                reflect_kwargs["context"] = custom_reflect_context
            elif defaults.reflect_context:
                reflect_kwargs["context"] = defaults.reflect_context
            # Add response_schema for structured output
            if defaults.reflect_response_schema:
                reflect_kwargs["response_schema"] = defaults.reflect_response_schema

            # If reflect_include_facts is enabled, use the API directly to include facts
            if defaults.reflect_include_facts:
                from hindsight_client_api.models import reflect_request, reflect_include_options
                request_obj = reflect_request.ReflectRequest(
                    include=reflect_include_options.ReflectIncludeOptions(facts={}),
                    **reflect_kwargs,
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
                    **reflect_kwargs,
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
                budget=defaults.budget or "mid",
                max_tokens=defaults.max_memory_tokens or 4096,
                types=defaults.fact_types,
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
            results_to_use = results[:defaults.max_memories] if defaults.max_memories else results
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
        if config and config.verbose:
            logging.getLogger("hindsight_litellm").warning(
                f"hindsight_client not installed: {e}. Install with: pip install hindsight-client"
            )
            _last_injection_debug = InjectionDebugInfo(
                mode="reflect" if (defaults and defaults.use_reflect) else "recall",
                query=user_query or "",
                bank_id=(defaults.bank_id if defaults else "") or "",
                memory_context="",
                results_count=0,
                injected=False,
                error=f"hindsight_client not installed: {e}",
            )
        return messages
    except Exception as e:
        # Always set debug info on error when verbose mode is on
        if config and config.verbose:
            logging.getLogger("hindsight_litellm").warning(f"Failed to inject memories: {e}")
            _last_injection_debug = InjectionDebugInfo(
                mode="reflect" if (defaults and defaults.use_reflect) else "recall",
                query=user_query or "",
                bank_id=(defaults.bank_id if defaults else "") or "",
                memory_context="",
                results_count=0,
                injected=False,
                error=str(e),
            )
        return messages
    finally:
        # Always close the client to avoid "Unclosed client session" warnings
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


def _wrapped_completion(*args, **kwargs):
    """Wrapper for litellm.completion that handles memory injection and storage.

    This wrapper:
    1. Injects memories before the LLM call (raises HindsightError on failure)
    2. Calls the original litellm.completion
    3. Stores the conversation after success (raises HindsightError on failure)
    """
    config = get_config()

    # Extract hindsight-specific kwargs
    custom_query = kwargs.pop("hindsight_query", None)
    custom_reflect_context = kwargs.pop("hindsight_reflect_context", None)

    # Extract messages from kwargs or args
    messages = kwargs.get("messages")
    if messages is None and len(args) > 1:
        messages = args[1]

    model = kwargs.get("model")
    if model is None and len(args) > 0:
        model = args[0]

    # Step 1: Inject memories (raises HindsightError on failure)
    if config and config.inject_memories and messages:
        try:
            injected_messages = _inject_memories(messages, custom_query=custom_query, custom_reflect_context=custom_reflect_context)
            kwargs["messages"] = injected_messages
        except Exception as e:
            raise HindsightError(f"Failed to inject memories: {e}") from e

    # Step 2: Call original LLM
    response = _original_completion(*args, **kwargs)

    # Step 3: Store conversation (raises HindsightError on failure)
    if config and config.store_conversations:
        final_messages = kwargs.get("messages", messages)
        if final_messages:
            _store_conversation(final_messages, response, model or "unknown")

    return response


async def _wrapped_acompletion(*args, **kwargs):
    """Wrapper for litellm.acompletion that handles memory injection and storage.

    This wrapper:
    1. Injects memories before the LLM call (raises HindsightError on failure)
    2. Calls the original litellm.acompletion
    3. Stores the conversation after success (raises HindsightError on failure)
    """
    config = get_config()

    # Extract hindsight-specific kwargs
    custom_query = kwargs.pop("hindsight_query", None)
    custom_reflect_context = kwargs.pop("hindsight_reflect_context", None)

    # Extract messages from kwargs or args
    messages = kwargs.get("messages")
    if messages is None and len(args) > 1:
        messages = args[1]

    model = kwargs.get("model")
    if model is None and len(args) > 0:
        model = args[0]

    # Step 1: Inject memories (raises HindsightError on failure)
    if config and config.inject_memories and messages:
        try:
            injected_messages = _inject_memories(messages, custom_query=custom_query, custom_reflect_context=custom_reflect_context)
            kwargs["messages"] = injected_messages
        except Exception as e:
            raise HindsightError(f"Failed to inject memories: {e}") from e

    # Step 2: Call original LLM
    response = await _original_acompletion(*args, **kwargs)

    # Step 3: Store conversation (raises HindsightError on failure)
    if config and config.store_conversations:
        final_messages = kwargs.get("messages", messages)
        if final_messages:
            _store_conversation(final_messages, response, model or "unknown")

    return response


def enable() -> None:
    """Enable Hindsight memory integration with LiteLLM.

    This monkeypatches litellm.completion and litellm.acompletion to:
    1. Inject relevant memories into prompts before LLM calls
    2. Store conversations to Hindsight after successful LLM calls

    STRICT ERROR HANDLING: Unlike LiteLLM's callback system which swallows
    exceptions, this integration raises HindsightError on any failure. If
    memory injection fails (when inject_memories=True) or storage fails
    (when store_conversations=True), the error will propagate to your code.

    Must be called after configure() and set_defaults(bank_id=...).

    Example:
        >>> from hindsight_litellm import configure, set_defaults, enable, HindsightError
        >>> import litellm
        >>>
        >>> configure(hindsight_api_url="http://localhost:8888")
        >>> set_defaults(bank_id="my-agent")
        >>> enable()
        >>>
        >>> # Now litellm.completion() has memory integration
        >>> try:
        ...     response = litellm.completion(
        ...         model="gpt-4",
        ...         messages=[{"role": "user", "content": "Hello!"}]
        ...     )
        ... except HindsightError as e:
        ...     print(f"Memory operation failed: {e}")

    Raises:
        RuntimeError: If configure() or set_defaults() hasn't been called
    """
    global _enabled, _original_completion, _original_acompletion

    if _enabled:
        return  # Already enabled

    config = get_config()
    defaults = get_defaults()

    if not config:
        raise RuntimeError(
            "Hindsight not configured. Call configure() before enable()."
        )

    if not defaults or not defaults.bank_id:
        raise RuntimeError(
            "Hindsight bank_id not set. Call set_defaults(bank_id=...) before enable()."
        )

    # Store original functions and monkeypatch for memory injection + storage
    _original_completion = litellm.completion
    _original_acompletion = litellm.acompletion
    litellm.completion = _wrapped_completion
    litellm.acompletion = _wrapped_acompletion

    _enabled = True

    if config.verbose:
        print(f"Hindsight memory enabled for bank: {defaults.bank_id}")


def disable() -> None:
    """Disable Hindsight memory integration with LiteLLM.

    This restores the original LiteLLM functions, stopping memory injection
    and conversation storage. Also closes any cached HTTP connections.

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

    # Close cached HTTP client to avoid "Unclosed client session" warnings
    _close_client()

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
    disable()  # This already calls _close_client()
    cleanup_callback()
    reset_config()


# =============================================================================
# Convenience wrappers - use hindsight_litellm.completion() directly
# =============================================================================

def _format_conversation_for_storage(
    messages: List[dict],
    response,
) -> str:
    """Format conversation messages and response for storage to Hindsight.

    Returns the formatted conversation text.
    """
    items = []

    for msg in messages:
        role = msg.get("role", "").upper()
        content = msg.get("content", "")

        # Skip system messages
        if role == "SYSTEM":
            continue

        # Skip injected memory context
        if isinstance(content, str) and content.startswith("# Relevant Memories"):
            continue

        # Handle tool results
        if role == "TOOL":
            items.append(f"TOOL_RESULT: {content}")
            continue

        # Handle assistant messages with tool calls
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            tc_strs = []
            for tc in tool_calls:
                if hasattr(tc, "function"):
                    tc_strs.append(f"{tc.function.name}({tc.function.arguments})")
                elif isinstance(tc, dict) and "function" in tc:
                    func = tc["function"]
                    tc_strs.append(f"{func.get('name', '')}({func.get('arguments', '')})")
            if tc_strs:
                items.append(f"ASSISTANT_TOOL_CALLS: {'; '.join(tc_strs)}")
            if content:
                items.append(f"ASSISTANT: {content}")
            continue

        # Handle structured content (vision messages)
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            content = " ".join(text_parts)

        if content:
            label = "USER" if role == "USER" else "ASSISTANT"
            items.append(f"{label}: {content}")

    # Add the response
    if response.choices and len(response.choices) > 0:
        choice = response.choices[0]
        if hasattr(choice, "message") and choice.message:
            assistant_content = choice.message.content or ""
            assistant_tool_calls = []
            if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    if hasattr(tc, "function"):
                        assistant_tool_calls.append(f"{tc.function.name}({tc.function.arguments})")

            if assistant_content:
                items.append(f"ASSISTANT: {assistant_content}")
            if assistant_tool_calls:
                items.append(f"ASSISTANT_TOOL_CALLS: {'; '.join(assistant_tool_calls)}")

    return "\n\n".join(items)


_storage_logger = logging.getLogger("hindsight_litellm.storage")

# Track storage errors from background threads - raised on next completion call
_pending_storage_errors: List[Exception] = []
_storage_error_lock = threading.Lock()


def _store_conversation_sync(
    conversation_text: str,
    bank_id: str,
    document_id: Optional[str],
    model: str,
    verbose: bool,
) -> None:
    """Actually store the conversation (runs in background thread)."""
    global _pending_storage_errors
    try:
        retain(
            content=conversation_text,
            bank_id=bank_id,
            context=f"conversation:litellm:{model}",
            document_id=document_id,
            metadata={"source": "litellm", "model": model},
        )
        if verbose:
            _storage_logger.info(f"Stored conversation to bank: {bank_id}")
    except Exception as e:
        _storage_logger.error(f"Failed to store conversation: {e}")
        # Store error to raise on next completion call
        with _storage_error_lock:
            _pending_storage_errors.append(
                HindsightError(f"Background storage failed: {e}")
            )


def _check_pending_storage_errors() -> None:
    """Check for and raise any pending storage errors from background threads."""
    global _pending_storage_errors
    with _storage_error_lock:
        if _pending_storage_errors:
            # Get first error and clear the list
            error = _pending_storage_errors[0]
            _pending_storage_errors.clear()
            raise error


def get_pending_storage_errors() -> List[Exception]:
    """Get any pending storage errors without raising them.

    Useful for checking/logging errors without interrupting flow.
    Clears the error queue after returning.

    Returns:
        List of HindsightError exceptions from failed background storage operations
    """
    global _pending_storage_errors
    with _storage_error_lock:
        errors = list(_pending_storage_errors)
        _pending_storage_errors.clear()
        return errors


def _store_conversation(
    messages: List[dict],
    response,
    model: str,
) -> None:
    """Store conversation to Hindsight.

    By default, storage runs in a background thread for performance.
    If sync_storage=True in config, runs synchronously and raises errors.
    Use get_pending_storage_errors() to check for async storage failures.
    """
    config = get_config()
    defaults = get_defaults()

    if not config or not config.store_conversations:
        return

    if not defaults or not defaults.bank_id:
        _storage_logger.warning(
            "No bank_id configured for storage. Call set_defaults(bank_id=...)."
        )
        return

    # Format conversation
    conversation_text = _format_conversation_for_storage(messages, response)

    if not conversation_text:
        return

    # Sync mode: run directly and raise errors
    if config.sync_storage:
        try:
            retain(
                content=conversation_text,
                bank_id=defaults.bank_id,
                context=f"conversation:litellm:{model}",
                document_id=defaults.document_id,
                metadata={"source": "litellm", "model": model},
            )
            if config.verbose:
                _storage_logger.info(f"Stored conversation to bank: {defaults.bank_id}")
        except Exception as e:
            raise HindsightError(f"Failed to store conversation: {e}") from e
        return

    # Async mode (default): run in background thread
    thread = threading.Thread(
        target=_store_conversation_sync,
        args=(
            conversation_text,
            defaults.bank_id,
            defaults.document_id,
            model,
            config.verbose,
        ),
        daemon=True,
    )
    thread.start()


def completion(*args, **kwargs):
    """Call LiteLLM completion with Hindsight memory integration.

    This wrapper handles memory injection and storage explicitly, ensuring
    that any Hindsight failures raise HindsightError instead of failing silently.

    Args:
        *args: Positional arguments passed to litellm.completion()
        **kwargs: Keyword arguments passed to litellm.completion()
            Special hindsight_* kwargs:
            - hindsight_query: Custom query for memory lookup (overrides user message)

    Returns:
        LiteLLM ModelResponse object

    Raises:
        HindsightError: If memory injection or storage fails

    Example:
        >>> import hindsight_litellm
        >>>
        >>> hindsight_litellm.configure(
        ...     hindsight_api_url="http://localhost:8888",
        ... )
        >>> hindsight_litellm.set_defaults(bank_id="my-agent")
        >>> hindsight_litellm.enable()
        >>>
        >>> # Use directly - no need to import litellm separately
        >>> response = hindsight_litellm.completion(
        ...     model="gpt-4o-mini",
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
        >>>
        >>> # With custom query for memory lookup
        >>> response = hindsight_litellm.completion(
        ...     model="gpt-4o-mini",
        ...     messages=[{"role": "user", "content": "Please deliver package to Alice"}],
        ...     hindsight_query="Where is Alice located?",  # Focused query for memory
        ... )
        >>>
        >>> # With custom reflect context (conversation history for reflect)
        >>> response = hindsight_litellm.completion(
        ...     model="gpt-4o-mini",
        ...     messages=[...],
        ...     hindsight_query="What should I do next?",
        ...     hindsight_reflect_context="Step 1: Checked floor 1. Step 2: Found elevator.",
        ... )
    """
    config = get_config()

    # Extract hindsight-specific kwargs
    custom_query = kwargs.pop("hindsight_query", None)
    custom_reflect_context = kwargs.pop("hindsight_reflect_context", None)

    # Extract messages from kwargs or args
    messages = kwargs.get("messages")
    if messages is None and len(args) > 1:
        messages = args[1]

    model = kwargs.get("model")
    if model is None and len(args) > 0:
        model = args[0]

    # Step 1: Inject memories (raises HindsightError on failure)
    if config and config.inject_memories and messages:
        try:
            injected_messages = _inject_memories(messages, custom_query=custom_query, custom_reflect_context=custom_reflect_context)
            kwargs["messages"] = injected_messages
        except Exception as e:
            raise HindsightError(f"Failed to inject memories: {e}") from e

    # Step 2: Call LLM
    response = litellm.completion(*args, **kwargs)

    # Step 3: Store conversation (raises HindsightError on failure)
    if config and config.store_conversations:
        final_messages = kwargs.get("messages", messages)
        _store_conversation(final_messages, response, model or "unknown")

    return response


async def acompletion(*args, **kwargs):
    """Call LiteLLM async completion with Hindsight memory integration.

    This wrapper handles memory injection and storage explicitly, ensuring
    that any Hindsight failures raise HindsightError instead of failing silently.

    Args:
        *args: Positional arguments passed to litellm.acompletion()
        **kwargs: Keyword arguments passed to litellm.acompletion()
            Special hindsight_* kwargs:
            - hindsight_query: Custom query for memory lookup (overrides user message)

    Returns:
        LiteLLM ModelResponse object

    Raises:
        HindsightError: If memory injection or storage fails

    Example:
        >>> import hindsight_litellm
        >>> import asyncio
        >>>
        >>> hindsight_litellm.configure(
        ...     hindsight_api_url="http://localhost:8888",
        ... )
        >>> hindsight_litellm.set_defaults(bank_id="my-agent")
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
    config = get_config()

    # Extract hindsight-specific kwargs
    custom_query = kwargs.pop("hindsight_query", None)
    custom_reflect_context = kwargs.pop("hindsight_reflect_context", None)

    # Extract messages from kwargs or args
    messages = kwargs.get("messages")
    if messages is None and len(args) > 1:
        messages = args[1]

    model = kwargs.get("model")
    if model is None and len(args) > 0:
        model = args[0]

    # Step 1: Inject memories (raises HindsightError on failure)
    if config and config.inject_memories and messages:
        try:
            injected_messages = _inject_memories(messages, custom_query=custom_query, custom_reflect_context=custom_reflect_context)
            kwargs["messages"] = injected_messages
        except Exception as e:
            raise HindsightError(f"Failed to inject memories: {e}") from e

    # Step 2: Call LLM
    response = await litellm.acompletion(*args, **kwargs)

    # Step 3: Store conversation (raises HindsightError on failure)
    if config and config.store_conversations:
        final_messages = kwargs.get("messages", messages)
        _store_conversation(final_messages, response, model or "unknown")

    return response


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
    budget: str = "mid",
    fact_types: Optional[List[str]] = None,
    document_id: Optional[str] = None,
    excluded_models: Optional[List[str]] = None,
    verbose: bool = False,
    include_entities: bool = True,
    trace: bool = False,
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
        budget: Budget for memory recall (low, mid, high)
        fact_types: List of fact types to filter (world, experience, opinion, observation)
        document_id: Optional document ID for grouping conversations
        excluded_models: List of model patterns to exclude
        verbose: Enable verbose logging
        include_entities: Include entity observations in recall (default True)
        trace: Enable trace info for debugging (default False)

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
    previous_defaults = get_defaults()

    try:
        # Configure and enable
        configure(
            hindsight_api_url=hindsight_api_url,
            api_key=api_key,
            store_conversations=store_conversations,
            inject_memories=inject_memories,
            injection_mode=injection_mode,
            excluded_models=excluded_models,
            verbose=verbose,
        )
        set_defaults(
            bank_id=bank_id,
            document_id=document_id,
            budget=budget,
            fact_types=fact_types,
            max_memories=max_memories,
            max_memory_tokens=max_memory_tokens,
            include_entities=include_entities,
            trace=trace,
        )
        enable()
        yield
    finally:
        # Restore previous state
        disable()
        if previous_config:
            configure(
                hindsight_api_url=previous_config.hindsight_api_url,
                api_key=previous_config.api_key,
                store_conversations=previous_config.store_conversations,
                inject_memories=previous_config.inject_memories,
                injection_mode=previous_config.injection_mode,
                excluded_models=previous_config.excluded_models,
                verbose=previous_config.verbose,
            )
        if previous_defaults:
            set_defaults(
                bank_id=previous_defaults.bank_id,
                document_id=previous_defaults.document_id,
                budget=previous_defaults.budget,
                fact_types=previous_defaults.fact_types,
                max_memories=previous_defaults.max_memories,
                max_memory_tokens=previous_defaults.max_memory_tokens,
                use_reflect=previous_defaults.use_reflect,
                reflect_include_facts=previous_defaults.reflect_include_facts,
                include_entities=previous_defaults.include_entities,
                trace=previous_defaults.trace,
            )
            if was_enabled:
                enable()
        else:
            reset_config()


__all__ = [
    # Main API
    "configure",
    "set_defaults",
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
    "get_defaults",
    "is_configured",
    "reset_config",
    "set_document_id",
    "set_bank_mission",
    "HindsightConfig",
    "HindsightDefaults",
    "MemoryInjectionMode",
    # Injection debug (verbose mode)
    "get_last_injection_debug",
    "clear_injection_debug",
    "InjectionDebugInfo",
    # Storage errors (async mode)
    "get_pending_storage_errors",
    "get_pending_retain_errors",
    # Callback (for advanced usage)
    "HindsightCallback",
    "get_callback",
    "cleanup_callback",
    # Exceptions
    "HindsightError",
]
