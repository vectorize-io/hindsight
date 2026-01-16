"""Global configuration for Hindsight-LiteLLM integration.

This module provides a clean API for configuring Hindsight integration:

1. configure() - Static settings that rarely change during a session
   - API URL, authentication, logging, injection mode, etc.

2. set_defaults() - Default values for per-call settings
   - bank_id, document_id, budget, fact_types, etc.
   - These are used when per-call kwargs are not provided

3. Per-call kwargs (hindsight_* prefix) - Override any default per-call
   - hindsight_bank_id, hindsight_document_id, etc.

4. set_bank_mission() - Set the mission for a memory bank (for mental models)
"""

from typing import Optional, List, Any, Dict
from dataclasses import dataclass, field
from enum import Enum


class MemoryInjectionMode(str, Enum):
    """How memories should be injected into the prompt.

    Use inject_memories=False if you don't want memory injection.
    """
    SYSTEM_MESSAGE = "system_message"  # Add to/create system message
    PREPEND_USER = "prepend_user"  # Prepend to last user message


@dataclass
class HindsightConfig:
    """Static configuration for Hindsight integration with LiteLLM.

    These settings typically don't change during a session.

    Attributes:
        hindsight_api_url: URL of the Hindsight API server
        api_key: Optional API key for Hindsight authentication
        store_conversations: Whether to store conversations to Hindsight
        inject_memories: Whether to inject relevant memories into prompts
        injection_mode: How to inject memories (system_message or prepend_user)
        excluded_models: List of model patterns to exclude from interception
        verbose: Enable verbose logging
        sync_storage: If True, storage runs synchronously and raises errors immediately.
            If False (default), storage runs in background thread for better performance.
    """

    hindsight_api_url: str = "http://localhost:8888"
    api_key: Optional[str] = None
    store_conversations: bool = True
    inject_memories: bool = True
    injection_mode: MemoryInjectionMode = MemoryInjectionMode.SYSTEM_MESSAGE
    excluded_models: List[str] = field(default_factory=list)
    verbose: bool = False
    sync_storage: bool = False


@dataclass
class HindsightDefaults:
    """Default values for per-call settings.

    These can be overridden on a per-call basis using hindsight_* kwargs.

    Attributes:
        bank_id: Memory bank ID for memory operations
        document_id: Optional document ID for grouping stored conversations
        budget: Budget level for memory recall (low, mid, high)
        fact_types: List of fact types to filter recall (world, experience, opinion, observation)
        max_memories: Maximum number of memories to inject (None = no limit)
        max_memory_tokens: Maximum tokens for injected memory context
        use_reflect: Use reflect API instead of recall for memory injection
        reflect_include_facts: Include facts used by reflect in debug info
        reflect_context: Additional context for reflect reasoning (does not affect retrieval)
        reflect_response_schema: JSON Schema for structured reflect output
        include_entities: Include entity observations in recall results
        trace: Enable trace info for recall debugging

    Note:
        For custom queries, use the hindsight_query kwarg per-call instead of a default,
        since queries typically need to be dynamic (e.g., include recipient name).
    """

    bank_id: Optional[str] = None
    document_id: Optional[str] = None
    budget: str = "mid"  # low, mid, high
    fact_types: Optional[List[str]] = None  # world, experience, opinion, observation
    max_memories: Optional[int] = None  # None = no limit
    max_memory_tokens: int = 4096
    use_reflect: bool = False
    reflect_include_facts: bool = False
    reflect_context: Optional[str] = None  # Context for reflect reasoning
    reflect_response_schema: Optional[Dict[str, Any]] = None  # JSON Schema for structured output
    include_entities: bool = True  # Include entity observations by default
    trace: bool = False  # Enable trace info for debugging


# Global instances
_global_config: Optional[HindsightConfig] = None
_global_defaults: Optional[HindsightDefaults] = None


def configure(
    hindsight_api_url: str = "http://localhost:8888",
    api_key: Optional[str] = None,
    store_conversations: bool = True,
    inject_memories: bool = True,
    injection_mode: MemoryInjectionMode = MemoryInjectionMode.SYSTEM_MESSAGE,
    excluded_models: Optional[List[str]] = None,
    verbose: bool = False,
    sync_storage: bool = False,
) -> HindsightConfig:
    """Configure static Hindsight integration settings for LiteLLM.

    This sets up settings that typically don't change during a session.
    For per-call settings like bank_id, use set_defaults() or per-call kwargs.

    Args:
        hindsight_api_url: URL of the Hindsight API server
        api_key: Optional API key for Hindsight authentication
        store_conversations: Whether to store conversations to Hindsight
        inject_memories: Whether to inject relevant memories into prompts
        injection_mode: How to inject memories into the prompt
        excluded_models: List of model patterns to exclude from interception
        verbose: Enable verbose logging
        sync_storage: If True, storage runs synchronously and raises errors immediately.
            If False (default), storage runs in background for better performance.
            Use get_pending_storage_errors() to check for async storage failures.

    Returns:
        The configured HindsightConfig instance

    Example:
        >>> from hindsight_litellm import configure, set_defaults, enable
        >>> configure(
        ...     hindsight_api_url="http://localhost:8888",
        ...     api_key="your-api-key",
        ...     verbose=True,
        ... )
        >>> set_defaults(bank_id="user-123")
        >>> enable()  # Start memory integration
    """
    global _global_config

    _global_config = HindsightConfig(
        hindsight_api_url=hindsight_api_url,
        api_key=api_key,
        store_conversations=store_conversations,
        inject_memories=inject_memories,
        injection_mode=injection_mode,
        excluded_models=excluded_models or [],
        verbose=verbose,
        sync_storage=sync_storage,
    )

    return _global_config


def set_defaults(
    bank_id: Optional[str] = None,
    document_id: Optional[str] = None,
    budget: Optional[str] = None,
    fact_types: Optional[List[str]] = None,
    max_memories: Optional[int] = None,
    max_memory_tokens: Optional[int] = None,
    use_reflect: Optional[bool] = None,
    reflect_include_facts: Optional[bool] = None,
    reflect_context: Optional[str] = None,
    reflect_response_schema: Optional[Dict[str, Any]] = None,
    include_entities: Optional[bool] = None,
    trace: Optional[bool] = None,
) -> HindsightDefaults:
    """Set default values for per-call settings.

    These defaults are used when per-call kwargs are not provided.
    Any of these can be overridden on individual LLM calls using
    hindsight_* kwargs (e.g., hindsight_bank_id="other-bank").

    Args:
        bank_id: Default memory bank ID for memory operations
        document_id: Default document ID for grouping stored conversations
        budget: Default budget level for memory recall (low, mid, high)
        fact_types: Default fact types to filter (world, experience, opinion, observation)
        max_memories: Default max number of memories to inject
        max_memory_tokens: Default max tokens for memory context
        use_reflect: Default whether to use reflect API instead of recall
        reflect_include_facts: Default whether to include facts in reflect debug info
        reflect_context: Default context for reflect reasoning (shapes LLM response, not retrieval)
        reflect_response_schema: Default JSON Schema for structured reflect output
        include_entities: Default whether to include entity observations in recall (default True)
        trace: Default whether to enable trace info for debugging (default False)

    Returns:
        The configured HindsightDefaults instance

    Note:
        For custom memory queries, use hindsight_query per-call instead of a default,
        since queries typically need to be dynamic (e.g., include recipient name).

    Example:
        >>> from hindsight_litellm import set_defaults
        >>> set_defaults(
        ...     bank_id="my-agent",
        ...     budget="high",
        ...     fact_types=["world", "opinion"],
        ...     reflect_context="I am a delivery agent finding package recipients.",
        ... )
        >>>
        >>> # Override per-call with dynamic query:
        >>> response = litellm.completion(
        ...     model="gpt-4",
        ...     messages=[...],
        ...     hindsight_query=f"Where is {recipient_name} located?",  # Dynamic query
        ... )
    """
    global _global_defaults

    # Get current defaults or create new
    current = _global_defaults or HindsightDefaults()

    # Update only provided values
    _global_defaults = HindsightDefaults(
        bank_id=bank_id if bank_id is not None else current.bank_id,
        document_id=document_id if document_id is not None else current.document_id,
        budget=budget if budget is not None else current.budget,
        fact_types=fact_types if fact_types is not None else current.fact_types,
        max_memories=max_memories if max_memories is not None else current.max_memories,
        max_memory_tokens=max_memory_tokens if max_memory_tokens is not None else current.max_memory_tokens,
        use_reflect=use_reflect if use_reflect is not None else current.use_reflect,
        reflect_include_facts=reflect_include_facts if reflect_include_facts is not None else current.reflect_include_facts,
        reflect_context=reflect_context if reflect_context is not None else current.reflect_context,
        reflect_response_schema=reflect_response_schema if reflect_response_schema is not None else current.reflect_response_schema,
        include_entities=include_entities if include_entities is not None else current.include_entities,
        trace=trace if trace is not None else current.trace,
    )

    return _global_defaults


def _create_or_update_bank(
    hindsight_api_url: str,
    bank_id: str,
    name: Optional[str] = None,
    mission: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """Create or update a memory bank with the given configuration.

    Args:
        hindsight_api_url: URL of the Hindsight API server
        bank_id: The bank ID to create/update
        name: Optional display name for the bank
        mission: Instructions guiding what Hindsight should learn and remember
        verbose: Enable verbose logging
    """
    try:
        from hindsight_client import Hindsight

        client = Hindsight(hindsight_api_url)
        client.create_bank(
            bank_id=bank_id,
            name=name,
            mission=mission,
        )
        if verbose:
            import logging
            logging.getLogger("hindsight_litellm").info(
                f"Created/updated bank '{bank_id}' with mission"
            )
    except ImportError:
        if verbose:
            import logging
            logging.getLogger("hindsight_litellm").warning(
                "hindsight_client not installed. Cannot create bank. "
                "Install with: pip install hindsight-client"
            )
    except Exception as e:
        if verbose:
            import logging
            logging.getLogger("hindsight_litellm").warning(
                f"Failed to create/update bank: {e}"
            )


def get_config() -> Optional[HindsightConfig]:
    """Get the current global static configuration.

    Returns:
        The current HindsightConfig instance, or None if not configured
    """
    return _global_config


def get_defaults() -> Optional[HindsightDefaults]:
    """Get the current global defaults for per-call settings.

    Returns:
        The current HindsightDefaults instance, or None if not set
    """
    return _global_defaults


def is_configured() -> bool:
    """Check if Hindsight has been configured with a valid bank_id.

    Returns:
        True if configure() has been called and a bank_id is set in defaults
    """
    return (
        _global_config is not None
        and _global_defaults is not None
        and _global_defaults.bank_id is not None
    )


def reset_config() -> None:
    """Reset all global configuration to None."""
    global _global_config, _global_defaults
    _global_config = None
    _global_defaults = None


def set_document_id(document_id: str | None) -> None:
    """Set the document_id for grouping stored conversations.

    This is a convenience function that updates just the document_id
    in the defaults without requiring a full set_defaults() call.

    When document_id is set, Hindsight uses upsert behavior:
    - Same document_id = replace previous version
    - Hindsight deduplicates facts automatically

    Args:
        document_id: Document ID for grouping conversations, or None to clear

    Example:
        >>> from hindsight_litellm import configure, set_defaults, enable, set_document_id
        >>> configure(hindsight_api_url="http://localhost:8888")
        >>> set_defaults(bank_id="my-agent")
        >>> enable()
        >>>
        >>> # Start a new conversation
        >>> set_document_id("conversation-123")
        >>> response = litellm.completion(model="gpt-4", messages=[...])
        >>>
        >>> # Switch to another conversation
        >>> set_document_id("conversation-456")
        >>> response = litellm.completion(model="gpt-4", messages=[...])
    """
    global _global_defaults
    if _global_defaults is not None:
        _global_defaults = HindsightDefaults(
            bank_id=_global_defaults.bank_id,
            document_id=document_id,
            budget=_global_defaults.budget,
            fact_types=_global_defaults.fact_types,
            max_memories=_global_defaults.max_memories,
            max_memory_tokens=_global_defaults.max_memory_tokens,
            use_reflect=_global_defaults.use_reflect,
            reflect_include_facts=_global_defaults.reflect_include_facts,
            reflect_context=_global_defaults.reflect_context,
            reflect_response_schema=_global_defaults.reflect_response_schema,
            include_entities=_global_defaults.include_entities,
            trace=_global_defaults.trace,
        )
    else:
        # Create defaults with just document_id if none exist
        _global_defaults = HindsightDefaults(document_id=document_id)


def set_bank_mission(
    bank_id: Optional[str] = None,
    mission: Optional[str] = None,
    name: Optional[str] = None,
) -> None:
    """Set or update the mission for a memory bank.

    The mission guides Hindsight on what information to learn and remember,
    and is used for mental model generation. If the bank doesn't exist,
    it will be auto-created.

    Args:
        bank_id: The bank ID to update. If not provided, uses the default bank_id.
        mission: Instructions guiding what Hindsight should learn and remember.
        name: Optional display name for the bank.

    Raises:
        ValueError: If no bank_id is provided and no default is set.
        RuntimeError: If configure() hasn't been called.

    Example:
        >>> from hindsight_litellm import configure, set_defaults, set_bank_mission
        >>> configure(hindsight_api_url="http://localhost:8888")
        >>> set_defaults(bank_id="delivery-agent")
        >>> set_bank_mission(
        ...     mission="You are a delivery agent navigating a building. "
        ...             "Remember employee locations, building layout, and optimal paths."
        ... )
    """
    config = get_config()
    if not config:
        raise RuntimeError("Hindsight not configured. Call configure() first.")

    # Determine which bank_id to use
    effective_bank_id = bank_id
    if effective_bank_id is None:
        defaults = get_defaults()
        if defaults:
            effective_bank_id = defaults.bank_id

    if not effective_bank_id:
        raise ValueError(
            "No bank_id provided and no default bank_id set. "
            "Either pass bank_id or call set_defaults(bank_id=...) first."
        )

    # Use the Hindsight API to create/update the bank
    _create_or_update_bank(
        hindsight_api_url=config.hindsight_api_url,
        bank_id=effective_bank_id,
        name=name,
        mission=mission,
        verbose=config.verbose,
    )


