"""Global configuration for Hindsight-LiteLLM integration."""

from typing import Optional, List
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4


class MemoryInjectionMode(str, Enum):
    """How memories should be injected into the prompt."""
    SYSTEM_MESSAGE = "system_message"  # Add as system message
    PREPEND_USER = "prepend_user"  # Prepend to user message
    DISABLED = "disabled"  # Don't inject memories


@dataclass
class HindsightConfig:
    """Configuration for Hindsight integration with LiteLLM.

    Attributes:
        hindsight_api_url: URL of the Hindsight API server
        bank_id: Memory bank ID for memory operations (required)
        api_key: Optional API key for Hindsight authentication
        entity_id: User/entity identifier for memory scoping (multi-user support)
        session_id: Session identifier for conversation grouping
        store_conversations: Whether to store conversations to Hindsight
        inject_memories: Whether to inject relevant memories into prompts
        injection_mode: How to inject memories (system_message or prepend_user)
        max_memories: Maximum number of memories to inject
        max_memory_tokens: Maximum tokens for injected memory context
        recall_budget: Budget level for memory recall (low, mid, high)
        fact_types: List of fact types to filter recall (world, agent, opinion, observation)
        document_id: Optional document ID for grouping stored conversations
        enabled: Master switch to enable/disable Hindsight integration
        excluded_models: List of model patterns to exclude from interception
        verbose: Enable verbose logging
    """

    hindsight_api_url: str = "http://localhost:8888"
    bank_id: Optional[str] = None
    api_key: Optional[str] = None
    entity_id: Optional[str] = None  # User identifier for multi-user memory isolation
    session_id: Optional[str] = None  # Session identifier for conversation grouping
    store_conversations: bool = True
    inject_memories: bool = True
    injection_mode: MemoryInjectionMode = MemoryInjectionMode.SYSTEM_MESSAGE
    max_memories: int = 10
    max_memory_tokens: int = 2000
    recall_budget: str = "mid"  # low, mid, high
    fact_types: Optional[List[str]] = None  # world, agent, opinion, observation
    document_id: Optional[str] = None
    enabled: bool = True
    excluded_models: List[str] = field(default_factory=list)
    verbose: bool = False


# Global configuration instance
_global_config: Optional[HindsightConfig] = None


def configure(
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
    enabled: bool = True,
    excluded_models: Optional[List[str]] = None,
    verbose: bool = False,
) -> HindsightConfig:
    """Configure global Hindsight integration settings for LiteLLM.

    This function sets up the global configuration that will be used by the
    LiteLLM callbacks to inject memories and store conversations.

    Args:
        hindsight_api_url: URL of the Hindsight API server
        bank_id: Memory bank ID for memory operations (required)
        api_key: Optional API key for Hindsight authentication
        entity_id: User/entity identifier for multi-user memory isolation
        session_id: Session identifier for conversation grouping
        store_conversations: Whether to store conversations to Hindsight
        inject_memories: Whether to inject relevant memories into prompts
        injection_mode: How to inject memories into the prompt
        max_memories: Maximum number of memories to inject
        max_memory_tokens: Maximum tokens for injected memory context
        recall_budget: Budget level for memory recall (low, mid, high)
        fact_types: List of fact types to filter (world, agent, opinion, observation)
        document_id: Optional document ID for grouping stored conversations
        enabled: Master switch to enable/disable Hindsight integration
        excluded_models: List of model patterns to exclude from interception
        verbose: Enable verbose logging

    Returns:
        The configured HindsightConfig instance

    Example:
        >>> from hindsight_litellm import configure, enable
        >>> configure(
        ...     hindsight_api_url="http://localhost:8888",
        ...     bank_id="my-agent",
        ...     entity_id="user-123",  # Multi-user support
        ...     store_conversations=True,
        ...     inject_memories=True,
        ... )
        >>> enable()  # Register callbacks with LiteLLM
    """
    global _global_config

    _global_config = HindsightConfig(
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
        enabled=enabled,
        excluded_models=excluded_models or [],
        verbose=verbose,
    )

    return _global_config


def get_config() -> Optional[HindsightConfig]:
    """Get the current global configuration.

    Returns:
        The current HindsightConfig instance, or None if not configured
    """
    return _global_config


def is_configured() -> bool:
    """Check if Hindsight has been configured.

    Returns:
        True if configure() has been called with a valid bank_id
    """
    return (
        _global_config is not None
        and _global_config.enabled
        and _global_config.bank_id is not None
    )


def reset_config() -> None:
    """Reset the global configuration to None."""
    global _global_config
    _global_config = None


def new_session() -> str:
    """Generate and set a new session ID.

    Creates a new UUID-based session ID and updates the global config.
    This is useful for starting fresh conversation threads.

    Returns:
        The new session ID string

    Raises:
        RuntimeError: If Hindsight has not been configured

    Example:
        >>> from hindsight_litellm import configure, new_session
        >>> configure(bank_id="my-agent")
        >>> session_id = new_session()
        >>> print(f"Started new session: {session_id}")
    """
    global _global_config

    if _global_config is None:
        raise RuntimeError(
            "Hindsight not configured. Call configure() before new_session()."
        )

    new_id = str(uuid4())
    _global_config.session_id = new_id
    return new_id


def set_session(session_id: str) -> None:
    """Set a specific session ID.

    Use this to resume a previous conversation session.

    Args:
        session_id: The session ID to set

    Raises:
        RuntimeError: If Hindsight has not been configured

    Example:
        >>> from hindsight_litellm import configure, set_session
        >>> configure(bank_id="my-agent")
        >>> set_session("previous-session-id")  # Resume conversation
    """
    global _global_config

    if _global_config is None:
        raise RuntimeError(
            "Hindsight not configured. Call configure() before set_session()."
        )

    _global_config.session_id = session_id


def get_session() -> Optional[str]:
    """Get the current session ID.

    Returns:
        The current session ID, or None if not set
    """
    if _global_config is None:
        return None
    return _global_config.session_id


def set_entity(entity_id: str) -> None:
    """Set the entity ID for multi-user memory isolation.

    Args:
        entity_id: The entity/user identifier

    Raises:
        RuntimeError: If Hindsight has not been configured

    Example:
        >>> from hindsight_litellm import configure, set_entity
        >>> configure(bank_id="my-agent")
        >>> set_entity("user-123")  # Switch to this user's memories
    """
    global _global_config

    if _global_config is None:
        raise RuntimeError(
            "Hindsight not configured. Call configure() before set_entity()."
        )

    _global_config.entity_id = entity_id


def get_entity() -> Optional[str]:
    """Get the current entity ID.

    Returns:
        The current entity ID, or None if not set
    """
    if _global_config is None:
        return None
    return _global_config.entity_id
