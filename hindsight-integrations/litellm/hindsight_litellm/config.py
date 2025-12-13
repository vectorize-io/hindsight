"""Global configuration for Hindsight-LiteLLM integration."""

from typing import Optional, List
from dataclasses import dataclass, field
from enum import Enum


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
        bank_id: Memory bank ID for memory operations (required). For multi-user
            support, use different bank_ids per user (e.g., f"user-{user_id}")
        api_key: Optional API key for Hindsight authentication
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
        bank_name: Optional display name for the memory bank
        background: Optional background/instructions for memory extraction
        use_reflect: Use reflect API instead of recall for memory injection (synthesizes answer)
    """

    hindsight_api_url: str = "http://localhost:8888"
    bank_id: Optional[str] = None
    api_key: Optional[str] = None
    store_conversations: bool = True
    inject_memories: bool = True
    injection_mode: MemoryInjectionMode = MemoryInjectionMode.SYSTEM_MESSAGE
    max_memories: Optional[int] = None  # None = no limit (use all results from API)
    max_memory_tokens: int = 4096
    recall_budget: str = "mid"  # low, mid, high
    fact_types: Optional[List[str]] = None  # world, agent, opinion, observation
    document_id: Optional[str] = None
    enabled: bool = True
    excluded_models: List[str] = field(default_factory=list)
    verbose: bool = False
    bank_name: Optional[str] = None  # Display name for the memory bank
    background: Optional[str] = None  # Background/instructions for memory extraction
    use_reflect: bool = False  # Use reflect instead of recall for memory injection
    reflect_include_facts: bool = False  # Include facts used by reflect in debug info


# Global configuration instance
_global_config: Optional[HindsightConfig] = None


def configure(
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
    enabled: bool = True,
    excluded_models: Optional[List[str]] = None,
    verbose: bool = False,
    bank_name: Optional[str] = None,
    background: Optional[str] = None,
    use_reflect: bool = False,
    reflect_include_facts: bool = False,
) -> HindsightConfig:
    """Configure global Hindsight integration settings for LiteLLM.

    This function sets up the global configuration that will be used by the
    LiteLLM callbacks to inject memories and store conversations.

    Args:
        hindsight_api_url: URL of the Hindsight API server
        bank_id: Memory bank ID for memory operations (required). For multi-user
            support, use different bank_ids per user (e.g., f"user-{user_id}")
        api_key: Optional API key for Hindsight authentication
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
        bank_name: Optional display name for the memory bank
        background: Optional background/instructions that help Hindsight understand
            what information is important to extract and remember from conversations.
            This is passed to create_bank() to configure the memory bank.
        use_reflect: Use reflect API instead of recall for memory injection.
            When True, Hindsight will synthesize a contextual answer based on
            memories rather than returning raw memory facts.
        reflect_include_facts: When use_reflect=True, include the facts that
            were used to generate the reflect response in the debug info.
            This is useful for debugging what memories the reflect API used.

    Returns:
        The configured HindsightConfig instance

    Example:
        >>> from hindsight_litellm import configure, enable
        >>> configure(
        ...     hindsight_api_url="http://localhost:8888",
        ...     bank_id="user-123",  # Per-user bank for multi-user support
        ...     store_conversations=True,
        ...     inject_memories=True,
        ...     background="This agent routes customer requests to support channels. "
        ...                "Remember which types of issues should go to which channels.",
        ... )
        >>> enable()  # Register callbacks with LiteLLM
    """
    global _global_config

    _global_config = HindsightConfig(
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
        enabled=enabled,
        excluded_models=excluded_models or [],
        verbose=verbose,
        bank_name=bank_name,
        background=background,
        use_reflect=use_reflect,
        reflect_include_facts=reflect_include_facts,
    )

    # If background or bank_name is provided, create/update the bank
    if bank_id and (background or bank_name):
        _create_or_update_bank(
            hindsight_api_url=hindsight_api_url,
            bank_id=bank_id,
            name=bank_name,
            background=background,
            verbose=verbose,
        )

    return _global_config


def _create_or_update_bank(
    hindsight_api_url: str,
    bank_id: str,
    name: Optional[str] = None,
    background: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """Create or update a memory bank with the given configuration.

    This is called automatically by configure() when background or bank_name is provided.
    """
    try:
        from hindsight_client import Hindsight

        client = Hindsight(hindsight_api_url)
        client.create_bank(
            bank_id=bank_id,
            name=name,
            background=background,
        )
        if verbose:
            import logging
            logging.getLogger("hindsight_litellm").info(
                f"Created/updated bank '{bank_id}' with background"
            )
    except ImportError:
        if verbose:
            import logging
            logging.getLogger("hindsight_litellm").warning(
                "hindsight_client not installed. Cannot create bank with background. "
                "Install with: pip install hindsight-client"
            )
    except Exception as e:
        if verbose:
            import logging
            logging.getLogger("hindsight_litellm").warning(
                f"Failed to create/update bank: {e}"
            )


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
