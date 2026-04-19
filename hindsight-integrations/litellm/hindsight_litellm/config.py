"""Global configuration for Hindsight-LiteLLM integration.

This module provides a clean API for configuring Hindsight integration:

1. configure() - Connection settings + default per-call settings
   - API URL, authentication, and default values for all per-call settings

2. set_defaults() - Update default values for per-call settings
   - Convenience function to update defaults without reconfiguring connection

3. Per-call kwargs (hindsight_* prefix) - Override any setting per-call
   - hindsight_bank_id, hindsight_budget, hindsight_inject_memories, etc.

4. set_bank_mission() - Set the mission for a memory bank (for mental models)
"""

import os
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from importlib import metadata
from typing import Any, Dict, List, Optional

try:
    _VERSION = metadata.version("hindsight-litellm")
except metadata.PackageNotFoundError:
    _VERSION = "0.0.0"
USER_AGENT = f"hindsight-litellm/{_VERSION}"

# Default Hindsight API URL (production)
DEFAULT_HINDSIGHT_API_URL = "https://api.hindsight.vectorize.io"
DEFAULT_BANK_ID = "default"
HINDSIGHT_API_KEY_ENV = "HINDSIGHT_API_KEY"


class MemoryInjectionMode(str, Enum):
    """How memories should be injected into the prompt.

    Use inject_memories=False if you don't want memory injection.
    """

    SYSTEM_MESSAGE = "system_message"  # Add to/create system message
    PREPEND_USER = "prepend_user"  # Prepend to last user message


@dataclass
class HindsightCallSettings:
    """Unified settings for Hindsight memory operations.

    All fields here can be:
    - Set as defaults via configure() or set_defaults()
    - Overridden per-call via hindsight_* kwargs (e.g., hindsight_bank_id="other")

    To add a new setting, just add a field here - it automatically works everywhere.

    Attributes:
        bank_id: Memory bank ID for operations. Use different bank_ids per user
            for multi-user support (e.g., f"user-{user_id}")
        session_id: Session ID for grouping conversations (maps to Hindsight's
            document_id). Use this to group related messages in a conversation.
            When set, Hindsight uses upsert behavior (same session = replace).
        document_id: DEPRECATED - Use session_id instead. Kept for backward
            compatibility. If both are set, session_id takes precedence.

        store_conversations: Whether to store conversations to Hindsight
        inject_memories: Whether to inject relevant memories into prompts
        injection_mode: How to inject memories (system_message or prepend_user)

        budget: Budget level for memory recall (low, mid, high)
        fact_types: Filter by fact types (world, experience, opinion, observation)
        max_memories: Maximum memories to inject (None = no limit)
        max_memory_tokens: Maximum tokens for memory context
        include_entities: Include entity observations in recall results
        trace: Enable trace info for recall debugging

        tags: Tags to apply when storing conversations. Use for visibility scoping
            (e.g., ["user:alice", "session:123"]). Stored memories will have these tags.
        recall_tags: Tags to filter by when recalling/reflecting memories. Only memories
            matching these tags (based on recall_tags_match mode) will be retrieved.
        recall_tags_match: How to match recall_tags. Options:
            - "any": OR matching, includes untagged memories (default)
            - "all": AND matching, includes untagged memories
            - "any_strict": OR matching, excludes untagged memories
            - "all_strict": AND matching, excludes untagged memories

        use_reflect: Use reflect API instead of recall for memory injection
        reflect_context: Context for reflect reasoning (shapes response, not retrieval)
        reflect_response_schema: JSON Schema for structured reflect output
        reflect_include_facts: Include facts used by reflect in debug info

        query: Custom query for memory recall (if not set, extracts from user message)
        verbose: Enable verbose logging
    """

    # Memory bank settings
    bank_id: Optional[str] = None
    session_id: Optional[str] = None  # Primary - maps to Hindsight's document_id
    document_id: Optional[str] = None  # Deprecated - use session_id instead

    # Feature toggles
    store_conversations: bool = True
    inject_memories: bool = True
    injection_mode: MemoryInjectionMode = MemoryInjectionMode.SYSTEM_MESSAGE

    # Recall settings
    budget: str = "mid"  # low, mid, high
    fact_types: Optional[List[str]] = None  # world, experience, opinion, observation
    max_memories: Optional[int] = None  # None = no limit
    max_memory_tokens: int = 4096
    include_entities: bool = True
    trace: bool = False

    # Tags for visibility scoping
    tags: Optional[List[str]] = None  # Tags applied when storing conversations
    recall_tags: Optional[List[str]] = None  # Tags to filter recall/reflect
    recall_tags_match: str = "any"  # any, all, any_strict, all_strict

    # Reflect settings (alternative to recall)
    use_reflect: bool = False
    reflect_context: Optional[str] = None
    reflect_response_schema: Optional[Dict[str, Any]] = None
    reflect_include_facts: bool = False

    # Query override (if not set, extracts from last user message)
    query: Optional[str] = None

    # Logging
    verbose: bool = False

    @property
    def effective_document_id(self) -> Optional[str]:
        """Get the effective document_id for Hindsight API calls.

        Returns session_id if set, otherwise falls back to document_id.
        This maps to Hindsight's document_id parameter for retain operations.
        """
        return self.session_id if self.session_id is not None else self.document_id


def _merge_call_settings(defaults: HindsightCallSettings, kwargs: Dict[str, Any]) -> HindsightCallSettings:
    """Merge per-call kwargs (hindsight_*) with defaults.

    This automatically handles all fields in HindsightCallSettings.
    When a new field is added to the dataclass, it works here automatically.

    Args:
        defaults: The default settings
        kwargs: The kwargs passed to the call, may contain hindsight_* overrides

    Returns:
        Merged settings with per-call values overriding defaults
    """
    # Start with defaults as dict
    merged = asdict(defaults)

    # Get valid field names from the dataclass
    valid_fields = {f.name for f in fields(HindsightCallSettings)}

    # Override with hindsight_* kwargs
    for key, value in kwargs.items():
        if key.startswith("hindsight_"):
            setting_name = key[len("hindsight_") :]
            if setting_name in valid_fields:
                merged[setting_name] = value

    return HindsightCallSettings(**merged)


# Backward compatibility alias
HindsightDefaults = HindsightCallSettings


@dataclass
class HindsightConfig:
    """Connection-level configuration for Hindsight integration.

    These are settings that require a new client connection to change:
    - API URL and authentication
    - Session-level settings (excluded_models, sync_storage)

    Per-call settings (bank_id, budget, etc.) are in default_settings.

    Attributes:
        hindsight_api_url: URL of the Hindsight API server
        api_key: API key for Hindsight authentication
        excluded_models: List of model patterns to exclude from interception
        sync_storage: If True, storage runs synchronously and raises errors immediately
        default_settings: Default values for all per-call settings
    """

    hindsight_api_url: str = DEFAULT_HINDSIGHT_API_URL
    api_key: Optional[str] = None
    excluded_models: List[str] = field(default_factory=list)
    sync_storage: bool = False
    default_settings: HindsightCallSettings = field(default_factory=HindsightCallSettings)

    # Backward compatibility properties - delegate to default_settings
    @property
    def bank_id(self) -> Optional[str]:
        return self.default_settings.bank_id

    @property
    def store_conversations(self) -> bool:
        return self.default_settings.store_conversations

    @property
    def inject_memories(self) -> bool:
        return self.default_settings.inject_memories

    @property
    def injection_mode(self) -> MemoryInjectionMode:
        return self.default_settings.injection_mode

    @property
    def verbose(self) -> bool:
        return self.default_settings.verbose


# Global instances
_global_config: Optional[HindsightConfig] = None


def configure(
    hindsight_api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    excluded_models: Optional[List[str]] = None,
    sync_storage: bool = False,
    # Bank setup (one-time)
    mission: Optional[str] = None,
    bank_name: Optional[str] = None,
    # Per-call defaults (all HindsightCallSettings fields)
    bank_id: Optional[str] = None,
    session_id: Optional[str] = None,
    document_id: Optional[str] = None,  # Deprecated - use session_id
    store_conversations: bool = True,
    inject_memories: bool = True,
    injection_mode: MemoryInjectionMode = MemoryInjectionMode.SYSTEM_MESSAGE,
    budget: str = "mid",
    fact_types: Optional[List[str]] = None,
    max_memories: Optional[int] = None,
    max_memory_tokens: int = 4096,
    include_entities: bool = True,
    trace: bool = False,
    tags: Optional[List[str]] = None,
    recall_tags: Optional[List[str]] = None,
    recall_tags_match: str = "any",
    use_reflect: bool = False,
    reflect_context: Optional[str] = None,
    reflect_response_schema: Optional[Dict[str, Any]] = None,
    reflect_include_facts: bool = False,
    verbose: bool = False,
) -> HindsightConfig:
    """Configure Hindsight integration settings.

    Sets up connection settings and default values for per-call settings.
    All per-call settings can be overridden using hindsight_* kwargs.

    Args:
        hindsight_api_url: URL of the Hindsight API server
            (default: https://api.hindsight.vectorize.io)
        api_key: API key for Hindsight authentication. If not provided,
            reads from HINDSIGHT_API_KEY environment variable.
        excluded_models: List of model patterns to exclude from interception
        sync_storage: If True, storage runs synchronously and raises errors immediately.
            If False (default), storage runs in background for better performance.
        mission: Instructions guiding what Hindsight should learn and remember
            (used for mental model generation).
        bank_name: Optional display name for the bank.

        # Per-call defaults (can be overridden with hindsight_* kwargs):
        bank_id: Memory bank ID (default: "default"). For multi-user support,
            use different bank_ids per user (e.g., f"user-{user_id}")
        session_id: Session ID for grouping conversations. Maps to Hindsight's
            document_id. When set, enables upsert behavior (same session = replace).
        document_id: DEPRECATED - Use session_id instead.
        store_conversations: Whether to store conversations (default: True)
        inject_memories: Whether to inject memories (default: True)
        injection_mode: How to inject memories (system_message or prepend_user)
        budget: Recall budget level - low/mid/high (default: "mid")
        fact_types: Filter by fact types (world/experience/opinion/observation)
        max_memories: Max memories to inject (None = no limit)
        max_memory_tokens: Max tokens for memory context (default: 4096)
        include_entities: Include entity observations in recall (default: True)
        trace: Enable trace info for debugging (default: False)
        tags: Tags to apply when storing conversations (e.g., ["user:alice"])
        recall_tags: Tags to filter by when recalling/reflecting memories
        recall_tags_match: Tag matching mode - any/all/any_strict/all_strict (default: "any")
        use_reflect: Use reflect API instead of recall (default: False)
        reflect_context: Context for reflect reasoning
        reflect_response_schema: JSON Schema for structured reflect output
        reflect_include_facts: Include facts in reflect debug info (default: False)
        verbose: Enable verbose logging (default: False)

    Returns:
        The configured HindsightConfig instance

    Example:
        >>> from hindsight_litellm import configure, enable
        >>>
        >>> # Minimal usage - just set HINDSIGHT_API_KEY env var
        >>> configure()
        >>> enable()
        >>>
        >>> # With per-call defaults
        >>> configure(
        ...     bank_id="user-123",
        ...     budget="high",
        ...     background="Remember user preferences.",
        ... )
        >>> enable()
        >>>
        >>> # Override per-call:
        >>> response = litellm.completion(
        ...     model="gpt-4",
        ...     messages=[...],
        ...     hindsight_bank_id="other-user",  # Override default
        ... )
    """
    global _global_config

    # Apply connection-level defaults
    resolved_api_url = hindsight_api_url or DEFAULT_HINDSIGHT_API_URL
    resolved_api_key = api_key or os.environ.get(HINDSIGHT_API_KEY_ENV)
    resolved_bank_id = bank_id or DEFAULT_BANK_ID

    # Build default settings
    default_settings = HindsightCallSettings(
        bank_id=resolved_bank_id,
        document_id=document_id,
        session_id=session_id,
        store_conversations=store_conversations,
        inject_memories=inject_memories,
        injection_mode=injection_mode,
        budget=budget,
        fact_types=fact_types,
        max_memories=max_memories,
        max_memory_tokens=max_memory_tokens,
        include_entities=include_entities,
        trace=trace,
        tags=tags,
        recall_tags=recall_tags,
        recall_tags_match=recall_tags_match,
        use_reflect=use_reflect,
        reflect_context=reflect_context,
        reflect_response_schema=reflect_response_schema,
        reflect_include_facts=reflect_include_facts,
        verbose=verbose,
    )

    _global_config = HindsightConfig(
        hindsight_api_url=resolved_api_url,
        api_key=resolved_api_key,
        excluded_models=excluded_models or [],
        sync_storage=sync_storage,
        default_settings=default_settings,
    )

    # If mission or bank_name is provided, create/update the bank
    if mission or bank_name:
        _create_or_update_bank(
            hindsight_api_url=resolved_api_url,
            bank_id=resolved_bank_id,
            name=bank_name,
            mission=mission,
            verbose=verbose,
            api_key=resolved_api_key,
        )

    return _global_config


def set_defaults(
    bank_id: Optional[str] = None,
    session_id: Optional[str] = None,
    document_id: Optional[str] = None,  # Deprecated - use session_id
    store_conversations: Optional[bool] = None,
    inject_memories: Optional[bool] = None,
    injection_mode: Optional[MemoryInjectionMode] = None,
    budget: Optional[str] = None,
    fact_types: Optional[List[str]] = None,
    max_memories: Optional[int] = None,
    max_memory_tokens: Optional[int] = None,
    include_entities: Optional[bool] = None,
    trace: Optional[bool] = None,
    tags: Optional[List[str]] = None,
    recall_tags: Optional[List[str]] = None,
    recall_tags_match: Optional[str] = None,
    use_reflect: Optional[bool] = None,
    reflect_context: Optional[str] = None,
    reflect_response_schema: Optional[Dict[str, Any]] = None,
    reflect_include_facts: Optional[bool] = None,
    verbose: Optional[bool] = None,
) -> HindsightCallSettings:
    """Update default values for per-call settings.

    Updates only the specified fields, preserving other defaults.
    Any of these can be overridden on individual calls using
    hindsight_* kwargs (e.g., hindsight_bank_id="other-bank").

    Args:
        bank_id: Memory bank ID for memory operations
        session_id: Session ID for grouping conversations. Maps to Hindsight's
            document_id. When set, enables upsert behavior (same session = replace).
        document_id: DEPRECATED - Use session_id instead.
        store_conversations: Whether to store conversations
        inject_memories: Whether to inject memories
        injection_mode: How to inject memories (system_message or prepend_user)
        budget: Budget level for memory recall (low, mid, high)
        fact_types: Fact types to filter (world, experience, opinion, observation)
        max_memories: Max number of memories to inject
        max_memory_tokens: Max tokens for memory context
        include_entities: Include entity observations in recall
        trace: Enable trace info for debugging
        tags: Tags to apply when storing conversations
        recall_tags: Tags to filter by when recalling/reflecting memories
        recall_tags_match: Tag matching mode - any/all/any_strict/all_strict
        use_reflect: Use reflect API instead of recall
        reflect_context: Context for reflect reasoning
        reflect_response_schema: JSON Schema for structured reflect output
        reflect_include_facts: Include facts in reflect debug info
        verbose: Enable verbose logging

    Returns:
        The updated HindsightCallSettings instance

    Example:
        >>> from hindsight_litellm import configure, set_defaults
        >>> configure()
        >>> set_defaults(bank_id="my-agent", budget="high")
    """
    global _global_config

    # Ensure configure() was called
    if _global_config is None:
        # Auto-configure with defaults if not configured
        configure()

    # Get current defaults
    current = _global_config.default_settings

    # Update only provided values using dataclass fields
    updated_settings = HindsightCallSettings(
        bank_id=bank_id if bank_id is not None else current.bank_id,
        document_id=document_id if document_id is not None else current.document_id,
        session_id=session_id if session_id is not None else current.session_id,
        store_conversations=store_conversations if store_conversations is not None else current.store_conversations,
        inject_memories=inject_memories if inject_memories is not None else current.inject_memories,
        injection_mode=injection_mode if injection_mode is not None else current.injection_mode,
        budget=budget if budget is not None else current.budget,
        fact_types=fact_types if fact_types is not None else current.fact_types,
        max_memories=max_memories if max_memories is not None else current.max_memories,
        max_memory_tokens=max_memory_tokens if max_memory_tokens is not None else current.max_memory_tokens,
        include_entities=include_entities if include_entities is not None else current.include_entities,
        trace=trace if trace is not None else current.trace,
        tags=tags if tags is not None else current.tags,
        recall_tags=recall_tags if recall_tags is not None else current.recall_tags,
        recall_tags_match=recall_tags_match if recall_tags_match is not None else current.recall_tags_match,
        use_reflect=use_reflect if use_reflect is not None else current.use_reflect,
        reflect_context=reflect_context if reflect_context is not None else current.reflect_context,
        reflect_response_schema=reflect_response_schema
        if reflect_response_schema is not None
        else current.reflect_response_schema,
        reflect_include_facts=reflect_include_facts
        if reflect_include_facts is not None
        else current.reflect_include_facts,
        verbose=verbose if verbose is not None else current.verbose,
    )

    # Update the config's default settings
    _global_config = HindsightConfig(
        hindsight_api_url=_global_config.hindsight_api_url,
        api_key=_global_config.api_key,
        excluded_models=_global_config.excluded_models,
        sync_storage=_global_config.sync_storage,
        default_settings=updated_settings,
    )

    return updated_settings


def _create_or_update_bank(
    hindsight_api_url: str,
    bank_id: str,
    name: Optional[str] = None,
    mission: Optional[str] = None,
    verbose: bool = False,
    api_key: Optional[str] = None,
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

        client = Hindsight(base_url=hindsight_api_url, api_key=api_key, user_agent=USER_AGENT)
        client.create_bank(
            bank_id=bank_id,
            name=name,
            mission=mission,
        )
        if verbose:
            import logging

            logging.getLogger("hindsight_litellm").info(f"Created/updated bank '{bank_id}' with mission")
    except ImportError:
        if verbose:
            import logging

            logging.getLogger("hindsight_litellm").warning(
                "hindsight_client not installed. Cannot create bank. Install with: pip install hindsight-client"
            )
    except Exception as e:
        if verbose:
            import logging

            logging.getLogger("hindsight_litellm").warning(f"Failed to create/update bank: {e}")


def get_config() -> Optional[HindsightConfig]:
    """Get the current global configuration.

    Returns:
        The current HindsightConfig instance, or None if not configured
    """
    return _global_config


def get_defaults() -> Optional[HindsightCallSettings]:
    """Get the current global defaults for per-call settings.

    Returns:
        The current HindsightCallSettings instance, or None if not configured
    """
    if _global_config is not None:
        return _global_config.default_settings
    return None


def is_configured() -> bool:
    """Check if Hindsight has been configured with a valid bank_id.

    Returns:
        True if configure() has been called and a bank_id is set
    """
    if _global_config is not None and _global_config.bank_id:
        return True
    return False


def reset_config() -> None:
    """Reset all global configuration to None."""
    global _global_config
    _global_config = None
