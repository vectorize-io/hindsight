"""Shared MCP tool implementations for Hindsight.

This module provides the core tool logic used by both:
- mcp_local.py (stdio transport for Claude Code)
- api/mcp.py (HTTP transport for API server)
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Literal, get_args

from fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

from hindsight_api import MemoryEngine
from hindsight_api.api import page_markdown
from hindsight_api.config import (
    DEFAULT_MCP_RECALL_DESCRIPTION,
    DEFAULT_MCP_RETAIN_DESCRIPTION,
)
from hindsight_api.engine.audit import AuditEntry, AuditLogger
from hindsight_api.engine.memory_engine import KEEP_PARENT, Budget
from hindsight_api.engine.response_models import VALID_RECALL_FACT_TYPES, MinScores, TemporalWindow
from hindsight_api.engine.search.tags import TagGroup, TagsMatch
from hindsight_api.extensions import OperationValidationError
from hindsight_api.models import RequestContext

_TAG_GROUP_LIST_ADAPTER = TypeAdapter(list[TagGroup])

# All tools available in the system (explicit list — no wildcards).
# Defined here (shared module) to avoid circular imports with api/mcp.py.
_ALL_TOOLS: frozenset[str] = frozenset(
    {
        "retain",
        "sync_retain",
        "recall",
        "reflect",
        "list_banks",
        "create_bank",
        "list_mental_models",
        "get_mental_model",
        "create_mental_model",
        "update_mental_model",
        "delete_mental_model",
        "refresh_mental_model",
        "clear_mental_model",
        "list_directives",
        "create_directive",
        "delete_directive",
        "list_memories",
        "get_memory",
        "update_memory",
        "invalidate_memory",
        "list_documents",
        "get_document",
        "delete_document",
        "list_operations",
        "get_operation",
        "cancel_operation",
        "list_tags",
        "get_bank",
        "get_bank_stats",
        "update_bank",
        "delete_bank",
        "clear_memories",
        "get_knowledge_base_tree",
        "search_knowledge_base",
        "get_knowledge_page",
        "create_knowledge_folder",
        "create_knowledge_page",
        "update_knowledge_node",
        "delete_knowledge_node",
    }
)

logger = logging.getLogger(__name__)


class MentalModelTriggerInput(BaseModel):
    """The refresh policy of a mental model or knowledge page, as an MCP tool input.

    Mirrors the HTTP ``MentalModelTrigger`` field for field — an agent that read
    the API docs must not have a call rejected for naming a setting that exists.
    ``tests/test_mcp_tools.py::test_trigger_input_covers_every_http_trigger_field``
    fails if the two drift apart.

    The one deliberate difference is that every field is optional with no default:
    the HTTP model fills unset fields with its own defaults, which makes a partial
    trigger silently reset the rest, while these tools send only what the caller
    actually set (``model_dump(exclude_unset=True)``) and the engine merges that
    over the stored trigger. Passing an explicit ``null`` still clears a setting.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["full", "delta"] | None = Field(
        default=None,
        description=(
            "Refresh mode. 'full' regenerates the content from scratch on each refresh; 'delta' makes "
            "surgical edits to the existing content, preserving unchanged sections byte-for-byte. Delta "
            "falls back to a full regeneration when there is no existing content or the source_query changed."
        ),
    )
    refresh_after_consolidation: bool | None = Field(
        default=None,
        description="Refresh automatically after observations are consolidated. Mutually exclusive with refresh_cron.",
    )
    refresh_cron: str | None = Field(
        default=None,
        description=(
            "UTC five-field cron schedule, e.g. '0 3 * * *' for daily at 03:00 UTC. A scheduled refresh runs "
            "only when the model is stale, so an unchanged scope costs no LLM call. Mutually exclusive with "
            "refresh_after_consolidation. null = no schedule."
        ),
    )
    min_refresh_interval_seconds: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Minimum seconds between two AUTOMATIC refreshes. A trigger that arrives sooner is queued and "
            "parked until the window expires, and further triggers fold into that one queued refresh, so a "
            "burst of retains costs one refresh. Explicit refreshes ignore it. 0 disables the floor; null "
            "falls back to the bank/global setting."
        ),
    )
    fact_types: list[Literal["world", "experience", "observation"]] | None = Field(
        default=None,
        description="Fact types to retrieve during refresh; null includes all of world, experience and observation.",
    )
    exclude_mental_models: bool | None = Field(
        default=None,
        description="Exclude ALL mental models from the refresh's reflect loop, so a model never reflects on its siblings.",
    )
    exclude_mental_model_ids: list[str] | None = Field(
        default=None, description="Exclude specific mental models from the refresh's reflect loop, by ID."
    )
    tags_match: TagsMatch | None = Field(
        default=None,
        description=(
            "How this model's tags select memories during refresh: any, all, any_strict, all_strict, or exact. "
            "Unset means 'all_strict' for a tagged model and 'any' for an untagged one."
        ),
    )
    tag_groups: list[TagGroup] | None = Field(
        default=None,
        description=(
            "Compound boolean tag expressions (nested and/or/not) used during refresh INSTEAD of the model's "
            "flat tags. When set, the model's own tags are not used for filtering."
        ),
    )
    include_chunks: bool | None = Field(
        default=None,
        description="Override whether the refresh's internal recall returns raw chunk text. null = bank/global default.",
    )
    recall_max_tokens: int | None = Field(
        default=None,
        description="Override the token budget for facts from the refresh's internal recall. null = bank/global default.",
    )
    recall_chunks_max_tokens: int | None = Field(
        default=None,
        description="Override the token budget for raw chunks from the refresh's internal recall. null = bank/global default.",
    )
    response_schema: dict | None = Field(
        default=None,
        description=(
            "JSON Schema for structured output. Each refresh then also stores a parsed result under "
            "reflect_response.structured_output, alongside the markdown content."
        ),
    )
    keep_trace: bool | None = Field(
        default=None,
        description=(
            "Record how each refresh reached its result under reflect_response.trace (mode and why, resolved "
            "scope and window, facts retrieved vs used, tool and LLM calls, delta operations). Only the latest "
            "refresh's trace is kept. This is the only way to diagnose a cron- or consolidation-driven refresh, "
            "since nobody watches those run."
        ),
    )

    @field_validator("refresh_cron")
    @classmethod
    def validate_refresh_cron(cls, value: str | None) -> str | None:
        if value is None:
            return None
        from croniter import croniter

        # An empty string reads as "no schedule", not as a malformed one — same
        # normalisation the HTTP model does, so the two accept the same input.
        value = value.strip()
        if not value:
            return None
        if not croniter.is_valid(value):
            raise ValueError(f"refresh_cron is not a valid cron expression: {value!r}")
        return value

    @field_validator("fact_types")
    @classmethod
    def validate_fact_types(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and not value:
            raise ValueError("fact_types must not be empty; use null to include all fact types")
        return value

    @model_validator(mode="after")
    def validate_refresh_exclusivity(self) -> "MentalModelTriggerInput":
        if self.refresh_after_consolidation and self.refresh_cron:
            raise ValueError(
                "refresh_after_consolidation and refresh_cron are mutually exclusive: "
                "a mental model refreshes either after consolidation or on a cron schedule, not both."
            )
        return self


def _mental_model_trigger_patch(
    trigger: MentalModelTriggerInput | None,
    *,
    tags_match: str | None = None,
    refresh_after_consolidation: bool | None = None,
) -> dict[str, Any] | None:
    """The trigger patch to send down, folding in the legacy flat MCP arguments.

    ``tags_match`` and ``trigger_refresh_after_consolidation`` predate the trigger
    object and stay accepted as shorthands. Passing a shorthand AND the same field
    inside ``trigger`` is a contradiction the caller has to resolve rather than one
    of the two silently winning.
    """
    patch = trigger.model_dump(exclude_unset=True) if trigger is not None else {}
    for field, param, legacy_value in (
        ("tags_match", "tags_match", tags_match),
        ("refresh_after_consolidation", "trigger_refresh_after_consolidation", refresh_after_consolidation),
    ):
        if legacy_value is None:
            continue
        if field in patch and patch[field] != legacy_value:
            raise ValueError(
                f"trigger.{field}={patch[field]!r} conflicts with {param}={legacy_value!r}; set it in one place"
            )
        patch[field] = legacy_value
    if patch.get("refresh_after_consolidation") and patch.get("refresh_cron"):
        raise ValueError(
            "refresh_after_consolidation and refresh_cron are mutually exclusive: "
            "a mental model refreshes either after consolidation or on a cron schedule, not both."
        )
    return patch or None


@dataclass
class MCPToolsConfig:
    """Configuration for MCP tools registration."""

    # How to resolve bank_id for operations
    bank_id_resolver: Callable[[], str | None]

    # How to resolve API key for tenant auth (optional)
    api_key_resolver: Callable[[], str | None] | None = None

    # How to resolve tenant_id for usage metering (set by MCP middleware after auth)
    tenant_id_resolver: Callable[[], str | None] | None = None

    # How to resolve api_key_id for usage metering (set by MCP middleware after auth)
    api_key_id_resolver: Callable[[], str | None] | None = None

    # How to resolve mcp_authenticated flag (set when MCP_AUTH_TOKEN validates)
    mcp_authenticated_resolver: Callable[[], bool] | None = None

    # Whether to include bank_id as a parameter on tools (for multi-bank support)
    include_bank_id_param: bool = False

    # Which tools to register
    tools: set[str] | None = None  # None means all tools

    # Custom descriptions (if None, uses defaults)
    retain_description: str | None = None
    recall_description: str | None = None

    # How to resolve the allowlisted passthrough headers (set by MCP middleware).
    # Appended last so existing positional construction keeps its meaning.
    extra_headers_resolver: Callable[[], dict[str, str]] | None = None

    # Retain behavior


def _get_request_context(config: MCPToolsConfig) -> RequestContext:
    """Create RequestContext with auth details from resolvers.

    This enables tenant auth and usage metering to work with MCP tools by propagating
    the authentication results from the MCP middleware to the memory engine.
    """
    api_key = config.api_key_resolver() if config.api_key_resolver else None
    tenant_id = config.tenant_id_resolver() if config.tenant_id_resolver else None
    api_key_id = config.api_key_id_resolver() if config.api_key_id_resolver else None
    mcp_authenticated = config.mcp_authenticated_resolver() if config.mcp_authenticated_resolver else False
    extra_headers = config.extra_headers_resolver() if config.extra_headers_resolver else {}
    return RequestContext(
        api_key=api_key,
        tenant_id=tenant_id,
        api_key_id=api_key_id,
        mcp_authenticated=mcp_authenticated,
        extra_headers=extra_headers,
    )


def parse_timestamp(timestamp: str) -> datetime | None:
    """Parse an ISO format timestamp string.

    Args:
        timestamp: ISO format timestamp (e.g., '2024-01-15T10:30:00Z')

    Returns:
        Parsed datetime or None if invalid

    Raises:
        ValueError: If timestamp format is invalid
    """
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as e:
        raise ValueError(
            f"Invalid timestamp format '{timestamp}'. "
            "Expected ISO format like '2024-01-15T10:30:00' or '2024-01-15T10:30:00Z'"
        ) from e


def build_content_dict(
    content: str,
    context: str,
    timestamp: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, str] | None = None,
    document_id: str | None = None,
    strategy: str | None = None,
    update_mode: str | None = None,
) -> tuple[dict[str, Any], str | None]:
    """Build a content dict for retain operations.

    Args:
        content: The memory content
        context: Category for the memory
        timestamp: Optional ISO timestamp
        tags: Optional tags for scoped visibility filtering
        metadata: Optional key-value metadata to attach to the memory
        document_id: Optional document ID to associate the memory with
        strategy: Optional named retain strategy override (e.g., 'exact', 'verbose')
        update_mode: How to handle existing documents ('replace' or 'append')

    Returns:
        Tuple of (content_dict, error_message). error_message is None if successful.
    """
    # Coerce tags from JSON string to list if needed.
    # MCP tool bridges sometimes serialize JSON arrays as strings during
    # transport, e.g. '["a", "b"]' instead of ["a", "b"].
    if isinstance(tags, str):
        try:
            parsed = json.loads(tags)
            if isinstance(parsed, list):
                tags = parsed
        except (json.JSONDecodeError, TypeError):
            pass
        if isinstance(tags, str):
            tags = [tags]

    content_dict: dict[str, Any] = {"content": content, "context": context}

    if timestamp:
        try:
            parsed_timestamp = parse_timestamp(timestamp)
            content_dict["event_date"] = parsed_timestamp
        except ValueError as e:
            return {}, str(e)

    if tags is not None:
        content_dict["tags"] = tags
    if metadata is not None:
        content_dict["metadata"] = metadata
    if document_id is not None:
        content_dict["document_id"] = document_id
    if strategy is not None:
        content_dict["strategy"] = strategy
    if update_mode is not None:
        content_dict["update_mode"] = update_mode

    return content_dict, None


# MCP tool annotations. Hindsight is a closed memory store (no open-world / internet
# access), so openWorldHint=False throughout. readOnlyHint lets clients group and
# auto-approve safe reads; destructiveHint flags tools that delete or clear memory.
_READ_ONLY_TOOLS = {
    "recall",
    "reflect",
    "list_banks",
    "get_bank",
    "get_bank_stats",
    "list_mental_models",
    "get_mental_model",
    "list_directives",
    "list_memories",
    "get_memory",
    "list_documents",
    "get_document",
    "list_operations",
    "get_operation",
    "list_tags",
    "get_knowledge_base_tree",
    "search_knowledge_base",
    "get_knowledge_page",
}
_DESTRUCTIVE_TOOLS = {
    "delete_bank",
    "clear_memories",
    "clear_mental_model",
    "delete_mental_model",
    "delete_directive",
    "delete_document",
    "invalidate_memory",
    "delete_knowledge_node",
}


def _tool_annotations(name: str) -> ToolAnnotations:
    if name in _READ_ONLY_TOOLS:
        return ToolAnnotations(readOnlyHint=True, openWorldHint=False)
    if name in _DESTRUCTIVE_TOOLS:
        return ToolAnnotations(readOnlyHint=False, destructiveHint=True, openWorldHint=False)
    # Everything else writes but does not destructively delete/clear memory
    # (retain, create_*, update_*, refresh_mental_model, cancel_operation).
    return ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)


def register_mcp_tools(
    mcp: FastMCP,
    memory: MemoryEngine,
    config: MCPToolsConfig,
) -> None:
    """Register MCP tools on a FastMCP server.

    Args:
        mcp: FastMCP server instance
        memory: MemoryEngine instance
        config: Tool configuration
    """
    tools_to_register = config.tools or {
        "retain",
        "sync_retain",
        "recall",
        "reflect",
        "list_banks",
        "create_bank",
        "list_mental_models",
        "get_mental_model",
        "create_mental_model",
        "update_mental_model",
        "delete_mental_model",
        "refresh_mental_model",
        "clear_mental_model",
        "list_directives",
        "create_directive",
        "delete_directive",
        "list_memories",
        "get_memory",
        "update_memory",
        "invalidate_memory",
        "list_documents",
        "get_document",
        "delete_document",
        "list_operations",
        "get_operation",
        "cancel_operation",
        "list_tags",
        "get_bank",
        "get_bank_stats",
        "update_bank",
        "delete_bank",
        "clear_memories",
        "get_knowledge_base_tree",
        "search_knowledge_base",
        "get_knowledge_page",
        "create_knowledge_folder",
        "create_knowledge_page",
        "update_knowledge_node",
        "delete_knowledge_node",
    }

    if "retain" in tools_to_register:
        _register_retain(mcp, memory, config)

    if "sync_retain" in tools_to_register:
        _register_sync_retain(mcp, memory, config)

    if "recall" in tools_to_register:
        _register_recall(mcp, memory, config)

    if "reflect" in tools_to_register:
        _register_reflect(mcp, memory, config)

    if "list_banks" in tools_to_register:
        _register_list_banks(mcp, memory, config)

    if "create_bank" in tools_to_register:
        _register_create_bank(mcp, memory, config)

    # Mental model tools
    if "list_mental_models" in tools_to_register:
        _register_list_mental_models(mcp, memory, config)

    if "get_mental_model" in tools_to_register:
        _register_get_mental_model(mcp, memory, config)

    if "create_mental_model" in tools_to_register:
        _register_create_mental_model(mcp, memory, config)

    if "update_mental_model" in tools_to_register:
        _register_update_mental_model(mcp, memory, config)

    if "delete_mental_model" in tools_to_register:
        _register_delete_mental_model(mcp, memory, config)

    if "refresh_mental_model" in tools_to_register:
        _register_refresh_mental_model(mcp, memory, config)

    if "clear_mental_model" in tools_to_register:
        _register_clear_mental_model(mcp, memory, config)

    # Directive tools
    if "list_directives" in tools_to_register:
        _register_list_directives(mcp, memory, config)

    if "create_directive" in tools_to_register:
        _register_create_directive(mcp, memory, config)

    if "delete_directive" in tools_to_register:
        _register_delete_directive(mcp, memory, config)

    # Memory browsing tools
    if "list_memories" in tools_to_register:
        _register_list_memories(mcp, memory, config)

    if "get_memory" in tools_to_register:
        _register_get_memory(mcp, memory, config)

    if "update_memory" in tools_to_register:
        _register_update_memory(mcp, memory, config)

    if "invalidate_memory" in tools_to_register:
        _register_invalidate_memory(mcp, memory, config)

    # Document tools
    if "list_documents" in tools_to_register:
        _register_list_documents(mcp, memory, config)

    if "get_document" in tools_to_register:
        _register_get_document(mcp, memory, config)

    if "delete_document" in tools_to_register:
        _register_delete_document(mcp, memory, config)

    # Operation tools
    if "list_operations" in tools_to_register:
        _register_list_operations(mcp, memory, config)

    if "get_operation" in tools_to_register:
        _register_get_operation(mcp, memory, config)

    if "cancel_operation" in tools_to_register:
        _register_cancel_operation(mcp, memory, config)

    # Tags & bank tools
    if "list_tags" in tools_to_register:
        _register_list_tags(mcp, memory, config)

    if "get_bank" in tools_to_register:
        _register_get_bank(mcp, memory, config)

    if "get_bank_stats" in tools_to_register:
        _register_get_bank_stats(mcp, memory, config)

    if "update_bank" in tools_to_register:
        _register_update_bank(mcp, memory, config)

    if "delete_bank" in tools_to_register:
        _register_delete_bank(mcp, memory, config)

    if "clear_memories" in tools_to_register:
        _register_clear_memories(mcp, memory, config)

    # Knowledge base tools
    if "get_knowledge_base_tree" in tools_to_register:
        _register_get_knowledge_base_tree(mcp, memory, config)

    if "search_knowledge_base" in tools_to_register:
        _register_search_knowledge_base(mcp, memory, config)

    if "get_knowledge_page" in tools_to_register:
        _register_get_knowledge_page(mcp, memory, config)

    if "create_knowledge_folder" in tools_to_register:
        _register_create_knowledge_folder(mcp, memory, config)

    if "create_knowledge_page" in tools_to_register:
        _register_create_knowledge_page(mcp, memory, config)

    if "update_knowledge_node" in tools_to_register:
        _register_update_knowledge_node(mcp, memory, config)

    if "delete_knowledge_node" in tools_to_register:
        _register_delete_knowledge_node(mcp, memory, config)

    _apply_bank_tool_filtering(mcp, memory, config)
    _apply_audit_logging(mcp, memory, config)


def _apply_bank_tool_filtering(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Filter bank-level mcp_enabled_tools from both tools/list and tool invocation.

    Compatible with FastMCP 2.x (_tool_manager pattern) and 3.x (provider pattern).
    """

    async def _get_enabled_tools() -> set[str] | None:
        """Return the enabled tool set for the current bank, or None if unrestricted."""
        bank_id = config.bank_id_resolver()
        if not bank_id:
            return None
        request_context = _get_request_context(config)

        # Layer 1: bank config filter (existing)
        bank_cfg = await memory._config_resolver.get_bank_config(bank_id, request_context)
        bank_tools: list[str] | None = bank_cfg.get("mcp_enabled_tools")
        enabled: set[str] | None = set(bank_tools) if bank_tools is not None else None

        # Layer 2: operation validator filter
        validator = memory._operation_validator
        if validator is not None:
            candidate = frozenset(enabled) if enabled is not None else _ALL_TOOLS
            try:
                filtered = await validator.filter_mcp_tools(bank_id, request_context, candidate)
            except Exception:
                logger.warning("filter_mcp_tools raised, returning unfiltered tools", exc_info=True)
                return enabled
            if filtered != candidate:
                # Validator can only narrow, never expand beyond the bank config ceiling.
                if bank_tools is not None:
                    enabled = set(filtered) & set(bank_tools)
                else:
                    enabled = set(filtered)

        return enabled

    if hasattr(mcp, "list_tools"):
        # FastMCP 3.x: wrap list_tools() and get_tool() on the instance
        original_list_tools = mcp.list_tools
        original_get_tool = mcp.get_tool

        async def _filtered_list_tools(**kwargs):
            tools = await original_list_tools(**kwargs)
            enabled_set = await _get_enabled_tools()
            if enabled_set is None:
                return tools
            return [t for t in tools if t.name in enabled_set]

        async def _filtered_get_tool(name, **kwargs):
            enabled_set = await _get_enabled_tools()
            if enabled_set is not None and name not in enabled_set:
                return None  # FastMCP treats None as "not found" → raises NotFoundError
            return await original_get_tool(name, **kwargs)

        object.__setattr__(mcp, "list_tools", _filtered_list_tools)
        object.__setattr__(mcp, "get_tool", _filtered_get_tool)

    elif hasattr(mcp, "_tool_manager"):
        # FastMCP 2.x: wrap _tool_manager.get_tools() and tool.run()
        try:
            tool_manager = mcp._tool_manager
            original_get_tools = tool_manager.get_tools

            async def _filtered_get_tools():
                all_tools = await original_get_tools()
                enabled_set = await _get_enabled_tools()
                if enabled_set is None:
                    return all_tools
                return {k: v for k, v in all_tools.items() if k in enabled_set}

            setattr(tool_manager, "get_tools", _filtered_get_tools)

            for name, tool in tool_manager._tools.items():
                original_run = tool.run

                async def _filtered_run(arguments, _name=name, _orig=original_run):
                    enabled_set = await _get_enabled_tools()
                    if enabled_set is not None and _name not in enabled_set:
                        raise ValueError(f"Tool '{_name}' is not enabled for bank '{config.bank_id_resolver()}'")
                    return await _orig(arguments)

                object.__setattr__(tool, "run", _filtered_run)
        except (AttributeError, KeyError) as e:
            logger.warning(f"Could not apply bank tool filtering (v2): {e}")
    else:
        logger.warning("Could not apply bank tool filtering: unknown FastMCP version")


_AUDITABLE_MCP_TOOLS: frozenset[str] = frozenset(
    {
        "retain",
        "recall",
        "reflect",
        "create_bank",
        "update_bank",
        "delete_bank",
        "clear_memories",
        "create_mental_model",
        "update_mental_model",
        "delete_mental_model",
        "refresh_mental_model",
        "clear_mental_model",
        "create_directive",
        "delete_directive",
        "delete_document",
        "cancel_operation",
        "create_knowledge_folder",
        "create_knowledge_page",
        "update_knowledge_node",
        "delete_knowledge_node",
    }
)


def _apply_audit_logging(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Wrap auditable MCP tool run methods with audit logging."""
    audit_logger: AuditLogger = memory.audit_logger

    def _wrap_tool_run(tool_name: str, original_run):
        """Create an audited wrapper for a tool's run method."""

        async def _audited_run(arguments, _name=tool_name, _orig=original_run):
            # Cheap bank-independent pre-filter before resolving bank_id.
            if not audit_logger.action_allowed(_name):
                return await _orig(arguments)

            bank_id = None
            if isinstance(arguments, dict):
                bank_id = arguments.get("bank_id") or (config.bank_id_resolver() if config.bank_id_resolver else None)
            elif hasattr(arguments, "get"):
                bank_id = arguments.get("bank_id")

            # Per-bank decision, resolved after bank_id is known.
            if not await audit_logger.should_log(_name, bank_id):
                return await _orig(arguments)

            entry = AuditEntry(
                action=_name,
                transport="mcp",
                bank_id=bank_id,
                started_at=datetime.now(timezone.utc),
                request=dict(arguments) if isinstance(arguments, dict) else {"raw": str(arguments)},
            )

            try:
                result = await _orig(arguments)
                if isinstance(result, dict):
                    entry.response = result
                elif isinstance(result, list):
                    entry.response = {"items": result}
                elif isinstance(result, str):
                    entry.response = {"text": result}
                return result
            finally:
                entry.ended_at = datetime.now(timezone.utc)
                audit_logger.log_fire_and_forget(entry)

        return _audited_run

    if hasattr(mcp, "_tool_manager"):
        # FastMCP 2.x
        try:
            for name, tool in mcp._tool_manager._tools.items():  # type: ignore[unresolved-attribute]  # FastMCP 2.x internal; guarded by hasattr
                if name in _AUDITABLE_MCP_TOOLS:
                    object.__setattr__(tool, "run", _wrap_tool_run(name, tool.run))
        except (AttributeError, KeyError) as e:
            logger.warning(f"Could not apply MCP audit logging (v2): {e}")
    elif hasattr(mcp, "get_tool"):
        # FastMCP 3.x: wrap call_tool
        original_call_tool = getattr(mcp, "call_tool", None)
        if original_call_tool:

            async def _audited_call_tool(name, arguments=None, **kwargs):
                # Cheap bank-independent pre-filter before resolving bank_id.
                if name not in _AUDITABLE_MCP_TOOLS or not audit_logger.action_allowed(name):
                    return await original_call_tool(name, arguments, **kwargs)

                bank_id = None
                if isinstance(arguments, dict):
                    bank_id = arguments.get("bank_id") or (
                        config.bank_id_resolver() if config.bank_id_resolver else None
                    )

                # Per-bank decision, resolved after bank_id is known.
                if not await audit_logger.should_log(name, bank_id):
                    return await original_call_tool(name, arguments, **kwargs)

                entry = AuditEntry(
                    action=name,
                    transport="mcp",
                    bank_id=bank_id,
                    started_at=datetime.now(timezone.utc),
                    request=dict(arguments) if isinstance(arguments, dict) else {},
                )

                try:
                    result = await original_call_tool(name, arguments, **kwargs)
                    entry.response = {"result": str(result)[:4096]}
                    return result
                finally:
                    entry.ended_at = datetime.now(timezone.utc)
                    audit_logger.log_fire_and_forget(entry)

            object.__setattr__(mcp, "call_tool", _audited_call_tool)
    else:
        logger.warning("Could not apply MCP audit logging: unknown FastMCP version")


def _register_retain(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the retain tool."""
    description = config.retain_description or DEFAULT_MCP_RETAIN_DESCRIPTION

    if config.include_bank_id_param:

        @mcp.tool(description=description, annotations=_tool_annotations("retain"))
        async def retain(
            content: str,
            context: str = "general",
            timestamp: str | None = None,
            tags: list[str] | None = None,
            metadata: dict[str, str] | None = None,
            document_id: str | None = None,
            bank_id: str | None = None,
            strategy: str | None = None,
            update_mode: str | None = None,
        ) -> dict:
            """
            Args:
                content: The fact/memory to store (be specific and include relevant details)
                context: Category for the memory (e.g., 'preferences', 'work', 'hobbies', 'family'). Default: 'general'
                timestamp: When this event/fact occurred (ISO format, e.g., '2024-01-15T10:30:00Z'). Useful for timeline tracking.
                tags: Optional tags for scoped visibility filtering (e.g., ['project:alpha', 'user:123'])
                metadata: Optional key-value metadata to attach (e.g., {'source': 'slack', 'channel': 'general'})
                document_id: Optional document ID to associate this memory with
                bank_id: Optional bank to store in (defaults to session bank). Use for cross-bank operations.
                strategy: Optional named retain strategy (e.g., 'exact' for verbatim storage). Strategies are defined in the bank config.
                update_mode: How to handle existing documents with the same document_id. 'replace' (default) or 'append' (concatenates new content to existing).
            """
            target_bank = bank_id or config.bank_id_resolver()
            if target_bank is None:
                return {"status": "error", "message": "No bank_id configured"}

            content_dict, error = build_content_dict(
                content, context, timestamp, tags, metadata, document_id, strategy, update_mode
            )
            if error:
                return {"status": "error", "message": error}

            request_context = _get_request_context(config)

            try:
                result = await memory.submit_async_retain(
                    bank_id=target_bank,
                    contents=[content_dict],
                    # `get`, not `pop`: the list above holds this same dict, so popping would
                    # strip `strategy` off the item before the call runs and retain_params
                    # would not capture it — a later reprocess then re-extracts under the
                    # bank's default strategy.
                    strategy=content_dict.get("strategy"),
                    request_context=request_context,
                )
                return {
                    "status": "accepted",
                    "message": "Memory storage initiated",
                    "operation_id": result.get("operation_id"),
                }
            except OperationValidationError as e:
                logger.warning(f"Retain rejected: {e}")
                return {"status": "error", "message": str(e)}
            except Exception as e:
                logger.error(f"Error storing memory: {e}", exc_info=True)
                return {"status": "error", "message": str(e)}

    else:

        @mcp.tool(description=description, annotations=_tool_annotations("retain"))
        async def retain(
            content: str,
            context: str = "general",
            timestamp: str | None = None,
            tags: list[str] | None = None,
            metadata: dict[str, str] | None = None,
            document_id: str | None = None,
            strategy: str | None = None,
            update_mode: str | None = None,
        ) -> dict:
            """
            Args:
                content: The fact/memory to store (be specific and include relevant details)
                context: Category for the memory (e.g., 'preferences', 'work', 'hobbies', 'family'). Default: 'general'
                timestamp: When this event/fact occurred (ISO format, e.g., '2024-01-15T10:30:00Z'). Useful for timeline tracking.
                tags: Optional tags for scoped visibility filtering (e.g., ['project:alpha', 'user:123'])
                metadata: Optional key-value metadata to attach (e.g., {'source': 'slack', 'channel': 'general'})
                document_id: Optional document ID to associate this memory with
                strategy: Optional named retain strategy (e.g., 'exact' for verbatim storage). Strategies are defined in the bank config.
                update_mode: How to handle existing documents with the same document_id. 'replace' (default) or 'append' (concatenates new content to existing).
            """
            target_bank = config.bank_id_resolver()
            if target_bank is None:
                return {"status": "error", "message": "No bank_id configured"}

            content_dict, error = build_content_dict(
                content, context, timestamp, tags, metadata, document_id, strategy, update_mode
            )
            if error:
                return {"status": "error", "message": error}

            request_context = _get_request_context(config)

            try:
                result = await memory.submit_async_retain(
                    bank_id=target_bank,
                    contents=[content_dict],
                    # `get`, not `pop`: the list above holds this same dict, so popping would
                    # strip `strategy` off the item before the call runs and retain_params
                    # would not capture it — a later reprocess then re-extracts under the
                    # bank's default strategy.
                    strategy=content_dict.get("strategy"),
                    request_context=request_context,
                )
                return {
                    "status": "accepted",
                    "message": "Memory storage initiated",
                    "operation_id": result.get("operation_id"),
                }
            except OperationValidationError as e:
                logger.warning(f"Retain rejected: {e}")
                return {"status": "error", "message": str(e)}
            except Exception as e:
                logger.error(f"Error storing memory: {e}", exc_info=True)
                return {"status": "error", "message": str(e)}


def _register_sync_retain(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the sync_retain tool (synchronous retain that waits for completion)."""

    if config.include_bank_id_param:

        @mcp.tool(annotations=_tool_annotations("sync_retain"))
        async def sync_retain(
            content: str,
            context: str = "general",
            timestamp: str | None = None,
            tags: list[str] | None = None,
            metadata: dict[str, str] | None = None,
            document_id: str | None = None,
            bank_id: str | None = None,
            strategy: str | None = None,
        ) -> dict:
            """Store information to long-term memory and wait for completion.

            Unlike retain (which is asynchronous), this tool blocks until the memory
            is fully stored and immediately available for recall.

            Args:
                content: The fact/memory to store (be specific and include relevant details)
                context: Category for the memory (e.g., 'preferences', 'work', 'hobbies', 'family'). Default: 'general'
                timestamp: When this event/fact occurred (ISO format, e.g., '2024-01-15T10:30:00Z'). Useful for timeline tracking.
                tags: Optional tags for scoped visibility filtering (e.g., ['project:alpha', 'user:123'])
                metadata: Optional key-value metadata to attach (e.g., {'source': 'slack', 'channel': 'general'})
                document_id: Optional document ID to associate this memory with
                bank_id: Optional bank to store in (defaults to session bank). Use for cross-bank operations.
                strategy: Optional named retain strategy (e.g., 'exact' for verbatim storage). Strategies are defined in the bank config.
            """
            target_bank = bank_id or config.bank_id_resolver()
            if target_bank is None:
                return {"status": "error", "message": "No bank_id configured"}

            content_dict, error = build_content_dict(content, context, timestamp, tags, metadata, document_id, strategy)
            if error:
                return {"status": "error", "message": error}

            request_context = _get_request_context(config)

            try:
                result = await memory.retain_batch_async(
                    bank_id=target_bank,
                    contents=[content_dict],
                    request_context=request_context,
                    # `get`, not `pop`: the list above holds this same dict, so popping would
                    # strip `strategy` off the item before the call runs and retain_params
                    # would not capture it — a later reprocess then re-extracts under the
                    # bank's default strategy.
                    strategy=content_dict.get("strategy"),
                )
                memory_ids = [uid for batch in result for uid in batch]
                return {
                    "status": "completed",
                    "message": "Memory stored successfully",
                    "memory_ids": memory_ids,
                }
            except OperationValidationError as e:
                logger.warning(f"Sync retain rejected: {e}")
                return {"status": "error", "message": str(e)}
            except Exception as e:
                logger.error(f"Error in sync retain: {e}", exc_info=True)
                return {"status": "error", "message": str(e)}

    else:

        @mcp.tool(annotations=_tool_annotations("sync_retain"))
        async def sync_retain(
            content: str,
            context: str = "general",
            timestamp: str | None = None,
            tags: list[str] | None = None,
            metadata: dict[str, str] | None = None,
            document_id: str | None = None,
            strategy: str | None = None,
        ) -> dict:
            """Store information to long-term memory and wait for completion.

            Unlike retain (which is asynchronous), this tool blocks until the memory
            is fully stored and immediately available for recall.

            Args:
                content: The fact/memory to store (be specific and include relevant details)
                context: Category for the memory (e.g., 'preferences', 'work', 'hobbies', 'family'). Default: 'general'
                timestamp: When this event/fact occurred (ISO format, e.g., '2024-01-15T10:30:00Z'). Useful for timeline tracking.
                tags: Optional tags for scoped visibility filtering (e.g., ['project:alpha', 'user:123'])
                metadata: Optional key-value metadata to attach (e.g., {'source': 'slack', 'channel': 'general'})
                document_id: Optional document ID to associate this memory with
                strategy: Optional named retain strategy (e.g., 'exact' for verbatim storage). Strategies are defined in the bank config.
            """
            target_bank = config.bank_id_resolver()
            if target_bank is None:
                return {"status": "error", "message": "No bank_id configured"}

            content_dict, error = build_content_dict(content, context, timestamp, tags, metadata, document_id, strategy)
            if error:
                return {"status": "error", "message": error}

            request_context = _get_request_context(config)

            try:
                result = await memory.retain_batch_async(
                    bank_id=target_bank,
                    contents=[content_dict],
                    request_context=request_context,
                    # `get`, not `pop`: the list above holds this same dict, so popping would
                    # strip `strategy` off the item before the call runs and retain_params
                    # would not capture it — a later reprocess then re-extracts under the
                    # bank's default strategy.
                    strategy=content_dict.get("strategy"),
                )
                memory_ids = [uid for batch in result for uid in batch]
                return {
                    "status": "completed",
                    "message": "Memory stored successfully",
                    "memory_ids": memory_ids,
                }
            except OperationValidationError as e:
                logger.warning(f"Sync retain rejected: {e}")
                return {"status": "error", "message": str(e)}
            except Exception as e:
                logger.error(f"Error in sync retain: {e}", exc_info=True)
                return {"status": "error", "message": str(e)}


def _register_recall(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the recall tool."""
    description = config.recall_description or DEFAULT_MCP_RECALL_DESCRIPTION

    if config.include_bank_id_param:

        @mcp.tool(description=description, annotations=_tool_annotations("recall"))
        async def recall(
            query: str,
            max_tokens: int = 4096,
            budget: str = "high",
            types: list[str] | None = None,
            prefer_observations: bool = False,
            tags: list[str] | None = None,
            tags_match: str = "any",
            tag_groups: list[dict] | None = None,
            query_timestamp: str | None = None,
            min_scores: dict | None = None,
            temporal_window: dict | None = None,
            bank_id: str | None = None,
        ) -> str | dict:
            """
            Args:
                query: Natural language search query (e.g., "user's food preferences", "what projects is user working on")
                max_tokens: Maximum tokens to return in results (default: 4096)
                budget: Search budget - 'low', 'mid', or 'high' (default: 'high'). Higher budgets search more thoroughly.
                types: Fact types to include (e.g., ['world', 'experience']). Default: all types.
                prefer_observations: When recalling raw facts together with 'observation', drop any raw fact
                    that a returned observation was consolidated from, so the observation supersedes it (no
                    duplicate content). Disabled by default; set true to enable. No effect unless
                    'observation' and a raw type are both in types. Default: False.
                tags: Optional tags to filter results by (e.g., ['project:alpha']). Mutually exclusive with tag_groups.
                tags_match: How to match tags - 'any' (match any tag) or 'all' (match all tags). Default: 'any'
                tag_groups: Compound tag filter using boolean groups (AND-ed together). Each group is a leaf
                    {"tags": [...], "match": "any_strict"} or compound {"and": [...]}, {"or": [...]}, {"not": {...}}.
                    Example: [{"not": {"tags": ["closeout"], "match": "any_strict"}}] excludes memories tagged closeout.
                    Mutually exclusive with tags.
                query_timestamp: Temporal context for the query (ISO format, e.g., '2024-01-15T10:30:00Z').
                    Anchors relative temporal expressions and recency scoring.
                min_scores: Optional per-stage score floors as an object with any of: "semantic", "keyword"
                    (retrieval-level cutoffs), "reranker", "final" (post-ranking). E.g. {"reranker": 0.5}.
                    Each floor is inclusive; omit for no score filtering. "semantic" and "keyword" prune only
                    the retrieval arm they name — recall fuses four arms (semantic, keyword, graph, temporal)
                    and returns what any of them surfaced, so a result may report null or a lower score for an
                    arm that did not surface it, and setting both does not restrict results to those clearing
                    both. Use "reranker"/"final" — applied to every scored result — to make recall abstain.
                    The reranker's absolute scores are not calibrated across queries, so only threshold
                    against scores you've calibrated for your own data.
                temporal_window: Window for the temporal arm as {"start": ISO, "end": ISO}, used instead of
                    extracting dates from the query text — pass it when you already know the range you mean.
                    It ranks memories dated inside the window higher; it does NOT drop memories dated outside
                    it, so do not use it to restrict results to a period.
                bank_id: Optional bank to search in (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return "Error: No bank_id configured"

                if tags is not None and tag_groups is not None:
                    raise ValueError(
                        "'tags' and 'tag_groups' are mutually exclusive. Use 'tag_groups' for compound filtering."
                    )

                budget_map = {"low": Budget.LOW, "mid": Budget.MID, "high": Budget.HIGH}
                budget_enum = budget_map.get(budget.lower(), Budget.HIGH)
                fact_types = types if types is not None else list(VALID_RECALL_FACT_TYPES)

                recall_kwargs: dict[str, Any] = {
                    "bank_id": target_bank,
                    "query": query,
                    "fact_type": fact_types,
                    "prefer_observations": prefer_observations,
                    "budget": budget_enum,
                    "max_tokens": max_tokens,
                    "request_context": _get_request_context(config),
                }
                if tags is not None:
                    recall_kwargs["tags"] = tags
                    recall_kwargs["tags_match"] = tags_match
                if tag_groups is not None:
                    recall_kwargs["tag_groups"] = _TAG_GROUP_LIST_ADAPTER.validate_python(tag_groups)
                if query_timestamp is not None:
                    recall_kwargs["question_date"] = parse_timestamp(query_timestamp)
                if min_scores is not None:
                    recall_kwargs["min_scores"] = MinScores.model_validate(min_scores)
                if temporal_window is not None:
                    recall_kwargs["temporal_window"] = TemporalWindow.model_validate(temporal_window)

                recall_result = await memory.recall_async(**recall_kwargs)

                return recall_result.model_dump_json(indent=2)
            except OperationValidationError as e:
                logger.warning(f"Recall rejected: {e}")
                return json.dumps({"error": str(e), "results": []})
            except ValueError as e:
                return f'{{"error": "{e}", "results": []}}'
            except Exception as e:
                logger.error(f"Error searching: {e}", exc_info=True)
                return f'{{"error": "{e}", "results": []}}'

    else:

        @mcp.tool(description=description, annotations=_tool_annotations("recall"))
        async def recall(
            query: str,
            max_tokens: int = 4096,
            budget: str = "high",
            types: list[str] | None = None,
            prefer_observations: bool = False,
            tags: list[str] | None = None,
            tags_match: str = "any",
            tag_groups: list[dict] | None = None,
            query_timestamp: str | None = None,
            min_scores: dict | None = None,
            temporal_window: dict | None = None,
        ) -> dict:
            """
            Args:
                query: Natural language search query (e.g., "user's food preferences", "what projects is user working on")
                max_tokens: Maximum tokens to return in results (default: 4096)
                budget: Search budget - 'low', 'mid', or 'high' (default: 'high'). Higher budgets search more thoroughly.
                types: Fact types to include (e.g., ['world', 'experience']). Default: all types.
                prefer_observations: When recalling raw facts together with 'observation', drop any raw fact
                    that a returned observation was consolidated from, so the observation supersedes it (no
                    duplicate content). Disabled by default; set true to enable. No effect unless
                    'observation' and a raw type are both in types. Default: False.
                tags: Optional tags to filter results by (e.g., ['project:alpha']). Mutually exclusive with tag_groups.
                tags_match: How to match tags - 'any' (match any tag) or 'all' (match all tags). Default: 'any'
                tag_groups: Compound tag filter using boolean groups (AND-ed together). Each group is a leaf
                    {"tags": [...], "match": "any_strict"} or compound {"and": [...]}, {"or": [...]}, {"not": {...}}.
                    Example: [{"not": {"tags": ["closeout"], "match": "any_strict"}}] excludes memories tagged closeout.
                    Mutually exclusive with tags.
                query_timestamp: Temporal context for the query (ISO format, e.g., '2024-01-15T10:30:00Z').
                    Anchors relative temporal expressions and recency scoring.
                min_scores: Optional per-stage score floors as an object with any of: "semantic", "keyword"
                    (retrieval-level cutoffs), "reranker", "final" (post-ranking). E.g. {"reranker": 0.5}.
                    Each floor is inclusive; omit for no score filtering. "semantic" and "keyword" prune only
                    the retrieval arm they name — recall fuses four arms (semantic, keyword, graph, temporal)
                    and returns what any of them surfaced, so a result may report null or a lower score for an
                    arm that did not surface it, and setting both does not restrict results to those clearing
                    both. Use "reranker"/"final" — applied to every scored result — to make recall abstain.
                    The reranker's absolute scores are not calibrated across queries, so only threshold
                    against scores you've calibrated for your own data.
                temporal_window: Window for the temporal arm as {"start": ISO, "end": ISO}, used instead of
                    extracting dates from the query text — pass it when you already know the range you mean.
                    It ranks memories dated inside the window higher; it does NOT drop memories dated outside
                    it, so do not use it to restrict results to a period.
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured", "results": []}

                if tags is not None and tag_groups is not None:
                    raise ValueError(
                        "'tags' and 'tag_groups' are mutually exclusive. Use 'tag_groups' for compound filtering."
                    )

                budget_map = {"low": Budget.LOW, "mid": Budget.MID, "high": Budget.HIGH}
                budget_enum = budget_map.get(budget.lower(), Budget.HIGH)
                fact_types = types if types is not None else list(VALID_RECALL_FACT_TYPES)

                recall_kwargs: dict[str, Any] = {
                    "bank_id": target_bank,
                    "query": query,
                    "fact_type": fact_types,
                    "prefer_observations": prefer_observations,
                    "budget": budget_enum,
                    "max_tokens": max_tokens,
                    "request_context": _get_request_context(config),
                }
                if tags is not None:
                    recall_kwargs["tags"] = tags
                    recall_kwargs["tags_match"] = tags_match
                if tag_groups is not None:
                    recall_kwargs["tag_groups"] = _TAG_GROUP_LIST_ADAPTER.validate_python(tag_groups)
                if query_timestamp is not None:
                    recall_kwargs["question_date"] = parse_timestamp(query_timestamp)
                if min_scores is not None:
                    recall_kwargs["min_scores"] = MinScores.model_validate(min_scores)
                if temporal_window is not None:
                    recall_kwargs["temporal_window"] = TemporalWindow.model_validate(temporal_window)

                recall_result = await memory.recall_async(**recall_kwargs)

                return recall_result.model_dump()
            except OperationValidationError as e:
                logger.warning(f"Recall rejected: {e}")
                return {"error": str(e), "results": []}
            except ValueError as e:
                return {"error": str(e), "results": []}
            except Exception as e:
                logger.error(f"Error searching: {e}", exc_info=True)
                return {"error": str(e), "results": []}


def _register_reflect(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the reflect tool."""

    if config.include_bank_id_param:

        @mcp.tool(annotations=_tool_annotations("reflect"))
        async def reflect(
            query: str,
            context: str | None = None,
            budget: str = "low",
            max_tokens: int = 4096,
            response_schema: dict | None = None,
            tags: list[str] | None = None,
            tags_match: str = "any",
            apply_all_directives: bool = False,
            include_based_on: bool = False,
            include_trace: bool = False,
            bank_id: str | None = None,
        ) -> str:
            """
            Generate thoughtful analysis by synthesizing stored memories with the bank's personality.

            WHEN TO USE THIS TOOL:
            Use reflect when you need reasoned analysis, not just fact retrieval. This tool
            thinks through the question using everything the bank knows and its personality traits.

            EXAMPLES OF GOOD QUERIES:
            - "What patterns have emerged in how I approach debugging?"
            - "Based on my past decisions, what architectural style do I prefer?"
            - "What might be the best approach for this problem given what you know about me?"
            - "How should I prioritize these tasks based on my goals?"

            HOW IT DIFFERS FROM RECALL:
            - recall: Returns raw facts matching your search (fast lookup)
            - reflect: Reasons across memories to form a synthesized answer (deeper analysis)

            Use recall for "what did I say about X?" and reflect for "what should I do about X?"

            Args:
                query: The question or topic to reflect on
                context: Optional context about why this reflection is needed
                budget: Search budget - 'low', 'mid', or 'high' (default: 'low')
                max_tokens: Maximum tokens for the response (default: 4096)
                response_schema: Optional JSON schema for structured output. When provided, the response includes a 'structured_output' field.
                tags: Optional tags to filter memories by (e.g., ['project:alpha'])
                tags_match: How to match tags - 'any' (match any tag) or 'all' (match all tags). Default: 'any'
                apply_all_directives: Apply every active directive regardless of tags. By default directives are scoped like memories (untagged always apply; tagged apply only when tags match). Set true to apply all directives, ignoring tag scope.
                include_based_on: Include source facts used for synthesis. Defaults to false because broad reflections can exceed MCP client result limits.
                include_trace: Include the reflection's internal trace fields (tool_trace/llm_trace and directives_applied). Defaults to false because the trace can be tens of KB and overflow MCP client context; enable only for debugging.
                bank_id: Optional bank to reflect in (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return "Error: No bank_id configured"

                budget_map = {"low": Budget.LOW, "mid": Budget.MID, "high": Budget.HIGH}
                budget_enum = budget_map.get(budget.lower(), Budget.LOW)

                reflect_kwargs: dict[str, Any] = {
                    "bank_id": target_bank,
                    "query": query,
                    "budget": budget_enum,
                    "context": context,
                    "max_tokens": max_tokens,
                    "apply_all_directives": apply_all_directives,
                    "request_context": _get_request_context(config),
                }
                if response_schema is not None:
                    reflect_kwargs["response_schema"] = response_schema
                if tags is not None:
                    reflect_kwargs["tags"] = tags
                    reflect_kwargs["tags_match"] = tags_match

                reflect_result = await memory.reflect_async(**reflect_kwargs)

                result_data = json.loads(reflect_result.model_dump_json(indent=2))
                if not include_based_on:
                    result_data.pop("based_on", None)
                if not include_trace:
                    # The agentic reflect loop's trace fields can be tens of KB (full
                    # mental-model text) and silently overflow MCP client context; the
                    # REST API omits them by default too. directives_applied is built by
                    # the engine "for the trace" and carries full directive content, so it
                    # belongs with tool_trace/llm_trace here. Opt in via include_trace.
                    result_data.pop("tool_trace", None)
                    result_data.pop("llm_trace", None)
                    result_data.pop("directives_applied", None)
                if response_schema is not None and hasattr(reflect_result, "structured_output"):
                    result_data["structured_output"] = reflect_result.structured_output
                return json.dumps(result_data, indent=2)
            except OperationValidationError as e:
                logger.warning(f"Reflect rejected: {e}")
                return json.dumps({"error": str(e)})
            except Exception as e:
                logger.error(f"Error reflecting: {e}", exc_info=True)
                return f'{{"error": "{e}", "text": ""}}'

    else:

        @mcp.tool(annotations=_tool_annotations("reflect"))
        async def reflect(
            query: str,
            context: str | None = None,
            budget: str = "low",
            max_tokens: int = 4096,
            response_schema: dict | None = None,
            tags: list[str] | None = None,
            tags_match: str = "any",
            apply_all_directives: bool = False,
            include_based_on: bool = False,
            include_trace: bool = False,
        ) -> dict:
            """
            Generate thoughtful analysis by synthesizing stored memories with the bank's personality.

            WHEN TO USE THIS TOOL:
            Use reflect when you need reasoned analysis, not just fact retrieval. This tool
            thinks through the question using everything the bank knows and its personality traits.

            EXAMPLES OF GOOD QUERIES:
            - "What patterns have emerged in how I approach debugging?"
            - "Based on my past decisions, what architectural style do I prefer?"
            - "What might be the best approach for this problem given what you know about me?"
            - "How should I prioritize these tasks based on my goals?"

            HOW IT DIFFERS FROM RECALL:
            - recall: Returns raw facts matching your search (fast lookup)
            - reflect: Reasons across memories to form a synthesized answer (deeper analysis)

            Use recall for "what did I say about X?" and reflect for "what should I do about X?"

            Args:
                query: The question or topic to reflect on
                context: Optional context about why this reflection is needed
                budget: Search budget - 'low', 'mid', or 'high' (default: 'low')
                max_tokens: Maximum tokens for the response (default: 4096)
                response_schema: Optional JSON schema for structured output. When provided, the response includes a 'structured_output' field.
                tags: Optional tags to filter memories by (e.g., ['project:alpha'])
                tags_match: How to match tags - 'any' (match any tag) or 'all' (match all tags). Default: 'any'
                apply_all_directives: Apply every active directive regardless of tags. By default directives are scoped like memories (untagged always apply; tagged apply only when tags match). Set true to apply all directives, ignoring tag scope.
                include_based_on: Include source facts used for synthesis. Defaults to false because broad reflections can exceed MCP client result limits.
                include_trace: Include the reflection's internal trace fields (tool_trace/llm_trace and directives_applied). Defaults to false because the trace can be tens of KB and overflow MCP client context; enable only for debugging.
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured", "text": ""}

                budget_map = {"low": Budget.LOW, "mid": Budget.MID, "high": Budget.HIGH}
                budget_enum = budget_map.get(budget.lower(), Budget.LOW)

                reflect_kwargs: dict[str, Any] = {
                    "bank_id": target_bank,
                    "query": query,
                    "budget": budget_enum,
                    "context": context,
                    "max_tokens": max_tokens,
                    "apply_all_directives": apply_all_directives,
                    "request_context": _get_request_context(config),
                }
                if response_schema is not None:
                    reflect_kwargs["response_schema"] = response_schema
                if tags is not None:
                    reflect_kwargs["tags"] = tags
                    reflect_kwargs["tags_match"] = tags_match

                reflect_result = await memory.reflect_async(**reflect_kwargs)

                result_data = reflect_result.model_dump()
                if not include_based_on:
                    result_data.pop("based_on", None)
                if not include_trace:
                    # The agentic reflect loop's trace fields can be tens of KB (full
                    # mental-model text) and silently overflow MCP client context; the
                    # REST API omits them by default too. directives_applied is built by
                    # the engine "for the trace" and carries full directive content, so it
                    # belongs with tool_trace/llm_trace here. Opt in via include_trace.
                    result_data.pop("tool_trace", None)
                    result_data.pop("llm_trace", None)
                    result_data.pop("directives_applied", None)
                if response_schema is not None and hasattr(reflect_result, "structured_output"):
                    result_data["structured_output"] = reflect_result.structured_output
                return result_data
            except OperationValidationError as e:
                logger.warning(f"Reflect rejected: {e}")
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Error reflecting: {e}", exc_info=True)
                return {"error": str(e), "text": ""}


def _register_list_banks(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the list_banks tool."""

    @mcp.tool(annotations=_tool_annotations("list_banks"))
    async def list_banks(query: str | None = None, limit: int = 100, offset: int = 0) -> str:
        """
        List available memory banks, most recently written first.

        Use this tool to discover what memory banks exist in the system.
        Each bank is an isolated memory store (like a separate "brain").

        Args:
            query: Optional case-insensitive substring to match against bank ID and name.
            limit: Maximum number of banks to return (default 100).
            offset: Number of banks to skip, for paging through `total`.

        Returns:
            JSON with the page of banks (IDs, names, dispositions, missions) plus
            the total number of matching banks and the limit/offset used.
        """
        try:
            data = await memory.list_banks(
                search_query=query,
                limit=limit,
                offset=offset,
                request_context=_get_request_context(config),
            )
            return json.dumps(data, indent=2)
        except OperationValidationError as e:
            logger.warning(f"Operation rejected: {e}")
            return json.dumps({"error": str(e), "banks": []})
        except Exception as e:
            logger.error(f"Error listing banks: {e}", exc_info=True)
            return f'{{"error": "{e}", "banks": []}}'


def _register_create_bank(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the create_bank tool."""

    @mcp.tool(annotations=_tool_annotations("create_bank"))
    async def create_bank(bank_id: str, name: str | None = None, mission: str | None = None) -> str:
        """
        Create a new memory bank or get an existing one.

        Memory banks are isolated stores - each one is like a separate "brain" for a user/agent.
        Banks are auto-created with default settings if they don't exist.

        Args:
            bank_id: Unique identifier for the bank (e.g., 'user-123', 'agent-alpha')
            name: Optional human-friendly name for the bank
            mission: Optional mission describing who the agent is and what they're trying to accomplish
        """
        try:
            request_context = _get_request_context(config)
            if name is not None or mission is not None:
                profile = await memory.update_bank(
                    bank_id,
                    name=name,
                    mission=mission,
                    request_context=request_context,
                )
            else:
                # The public profile API owns bank creation and its lifecycle
                # validation when no profile fields need updating.
                profile = await memory.get_bank_profile(bank_id, request_context=request_context)

            # Serialize disposition if it's a Pydantic model
            if "disposition" in profile and hasattr(profile["disposition"], "model_dump"):
                profile["disposition"] = profile["disposition"].model_dump()
            return json.dumps(profile, indent=2)
        except OperationValidationError as e:
            logger.warning(f"Operation rejected: {e}")
            return json.dumps({"error": str(e)})
        except Exception as e:
            logger.error(f"Error creating bank: {e}", exc_info=True)
            return f'{{"error": "{e}"}}'


def _validate_mental_model_inputs(
    name: str | None = None,
    source_query: str | None = None,
    max_tokens: int | None = None,
    tags_match: str | None = None,
) -> str | None:
    """Validate mental model inputs, returning an error message or None if valid."""
    if name is not None and not name.strip():
        return "name cannot be empty"
    if source_query is not None and not source_query.strip():
        return "source_query cannot be empty"
    if max_tokens is not None and (max_tokens < 256 or max_tokens > 8192):
        return f"max_tokens must be between 256 and 8192, got {max_tokens}"
    if tags_match is not None and tags_match not in get_args(TagsMatch):
        valid = ", ".join(get_args(TagsMatch))
        return f"tags_match must be one of {valid}, got {tags_match!r}"
    return None


# =========================================================================
# MENTAL MODEL TOOLS
# =========================================================================


def _register_list_mental_models(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the list_mental_models tool."""

    if config.include_bank_id_param:

        @mcp.tool(annotations=_tool_annotations("list_mental_models"))
        async def list_mental_models(
            tags: list[str] | None = None,
            detail: str = "full",
            limit: int = 100,
            offset: int = 0,
            bank_id: str | None = None,
        ) -> str:
            """
            List mental models (pinned reflections) for a memory bank.

            Mental models are living documents that stay current by periodically re-running
            a source query through reflect. Use them to maintain up-to-date summaries,
            preferences, or synthesized knowledge.

            Args:
                tags: Optional tags to filter by (returns models matching any tag)
                detail: Detail level - 'metadata' (names/tags only), 'content' (adds content/config), 'full' (includes reflect_response). Default: 'full'
                limit: Maximum number of results (default: 100)
                offset: Pagination offset (default: 0). Page until the returned items add up to 'total'.
                bank_id: Optional bank to list from (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured", "items": []}'

                page = await memory.list_mental_models(
                    bank_id=target_bank,
                    tags=tags,
                    detail=detail,
                    limit=limit,
                    offset=offset,
                    request_context=_get_request_context(config),
                )
                return json.dumps({"items": page.items, "total": page.total}, indent=2, default=str)
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return json.dumps({"error": str(e)})
            except Exception as e:
                logger.error(f"Error listing mental models: {e}", exc_info=True)
                return f'{{"error": "{e}", "items": []}}'

    else:

        @mcp.tool(annotations=_tool_annotations("list_mental_models"))
        async def list_mental_models(
            tags: list[str] | None = None,
            detail: str = "full",
            limit: int = 100,
            offset: int = 0,
        ) -> dict:
            """
            List mental models (pinned reflections) for this memory bank.

            Mental models are living documents that stay current by periodically re-running
            a source query through reflect. Use them to maintain up-to-date summaries,
            preferences, or synthesized knowledge.

            Args:
                tags: Optional tags to filter by (returns models matching any tag)
                detail: Detail level - 'metadata' (names/tags only), 'content' (adds content/config), 'full' (includes reflect_response). Default: 'full'
                limit: Maximum number of results (default: 100)
                offset: Pagination offset (default: 0). Page until the returned items add up to 'total'.
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured", "items": []}

                page = await memory.list_mental_models(
                    bank_id=target_bank,
                    tags=tags,
                    detail=detail,
                    limit=limit,
                    offset=offset,
                    request_context=_get_request_context(config),
                )
                return {"items": page.items, "total": page.total}
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Error listing mental models: {e}", exc_info=True)
                return {"error": str(e), "items": []}


def _register_get_mental_model(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the get_mental_model tool."""

    if config.include_bank_id_param:

        @mcp.tool(annotations=_tool_annotations("get_mental_model"))
        async def get_mental_model(
            mental_model_id: str,
            detail: str = "full",
            bank_id: str | None = None,
        ) -> str:
            """
            Get a specific mental model by ID.

            Returns the mental model with the requested detail level. Use list_mental_models
            first to discover available model IDs.

            Args:
                mental_model_id: The ID of the mental model to retrieve
                detail: Detail level - 'metadata' (names/tags only), 'content' (adds content/config), 'full' (includes reflect_response). Default: 'full'
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                model = await memory.get_mental_model(
                    bank_id=target_bank,
                    mental_model_id=mental_model_id,
                    detail=detail,
                    request_context=_get_request_context(config),
                )
                if model is None:
                    return json.dumps({"error": f"Mental model '{mental_model_id}' not found in bank '{target_bank}'"})
                return json.dumps(model, indent=2, default=str)
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return json.dumps({"error": str(e)})
            except Exception as e:
                logger.error(f"Error getting mental model: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool(annotations=_tool_annotations("get_mental_model"))
        async def get_mental_model(
            mental_model_id: str,
            detail: str = "full",
        ) -> dict:
            """
            Get a specific mental model by ID.

            Returns the mental model with the requested detail level. Use list_mental_models
            first to discover available model IDs.

            Args:
                mental_model_id: The ID of the mental model to retrieve
                detail: Detail level - 'metadata' (names/tags only), 'content' (adds content/config), 'full' (includes reflect_response). Default: 'full'
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                model = await memory.get_mental_model(
                    bank_id=target_bank,
                    mental_model_id=mental_model_id,
                    detail=detail,
                    request_context=_get_request_context(config),
                )
                if model is None:
                    return {"error": f"Mental model '{mental_model_id}' not found in bank '{target_bank}'"}
                return model
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Error getting mental model: {e}", exc_info=True)
                return {"error": str(e)}


def _register_create_mental_model(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the create_mental_model tool."""

    if config.include_bank_id_param:

        @mcp.tool(annotations=_tool_annotations("create_mental_model"))
        async def create_mental_model(
            name: str,
            source_query: str,
            mental_model_id: str | None = None,
            tags: list[str] | None = None,
            trigger: MentalModelTriggerInput | None = None,
            tags_match: str | None = None,
            max_tokens: int = 2048,
            trigger_refresh_after_consolidation: bool | None = None,
            bank_id: str | None = None,
        ) -> str:
            """
            Create a new mental model (pinned reflection).

            A mental model is a living document generated by running the source_query through
            reflect. The content is auto-generated asynchronously - use the returned operation_id
            to track progress.

            EXAMPLES:
            - name="Coding Preferences", source_query="What coding patterns and tools does the user prefer?"
            - name="Project Goals", source_query="What are the user's current project goals and priorities?"
            - name="Communication Style", source_query="How does the user prefer to communicate?"

            Args:
                name: Human-readable name for the mental model
                source_query: The query to run through reflect to generate content
                mental_model_id: Optional custom ID (alphanumeric lowercase with hyphens). Auto-generated if not provided.
                tags: Optional tags for scoped visibility filtering
                tags_match: How this model's tags are matched against memories when the content
                    is (re)generated. One of 'any' (match any tag, like recall/reflect), 'all'
                    (match all tags), 'any_strict', 'all_strict', or 'exact'. If omitted, a tagged
                    model defaults to 'all_strict' — a memory must carry EVERY one of the model's
                    tags to be included, which silently filters out memories that only carry a
                    subset. Pass 'any' when your memories use narrow single-topic tags.
                trigger: Refresh policy for this model — when it rebuilds itself (mode,
                    refresh_after_consolidation, refresh_cron) and what it rebuilds from
                    (fact_types, tags_match, tag_groups, exclude_mental_models,
                    recall_max_tokens, ...). Omitted fields use engine defaults. Prefer
                    this over the flat tags_match/trigger_refresh_after_consolidation
                    shorthands, which are kept only for existing integrations.
                max_tokens: Maximum tokens for generated content (256-8192, default: 2048)
                trigger_refresh_after_consolidation: If True, automatically refresh this model after memory consolidation. Default: False
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                validation_error = _validate_mental_model_inputs(
                    name=name, source_query=source_query, max_tokens=max_tokens, tags_match=tags_match
                )
                if validation_error:
                    return json.dumps({"error": validation_error})

                request_context = _get_request_context(config)
                trigger_patch = _mental_model_trigger_patch(
                    trigger,
                    tags_match=tags_match,
                    refresh_after_consolidation=trigger_refresh_after_consolidation,
                )
                if trigger_patch is None and trigger is None:
                    trigger_patch = {"refresh_after_consolidation": False}

                # Create with placeholder content
                model = await memory.create_mental_model(
                    bank_id=target_bank,
                    name=name,
                    source_query=source_query,
                    content="Generating content...",
                    mental_model_id=mental_model_id,
                    tags=tags,
                    max_tokens=max_tokens,
                    trigger=trigger_patch,
                    request_context=request_context,
                )

                # Schedule async refresh to generate actual content
                result = await memory.submit_async_refresh_mental_model(
                    bank_id=target_bank,
                    mental_model_id=model["id"],
                    request_context=request_context,
                )

                return json.dumps(
                    {
                        "mental_model_id": model["id"],
                        "operation_id": result["operation_id"],
                        "status": "created",
                        "message": f"Mental model '{name}' created. Content is being generated asynchronously.",
                    }
                )
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return json.dumps({"error": str(e)})
            except ValueError as e:
                return json.dumps({"error": str(e)})
            except Exception as e:
                logger.error(f"Error creating mental model: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool(annotations=_tool_annotations("create_mental_model"))
        async def create_mental_model(
            name: str,
            source_query: str,
            mental_model_id: str | None = None,
            tags: list[str] | None = None,
            trigger: MentalModelTriggerInput | None = None,
            tags_match: str | None = None,
            max_tokens: int = 2048,
            trigger_refresh_after_consolidation: bool | None = None,
        ) -> dict:
            """
            Create a new mental model (pinned reflection).

            A mental model is a living document generated by running the source_query through
            reflect. The content is auto-generated asynchronously - use the returned operation_id
            to track progress.

            EXAMPLES:
            - name="Coding Preferences", source_query="What coding patterns and tools does the user prefer?"
            - name="Project Goals", source_query="What are the user's current project goals and priorities?"
            - name="Communication Style", source_query="How does the user prefer to communicate?"

            Args:
                name: Human-readable name for the mental model
                source_query: The query to run through reflect to generate content
                mental_model_id: Optional custom ID (alphanumeric lowercase with hyphens). Auto-generated if not provided.
                tags: Optional tags for scoped visibility filtering
                tags_match: How this model's tags are matched against memories when the content
                    is (re)generated. One of 'any' (match any tag, like recall/reflect), 'all'
                    (match all tags), 'any_strict', 'all_strict', or 'exact'. If omitted, a tagged
                    model defaults to 'all_strict' — a memory must carry EVERY one of the model's
                    tags to be included, which silently filters out memories that only carry a
                    subset. Pass 'any' when your memories use narrow single-topic tags.
                trigger: Refresh policy for this model — when it rebuilds itself (mode,
                    refresh_after_consolidation, refresh_cron) and what it rebuilds from
                    (fact_types, tags_match, tag_groups, exclude_mental_models,
                    recall_max_tokens, ...). Omitted fields use engine defaults. Prefer
                    this over the flat tags_match/trigger_refresh_after_consolidation
                    shorthands, which are kept only for existing integrations.
                max_tokens: Maximum tokens for generated content (256-8192, default: 2048)
                trigger_refresh_after_consolidation: If True, automatically refresh this model after memory consolidation. Default: False
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                validation_error = _validate_mental_model_inputs(
                    name=name, source_query=source_query, max_tokens=max_tokens, tags_match=tags_match
                )
                if validation_error:
                    return {"error": validation_error}

                request_context = _get_request_context(config)
                trigger_patch = _mental_model_trigger_patch(
                    trigger,
                    tags_match=tags_match,
                    refresh_after_consolidation=trigger_refresh_after_consolidation,
                )
                if trigger_patch is None and trigger is None:
                    trigger_patch = {"refresh_after_consolidation": False}

                model = await memory.create_mental_model(
                    bank_id=target_bank,
                    name=name,
                    source_query=source_query,
                    content="Generating content...",
                    mental_model_id=mental_model_id,
                    tags=tags,
                    max_tokens=max_tokens,
                    trigger=trigger_patch,
                    request_context=request_context,
                )

                result = await memory.submit_async_refresh_mental_model(
                    bank_id=target_bank,
                    mental_model_id=model["id"],
                    request_context=request_context,
                )

                return {
                    "mental_model_id": model["id"],
                    "operation_id": result["operation_id"],
                    "status": "created",
                    "message": f"Mental model '{name}' created. Content is being generated asynchronously.",
                }
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return {"error": str(e)}
            except ValueError as e:
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Error creating mental model: {e}", exc_info=True)
                return {"error": str(e)}


def _register_update_mental_model(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the update_mental_model tool."""

    if config.include_bank_id_param:

        @mcp.tool(annotations=_tool_annotations("update_mental_model"))
        async def update_mental_model(
            mental_model_id: str,
            name: str | None = None,
            source_query: str | None = None,
            max_tokens: int | None = None,
            tags: list[str] | None = None,
            trigger: MentalModelTriggerInput | None = None,
            tags_match: str | None = None,
            trigger_refresh_after_consolidation: bool | None = None,
            bank_id: str | None = None,
        ) -> str:
            """
            Update a mental model's metadata.

            Changes the name, source query, or tags of an existing mental model.
            To regenerate the content, use refresh_mental_model after updating the source query.

            Args:
                mental_model_id: The ID of the mental model to update
                name: New name (leave None to keep current)
                source_query: New source query (leave None to keep current)
                max_tokens: New max tokens for content generation (256-8192, leave None to keep current)
                tags: New tags (leave None to keep current)
                trigger: Refresh policy fields to change — mode, refresh_after_consolidation,
                    refresh_cron, fact_types, tags_match, tag_groups, exclude_mental_models,
                    recall_max_tokens, and so on. This is a PATCH: fields you omit keep their
                    current values, so setting a cron schedule does not reset the model's
                    fact_types. Pass an explicit null to clear a setting.
                tags_match: Legacy shorthand for trigger.tags_match
                trigger_refresh_after_consolidation: If set, update whether this model auto-refreshes after consolidation
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                validation_error = _validate_mental_model_inputs(
                    name=name, source_query=source_query, max_tokens=max_tokens, tags_match=tags_match
                )
                if validation_error:
                    return json.dumps({"error": validation_error})

                trigger_patch = _mental_model_trigger_patch(
                    trigger,
                    tags_match=tags_match,
                    refresh_after_consolidation=trigger_refresh_after_consolidation,
                )

                update_kwargs: dict[str, Any] = {
                    "bank_id": target_bank,
                    "mental_model_id": mental_model_id,
                    "name": name,
                    "source_query": source_query,
                    "max_tokens": max_tokens,
                    "tags": tags,
                    "request_context": _get_request_context(config),
                }
                if trigger_patch is not None:
                    update_kwargs["trigger"] = trigger_patch

                model = await memory.update_mental_model(**update_kwargs)
                if model is None:
                    return json.dumps({"error": f"Mental model '{mental_model_id}' not found in bank '{target_bank}'"})
                return json.dumps(model, indent=2, default=str)
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return json.dumps({"error": str(e)})
            except Exception as e:
                logger.error(f"Error updating mental model: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool(annotations=_tool_annotations("update_mental_model"))
        async def update_mental_model(
            mental_model_id: str,
            name: str | None = None,
            source_query: str | None = None,
            max_tokens: int | None = None,
            tags: list[str] | None = None,
            trigger: MentalModelTriggerInput | None = None,
            tags_match: str | None = None,
            trigger_refresh_after_consolidation: bool | None = None,
        ) -> dict:
            """
            Update a mental model's metadata.

            Changes the name, source query, or tags of an existing mental model.
            To regenerate the content, use refresh_mental_model after updating the source query.

            Args:
                mental_model_id: The ID of the mental model to update
                name: New name (leave None to keep current)
                source_query: New source query (leave None to keep current)
                max_tokens: New max tokens for content generation (256-8192, leave None to keep current)
                tags: New tags (leave None to keep current)
                trigger: Refresh policy fields to change — mode, refresh_after_consolidation,
                    refresh_cron, fact_types, tags_match, tag_groups, exclude_mental_models,
                    recall_max_tokens, and so on. This is a PATCH: fields you omit keep their
                    current values, so setting a cron schedule does not reset the model's
                    fact_types. Pass an explicit null to clear a setting.
                tags_match: Legacy shorthand for trigger.tags_match
                trigger_refresh_after_consolidation: If set, update whether this model auto-refreshes after consolidation
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                validation_error = _validate_mental_model_inputs(
                    name=name, source_query=source_query, max_tokens=max_tokens, tags_match=tags_match
                )
                if validation_error:
                    return {"error": validation_error}

                trigger_patch = _mental_model_trigger_patch(
                    trigger,
                    tags_match=tags_match,
                    refresh_after_consolidation=trigger_refresh_after_consolidation,
                )

                update_kwargs: dict[str, Any] = {
                    "bank_id": target_bank,
                    "mental_model_id": mental_model_id,
                    "name": name,
                    "source_query": source_query,
                    "max_tokens": max_tokens,
                    "tags": tags,
                    "request_context": _get_request_context(config),
                }
                if trigger_patch is not None:
                    update_kwargs["trigger"] = trigger_patch

                model = await memory.update_mental_model(**update_kwargs)
                if model is None:
                    return {"error": f"Mental model '{mental_model_id}' not found in bank '{target_bank}'"}
                return model
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Error updating mental model: {e}", exc_info=True)
                return {"error": str(e)}


def _register_delete_mental_model(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the delete_mental_model tool."""

    if config.include_bank_id_param:

        @mcp.tool(annotations=_tool_annotations("delete_mental_model"))
        async def delete_mental_model(
            mental_model_id: str,
            bank_id: str | None = None,
        ) -> str:
            """
            Delete a mental model.

            Permanently removes a mental model and its generated content.

            Args:
                mental_model_id: The ID of the mental model to delete
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                deleted = await memory.delete_mental_model(
                    bank_id=target_bank,
                    mental_model_id=mental_model_id,
                    request_context=_get_request_context(config),
                )
                if not deleted:
                    return json.dumps({"error": f"Mental model '{mental_model_id}' not found in bank '{target_bank}'"})
                return json.dumps({"status": "deleted", "mental_model_id": mental_model_id})
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return json.dumps({"error": str(e)})
            except Exception as e:
                logger.error(f"Error deleting mental model: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool(annotations=_tool_annotations("delete_mental_model"))
        async def delete_mental_model(
            mental_model_id: str,
        ) -> dict:
            """
            Delete a mental model.

            Permanently removes a mental model and its generated content.

            Args:
                mental_model_id: The ID of the mental model to delete
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                deleted = await memory.delete_mental_model(
                    bank_id=target_bank,
                    mental_model_id=mental_model_id,
                    request_context=_get_request_context(config),
                )
                if not deleted:
                    return {"error": f"Mental model '{mental_model_id}' not found in bank '{target_bank}'"}
                return {"status": "deleted", "mental_model_id": mental_model_id}
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Error deleting mental model: {e}", exc_info=True)
                return {"error": str(e)}


def _register_refresh_mental_model(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the refresh_mental_model tool."""

    if config.include_bank_id_param:

        @mcp.tool(annotations=_tool_annotations("refresh_mental_model"))
        async def refresh_mental_model(
            mental_model_id: str,
            bank_id: str | None = None,
        ) -> str:
            """
            Refresh a mental model by re-running its source query.

            Schedules an async task to re-run the source query through reflect and update the
            mental model's content with fresh results. Use this after adding new memories or
            when the mental model's content may be stale.

            Args:
                mental_model_id: The ID of the mental model to refresh
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                result = await memory.submit_async_refresh_mental_model(
                    bank_id=target_bank,
                    mental_model_id=mental_model_id,
                    request_context=_get_request_context(config),
                )
                return json.dumps(
                    {
                        "operation_id": result["operation_id"],
                        "status": "queued",
                        "message": f"Refresh queued for mental model '{mental_model_id}'.",
                    }
                )
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return json.dumps({"error": str(e)})
            except ValueError as e:
                return json.dumps({"error": str(e)})
            except Exception as e:
                logger.error(f"Error refreshing mental model: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool(annotations=_tool_annotations("refresh_mental_model"))
        async def refresh_mental_model(
            mental_model_id: str,
        ) -> dict:
            """
            Refresh a mental model by re-running its source query.

            Schedules an async task to re-run the source query through reflect and update the
            mental model's content with fresh results. Use this after adding new memories or
            when the mental model's content may be stale.

            Args:
                mental_model_id: The ID of the mental model to refresh
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                result = await memory.submit_async_refresh_mental_model(
                    bank_id=target_bank,
                    mental_model_id=mental_model_id,
                    request_context=_get_request_context(config),
                )
                return {
                    "operation_id": result["operation_id"],
                    "status": "queued",
                    "message": f"Refresh queued for mental model '{mental_model_id}'.",
                }
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return {"error": str(e)}
            except ValueError as e:
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Error refreshing mental model: {e}", exc_info=True)
                return {"error": str(e)}


def _register_clear_mental_model(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the clear_mental_model tool."""

    if config.include_bank_id_param:

        @mcp.tool(annotations=_tool_annotations("clear_mental_model"))
        async def clear_mental_model(
            mental_model_id: str,
            bank_id: str | None = None,
        ) -> str:
            """
            Clear a mental model's content so the next refresh performs a full re-synthesis.

            This is useful for delta-mode models that have accumulated drift over many
            incremental refreshes. After clearing, call refresh_mental_model to trigger
            a clean full rebuild.

            Args:
                mental_model_id: The ID of the mental model to clear
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                result = await memory.clear_mental_model(
                    bank_id=target_bank,
                    mental_model_id=mental_model_id,
                    request_context=_get_request_context(config),
                )
                if result is None:
                    return json.dumps({"error": f"Mental model '{mental_model_id}' not found"})
                return json.dumps(
                    {
                        "mental_model_id": result["id"],
                        "status": "cleared",
                        "message": f"Mental model '{mental_model_id}' content cleared. Call refresh_mental_model to rebuild.",
                    }
                )
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return json.dumps({"error": str(e)})
            except ValueError as e:
                return json.dumps({"error": str(e)})
            except Exception as e:
                logger.error(f"Error clearing mental model: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool(annotations=_tool_annotations("clear_mental_model"))
        async def clear_mental_model(
            mental_model_id: str,
        ) -> dict:
            """
            Clear a mental model's content so the next refresh performs a full re-synthesis.

            This is useful for delta-mode models that have accumulated drift over many
            incremental refreshes. After clearing, call refresh_mental_model to trigger
            a clean full rebuild.

            Args:
                mental_model_id: The ID of the mental model to clear
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                result = await memory.clear_mental_model(
                    bank_id=target_bank,
                    mental_model_id=mental_model_id,
                    request_context=_get_request_context(config),
                )
                if result is None:
                    return {"error": f"Mental model '{mental_model_id}' not found"}
                return {
                    "mental_model_id": result["id"],
                    "status": "cleared",
                    "message": f"Mental model '{mental_model_id}' content cleared. Call refresh_mental_model to rebuild.",
                }
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return {"error": str(e)}
            except ValueError as e:
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Error clearing mental model: {e}", exc_info=True)
                return {"error": str(e)}


# =========================================================================
# KNOWLEDGE BASE TOOLS
# =========================================================================
# A tree of folders and pages over mental models (see the HTTP
# /knowledge-base endpoints). Pages read as markdown documents; folders are
# containers. ``export_knowledge_base`` is deliberately NOT exposed here — it
# returns the whole bank as one markdown bundle, which belongs on the HTTP/CLI
# path rather than in an agent's context window.

# MCP tool arguments cannot express an explicit null: an omitted argument and a
# null one both arrive as ``None``, so update_knowledge_node reads this literal
# as "move to the top level". Node ids are prefixed (``kf-``/``kp-``), so it
# cannot collide with a real folder id.
KNOWLEDGE_ROOT_PARENT = "root"


def _knowledge_node_json(node: dict[str, Any]) -> dict[str, Any]:
    """Project an engine node dict into the compact JSON an MCP client sees.

    Mirrors the HTTP ``KnowledgeNode`` projection, minus the fields an agent has
    no use for (bank_id, sort_order): page metadata comes from the backing
    mental model, folders carry structure only.
    """
    is_page = node.get("kind") == "page"
    projected: dict[str, Any] = {
        "id": node["id"],
        "kind": node["kind"],
        "name": node["name"],
        "parent_id": node.get("parent_id"),
        "managed": bool(node.get("managed")),
    }
    if is_page:
        projected["mental_model_id"] = node.get("mental_model_id")
        projected["description"] = node.get("source_query")
        projected["tags"] = list(node.get("tags") or [])
        projected["timestamp"] = node.get("last_refreshed_at")
        if node.get("trigger") is not None:
            projected["trigger"] = node["trigger"]
        if node.get("is_stale") is not None:
            projected["is_stale"] = node["is_stale"]
    else:
        projected["timestamp"] = node.get("updated_at")
        projected["children"] = []
    return projected


def _knowledge_tree_json(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assemble the flat node list into nested roots (mirrors the HTTP tree)."""
    projected = {n["id"]: _knowledge_node_json(n) for n in nodes}
    roots: list[dict[str, Any]] = []
    for node in nodes:
        parent_id = node.get("parent_id")
        # Only folders can be parents (enforced on write), so the parent normally
        # carries a children list; setdefault keeps a malformed row from raising.
        if parent_id and parent_id in projected:
            projected[parent_id].setdefault("children", []).append(projected[node["id"]])
        else:
            roots.append(projected[node["id"]])
    return roots


async def _do_get_knowledge_base_tree(
    memory: MemoryEngine, target_bank: str, request_context: RequestContext
) -> dict[str, Any]:
    """Shared implementation for the get_knowledge_base_tree MCP tool variants."""
    nodes = await memory.list_knowledge_nodes(bank_id=target_bank, with_staleness=True, request_context=request_context)
    return {"roots": _knowledge_tree_json(nodes)}


async def _do_search_knowledge_base(
    memory: MemoryEngine, target_bank: str, request_context: RequestContext, *, query: str, limit: int
) -> dict[str, Any]:
    """Shared implementation for the search_knowledge_base MCP tool variants.

    ``limit`` is clamped rather than rejected: the HTTP route answers an
    out-of-range limit with a 422, but an agent that asked for 500 pages wants
    results, not a validation round trip.
    """
    results = await memory.search_knowledge_pages(
        bank_id=target_bank, query=query, limit=max(1, min(limit, 50)), request_context=request_context
    )
    return {"results": results, "total": len(results)}


async def _do_get_knowledge_page(
    memory: MemoryEngine, target_bank: str, request_context: RequestContext, *, page_id: str
) -> dict[str, Any]:
    """Shared implementation for the get_knowledge_page MCP tool variants."""
    node = await memory.get_knowledge_page(bank_id=target_bank, page_id=page_id, request_context=request_context)
    if node is None:
        return {"error": f"Knowledge page '{page_id}' not found in bank '{target_bank}'"}
    page = page_markdown.page_type(node.get("tags"))
    # The rendered document already carries the body under a frontmatter block,
    # so it is returned once rather than alongside a duplicate `body` field.
    return {
        "id": node["id"],
        "name": node["name"],
        "type": page.type,
        "description": node.get("source_query"),
        "tags": page.display_tags,
        "timestamp": node.get("last_refreshed_at") or node.get("created_at"),
        "markdown": page_markdown.render_document(node),
    }


async def _do_create_knowledge_folder(
    memory: MemoryEngine, target_bank: str, request_context: RequestContext, *, name: str, parent_id: str | None
) -> dict[str, Any]:
    """Shared implementation for the create_knowledge_folder MCP tool variants."""
    node = await memory.create_knowledge_folder(
        bank_id=target_bank, name=name, parent_id=parent_id, request_context=request_context
    )
    return _knowledge_node_json(node)


async def _do_create_knowledge_page(
    memory: MemoryEngine,
    target_bank: str,
    request_context: RequestContext,
    *,
    name: str,
    source_query: str,
    parent_id: str | None,
    tags: list[str] | None,
    max_tokens: int | None,
    trigger: MentalModelTriggerInput | None,
    refresh_after_consolidation: bool | None,
) -> dict[str, Any]:
    """Shared implementation for the create_knowledge_page MCP tool variants."""
    node = await memory.create_knowledge_page(
        bank_id=target_bank,
        name=name,
        source_query=source_query,
        content="Generating content...",
        parent_id=parent_id,
        tags=tags or None,
        max_tokens=max_tokens,
        # Only the fields the caller stated: the engine merges them over
        # KNOWLEDGE_PAGE_DEFAULT_TRIGGER, so an unmentioned setting keeps the page
        # contract (delta mode, observation-only facts, sibling pages excluded).
        trigger=_mental_model_trigger_patch(trigger, refresh_after_consolidation=refresh_after_consolidation),
        request_context=request_context,
    )
    if node is None:
        return {"error": f"A page named '{name}' already exists in this folder"}
    result = await memory.submit_async_refresh_mental_model(
        bank_id=target_bank, mental_model_id=node["mental_model_id"], request_context=request_context
    )
    return {
        "page_id": node["id"],
        "mental_model_id": node["mental_model_id"],
        "operation_id": result["operation_id"],
        "status": "created",
        "message": f"Page '{name}' created. Content is being generated asynchronously.",
    }


async def _do_update_knowledge_node(
    memory: MemoryEngine,
    target_bank: str,
    request_context: RequestContext,
    *,
    node_id: str,
    name: str | None,
    parent_id: str | None,
    source_query: str | None,
    tags: list[str] | None,
    max_tokens: int | None,
    trigger: MentalModelTriggerInput | None,
    refresh_after_consolidation: bool | None,
) -> dict[str, Any]:
    """Shared implementation for the update_knowledge_node MCP tool variants.

    Each field is applied only when provided, so a rename never resets a page's
    query and moving a page never drops its tags.
    """
    # A patch, merged over the page's CURRENT trigger by the engine, so putting a
    # page on a cron schedule does not reset how or from what it rebuilds.
    trigger_patch = _mental_model_trigger_patch(trigger, refresh_after_consolidation=refresh_after_consolidation)
    page_update = source_query is not None or tags is not None or max_tokens is not None or trigger_patch is not None
    if name is None and parent_id is None and not page_update:
        return {
            "error": "Provide name, parent_id, source_query, tags, max_tokens, "
            "trigger, and/or refresh_after_consolidation to update"
        }

    # One call, one transaction: a rename must not survive the move that fails
    # after it. This tool is driven by agents that retry on error, and a partly
    # applied patch made the retry read a tree nobody asked for.
    updated = await memory.update_knowledge_node(
        bank_id=target_bank,
        node_id=node_id,
        name=name,
        parent_id=(None if parent_id == KNOWLEDGE_ROOT_PARENT else parent_id) if parent_id is not None else KEEP_PARENT,
        source_query=source_query,
        tags=tags,
        max_tokens=max_tokens,
        trigger=trigger_patch,
        request_context=request_context,
    )
    if updated is None:
        return {"error": f"Knowledge node '{node_id}' not found in bank '{target_bank}'"}
    # A new source query means the page's content no longer answers it — rebuild.
    # Scheduled only once the patch has committed.
    if source_query is not None and updated.get("mental_model_id"):
        await memory.submit_async_refresh_mental_model(
            bank_id=target_bank, mental_model_id=updated["mental_model_id"], request_context=request_context
        )
    return _knowledge_node_json(updated)


async def _do_delete_knowledge_node(
    memory: MemoryEngine, target_bank: str, request_context: RequestContext, *, node_id: str
) -> dict[str, Any]:
    """Shared implementation for the delete_knowledge_node MCP tool variants."""
    deleted = await memory.delete_knowledge_node(bank_id=target_bank, node_id=node_id, request_context=request_context)
    if not deleted:
        return {"error": f"Knowledge node '{node_id}' not found in bank '{target_bank}'"}
    return {"status": "deleted", "node_id": node_id}


def _register_get_knowledge_base_tree(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the get_knowledge_base_tree tool."""

    if config.include_bank_id_param:

        @mcp.tool(annotations=_tool_annotations("get_knowledge_base_tree"))
        async def get_knowledge_base_tree(
            bank_id: str | None = None,
        ) -> str:
            """
            Browse the knowledge base as a nested tree of folders and pages.

            Start here to discover what the bank documents: each page is a living
            markdown document synthesized from the bank's memories, and folders
            group them. Use get_knowledge_page to read a page's content, or
            search_knowledge_base when you know what you are looking for.

            Pages report `is_stale`: false means the page is provably up to date;
            true means something was written since its last refresh, so it MAY be
            out of date.

            Args:
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                tree = await _do_get_knowledge_base_tree(memory, target_bank, _get_request_context(config))
                return json.dumps(tree, indent=2, default=str)
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return json.dumps({"error": str(e)})
            except Exception as e:
                logger.error(f"Error getting knowledge base tree: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool(annotations=_tool_annotations("get_knowledge_base_tree"))
        async def get_knowledge_base_tree() -> dict:
            """
            Browse the knowledge base as a nested tree of folders and pages.

            Start here to discover what the bank documents: each page is a living
            markdown document synthesized from the bank's memories, and folders
            group them. Use get_knowledge_page to read a page's content, or
            search_knowledge_base when you know what you are looking for.

            Pages report `is_stale`: false means the page is provably up to date;
            true means something was written since its last refresh, so it MAY be
            out of date.
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                return await _do_get_knowledge_base_tree(memory, target_bank, _get_request_context(config))
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Error getting knowledge base tree: {e}", exc_info=True)
                return {"error": str(e)}


def _register_search_knowledge_base(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the search_knowledge_base tool."""

    if config.include_bank_id_param:

        @mcp.tool(annotations=_tool_annotations("search_knowledge_base"))
        async def search_knowledge_base(
            query: str,
            limit: int = 10,
            bank_id: str | None = None,
        ) -> str:
            """
            Find knowledge pages by relevance (hybrid keyword + semantic search).

            Searches page names and content, returning ranked pages with a short
            snippet each. Read a hit in full with get_knowledge_page. This searches
            the curated knowledge base only — use recall to search raw memories.

            Args:
                query: What to search for
                limit: Maximum pages to return (1-50, default: 10)
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                results = await _do_search_knowledge_base(
                    memory, target_bank, _get_request_context(config), query=query, limit=limit
                )
                return json.dumps(results, indent=2, default=str)
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return json.dumps({"error": str(e)})
            except Exception as e:
                logger.error(f"Error searching knowledge base: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool(annotations=_tool_annotations("search_knowledge_base"))
        async def search_knowledge_base(
            query: str,
            limit: int = 10,
        ) -> dict:
            """
            Find knowledge pages by relevance (hybrid keyword + semantic search).

            Searches page names and content, returning ranked pages with a short
            snippet each. Read a hit in full with get_knowledge_page. This searches
            the curated knowledge base only — use recall to search raw memories.

            Args:
                query: What to search for
                limit: Maximum pages to return (1-50, default: 10)
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                return await _do_search_knowledge_base(
                    memory, target_bank, _get_request_context(config), query=query, limit=limit
                )
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Error searching knowledge base: {e}", exc_info=True)
                return {"error": str(e)}


def _register_get_knowledge_page(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the get_knowledge_page tool."""

    if config.include_bank_id_param:

        @mcp.tool(annotations=_tool_annotations("get_knowledge_page"))
        async def get_knowledge_page(
            page_id: str,
            bank_id: str | None = None,
        ) -> str:
            """
            Read a knowledge page as a markdown document.

            Returns the page's YAML frontmatter (id, type, title, description,
            tags, timestamp) followed by its synthesized markdown body. Discover
            page ids with get_knowledge_base_tree or search_knowledge_base.

            Args:
                page_id: The ID of the page to read (a `kp-...` node id)
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                page = await _do_get_knowledge_page(memory, target_bank, _get_request_context(config), page_id=page_id)
                return json.dumps(page, indent=2, default=str)
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return json.dumps({"error": str(e)})
            except Exception as e:
                logger.error(f"Error getting knowledge page: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool(annotations=_tool_annotations("get_knowledge_page"))
        async def get_knowledge_page(
            page_id: str,
        ) -> dict:
            """
            Read a knowledge page as a markdown document.

            Returns the page's YAML frontmatter (id, type, title, description,
            tags, timestamp) followed by its synthesized markdown body. Discover
            page ids with get_knowledge_base_tree or search_knowledge_base.

            Args:
                page_id: The ID of the page to read (a `kp-...` node id)
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                return await _do_get_knowledge_page(memory, target_bank, _get_request_context(config), page_id=page_id)
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Error getting knowledge page: {e}", exc_info=True)
                return {"error": str(e)}


def _register_create_knowledge_folder(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the create_knowledge_folder tool."""

    if config.include_bank_id_param:

        @mcp.tool(annotations=_tool_annotations("create_knowledge_folder"))
        async def create_knowledge_folder(
            name: str,
            parent_id: str | None = None,
            bank_id: str | None = None,
        ) -> str:
            """
            Create a folder in the knowledge base.

            Folders group pages; they hold no content of their own.

            Args:
                name: Folder name
                parent_id: Optional parent folder id (a `kf-...` node id). Omit to create at the top level.
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                node = await _do_create_knowledge_folder(
                    memory, target_bank, _get_request_context(config), name=name, parent_id=parent_id
                )
                return json.dumps(node, indent=2, default=str)
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return json.dumps({"error": str(e)})
            except ValueError as e:
                return json.dumps({"error": str(e)})
            except Exception as e:
                logger.error(f"Error creating knowledge folder: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool(annotations=_tool_annotations("create_knowledge_folder"))
        async def create_knowledge_folder(
            name: str,
            parent_id: str | None = None,
        ) -> dict:
            """
            Create a folder in the knowledge base.

            Folders group pages; they hold no content of their own.

            Args:
                name: Folder name
                parent_id: Optional parent folder id (a `kf-...` node id). Omit to create at the top level.
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                return await _do_create_knowledge_folder(
                    memory, target_bank, _get_request_context(config), name=name, parent_id=parent_id
                )
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return {"error": str(e)}
            except ValueError as e:
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Error creating knowledge folder: {e}", exc_info=True)
                return {"error": str(e)}


def _register_create_knowledge_page(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the create_knowledge_page tool."""

    if config.include_bank_id_param:

        @mcp.tool(annotations=_tool_annotations("create_knowledge_page"))
        async def create_knowledge_page(
            name: str,
            source_query: str,
            parent_id: str | None = None,
            tags: list[str] | None = None,
            max_tokens: int | None = None,
            trigger: MentalModelTriggerInput | None = None,
            refresh_after_consolidation: bool | None = None,
            bank_id: str | None = None,
        ) -> str:
            """
            Create a knowledge page — a living document answering a question.

            The page's content is synthesized from the bank's memories by running
            source_query, asynchronously: use the returned operation_id to track
            completion, then read it with get_knowledge_page. By default the page
            keeps itself current, rebuilding after each consolidation.

            EXAMPLES:
            - name="Deployment Runbook", source_query="How is this service deployed and rolled back?"
            - name="Team Preferences", source_query="What tools and conventions does the team prefer?"

            Args:
                name: Page name (must be unique within its folder)
                source_query: The question this page answers and rebuilds itself from
                parent_id: Optional parent folder id (a `kf-...` node id). Omit to create at the top level.
                tags: Optional tags scoping which memories the page is built from
                max_tokens: Maximum tokens for the generated content (default: 4096)
                trigger: Refresh policy for this page — when it rebuilds itself (mode,
                    refresh_after_consolidation, refresh_cron) and what it rebuilds from
                    (fact_types, tags_match, tag_groups, exclude_mental_models,
                    recall_max_tokens, ...). Omitted fields keep the knowledge-page
                    defaults: incremental (delta) rebuilds from consolidated observations
                    after each consolidation, ignoring sibling pages. Set refresh_cron
                    instead to move the page onto a fixed UTC schedule.
                refresh_after_consolidation: Legacy shorthand for trigger.refresh_after_consolidation.
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                result = await _do_create_knowledge_page(
                    memory,
                    target_bank,
                    _get_request_context(config),
                    name=name,
                    source_query=source_query,
                    parent_id=parent_id,
                    tags=tags,
                    max_tokens=max_tokens,
                    trigger=trigger,
                    refresh_after_consolidation=refresh_after_consolidation,
                )
                return json.dumps(result, indent=2, default=str)
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return json.dumps({"error": str(e)})
            except ValueError as e:
                return json.dumps({"error": str(e)})
            except Exception as e:
                logger.error(f"Error creating knowledge page: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool(annotations=_tool_annotations("create_knowledge_page"))
        async def create_knowledge_page(
            name: str,
            source_query: str,
            parent_id: str | None = None,
            tags: list[str] | None = None,
            max_tokens: int | None = None,
            trigger: MentalModelTriggerInput | None = None,
            refresh_after_consolidation: bool | None = None,
        ) -> dict:
            """
            Create a knowledge page — a living document answering a question.

            The page's content is synthesized from the bank's memories by running
            source_query, asynchronously: use the returned operation_id to track
            completion, then read it with get_knowledge_page. By default the page
            keeps itself current, rebuilding after each consolidation.

            EXAMPLES:
            - name="Deployment Runbook", source_query="How is this service deployed and rolled back?"
            - name="Team Preferences", source_query="What tools and conventions does the team prefer?"

            Args:
                name: Page name (must be unique within its folder)
                source_query: The question this page answers and rebuilds itself from
                parent_id: Optional parent folder id (a `kf-...` node id). Omit to create at the top level.
                tags: Optional tags scoping which memories the page is built from
                max_tokens: Maximum tokens for the generated content (default: 4096)
                trigger: Refresh policy for this page — when it rebuilds itself (mode,
                    refresh_after_consolidation, refresh_cron) and what it rebuilds from
                    (fact_types, tags_match, tag_groups, exclude_mental_models,
                    recall_max_tokens, ...). Omitted fields keep the knowledge-page
                    defaults: incremental (delta) rebuilds from consolidated observations
                    after each consolidation, ignoring sibling pages. Set refresh_cron
                    instead to move the page onto a fixed UTC schedule.
                refresh_after_consolidation: Legacy shorthand for trigger.refresh_after_consolidation.
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                return await _do_create_knowledge_page(
                    memory,
                    target_bank,
                    _get_request_context(config),
                    name=name,
                    source_query=source_query,
                    parent_id=parent_id,
                    tags=tags,
                    max_tokens=max_tokens,
                    trigger=trigger,
                    refresh_after_consolidation=refresh_after_consolidation,
                )
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return {"error": str(e)}
            except ValueError as e:
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Error creating knowledge page: {e}", exc_info=True)
                return {"error": str(e)}


def _register_update_knowledge_node(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the update_knowledge_node tool."""

    if config.include_bank_id_param:

        @mcp.tool(annotations=_tool_annotations("update_knowledge_node"))
        async def update_knowledge_node(
            node_id: str,
            name: str | None = None,
            parent_id: str | None = None,
            source_query: str | None = None,
            tags: list[str] | None = None,
            max_tokens: int | None = None,
            trigger: MentalModelTriggerInput | None = None,
            refresh_after_consolidation: bool | None = None,
            bank_id: str | None = None,
        ) -> str:
            """
            Rename or move a folder/page, and/or update a page's options.

            Only the arguments you pass are changed; everything else keeps its
            current value. Changing source_query schedules an async refresh so the
            page rebuilds against the new question.

            Args:
                node_id: The ID of the folder (`kf-...`) or page (`kp-...`) to update
                name: New name for the node
                parent_id: Folder id to move the node into, or "root" to move it to the top level
                source_query: Pages only — the new question the page answers
                tags: Pages only — replacement tag list (pass [] to clear)
                max_tokens: Pages only — new maximum tokens for the generated content
                trigger: Pages only — refresh policy fields to change: mode,
                    refresh_after_consolidation, refresh_cron, fact_types, tags_match,
                    tag_groups, exclude_mental_models, recall_max_tokens, and so on. This
                    is a PATCH: fields you omit keep their current values, so putting a
                    page on a cron schedule does not reset its delta mode or its
                    observation-only scope.
                refresh_after_consolidation: Pages only — legacy shorthand for
                    trigger.refresh_after_consolidation
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                result = await _do_update_knowledge_node(
                    memory,
                    target_bank,
                    _get_request_context(config),
                    node_id=node_id,
                    name=name,
                    parent_id=parent_id,
                    source_query=source_query,
                    tags=tags,
                    max_tokens=max_tokens,
                    trigger=trigger,
                    refresh_after_consolidation=refresh_after_consolidation,
                )
                return json.dumps(result, indent=2, default=str)
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return json.dumps({"error": str(e)})
            except ValueError as e:
                return json.dumps({"error": str(e)})
            except Exception as e:
                logger.error(f"Error updating knowledge node: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool(annotations=_tool_annotations("update_knowledge_node"))
        async def update_knowledge_node(
            node_id: str,
            name: str | None = None,
            parent_id: str | None = None,
            source_query: str | None = None,
            tags: list[str] | None = None,
            max_tokens: int | None = None,
            trigger: MentalModelTriggerInput | None = None,
            refresh_after_consolidation: bool | None = None,
        ) -> dict:
            """
            Rename or move a folder/page, and/or update a page's options.

            Only the arguments you pass are changed; everything else keeps its
            current value. Changing source_query schedules an async refresh so the
            page rebuilds against the new question.

            Args:
                node_id: The ID of the folder (`kf-...`) or page (`kp-...`) to update
                name: New name for the node
                parent_id: Folder id to move the node into, or "root" to move it to the top level
                source_query: Pages only — the new question the page answers
                tags: Pages only — replacement tag list (pass [] to clear)
                max_tokens: Pages only — new maximum tokens for the generated content
                trigger: Pages only — refresh policy fields to change: mode,
                    refresh_after_consolidation, refresh_cron, fact_types, tags_match,
                    tag_groups, exclude_mental_models, recall_max_tokens, and so on. This
                    is a PATCH: fields you omit keep their current values, so putting a
                    page on a cron schedule does not reset its delta mode or its
                    observation-only scope.
                refresh_after_consolidation: Pages only — legacy shorthand for
                    trigger.refresh_after_consolidation
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                return await _do_update_knowledge_node(
                    memory,
                    target_bank,
                    _get_request_context(config),
                    node_id=node_id,
                    name=name,
                    parent_id=parent_id,
                    source_query=source_query,
                    tags=tags,
                    max_tokens=max_tokens,
                    trigger=trigger,
                    refresh_after_consolidation=refresh_after_consolidation,
                )
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return {"error": str(e)}
            except ValueError as e:
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Error updating knowledge node: {e}", exc_info=True)
                return {"error": str(e)}


def _register_delete_knowledge_node(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the delete_knowledge_node tool."""

    if config.include_bank_id_param:

        @mcp.tool(annotations=_tool_annotations("delete_knowledge_node"))
        async def delete_knowledge_node(
            node_id: str,
            bank_id: str | None = None,
        ) -> str:
            """
            Delete a knowledge-base folder or page and everything under it.

            Deleting a folder also deletes its whole subtree, and each deleted page
            takes its backing mental model with it. This cannot be undone.

            Args:
                node_id: The ID of the folder (`kf-...`) or page (`kp-...`) to delete
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                result = await _do_delete_knowledge_node(
                    memory, target_bank, _get_request_context(config), node_id=node_id
                )
                return json.dumps(result)
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return json.dumps({"error": str(e)})
            except Exception as e:
                logger.error(f"Error deleting knowledge node: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool(annotations=_tool_annotations("delete_knowledge_node"))
        async def delete_knowledge_node(
            node_id: str,
        ) -> dict:
            """
            Delete a knowledge-base folder or page and everything under it.

            Deleting a folder also deletes its whole subtree, and each deleted page
            takes its backing mental model with it. This cannot be undone.

            Args:
                node_id: The ID of the folder (`kf-...`) or page (`kp-...`) to delete
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                return await _do_delete_knowledge_node(
                    memory, target_bank, _get_request_context(config), node_id=node_id
                )
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Error deleting knowledge node: {e}", exc_info=True)
                return {"error": str(e)}


# =========================================================================
# DIRECTIVE TOOLS
# =========================================================================


def _register_list_directives(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the list_directives tool."""

    if config.include_bank_id_param:

        @mcp.tool(annotations=_tool_annotations("list_directives"))
        async def list_directives(
            tags: list[str] | None = None,
            active_only: bool = True,
            limit: int = 100,
            offset: int = 0,
            bank_id: str | None = None,
        ) -> str:
            """
            List directives for a memory bank.

            Directives are instructions that guide how the memory engine processes and
            responds to queries. They influence reflect behavior and memory organization.

            Args:
                tags: Optional tags to filter by
                active_only: If True, only return active directives (default: True)
                limit: Maximum number of results (default: 100)
                offset: Pagination offset (default: 0). Page until the returned items add up to 'total'.
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                page = await memory.list_directives(
                    target_bank,
                    tags=tags,
                    active_only=active_only,
                    limit=limit,
                    offset=offset,
                    request_context=_get_request_context(config),
                )
                return json.dumps({"items": page.items, "total": page.total}, indent=2, default=str)
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return json.dumps({"error": str(e)})
            except Exception as e:
                logger.error(f"Error listing directives: {e}", exc_info=True)
                return f'{{"error": "{e}", "items": []}}'

    else:

        @mcp.tool(annotations=_tool_annotations("list_directives"))
        async def list_directives(
            tags: list[str] | None = None,
            active_only: bool = True,
            limit: int = 100,
            offset: int = 0,
        ) -> dict:
            """
            List directives for this memory bank.

            Directives are instructions that guide how the memory engine processes and
            responds to queries. They influence reflect behavior and memory organization.

            Args:
                tags: Optional tags to filter by
                active_only: If True, only return active directives (default: True)
                limit: Maximum number of results (default: 100)
                offset: Pagination offset (default: 0). Page until the returned items add up to 'total'.
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured", "items": []}

                page = await memory.list_directives(
                    target_bank,
                    tags=tags,
                    active_only=active_only,
                    limit=limit,
                    offset=offset,
                    request_context=_get_request_context(config),
                )
                return {"items": page.items, "total": page.total}
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Error listing directives: {e}", exc_info=True)
                return {"error": str(e), "items": []}


def _register_create_directive(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the create_directive tool."""

    if config.include_bank_id_param:

        @mcp.tool(annotations=_tool_annotations("create_directive"))
        async def create_directive(
            name: str,
            content: str,
            priority: int = 0,
            is_active: bool = True,
            tags: list[str] | None = None,
            bank_id: str | None = None,
        ) -> str:
            """
            Create a new directive for a memory bank.

            Directives guide how the memory engine processes queries and generates reflections.

            Args:
                name: Human-readable name for the directive
                content: The directive content/instructions
                priority: Priority level (higher = more important, default: 0)
                is_active: Whether the directive is active (default: True)
                tags: Optional tags for filtering
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                directive = await memory.create_directive(
                    target_bank,
                    name=name,
                    content=content,
                    priority=priority,
                    is_active=is_active,
                    tags=tags,
                    request_context=_get_request_context(config),
                )
                return json.dumps(directive, indent=2, default=str)
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return json.dumps({"error": str(e)})
            except Exception as e:
                logger.error(f"Error creating directive: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool(annotations=_tool_annotations("create_directive"))
        async def create_directive(
            name: str,
            content: str,
            priority: int = 0,
            is_active: bool = True,
            tags: list[str] | None = None,
        ) -> dict:
            """
            Create a new directive for this memory bank.

            Directives guide how the memory engine processes queries and generates reflections.

            Args:
                name: Human-readable name for the directive
                content: The directive content/instructions
                priority: Priority level (higher = more important, default: 0)
                is_active: Whether the directive is active (default: True)
                tags: Optional tags for filtering
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                directive = await memory.create_directive(
                    target_bank,
                    name=name,
                    content=content,
                    priority=priority,
                    is_active=is_active,
                    tags=tags,
                    request_context=_get_request_context(config),
                )
                return directive
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Error creating directive: {e}", exc_info=True)
                return {"error": str(e)}


def _register_delete_directive(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the delete_directive tool."""

    if config.include_bank_id_param:

        @mcp.tool(annotations=_tool_annotations("delete_directive"))
        async def delete_directive(
            directive_id: str,
            bank_id: str | None = None,
        ) -> str:
            """
            Delete a directive.

            Permanently removes a directive from the memory bank.

            Args:
                directive_id: The ID of the directive to delete
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                deleted = await memory.delete_directive(
                    target_bank,
                    directive_id,
                    request_context=_get_request_context(config),
                )
                if not deleted:
                    return json.dumps({"error": f"Directive '{directive_id}' not found"})
                return json.dumps({"status": "deleted", "directive_id": directive_id})
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return json.dumps({"error": str(e)})
            except Exception as e:
                logger.error(f"Error deleting directive: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool(annotations=_tool_annotations("delete_directive"))
        async def delete_directive(
            directive_id: str,
        ) -> dict:
            """
            Delete a directive.

            Permanently removes a directive from this memory bank.

            Args:
                directive_id: The ID of the directive to delete
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                deleted = await memory.delete_directive(
                    target_bank,
                    directive_id,
                    request_context=_get_request_context(config),
                )
                if not deleted:
                    return {"error": f"Directive '{directive_id}' not found"}
                return {"status": "deleted", "directive_id": directive_id}
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Error deleting directive: {e}", exc_info=True)
                return {"error": str(e)}


# =========================================================================
# MEMORY BROWSING TOOLS
# =========================================================================


def _register_list_memories(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the list_memories tool."""

    if config.include_bank_id_param:

        @mcp.tool(annotations=_tool_annotations("list_memories"))
        async def list_memories(
            type: str | None = None,
            q: str | None = None,
            limit: int = 100,
            offset: int = 0,
            bank_id: str | None = None,
            tags: list[str] | None = None,
            tags_match: TagsMatch = "any",
        ) -> str:
            """
            Browse stored memories with optional filtering.

            Lists memory units (facts) stored in the bank. Unlike recall, this is a direct
            browse/search without relevance ranking.

            Args:
                type: Filter by fact type: 'world', 'experience', or 'observation'
                q: Optional text search query to filter memories
                limit: Maximum number of results (default: 100)
                offset: Pagination offset (default: 0)
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
                tags: Optional list of tag names to filter by.
                tags_match: How to combine tags: 'any' (OR, default) or 'all' (AND)
                    both also include untagged memories; 'any_strict'/'all_strict'
                    exclude untagged; 'exact' matches the tag set exactly.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                result = await memory.list_memory_units(
                    target_bank,
                    fact_type=type,
                    search_query=q,
                    limit=limit,
                    offset=offset,
                    tags=tags,
                    tags_match=tags_match,
                    request_context=_get_request_context(config),
                )
                return json.dumps(result, indent=2, default=str)
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return json.dumps({"error": str(e)})
            except Exception as e:
                logger.error(f"Error listing memories: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool(annotations=_tool_annotations("list_memories"))
        async def list_memories(
            type: str | None = None,
            q: str | None = None,
            limit: int = 100,
            offset: int = 0,
            tags: list[str] | None = None,
            tags_match: TagsMatch = "any",
        ) -> dict:
            """
            Browse stored memories with optional filtering.

            Lists memory units (facts) stored in the bank. Unlike recall, this is a direct
            browse/search without relevance ranking.

            Args:
                type: Filter by fact type: 'world', 'experience', or 'observation'
                q: Optional text search query to filter memories
                limit: Maximum number of results (default: 100)
                offset: Pagination offset (default: 0)
                tags: Optional list of tag names to filter by.
                tags_match: How to combine tags: 'any' (OR, default) or 'all' (AND)
                    both also include untagged memories; 'any_strict'/'all_strict'
                    exclude untagged; 'exact' matches the tag set exactly.
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                result = await memory.list_memory_units(
                    target_bank,
                    fact_type=type,
                    search_query=q,
                    limit=limit,
                    offset=offset,
                    tags=tags,
                    tags_match=tags_match,
                    request_context=_get_request_context(config),
                )
                return result
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Error listing memories: {e}", exc_info=True)
                return {"error": str(e)}


def _register_get_memory(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the get_memory tool."""

    if config.include_bank_id_param:

        @mcp.tool(annotations=_tool_annotations("get_memory"))
        async def get_memory(
            memory_id: str,
            bank_id: str | None = None,
        ) -> str:
            """
            Get a specific memory by ID.

            Returns the full memory unit including content, metadata, and timestamps.

            Args:
                memory_id: The ID of the memory to retrieve
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                result = await memory.get_memory_unit(
                    target_bank,
                    memory_id,
                    request_context=_get_request_context(config),
                )
                if result is None:
                    return json.dumps({"error": f"Memory '{memory_id}' not found"})
                return json.dumps(result, indent=2, default=str)
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return json.dumps({"error": str(e)})
            except Exception as e:
                logger.error(f"Error getting memory: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool(annotations=_tool_annotations("get_memory"))
        async def get_memory(
            memory_id: str,
        ) -> dict:
            """
            Get a specific memory by ID.

            Returns the full memory unit including content, metadata, and timestamps.

            Args:
                memory_id: The ID of the memory to retrieve
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                result = await memory.get_memory_unit(
                    target_bank,
                    memory_id,
                    request_context=_get_request_context(config),
                )
                if result is None:
                    return {"error": f"Memory '{memory_id}' not found"}
                return result
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Error getting memory: {e}", exc_info=True)
                return {"error": str(e)}


def _register_update_memory(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the update_memory (edit) tool."""

    _EDIT_DOC = """
            Edit a memory unit to correct what was extracted.

            Pass any of text / context / occurred_start / occurred_end / fact_type /
            entities. For context and the dates, "" clears the field and omitting it
            leaves it unchanged; entities replaces the fact's entity set ([] detaches
            all). The memory is re-embedded and its derived observations, links, and
            graph are recomputed automatically.

            resolve_entities controls how the names in entities are matched. The
            default True behaves like retain and may resolve a name onto a similar
            entity that already exists, which silently discards a correction when the
            bank holds a near-duplicate name. Pass False whenever you are correcting a
            fact deliberately: an existing entity is then reused only on a
            case-insensitive name match, and any other name becomes its own entity.

            Only raw world/experience facts can be edited; observations are derived.
            To retire or restore a fact, use invalidate_memory instead.
    """

    if config.include_bank_id_param:

        @mcp.tool(description=_EDIT_DOC, annotations=_tool_annotations("update_memory"))
        async def update_memory(
            memory_id: str,
            text: str | None = None,
            context: str | None = None,
            occurred_start: str | None = None,
            occurred_end: str | None = None,
            fact_type: str | None = None,
            entities: list[str] | None = None,
            resolve_entities: bool = True,
            bank_id: str | None = None,
        ) -> str:
            """
            Args:
                memory_id: The ID of the memory unit to edit.
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                result = await memory.update_memory_unit(
                    target_bank,
                    memory_id,
                    text=text,
                    context=context,
                    occurred_start=occurred_start,
                    occurred_end=occurred_end,
                    new_fact_type=fact_type,
                    entities=entities,
                    resolve_entities=resolve_entities,
                    request_context=_get_request_context(config),
                )
                if result is None:
                    return json.dumps({"error": f"Memory '{memory_id}' not found"})
                return json.dumps(result, indent=2, default=str)
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return json.dumps({"error": str(e)})
            except ValueError as e:
                return json.dumps({"error": str(e)})
            except Exception as e:
                logger.error(f"Error updating memory: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool(description=_EDIT_DOC, annotations=_tool_annotations("update_memory"))
        async def update_memory(
            memory_id: str,
            text: str | None = None,
            context: str | None = None,
            occurred_start: str | None = None,
            occurred_end: str | None = None,
            fact_type: str | None = None,
            entities: list[str] | None = None,
            resolve_entities: bool = True,
        ) -> dict:
            """
            Args:
                memory_id: The ID of the memory unit to edit.
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                result = await memory.update_memory_unit(
                    target_bank,
                    memory_id,
                    text=text,
                    context=context,
                    occurred_start=occurred_start,
                    occurred_end=occurred_end,
                    new_fact_type=fact_type,
                    entities=entities,
                    resolve_entities=resolve_entities,
                    request_context=_get_request_context(config),
                )
                if result is None:
                    return {"error": f"Memory '{memory_id}' not found"}
                return result
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return {"error": str(e)}
            except ValueError as e:
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Error updating memory: {e}", exc_info=True)
                return {"error": str(e)}


def _register_invalidate_memory(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the invalidate_memory (retire / restore) tool."""

    _INVALIDATE_DOC = """
            Soft-retire a memory unit (or restore a previously retired one).

            Invalidating moves the fact out of the active set: it's excluded from
            recall, consolidation, and the knowledge graph, its links are pruned, and
            its derived observations are recomputed without it — but it's kept for
            audit and is fully reversible. Pass restore=True to bring it back.

            Only raw world/experience facts can be invalidated; observations are derived.
    """

    if config.include_bank_id_param:

        @mcp.tool(description=_INVALIDATE_DOC, annotations=_tool_annotations("invalidate_memory"))
        async def invalidate_memory(
            memory_id: str,
            reason: str | None = None,
            restore: bool = False,
            bank_id: str | None = None,
        ) -> str:
            """
            Args:
                memory_id: The ID of the memory unit to retire (or restore).
                reason: Optional free-text reason recorded when invalidating.
                restore: Set True to restore a previously invalidated fact.
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                result = await memory.update_memory_unit(
                    target_bank,
                    memory_id,
                    state="valid" if restore else "invalidated",
                    reason=reason,
                    request_context=_get_request_context(config),
                )
                if result is None:
                    return json.dumps({"error": f"Memory '{memory_id}' not found"})
                return json.dumps(result, indent=2, default=str)
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return json.dumps({"error": str(e)})
            except ValueError as e:
                return json.dumps({"error": str(e)})
            except Exception as e:
                logger.error(f"Error invalidating memory: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool(description=_INVALIDATE_DOC, annotations=_tool_annotations("invalidate_memory"))
        async def invalidate_memory(
            memory_id: str,
            reason: str | None = None,
            restore: bool = False,
        ) -> dict:
            """
            Args:
                memory_id: The ID of the memory unit to retire (or restore).
                reason: Optional free-text reason recorded when invalidating.
                restore: Set True to restore a previously invalidated fact.
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                result = await memory.update_memory_unit(
                    target_bank,
                    memory_id,
                    state="valid" if restore else "invalidated",
                    reason=reason,
                    request_context=_get_request_context(config),
                )
                if result is None:
                    return {"error": f"Memory '{memory_id}' not found"}
                return result
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return {"error": str(e)}
            except ValueError as e:
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Error invalidating memory: {e}", exc_info=True)
                return {"error": str(e)}


# =========================================================================
# DOCUMENT TOOLS
# =========================================================================


def _register_list_documents(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the list_documents tool."""

    if config.include_bank_id_param:

        @mcp.tool(annotations=_tool_annotations("list_documents"))
        async def list_documents(
            q: str | None = None,
            limit: int = 100,
            bank_id: str | None = None,
        ) -> str:
            """
            List documents in a memory bank.

            Documents are containers for related memories (e.g., a conversation transcript,
            a meeting notes file). Memories created with a document_id are grouped under that document.

            Args:
                q: Optional search query to filter documents
                limit: Maximum number of results (default: 100)
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                result = await memory.list_documents(
                    target_bank,
                    search_query=q,
                    limit=limit,
                    request_context=_get_request_context(config),
                )
                return json.dumps(result, indent=2, default=str)
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return json.dumps({"error": str(e)})
            except Exception as e:
                logger.error(f"Error listing documents: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool(annotations=_tool_annotations("list_documents"))
        async def list_documents(
            q: str | None = None,
            limit: int = 100,
        ) -> dict:
            """
            List documents in this memory bank.

            Documents are containers for related memories (e.g., a conversation transcript,
            a meeting notes file). Memories created with a document_id are grouped under that document.

            Args:
                q: Optional search query to filter documents
                limit: Maximum number of results (default: 100)
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                result = await memory.list_documents(
                    target_bank,
                    search_query=q,
                    limit=limit,
                    request_context=_get_request_context(config),
                )
                return result
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Error listing documents: {e}", exc_info=True)
                return {"error": str(e)}


def _register_get_document(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the get_document tool."""

    if config.include_bank_id_param:

        @mcp.tool(annotations=_tool_annotations("get_document"))
        async def get_document(
            document_id: str,
            bank_id: str | None = None,
        ) -> str:
            """
            Get a specific document by ID.

            Returns document metadata and associated memory information.

            Args:
                document_id: The ID of the document to retrieve
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                result = await memory.get_document(
                    document_id,
                    target_bank,
                    request_context=_get_request_context(config),
                )
                if result is None:
                    return json.dumps({"error": f"Document '{document_id}' not found"})
                return json.dumps(result, indent=2, default=str)
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return json.dumps({"error": str(e)})
            except Exception as e:
                logger.error(f"Error getting document: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool(annotations=_tool_annotations("get_document"))
        async def get_document(
            document_id: str,
        ) -> dict:
            """
            Get a specific document by ID.

            Returns document metadata and associated memory information.

            Args:
                document_id: The ID of the document to retrieve
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                result = await memory.get_document(
                    document_id,
                    target_bank,
                    request_context=_get_request_context(config),
                )
                if result is None:
                    return {"error": f"Document '{document_id}' not found"}
                return result
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Error getting document: {e}", exc_info=True)
                return {"error": str(e)}


def _register_delete_document(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the delete_document tool."""

    if config.include_bank_id_param:

        @mcp.tool(annotations=_tool_annotations("delete_document"))
        async def delete_document(
            document_id: str,
            bank_id: str | None = None,
        ) -> str:
            """
            Delete a document and its associated memories.

            Permanently removes a document and all memories linked to it.

            Args:
                document_id: The ID of the document to delete
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                result = await memory.delete_document(
                    document_id,
                    target_bank,
                    request_context=_get_request_context(config),
                )
                return json.dumps({"status": "deleted", "document_id": document_id, **result}, default=str)
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return json.dumps({"error": str(e)})
            except Exception as e:
                logger.error(f"Error deleting document: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool(annotations=_tool_annotations("delete_document"))
        async def delete_document(
            document_id: str,
        ) -> dict:
            """
            Delete a document and its associated memories.

            Permanently removes a document and all memories linked to it.

            Args:
                document_id: The ID of the document to delete
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                result = await memory.delete_document(
                    document_id,
                    target_bank,
                    request_context=_get_request_context(config),
                )
                return {"status": "deleted", "document_id": document_id, **result}
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Error deleting document: {e}", exc_info=True)
                return {"error": str(e)}


# =========================================================================
# OPERATION TOOLS
# =========================================================================


def _register_list_operations(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the list_operations tool."""

    if config.include_bank_id_param:

        @mcp.tool(annotations=_tool_annotations("list_operations"))
        async def list_operations(
            status: str | None = None,
            limit: int = 20,
            bank_id: str | None = None,
        ) -> str:
            """
            List async operations for a memory bank.

            Operations track background tasks like retain processing, mental model refresh, etc.

            Args:
                status: Filter by status: 'pending', 'running', 'completed', 'failed', 'cancelled'
                limit: Maximum number of results (default: 20)
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                result = await memory.list_operations(
                    target_bank,
                    status=status,
                    limit=limit,
                    request_context=_get_request_context(config),
                )
                return json.dumps(result, indent=2, default=str)
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return json.dumps({"error": str(e)})
            except Exception as e:
                logger.error(f"Error listing operations: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool(annotations=_tool_annotations("list_operations"))
        async def list_operations(
            status: str | None = None,
            limit: int = 20,
        ) -> dict:
            """
            List async operations for this memory bank.

            Operations track background tasks like retain processing, mental model refresh, etc.

            Args:
                status: Filter by status: 'pending', 'running', 'completed', 'failed', 'cancelled'
                limit: Maximum number of results (default: 20)
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                result = await memory.list_operations(
                    target_bank,
                    status=status,
                    limit=limit,
                    request_context=_get_request_context(config),
                )
                return result
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Error listing operations: {e}", exc_info=True)
                return {"error": str(e)}


def _register_get_operation(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the get_operation tool."""

    if config.include_bank_id_param:

        @mcp.tool(annotations=_tool_annotations("get_operation"))
        async def get_operation(
            operation_id: str,
            bank_id: str | None = None,
        ) -> str:
            """
            Get the status of an async operation.

            Check progress of background tasks like retain processing or mental model refresh.

            Args:
                operation_id: The ID of the operation to check
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                result = await memory.get_operation_status(
                    target_bank,
                    operation_id,
                    request_context=_get_request_context(config),
                )
                return json.dumps(result, indent=2, default=str)
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return json.dumps({"error": str(e)})
            except Exception as e:
                logger.error(f"Error getting operation: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool(annotations=_tool_annotations("get_operation"))
        async def get_operation(
            operation_id: str,
        ) -> dict:
            """
            Get the status of an async operation.

            Check progress of background tasks like retain processing or mental model refresh.

            Args:
                operation_id: The ID of the operation to check
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                result = await memory.get_operation_status(
                    target_bank,
                    operation_id,
                    request_context=_get_request_context(config),
                )
                return result
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Error getting operation: {e}", exc_info=True)
                return {"error": str(e)}


def _register_cancel_operation(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the cancel_operation tool."""

    if config.include_bank_id_param:

        @mcp.tool(annotations=_tool_annotations("cancel_operation"))
        async def cancel_operation(
            operation_id: str,
            bank_id: str | None = None,
        ) -> str:
            """
            Cancel a pending or running async operation.

            Args:
                operation_id: The ID of the operation to cancel
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                result = await memory.cancel_operation(
                    target_bank,
                    operation_id,
                    request_context=_get_request_context(config),
                )
                return json.dumps(result, indent=2, default=str)
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return json.dumps({"error": str(e)})
            except Exception as e:
                logger.error(f"Error cancelling operation: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool(annotations=_tool_annotations("cancel_operation"))
        async def cancel_operation(
            operation_id: str,
        ) -> dict:
            """
            Cancel a pending or running async operation.

            Args:
                operation_id: The ID of the operation to cancel
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                result = await memory.cancel_operation(
                    target_bank,
                    operation_id,
                    request_context=_get_request_context(config),
                )
                return result
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Error cancelling operation: {e}", exc_info=True)
                return {"error": str(e)}


# =========================================================================
# TAGS & BANK TOOLS
# =========================================================================


def _register_list_tags(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the list_tags tool."""

    if config.include_bank_id_param:

        @mcp.tool(annotations=_tool_annotations("list_tags"))
        async def list_tags(
            q: str | None = None,
            limit: int = 100,
            bank_id: str | None = None,
        ) -> str:
            """
            List tags used in a memory bank.

            Tags are used to organize and filter memories, directives, and mental models.

            Args:
                q: Optional pattern to filter tags (e.g., 'project:*')
                limit: Maximum number of results (default: 100)
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                result = await memory.list_tags(
                    target_bank,
                    pattern=q,
                    limit=limit,
                    request_context=_get_request_context(config),
                )
                return json.dumps(result, indent=2, default=str)
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return json.dumps({"error": str(e)})
            except Exception as e:
                logger.error(f"Error listing tags: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool(annotations=_tool_annotations("list_tags"))
        async def list_tags(
            q: str | None = None,
            limit: int = 100,
        ) -> dict:
            """
            List tags used in this memory bank.

            Tags are used to organize and filter memories, directives, and mental models.

            Args:
                q: Optional pattern to filter tags (e.g., 'project:*')
                limit: Maximum number of results (default: 100)
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                result = await memory.list_tags(
                    target_bank,
                    pattern=q,
                    limit=limit,
                    request_context=_get_request_context(config),
                )
                return result
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Error listing tags: {e}", exc_info=True)
                return {"error": str(e)}


def _register_get_bank(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the get_bank tool."""

    if config.include_bank_id_param:

        @mcp.tool(annotations=_tool_annotations("get_bank"))
        async def get_bank(
            bank_id: str | None = None,
        ) -> str:
            """
            Get the profile of a memory bank.

            Returns bank metadata including name, disposition, and mission.

            Args:
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                profile = await memory.get_bank_profile(
                    target_bank,
                    request_context=_get_request_context(config),
                    create_if_missing=False,
                )
                if profile is None:
                    return json.dumps({"error": f"Bank '{target_bank}' not found"})
                if "disposition" in profile and hasattr(profile["disposition"], "model_dump"):
                    profile["disposition"] = profile["disposition"].model_dump()
                return json.dumps(profile, indent=2, default=str)
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return json.dumps({"error": str(e)})
            except Exception as e:
                logger.error(f"Error getting bank: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool(annotations=_tool_annotations("get_bank"))
        async def get_bank() -> dict:
            """
            Get the profile of this memory bank.

            Returns bank metadata including name, disposition, and mission.
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                profile = await memory.get_bank_profile(
                    target_bank,
                    request_context=_get_request_context(config),
                    create_if_missing=False,
                )
                if profile is None:
                    return {"error": f"Bank '{target_bank}' not found"}
                if "disposition" in profile and hasattr(profile["disposition"], "model_dump"):
                    profile["disposition"] = profile["disposition"].model_dump()
                return profile
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Error getting bank: {e}", exc_info=True)
                return {"error": str(e)}


def _register_get_bank_stats(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the get_bank_stats tool (multi-bank only)."""

    @mcp.tool(annotations=_tool_annotations("get_bank_stats"))
    async def get_bank_stats(
        bank_id: str | None = None,
    ) -> str:
        """
        Get statistics for a memory bank.

        Returns counts of nodes, links, and other metrics.

        Args:
            bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
        """
        try:
            target_bank = bank_id or config.bank_id_resolver()
            if target_bank is None:
                return '{"error": "No bank_id configured"}'

            result = await memory.get_bank_stats(
                target_bank,
                request_context=_get_request_context(config),
            )
            return json.dumps(result, indent=2, default=str)
        except OperationValidationError as e:
            logger.warning(f"Operation rejected: {e}")
            return json.dumps({"error": str(e)})
        except Exception as e:
            logger.error(f"Error getting bank stats: {e}", exc_info=True)
            return f'{{"error": "{e}"}}'


async def _do_update_bank(
    memory: MemoryEngine,
    target_bank: str,
    request_context: RequestContext,
    *,
    name: str | None = None,
    mission: str | None = None,
    config_updates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Shared implementation for update_bank MCP tool variants.

    Args:
        name: Display name (stored in banks table).
        mission: Deprecated alias for reflect_mission — mapped into config_updates.
        config_updates: Arbitrary config overrides passed to MemoryEngine.update_bank_config().
            Supports all configurable fields (retain_mission, disposition_*, etc.).
            The config resolver validates keys and rejects non-configurable/credential fields.
    """
    # Merge deprecated mission alias into config_updates as reflect_mission
    effective_config: dict[str, Any] = dict(config_updates) if config_updates else {}
    if mission is not None and "reflect_mission" not in effective_config:
        effective_config["reflect_mission"] = mission

    return await memory.update_bank(
        target_bank,
        name=name,
        config_updates=effective_config or None,
        request_context=request_context,
    )


def _register_update_bank(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the update_bank tool."""

    if config.include_bank_id_param:

        @mcp.tool(annotations=_tool_annotations("update_bank"))
        async def update_bank(
            name: str | None = None,
            mission: str | None = None,
            config_updates: dict[str, Any] | None = None,
            bank_id: str | None = None,
        ) -> str:
            """
            Update a memory bank's configuration.

            Updates the bank's name and/or any bank-level configuration fields.
            Only provided fields will be updated; omitted fields remain unchanged.

            Args:
                name: Human-friendly display name for the bank.
                mission: Deprecated alias for config_updates.reflect_mission.
                config_updates: Dictionary of configuration fields to update. Supports all
                    bank-configurable fields including:
                    - reflect_mission: Mission/context for Reflect operations.
                    - retain_mission: Steers what gets extracted during retain().
                    - retain_extraction_mode: 'concise' (default), 'verbose', or 'custom'.
                    - retain_custom_instructions: Custom extraction prompt (active when mode is 'custom').
                    - retain_chunk_size: Target maximum characters for each content chunk.
                    - retain_structured_chunk_size: Maximum characters for a single JSONL line or conversation turn to keep whole.
                    - retain_chunk_batch_size: Number of chunks to process in parallel.
                    - enable_observations: Toggle observation consolidation after retain().
                    - observations_mission: Controls observation synthesis rules.
                    - disposition_skepticism: Critical evaluation level (1-5).
                    - disposition_literalism: Literal vs. abstract interpretation (1-5).
                    - disposition_empathy: Emotional context consideration (1-5).
                    - entity_labels: Controlled vocabulary for entity classification.
                    - entities_allow_free_form: Allow labels outside entity_labels.
                    - recall_include_chunks: Include raw chunks in recall results.
                    - recall_max_tokens: Max tokens for recall results.
                    - mcp_enabled_tools: Tool allowlist for this bank.
                    Any configurable field name is accepted (use Python field names).
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                result = await _do_update_bank(
                    memory,
                    target_bank,
                    _get_request_context(config),
                    name=name,
                    mission=mission,
                    config_updates=config_updates,
                )
                return json.dumps(result, indent=2, default=str)
            except (OperationValidationError, ValueError) as e:
                logger.warning(f"Operation rejected: {e}")
                return json.dumps({"error": str(e)})
            except Exception as e:
                logger.error(f"Error updating bank: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool(annotations=_tool_annotations("update_bank"))
        async def update_bank(
            name: str | None = None,
            mission: str | None = None,
            config_updates: dict[str, Any] | None = None,
        ) -> dict:
            """
            Update this memory bank's configuration.

            Updates the bank's name and/or any bank-level configuration fields.
            Only provided fields will be updated; omitted fields remain unchanged.

            Args:
                name: Human-friendly display name for the bank.
                mission: Deprecated alias for config_updates.reflect_mission.
                config_updates: Dictionary of configuration fields to update. Supports all
                    bank-configurable fields including:
                    - reflect_mission: Mission/context for Reflect operations.
                    - retain_mission: Steers what gets extracted during retain().
                    - retain_extraction_mode: 'concise' (default), 'verbose', or 'custom'.
                    - retain_custom_instructions: Custom extraction prompt (active when mode is 'custom').
                    - retain_chunk_size: Target maximum characters for each content chunk.
                    - retain_structured_chunk_size: Maximum characters for a single JSONL line or conversation turn to keep whole.
                    - retain_chunk_batch_size: Number of chunks to process in parallel.
                    - enable_observations: Toggle observation consolidation after retain().
                    - observations_mission: Controls observation synthesis rules.
                    - disposition_skepticism: Critical evaluation level (1-5).
                    - disposition_literalism: Literal vs. abstract interpretation (1-5).
                    - disposition_empathy: Emotional context consideration (1-5).
                    - entity_labels: Controlled vocabulary for entity classification.
                    - entities_allow_free_form: Allow labels outside entity_labels.
                    - recall_include_chunks: Include raw chunks in recall results.
                    - recall_max_tokens: Max tokens for recall results.
                    - mcp_enabled_tools: Tool allowlist for this bank.
                    Any configurable field name is accepted (use Python field names).
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                result = await _do_update_bank(
                    memory,
                    target_bank,
                    _get_request_context(config),
                    name=name,
                    mission=mission,
                    config_updates=config_updates,
                )
                return result
            except (OperationValidationError, ValueError) as e:
                logger.warning(f"Operation rejected: {e}")
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Error updating bank: {e}", exc_info=True)
                return {"error": str(e)}


def _register_delete_bank(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the delete_bank tool."""

    if config.include_bank_id_param:

        @mcp.tool(annotations=_tool_annotations("delete_bank"))
        async def delete_bank(
            bank_id: str | None = None,
        ) -> str:
            """
            Delete a memory bank and all its data.

            WARNING: This permanently deletes the bank and all its memories, documents,
            mental models, directives, and other data. This action cannot be undone.

            Args:
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                result = await memory.delete_bank(
                    target_bank,
                    request_context=_get_request_context(config),
                )
                return json.dumps({"status": "deleted", "bank_id": target_bank, **result}, default=str)
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return json.dumps({"error": str(e)})
            except Exception as e:
                logger.error(f"Error deleting bank: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool(annotations=_tool_annotations("delete_bank"))
        async def delete_bank() -> dict:
            """
            Delete this memory bank and all its data.

            WARNING: This permanently deletes the bank and all its memories, documents,
            mental models, directives, and other data. This action cannot be undone.
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                result = await memory.delete_bank(
                    target_bank,
                    request_context=_get_request_context(config),
                )
                return {"status": "deleted", "bank_id": target_bank, **result}
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Error deleting bank: {e}", exc_info=True)
                return {"error": str(e)}


def _register_clear_memories(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the clear_memories tool."""

    if config.include_bank_id_param:

        @mcp.tool(annotations=_tool_annotations("clear_memories"))
        async def clear_memories(
            type: str | None = None,
            bank_id: str | None = None,
        ) -> str:
            """
            Clear all memories from a bank without deleting the bank itself.

            Optionally filter by fact type to only clear specific kinds of memories.

            Args:
                type: Optional fact type filter: 'world', 'experience', or 'observation'. If not specified, clears all.
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                result = await memory.delete_bank(
                    target_bank,
                    fact_type=type,
                    delete_bank_profile=False,
                    request_context=_get_request_context(config),
                )
                return json.dumps({"status": "cleared", "bank_id": target_bank, **result}, default=str)
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return json.dumps({"error": str(e)})
            except Exception as e:
                logger.error(f"Error clearing memories: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool(annotations=_tool_annotations("clear_memories"))
        async def clear_memories(
            type: str | None = None,
        ) -> dict:
            """
            Clear all memories from this bank without deleting the bank itself.

            Optionally filter by fact type to only clear specific kinds of memories.

            Args:
                type: Optional fact type filter: 'world', 'experience', or 'observation'. If not specified, clears all.
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                result = await memory.delete_bank(
                    target_bank,
                    fact_type=type,
                    delete_bank_profile=False,
                    request_context=_get_request_context(config),
                )
                return {"status": "cleared", "bank_id": target_bank, **result}
            except OperationValidationError as e:
                logger.warning(f"Operation rejected: {e}")
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Error clearing memories: {e}", exc_info=True)
                return {"error": str(e)}
