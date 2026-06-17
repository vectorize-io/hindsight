"""Abstract interface for MemoryEngine public methods.

This module defines the public API that HTTP endpoints and extensions should use
to interact with the memory system. All methods require a RequestContext for
authentication when a TenantExtension is configured.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hindsight_api.engine.memory_engine import BankLlmHealthInfo, Budget
    from hindsight_api.engine.response_models import RecallResult, ReflectResult
    from hindsight_api.engine.search.tags import TagsMatch
    from hindsight_api.models import RequestContext


class MemoryEngineInterface(ABC):
    """
    Abstract interface for the Memory Engine.

    This defines the public API that should be used by HTTP endpoints and extensions.
    All methods require a RequestContext for authentication.
    """

    # =========================================================================
    # Health & Status
    # =========================================================================

    @abstractmethod
    async def health_check(self) -> dict:
        """
        Check the health of the memory system.

        Returns:
            Dict with 'status' key ('healthy' or 'unhealthy') and additional info.
        """
        ...

    # =========================================================================
    # Core Memory Operations
    # =========================================================================

    @abstractmethod
    async def retain_batch_async(
        self,
        bank_id: str,
        contents: list[dict[str, Any]],
        *,
        request_context: "RequestContext",
        document_tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Retain a batch of memory items.

        Args:
            bank_id: The memory bank ID.
            contents: List of content dicts with 'content', optional 'event_date',
                     'context', 'metadata', 'document_id', and per-item 'tags'.
            request_context: Request context for authentication.
            document_tags: Optional tags applied to all items in the batch.

        Returns:
            Dict with processing results.
        """
        ...

    @abstractmethod
    async def recall_async(
        self,
        bank_id: str,
        query: str,
        *,
        budget: "Budget | None" = None,
        max_tokens: int = 4096,
        enable_trace: bool = False,
        fact_type: list[str] | None = None,
        question_date: datetime | None = None,
        include_entities: bool = False,
        max_entity_tokens: int = 500,
        include_chunks: bool = False,
        max_chunk_tokens: int = 8192,
        request_context: "RequestContext",
    ) -> "RecallResult":
        """
        Recall memories relevant to a query.

        Args:
            bank_id: The memory bank ID.
            query: The search query.
            budget: Search budget (LOW, MID, HIGH).
            max_tokens: Maximum tokens in response.
            enable_trace: Include trace information.
            fact_type: Filter by fact types.
            question_date: Context date for temporal relevance.
            include_entities: Include entity observations.
            max_entity_tokens: Max tokens for entity observations.
            include_chunks: Include raw chunks.
            max_chunk_tokens: Max tokens for chunks.
            request_context: Request context for authentication.

        Returns:
            RecallResult with matching memories.
        """
        ...

    @abstractmethod
    async def reflect_async(
        self,
        bank_id: str,
        query: str,
        *,
        budget: "Budget | None" = None,
        context: str | None = None,
        max_tokens: int = 4096,
        response_schema: dict | None = None,
        request_context: "RequestContext",
    ) -> "ReflectResult":
        """
        Reflect on a query and generate a thoughtful response.

        Args:
            bank_id: The memory bank ID.
            query: The question to reflect on.
            budget: Search budget for retrieving context.
            context: Additional context for the reflection.
            max_tokens: Maximum tokens for the response.
            response_schema: Optional JSON Schema for structured output.
            request_context: Request context for authentication.

        Returns:
            ReflectResult with generated response and supporting facts.
        """
        ...

    # =========================================================================
    # Bank Management
    # =========================================================================

    @abstractmethod
    async def list_banks(
        self,
        *,
        request_context: "RequestContext",
    ) -> list[dict[str, Any]]:
        """
        List all memory banks.

        Args:
            request_context: Request context for authentication.

        Returns:
            List of bank info dicts.
        """
        ...

    @abstractmethod
    async def get_bank_profile(
        self,
        bank_id: str,
        *,
        request_context: "RequestContext",
        create_if_missing: bool = True,
    ) -> dict[str, Any] | None:
        """
        Get bank profile including disposition and mission.

        Args:
            bank_id: The memory bank ID.
            request_context: Request context for authentication.
            create_if_missing: If True (default), the bank is auto-created
                with defaults if it does not exist. Pass False to make this
                a strict read — returns None if the bank does not exist.

        Returns:
            Bank profile dict with bank_id, name, disposition, and mission,
            or None when create_if_missing=False and the bank does not
            exist.
        """
        ...

    @abstractmethod
    async def update_bank_disposition(
        self,
        bank_id: str,
        disposition: dict[str, int],
        *,
        request_context: "RequestContext",
    ) -> None:
        """
        Update bank disposition traits.

        Args:
            bank_id: The memory bank ID.
            disposition: Dict with trait values.
            request_context: Request context for authentication.
        """
        ...

    @abstractmethod
    async def merge_bank_mission(
        self,
        bank_id: str,
        new_info: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """
        Merge new mission information into bank profile.

        Args:
            bank_id: The memory bank ID.
            new_info: New mission information to merge.
            request_context: Request context for authentication.

        Returns:
            Updated mission info.
        """
        ...

    @abstractmethod
    async def set_bank_mission(
        self,
        bank_id: str,
        mission: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """
        Set the bank's mission (replaces existing).

        Args:
            bank_id: The memory bank ID.
            mission: The mission text.
            request_context: Request context for authentication.

        Returns:
            Dict with bank_id and mission.
        """
        ...

    @abstractmethod
    async def delete_bank(
        self,
        bank_id: str,
        *,
        fact_type: str | None = None,
        delete_bank_profile: bool = True,
        request_context: "RequestContext",
    ) -> dict[str, int]:
        """
        Delete a bank or its memories.

        Args:
            bank_id: The memory bank ID.
            fact_type: If specified, only delete memories of this type.
            delete_bank_profile: If True, also delete the bank profile row itself.
                If False, only delete memories/entities/documents but preserve the bank.
            request_context: Request context for authentication.

        Returns:
            Dict with deletion counts.
        """
        ...

    # =========================================================================
    # Memory Units
    # =========================================================================

    @abstractmethod
    async def list_memory_units(
        self,
        bank_id: str,
        *,
        fact_type: str | None = None,
        search_query: str | None = None,
        limit: int = 100,
        offset: int = 0,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """
        List memory units with pagination.

        Args:
            bank_id: The memory bank ID.
            fact_type: Filter by fact type.
            search_query: Full-text search query.
            limit: Maximum results.
            offset: Pagination offset.
            request_context: Request context for authentication.

        Returns:
            Dict with 'items', 'total', 'limit', 'offset'.
        """
        ...

    @abstractmethod
    async def get_graph_data(
        self,
        bank_id: str,
        *,
        fact_type: str | None = None,
        limit: int = 1000,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """
        Get graph data for visualization.

        Args:
            bank_id: The memory bank ID.
            fact_type: Filter by fact type.
            limit: Maximum number of items to return (default: 1000).
            request_context: Request context for authentication.

        Returns:
            Dict with nodes, edges, table_rows, total_units, limit.
        """
        ...

    # =========================================================================
    # Documents
    # =========================================================================

    @abstractmethod
    async def list_documents(
        self,
        bank_id: str,
        *,
        search_query: str | None = None,
        tags: list[str] | None = None,
        tags_match: "TagsMatch" = "any_strict",
        limit: int = 100,
        offset: int = 0,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """
        List documents with pagination.

        Args:
            bank_id: The memory bank ID.
            search_query: Case-insensitive substring filter on document ID.
            tags: Filter by tags.
            tags_match: How to match tags (any, all, any_strict, all_strict).
            limit: Maximum results.
            offset: Pagination offset.
            request_context: Request context for authentication.

        Returns:
            Dict with 'items', 'total', 'limit', 'offset'.
        """
        ...

    @abstractmethod
    async def get_document(
        self,
        document_id: str,
        bank_id: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, Any] | None:
        """
        Get a specific document.

        Args:
            document_id: The document ID.
            bank_id: The memory bank ID.
            request_context: Request context for authentication.

        Returns:
            Document dict or None if not found.
        """
        ...

    @abstractmethod
    async def delete_document(
        self,
        document_id: str,
        bank_id: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, int]:
        """
        Delete a document and its memory units.

        Args:
            document_id: The document ID.
            bank_id: The memory bank ID.
            request_context: Request context for authentication.

        Returns:
            Dict with deletion counts.
        """
        ...

    @abstractmethod
    async def get_chunk(
        self,
        chunk_id: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, Any] | None:
        """
        Get a specific chunk.

        Args:
            chunk_id: The chunk ID.
            request_context: Request context for authentication.

        Returns:
            Chunk dict or None if not found.
        """
        ...

    # =========================================================================
    # Entities
    # =========================================================================

    @abstractmethod
    async def list_entities(
        self,
        bank_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """
        List entities for a bank with pagination.

        Args:
            bank_id: The memory bank ID.
            limit: Maximum results.
            offset: Offset for pagination.
            request_context: Request context for authentication.

        Returns:
            Dict with items, total, limit, offset.
        """
        ...

    # =========================================================================
    # Statistics & Operations
    # =========================================================================

    @abstractmethod
    async def get_bank_stats(
        self,
        bank_id: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """
        Get statistics about memory nodes and links for a bank.

        Args:
            bank_id: The memory bank ID.
            request_context: Request context for authentication.

        Returns:
            Dict with node_counts, link_counts, link_counts_by_fact_type
            (deprecated, returns empty), link_breakdown (deprecated, returns
            empty), and operations stats.
        """
        ...

    @abstractmethod
    async def get_bank_freshness(
        self,
        bank_id: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """
        Get consolidation freshness for a bank.

        Cheap alternative to get_bank_stats when callers only need
        last_consolidated_at / pending_consolidation / failed_consolidation.

        Returns:
            Dict with last_consolidated_at (ISO-8601 string or None),
            pending_consolidation (int), and failed_consolidation (int).
        """
        ...

    @abstractmethod
    async def check_bank_llm(
        self,
        bank_id: str,
        *,
        request_context: "RequestContext",
    ) -> "BankLlmHealthInfo":
        """
        Probe the LLM consolidation would use for this bank. Deliberate connectivity
        test (one real minimal call); never returns the API key. See
        MemoryEngine.check_bank_llm.
        """
        ...

    @abstractmethod
    async def get_entity(
        self,
        bank_id: str,
        entity_id: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, Any] | None:
        """
        Get entity details including metadata and observations.

        Args:
            bank_id: The memory bank ID.
            entity_id: The entity ID.
            request_context: Request context for authentication.

        Returns:
            Entity dict with id, canonical_name, mention_count, first_seen,
            last_seen, metadata, and observations. None if not found.
        """
        ...

    @abstractmethod
    async def list_operations(
        self,
        bank_id: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """
        List async operations for a bank.

        Args:
            bank_id: The memory bank ID.
            request_context: Request context for authentication.

        Returns:
            Dict with 'total' (int) and 'operations' (list of operation dicts).
        """
        ...

    @abstractmethod
    async def cancel_operation(
        self,
        bank_id: str,
        operation_id: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """
        Cancel a pending async operation.

        Args:
            bank_id: The memory bank ID.
            operation_id: The operation ID to cancel.
            request_context: Request context for authentication.

        Returns:
            Dict with success status and message.

        Raises:
            ValueError: If operation not found.
        """
        ...

    @abstractmethod
    async def update_bank(
        self,
        bank_id: str,
        *,
        name: str | None = None,
        mission: str | None = None,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """
        Update bank name and/or mission.

        Args:
            bank_id: The memory bank ID.
            name: New bank name (optional).
            mission: New mission text (optional, replaces existing).
            request_context: Request context for authentication.

        Returns:
            Updated bank profile dict.
        """
        ...

    @abstractmethod
    async def submit_async_retain(
        self,
        bank_id: str,
        contents: list[dict[str, Any]],
        *,
        request_context: "RequestContext",
        document_tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Submit a batch retain operation to run asynchronously.

        Args:
            bank_id: The memory bank ID.
            contents: List of content dicts to retain.
            request_context: Request context for authentication.
            document_tags: Optional tags applied to all items in the async batch.

        Returns:
            Dict with operation_id and items_count.
        """
        ...
