"""Abstract interface for MemoryEngine public methods.

This module defines the public API that HTTP endpoints and extensions should use
to interact with the memory system. All methods require a RequestContext for
authentication when a TenantExtension is configured.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hindsight_api.engine.memory_engine import Budget
    from hindsight_api.engine.response_models import RecallResult, ReflectResult
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
    ) -> dict[str, Any]:
        """
        Retain a batch of memory items.

        Args:
            bank_id: The memory bank ID.
            contents: List of content dicts with 'content', optional 'event_date',
                     'context', 'metadata', 'document_id'.
            request_context: Request context for authentication.

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
    ) -> dict[str, Any]:
        """
        Get bank profile including disposition and background.

        Args:
            bank_id: The memory bank ID.
            request_context: Request context for authentication.

        Returns:
            Bank profile dict.
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
    async def merge_bank_background(
        self,
        bank_id: str,
        new_info: str,
        *,
        update_disposition: bool = True,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """
        Merge new background information into bank profile.

        Args:
            bank_id: The memory bank ID.
            new_info: New background information to merge.
            update_disposition: Whether to infer disposition from background.
            request_context: Request context for authentication.

        Returns:
            Updated background info.
        """
        ...

    @abstractmethod
    async def delete_bank(
        self,
        bank_id: str,
        *,
        fact_type: str | None = None,
        request_context: "RequestContext",
    ) -> dict[str, int]:
        """
        Delete a bank or its memories.

        Args:
            bank_id: The memory bank ID.
            fact_type: If specified, only delete memories of this type.
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
    async def delete_memory_unit(
        self,
        unit_id: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """
        Delete a specific memory unit.

        Args:
            unit_id: The memory unit ID.
            request_context: Request context for authentication.

        Returns:
            Deletion result.
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
        limit: int = 100,
        offset: int = 0,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """
        List documents with pagination.

        Args:
            bank_id: The memory bank ID.
            search_query: Search query.
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
        request_context: "RequestContext",
    ) -> list[dict[str, Any]]:
        """
        List entities for a bank.

        Args:
            bank_id: The memory bank ID.
            limit: Maximum results.
            request_context: Request context for authentication.

        Returns:
            List of entity dicts.
        """
        ...

    @abstractmethod
    async def get_entity_observations(
        self,
        bank_id: str,
        entity_id: str,
        *,
        limit: int = 10,
        request_context: "RequestContext",
    ) -> list[Any]:
        """
        Get observations for an entity.

        Args:
            bank_id: The memory bank ID.
            entity_id: The entity ID.
            limit: Maximum observations.
            request_context: Request context for authentication.

        Returns:
            List of EntityObservation objects.
        """
        ...

    @abstractmethod
    async def regenerate_entity_observations(
        self,
        bank_id: str,
        entity_id: str,
        entity_name: str,
        *,
        request_context: "RequestContext",
    ) -> None:
        """
        Regenerate observations for an entity.

        Args:
            bank_id: The memory bank ID.
            entity_id: The entity ID.
            entity_name: The entity's canonical name.
            request_context: Request context for authentication.
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
            Dict with node_counts, link_counts, link_counts_by_fact_type,
            link_breakdown, and operations stats.
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
    ) -> list[dict[str, Any]]:
        """
        List async operations for a bank.

        Args:
            bank_id: The memory bank ID.
            request_context: Request context for authentication.

        Returns:
            List of operation dicts with id, task_type, status, etc.
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
        background: str | None = None,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """
        Update bank name and/or background.

        Args:
            bank_id: The memory bank ID.
            name: New bank name (optional).
            background: New background text (optional, replaces existing).
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
    ) -> dict[str, Any]:
        """
        Submit a batch retain operation to run asynchronously.

        Args:
            bank_id: The memory bank ID.
            contents: List of content dicts to retain.
            request_context: Request context for authentication.

        Returns:
            Dict with operation_id and items_count.
        """
        ...

    # =========================================================================
    # Feedback Signal Methods
    # =========================================================================

    @abstractmethod
    async def submit_signals(
        self,
        bank_id: str,
        signals: list[dict[str, Any]],
        *,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """
        Submit feedback signals for facts.

        Args:
            bank_id: The memory bank ID.
            signals: List of signal dicts with fact_id, signal_type, confidence, query, context.
            request_context: Request context for authentication.

        Returns:
            Dict with success status and list of updated fact IDs.
        """
        ...

    @abstractmethod
    async def get_fact_usefulness_stats(
        self,
        bank_id: str,
        fact_id: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, Any] | None:
        """
        Get usefulness statistics for a specific fact.

        Args:
            bank_id: The memory bank ID.
            fact_id: The fact UUID.
            request_context: Request context for authentication.

        Returns:
            Dict with usefulness stats or None if no signals exist.
        """
        ...

    @abstractmethod
    async def get_bank_usefulness_stats(
        self,
        bank_id: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """
        Get aggregate usefulness statistics for a bank.

        Args:
            bank_id: The memory bank ID.
            request_context: Request context for authentication.

        Returns:
            Dict with bank-level usefulness stats.
        """
        ...
