"""
Clean, pythonic wrapper for the Hindsight API client.

This file is MAINTAINED and NOT auto-generated. It provides a high-level,
easy-to-use interface on top of the auto-generated OpenAPI client.
"""

import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime

import hindsight_client_api
from hindsight_client_api.api import default_api
from hindsight_client_api.models import (
    recall_request,
    retain_request,
    memory_item,
    reflect_request,
)
from hindsight_client_api.models.retain_response import RetainResponse
from hindsight_client_api.models.recall_response import RecallResponse
from hindsight_client_api.models.recall_result import RecallResult
from hindsight_client_api.models.reflect_response import ReflectResponse
from hindsight_client_api.models.list_memory_units_response import ListMemoryUnitsResponse
from hindsight_client_api.models.bank_profile_response import BankProfileResponse


def _run_async(coro):
    """Run an async coroutine synchronously."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coro)


class Hindsight:
    """
    High-level, easy-to-use Hindsight API client.

    Example:
        ```python
        from hindsight_client import Hindsight

        client = Hindsight(base_url="http://localhost:8888")

        # Store a memory
        client.retain(bank_id="alice", content="Alice loves AI")

        # Recall memories
        response = client.recall(bank_id="alice", query="What does Alice like?")
        for r in response.results:
            print(r.text)

        # Generate contextual answer
        answer = client.reflect(bank_id="alice", query="What are my interests?")
        ```
    """

    def __init__(self, base_url: str, timeout: float = 30.0):
        """
        Initialize the Hindsight client.

        Args:
            base_url: The base URL of the Hindsight API server
            timeout: Request timeout in seconds (default: 30.0)
        """
        config = hindsight_client_api.Configuration(host=base_url)
        self._api_client = hindsight_client_api.ApiClient(config)
        self._api = default_api.DefaultApi(self._api_client)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def close(self):
        """Close the API client."""
        if self._api_client:
            _run_async(self._api_client.close())

    # Simplified methods for main operations

    def retain(
        self,
        bank_id: str,
        content: str,
        timestamp: Optional[datetime] = None,
        context: Optional[str] = None,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> RetainResponse:
        """
        Store a single memory (simplified interface).

        Args:
            bank_id: The memory bank ID
            content: Memory content
            timestamp: Optional event timestamp
            context: Optional context description
            document_id: Optional document ID for grouping
            metadata: Optional user-defined metadata

        Returns:
            RetainResponse with success status
        """
        return self.retain_batch(
            bank_id=bank_id,
            items=[{"content": content, "timestamp": timestamp, "context": context, "metadata": metadata}],
            document_id=document_id,
        )

    def retain_batch(
        self,
        bank_id: str,
        items: List[Dict[str, Any]],
        document_id: Optional[str] = None,
        retain_async: bool = False,
    ) -> RetainResponse:
        """
        Store multiple memories in batch.

        Args:
            bank_id: The memory bank ID
            items: List of memory items with 'content' and optional 'timestamp', 'context', 'metadata', 'document_id'
            document_id: Optional document ID for grouping memories (applied to items that don't have their own)
            retain_async: If True, process asynchronously in background (default: False)

        Returns:
            RetainResponse with success status and item count
        """
        memory_items = [
            memory_item.MemoryItem(
                content=item["content"],
                timestamp=item.get("timestamp"),
                context=item.get("context"),
                metadata=item.get("metadata"),
                # Use item's document_id if provided, otherwise fall back to batch-level document_id
                document_id=item.get("document_id") or document_id,
            )
            for item in items
        ]

        request_obj = retain_request.RetainRequest(
            items=memory_items,
            async_=retain_async,
        )

        return _run_async(self._api.retain_memories(bank_id, request_obj))

    def recall(
        self,
        bank_id: str,
        query: str,
        types: Optional[List[str]] = None,
        max_tokens: int = 4096,
        budget: str = "mid",
        trace: bool = False,
        query_timestamp: Optional[str] = None,
        include_entities: bool = False,
        max_entity_tokens: int = 500,
        include_chunks: bool = False,
        max_chunk_tokens: int = 8192,
    ) -> RecallResponse:
        """
        Recall memories using semantic similarity.

        Args:
            bank_id: The memory bank ID
            query: Search query
            types: Optional list of fact types to filter (world, experience, opinion, observation)
            max_tokens: Maximum tokens in results (default: 4096)
            budget: Budget level for recall - "low", "mid", or "high" (default: "mid")
            trace: Enable trace output (default: False)
            query_timestamp: Optional ISO format date string (e.g., '2023-05-30T23:40:00')
            include_entities: Include entity observations in results (default: False)
            max_entity_tokens: Maximum tokens for entity observations (default: 500)
            include_chunks: Include raw text chunks in results (default: False)
            max_chunk_tokens: Maximum tokens for chunks (default: 8192)

        Returns:
            RecallResponse with results, optional entities, optional chunks, and optional trace
        """
        from hindsight_client_api.models import include_options, entity_include_options, chunk_include_options

        include_opts = include_options.IncludeOptions(
            entities=entity_include_options.EntityIncludeOptions(max_tokens=max_entity_tokens) if include_entities else None,
            chunks=chunk_include_options.ChunkIncludeOptions(max_tokens=max_chunk_tokens) if include_chunks else None,
        )

        request_obj = recall_request.RecallRequest(
            query=query,
            types=types,
            budget=budget,
            max_tokens=max_tokens,
            trace=trace,
            query_timestamp=query_timestamp,
            include=include_opts,
        )

        return _run_async(self._api.recall_memories(bank_id, request_obj))

    def reflect(
        self,
        bank_id: str,
        query: str,
        budget: str = "low",
        context: Optional[str] = None,
    ) -> ReflectResponse:
        """
        Generate a contextual answer based on bank identity and memories.

        Args:
            bank_id: The memory bank ID
            query: The question or prompt
            budget: Budget level for reflection - "low", "mid", or "high" (default: "low")
            context: Optional additional context

        Returns:
            ReflectResponse with answer text and optionally facts used
        """
        request_obj = reflect_request.ReflectRequest(
            query=query,
            budget=budget,
            context=context,
        )

        return _run_async(self._api.reflect(bank_id, request_obj))

    def list_memories(
        self,
        bank_id: str,
        type: Optional[str] = None,
        search_query: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> ListMemoryUnitsResponse:
        """List memory units with pagination."""
        return _run_async(self._api.list_memories(
            bank_id=bank_id,
            type=type,
            q=search_query,
            limit=limit,
            offset=offset,
        ))

    def create_bank(
        self,
        bank_id: str,
        name: Optional[str] = None,
        background: Optional[str] = None,
        disposition: Optional[Dict[str, float]] = None,
    ) -> BankProfileResponse:
        """Create or update a memory bank."""
        from hindsight_client_api.models import create_bank_request, disposition_traits

        disposition_obj = None
        if disposition:
            disposition_obj = disposition_traits.DispositionTraits(**disposition)

        request_obj = create_bank_request.CreateBankRequest(
            name=name,
            background=background,
            disposition=disposition_obj,
        )

        return _run_async(self._api.create_or_update_bank(bank_id, request_obj))

    # Async methods (native async, no _run_async wrapper)

    async def aretain_batch(
        self,
        bank_id: str,
        items: List[Dict[str, Any]],
        document_id: Optional[str] = None,
        retain_async: bool = False,
    ) -> RetainResponse:
        """
        Store multiple memories in batch (async).

        Args:
            bank_id: The memory bank ID
            items: List of memory items with 'content' and optional 'timestamp', 'context', 'metadata', 'document_id'
            document_id: Optional document ID for grouping memories (applied to items that don't have their own)
            retain_async: If True, process asynchronously in background (default: False)

        Returns:
            RetainResponse with success status and item count
        """
        memory_items = [
            memory_item.MemoryItem(
                content=item["content"],
                timestamp=item.get("timestamp"),
                context=item.get("context"),
                metadata=item.get("metadata"),
                # Use item's document_id if provided, otherwise fall back to batch-level document_id
                document_id=item.get("document_id") or document_id,
            )
            for item in items
        ]

        request_obj = retain_request.RetainRequest(
            items=memory_items,
            async_=retain_async,
        )

        return await self._api.retain_memories(bank_id, request_obj)

    async def aretain(
        self,
        bank_id: str,
        content: str,
        timestamp: Optional[datetime] = None,
        context: Optional[str] = None,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> RetainResponse:
        """
        Store a single memory (async).

        Args:
            bank_id: The memory bank ID
            content: Memory content
            timestamp: Optional event timestamp
            context: Optional context description
            document_id: Optional document ID for grouping
            metadata: Optional user-defined metadata

        Returns:
            RetainResponse with success status
        """
        return await self.aretain_batch(
            bank_id=bank_id,
            items=[{"content": content, "timestamp": timestamp, "context": context, "metadata": metadata}],
            document_id=document_id,
        )

    async def arecall(
        self,
        bank_id: str,
        query: str,
        types: Optional[List[str]] = None,
        max_tokens: int = 4096,
        budget: str = "mid",
    ) -> List[RecallResult]:
        """
        Recall memories using semantic similarity (async).

        Args:
            bank_id: The memory bank ID
            query: Search query
            types: Optional list of fact types to filter (world, experience, opinion, observation)
            max_tokens: Maximum tokens in results (default: 4096)
            budget: Budget level for recall - "low", "mid", or "high" (default: "mid")

        Returns:
            List of RecallResult objects
        """
        request_obj = recall_request.RecallRequest(
            query=query,
            types=types,
            budget=budget,
            max_tokens=max_tokens,
            trace=False,
        )

        response = await self._api.recall_memories(bank_id, request_obj)
        return response.results if hasattr(response, 'results') else []

    async def areflect(
        self,
        bank_id: str,
        query: str,
        budget: str = "low",
        context: Optional[str] = None,
    ) -> ReflectResponse:
        """
        Generate a contextual answer based on bank identity and memories (async).

        Args:
            bank_id: The memory bank ID
            query: The question or prompt
            budget: Budget level for reflection - "low", "mid", or "high" (default: "low")
            context: Optional additional context

        Returns:
            ReflectResponse with answer text and optionally facts used
        """
        request_obj = reflect_request.ReflectRequest(
            query=query,
            budget=budget,
            context=context,
        )

        return await self._api.reflect(bank_id, request_obj)
