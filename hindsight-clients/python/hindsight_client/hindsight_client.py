"""
Clean, pythonic wrapper for the Hindsight API client.

This file is MAINTAINED and NOT auto-generated. It provides a high-level,
easy-to-use interface on top of the auto-generated OpenAPI client.
"""

import asyncio
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime

import hindsight_client_api
from hindsight_client_api.api import memory_api, banks_api, mental_models_api
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
from hindsight_client_api.models.mental_model_response import MentalModelResponse
from hindsight_client_api.models.mental_model_list_response import MentalModelListResponse
from hindsight_client_api.models.async_operation_submit_response import AsyncOperationSubmitResponse


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

        # Without authentication
        client = Hindsight(base_url="http://localhost:8888")

        # With API key authentication
        client = Hindsight(base_url="http://localhost:8888", api_key="your-api-key")

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

    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: float = 30.0):
        """
        Initialize the Hindsight client.

        Args:
            base_url: The base URL of the Hindsight API server
            api_key: Optional API key for authentication (sent as Bearer token)
            timeout: Request timeout in seconds (default: 30.0)
        """
        config = hindsight_client_api.Configuration(host=base_url, access_token=api_key)
        self._api_client = hindsight_client_api.ApiClient(config)
        if api_key:
            self._api_client.set_default_header("Authorization", f"Bearer {api_key}")
        self._memory_api = memory_api.MemoryApi(self._api_client)
        self._banks_api = banks_api.BanksApi(self._api_client)
        self._mental_models_api = mental_models_api.MentalModelsApi(self._api_client)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def close(self):
        """Close the API client (sync version - use aclose() in async code)."""
        if self._api_client:
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context - schedule but don't wait
                # The caller should use aclose() instead
                loop.create_task(self._api_client.close())
            except RuntimeError:
                # No running loop - safe to run synchronously
                _run_async(self._api_client.close())

    async def aclose(self):
        """Close the API client (async version)."""
        if self._api_client:
            await self._api_client.close()

    # Simplified methods for main operations

    def retain(
        self,
        bank_id: str,
        content: str,
        timestamp: Optional[datetime] = None,
        context: Optional[str] = None,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        entities: Optional[List[Dict[str, str]]] = None,
        tags: Optional[List[str]] = None,
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
            entities: Optional list of entities [{"text": "...", "type": "..."}]
            tags: Optional list of tags for this memory

        Returns:
            RetainResponse with success status
        """
        return self.retain_batch(
            bank_id=bank_id,
            items=[{"content": content, "timestamp": timestamp, "context": context, "metadata": metadata, "entities": entities, "tags": tags}],
            document_id=document_id,
        )

    def retain_batch(
        self,
        bank_id: str,
        items: List[Dict[str, Any]],
        document_id: Optional[str] = None,
        retain_async: bool = False,
        document_tags: Optional[List[str]] = None,
    ) -> RetainResponse:
        """
        Store multiple memories in batch.

        Args:
            bank_id: The memory bank ID
            items: List of memory items with 'content' and optional 'timestamp', 'context', 'metadata', 'document_id', 'entities', 'tags'
            document_id: Optional document ID for grouping memories (applied to items that don't have their own)
            retain_async: If True, process asynchronously in background (default: False)
            document_tags: Optional list of tags to apply to all memories in this batch

        Returns:
            RetainResponse with success status and item count
        """
        from hindsight_client_api.models.entity_input import EntityInput

        memory_items = []
        for item in items:
            entities = None
            if item.get("entities"):
                entities = [
                    EntityInput(text=e["text"], type=e.get("type"))
                    for e in item["entities"]
                ]
            memory_items.append(
                memory_item.MemoryItem(
                    content=item["content"],
                    timestamp=item.get("timestamp"),
                    context=item.get("context"),
                    metadata=item.get("metadata"),
                    # Use item's document_id if provided, otherwise fall back to batch-level document_id
                    document_id=item.get("document_id") or document_id,
                    entities=entities,
                    tags=item.get("tags"),
                )
            )

        request_obj = retain_request.RetainRequest(
            items=memory_items,
            async_=retain_async,
            document_tags=document_tags,
        )

        return _run_async(self._memory_api.retain_memories(bank_id, request_obj))

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
        tags: Optional[List[str]] = None,
        tags_match: str = "any",
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
            tags: Optional list of tags to filter memories by
            tags_match: How to match tags: 'any' (OR, includes untagged), 'all' (AND, includes untagged),
                'any_strict' (OR, excludes untagged), 'all_strict' (AND, excludes untagged). Default: 'any'

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
            tags=tags,
            tags_match=tags_match,
        )

        return _run_async(self._memory_api.recall_memories(bank_id, request_obj))

    def reflect(
        self,
        bank_id: str,
        query: str,
        budget: str = "low",
        context: Optional[str] = None,
        max_tokens: Optional[int] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        tags_match: str = "any",
    ) -> ReflectResponse:
        """
        Generate a contextual answer based on bank identity and memories.

        Args:
            bank_id: The memory bank ID
            query: The question or prompt
            budget: Budget level for reflection - "low", "mid", or "high" (default: "low")
            context: Optional additional context
            max_tokens: Maximum tokens for the response (server default: 4096)
            response_schema: Optional JSON Schema for structured output. When provided,
                the response will include a 'structured_output' field with the LLM
                response parsed according to this schema.
            tags: Optional list of tags to filter memories by
            tags_match: How to match tags: 'any' (OR, includes untagged), 'all' (AND, includes untagged),
                'any_strict' (OR, excludes untagged), 'all_strict' (AND, excludes untagged). Default: 'any'

        Returns:
            ReflectResponse with answer text, optionally facts used, and optionally
            structured_output if response_schema was provided
        """
        request_obj = reflect_request.ReflectRequest(
            query=query,
            budget=budget,
            context=context,
            max_tokens=max_tokens,
            response_schema=response_schema,
            tags=tags,
            tags_match=tags_match,
        )

        return _run_async(self._memory_api.reflect(bank_id, request_obj))

    def list_memories(
        self,
        bank_id: str,
        type: Optional[str] = None,
        search_query: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> ListMemoryUnitsResponse:
        """List memory units with pagination."""
        return _run_async(self._memory_api.list_memories(
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
        mission: Optional[str] = None,
        disposition: Optional[Dict[str, float]] = None,
    ) -> BankProfileResponse:
        """Create or update a memory bank.

        Args:
            bank_id: Unique identifier for the bank
            name: Human-readable display name
            mission: Instructions guiding what Hindsight should learn and remember (for mental models)
            disposition: Optional disposition traits (skepticism, literalism, empathy)
        """
        from hindsight_client_api.models import create_bank_request, disposition_traits

        disposition_obj = None
        if disposition:
            disposition_obj = disposition_traits.DispositionTraits(**disposition)

        request_obj = create_bank_request.CreateBankRequest(
            name=name,
            mission=mission,
            disposition=disposition_obj,
        )

        return _run_async(self._banks_api.create_or_update_bank(bank_id, request_obj))

    def set_mission(
        self,
        bank_id: str,
        mission: str,
    ) -> BankProfileResponse:
        """
        Set or update the mission for a memory bank.

        Args:
            bank_id: The memory bank ID
            mission: The mission text describing the agent's purpose

        Returns:
            BankProfileResponse with updated bank profile
        """
        from hindsight_client_api.models import create_bank_request

        request_obj = create_bank_request.CreateBankRequest(mission=mission)
        return _run_async(self._banks_api.create_or_update_bank(bank_id, request_obj))

    def list_mental_models(
        self,
        bank_id: str,
        subtype: Optional[Literal["structural", "emergent", "pinned", "learned", "directive"]] = None,
        tags: Optional[List[str]] = None,
        tags_match: Optional[Literal["any", "all", "exact"]] = None,
    ) -> MentalModelListResponse:
        """
        List mental models for a bank.

        Args:
            bank_id: The memory bank ID
            subtype: Optional filter by subtype (structural, emergent, pinned, learned, directive)
            tags: Optional list of tags to filter by
            tags_match: How to match tags - 'any' (OR), 'all' (AND), or 'exact'

        Returns:
            MentalModelListResponse with list of mental models
        """
        return _run_async(self._mental_models_api.list_mental_models(
            bank_id=bank_id,
            subtype=subtype,
            tags=tags,
            tags_match=tags_match,
        ))

    def get_mental_model(
        self,
        bank_id: str,
        model_id: str,
    ) -> MentalModelResponse:
        """
        Get a specific mental model by ID.

        Args:
            bank_id: The memory bank ID
            model_id: The mental model ID

        Returns:
            MentalModelResponse with full mental model details including observations
        """
        return _run_async(self._mental_models_api.get_mental_model(
            bank_id=bank_id,
            model_id=model_id,
        ))

    def create_mental_model(
        self,
        bank_id: str,
        name: str,
        description: str,
        subtype: Literal["pinned", "directive"] = "pinned",
        observations: Optional[List[Dict[str, str]]] = None,
        tags: Optional[List[str]] = None,
    ) -> MentalModelResponse:
        """
        Create a mental model.

        Args:
            bank_id: The memory bank ID
            name: Human-readable name for the mental model
            description: One-liner description for quick scanning
            subtype: Type of mental model - 'pinned' (LLM-generated observations) or 'directive' (user-provided observations)
            observations: For directives only - list of observations with 'title' and 'content' keys
            tags: Optional list of tags for scoped visibility

        Returns:
            MentalModelResponse with created mental model
        """
        from hindsight_client_api.models.create_mental_model_request import CreateMentalModelRequest
        from hindsight_client_api.models.observation_input import ObservationInput

        obs_list = None
        if observations:
            obs_list = [ObservationInput(title=o.get("title", ""), content=o.get("content", "")) for o in observations]

        request_obj = CreateMentalModelRequest(
            name=name,
            description=description,
            subtype=subtype,
            observations=obs_list,
            tags=tags or [],
        )
        return _run_async(self._mental_models_api.create_mental_model(
            bank_id=bank_id,
            create_mental_model_request=request_obj,
        ))

    def update_mental_model(
        self,
        bank_id: str,
        model_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> MentalModelResponse:
        """
        Update a mental model's name and/or description.

        Args:
            bank_id: The memory bank ID
            model_id: The mental model ID
            name: Optional new name
            description: Optional new description

        Returns:
            MentalModelResponse with updated mental model
        """
        from hindsight_client_api.models.update_mental_model_request import UpdateMentalModelRequest

        request_obj = UpdateMentalModelRequest(
            name=name,
            description=description,
        )
        return _run_async(self._mental_models_api.update_mental_model(
            bank_id=bank_id,
            model_id=model_id,
            update_mental_model_request=request_obj,
        ))

    def delete_mental_model(
        self,
        bank_id: str,
        model_id: str,
    ):
        """
        Delete a mental model.

        Args:
            bank_id: The memory bank ID
            model_id: The mental model ID

        Returns:
            DeleteResponse confirming deletion
        """
        return _run_async(self._mental_models_api.delete_mental_model(
            bank_id=bank_id,
            model_id=model_id,
        ))

    def refresh_mental_models(
        self,
        bank_id: str,
        subtype: Optional[Literal["structural", "emergent", "pinned", "learned"]] = None,
        tags: Optional[List[str]] = None,
    ) -> AsyncOperationSubmitResponse:
        """
        Submit a background job to refresh mental models for a bank.

        Args:
            bank_id: The memory bank ID
            subtype: Optional - only refresh models of this subtype
            tags: Optional - tags to apply to newly created mental models

        Returns:
            AsyncOperationSubmitResponse with operation_id to track progress
        """
        from hindsight_client_api.models.refresh_mental_models_request import RefreshMentalModelsRequest

        request_obj = RefreshMentalModelsRequest(
            subtype=subtype,
            tags=tags,
        )
        return _run_async(self._mental_models_api.refresh_mental_models(
            bank_id=bank_id,
            refresh_mental_models_request=request_obj,
        ))

    def refresh_mental_model(
        self,
        bank_id: str,
        model_id: str,
    ) -> AsyncOperationSubmitResponse:
        """
        Submit a background job to refresh content for a specific mental model.

        Args:
            bank_id: The memory bank ID
            model_id: The mental model ID to refresh

        Returns:
            AsyncOperationSubmitResponse with operation_id to track progress
        """
        return _run_async(self._mental_models_api.refresh_mental_model(
            bank_id=bank_id,
            model_id=model_id,
        ))

    def list_mental_model_versions(
        self,
        bank_id: str,
        model_id: str,
    ):
        """
        List all saved versions of a mental model's observations.

        Args:
            bank_id: The memory bank ID
            model_id: The mental model ID

        Returns:
            List of version objects ordered by version descending
        """
        return _run_async(self._mental_models_api.list_mental_model_versions(
            bank_id=bank_id,
            model_id=model_id,
        ))

    def get_mental_model_version(
        self,
        bank_id: str,
        model_id: str,
        version: int,
    ):
        """
        Get observations from a specific version of a mental model.

        Args:
            bank_id: The memory bank ID
            model_id: The mental model ID
            version: The version number

        Returns:
            Version object with observations at that version
        """
        return _run_async(self._mental_models_api.get_mental_model_version(
            bank_id=bank_id,
            model_id=model_id,
            version=version,
        ))

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
            items: List of memory items with 'content' and optional 'timestamp', 'context', 'metadata', 'document_id', 'entities'
            document_id: Optional document ID for grouping memories (applied to items that don't have their own)
            retain_async: If True, process asynchronously in background (default: False)

        Returns:
            RetainResponse with success status and item count
        """
        from hindsight_client_api.models.entity_input import EntityInput

        memory_items = []
        for item in items:
            entities = None
            if item.get("entities"):
                entities = [
                    EntityInput(text=e["text"], type=e.get("type"))
                    for e in item["entities"]
                ]
            memory_items.append(
                memory_item.MemoryItem(
                    content=item["content"],
                    timestamp=item.get("timestamp"),
                    context=item.get("context"),
                    metadata=item.get("metadata"),
                    # Use item's document_id if provided, otherwise fall back to batch-level document_id
                    document_id=item.get("document_id") or document_id,
                    entities=entities,
                )
            )

        request_obj = retain_request.RetainRequest(
            items=memory_items,
            async_=retain_async,
        )

        return await self._memory_api.retain_memories(bank_id, request_obj)

    async def aretain(
        self,
        bank_id: str,
        content: str,
        timestamp: Optional[datetime] = None,
        context: Optional[str] = None,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        entities: Optional[List[Dict[str, str]]] = None,
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
            entities: Optional list of entities [{"text": "...", "type": "..."}]

        Returns:
            RetainResponse with success status
        """
        return await self.aretain_batch(
            bank_id=bank_id,
            items=[{"content": content, "timestamp": timestamp, "context": context, "metadata": metadata, "entities": entities}],
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

        response = await self._memory_api.recall_memories(bank_id, request_obj)
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

        return await self._memory_api.reflect(bank_id, request_obj)
