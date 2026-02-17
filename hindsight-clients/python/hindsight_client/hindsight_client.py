"""
Clean, pythonic wrapper for the Hindsight API client.

This file is MAINTAINED and NOT auto-generated. It provides a high-level,
easy-to-use interface on top of the auto-generated OpenAPI client.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import hindsight_client_api
from hindsight_client_api.api import banks_api, directives_api, files_api, memory_api, mental_models_api
from hindsight_client_api.models import (
    memory_item,
    recall_request,
    reflect_request,
    retain_request,
)
from hindsight_client_api.models.bank_profile_response import BankProfileResponse
from hindsight_client_api.models.file_retain_response import FileRetainResponse
from hindsight_client_api.models.list_memory_units_response import ListMemoryUnitsResponse
from hindsight_client_api.models.recall_response import RecallResponse
from hindsight_client_api.models.recall_result import RecallResult
from hindsight_client_api.models.reflect_response import ReflectResponse
from hindsight_client_api.models.retain_response import RetainResponse


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

    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 30.0):
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
        self._directives_api = directives_api.DirectivesApi(self._api_client)
        self._files_api = files_api.FilesApi(self._api_client)

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
        timestamp: datetime | None = None,
        context: str | None = None,
        document_id: str | None = None,
        metadata: dict[str, str] | None = None,
        entities: list[dict[str, str]] | None = None,
        tags: list[str] | None = None,
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
            tags: Optional list of tags for filtering memories during recall/reflect

        Returns:
            RetainResponse with success status
        """
        return self.retain_batch(
            bank_id=bank_id,
            items=[
                {
                    "content": content,
                    "timestamp": timestamp,
                    "context": context,
                    "metadata": metadata,
                    "entities": entities,
                    "tags": tags,
                }
            ],
            document_id=document_id,
        )

    def retain_batch(
        self,
        bank_id: str,
        items: list[dict[str, Any]],
        document_id: str | None = None,
        document_tags: list[str] | None = None,
        retain_async: bool = False,
    ) -> RetainResponse:
        """
        Store multiple memories in batch.

        Args:
            bank_id: The memory bank ID
            items: List of memory items with 'content' and optional 'timestamp', 'context', 'metadata', 'document_id', 'entities', 'tags'
            document_id: Optional document ID for grouping memories (applied to items that don't have their own)
            document_tags: Optional list of tags applied to all items in this batch (merged with per-item tags)
            retain_async: If True, process asynchronously in background (default: False)

        Returns:
            RetainResponse with success status and item count
        """
        from hindsight_client_api.models.entity_input import EntityInput

        memory_items = []
        for item in items:
            entities = None
            if item.get("entities"):
                entities = [EntityInput(text=e["text"], type=e.get("type")) for e in item["entities"]]
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

    def retain_files(
        self,
        bank_id: str,
        files: list[str | Path],
        context: str | None = None,
        files_metadata: list[dict[str, Any]] | None = None,
    ) -> FileRetainResponse:
        """
        Upload files and retain their contents as memories.

        Files are automatically converted to text (PDF, DOCX, images via OCR, audio via
        transcription, and more) and ingested as memories. Processing is always asynchronous
        — use the returned operation IDs to track progress.

        Args:
            bank_id: The memory bank ID
            files: List of file paths to upload
            context: Optional context description applied to all files
            files_metadata: Optional per-file metadata list. If provided, must match the
                length of `files`. Each entry can have: context, document_id, tags, metadata.

        Returns:
            FileRetainResponse with operation_ids for tracking progress
        """
        file_data = []
        for file_path in files:
            path = Path(file_path)
            file_data.append((path.name, path.read_bytes()))

        meta = files_metadata or [{"context": context} if context else {} for _ in files]

        request_body = json.dumps({"files_metadata": meta})

        return _run_async(self._files_api.file_retain(bank_id=bank_id, files=file_data, request=request_body))

    def recall(
        self,
        bank_id: str,
        query: str,
        types: list[str] | None = None,
        max_tokens: int = 4096,
        budget: str = "mid",
        trace: bool = False,
        query_timestamp: str | None = None,
        include_entities: bool = False,
        max_entity_tokens: int = 500,
        include_chunks: bool = False,
        max_chunk_tokens: int = 8192,
        tags: list[str] | None = None,
        tags_match: Literal["any", "all", "any_strict", "all_strict"] = "any",
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
            tags_match: How to match tags - "any" (OR, includes untagged), "all" (AND, includes untagged),
                "any_strict" (OR, excludes untagged), "all_strict" (AND, excludes untagged). Default: "any"

        Returns:
            RecallResponse with results, optional entities, optional chunks, and optional trace
        """
        from hindsight_client_api.models import chunk_include_options, entity_include_options, include_options

        include_opts = include_options.IncludeOptions(
            entities=entity_include_options.EntityIncludeOptions(max_tokens=max_entity_tokens)
            if include_entities
            else None,
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
        context: str | None = None,
        max_tokens: int | None = None,
        response_schema: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        tags_match: Literal["any", "all", "any_strict", "all_strict"] = "any",
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
            tags_match: How to match tags - "any" (OR, includes untagged), "all" (AND, includes untagged),
                "any_strict" (OR, excludes untagged), "all_strict" (AND, excludes untagged). Default: "any"

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
        type: str | None = None,
        search_query: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> ListMemoryUnitsResponse:
        """List memory units with pagination."""
        return _run_async(
            self._memory_api.list_memories(
                bank_id=bank_id,
                type=type,
                q=search_query,
                limit=limit,
                offset=offset,
            )
        )

    def create_bank(
        self,
        bank_id: str,
        name: str | None = None,
        mission: str | None = None,
        disposition: dict[str, float] | None = None,
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

    # Async methods (native async, no _run_async wrapper)

    async def aretain_batch(
        self,
        bank_id: str,
        items: list[dict[str, Any]],
        document_id: str | None = None,
        document_tags: list[str] | None = None,
        retain_async: bool = False,
    ) -> RetainResponse:
        """
        Store multiple memories in batch (async).

        Args:
            bank_id: The memory bank ID
            items: List of memory items with 'content' and optional 'timestamp', 'context', 'metadata', 'document_id', 'entities', 'tags'
            document_id: Optional document ID for grouping memories (applied to items that don't have their own)
            document_tags: Optional list of tags applied to all items in this batch (merged with per-item tags)
            retain_async: If True, process asynchronously in background (default: False)

        Returns:
            RetainResponse with success status and item count
        """
        from hindsight_client_api.models.entity_input import EntityInput

        memory_items = []
        for item in items:
            entities = None
            if item.get("entities"):
                entities = [EntityInput(text=e["text"], type=e.get("type")) for e in item["entities"]]
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

        return await self._memory_api.retain_memories(bank_id, request_obj)

    async def aretain(
        self,
        bank_id: str,
        content: str,
        timestamp: datetime | None = None,
        context: str | None = None,
        document_id: str | None = None,
        metadata: dict[str, str] | None = None,
        entities: list[dict[str, str]] | None = None,
        tags: list[str] | None = None,
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
            tags: Optional list of tags for filtering memories during recall/reflect

        Returns:
            RetainResponse with success status
        """
        return await self.aretain_batch(
            bank_id=bank_id,
            items=[
                {
                    "content": content,
                    "timestamp": timestamp,
                    "context": context,
                    "metadata": metadata,
                    "entities": entities,
                    "tags": tags,
                }
            ],
            document_id=document_id,
        )

    async def arecall(
        self,
        bank_id: str,
        query: str,
        types: list[str] | None = None,
        max_tokens: int = 4096,
        budget: str = "mid",
        tags: list[str] | None = None,
        tags_match: Literal["any", "all", "any_strict", "all_strict"] = "any",
    ) -> list[RecallResult]:
        """
        Recall memories using semantic similarity (async).

        Args:
            bank_id: The memory bank ID
            query: Search query
            types: Optional list of fact types to filter (world, experience, opinion, observation)
            max_tokens: Maximum tokens in results (default: 4096)
            budget: Budget level for recall - "low", "mid", or "high" (default: "mid")
            tags: Optional list of tags to filter memories by
            tags_match: How to match tags - "any" (OR, includes untagged), "all" (AND, includes untagged),
                "any_strict" (OR, excludes untagged), "all_strict" (AND, excludes untagged). Default: "any"

        Returns:
            List of RecallResult objects
        """
        request_obj = recall_request.RecallRequest(
            query=query,
            types=types,
            budget=budget,
            max_tokens=max_tokens,
            trace=False,
            tags=tags,
            tags_match=tags_match,
        )

        response = await self._memory_api.recall_memories(bank_id, request_obj)
        return response.results if hasattr(response, "results") else []

    async def areflect(
        self,
        bank_id: str,
        query: str,
        budget: str = "low",
        context: str | None = None,
        tags: list[str] | None = None,
        tags_match: Literal["any", "all", "any_strict", "all_strict"] = "any",
    ) -> ReflectResponse:
        """
        Generate a contextual answer based on bank identity and memories (async).

        Args:
            bank_id: The memory bank ID
            query: The question or prompt
            budget: Budget level for reflection - "low", "mid", or "high" (default: "low")
            context: Optional additional context
            tags: Optional list of tags to filter memories by
            tags_match: How to match tags - "any" (OR, includes untagged), "all" (AND, includes untagged),
                "any_strict" (OR, excludes untagged), "all_strict" (AND, excludes untagged). Default: "any"

        Returns:
            ReflectResponse with answer text and optionally facts used
        """
        request_obj = reflect_request.ReflectRequest(
            query=query,
            budget=budget,
            context=context,
            tags=tags,
            tags_match=tags_match,
        )

        return await self._memory_api.reflect(bank_id, request_obj)

    # Mental Models methods

    def create_mental_model(
        self,
        bank_id: str,
        name: str,
        source_query: str,
        tags: list[str] | None = None,
        max_tokens: int | None = None,
        trigger: dict[str, Any] | None = None,
    ):
        """
        Create a mental model (runs reflect in background).

        Args:
            bank_id: The memory bank ID
            name: Human-readable name for the mental model
            source_query: The query to run to generate content
            tags: Optional tags for filtering during retrieval
            max_tokens: Optional maximum tokens for the mental model content
            trigger: Optional trigger settings (e.g., {"refresh_after_consolidation": True})

        Returns:
            CreateMentalModelResponse with operation_id
        """
        from hindsight_client_api.models import create_mental_model_request, mental_model_trigger

        trigger_obj = None
        if trigger:
            trigger_obj = mental_model_trigger.MentalModelTrigger(**trigger)

        request_obj = create_mental_model_request.CreateMentalModelRequest(
            name=name,
            source_query=source_query,
            tags=tags,
            max_tokens=max_tokens,
            trigger=trigger_obj,
        )

        return _run_async(self._mental_models_api.create_mental_model(bank_id, request_obj))

    def list_mental_models(self, bank_id: str, tags: list[str] | None = None):
        """
        List all mental models in a bank.

        Args:
            bank_id: The memory bank ID
            tags: Optional tags to filter by

        Returns:
            ListMentalModelsResponse with items
        """
        return _run_async(self._mental_models_api.list_mental_models(bank_id, tags=tags))

    def get_mental_model(self, bank_id: str, mental_model_id: str):
        """
        Get a specific mental model.

        Args:
            bank_id: The memory bank ID
            mental_model_id: The mental model ID

        Returns:
            MentalModelResponse
        """
        return _run_async(self._mental_models_api.get_mental_model(bank_id, mental_model_id))

    def refresh_mental_model(self, bank_id: str, mental_model_id: str):
        """
        Refresh a mental model to update with current knowledge.

        Args:
            bank_id: The memory bank ID
            mental_model_id: The mental model ID

        Returns:
            RefreshMentalModelResponse with operation_id
        """
        return _run_async(self._mental_models_api.refresh_mental_model(bank_id, mental_model_id))

    def update_mental_model(
        self,
        bank_id: str,
        mental_model_id: str,
        name: str | None = None,
        source_query: str | None = None,
        tags: list[str] | None = None,
        max_tokens: int | None = None,
        trigger: dict[str, Any] | None = None,
    ):
        """
        Update a mental model's metadata.

        Args:
            bank_id: The memory bank ID
            mental_model_id: The mental model ID
            name: Optional new name
            source_query: Optional new source query
            tags: Optional new tags
            max_tokens: Optional new max tokens
            trigger: Optional trigger settings (e.g., {"refresh_after_consolidation": True})

        Returns:
            MentalModelResponse
        """
        from hindsight_client_api.models import mental_model_trigger, update_mental_model_request

        trigger_obj = None
        if trigger:
            trigger_obj = mental_model_trigger.MentalModelTrigger(**trigger)

        request_obj = update_mental_model_request.UpdateMentalModelRequest(
            name=name,
            source_query=source_query,
            tags=tags,
            max_tokens=max_tokens,
            trigger=trigger_obj,
        )

        return _run_async(self._mental_models_api.update_mental_model(bank_id, mental_model_id, request_obj))

    def delete_mental_model(self, bank_id: str, mental_model_id: str):
        """
        Delete a mental model.

        Args:
            bank_id: The memory bank ID
            mental_model_id: The mental model ID
        """
        return _run_async(self._mental_models_api.delete_mental_model(bank_id, mental_model_id))

    # Directives methods

    def create_directive(
        self,
        bank_id: str,
        name: str,
        content: str,
        priority: int = 0,
        is_active: bool = True,
        tags: list[str] | None = None,
    ):
        """
        Create a directive (hard rule for reflect).

        Args:
            bank_id: The memory bank ID
            name: Human-readable name for the directive
            content: The directive content/rules
            priority: Priority level (higher = injected first)
            is_active: Whether the directive is active
            tags: Optional tags for filtering

        Returns:
            DirectiveResponse
        """
        from hindsight_client_api.models import create_directive_request

        request_obj = create_directive_request.CreateDirectiveRequest(
            name=name,
            content=content,
            priority=priority,
            is_active=is_active,
            tags=tags,
        )

        return _run_async(self._directives_api.create_directive(bank_id, request_obj))

    def list_directives(self, bank_id: str, tags: list[str] | None = None):
        """
        List all directives in a bank.

        Args:
            bank_id: The memory bank ID
            tags: Optional tags to filter by

        Returns:
            ListDirectivesResponse with items
        """
        return _run_async(self._directives_api.list_directives(bank_id, tags=tags))

    def get_directive(self, bank_id: str, directive_id: str):
        """
        Get a specific directive.

        Args:
            bank_id: The memory bank ID
            directive_id: The directive ID

        Returns:
            DirectiveResponse
        """
        return _run_async(self._directives_api.get_directive(bank_id, directive_id))

    def update_directive(
        self,
        bank_id: str,
        directive_id: str,
        name: str | None = None,
        content: str | None = None,
        priority: int | None = None,
        is_active: bool | None = None,
        tags: list[str] | None = None,
    ):
        """
        Update a directive.

        Args:
            bank_id: The memory bank ID
            directive_id: The directive ID
            name: Optional new name
            content: Optional new content
            priority: Optional new priority
            is_active: Optional new active status
            tags: Optional new tags

        Returns:
            DirectiveResponse
        """
        from hindsight_client_api.models import update_directive_request

        request_obj = update_directive_request.UpdateDirectiveRequest(
            name=name,
            content=content,
            priority=priority,
            is_active=is_active,
            tags=tags,
        )

        return _run_async(self._directives_api.update_directive(bank_id, directive_id, request_obj))

    def delete_directive(self, bank_id: str, directive_id: str):
        """
        Delete a directive.

        Args:
            bank_id: The memory bank ID
            directive_id: The directive ID
        """
        return _run_async(self._directives_api.delete_directive(bank_id, directive_id))

    def delete_bank(self, bank_id: str):
        """
        Delete a memory bank.

        Args:
            bank_id: The memory bank ID
        """
        return _run_async(self._banks_api.delete_bank(bank_id))
