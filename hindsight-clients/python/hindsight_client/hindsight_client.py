"""
Clean, pythonic wrapper for the Hindsight API client.

This file is MAINTAINED and NOT auto-generated. It provides a high-level,
easy-to-use interface on top of the auto-generated OpenAPI client.
"""

import asyncio
import json
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

import hindsight_client_api

try:
    _CLIENT_VERSION = metadata.version("hindsight-client")
except metadata.PackageNotFoundError:
    _CLIENT_VERSION = "0.0.0"

DEFAULT_USER_AGENT = f"hindsight-client-python/{_CLIENT_VERSION}"
from hindsight_client_api.api import (
    banks_api,
    directives_api,
    documents_api,
    entities_api,
    files_api,
    memory_api,
    mental_models_api,
    monitoring_api,
    operations_api,
    webhooks_api,
)
from hindsight_client_api.models import (
    memory_item,
    recall_request,
    reflect_request,
    retain_request,
)
from hindsight_client_api.models.reflect_include_options import ReflectIncludeOptions
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

    This class provides simplified methods for the most common operations:
    retain, recall, reflect, bank management, mental models, and directives.

    **Async vs sync:** Every convenience method has an async counterpart
    prefixed with ``a`` (e.g. ``aretain``, ``arecall``, ``areflect``).
    **Prefer the async variants** (``aretain``, ``arecall``, ``areflect``, etc.)
    whenever you are inside an async context (``async def``, event loops,
    frameworks like FastAPI/LangGraph/CrewAI). The sync versions (``retain``,
    ``recall``, ``reflect``) are convenience wrappers that call
    ``asyncio.run_until_complete`` under the hood — they exist for scripts and
    REPLs but will raise errors if an event loop is already running.

    For operations not covered here (documents, entities, operations/async jobs,
    webhooks, file uploads, monitoring), use the low-level API clients exposed
    as properties on this class. These are auto-generated from the OpenAPI spec
    and cover the full API surface. **All low-level methods are async-only.**

    Example — async (preferred)::

        from hindsight_client import Hindsight

        client = Hindsight(base_url="http://localhost:8888", api_key="your-api-key")

        # Inside an async function — use the a* methods
        await client.aretain(bank_id="alice", content="Alice loves AI")
        response = await client.arecall(bank_id="alice", query="What does Alice like?")
        answer = await client.areflect(bank_id="alice", query="What are my interests?")

    Example — sync (scripts / REPLs only)::

        # Outside an async context — sync wrappers are available
        client.retain(bank_id="alice", content="Alice loves AI")
        response = client.recall(bank_id="alice", query="What does Alice like?")

    Example — low-level API for advanced operations::

        # Access documents, entities, operations, webhooks, etc.
        # All low-level methods are async-only — use 'await' or asyncio.run().

        # List documents in a bank
        docs = await client.documents.list_documents("alice")

        # Delete a specific document
        await client.documents.delete_document("alice", "doc-123")

        # Check async operation status
        status = await client.operations.get_operation_status("alice", "op-456")

        # List entities
        entities = await client.entities.list_entities("alice")

    Available low-level API properties:
        - ``client.memory``: Core memory operations (MemoryApi)
        - ``client.banks``: Bank management (BanksApi)
        - ``client.documents``: Document CRUD (DocumentsApi)
        - ``client.entities``: Entity browsing (EntitiesApi)
        - ``client.mental_models``: Mental model management (MentalModelsApi)
        - ``client.directives``: Directive management (DirectivesApi)
        - ``client.operations``: Async operation tracking (OperationsApi)
        - ``client.webhooks``: Webhook management (WebhooksApi)
        - ``client.files``: File upload (FilesApi)
        - ``client.monitoring``: Health/version checks (MonitoringApi)
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 300.0,
        user_agent: str | None = None,
    ):
        """
        Initialize the Hindsight client.

        Args:
            base_url: The base URL of the Hindsight API server
            api_key: Optional API key for authentication (sent as Bearer token)
            timeout: Request timeout in seconds (default: 300.0)
            user_agent: Override the default ``User-Agent`` header. Integrations
                should set this to identify themselves (e.g.
                ``"hindsight-crewai/1.2.0"``). Defaults to
                ``hindsight-client-python/<version>``.
        """
        config = hindsight_client_api.Configuration(host=base_url, access_token=api_key)
        self._api_client = hindsight_client_api.ApiClient(config)
        self._api_client.user_agent = user_agent or DEFAULT_USER_AGENT
        self._timeout = timeout
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        if api_key:
            self._api_client.set_default_header("Authorization", f"Bearer {api_key}")
        self._memory_api = memory_api.MemoryApi(self._api_client)
        self._banks_api = banks_api.BanksApi(self._api_client)
        self._mental_models_api = mental_models_api.MentalModelsApi(self._api_client)
        self._directives_api = directives_api.DirectivesApi(self._api_client)
        self._files_api = files_api.FilesApi(self._api_client)
        self._documents_api = documents_api.DocumentsApi(self._api_client)
        self._entities_api = entities_api.EntitiesApi(self._api_client)
        self._operations_api = operations_api.OperationsApi(self._api_client)
        self._webhooks_api = webhooks_api.WebhooksApi(self._api_client)
        self._monitoring_api = monitoring_api.MonitoringApi(self._api_client)

    # -- Low-level API accessors ------------------------------------------------
    # These expose the full, auto-generated API surface for operations not
    # covered by the convenience methods above.  All methods on these objects
    # are async — use ``await`` or ``asyncio.run()`` to call them.

    @property
    def memory(self) -> memory_api.MemoryApi:
        """Low-level Memory API — retain, recall, reflect, list/clear memories, tags, and graph."""
        return self._memory_api

    @property
    def banks(self) -> banks_api.BanksApi:
        """Low-level Banks API — create, update, delete banks; stats; consolidation; config."""
        return self._banks_api

    @property
    def documents(self) -> documents_api.DocumentsApi:
        """Low-level Documents API — list, get, update, delete documents and chunks."""
        return self._documents_api

    @property
    def entities(self) -> entities_api.EntitiesApi:
        """Low-level Entities API — list, get, and regenerate entity observations."""
        return self._entities_api

    @property
    def mental_models(self) -> mental_models_api.MentalModelsApi:
        """Low-level Mental Models API — create, list, get, update, refresh, delete, history."""
        return self._mental_models_api

    @property
    def directives(self) -> directives_api.DirectivesApi:
        """Low-level Directives API — create, list, get, update, delete."""
        return self._directives_api

    @property
    def operations(self) -> operations_api.OperationsApi:
        """Low-level Operations API — get status, list, cancel, retry async operations."""
        return self._operations_api

    @property
    def webhooks(self) -> webhooks_api.WebhooksApi:
        """Low-level Webhooks API — create, list, update, delete webhooks and deliveries."""
        return self._webhooks_api

    @property
    def files(self) -> files_api.FilesApi:
        """Low-level Files API — upload and retain files."""
        return self._files_api

    @property
    def monitoring(self) -> monitoring_api.MonitoringApi:
        """Low-level Monitoring API — health check, version, metrics."""
        return self._monitoring_api

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
        update_mode: str | None = None,
    ) -> RetainResponse:
        """
        Store a single memory (sync wrapper — prefer :meth:`aretain` in async code).

        Args:
            bank_id: The memory bank ID
            content: Memory content
            timestamp: Optional event timestamp
            context: Optional context description
            document_id: Optional document ID for grouping
            metadata: Optional user-defined metadata
            entities: Optional list of entities [{"text": "...", "type": "..."}]
            tags: Optional list of tags for filtering memories during recall/reflect
            update_mode: How to handle existing documents ('replace' or 'append')

        Returns:
            RetainResponse with success status
        """
        item: dict[str, Any] = {
            "content": content,
            "timestamp": timestamp,
            "context": context,
            "metadata": metadata,
            "entities": entities,
            "tags": tags,
        }
        if update_mode is not None:
            item["update_mode"] = update_mode
        return self.retain_batch(
            bank_id=bank_id,
            items=[item],
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
        Store multiple memories in batch (sync wrapper — prefer :meth:`aretain_batch` in async code).

        Args:
            bank_id: The memory bank ID
            items: List of memory items, each a dict with 'content' (required) and optional keys:
                'timestamp', 'context', 'metadata', 'document_id', 'entities', 'tags',
                'observation_scopes' (str or list[list[str]]), 'strategy'.
            document_id: Optional document ID for grouping memories (applied to items that don't have their own)
            document_tags: Optional list of tags applied to all items in this batch (merged with per-item tags)
            retain_async: If True, process asynchronously in background (default: False)

        Returns:
            RetainResponse with success status and item count
        """
        return _run_async(
            self.aretain_batch(
                bank_id=bank_id,
                items=items,
                document_id=document_id,
                document_tags=document_tags,
                retain_async=retain_async,
            )
        )

    def retain_files(
        self,
        bank_id: str,
        files: list[str | Path],
        context: str | None = None,
        files_metadata: list[dict[str, Any]] | None = None,
    ) -> FileRetainResponse:
        """
        Upload files and retain their contents as memories (sync wrapper).

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

        return _run_async(self._files_api.file_retain(bank_id=bank_id, files=file_data, request=request_body, _request_timeout=self._timeout))

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
        include_source_facts: bool = False,
        max_source_facts_tokens: int = 4096,
        tags: list[str] | None = None,
        tags_match: Literal["any", "all", "any_strict", "all_strict"] = "any",
        tag_groups: list[dict[str, Any]] | None = None,
    ) -> RecallResponse:
        """
        Recall memories using semantic similarity (sync wrapper — prefer :meth:`arecall` in async code).

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
            include_source_facts: Include source facts for observation-type results (default: False)
            max_source_facts_tokens: Maximum tokens for source facts (default: 4096)
            tags: Optional list of tags to filter memories by
            tags_match: How to match tags - "any" (OR, includes untagged), "all" (AND, includes untagged),
                "any_strict" (OR, excludes untagged), "all_strict" (AND, excludes untagged). Default: "any"
            tag_groups: Optional list of tag group filters for advanced boolean tag matching.

        Returns:
            RecallResponse with results, optional entities, optional chunks, optional source_facts, and optional trace
        """
        return _run_async(
            self.arecall(
                bank_id=bank_id,
                query=query,
                types=types,
                max_tokens=max_tokens,
                budget=budget,
                trace=trace,
                query_timestamp=query_timestamp,
                include_entities=include_entities,
                max_entity_tokens=max_entity_tokens,
                include_chunks=include_chunks,
                max_chunk_tokens=max_chunk_tokens,
                include_source_facts=include_source_facts,
                max_source_facts_tokens=max_source_facts_tokens,
                tags=tags,
                tags_match=tags_match,
                tag_groups=tag_groups,
            )
        )

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
        include_facts: bool = False,
        tag_groups: list[dict[str, Any]] | None = None,
        fact_types: list[str] | None = None,
        exclude_mental_models: bool = False,
        exclude_mental_model_ids: list[str] | None = None,
    ) -> ReflectResponse:
        """
        Generate a contextual answer based on bank identity and memories (sync wrapper — prefer :meth:`areflect` in async code).

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
            include_facts: If True, the response will include a 'based_on' field listing
                the memories, mental models, and directives used to construct the answer.
            tag_groups: Optional list of tag group filters for advanced boolean tag matching.
            fact_types: Optional list of fact types to include (world, experience, observation).
            exclude_mental_models: If True, exclude all mental models from reflection (default: False).
            exclude_mental_model_ids: Optional list of specific mental model IDs to exclude.

        Returns:
            ReflectResponse with answer text, optionally facts used, and optionally
            structured_output if response_schema was provided
        """
        return _run_async(
            self.areflect(
                bank_id=bank_id,
                query=query,
                budget=budget,
                context=context,
                max_tokens=max_tokens,
                response_schema=response_schema,
                tags=tags,
                tags_match=tags_match,
                include_facts=include_facts,
                tag_groups=tag_groups,
                fact_types=fact_types,
                exclude_mental_models=exclude_mental_models,
                exclude_mental_model_ids=exclude_mental_model_ids,
            )
        )

    def list_memories(
        self,
        bank_id: str,
        type: str | None = None,
        search_query: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> ListMemoryUnitsResponse:
        """List memory units with pagination (sync wrapper — use ``await client.memory.list_memories(...)`` in async code)."""
        return _run_async(
            self._memory_api.list_memories(
                bank_id=bank_id,
                type=type,
                q=search_query,
                limit=limit,
                offset=offset,
                _request_timeout=self._timeout,
            )
        )

    def create_bank(
        self,
        bank_id: str,
        name: str | None = None,
        mission: str | None = None,
        disposition_skepticism: int | None = None,
        disposition_literalism: int | None = None,
        disposition_empathy: int | None = None,
        disposition: dict[str, float] | None = None,
        retain_mission: str | None = None,
        retain_extraction_mode: str | None = None,
        retain_custom_instructions: str | None = None,
        retain_chunk_size: int | None = None,
        enable_observations: bool | None = None,
        observations_mission: str | None = None,
        reflect_mission: str | None = None,
        background: str | None = None,
    ) -> BankProfileResponse:
        """Create or update a memory bank (sync wrapper — prefer :meth:`acreate_bank` in async code).

        Args:
            bank_id: Unique identifier for the bank
            name: Deprecated. Display label only.
            mission: Deprecated. Use reflect_mission instead.
            disposition_skepticism: Deprecated. Use update_bank_config(disposition_skepticism=...) instead.
            disposition_literalism: Deprecated. Use update_bank_config(disposition_literalism=...) instead.
            disposition_empathy: Deprecated. Use update_bank_config(disposition_empathy=...) instead.
            disposition: Deprecated. Use update_bank_config(disposition_skepticism=...) instead.
            retain_mission: Steers what gets extracted during retain(). Injected alongside built-in rules.
            retain_extraction_mode: Fact extraction mode: 'concise' (default), 'verbose', or 'custom'.
            retain_custom_instructions: Custom extraction prompt (only active when mode is 'custom').
            retain_chunk_size: Maximum token size for each content chunk during retain.
            enable_observations: Toggle automatic observation consolidation after retain().
            observations_mission: Controls what gets synthesised into observations. Replaces built-in rules.
            reflect_mission: Mission/context for Reflect operations.
            background: Optional background context for the bank.
        """
        return _run_async(
            self._acreate_bank(
                bank_id,
                name=name,
                mission=mission,
                reflect_mission=reflect_mission,
                disposition_skepticism=disposition_skepticism,
                disposition_literalism=disposition_literalism,
                disposition_empathy=disposition_empathy,
                disposition=disposition,
                retain_mission=retain_mission,
                retain_extraction_mode=retain_extraction_mode,
                retain_custom_instructions=retain_custom_instructions,
                retain_chunk_size=retain_chunk_size,
                enable_observations=enable_observations,
                observations_mission=observations_mission,
                background=background,
            )
        )

    async def _acreate_bank(
        self,
        bank_id: str,
        name: str | None = None,
        mission: str | None = None,
        reflect_mission: str | None = None,
        disposition_skepticism: int | None = None,
        disposition_literalism: int | None = None,
        disposition_empathy: int | None = None,
        disposition: dict[str, float] | None = None,
        retain_mission: str | None = None,
        retain_extraction_mode: str | None = None,
        retain_custom_instructions: str | None = None,
        retain_chunk_size: int | None = None,
        enable_observations: bool | None = None,
        observations_mission: str | None = None,
        background: str | None = None,
    ) -> BankProfileResponse:
        import aiohttp

        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if mission is not None:
            body["mission"] = mission
        if reflect_mission is not None:
            body["reflect_mission"] = reflect_mission
        if background is not None:
            body["background"] = background
        # Individual disposition fields take priority over legacy disposition dict
        if disposition_skepticism is not None:
            body["disposition_skepticism"] = disposition_skepticism
        elif disposition is not None:
            body["disposition_skepticism"] = disposition.get("skepticism")
        if disposition_literalism is not None:
            body["disposition_literalism"] = disposition_literalism
        elif disposition is not None:
            body["disposition_literalism"] = disposition.get("literalism")
        if disposition_empathy is not None:
            body["disposition_empathy"] = disposition_empathy
        elif disposition is not None:
            body["disposition_empathy"] = disposition.get("empathy")
        if retain_mission is not None:
            body["retain_mission"] = retain_mission
        if retain_extraction_mode is not None:
            body["retain_extraction_mode"] = retain_extraction_mode
        if retain_custom_instructions is not None:
            body["retain_custom_instructions"] = retain_custom_instructions
        if retain_chunk_size is not None:
            body["retain_chunk_size"] = retain_chunk_size
        if enable_observations is not None:
            body["enable_observations"] = enable_observations
        if observations_mission is not None:
            body["observations_mission"] = observations_mission

        url = f"{self._base_url}/v1/default/banks/{bank_id}"
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        async with aiohttp.ClientSession() as session:
            async with session.put(
                url, json=body, headers=headers, timeout=aiohttp.ClientTimeout(total=self._timeout)
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return BankProfileResponse.model_validate(data)

    def set_mission(self, bank_id: str, mission: str) -> dict[str, Any]:
        """Deprecated. Use update_bank_config(reflect_mission=...) instead."""
        return self.create_bank(bank_id, mission=mission)

    def set_reflect_mission(self, bank_id: str, reflect_mission: str) -> dict[str, Any]:
        """Deprecated alias for set_mission()."""
        return self.set_mission(bank_id, reflect_mission)

    # Async methods (native async, no _run_async wrapper)

    async def acreate_bank(
        self,
        bank_id: str,
        name: str | None = None,
        mission: str | None = None,
        disposition_skepticism: int | None = None,
        disposition_literalism: int | None = None,
        disposition_empathy: int | None = None,
        disposition: dict[str, float] | None = None,
        retain_mission: str | None = None,
        retain_extraction_mode: str | None = None,
        retain_custom_instructions: str | None = None,
        retain_chunk_size: int | None = None,
        enable_observations: bool | None = None,
        observations_mission: str | None = None,
        reflect_mission: str | None = None,
        background: str | None = None,
    ) -> BankProfileResponse:
        """Create or update a memory bank (async — preferred over :meth:`create_bank`).

        Args:
            bank_id: Unique identifier for the bank
            name: Deprecated. Display label only.
            mission: Deprecated. Use reflect_mission instead.
            disposition_skepticism: Deprecated. Use update_bank_config(disposition_skepticism=...) instead.
            disposition_literalism: Deprecated. Use update_bank_config(disposition_literalism=...) instead.
            disposition_empathy: Deprecated. Use update_bank_config(disposition_empathy=...) instead.
            disposition: Deprecated. Use update_bank_config(disposition_skepticism=...) instead.
            retain_mission: Steers what gets extracted during retain(). Injected alongside built-in rules.
            retain_extraction_mode: Fact extraction mode: 'concise' (default), 'verbose', or 'custom'.
            retain_custom_instructions: Custom extraction prompt (only active when mode is 'custom').
            retain_chunk_size: Maximum token size for each content chunk during retain.
            enable_observations: Toggle automatic observation consolidation after retain().
            observations_mission: Controls what gets synthesised into observations. Replaces built-in rules.
            reflect_mission: Mission/context for Reflect operations.
            background: Optional background context for the bank.
        """
        return await self._acreate_bank(
            bank_id,
            name=name,
            mission=mission,
            reflect_mission=reflect_mission,
            disposition_skepticism=disposition_skepticism,
            disposition_literalism=disposition_literalism,
            disposition_empathy=disposition_empathy,
            disposition=disposition,
            retain_mission=retain_mission,
            retain_extraction_mode=retain_extraction_mode,
            retain_custom_instructions=retain_custom_instructions,
            retain_chunk_size=retain_chunk_size,
            enable_observations=enable_observations,
            observations_mission=observations_mission,
            background=background,
        )

    async def aset_mission(self, bank_id: str, mission: str) -> dict[str, Any]:
        """Deprecated. Use update_bank_config(reflect_mission=...) instead."""
        return await self.acreate_bank(bank_id, mission=mission)

    async def aset_reflect_mission(self, bank_id: str, reflect_mission: str) -> dict[str, Any]:
        """Deprecated alias for aset_mission()."""
        return await self.aset_mission(bank_id, reflect_mission)

    async def aretain_batch(
        self,
        bank_id: str,
        items: list[dict[str, Any]],
        document_id: str | None = None,
        document_tags: list[str] | None = None,
        retain_async: bool = False,
    ) -> RetainResponse:
        """
        Store multiple memories in batch (async — preferred over :meth:`retain_batch`).

        Args:
            bank_id: The memory bank ID
            items: List of memory items, each a dict with 'content' (required) and optional keys:
                'timestamp', 'context', 'metadata', 'document_id', 'entities', 'tags',
                'observation_scopes' (str or list[list[str]]), 'strategy'.
            document_id: Optional document ID for grouping memories (applied to items that don't have their own)
            document_tags: Optional list of tags applied to all items in this batch (merged with per-item tags)
            retain_async: If True, process asynchronously in background (default: False)

        Returns:
            RetainResponse with success status and item count
        """
        from hindsight_client_api.models.entity_input import EntityInput
        from hindsight_client_api.models.observation_scopes import ObservationScopes
        from hindsight_client_api.models.timestamp import Timestamp

        memory_items = []
        for item in items:
            entities = None
            if item.get("entities"):
                entities = [EntityInput(text=e["text"], type=e.get("type")) for e in item["entities"]]
            raw_ts = item.get("timestamp")
            timestamp_val = Timestamp(actual_instance=raw_ts) if raw_ts is not None else None
            obs_scopes = None
            if item.get("observation_scopes") is not None:
                obs_scopes = ObservationScopes(actual_instance=item["observation_scopes"])
            memory_items.append(
                memory_item.MemoryItem(
                    content=item["content"],
                    timestamp=timestamp_val,
                    context=item.get("context"),
                    metadata=item.get("metadata"),
                    # Use item's document_id if provided, otherwise fall back to batch-level document_id
                    document_id=item.get("document_id") or document_id,
                    entities=entities,
                    tags=item.get("tags"),
                    observation_scopes=obs_scopes,
                    strategy=item.get("strategy"),
                    update_mode=item.get("update_mode"),
                )
            )

        request_obj = retain_request.RetainRequest(
            items=memory_items,
            var_async=retain_async,
            document_tags=document_tags,
        )

        return await self._memory_api.retain_memories(bank_id, request_obj, _request_timeout=self._timeout)

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
        update_mode: str | None = None,
    ) -> RetainResponse:
        """
        Store a single memory (async — preferred over :meth:`retain`).

        Args:
            bank_id: The memory bank ID
            content: Memory content
            timestamp: Optional event timestamp
            context: Optional context description
            document_id: Optional document ID for grouping
            metadata: Optional user-defined metadata
            entities: Optional list of entities [{"text": "...", "type": "..."}]
            tags: Optional list of tags for filtering memories during recall/reflect
            update_mode: How to handle existing documents ('replace' or 'append')

        Returns:
            RetainResponse with success status
        """
        item: dict[str, Any] = {
            "content": content,
            "timestamp": timestamp,
            "context": context,
            "metadata": metadata,
            "entities": entities,
            "tags": tags,
        }
        if update_mode is not None:
            item["update_mode"] = update_mode
        return await self.aretain_batch(
            bank_id=bank_id,
            items=[item],
            document_id=document_id,
        )

    async def arecall(
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
        include_source_facts: bool = False,
        max_source_facts_tokens: int = 4096,
        tags: list[str] | None = None,
        tags_match: Literal["any", "all", "any_strict", "all_strict"] = "any",
        tag_groups: list[dict[str, Any]] | None = None,
    ) -> RecallResponse:
        """
        Recall memories using semantic similarity (async — preferred over :meth:`recall`).

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
            include_source_facts: Include source facts for observation-type results (default: False)
            max_source_facts_tokens: Maximum tokens for source facts (default: 4096)
            tags: Optional list of tags to filter memories by
            tags_match: How to match tags - "any" (OR, includes untagged), "all" (AND, includes untagged),
                "any_strict" (OR, excludes untagged), "all_strict" (AND, excludes untagged). Default: "any"
            tag_groups: Optional list of tag group filters for advanced boolean tag matching.
                Each element is a dict representing a tag group node (TagGroupLeaf, TagGroupAnd,
                TagGroupOr, or TagGroupNot). Example::

                    [{"tags": ["customer"], "match": "all"}, {"not": {"tags": ["internal"]}}]

        Returns:
            RecallResponse with results, optional entities, optional chunks, optional source_facts, and optional trace
        """
        from hindsight_client_api.models import (
            chunk_include_options,
            entity_include_options,
            include_options,
            source_facts_include_options,
        )

        include_opts = include_options.IncludeOptions(
            entities=entity_include_options.EntityIncludeOptions(max_tokens=max_entity_tokens)
            if include_entities
            else None,
            chunks=chunk_include_options.ChunkIncludeOptions(max_tokens=max_chunk_tokens) if include_chunks else None,
            source_facts=source_facts_include_options.SourceFactsIncludeOptions(max_tokens=max_source_facts_tokens)
            if include_source_facts
            else None,
        )

        tag_groups_objs = None
        if tag_groups is not None:
            from hindsight_client_api.models.recall_request_tag_groups_inner import RecallRequestTagGroupsInner

            tag_groups_objs = [RecallRequestTagGroupsInner.from_dict(tg) for tg in tag_groups]

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
            tag_groups=tag_groups_objs,
        )

        return await self._memory_api.recall_memories(bank_id, request_obj, _request_timeout=self._timeout)

    async def areflect(
        self,
        bank_id: str,
        query: str,
        budget: str = "low",
        context: str | None = None,
        max_tokens: int | None = None,
        response_schema: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        tags_match: Literal["any", "all", "any_strict", "all_strict"] = "any",
        include_facts: bool = False,
        tag_groups: list[dict[str, Any]] | None = None,
        fact_types: list[str] | None = None,
        exclude_mental_models: bool = False,
        exclude_mental_model_ids: list[str] | None = None,
    ) -> ReflectResponse:
        """
        Generate a contextual answer based on bank identity and memories (async — preferred over :meth:`reflect`).

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
            include_facts: If True, the response will include a 'based_on' field listing
                the memories, mental models, and directives used to construct the answer.
            tag_groups: Optional list of tag group filters for advanced boolean tag matching.
            fact_types: Optional list of fact types to include (world, experience, observation).
            exclude_mental_models: If True, exclude all mental models from reflection (default: False).
            exclude_mental_model_ids: Optional list of specific mental model IDs to exclude.

        Returns:
            ReflectResponse with answer text, optionally facts used, and optionally
            structured_output if response_schema was provided
        """
        include = ReflectIncludeOptions(facts={}) if include_facts else None

        tag_groups_objs = None
        if tag_groups is not None:
            from hindsight_client_api.models.recall_request_tag_groups_inner import RecallRequestTagGroupsInner

            tag_groups_objs = [RecallRequestTagGroupsInner.from_dict(tg) for tg in tag_groups]

        request_obj = reflect_request.ReflectRequest(
            query=query,
            budget=budget,
            context=context,
            max_tokens=max_tokens,
            response_schema=response_schema,
            tags=tags,
            tags_match=tags_match,
            include=include,
            tag_groups=tag_groups_objs,
            fact_types=fact_types,
            exclude_mental_models=exclude_mental_models or None,
            exclude_mental_model_ids=exclude_mental_model_ids,
        )

        return await self._memory_api.reflect(bank_id, request_obj, _request_timeout=self._timeout)

    # Mental Models methods

    def create_mental_model(
        self,
        bank_id: str,
        name: str,
        source_query: str,
        tags: list[str] | None = None,
        max_tokens: int | None = None,
        trigger: dict[str, Any] | None = None,
        id: str | None = None,
    ):
        """
        Create a mental model (sync wrapper — use ``await client.mental_models.create_mental_model(...)`` in async code).

        Args:
            bank_id: The memory bank ID
            name: Human-readable name for the mental model
            source_query: The query to run to generate content
            tags: Optional tags for filtering during retrieval
            max_tokens: Optional maximum tokens for the mental model content
            trigger: Optional trigger settings (e.g., {"refresh_after_consolidation": True})
            id: Optional custom ID for the mental model (alphanumeric lowercase with hyphens)

        Returns:
            CreateMentalModelResponse with operation_id
        """
        from hindsight_client_api.models import create_mental_model_request, mental_model_trigger_input

        trigger_obj = None
        if trigger:
            trigger_obj = mental_model_trigger_input.MentalModelTriggerInput(**trigger)

        request_obj = create_mental_model_request.CreateMentalModelRequest(
            id=id,
            name=name,
            source_query=source_query,
            tags=tags,
            max_tokens=max_tokens,
            trigger=trigger_obj,
        )

        return _run_async(self._mental_models_api.create_mental_model(bank_id, request_obj, _request_timeout=self._timeout))

    def list_mental_models(self, bank_id: str, tags: list[str] | None = None):
        """
        List all mental models in a bank (sync wrapper — use ``await client.mental_models.list_mental_models(...)`` in async code).

        Args:
            bank_id: The memory bank ID
            tags: Optional tags to filter by

        Returns:
            ListMentalModelsResponse with items
        """
        return _run_async(self._mental_models_api.list_mental_models(bank_id, tags=tags, _request_timeout=self._timeout))

    def get_mental_model(self, bank_id: str, mental_model_id: str):
        """
        Get a specific mental model (sync wrapper — use ``await client.mental_models.get_mental_model(...)`` in async code).

        Args:
            bank_id: The memory bank ID
            mental_model_id: The mental model ID

        Returns:
            MentalModelResponse
        """
        return _run_async(self._mental_models_api.get_mental_model(bank_id, mental_model_id, _request_timeout=self._timeout))

    def refresh_mental_model(self, bank_id: str, mental_model_id: str):
        """
        Refresh a mental model (sync wrapper — use ``await client.mental_models.refresh_mental_model(...)`` in async code).

        Args:
            bank_id: The memory bank ID
            mental_model_id: The mental model ID

        Returns:
            RefreshMentalModelResponse with operation_id
        """
        return _run_async(self._mental_models_api.refresh_mental_model(bank_id, mental_model_id, _request_timeout=self._timeout))

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
        Update a mental model's metadata (sync wrapper — use ``await client.mental_models.update_mental_model(...)`` in async code).

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
        from hindsight_client_api.models import mental_model_trigger_input, update_mental_model_request

        trigger_obj = None
        if trigger:
            trigger_obj = mental_model_trigger_input.MentalModelTriggerInput(**trigger)

        request_obj = update_mental_model_request.UpdateMentalModelRequest(
            name=name,
            source_query=source_query,
            tags=tags,
            max_tokens=max_tokens,
            trigger=trigger_obj,
        )

        return _run_async(self._mental_models_api.update_mental_model(bank_id, mental_model_id, request_obj, _request_timeout=self._timeout))

    def delete_mental_model(self, bank_id: str, mental_model_id: str):
        """
        Delete a mental model (sync wrapper — use ``await client.mental_models.delete_mental_model(...)`` in async code).

        Args:
            bank_id: The memory bank ID
            mental_model_id: The mental model ID
        """
        return _run_async(self._mental_models_api.delete_mental_model(bank_id, mental_model_id, _request_timeout=self._timeout))

    def get_mental_model_history(self, bank_id: str, mental_model_id: str):
        """
        Get the content change history of a mental model (sync wrapper — use ``await client.mental_models.get_mental_model_history(...)`` in async code).

        Returns a list of history entries (most recent first), each with
        ``previous_content`` and ``changed_at`` fields.

        Args:
            bank_id: The memory bank ID
            mental_model_id: The mental model ID
        """
        return _run_async(self._mental_models_api.get_mental_model_history(bank_id, mental_model_id, _request_timeout=self._timeout))

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
        Create a directive (sync wrapper — use ``await client.directives.create_directive(...)`` in async code).

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

        return _run_async(self._directives_api.create_directive(bank_id, request_obj, _request_timeout=self._timeout))

    def list_directives(self, bank_id: str, tags: list[str] | None = None):
        """
        List all directives in a bank (sync wrapper — use ``await client.directives.list_directives(...)`` in async code).

        Args:
            bank_id: The memory bank ID
            tags: Optional tags to filter by

        Returns:
            ListDirectivesResponse with items
        """
        return _run_async(self._directives_api.list_directives(bank_id, tags=tags, _request_timeout=self._timeout))

    def get_directive(self, bank_id: str, directive_id: str):
        """
        Get a specific directive (sync wrapper — use ``await client.directives.get_directive(...)`` in async code).

        Args:
            bank_id: The memory bank ID
            directive_id: The directive ID

        Returns:
            DirectiveResponse
        """
        return _run_async(self._directives_api.get_directive(bank_id, directive_id, _request_timeout=self._timeout))

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
        Update a directive (sync wrapper — use ``await client.directives.update_directive(...)`` in async code).

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

        return _run_async(self._directives_api.update_directive(bank_id, directive_id, request_obj, _request_timeout=self._timeout))

    def delete_directive(self, bank_id: str, directive_id: str):
        """
        Delete a directive (sync wrapper — use ``await client.directives.delete_directive(...)`` in async code).

        Args:
            bank_id: The memory bank ID
            directive_id: The directive ID
        """
        return _run_async(self._directives_api.delete_directive(bank_id, directive_id, _request_timeout=self._timeout))

    def get_bank_config(self, bank_id: str) -> dict[str, Any]:
        """
        Get the resolved configuration for a bank (sync wrapper — use ``await client.banks.get_bank_config(...)`` in async code).

        Can be disabled on the server by setting ``HINDSIGHT_API_ENABLE_BANK_CONFIG_API=false``.

        Args:
            bank_id: The memory bank ID

        Returns:
            dict with ``bank_id``, ``config`` (fully resolved), and ``overrides`` (bank-level only)
        """
        return _run_async(self._aget_bank_config(bank_id))

    async def _aget_bank_config(self, bank_id: str) -> dict[str, Any]:
        import aiohttp

        url = f"{self._base_url}/v1/default/banks/{bank_id}/config"
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=self._timeout)) as resp:
                resp.raise_for_status()
                return await resp.json()

    def update_bank_config(
        self,
        bank_id: str,
        *,
        # Reflect settings
        reflect_mission: str | None = None,
        reflect_source_facts_max_tokens: int | None = None,
        # Retain settings
        retain_mission: str | None = None,
        retain_extraction_mode: str | None = None,
        retain_custom_instructions: str | None = None,
        retain_chunk_size: int | None = None,
        retain_default_strategy: str | None = None,
        retain_strategies: dict[str, Any] | None = None,
        # Entity settings
        entity_labels: list[str] | None = None,
        entities_allow_free_form: bool | None = None,
        # Observation / consolidation settings
        enable_observations: bool | None = None,
        observations_mission: str | None = None,
        consolidation_llm_batch_size: int | None = None,
        consolidation_source_facts_max_tokens: int | None = None,
        consolidation_source_facts_max_tokens_per_observation: int | None = None,
        # Disposition settings
        disposition_skepticism: int | None = None,
        disposition_literalism: int | None = None,
        disposition_empathy: int | None = None,
        # MCP settings
        mcp_enabled_tools: list[str] | None = None,
        # Gemini safety settings
        llm_gemini_safety_settings: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Update configuration overrides for a bank (sync wrapper — use ``await client.banks.update_bank_config(...)`` in async code).

        Can be disabled on the server by setting ``HINDSIGHT_API_ENABLE_BANK_CONFIG_API=false``.

        Args:
            bank_id: The memory bank ID
            reflect_mission: Identity and reasoning framing for reflect().
            reflect_source_facts_max_tokens: Token budget for source facts in observation search
                during reflect. -1 to disable (default: -1).
            retain_mission: Steers what gets extracted during retain().
            retain_extraction_mode: Fact extraction mode: 'concise', 'verbose', or 'custom'.
            retain_custom_instructions: Custom extraction prompt (only active when mode is 'custom').
            retain_chunk_size: Maximum token size for each content chunk during retain.
            retain_default_strategy: Default retain strategy name.
            retain_strategies: Named strategy definitions (dict of strategy name to config).
            entity_labels: Controlled vocabulary for entity type classification.
                When set, extracted entities are classified into these labels.
            entities_allow_free_form: Whether to allow entity types outside entity_labels (default: True).
            enable_observations: Toggle automatic observation consolidation after retain().
            observations_mission: Controls what gets synthesised into observations.
            consolidation_llm_batch_size: Number of LLM calls to batch during consolidation.
            consolidation_source_facts_max_tokens: Max tokens for source facts across all observations
                in a consolidation pass.
            consolidation_source_facts_max_tokens_per_observation: Max tokens of source facts per
                individual observation in the consolidation prompt.
            disposition_skepticism: How skeptical vs trusting (1=trusting, 5=skeptical).
            disposition_literalism: How literally to interpret information (1=flexible, 5=literal).
            disposition_empathy: How much to consider emotional context (1=detached, 5=empathetic).
            mcp_enabled_tools: List of MCP tool names to enable for this bank.
            llm_gemini_safety_settings: Gemini/VertexAI safety setting overrides (category → threshold).

        Returns:
            dict with ``bank_id``, ``config`` (fully resolved), and ``overrides`` (bank-level only)
        """
        updates = {
            k: v
            for k, v in {
                "reflect_mission": reflect_mission,
                "reflect_source_facts_max_tokens": reflect_source_facts_max_tokens,
                "retain_mission": retain_mission,
                "retain_extraction_mode": retain_extraction_mode,
                "retain_custom_instructions": retain_custom_instructions,
                "retain_chunk_size": retain_chunk_size,
                "retain_default_strategy": retain_default_strategy,
                "retain_strategies": retain_strategies,
                "entity_labels": entity_labels,
                "entities_allow_free_form": entities_allow_free_form,
                "enable_observations": enable_observations,
                "observations_mission": observations_mission,
                "consolidation_llm_batch_size": consolidation_llm_batch_size,
                "consolidation_source_facts_max_tokens": consolidation_source_facts_max_tokens,
                "consolidation_source_facts_max_tokens_per_observation": consolidation_source_facts_max_tokens_per_observation,
                "disposition_skepticism": disposition_skepticism,
                "disposition_literalism": disposition_literalism,
                "disposition_empathy": disposition_empathy,
                "mcp_enabled_tools": mcp_enabled_tools,
                "llm_gemini_safety_settings": llm_gemini_safety_settings,
            }.items()
            if v is not None
        }
        return _run_async(self._aupdate_bank_config(bank_id, updates))

    async def _aupdate_bank_config(self, bank_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        import aiohttp

        url = f"{self._base_url}/v1/default/banks/{bank_id}/config"
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        async with aiohttp.ClientSession() as session:
            async with session.patch(
                url, json={"updates": updates}, headers=headers, timeout=aiohttp.ClientTimeout(total=self._timeout)
            ) as resp:
                resp.raise_for_status()
                return await resp.json()

    def reset_bank_config(self, bank_id: str) -> dict[str, Any]:
        """
        Reset all bank-level config overrides (sync wrapper — use ``await client.banks.reset_bank_config(...)`` in async code).

        Can be disabled on the server by setting ``HINDSIGHT_API_ENABLE_BANK_CONFIG_API=false``.

        Args:
            bank_id: The memory bank ID

        Returns:
            dict with ``bank_id``, ``config`` (fully resolved), and ``overrides`` (now empty)
        """
        return _run_async(self._areset_bank_config(bank_id))

    async def _areset_bank_config(self, bank_id: str) -> dict[str, Any]:
        import aiohttp

        url = f"{self._base_url}/v1/default/banks/{bank_id}/config"
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        async with aiohttp.ClientSession() as session:
            async with session.delete(url, headers=headers, timeout=aiohttp.ClientTimeout(total=self._timeout)) as resp:
                resp.raise_for_status()
                return await resp.json()

    def delete_bank(self, bank_id: str):
        """
        Delete a memory bank (sync wrapper — prefer :meth:`adelete_bank` in async code).

        Args:
            bank_id: The memory bank ID
        """
        return _run_async(self._banks_api.delete_bank(bank_id, _request_timeout=self._timeout))

    async def adelete_bank(self, bank_id: str):
        """
        Delete a memory bank (async — preferred over :meth:`delete_bank`).

        Args:
            bank_id: The memory bank ID
        """
        return await self._banks_api.delete_bank(bank_id, _request_timeout=self._timeout)
