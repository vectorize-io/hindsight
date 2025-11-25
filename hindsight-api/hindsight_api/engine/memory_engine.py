"""
Memory Engine for AI Agents.

This implements a sophisticated memory architecture that combines:
1. Temporal links: Memories connected by time proximity
2. Semantic links: Memories connected by meaning/similarity
3. Entity links: Memories connected by shared entities (PERSON, ORG, etc.)
4. Spreading activation: Search through the graph with activation decay
5. Dynamic weighting: Recency and frequency-based importance
"""
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
import asyncpg
import asyncio
from .embeddings import Embeddings, SentenceTransformersEmbeddings
from .cross_encoder import CrossEncoderReranker as CrossEncoderModel
import time
import numpy as np
import uuid
import logging
from pydantic import BaseModel, Field

from .query_analyzer import QueryAnalyzer
from .utils import (
    extract_facts,
    calculate_recency_weight,
    calculate_frequency_weight,
)
from .entity_resolver import EntityResolver
from . import (
    embedding_utils,
    link_utils,
    think_utils,
    agent_utils,
)
from .llm_wrapper import LLMConfig
from .response_models import SearchResult as SearchResultModel, ThinkResult, MemoryFact
from .task_backend import TaskBackend, AsyncIOQueueBackend
from .search.reranking import CrossEncoderReranker
from ..pg0 import EmbeddedPostgres


def utcnow():
    """Get current UTC time with timezone info."""
    return datetime.now(timezone.utc)


# Logger for memory system
logger = logging.getLogger(__name__)

from .db_utils import acquire_with_retry, retry_with_backoff

import tiktoken
from dateutil import parser as date_parser

# Cache tiktoken encoding for token budget filtering (module-level singleton)
_TIKTOKEN_ENCODING = None

def _get_tiktoken_encoding():
    """Get cached tiktoken encoding (cl100k_base for GPT-4/3.5)."""
    global _TIKTOKEN_ENCODING
    if _TIKTOKEN_ENCODING is None:
        _TIKTOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
    return _TIKTOKEN_ENCODING


class MemoryEngine:
    """
    Advanced memory system using temporal and semantic linking with PostgreSQL.

    This class provides:
    - Embedding generation for semantic search
    - Entity, temporal, and semantic link creation
    - Think operations for formulating answers with opinions
    - Agent profile and personality management
    """

    def __init__(
        self,
        db_url: str,
        memory_llm_provider: str,
        memory_llm_api_key: str,
        memory_llm_model: str,
        memory_llm_base_url: Optional[str] = None,
        embeddings: Optional[Embeddings] = None,
        cross_encoder: Optional[CrossEncoderModel] = None,
        query_analyzer: Optional[QueryAnalyzer] = None,
        pool_min_size: int = 5,
        pool_max_size: int = 100,
        task_backend: Optional[TaskBackend] = None,
    ):
        """
        Initialize the temporal + semantic memory system.

        Args:
            db_url: PostgreSQL connection URL (postgresql://user:pass@host:port/dbname). Required.
            memory_llm_provider: LLM provider for memory operations: "openai", "groq", or "ollama". Required.
            memory_llm_api_key: API key for the LLM provider. Required.
            memory_llm_model: Model name to use for all memory operations (put/think/opinions). Required.
            memory_llm_base_url: Base URL for the LLM API. Optional. Defaults based on provider:
                                - groq: https://api.groq.com/openai/v1
                                - ollama: http://localhost:11434/v1
            embeddings: Embeddings implementation to use. If not provided, uses SentenceTransformersEmbeddings
            cross_encoder: Cross-encoder model for reranking. If not provided, uses default when cross-encoder reranker is selected
            query_analyzer: Query analyzer implementation to use. If not provided, uses TransformerQueryAnalyzer
            pool_min_size: Minimum number of connections in the pool (default: 5)
            pool_max_size: Maximum number of connections in the pool (default: 100)
                          Increase for parallel think/search operations (e.g., 200-300 for 100+ parallel thinks)
            task_backend: Custom task backend for async task execution. If not provided, uses AsyncIOQueueBackend
        """
        # Track pg0 instance (if used)
        self._pg0: Optional[EmbeddedPostgres] = None

        # Initialize PostgreSQL connection URL
        # "pg0" or "embedded-pg" are special values that trigger embedded PostgreSQL via pg0
        # The actual URL will be set during initialize() after starting the server
        self._use_pg0 = db_url in ("pg0", "embedded-pg")
        self.db_url = db_url if not self._use_pg0 else None


        # Set default base URL if not provided
        if memory_llm_base_url is None:
            if memory_llm_provider.lower() == "groq":
                memory_llm_base_url = "https://api.groq.com/openai/v1"
            elif memory_llm_provider.lower() == "ollama":
                memory_llm_base_url = "http://localhost:11434/v1"
            else:
                memory_llm_base_url = ""

        # Connection pool (will be created in initialize())
        self._pool = None
        self._initialized = False
        self._pool_min_size = pool_min_size
        self._pool_max_size = pool_max_size

        # Initialize entity resolver (will be created in initialize())
        self.entity_resolver = None

        # Initialize embeddings
        if embeddings is not None:
            self.embeddings = embeddings
        else:
            self.embeddings = SentenceTransformersEmbeddings("BAAI/bge-small-en-v1.5")

        # Initialize query analyzer
        if query_analyzer is not None:
            self.query_analyzer = query_analyzer
        else:
            from .query_analyzer import TransformerQueryAnalyzer
            self.query_analyzer = TransformerQueryAnalyzer()

        # Initialize LLM configuration
        self._llm_config = LLMConfig(
            provider=memory_llm_provider,
            api_key=memory_llm_api_key,
            base_url=memory_llm_base_url,
            model=memory_llm_model,
        )

        # Store client and model for convenience (deprecated: use _llm_config.call() instead)
        self._llm_client = self._llm_config._client
        self._llm_model = self._llm_config.model

        # Initialize cross-encoder reranker (cached for performance)
        self._cross_encoder_reranker = CrossEncoderReranker(cross_encoder=cross_encoder)

        # Initialize task backend
        self._task_backend = task_backend or AsyncIOQueueBackend(
            batch_size=100,
            batch_interval=1.0
        )

        # Backpressure mechanism: limit concurrent searches to prevent overwhelming the database
        # Limit concurrent searches to prevent connection pool exhaustion
        # Each search can use 2-4 connections, so with 10 concurrent searches
        # we use ~20-40 connections max, staying well within pool limits
        self._search_semaphore = asyncio.Semaphore(10)

        # Backpressure for put operations: limit concurrent puts to prevent database contention
        # Each put_batch holds a connection for the entire transaction, so we limit to 5
        # concurrent puts to avoid connection pool exhaustion and reduce write contention
        self._put_semaphore = asyncio.Semaphore(5)

        # initialize encoding eagerly to avoid delaying the first time
        _get_tiktoken_encoding()

    async def _handle_access_count_update(self, task_dict: Dict[str, Any]):
        """
        Handler for access count update tasks.

        Args:
            task_dict: Dict with 'node_ids' key containing list of node IDs to update
        """
        node_ids = task_dict.get('node_ids', [])
        if not node_ids:
            return

        pool = await self._get_pool()
        try:
            # Convert string UUIDs to UUID type for faster matching
            uuid_list = [uuid.UUID(nid) for nid in node_ids]
            async with acquire_with_retry(pool) as conn:
                await conn.execute(
                    "UPDATE memory_units SET access_count = access_count + 1 WHERE id = ANY($1::uuid[])",
                    uuid_list
                )
        except Exception as e:
            logger.error(f"Access count handler: Error updating access counts: {e}")

    async def _handle_batch_put(self, task_dict: Dict[str, Any]):
        """
        Handler for batch put tasks.

        Args:
            task_dict: Dict with 'agent_id', 'contents', 'document_id'
        """
        try:
            agent_id = task_dict.get('agent_id')
            contents = task_dict.get('contents', [])
            document_id = task_dict.get('document_id')

            logger.info(f"[BATCH_PUT_TASK] Starting background batch put for agent_id={agent_id}, {len(contents)} items")

            await self.put_batch_async(
                agent_id=agent_id,
                contents=contents,
                document_id=document_id
            )

            logger.info(f"[BATCH_PUT_TASK] Completed background batch put for agent_id={agent_id}")
        except Exception as e:
            logger.error(f"Batch put handler: Error processing batch put: {e}")
            import traceback
            traceback.print_exc()

    async def execute_task(self, task_dict: Dict[str, Any]):
        """
        Execute a task by routing it to the appropriate handler.

        This method is called by the task backend to execute tasks.
        It receives a plain dict that can be serialized and sent over the network.

        Args:
            task_dict: Task dictionary with 'type' key and other payload data
                      Example: {'type': 'access_count_update', 'node_ids': [...]}
        """
        task_type = task_dict.get('type')
        operation_id = task_dict.get('operation_id')
        retry_count = task_dict.get('retry_count', 0)
        max_retries = 3

        # Check if operation was cancelled (only for tasks with operation_id)
        if operation_id:
            try:
                pool = await self._get_pool()
                async with acquire_with_retry(pool) as conn:
                    result = await conn.fetchrow(
                        "SELECT id FROM async_operations WHERE id = $1",
                        uuid.UUID(operation_id)
                    )
                    if not result:
                        # Operation was cancelled, skip processing
                        logger.info(f"Skipping cancelled operation: {operation_id}")
                        return
            except Exception as e:
                logger.error(f"Failed to check operation status {operation_id}: {e}")
                # Continue with processing if we can't check status

        try:
            if task_type == 'access_count_update':
                await self._handle_access_count_update(task_dict)
            elif task_type == 'reinforce_opinion':
                await self._handle_reinforce_opinion(task_dict)
            elif task_type == 'form_opinion':
                await self._handle_form_opinion(task_dict)
            elif task_type == 'batch_put':
                await self._handle_batch_put(task_dict)
            else:
                logger.error(f"Unknown task type: {task_type}")
                # Don't retry unknown task types
                if operation_id:
                    await self._delete_operation_record(operation_id)
                return

            # Task succeeded - delete operation record
            if operation_id:
                await self._delete_operation_record(operation_id)

        except Exception as e:
            # Task failed - check if we should retry
            logger.error(f"Task execution failed (attempt {retry_count + 1}/{max_retries + 1}): {task_type}, error: {e}")
            import traceback
            error_traceback = traceback.format_exc()
            traceback.print_exc()

            if retry_count < max_retries:
                # Reschedule with incremented retry count
                task_dict['retry_count'] = retry_count + 1
                logger.info(f"Rescheduling task {task_type} (retry {retry_count + 1}/{max_retries})")
                await self._task_backend.submit_task(task_dict)
            else:
                # Max retries exceeded - mark operation as failed
                logger.error(f"Max retries exceeded for task {task_type}, marking as failed")
                if operation_id:
                    await self._mark_operation_failed(operation_id, str(e), error_traceback)

    async def _delete_operation_record(self, operation_id: str):
        """Helper to delete an operation record from the database."""
        try:
            pool = await self._get_pool()
            async with acquire_with_retry(pool) as conn:
                await conn.execute(
                    "DELETE FROM async_operations WHERE id = $1",
                    uuid.UUID(operation_id)
                )
            logger.debug(f"Deleted async operation record: {operation_id}")
        except Exception as e:
            logger.error(f"Failed to delete async operation record {operation_id}: {e}")

    async def _mark_operation_failed(self, operation_id: str, error_message: str, error_traceback: str):
        """Helper to mark an operation as failed in the database."""
        try:
            pool = await self._get_pool()
            # Truncate error message to avoid extremely long strings
            full_error = f"{error_message}\n\nTraceback:\n{error_traceback}"
            truncated_error = full_error[:5000] if len(full_error) > 5000 else full_error

            async with acquire_with_retry(pool) as conn:
                await conn.execute(
                    """
                    UPDATE async_operations
                    SET status = 'failed', error_message = $2
                    WHERE id = $1
                    """,
                    uuid.UUID(operation_id),
                    truncated_error
                )
            logger.info(f"Marked async operation as failed: {operation_id}")
        except Exception as e:
            logger.error(f"Failed to mark operation as failed {operation_id}: {e}")

    async def initialize(self):
        """Initialize the connection pool and background workers."""
        if self._initialized:
            return

        # Start pg0 embedded PostgreSQL if configured
        if self._use_pg0:
            logger.info("Starting pg0 embedded PostgreSQL...")
            self._pg0 = EmbeddedPostgres()
            self.db_url = await self._pg0.ensure_running()
            logger.info(f"pg0 PostgreSQL running at: {self.db_url}")

        # Create connection pool
        # For read-heavy workloads with many parallel think/search operations,
        # we need a larger pool. Read operations don't need strong isolation.
        self._pool = await asyncpg.create_pool(
            self.db_url,
            min_size=self._pool_min_size,
            max_size=self._pool_max_size,
            command_timeout=60,
            statement_cache_size=0,  # Disable prepared statement cache
            timeout=30,  # Connection acquisition timeout (seconds)
        )

        # Initialize entity resolver with pool
        self.entity_resolver = EntityResolver(self._pool)

        # Set executor for task backend and initialize
        self._task_backend.set_executor(self.execute_task)
        await self._task_backend.initialize()

        self._initialized = True
        logger.info("Memory system initialized (pool and task backend started)")

    async def _get_pool(self) -> asyncpg.Pool:
        """Get the connection pool (must call initialize() first)."""
        if not self._initialized:
            await self.initialize()
        return self._pool

    async def _acquire_connection(self):
        """
        Acquire a connection from the pool with retry logic.

        Returns an async context manager that yields a connection.
        Retries on transient connection errors with exponential backoff.
        """
        pool = await self._get_pool()

        async def acquire():
            return await pool.acquire()

        return await _retry_with_backoff(acquire)

    async def close(self):
        """Close the connection pool and shutdown background workers."""
        logger.info("close() started")

        # Shutdown task backend
        logger.debug("shutting down task backend")
        await self._task_backend.shutdown()
        logger.debug("task backend shutdown complete")

        # Close pool
        if self._pool is not None:
            logger.debug("closing connection pool")
            self._pool.terminate()
            logger.debug("connection pool closed")
            self._pool = None
        else:
            logger.debug("no pool to close")

        self._initialized = False

        # Stop pg0 if we started it
        if self._pg0 is not None:
            logger.info("Stopping pg0...")
            await self._pg0.stop()
            self._pg0 = None
            logger.info("pg0 stopped")

        logger.debug("close() completed")

    async def wait_for_background_tasks(self):
        """
        Wait for all pending background tasks to complete.

        This is useful in tests to ensure background tasks (like opinion reinforcement)
        complete before making assertions.
        """
        if hasattr(self._task_backend, 'wait_for_pending_tasks'):
            await self._task_backend.wait_for_pending_tasks()

    def _format_readable_date(self, dt: datetime) -> str:
        """
        Format a datetime into a readable string for temporal matching.

        Examples:
            - June 2024
            - January 15, 2024
            - December 2023

        This helps queries like "camping in June" match facts that happened in June.

        Args:
            dt: datetime object to format

        Returns:
            Readable date string
        """
        # Format as "Month Year" for most cases
        # Could be extended to include day for very specific dates if needed
        month_name = dt.strftime("%B")  # Full month name (e.g., "June")
        year = dt.strftime("%Y")  # Year (e.g., "2024")

        # For now, use "Month Year" format
        # Could check if day is significant (not 1st or 15th) and include it
        return f"{month_name} {year}"

    async def _find_duplicate_facts_batch(
        self,
        conn,
        agent_id: str,
        texts: List[str],
        embeddings: List[List[float]],
        event_date: datetime,
        time_window_hours: int = 24,
        similarity_threshold: float = 0.95
    ) -> List[bool]:
        """
        Check which facts are duplicates using semantic similarity + temporal window.

        For each new fact, checks if a semantically similar fact already exists
        within the time window. Uses pgvector cosine similarity for efficiency.

        Args:
            conn: Database connection
            agent_id: Agent identifier
            texts: List of fact texts to check
            embeddings: Corresponding embeddings
            event_date: Event date for temporal filtering
            time_window_hours: Hours before/after event_date to search (default: 24)
            similarity_threshold: Minimum cosine similarity to consider duplicate (default: 0.95)

        Returns:
            List of booleans - True if fact is a duplicate (should skip), False if new
        """
        if not texts:
            return []

        time_lower = event_date - timedelta(hours=time_window_hours)
        time_upper = event_date + timedelta(hours=time_window_hours)

        # Fetch ALL existing facts in time window ONCE (much faster than N queries)
        import time as time_mod
        fetch_start = time_mod.time()
        existing_facts = await conn.fetch(
            """
            SELECT id, text, embedding
            FROM memory_units
            WHERE agent_id = $1
              AND event_date BETWEEN $2 AND $3
            """,
            agent_id, time_lower, time_upper
        )
        logger.debug(f"      [3.X] Fetched {len(existing_facts)} existing facts in {time_mod.time() - fetch_start:.3f}s")

        # If no existing facts, nothing is duplicate
        if not existing_facts:
            return [False] * len(texts)

        # Compute similarities in Python (vectorized with numpy)
        import numpy as np
        is_duplicate = []

        # Convert existing embeddings to numpy for faster computation
        embedding_arrays = []
        for row in existing_facts:
            raw_emb = row['embedding']
            # Handle different pgvector formats
            if isinstance(raw_emb, str):
                # Parse string format: "[1.0, 2.0, ...]"
                import json
                emb = np.array(json.loads(raw_emb), dtype=np.float32)
            elif isinstance(raw_emb, (list, tuple)):
                emb = np.array(raw_emb, dtype=np.float32)
            else:
                # Try direct conversion
                emb = np.array(raw_emb, dtype=np.float32)
            embedding_arrays.append(emb)

        if not embedding_arrays:
            existing_embeddings = np.array([])
        elif len(embedding_arrays) == 1:
            # Single embedding: reshape to (1, dim)
            existing_embeddings = embedding_arrays[0].reshape(1, -1)
        else:
            # Multiple embeddings: vstack
            existing_embeddings = np.vstack(embedding_arrays)

        comp_start = time_mod.time()
        for embedding in embeddings:
            # Compute cosine similarity with all existing facts
            emb_array = np.array(embedding)
            # Cosine similarity = 1 - cosine distance
            # For normalized vectors: cosine_sim = dot product
            similarities = np.dot(existing_embeddings, emb_array)

            # Check if any existing fact is too similar
            max_similarity = np.max(similarities) if len(similarities) > 0 else 0
            is_duplicate.append(max_similarity > similarity_threshold)

        logger.debug(f"      [3.X] Computed {len(texts)} x {len(existing_facts)} similarities in {time_mod.time() - comp_start:.3f}s")

        return is_duplicate

    def put(
        self,
        agent_id: str,
        content: str,
        context: str = "",
        event_date: Optional[datetime] = None,
    ) -> List[str]:
        """
        Store content as memory units (synchronous wrapper).

        This is a synchronous wrapper around put_async() for convenience.
        For best performance, use put_async() directly.

        Args:
            agent_id: Unique identifier for the agent
            content: Text content to store
            context: Context about when/why this memory was formed
            event_date: When the event occurred (defaults to now)

        Returns:
            List of created unit IDs
        """
        # Run async version synchronously
        return asyncio.run(self.put_async(agent_id, content, context, event_date))

    async def put_async(
        self,
        agent_id: str,
        content: str,
        context: str = "",
        event_date: Optional[datetime] = None,
        document_id: Optional[str] = None,
        fact_type_override: Optional[str] = None,
        confidence_score: Optional[float] = None,
    ) -> List[str]:
        """
        Store content as memory units with temporal and semantic links (ASYNC version).

        This is a convenience wrapper around put_batch_async for a single content item.

        Args:
            agent_id: Unique identifier for the agent
            content: Text content to store
            context: Context about when/why this memory was formed
            event_date: When the event occurred (defaults to now)
            document_id: Optional document ID for tracking (always upserts if document already exists)
            fact_type_override: Override fact type ('world', 'agent', 'opinion')
            confidence_score: Confidence score for opinions (0.0 to 1.0)

        Returns:
            List of created unit IDs
        """
        # Use put_batch_async with a single item (avoids code duplication)
        result = await self.put_batch_async(
            agent_id=agent_id,
            contents=[{
                "content": content,
                "context": context,
                "event_date": event_date
            }],
            document_id=document_id,
            fact_type_override=fact_type_override,
            confidence_score=confidence_score
        )

        # Return the first (and only) list of unit IDs
        return result[0] if result else []

    async def put_batch_async(
        self,
        agent_id: str,
        contents: List[Dict[str, Any]],
        document_id: Optional[str] = None,
        fact_type_override: Optional[str] = None,
        confidence_score: Optional[float] = None,
    ) -> List[List[str]]:
        """
        Store multiple content items as memory units in ONE batch operation.

        This is MUCH more efficient than calling put_async multiple times:
        - Extracts facts from all contents in parallel
        - Generates ALL embeddings in ONE batch
        - Does ALL database operations in ONE transaction
        - Automatically chunks large batches to prevent timeouts

        Args:
            agent_id: Unique identifier for the agent
            contents: List of dicts with keys:
                - "content" (required): Text content to store
                - "context" (optional): Context about the memory
                - "event_date" (optional): When the event occurred
            document_id: Optional document ID for tracking (always upserts if document already exists)
            fact_type_override: Override fact type for all facts ('world', 'agent', 'opinion')
            confidence_score: Confidence score for opinions (0.0 to 1.0)

        Returns:
            List of lists of unit IDs (one list per content item)

        Example:
            unit_ids = await memory.put_batch_async(
                agent_id="user123",
                contents=[
                    {"content": "Alice works at Google", "context": "conversation"},
                    {"content": "Bob loves Python", "context": "conversation"},
                ],
                document_id="meeting-2024-01-15"
            )
            # Returns: [["unit-id-1"], ["unit-id-2"]]
        """
        start_time = time.time()

        if not contents:
            return []

        # Auto-chunk large batches by character count to avoid timeouts and memory issues
        # Calculate total character count
        total_chars = sum(len(item.get("content", "")) for item in contents)

        CHARS_PER_BATCH = 500_000

        if total_chars > CHARS_PER_BATCH:
            # Split into smaller batches based on character count
            logger.info(f"Large batch detected ({total_chars:,} chars from {len(contents)} items). Splitting into sub-batches of ~{CHARS_PER_BATCH:,} chars each...")

            sub_batches = []
            current_batch = []
            current_batch_chars = 0

            for item in contents:
                item_chars = len(item.get("content", ""))

                # If adding this item would exceed the limit, start a new batch
                # (unless current batch is empty - then we must include it even if it's large)
                if current_batch and current_batch_chars + item_chars > CHARS_PER_BATCH:
                    sub_batches.append(current_batch)
                    current_batch = [item]
                    current_batch_chars = item_chars
                else:
                    current_batch.append(item)
                    current_batch_chars += item_chars

            # Add the last batch
            if current_batch:
                sub_batches.append(current_batch)

            logger.info(f"Split into {len(sub_batches)} sub-batches: {[len(b) for b in sub_batches]} items each")

            # Process each sub-batch using internal method (skip chunking check)
            all_results = []
            for i, sub_batch in enumerate(sub_batches, 1):
                sub_batch_chars = sum(len(item.get("content", "")) for item in sub_batch)
                logger.info(f"Processing sub-batch {i}/{len(sub_batches)}: {len(sub_batch)} items, {sub_batch_chars:,} chars")

                sub_results = await self._put_batch_async_internal(
                    agent_id=agent_id,
                    contents=sub_batch,
                    document_id=document_id,
                    is_first_batch=i == 1,  # Only upsert on first batch
                    fact_type_override=fact_type_override,
                    confidence_score=confidence_score
                )
                all_results.extend(sub_results)

            total_time = time.time() - start_time
            logger.info(f"PUT_BATCH_ASYNC (chunked) COMPLETE: {len(all_results)} results from {len(contents)} contents in {total_time:.3f}s")
            return all_results

        # Small batch - use internal method directly
        return await self._put_batch_async_internal(
            agent_id=agent_id,
            contents=contents,
            document_id=document_id,
            is_first_batch=True,
            fact_type_override=fact_type_override,
            confidence_score=confidence_score
        )

    async def _put_batch_async_internal(
        self,
        agent_id: str,
        contents: List[Dict[str, Any]],
        document_id: Optional[str] = None,
        is_first_batch: bool = True,
        fact_type_override: Optional[str] = None,
        confidence_score: Optional[float] = None,
    ) -> List[List[str]]:
        """
        Internal method for batch processing without chunking logic.

        Assumes contents are already appropriately sized (< 50k chars).
        Called by put_batch_async after chunking large batches.

        Uses semaphore for backpressure to limit concurrent puts.

        Args:
            agent_id: Unique identifier for the agent
            contents: List of dicts with content, context, event_date
            document_id: Optional document ID (always upserts if exists)
            is_first_batch: Whether this is the first batch (for chunked operations, only delete on first batch)
            fact_type_override: Override fact type for all facts
            confidence_score: Confidence score for opinions
        """
        # Backpressure: limit concurrent puts to prevent database contention
        async with self._put_semaphore:
            start_time = time.time()
            total_chars = sum(len(item.get("content", "")) for item in contents)

            # Buffer all logs to avoid interleaving
            log_buffer = []
            log_buffer.append(f"{'='*60}")
            log_buffer.append(f"PUT_BATCH_ASYNC START: {agent_id}")
            log_buffer.append(f"Batch size: {len(contents)} content items, {total_chars:,} chars")
            log_buffer.append(f"{'='*60}")

            # Get agent name for fact extraction
            pool = await self._get_pool()
            profile = await agent_utils.get_agent_profile(pool, agent_id)
            agent_name = profile["name"]

            # Step 1: Extract facts from ALL contents in parallel
            step_start = time.time()

            # If fact_type_override is 'opinion', extract only opinions; otherwise extract world and agent facts
            extract_opinions = (fact_type_override == 'opinion')

            # Create tasks for parallel fact extraction using configured LLM
            fact_extraction_tasks = []
            for item in contents:
                content = item["content"]
                context = item.get("context", "")
                event_date = item.get("event_date") or utcnow()
                metadata = item.get("metadata") or {}

                task = extract_facts(content, event_date, context, llm_config=self._llm_config, agent_name=agent_name, extract_opinions=extract_opinions)
                fact_extraction_tasks.append((task, event_date, context, metadata))

            # Wait for all fact extractions to complete
            all_fact_results = await asyncio.gather(*[task for task, _, _, _ in fact_extraction_tasks])
            log_buffer.append(f"[1] Extract facts (parallel): {len(fact_extraction_tasks)} contents in {time.time() - step_start:.3f}s")

            # Flatten and track which facts belong to which content
            all_fact_texts = []
            all_fact_dates = []
            all_occurred_starts = []  # NEW: When fact occurred (range start)
            all_occurred_ends = []    # NEW: When fact occurred (range end)
            all_mentioned_ats = []    # NEW: When fact was mentioned
            all_contexts = []
            all_fact_entities = []  # NEW: Store LLM-extracted entities per fact
            all_fact_types = []  # Store fact type (world or agent)
            all_causal_relations = []  # NEW: Store causal relationships per fact
            all_metadata = []  # User-defined metadata for each fact
            content_boundaries = []  # [(start_idx, end_idx), ...]

            current_idx = 0
            for i, ((_, event_date, context, metadata), fact_dicts) in enumerate(zip(fact_extraction_tasks, all_fact_results)):
                start_idx = current_idx

                for fact_dict in fact_dicts:
                    all_fact_texts.append(fact_dict['fact'])

                    # Extract temporal fields (new schema with ranges)

                    try:
                        # Try new schema first (occurred_start/end)
                        occurred_start = date_parser.isoparse(fact_dict['occurred_start'])
                        occurred_end = date_parser.isoparse(fact_dict['occurred_end'])
                        all_occurred_starts.append(occurred_start)
                        all_occurred_ends.append(occurred_end)
                        # Use occurred_start as event_date for backward compatibility
                        all_fact_dates.append(occurred_start)
                    except (KeyError, Exception):
                        # Fallback to old schema (single 'date' field)
                        try:
                            fact_date = date_parser.isoparse(fact_dict['date'])
                        except Exception:
                            fact_date = event_date
                        all_fact_dates.append(fact_date)
                        # For old schema, use same date for start and end (point event)
                        all_occurred_starts.append(fact_date)
                        all_occurred_ends.append(fact_date)

                    # mentioned_at is when the fact was mentioned (conversation date)
                    all_mentioned_ats.append(event_date)

                    all_contexts.append(context)
                    # Extract entities from fact (default to empty list if not present)
                    all_fact_entities.append(fact_dict.get('entities', []))
                    # Extract fact type (use override if provided, else use extracted type or default to 'world')
                    if fact_type_override:
                        all_fact_types.append(fact_type_override)
                    else:
                        all_fact_types.append(fact_dict.get('fact_type', 'world'))
                    # Extract causal relations (with global index adjustment)
                    # Causal relations use fact indices within each content, need to adjust to global indices
                    causal_relations = fact_dict.get('causal_relations', []) or []
                    # Adjust target_fact_index to global index by adding start_idx
                    adjusted_relations = []
                    for rel in causal_relations:
                        adjusted_rel = dict(rel)
                        adjusted_rel['target_fact_index'] = start_idx + rel['target_fact_index']
                        adjusted_relations.append(adjusted_rel)
                    all_causal_relations.append(adjusted_relations)
                    # Each fact inherits metadata from its source content item
                    all_metadata.append(metadata)

                end_idx = current_idx + len(fact_dicts)
                content_boundaries.append((start_idx, end_idx))
                current_idx = end_idx

            total_facts = len(all_fact_texts)

            if total_facts == 0:
                return [[] for _ in contents]

            # Step 1.5: Add time offsets to preserve fact ordering within each document
            # This allows retrieval to distinguish between facts that happened earlier vs later
            # in the same conversation, even when the base event_date is the same
            SECONDS_PER_FACT = 10  # Each fact gets 10 seconds offset
            for start_idx, end_idx in content_boundaries:
                # For each content item, offset its facts sequentially
                for i in range(start_idx, end_idx):
                    fact_position = i - start_idx  # 0, 1, 2, ...
                    offset = timedelta(seconds=fact_position * SECONDS_PER_FACT)
                    # Add incremental offset to preserve order (facts appear in extraction order)
                    all_fact_dates[i] = all_fact_dates[i] + offset
                    all_occurred_starts[i] = all_occurred_starts[i] + offset
                    all_occurred_ends[i] = all_occurred_ends[i] + offset

            log_buffer.append(f"[1.5] Added time offsets: {SECONDS_PER_FACT}s per fact to preserve ordering")

            # Step 2: Augment fact texts with readable dates for better temporal matching
            # This allows queries like "camping in June" to match facts that happened in June
            augmented_texts = []
            for fact_text, fact_date in zip(all_fact_texts, all_fact_dates):
                # Format date in readable form
                readable_date = self._format_readable_date(fact_date)
                # Augment text with date for embedding (but store original text in DB)
                augmented_text = f"{fact_text} (happened in {readable_date})"
                augmented_texts.append(augmented_text)

            # Step 2b: Generate ALL embeddings in ONE batch using augmented texts (HUGE speedup!)
            step_start = time.time()
            all_embeddings = await embedding_utils.generate_embeddings_batch(self.embeddings, augmented_texts)
            log_buffer.append(f"[2] Generate embeddings (parallel): {len(all_embeddings)} embeddings in {time.time() - step_start:.3f}s")

            # Step 3: Process everything in ONE database transaction
            logger.debug("Getting connection pool")
            pool = await self._get_pool()
            logger.debug("Acquiring connection from pool")
            async with acquire_with_retry(pool) as conn:
                logger.debug("Starting transaction")
                async with conn.transaction():
                    logger.debug("Inside transaction")
                    try:
                        # Ensure agent exists in agents table (create with defaults if not exists)
                        # Update updated_at to reflect recent activity
                        logger.debug(f"Ensuring agent '{agent_id}' exists in agents table")
                        await conn.execute(
                            """
                            INSERT INTO agents (agent_id, personality, background)
                            VALUES ($1, $2::jsonb, $3)
                            ON CONFLICT (agent_id) DO UPDATE
                            SET updated_at = NOW()
                            """,
                            agent_id,
                            '{"openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5, "agreeableness": 0.5, "neuroticism": 0.5, "bias_strength": 0.5}',
                            ""
                        )

                        # Handle document tracking with automatic upsert
                        if document_id:
                            logger.debug(f"Handling document tracking for {document_id}")
                            import hashlib
                            import json

                            # Calculate content hash from all content items
                            combined_content = "\n".join([c.get("content", "") for c in contents])
                            content_hash = hashlib.sha256(combined_content.encode()).hexdigest()

                            # Always delete old document first if it exists (cascades to units and links)
                            # Only delete on the first batch to avoid deleting data we just inserted
                            if is_first_batch:
                                deleted = await conn.fetchval(
                                    "DELETE FROM documents WHERE id = $1 AND agent_id = $2 RETURNING id",
                                    document_id, agent_id
                                )
                                if deleted:
                                    logger.debug(f"[3.1] Upsert: Deleted existing document '{document_id}' and all its units")

                            # Insert document (or update if exists from concurrent operations)
                            # Use ON CONFLICT for idempotent behavior in edge cases
                            await conn.execute(
                                """
                                INSERT INTO documents (id, agent_id, original_text, content_hash, metadata)
                                VALUES ($1, $2, $3, $4, $5)
                                ON CONFLICT (id, agent_id) DO UPDATE
                                SET original_text = EXCLUDED.original_text,
                                    content_hash = EXCLUDED.content_hash,
                                    metadata = EXCLUDED.metadata,
                                    updated_at = NOW()
                                """,
                                document_id,
                                agent_id,
                                combined_content,
                                content_hash,
                                json.dumps({})  # Empty metadata dict
                            )
                            logger.debug(f"[3.2] Document '{document_id}' stored/updated")

                        # Deduplication check for all facts (batched by time window)
                        logger.debug("Starting deduplication check")
                        step_start = time.time()

                        # Group facts by event_date (rounded to 12-hour buckets) for batching
                        from collections import defaultdict
                        time_buckets = defaultdict(list)
                        for idx, (sentence, embedding, fact_date) in enumerate(zip(all_fact_texts, all_embeddings, all_fact_dates)):
                            # Round to 12-hour bucket to group similar times
                            bucket_key = fact_date.replace(hour=(fact_date.hour // 12) * 12, minute=0, second=0, microsecond=0)
                            time_buckets[bucket_key].append((idx, sentence, embedding, fact_date))

                        # Process each bucket in batch
                        all_is_duplicate = [False] * total_facts  # Initialize all as not duplicate
                        for bucket_date, bucket_items in time_buckets.items():
                            indices = [item[0] for item in bucket_items]
                            sentences = [item[1] for item in bucket_items]
                            embeddings = [item[2] for item in bucket_items]
                            # Use bucket_date as representative for time window
                            dup_flags = await self._find_duplicate_facts_batch(
                                conn, agent_id, sentences, embeddings, bucket_date, time_window_hours=24
                            )
                            # Map results back to original indices
                            for idx, is_dup in zip(indices, dup_flags):
                                all_is_duplicate[idx] = is_dup

                        duplicates_filtered = sum(all_is_duplicate)
                        new_facts = total_facts - duplicates_filtered
                        logger.debug(f"Deduplication complete: {duplicates_filtered} duplicates filtered, {new_facts} new facts ({len(time_buckets)} time buckets)")
                        log_buffer.append(f"[3] Deduplication check: {duplicates_filtered} duplicates filtered, {new_facts} new facts in {time.time() - step_start:.3f}s")

                        # Filter out duplicates
                        filtered_sentences = [s for s, is_dup in zip(all_fact_texts, all_is_duplicate) if not is_dup]
                        filtered_embeddings = [e for e, is_dup in zip(all_embeddings, all_is_duplicate) if not is_dup]
                        filtered_dates = [d for d, is_dup in zip(all_fact_dates, all_is_duplicate) if not is_dup]
                        filtered_occurred_starts = [d for d, is_dup in zip(all_occurred_starts, all_is_duplicate) if not is_dup]
                        filtered_occurred_ends = [d for d, is_dup in zip(all_occurred_ends, all_is_duplicate) if not is_dup]
                        filtered_mentioned_ats = [d for d, is_dup in zip(all_mentioned_ats, all_is_duplicate) if not is_dup]
                        filtered_contexts = [c for c, is_dup in zip(all_contexts, all_is_duplicate) if not is_dup]
                        filtered_entities = [ents for ents, is_dup in zip(all_fact_entities, all_is_duplicate) if not is_dup]
                        filtered_fact_types = [ft for ft, is_dup in zip(all_fact_types, all_is_duplicate) if not is_dup]
                        filtered_metadata = [m for m, is_dup in zip(all_metadata, all_is_duplicate) if not is_dup]

                        # Build index mapping from old indices to new indices (accounting for removed duplicates)
                        old_to_new_index = {}
                        new_idx = 0
                        for old_idx, is_dup in enumerate(all_is_duplicate):
                            if not is_dup:
                                old_to_new_index[old_idx] = new_idx
                                new_idx += 1

                        # Filter and remap causal relations
                        filtered_causal_relations = []
                        for old_idx, (relations, is_dup) in enumerate(zip(all_causal_relations, all_is_duplicate)):
                            if not is_dup:
                                # Keep relations where both source and target survived deduplication
                                valid_relations = []
                                for rel in relations:
                                    target_idx = rel['target_fact_index']
                                    # Only keep if target fact wasn't filtered out
                                    if target_idx in old_to_new_index:
                                        remapped_rel = dict(rel)
                                        remapped_rel['target_fact_index'] = old_to_new_index[target_idx]
                                        valid_relations.append(remapped_rel)
                                filtered_causal_relations.append(valid_relations)

                        if not filtered_sentences:
                            logger.debug(f"[PUT_BATCH_ASYNC] All facts were duplicates, returning empty")
                            return [[] for _ in contents]

                        # Batch insert ALL units
                        step_start = time.time()
                        # Convert embeddings to strings for asyncpg vector type
                        filtered_embeddings_str = [str(emb) for emb in filtered_embeddings]
                        # Prepare confidence scores (only for opinions)
                        # If fact_type is 'opinion' and no confidence_score provided, use default of 1.0
                        confidence_scores = [
                            confidence_score if confidence_score is not None else 1.0
                            if ft == 'opinion'
                            else None
                            for ft in filtered_fact_types
                        ]
                        # Convert metadata dicts to JSON strings for asyncpg
                        import json
                        filtered_metadata_json = [json.dumps(m) if m else '{}' for m in filtered_metadata]
                        results = await conn.fetch(
                            """
                            INSERT INTO memory_units (agent_id, document_id, text, context, embedding, event_date, occurred_start, occurred_end, mentioned_at, fact_type, confidence_score, access_count, metadata)
                            SELECT * FROM unnest($1::text[], $2::text[], $3::text[], $4::text[], $5::vector[], $6::timestamptz[], $7::timestamptz[], $8::timestamptz[], $9::timestamptz[], $10::text[], $11::float[], $12::integer[], $13::jsonb[])
                            RETURNING id
                            """,
                            [agent_id] * len(filtered_sentences),
                            [document_id] * len(filtered_sentences) if document_id else [None] * len(filtered_sentences),
                            filtered_sentences,
                            filtered_contexts,
                            filtered_embeddings_str,
                            filtered_dates,
                            filtered_occurred_starts,
                            filtered_occurred_ends,
                            filtered_mentioned_ats,
                            filtered_fact_types,
                            confidence_scores,
                            [0] * len(filtered_sentences),
                            filtered_metadata_json
                        )

                        created_unit_ids = [str(row['id']) for row in results]
                        logger.debug(f"Batch insert complete: {len(created_unit_ids)} units created")
                        log_buffer.append(f"[5] Batch insert units: {len(created_unit_ids)} units in {time.time() - step_start:.3f}s")

                        # Process entities for ALL units
                        logger.debug("Processing entities")
                        step_start = time.time()
                        all_entity_links = await link_utils.extract_entities_batch_optimized(
                            self.entity_resolver, conn, agent_id, created_unit_ids, filtered_sentences, "", filtered_dates, filtered_entities, log_buffer
                        )
                        logger.debug(f"Entity processing complete: {len(all_entity_links)} links")
                        log_buffer.append(f"[6] Process entities (batched): {time.time() - step_start:.3f}s")

                        # Create temporal links
                        logger.debug("Creating temporal links")
                        step_start = time.time()
                        await link_utils.create_temporal_links_batch_per_fact(conn, agent_id, created_unit_ids, log_buffer=log_buffer)
                        logger.debug("Temporal links complete")
                        log_buffer.append(f"[7] Batch create temporal links: {time.time() - step_start:.3f}s")

                        # Create semantic links
                        logger.debug("Creating semantic links")
                        step_start = time.time()
                        await link_utils.create_semantic_links_batch(conn, agent_id, created_unit_ids, filtered_embeddings, log_buffer=log_buffer)
                        logger.debug("Semantic links complete")
                        log_buffer.append(f"[8] Batch create semantic links: {time.time() - step_start:.3f}s")

                        # Insert entity links
                        logger.debug("Inserting entity links")
                        step_start = time.time()
                        if all_entity_links:
                            await link_utils.insert_entity_links_batch(conn, all_entity_links)
                        logger.debug("Entity links inserted")
                        log_buffer.append(f"[9] Batch insert entity links: {time.time() - step_start:.3f}s")

                        # Create causal links
                        logger.debug("Creating causal links")
                        step_start = time.time()
                        causal_link_count = await link_utils.create_causal_links_batch(
                            conn, created_unit_ids, filtered_causal_relations
                        )
                        logger.debug(f"Causal links complete: {causal_link_count} links created")
                        log_buffer.append(f"[10] Batch create causal links: {causal_link_count} links in {time.time() - step_start:.3f}s")

                        # Transaction auto-commits on success
                        commit_start = time.time()
                        logger.debug(f"[10] Commit: {time.time() - commit_start:.3f}s")

                        # Map created unit IDs back to original content items
                        # Account for duplicates when mapping back
                        result_unit_ids = []
                        filtered_idx = 0

                        for start_idx, end_idx in content_boundaries:
                            content_unit_ids = []
                            for i in range(start_idx, end_idx):
                                if not all_is_duplicate[i]:
                                    content_unit_ids.append(created_unit_ids[filtered_idx])
                                    filtered_idx += 1
                            result_unit_ids.append(content_unit_ids)

                        total_time = time.time() - start_time
                        log_buffer.append(f"{'='*60}")
                        log_buffer.append(f"PUT_BATCH_ASYNC COMPLETE: {len(created_unit_ids)} units from {len(contents)} contents in {total_time:.3f}s")
                        log_buffer.append(f"{'='*60}")

                        # Flush all logs at once to avoid interleaving
                        logger.info("\n" + "\n".join(log_buffer) + "\n")

                        # Trigger opinion reinforcement in background (non-blocking)
                        # Only trigger if there are entities in the new units
                        if any(filtered_entities):
                            await self._task_backend.submit_task({
                                'type': 'reinforce_opinion',
                                'agent_id': agent_id,
                                'created_unit_ids': created_unit_ids,
                                'unit_texts': filtered_sentences,
                                'unit_entities': filtered_entities
                            })
                            logger.debug("[PUT_BATCH_ASYNC] Opinion reinforcement task queued in background")

                        return result_unit_ids

                    except Exception as e:
                        # Transaction auto-rolls back on exception
                        import traceback
                        traceback.print_exc()
                        raise Exception(f"Failed to store batch memory: {str(e)}")

    def search(
        self,
        agent_id: str,
        query: str,
        fact_type: str,
        thinking_budget: int = 50,
        max_tokens: int = 4096,
        enable_trace: bool = False,
    ) -> tuple[List[Dict[str, Any]], Optional[Any]]:
        """
        Search memories using 4-way parallel retrieval (synchronous wrapper).

        This is a synchronous wrapper around search_async() for convenience.
        For best performance, use search_async() directly.

        Args:
            agent_id: Agent ID to search for
            query: Search query
            fact_type: Required filter for fact type ('world', 'agent', or 'opinion')
            thinking_budget: How many units to explore (computational budget)
            max_tokens: Maximum tokens to return (counts only 'text' field, default 4096)
            enable_trace: If True, returns detailed SearchTrace object

        Returns:
            Tuple of (results, trace)
        """
        # Run async version synchronously
        return asyncio.run(self.search_async(
            agent_id, query, fact_type, thinking_budget, max_tokens, enable_trace
        ))

    async def search_async(
        self,
        agent_id: str,
        query: str,
        fact_type: List[str],
        thinking_budget: int = 50,
        max_tokens: int = 4096,
        enable_trace: bool = False,
        question_date: Optional[datetime] = None,
    ) -> SearchResultModel:
        """
        Search memories using N*4-way parallel retrieval (N fact types × 4 retrieval methods).

        This implements the core SEARCH operation:
        1. Retrieval: For each fact type, run 4 parallel retrievals (semantic vector, BM25 keyword, graph activation, temporal graph)
        2. Merge: Combine using Reciprocal Rank Fusion (RRF)
        3. Rerank: Score using selected reranker (heuristic or cross-encoder)
        4. Diversify: Apply MMR for diversity
        5. Token Filter: Return results up to max_tokens budget

        Args:
            agent_id: Agent ID to search for
            query: Search query
            fact_type: List of fact types to search (e.g., ['world', 'agent'])
            thinking_budget: How many units to explore in graph traversal (controls compute cost)
            max_tokens: Maximum tokens to return (counts only 'text' field, default 4096)
                       Results are returned until token budget is reached, stopping before
                       including a fact that would exceed the limit
            enable_trace: Whether to return search trace for debugging (deprecated)
            question_date: Optional date when question was asked (for temporal filtering)

        Returns:
            SearchResultModel containing:
            - results: List of MemoryFact objects
            - trace: Optional trace information for debugging
        """
        # Backpressure: limit concurrent searches to prevent overwhelming the database
        async with self._search_semaphore:
            # Retry loop for connection errors
            max_retries = 3
            for attempt in range(max_retries + 1):
                try:
                    return await self._search_with_retries(
                        agent_id, query, fact_type, thinking_budget, max_tokens, enable_trace, question_date
                    )
                except Exception as e:
                    # Check if it's a connection error
                    is_connection_error = (
                        isinstance(e, asyncpg.TooManyConnectionsError) or
                        isinstance(e, asyncpg.CannotConnectNowError) or
                        (isinstance(e, asyncpg.PostgresError) and 'connection' in str(e).lower())
                    )

                    if is_connection_error and attempt < max_retries:
                        # Wait with exponential backoff before retry
                        wait_time = 0.5 * (2 ** attempt)  # 0.5s, 1s, 2s
                        logger.warning(
                            f"Connection error on search attempt {attempt + 1}/{max_retries + 1}: {str(e)}. "
                            f"Retrying in {wait_time:.1f}s..."
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        # Not a connection error or out of retries - raise
                        raise

    async def _search_with_retries(
        self,
        agent_id: str,
        query: str,
        fact_type: List[str],
        thinking_budget: int,
        max_tokens: int,
        enable_trace: bool,
        question_date: Optional[datetime] = None,
    ) -> tuple[List[Dict[str, Any]], Optional[Any]]:
        """
        Search implementation with modular retrieval and reranking.

        Architecture:
        1. Retrieval: 4-way parallel (semantic, keyword, graph, temporal graph)
        2. Merge: RRF to combine ranked lists
        3. Reranking: Pluggable strategy (heuristic or cross-encoder)
        4. Diversity: MMR with λ=0.5
        5. Token Filter: Limit results to max_tokens budget

        Args:
            agent_id: Agent identifier
            query: Search query
            fact_type: Type of facts to search
            thinking_budget: Nodes to explore in graph traversal
            max_tokens: Maximum tokens to return (counts only 'text' field)
            enable_trace: Whether to return search trace (deprecated)

        Returns:
            (results, trace) tuple where trace is None (tracing removed)
        """
        # Initialize tracer if requested
        from .search_tracer import SearchTracer
        tracer = SearchTracer(query, thinking_budget, max_tokens) if enable_trace else None
        if tracer:
            tracer.start()

        pool = await self._get_pool()
        search_start = time.time()

        # Buffer logs for clean output in concurrent scenarios
        search_id = f"{agent_id[:8]}-{int(time.time() * 1000) % 100000}"
        log_buffer = []
        log_buffer.append(f"[SEARCH {search_id}] Query: '{query[:50]}...' (budget={thinking_budget}, max_tokens={max_tokens})")

        try:
            # Step 1: Generate query embedding (for semantic search)
            step_start = time.time()
            query_embedding = embedding_utils.generate_embedding(self.embeddings, query)
            step_duration = time.time() - step_start
            log_buffer.append(f"  [1] Generate query embedding: {step_duration:.3f}s")

            if tracer:
                tracer.record_query_embedding(query_embedding)
                tracer.add_phase_metric("generate_query_embedding", step_duration)

            # Step 2: N*4-Way Parallel Retrieval (N fact types × 4 retrieval methods)
            step_start = time.time()
            query_embedding_str = str(query_embedding)

            from .search.retrieval import retrieve_parallel

            # Track each retrieval start time
            retrieval_start = time.time()

            # Run retrieval for each fact type in parallel
            retrieval_tasks = [
                retrieve_parallel(
                    pool, query, query_embedding_str, agent_id, ft, thinking_budget,
                    question_date, self.query_analyzer
                )
                for ft in fact_type
            ]
            all_retrievals = await asyncio.gather(*retrieval_tasks)

            # Combine all results from all fact types
            semantic_results = []
            bm25_results = []
            graph_results = []
            temporal_results = []

            for ft_semantic, ft_bm25, ft_graph, ft_temporal in all_retrievals:
                semantic_results.extend(ft_semantic)
                bm25_results.extend(ft_bm25)
                graph_results.extend(ft_graph)
                if ft_temporal:
                    temporal_results.extend(ft_temporal)

            # If no temporal results from any fact type, set to None
            if not temporal_results:
                temporal_results = None

            retrieval_duration = time.time() - retrieval_start

            step_duration = time.time() - step_start
            total_retrievals = len(fact_type) * (4 if temporal_results else 3)
            if temporal_results:
                log_buffer.append(f"  [2] {total_retrievals}-way retrieval ({len(fact_type)} fact_types): semantic={len(semantic_results)}, bm25={len(bm25_results)}, graph={len(graph_results)}, temporal={len(temporal_results)} in {step_duration:.3f}s")
            else:
                log_buffer.append(f"  [2] {total_retrievals}-way retrieval ({len(fact_type)} fact_types): semantic={len(semantic_results)}, bm25={len(bm25_results)}, graph={len(graph_results)} in {step_duration:.3f}s")

            # Record retrieval results for tracer
            if tracer:
                # Estimate duration for each method (since they run in parallel)
                estimated_duration = retrieval_duration

                # Add semantic retrieval results
                tracer.add_retrieval_results(
                    method_name="semantic",
                    results=semantic_results,
                    duration_seconds=estimated_duration,
                    score_field="similarity",
                    metadata={"limit": thinking_budget}
                )

                # Add BM25 retrieval results
                tracer.add_retrieval_results(
                    method_name="bm25",
                    results=bm25_results,
                    duration_seconds=estimated_duration,
                    score_field="bm25_score",
                    metadata={"limit": thinking_budget}
                )

                # Add graph retrieval results
                tracer.add_retrieval_results(
                    method_name="graph",
                    results=graph_results,
                    duration_seconds=estimated_duration,
                    score_field="similarity",  # Graph uses similarity for activation
                    metadata={"budget": thinking_budget}
                )

                # Add temporal retrieval results if present
                if temporal_results:
                    tracer.add_retrieval_results(
                        method_name="temporal",
                        results=temporal_results,
                        duration_seconds=estimated_duration,
                        score_field="temporal_score",
                        metadata={"budget": thinking_budget}
                    )

                # Record entry points (from semantic results) for legacy graph view
                for rank, (doc_id, data) in enumerate(semantic_results[:10], start=1):  # Top 10 as entry points
                    similarity = data.get("similarity", 0.0)
                    tracer.add_entry_point(doc_id, data.get("text", ""), similarity, rank)

                tracer.add_phase_metric("parallel_retrieval", step_duration, {
                    "semantic_count": len(semantic_results),
                    "bm25_count": len(bm25_results),
                    "graph_count": len(graph_results),
                    "temporal_count": len(temporal_results) if temporal_results else 0
                })

            # Step 3: Merge with RRF
            step_start = time.time()
            from .search_helpers import reciprocal_rank_fusion

            # Merge 3 or 4 result lists depending on temporal constraint
            if temporal_results:
                merged_candidates = reciprocal_rank_fusion([semantic_results, bm25_results, graph_results, temporal_results])
            else:
                merged_candidates = reciprocal_rank_fusion([semantic_results, bm25_results, graph_results])

            step_duration = time.time() - step_start
            log_buffer.append(f"  [3] RRF merge: {len(merged_candidates)} unique candidates in {step_duration:.3f}s")

            if tracer:
                tracer.add_rrf_merged(merged_candidates)
                tracer.add_phase_metric("rrf_merge", step_duration, {"candidates_merged": len(merged_candidates)})

            # Step 4: Build candidate objects for reranking
            step_start = time.time()

            # Build result objects with all necessary data
            results = []
            for doc_id, data, rrf_meta in merged_candidates:
                # Extract scores from different sources
                semantic_sim = data.get("similarity", 0.0)
                bm25_score = data.get("bm25_score", 0.0)

                # Convert embedding from string to list if needed
                embedding = data.get("embedding")
                if embedding is not None:
                    if isinstance(embedding, str):
                        import json
                        embedding = json.loads(embedding)
                    elif not isinstance(embedding, (list, np.ndarray)):
                        embedding = list(embedding)

                result_obj = {
                    "id": doc_id,
                    "text": data["text"],
                    "context": data.get("context", ""),
                    "event_date": data["event_date"],  # Keep as datetime for now
                    "fact_type": data.get("fact_type"),  # Include fact type for filtering
                    "access_count": data.get("access_count", 0),
                    "semantic_similarity": semantic_sim,
                    "bm25_score": bm25_score,
                    "embedding": embedding,
                    "rrf_score": rrf_meta.get("rrf_score", 0.0),
                    **rrf_meta  # Include all RRF metadata
                }

                # Include temporal scores if present
                if "temporal_score" in data:
                    result_obj["temporal_score"] = data["temporal_score"]
                if "temporal_proximity" in data:
                    result_obj["temporal_proximity"] = data["temporal_proximity"]

                results.append(result_obj)

            # Step 5: Rerank using cross-encoder
            reranker_instance = self._cross_encoder_reranker
            log_buffer.append(f"  [4] Using cross-encoder reranker")

            # Rerank using cross-encoder
            results = reranker_instance.rerank(query, results)

            step_duration = time.time() - step_start
            log_buffer.append(f"  [4] Reranking: {len(results)} candidates scored in {step_duration:.3f}s")

            if tracer:
                tracer.add_reranked(results, merged_candidates)
                tracer.add_phase_metric("reranking", step_duration, {
                    "reranker_type": "cross-encoder",
                    "candidates_reranked": len(results)
                })

            # Step 5: Truncate to thinking_budget * 2 for token filtering
            rerank_limit = thinking_budget * 2
            top_results = results[:rerank_limit]
            log_buffer.append(f"  [5] Truncated to top {len(top_results)} results")

            # Step 6: Token budget filtering
            step_start = time.time()

            # Filter results to fit within max_tokens budget
            # Token counting using tiktoken (cached at module level)
            filtered_results, total_tokens = self._filter_by_token_budget(top_results, max_tokens)

            top_results = filtered_results
            step_duration = time.time() - step_start
            log_buffer.append(f"  [6] Token filtering: {len(top_results)} results, {total_tokens}/{max_tokens} tokens in {step_duration:.3f}s")

            if tracer:
                tracer.add_phase_metric("token_filtering", step_duration, {
                    "results_selected": len(top_results),
                    "tokens_used": total_tokens,
                    "max_tokens": max_tokens
                })

            # Record visits for all retrieved nodes
            if tracer:
                for result in results:
                    tracer.visit_node(
                        node_id=result["id"],
                        text=result["text"],
                        context=result.get("context", ""),
                        event_date=result["event_date"],
                        access_count=result.get("access_count", 0),
                        is_entry_point=(result["id"] in [ep.node_id for ep in tracer.entry_points]),
                        parent_node_id=None,  # In parallel retrieval, there's no clear parent
                        link_type=None,
                        link_weight=None,
                        activation=result.get("rrf_score", 0.0),  # Use RRF score as activation
                        semantic_similarity=result.get("semantic_similarity", 0.0),
                        recency=result.get("recency_normalized", 0.0),
                        frequency=result.get("frequency_normalized", 0.0),
                        final_weight=result.get("weight", 0.0)
                    )

            # Step 8: Queue access count updates for visited nodes
            visited_ids = list(set([r["id"] for r in results[:50]]))  # Top 50
            if visited_ids:
                await self._task_backend.submit_task({
                    'type': 'access_count_update',
                    'node_ids': visited_ids
                })
                log_buffer.append(f"  [7] Queued access count updates for {len(visited_ids)} nodes")

            total_time = time.time() - search_start
            log_buffer.append(f"[SEARCH {search_id}] Complete: {len(top_results)} results ({total_tokens} tokens) in {total_time:.3f}s")

            # Log all buffered logs at once
            logger.info("\n" + "\n".join(log_buffer))

            # Convert datetime objects to ISO strings for JSON serialization
            for result in top_results:
                if result.get("event_date"):
                    event_date = result["event_date"]
                    result["event_date"] = event_date.isoformat() if hasattr(event_date, 'isoformat') else event_date
                if result.get("occurred_start"):
                    occurred_start = result["occurred_start"]
                    result["occurred_start"] = occurred_start.isoformat() if hasattr(occurred_start, 'isoformat') else occurred_start
                if result.get("occurred_end"):
                    occurred_end = result["occurred_end"]
                    result["occurred_end"] = occurred_end.isoformat() if hasattr(occurred_end, 'isoformat') else occurred_end
                if result.get("mentioned_at"):
                    mentioned_at = result["mentioned_at"]
                    result["mentioned_at"] = mentioned_at.isoformat() if hasattr(mentioned_at, 'isoformat') else mentioned_at

            # Convert results to MemoryFact objects
            memory_facts = []
            for result in top_results:
                memory_facts.append(MemoryFact(
                    id=str(result.get("id")),
                    text=result.get("text"),
                    fact_type=result.get("fact_type", "world"),
                    context=result.get("context"),
                    event_date=result.get("event_date"),
                    occurred_start=result.get("occurred_start"),
                    occurred_end=result.get("occurred_end"),
                    mentioned_at=result.get("mentioned_at"),
                    activation=result.get("activation")
                ))

            # Finalize trace if enabled
            trace_dict = None
            if tracer:
                trace = tracer.finalize(top_results)
                trace_dict = trace.to_dict() if trace else None

            return SearchResultModel(results=memory_facts, trace=trace_dict)

        except Exception as e:
            log_buffer.append(f"[SEARCH {search_id}] ERROR after {time.time() - search_start:.3f}s: {str(e)}")
            logger.error("\n" + "\n".join(log_buffer))
            raise Exception(f"Failed to search memories: {str(e)}")

    def _filter_by_token_budget(
        self,
        results: List[Dict[str, Any]],
        max_tokens: int
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Filter results to fit within token budget.

        Counts tokens only for the 'text' field using tiktoken (cl100k_base encoding).
        Stops before including a fact that would exceed the budget.

        Args:
            results: List of search results
            max_tokens: Maximum tokens allowed

        Returns:
            Tuple of (filtered_results, total_tokens_used)
        """
        encoding = _get_tiktoken_encoding()

        filtered_results = []
        total_tokens = 0

        for result in results:
            text = result.get("text", "")
            text_tokens = len(encoding.encode(text))

            # Check if adding this result would exceed budget
            if total_tokens + text_tokens <= max_tokens:
                filtered_results.append(result)
                total_tokens += text_tokens
            else:
                # Stop before including a fact that would exceed limit
                break

        return filtered_results, total_tokens

    async def get_document(self, document_id: str, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve document metadata and statistics.

        Args:
            document_id: Document ID to retrieve
            agent_id: Agent ID that owns the document

        Returns:
            Dictionary with document info or None if not found
        """
        pool = await self._get_pool()
        async with acquire_with_retry(pool) as conn:
            doc = await conn.fetchrow(
                """
                SELECT d.id, d.agent_id, d.original_text, d.content_hash,
                       d.created_at, d.updated_at, COUNT(mu.id) as unit_count
                FROM documents d
                LEFT JOIN memory_units mu ON mu.document_id = d.id
                WHERE d.id = $1 AND d.agent_id = $2
                GROUP BY d.id, d.agent_id, d.original_text, d.content_hash, d.created_at, d.updated_at
                """,
                document_id, agent_id
            )

            if not doc:
                return None

            return {
                "id": doc["id"],
                "agent_id": doc["agent_id"],
                "original_text": doc["original_text"],
                "content_hash": doc["content_hash"],
                "unit_count": doc["unit_count"],
                "created_at": doc["created_at"],
                "updated_at": doc["updated_at"]
            }

    async def delete_document(self, document_id: str, agent_id: str) -> Dict[str, int]:
        """
        Delete a document and all its associated memory units and links.

        Args:
            document_id: Document ID to delete
            agent_id: Agent ID that owns the document

        Returns:
            Dictionary with counts of deleted items
        """
        pool = await self._get_pool()
        async with acquire_with_retry(pool) as conn:
            async with conn.transaction():
                # Count units before deletion
                units_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM memory_units WHERE document_id = $1",
                    document_id
                )

                # Delete document (cascades to memory_units and all their links)
                deleted = await conn.fetchval(
                    "DELETE FROM documents WHERE id = $1 AND agent_id = $2 RETURNING id",
                    document_id, agent_id
                )

                return {
                    "document_deleted": 1 if deleted else 0,
                    "memory_units_deleted": units_count if deleted else 0
                }

    async def delete_memory_unit(self, unit_id: str) -> Dict[str, Any]:
        """
        Delete a single memory unit and all its associated links.

        Due to CASCADE DELETE constraints, this will automatically delete:
        - All links from this unit (memory_links where from_unit_id = unit_id)
        - All links to this unit (memory_links where to_unit_id = unit_id)
        - All entity associations (unit_entities where unit_id = unit_id)

        Args:
            unit_id: UUID of the memory unit to delete

        Returns:
            Dictionary with deletion result
        """
        pool = await self._get_pool()
        async with acquire_with_retry(pool) as conn:
            async with conn.transaction():
                # Delete the memory unit (cascades to links and associations)
                deleted = await conn.fetchval(
                    "DELETE FROM memory_units WHERE id = $1 RETURNING id",
                    unit_id
                )

                return {
                    "success": deleted is not None,
                    "unit_id": str(deleted) if deleted else None,
                    "message": "Memory unit and all its links deleted successfully" if deleted else "Memory unit not found"
                }

    async def delete_agent(self, agent_id: str, fact_type: Optional[str] = None) -> Dict[str, int]:
        """
        Delete all data for a specific agent (multi-tenant cleanup).

        This is much more efficient than dropping all tables and allows
        multiple agents to coexist in the same database.

        Deletes (with CASCADE):
        - All memory units for this agent (optionally filtered by fact_type)
        - All entities for this agent (if deleting all memory units)
        - All associated links, unit-entity associations, and co-occurrences

        Args:
            agent_id: Agent ID to delete
            fact_type: Optional fact type filter (world, agent, opinion). If provided, only deletes memories of that type.

        Returns:
            Dictionary with counts of deleted items
        """
        pool = await self._get_pool()
        async with acquire_with_retry(pool) as conn:
            async with conn.transaction():
                try:
                    if fact_type:
                        # Delete only memories of a specific fact type
                        units_count = await conn.fetchval(
                            "SELECT COUNT(*) FROM memory_units WHERE agent_id = $1 AND fact_type = $2",
                            agent_id, fact_type
                        )
                        await conn.execute(
                            "DELETE FROM memory_units WHERE agent_id = $1 AND fact_type = $2",
                            agent_id, fact_type
                        )

                        # Note: We don't delete entities when fact_type is specified,
                        # as they may be referenced by other memory units
                        return {
                            "memory_units_deleted": units_count,
                            "entities_deleted": 0
                        }
                    else:
                        # Delete all data for the agent
                        units_count = await conn.fetchval("SELECT COUNT(*) FROM memory_units WHERE agent_id = $1", agent_id)
                        entities_count = await conn.fetchval("SELECT COUNT(*) FROM entities WHERE agent_id = $1", agent_id)

                        # Delete memory units (cascades to unit_entities, memory_links)
                        await conn.execute("DELETE FROM memory_units WHERE agent_id = $1", agent_id)

                        # Delete entities (cascades to unit_entities, entity_cooccurrences, memory_links with entity_id)
                        await conn.execute("DELETE FROM entities WHERE agent_id = $1", agent_id)

                        return {
                            "memory_units_deleted": units_count,
                            "entities_deleted": entities_count
                        }

                except Exception as e:
                    raise Exception(f"Failed to delete agent data: {str(e)}")

    async def get_graph_data(self, agent_id: Optional[str] = None, fact_type: Optional[str] = None):
        """
        Get graph data for visualization.

        Args:
            agent_id: Filter by agent ID
            fact_type: Filter by fact type (world, agent, opinion)

        Returns:
            Dict with nodes, edges, and table_rows
        """
        pool = await self._get_pool()
        async with acquire_with_retry(pool) as conn:
            # Get memory units, optionally filtered by agent_id and fact_type
            query_conditions = []
            query_params = []
            param_count = 0

            if agent_id:
                param_count += 1
                query_conditions.append(f"agent_id = ${param_count}")
                query_params.append(agent_id)

            if fact_type:
                param_count += 1
                query_conditions.append(f"fact_type = ${param_count}")
                query_params.append(fact_type)

            where_clause = "WHERE " + " AND ".join(query_conditions) if query_conditions else ""

            units = await conn.fetch(f"""
                SELECT id, text, event_date, context
                FROM memory_units
                {where_clause}
                ORDER BY event_date DESC
                LIMIT 1000
            """, *query_params)

            # Get links, filtering to only include links between units of the selected agent
            unit_ids = [row['id'] for row in units]
            if unit_ids:
                links = await conn.fetch("""
                    SELECT
                        ml.from_unit_id,
                        ml.to_unit_id,
                        ml.link_type,
                        ml.weight,
                        e.canonical_name as entity_name
                    FROM memory_links ml
                    LEFT JOIN entities e ON ml.entity_id = e.id
                    WHERE ml.from_unit_id = ANY($1::uuid[]) AND ml.to_unit_id = ANY($1::uuid[])
                    ORDER BY ml.link_type, ml.weight DESC
                """, unit_ids)
            else:
                links = []

            # Get entity information
            unit_entities = await conn.fetch("""
                SELECT ue.unit_id, e.canonical_name
                FROM unit_entities ue
                JOIN entities e ON ue.entity_id = e.id
                ORDER BY ue.unit_id
            """)

        # Build entity mapping
        entity_map = {}
        for row in unit_entities:
            unit_id = row['unit_id']
            entity_name = row['canonical_name']
            if unit_id not in entity_map:
                entity_map[unit_id] = []
            entity_map[unit_id].append(entity_name)

        # Build nodes
        nodes = []
        for row in units:
            unit_id = row['id']
            text = row['text']
            event_date = row['event_date']
            context = row['context']

            entities = entity_map.get(unit_id, [])
            entity_count = len(entities)

            # Color by entity count
            if entity_count == 0:
                color = "#e0e0e0"
            elif entity_count == 1:
                color = "#90caf9"
            else:
                color = "#42a5f5"

            nodes.append({
                "data": {
                    "id": str(unit_id),
                    "label": f"{text[:30]}..." if len(text) > 30 else text,
                    "text": text,
                    "date": event_date.isoformat() if event_date else "",
                    "context": context if context else "",
                    "entities": ", ".join(entities) if entities else "None",
                    "color": color
                }
            })

        # Build edges
        edges = []
        for row in links:
            from_id = str(row['from_unit_id'])
            to_id = str(row['to_unit_id'])
            link_type = row['link_type']
            weight = row['weight']
            entity_name = row['entity_name']

            # Color by link type
            if link_type == 'temporal':
                color = "#00bcd4"
                line_style = "dashed"
            elif link_type == 'semantic':
                color = "#ff69b4"
                line_style = "solid"
            elif link_type == 'entity':
                color = "#ffd700"
                line_style = "solid"
            else:
                color = "#999999"
                line_style = "solid"

            edges.append({
                "data": {
                    "id": f"{from_id}-{to_id}-{link_type}",
                    "source": from_id,
                    "target": to_id,
                    "linkType": link_type,
                    "weight": weight,
                    "entityName": entity_name if entity_name else "",
                    "color": color,
                    "lineStyle": line_style
                }
            })

        # Build table rows
        table_rows = []
        for row in units:
            unit_id = row['id']
            entities = entity_map.get(unit_id, [])

            table_rows.append({
                "id": str(unit_id)[:8] + "...",
                "text": row['text'],
                "context": row['context'] if row['context'] else "N/A",
                "date": row['event_date'].strftime("%Y-%m-%d %H:%M") if row['event_date'] else "N/A",
                "entities": ", ".join(entities) if entities else "None"
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "table_rows": table_rows,
            "total_units": len(units)
        }

    async def list_memory_units(
        self,
        agent_id: Optional[str] = None,
        fact_type: Optional[str] = None,
        search_query: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ):
        """
        List memory units for table view with optional full-text search.

        Args:
            agent_id: Filter by agent ID
            fact_type: Filter by fact type (world, agent, opinion)
            search_query: Full-text search query (searches text and context fields)
            limit: Maximum number of results to return
            offset: Offset for pagination

        Returns:
            Dict with items (list of memory units) and total count
        """
        pool = await self._get_pool()
        async with acquire_with_retry(pool) as conn:
            # Build query conditions
            query_conditions = []
            query_params = []
            param_count = 0

            if agent_id:
                param_count += 1
                query_conditions.append(f"agent_id = ${param_count}")
                query_params.append(agent_id)

            if fact_type:
                param_count += 1
                query_conditions.append(f"fact_type = ${param_count}")
                query_params.append(fact_type)

            if search_query:
                # Full-text search on text and context fields using ILIKE
                param_count += 1
                query_conditions.append(f"(text ILIKE ${param_count} OR context ILIKE ${param_count})")
                query_params.append(f"%{search_query}%")

            where_clause = "WHERE " + " AND ".join(query_conditions) if query_conditions else ""

            # Get total count
            count_query = f"""
                SELECT COUNT(*) as total
                FROM memory_units
                {where_clause}
            """
            count_result = await conn.fetchrow(count_query, *query_params)
            total = count_result['total']

            # Get units with limit and offset
            param_count += 1
            limit_param = f"${param_count}"
            query_params.append(limit)

            param_count += 1
            offset_param = f"${param_count}"
            query_params.append(offset)

            units = await conn.fetch(f"""
                SELECT id, text, event_date, context, fact_type
                FROM memory_units
                {where_clause}
                ORDER BY event_date DESC
                LIMIT {limit_param} OFFSET {offset_param}
            """, *query_params)

            # Get entity information for these units
            if units:
                unit_ids = [row['id'] for row in units]
                unit_entities = await conn.fetch("""
                    SELECT ue.unit_id, e.canonical_name
                    FROM unit_entities ue
                    JOIN entities e ON ue.entity_id = e.id
                    WHERE ue.unit_id = ANY($1::uuid[])
                    ORDER BY ue.unit_id
                """, unit_ids)
            else:
                unit_entities = []

            # Build entity mapping
            entity_map = {}
            for row in unit_entities:
                unit_id = row['unit_id']
                entity_name = row['canonical_name']
                if unit_id not in entity_map:
                    entity_map[unit_id] = []
                entity_map[unit_id].append(entity_name)

            # Build result items
            items = []
            for row in units:
                unit_id = row['id']
                entities = entity_map.get(unit_id, [])

                items.append({
                    "id": str(unit_id),
                    "text": row['text'],
                    "context": row['context'] if row['context'] else "",
                    "date": row['event_date'].isoformat() if row['event_date'] else "",
                    "fact_type": row['fact_type'],
                    "entities": ", ".join(entities) if entities else ""
                })

            return {
                "items": items,
                "total": total,
                "limit": limit,
                "offset": offset
            }

    async def list_documents(
        self,
        agent_id: str,
        search_query: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ):
        """
        List documents with optional search and pagination.

        Args:
            agent_id: Agent ID (required)
            search_query: Search in document ID
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            Dict with items (list of documents without original_text) and total count
        """
        pool = await self._get_pool()
        async with acquire_with_retry(pool) as conn:
            # Build query conditions
            query_conditions = []
            query_params = []
            param_count = 0

            param_count += 1
            query_conditions.append(f"agent_id = ${param_count}")
            query_params.append(agent_id)

            if search_query:
                # Search in document ID
                param_count += 1
                query_conditions.append(f"id ILIKE ${param_count}")
                query_params.append(f"%{search_query}%")

            where_clause = "WHERE " + " AND ".join(query_conditions) if query_conditions else ""

            # Get total count
            count_query = f"""
                SELECT COUNT(*) as total
                FROM documents
                {where_clause}
            """
            count_result = await conn.fetchrow(count_query, *query_params)
            total = count_result['total']

            # Get documents with limit and offset (without original_text for performance)
            param_count += 1
            limit_param = f"${param_count}"
            query_params.append(limit)

            param_count += 1
            offset_param = f"${param_count}"
            query_params.append(offset)

            documents = await conn.fetch(f"""
                SELECT
                    id,
                    agent_id,
                    content_hash,
                    created_at,
                    updated_at,
                    LENGTH(original_text) as text_length
                FROM documents
                {where_clause}
                ORDER BY created_at DESC
                LIMIT {limit_param} OFFSET {offset_param}
            """, *query_params)

            # Get memory unit count for each document
            if documents:
                doc_ids = [(row['id'], row['agent_id']) for row in documents]

                # Create placeholders for the query
                placeholders = []
                params_for_count = []
                for i, (doc_id, agent_id_val) in enumerate(doc_ids):
                    idx_doc = i * 2 + 1
                    idx_agent = i * 2 + 2
                    placeholders.append(f"(document_id = ${idx_doc} AND agent_id = ${idx_agent})")
                    params_for_count.extend([doc_id, agent_id_val])

                where_clause_count = " OR ".join(placeholders)

                unit_counts = await conn.fetch(f"""
                    SELECT document_id, agent_id, COUNT(*) as unit_count
                    FROM memory_units
                    WHERE {where_clause_count}
                    GROUP BY document_id, agent_id
                """, *params_for_count)
            else:
                unit_counts = []

            # Build count mapping
            count_map = {(row['document_id'], row['agent_id']): row['unit_count'] for row in unit_counts}

            # Build result items
            items = []
            for row in documents:
                doc_id = row['id']
                agent_id_val = row['agent_id']
                unit_count = count_map.get((doc_id, agent_id_val), 0)

                items.append({
                    "id": doc_id,
                    "agent_id": agent_id_val,
                    "content_hash": row['content_hash'],
                    "created_at": row['created_at'].isoformat() if row['created_at'] else "",
                    "updated_at": row['updated_at'].isoformat() if row['updated_at'] else "",
                    "text_length": row['text_length'] or 0,
                    "memory_unit_count": unit_count
                })

            return {
                "items": items,
                "total": total,
                "limit": limit,
                "offset": offset
            }

    async def get_document(
        self,
        document_id: str,
        agent_id: str
    ):
        """
        Get a specific document including its original_text.

        Args:
            document_id: Document ID
            agent_id: Agent ID

        Returns:
            Dict with document details including original_text, or None if not found
        """
        pool = await self._get_pool()
        async with acquire_with_retry(pool) as conn:
            doc = await conn.fetchrow("""
                SELECT
                    id,
                    agent_id,
                    original_text,
                    content_hash,
                    created_at,
                    updated_at
                FROM documents
                WHERE id = $1 AND agent_id = $2
            """, document_id, agent_id)

            if not doc:
                return None

            # Get memory unit count
            unit_count_row = await conn.fetchrow("""
                SELECT COUNT(*) as unit_count
                FROM memory_units
                WHERE document_id = $1 AND agent_id = $2
            """, document_id, agent_id)

            return {
                "id": doc['id'],
                "agent_id": doc['agent_id'],
                "original_text": doc['original_text'],
                "content_hash": doc['content_hash'],
                "created_at": doc['created_at'].isoformat() if doc['created_at'] else "",
                "updated_at": doc['updated_at'].isoformat() if doc['updated_at'] else "",
                "memory_unit_count": unit_count_row['unit_count'] if unit_count_row else 0
            }

    async def _evaluate_opinion_update_async(
        self,
        opinion_text: str,
        opinion_confidence: float,
        new_event_text: str,
        entity_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate if an opinion should be updated based on a new event.

        Args:
            opinion_text: Current opinion text (includes reasons)
            opinion_confidence: Current confidence score (0.0-1.0)
            new_event_text: Text of the new event
            entity_name: Name of the entity this opinion is about

        Returns:
            Dict with 'action' ('keep'|'update'), 'new_confidence', 'new_text' (if action=='update')
            or None if no changes needed
        """
        from pydantic import BaseModel, Field

        class OpinionEvaluation(BaseModel):
            """Evaluation of whether an opinion should be updated."""
            action: str = Field(description="Action to take: 'keep' (no change) or 'update' (modify opinion)")
            reasoning: str = Field(description="Brief explanation of why this action was chosen")
            new_confidence: float = Field(description="New confidence score (0.0-1.0). Can be higher, lower, or same as before.")
            new_opinion_text: Optional[str] = Field(
                default=None,
                description="If action is 'update', the revised opinion text that acknowledges the previous view. Otherwise None."
            )

        evaluation_prompt = f"""You are evaluating whether an existing opinion should be updated based on new information.

ENTITY: {entity_name}

EXISTING OPINION:
{opinion_text}
Current confidence: {opinion_confidence:.2f}

NEW EVENT:
{new_event_text}

Evaluate whether this new event:
1. REINFORCES the opinion (increase confidence, keep text)
2. WEAKENS the opinion (decrease confidence, keep text)
3. CHANGES the opinion (update both text and confidence, noting "Previously I thought X, but now Y...")
4. IRRELEVANT (keep everything as is)

Guidelines:
- Only suggest 'update' action if the new event genuinely contradicts or significantly modifies the opinion
- If updating the text, acknowledge the previous opinion and explain the change
- Confidence should reflect accumulated evidence (0.0 = no confidence, 1.0 = very confident)
- Small changes in confidence are normal; large jumps should be rare"""

        try:
            result = await self._llm_config.call(
                messages=[
                    {"role": "system", "content": "You evaluate and update opinions based on new information."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                response_format=OpinionEvaluation,
                scope="memory_evaluate_opinion",
                temperature=0.3  # Lower temperature for more consistent evaluation
            )

            # Only return updates if something actually changed
            if result.action == 'keep' and abs(result.new_confidence - opinion_confidence) < 0.01:
                return None

            return {
                'action': result.action,
                'reasoning': result.reasoning,
                'new_confidence': result.new_confidence,
                'new_text': result.new_opinion_text if result.action == 'update' else None
            }

        except Exception as e:
            logger.warning(f"Failed to evaluate opinion update: {str(e)}")
            return None

    async def _handle_form_opinion(self, task_dict: Dict[str, Any]):
        """
        Handler for form opinion tasks.

        Args:
            task_dict: Dict with keys: 'agent_id', 'answer_text', 'query'
        """
        agent_id = task_dict['agent_id']
        answer_text = task_dict['answer_text']
        query = task_dict['query']

        logger.debug(f"[TASK] Handling form_opinion task for agent {agent_id}")
        await self._extract_and_store_opinions_async(
            agent_id=agent_id,
            answer_text=answer_text,
            query=query
        )

    async def _handle_reinforce_opinion(self, task_dict: Dict[str, Any]):
        """
        Handler for reinforce opinion tasks.

        Args:
            task_dict: Dict with keys: 'agent_id', 'created_unit_ids', 'unit_texts', 'unit_entities'
        """
        agent_id = task_dict['agent_id']
        created_unit_ids = task_dict['created_unit_ids']
        unit_texts = task_dict['unit_texts']
        unit_entities = task_dict['unit_entities']

        await self._reinforce_opinions_async(
            agent_id=agent_id,
            created_unit_ids=created_unit_ids,
            unit_texts=unit_texts,
            unit_entities=unit_entities
        )

    async def _reinforce_opinions_async(
        self,
        agent_id: str,
        created_unit_ids: List[str],
        unit_texts: List[str],
        unit_entities: List[List[Dict[str, str]]],
    ):
        """
        Background task to reinforce opinions based on newly ingested events.

        This runs asynchronously and does not block the put operation.

        Args:
            agent_id: Agent ID
            created_unit_ids: List of newly created memory unit IDs
            unit_texts: Texts of the newly created units
            unit_entities: Entities extracted from each unit
        """
        try:
            # Extract all unique entity names from the new units
            entity_names = set()
            for entities_list in unit_entities:
                for entity in entities_list:
                    entity_names.add(entity['text'])

            if not entity_names:
                logger.debug("[REINFORCE] No entities found in new units, skipping opinion reinforcement")
                return

            logger.debug(f"[REINFORCE] Starting opinion reinforcement for {len(entity_names)} entities")

            pool = await self._get_pool()
            async with acquire_with_retry(pool) as conn:
                # Find all opinions related to these entities
                opinions = await conn.fetch(
                    """
                    SELECT DISTINCT mu.id, mu.text, mu.confidence_score, e.canonical_name
                    FROM memory_units mu
                    JOIN unit_entities ue ON mu.id = ue.unit_id
                    JOIN entities e ON ue.entity_id = e.id
                    WHERE mu.agent_id = $1
                      AND mu.fact_type = 'opinion'
                      AND e.canonical_name = ANY($2::text[])
                    """,
                    agent_id,
                    list(entity_names)
                )

                if not opinions:
                    logger.debug("[REINFORCE] No existing opinions found for these entities")
                    return

                logger.debug(f"[REINFORCE] Found {len(opinions)} opinions to potentially reinforce")

                # Use cached LLM config
                if self._llm_config is None:
                    logger.error("[REINFORCE] LLM config not available, skipping opinion reinforcement")
                    return

                # Evaluate each opinion against the new events
                updates_to_apply = []
                for opinion in opinions:
                    opinion_id = str(opinion['id'])
                    opinion_text = opinion['text']
                    opinion_confidence = opinion['confidence_score']
                    entity_name = opinion['canonical_name']

                    # Find all new events mentioning this entity
                    relevant_events = []
                    for unit_text, entities_list in zip(unit_texts, unit_entities):
                        if any(e['text'] == entity_name for e in entities_list):
                            relevant_events.append(unit_text)

                    if not relevant_events:
                        continue

                    # Combine all relevant events
                    combined_events = "\n".join(relevant_events)

                    # Evaluate if opinion should be updated
                    evaluation = await self._evaluate_opinion_update_async(
                        opinion_text,
                        opinion_confidence,
                        combined_events,
                        entity_name
                    )

                    if evaluation:
                        updates_to_apply.append({
                            'opinion_id': opinion_id,
                            'evaluation': evaluation
                        })

                # Apply all updates in a single transaction
                if updates_to_apply:
                    async with conn.transaction():
                        for update in updates_to_apply:
                            opinion_id = update['opinion_id']
                            evaluation = update['evaluation']

                            if evaluation['action'] == 'update' and evaluation['new_text']:
                                # Update both text and confidence
                                await conn.execute(
                                    """
                                    UPDATE memory_units
                                    SET text = $1, confidence_score = $2, updated_at = NOW()
                                    WHERE id = $3
                                    """,
                                    evaluation['new_text'],
                                    evaluation['new_confidence'],
                                    uuid.UUID(opinion_id)
                                )
                                logger.debug(f"[REINFORCE] Updated opinion {opinion_id[:8]}... (action: {evaluation['action']}, confidence: {evaluation['new_confidence']:.2f})")
                            else:
                                # Only update confidence
                                await conn.execute(
                                    """
                                    UPDATE memory_units
                                    SET confidence_score = $1, updated_at = NOW()
                                    WHERE id = $2
                                    """,
                                    evaluation['new_confidence'],
                                    uuid.UUID(opinion_id)
                                )
                                logger.debug(f"[REINFORCE] Updated confidence for opinion {opinion_id[:8]}... (confidence: {evaluation['new_confidence']:.2f})")

                    logger.debug(f"[REINFORCE] Applied {len(updates_to_apply)} opinion updates")
                else:
                    logger.debug("[REINFORCE] No opinion updates needed")

        except Exception as e:
            logger.error(f"[REINFORCE] Error during opinion reinforcement: {str(e)}")
            import traceback
            traceback.print_exc()

    # ==================== Agent Profile Methods ====================

    async def get_agent_profile(self, agent_id: str) -> Dict:
        """
        Get agent profile (name, personality + background).
        Auto-creates agent with default values if not exists.

        Args:
            agent_id: Agent identifier

        Returns:
            Dict with 'name' (str), 'personality' (dict) and 'background' (str) keys
        """
        pool = await self._get_pool()
        return await agent_utils.get_agent_profile(pool, agent_id)

    async def update_agent_personality(
        self,
        agent_id: str,
        personality: Dict[str, float]
    ) -> None:
        """
        Update agent personality traits.

        Args:
            agent_id: Agent identifier
            personality: Dict with Big Five traits + bias_strength (all 0-1)
        """
        pool = await self._get_pool()
        await agent_utils.update_agent_personality(pool, agent_id, personality)

    async def merge_agent_background(
        self,
        agent_id: str,
        new_info: str,
        update_personality: bool = True
    ) -> dict:
        """
        Merge new background information with existing background using LLM.
        Normalizes to first person ("I") and resolves conflicts.
        Optionally infers personality traits from the merged background.

        Args:
            agent_id: Agent identifier
            new_info: New background information to add/merge
            update_personality: If True, infer Big Five traits from background (default: True)

        Returns:
            Dict with 'background' (str) and optionally 'personality' (dict) keys
        """
        pool = await self._get_pool()
        return await agent_utils.merge_agent_background(
            pool, self._llm_config, agent_id, new_info, update_personality
        )

    async def list_agents(self) -> list:
        """
        List all agents in the system.

        Returns:
            List of dicts with agent_id, name, personality, background, created_at, updated_at
        """
        pool = await self._get_pool()
        return await agent_utils.list_agents(pool)

    # ==================== Think Methods ====================

    async def think_async(
        self,
        agent_id: str,
        query: str,
        thinking_budget: int = 50,
        context: str = None,
    ) -> ThinkResult:
        """
        Think and formulate an answer using agent identity, world facts, and opinions.

        This method:
        1. Retrieves agent facts (agent's identity and past actions)
        2. Retrieves world facts (general knowledge)
        3. Retrieves existing opinions (agent's formed perspectives)
        4. Uses LLM to formulate an answer
        5. Extracts and stores any new opinions formed during thinking
        6. Returns plain text answer and the facts used

        Args:
            agent_id: Agent identifier
            query: Question to answer
            thinking_budget: Number of memory units to explore
            context: Additional context string to include in LLM prompt (not used in search)

        Returns:
            ThinkResult containing:
                - text: Plain text answer (no markdown)
                - based_on: Dict with 'world', 'agent', and 'opinion' fact lists (MemoryFact objects)
                - new_opinions: List of newly formed opinions
        """
        # Use cached LLM config
        if self._llm_config is None:
            raise ValueError("Memory LLM API key not set. Set HINDSIGHT_API_LLM_API_KEY environment variable.")

        # Steps 1-3: Run multi-fact-type search (12-way retrieval: 4 methods × 3 fact types)
        search_result = await self.search_async(
            agent_id=agent_id,
            query=query,
            thinking_budget=thinking_budget,
            max_tokens=4096,
            enable_trace=False,
            fact_type=['agent', 'world', 'opinion']
        )

        all_results = search_result.results
        logger.info(f"[THINK] Search returned {len(all_results)} results")

        # Split results by fact type for structured response
        agent_results = [r for r in all_results if r.fact_type == 'agent']
        world_results = [r for r in all_results if r.fact_type == 'world']
        opinion_results = [r for r in all_results if r.fact_type == 'opinion']

        logger.info(f"[THINK] Split results - agent: {len(agent_results)}, world: {len(world_results)}, opinion: {len(opinion_results)}")

        # Format facts for LLM
        agent_facts_text = think_utils.format_facts_for_prompt(agent_results)
        world_facts_text = think_utils.format_facts_for_prompt(world_results)
        opinion_facts_text = think_utils.format_facts_for_prompt(opinion_results)

        logger.info(f"[THINK] Formatted facts - agent: {len(agent_facts_text)} chars, world: {len(world_facts_text)} chars, opinion: {len(opinion_facts_text)} chars")

        # Get agent profile (name, personality + background)
        profile = await self.get_agent_profile(agent_id)
        name = profile["name"]
        personality = profile["personality"]
        background = profile["background"]

        # Build the prompt
        prompt = think_utils.build_think_prompt(
            agent_facts_text=agent_facts_text,
            world_facts_text=world_facts_text,
            opinion_facts_text=opinion_facts_text,
            query=query,
            name=name,
            personality=personality,
            background=background,
            context=context,
        )

        logger.info(f"[THINK] Full prompt length: {len(prompt)} chars")
        logger.debug(f"[THINK] Prompt preview (first 500 chars): {prompt[:500]}")

        system_message = think_utils.get_system_message(personality)

        answer_text = await self._llm_config.call(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            scope="memory_think",
            temperature=0.9,
            max_tokens=1000
        )

        answer_text = answer_text.strip()

        # Submit form_opinion task for background processing
        logger.debug(f"[THINK] Submitting form_opinion task for agent {agent_id}")
        await self._task_backend.submit_task({
            'type': 'form_opinion',
            'agent_id': agent_id,
            'answer_text': answer_text,
            'query': query
        })
        logger.debug(f"[THINK] form_opinion task submitted")

        # Return response with facts split by type
        return ThinkResult(
            text=answer_text,
            based_on={
                "world": world_results,
                "agent": agent_results,
                "opinion": opinion_results
            },
            new_opinions=[]  # Opinions are being extracted asynchronously
        )

    async def _extract_and_store_opinions_async(
        self,
        agent_id: str,
        answer_text: str,
        query: str
    ):
        """
        Background task to extract and store opinions from think response.

        This runs asynchronously and does not block the think response.

        Args:
            agent_id: Agent identifier
            answer_text: The generated answer text
            query: The original query
        """
        try:
            logger.debug(f"[THINK] Extracting opinions from answer for agent {agent_id}")
            # Extract opinions from the answer
            new_opinions = await think_utils.extract_opinions_from_text(
                self._llm_config, text=answer_text, query=query
            )
            logger.debug(f"[THINK] Extracted {len(new_opinions)} opinions")

            # Store new opinions
            if new_opinions:
                from datetime import datetime, timezone
                current_time = datetime.now(timezone.utc)
                for opinion_dict in new_opinions:
                    await self.put_async(
                        agent_id=agent_id,
                        content=opinion_dict["text"],
                        context=f"formed during thinking about: {query}",
                        event_date=current_time,
                        fact_type_override='opinion',
                        confidence_score=opinion_dict["confidence"]
                    )

                logger.debug(f"[THINK] Extracted and stored {len(new_opinions)} new opinions")
        except Exception as e:
            logger.warning(f"[THINK] Failed to extract/store opinions: {str(e)}")

