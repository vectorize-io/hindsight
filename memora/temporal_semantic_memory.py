"""
Temporal + Semantic + Entity Memory System for AI Agents.

This implements a sophisticated memory architecture that combines:
1. Temporal links: Memories connected by time proximity
2. Semantic links: Memories connected by meaning/similarity
3. Entity links: Memories connected by shared entities (PERSON, ORG, etc.)
4. Spreading activation: Search through the graph with activation decay
5. Dynamic weighting: Recency and frequency-based importance
"""
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import asyncpg
import asyncio
from .embeddings import Embeddings, SentenceTransformersEmbeddings
from .cross_encoder import CrossEncoderReranker as CrossEncoderModel
import time
import numpy as np
import uuid
import logging

from .utils import (
    extract_facts,
    calculate_recency_weight,
    calculate_frequency_weight,
)
from .entity_resolver import EntityResolver
from .operations import EmbeddingOperationsMixin, LinkOperationsMixin, ThinkOperationsMixin
from .llm_wrapper import LLMConfig
from .task_backend import TaskBackend, AsyncIOQueueBackend
from .search.reranking import HeuristicReranker, CrossEncoderReranker


def utcnow():
    """Get current UTC time with timezone info."""
    return datetime.now(timezone.utc)


# Logger for memory system
logger = logging.getLogger(__name__)

# Tiktoken for token budget filtering
import tiktoken

# Cache tiktoken encoding for token budget filtering (module-level singleton)
_TIKTOKEN_ENCODING = None

def _get_tiktoken_encoding():
    """Get cached tiktoken encoding (cl100k_base for GPT-4/3.5)."""
    global _TIKTOKEN_ENCODING
    if _TIKTOKEN_ENCODING is None:
        _TIKTOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
    return _TIKTOKEN_ENCODING


class TemporalSemanticMemory(
    EmbeddingOperationsMixin,
    LinkOperationsMixin,
    ThinkOperationsMixin,
):
    """
    Advanced memory system using temporal and semantic linking with PostgreSQL.

    Uses mixin architecture for code organization:
    - EmbeddingOperationsMixin: Embedding generation
    - LinkOperationsMixin: Entity, temporal, and semantic link creation
    - ThinkOperationsMixin: Think operations for formulating answers with opinions
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
            pool_min_size: Minimum number of connections in the pool (default: 5)
            pool_max_size: Maximum number of connections in the pool (default: 100)
                          Increase for parallel think/search operations (e.g., 200-300 for 100+ parallel thinks)
            task_backend: Custom task backend for async task execution. If not provided, uses AsyncIOQueueBackend
        """
        # Initialize PostgreSQL connection URL
        self.db_url = db_url

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

        # Initialize LLM configuration
        self._llm_config = LLMConfig(
            provider=memory_llm_provider,
            api_key=memory_llm_api_key,
            base_url=memory_llm_base_url,
            model=memory_llm_model,
        )

        # Store client and model for convenience
        self._llm_client = self._llm_config.client
        self._llm_model = self._llm_config.model

        # Initialize rerankers (cached for performance)
        self._heuristic_reranker = HeuristicReranker()
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
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE memory_units SET access_count = access_count + 1 WHERE id = ANY($1::uuid[])",
                    uuid_list
                )
        except Exception as e:
            logger.error(f"Access count handler: Error updating access counts: {e}")

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

        if task_type == 'access_count_update':
            await self._handle_access_count_update(task_dict)
        elif task_type == 'reinforce_opinion':
            await self._handle_reinforce_opinion(task_dict)
        elif task_type == 'form_opinion':
            await self._handle_form_opinion(task_dict)
        else:
            logger.error(f"Unknown task type: {task_type}")

    async def initialize(self):
        """Initialize the connection pool and background workers."""
        if self._initialized:
            return

        # Create connection pool
        # For read-heavy workloads with many parallel think/search operations,
        # we need a larger pool. Read operations don't need strong isolation.
        self._pool = await asyncpg.create_pool(
            self.db_url,
            min_size=self._pool_min_size,
            max_size=self._pool_max_size,
            command_timeout=60,
            statement_cache_size=0  # Disable prepared statement cache
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
        document_metadata: Optional[Dict[str, Any]] = None,
        upsert: bool = False,
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
            document_id: Optional document ID for tracking and upsert
            document_metadata: Optional metadata about the document
            upsert: If True and document_id exists, delete old units and create new ones
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
            document_metadata=document_metadata,
            upsert=upsert,
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
        document_metadata: Optional[Dict[str, Any]] = None,
        upsert: bool = False,
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
            document_id: Optional document ID for tracking and upsert
            document_metadata: Optional metadata about the document
            upsert: If True and document_id exists, delete old units and create new ones
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
                document_id="meeting-2024-01-15",
                upsert=True
            )
            # Returns: [["unit-id-1"], ["unit-id-2"]]
        """
        start_time = time.time()

        if not contents:
            return []

        # Auto-chunk large batches by character count to avoid timeouts and memory issues
        # Calculate total character count
        total_chars = sum(len(item.get("content", "")) for item in contents)

        # Threshold: 50,000 characters per sub-batch (roughly 12k tokens with average 4 chars/token)
        CHARS_PER_BATCH = 50_000

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
                    document_metadata=document_metadata,
                    upsert=upsert and i == 1,  # Only upsert on first batch
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
            document_metadata=document_metadata,
            upsert=upsert,
            fact_type_override=fact_type_override,
            confidence_score=confidence_score
        )

    async def _put_batch_async_internal(
        self,
        agent_id: str,
        contents: List[Dict[str, Any]],
        document_id: Optional[str] = None,
        document_metadata: Optional[Dict[str, Any]] = None,
        upsert: bool = False,
        fact_type_override: Optional[str] = None,
        confidence_score: Optional[float] = None,
    ) -> List[List[str]]:
        """
        Internal method for batch processing without chunking logic.

        Assumes contents are already appropriately sized (< 50k chars).
        Called by put_batch_async after chunking large batches.
        """
        start_time = time.time()
        total_chars = sum(len(item.get("content", "")) for item in contents)

        # Buffer all logs to avoid interleaving
        log_buffer = []
        log_buffer.append(f"{'='*60}")
        log_buffer.append(f"PUT_BATCH_ASYNC START: {agent_id}")
        log_buffer.append(f"Batch size: {len(contents)} content items, {total_chars:,} chars")
        log_buffer.append(f"{'='*60}")

        # Step 1: Extract facts from ALL contents in parallel
        step_start = time.time()

        # Create tasks for parallel fact extraction using configured LLM
        fact_extraction_tasks = []
        for item in contents:
            content = item["content"]
            context = item.get("context", "")
            event_date = item.get("event_date") or utcnow()

            task = extract_facts(content, event_date, context, llm_config=self._llm_config)
            fact_extraction_tasks.append((task, event_date, context))

        # Wait for all fact extractions to complete
        all_fact_results = await asyncio.gather(*[task for task, _, _ in fact_extraction_tasks])
        log_buffer.append(f"[1] Extract facts (parallel): {len(fact_extraction_tasks)} contents in {time.time() - step_start:.3f}s")

        # Flatten and track which facts belong to which content
        all_fact_texts = []
        all_fact_dates = []
        all_contexts = []
        all_fact_entities = []  # NEW: Store LLM-extracted entities per fact
        all_fact_types = []  # Store fact type (world or agent)
        content_boundaries = []  # [(start_idx, end_idx), ...]

        current_idx = 0
        for i, ((_, event_date, context), fact_dicts) in enumerate(zip(fact_extraction_tasks, all_fact_results)):
            start_idx = current_idx

            for fact_dict in fact_dicts:
                all_fact_texts.append(fact_dict['fact'])
                try:
                    from dateutil import parser as date_parser
                    fact_date = date_parser.isoparse(fact_dict['date'])
                    all_fact_dates.append(fact_date)
                except Exception:
                    all_fact_dates.append(event_date)
                all_contexts.append(context)
                # Extract entities from fact (default to empty list if not present)
                all_fact_entities.append(fact_dict.get('entities', []))
                # Extract fact type (use override if provided, else use extracted type or default to 'world')
                if fact_type_override:
                    all_fact_types.append(fact_type_override)
                else:
                    all_fact_types.append(fact_dict.get('fact_type', 'world'))

            end_idx = current_idx + len(fact_dicts)
            content_boundaries.append((start_idx, end_idx))
            current_idx = end_idx

        total_facts = len(all_fact_texts)

        if total_facts == 0:
            return [[] for _ in contents]

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
        all_embeddings = await self._generate_embeddings_batch(augmented_texts)
        log_buffer.append(f"[2] Generate embeddings (parallel): {len(all_embeddings)} embeddings in {time.time() - step_start:.3f}s")

        # Step 3: Process everything in ONE database transaction
        logger.debug("Getting connection pool")
        pool = await self._get_pool()
        logger.debug("Acquiring connection from pool")
        async with pool.acquire() as conn:
            logger.debug("Starting transaction")
            async with conn.transaction():
                logger.debug("Inside transaction")
                try:
                    # Handle document tracking and upsert
                    if document_id:
                        logger.debug(f"Handling document tracking for {document_id}")
                        import hashlib
                        import json

                        # Calculate content hash from all content items
                        combined_content = "\n".join([c.get("content", "") for c in contents])
                        content_hash = hashlib.sha256(combined_content.encode()).hexdigest()

                        # If upsert, delete old document first (cascades to units and links)
                        if upsert:
                            deleted = await conn.fetchval(
                                "DELETE FROM documents WHERE id = $1 AND agent_id = $2 RETURNING id",
                                document_id, agent_id
                            )
                            if deleted:
                                logger.debug(f"[3.1] Upsert: Deleted existing document '{document_id}' and all its units")

                        # Insert or update document
                        # Always use ON CONFLICT for idempotent behavior
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
                            json.dumps(document_metadata or {})
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
                    filtered_contexts = [c for c, is_dup in zip(all_contexts, all_is_duplicate) if not is_dup]
                    filtered_entities = [ents for ents, is_dup in zip(all_fact_entities, all_is_duplicate) if not is_dup]
                    filtered_fact_types = [ft for ft, is_dup in zip(all_fact_types, all_is_duplicate) if not is_dup]

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
                    results = await conn.fetch(
                        """
                        INSERT INTO memory_units (agent_id, document_id, text, context, embedding, event_date, fact_type, confidence_score, access_count)
                        SELECT * FROM unnest($1::text[], $2::text[], $3::text[], $4::text[], $5::vector[], $6::timestamptz[], $7::text[], $8::float[], $9::integer[])
                        RETURNING id
                        """,
                        [agent_id] * len(filtered_sentences),
                        [document_id] * len(filtered_sentences) if document_id else [None] * len(filtered_sentences),
                        filtered_sentences,
                        filtered_contexts,
                        filtered_embeddings_str,
                        filtered_dates,
                        filtered_fact_types,
                        confidence_scores,
                        [0] * len(filtered_sentences)
                    )

                    created_unit_ids = [str(row['id']) for row in results]
                    logger.debug(f"Batch insert complete: {len(created_unit_ids)} units created")
                    log_buffer.append(f"[5] Batch insert units: {len(created_unit_ids)} units in {time.time() - step_start:.3f}s")

                    # Process entities for ALL units
                    logger.debug("Processing entities")
                    step_start = time.time()
                    all_entity_links = await self._extract_entities_batch_optimized(
                        conn, agent_id, created_unit_ids, filtered_sentences, "", filtered_dates, filtered_entities, log_buffer
                    )
                    logger.debug(f"Entity processing complete: {len(all_entity_links)} links")
                    log_buffer.append(f"[6] Process entities (batched): {time.time() - step_start:.3f}s")

                    # Create temporal links
                    logger.debug("Creating temporal links")
                    step_start = time.time()
                    await self._create_temporal_links_batch_per_fact(conn, agent_id, created_unit_ids, log_buffer=log_buffer)
                    logger.debug("Temporal links complete")
                    log_buffer.append(f"[7] Batch create temporal links: {time.time() - step_start:.3f}s")

                    # Create semantic links
                    logger.debug("Creating semantic links")
                    step_start = time.time()
                    await self._create_semantic_links_batch(conn, agent_id, created_unit_ids, filtered_embeddings, log_buffer=log_buffer)
                    logger.debug("Semantic links complete")
                    log_buffer.append(f"[8] Batch create semantic links: {time.time() - step_start:.3f}s")

                    # Insert entity links
                    logger.debug("Inserting entity links")
                    step_start = time.time()
                    if all_entity_links:
                        await self._insert_entity_links_batch(conn, all_entity_links)
                    logger.debug("Entity links inserted")
                    log_buffer.append(f"[9] Batch insert entity links: {time.time() - step_start:.3f}s")

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
        reranker: str = "heuristic",
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
            reranker: Reranking strategy - "heuristic" (default) or "cross-encoder"

        Returns:
            Tuple of (results, trace)
        """
        # Run async version synchronously
        return asyncio.run(self.search_async(
            agent_id, query, fact_type, thinking_budget, max_tokens, enable_trace, reranker
        ))

    async def search_async(
        self,
        agent_id: str,
        query: str,
        fact_type: str,
        thinking_budget: int = 50,
        max_tokens: int = 4096,
        enable_trace: bool = False,
        reranker: str = "cross-encoder",
    ) -> tuple[List[Dict[str, Any]], Optional[Any]]:
        """
        Search memories using 4-way parallel retrieval (semantic + keyword + graph + temporal).

        This implements the core SEARCH operation:
        1. Retrieval: Run 4 parallel retrievals (semantic vector, BM25 keyword, graph activation, temporal graph)
        2. Merge: Combine using Reciprocal Rank Fusion (RRF)
        3. Rerank: Score using selected reranker (heuristic or cross-encoder)
        4. Diversify: Apply MMR for diversity
        5. Token Filter: Return results up to max_tokens budget

        Args:
            agent_id: Agent ID to search for
            query: Search query
            fact_type: Type of facts to search ('world', 'agent', 'opinion')
            thinking_budget: How many units to explore in graph traversal (controls compute cost)
            max_tokens: Maximum tokens to return (counts only 'text' field, default 4096)
                       Results are returned until token budget is reached, stopping before
                       including a fact that would exceed the limit
            enable_trace: Whether to return search trace for debugging (deprecated)
            reranker: Reranking strategy - "heuristic" (default) or "cross-encoder"
                     - heuristic: 60% semantic + 40% BM25 + normalized boosts (fast)
                     - cross-encoder: Neural reranking with ms-marco-MiniLM-L-6-v2 (slower but more accurate)

        Returns:
            Tuple of (results, trace) where results is a list of memory units
            and trace is None (tracing removed)
        """
        # Backpressure: limit concurrent searches to prevent overwhelming the database
        async with self._search_semaphore:
            # Retry loop for connection errors
            max_retries = 3
            for attempt in range(max_retries + 1):
                try:
                    return await self._search_with_retries(
                        agent_id, query, fact_type, thinking_budget, max_tokens, enable_trace, reranker
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
        fact_type: str,
        thinking_budget: int,
        max_tokens: int,
        enable_trace: bool,
        reranker: str,
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
            reranker: Reranking strategy ("heuristic" or "cross-encoder")

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
            query_embedding = self._generate_embedding(query)
            step_duration = time.time() - step_start
            log_buffer.append(f"  [1] Generate query embedding: {step_duration:.3f}s")

            if tracer:
                tracer.record_query_embedding(query_embedding)
                tracer.add_phase_metric("generate_query_embedding", step_duration)

            # Step 2: 3-Way or 4-Way Parallel Retrieval
            step_start = time.time()
            query_embedding_str = str(query_embedding)

            from .search.retrieval import retrieve_parallel

            # Track each retrieval start time
            retrieval_start = time.time()
            semantic_results, bm25_results, graph_results, temporal_results = await retrieve_parallel(
                pool, query, query_embedding_str, agent_id, fact_type, thinking_budget
            )
            retrieval_duration = time.time() - retrieval_start

            step_duration = time.time() - step_start
            if temporal_results:
                log_buffer.append(f"  [2] 4-way retrieval: semantic={len(semantic_results)}, bm25={len(bm25_results)}, graph={len(graph_results)}, temporal={len(temporal_results)} in {step_duration:.3f}s")
            else:
                log_buffer.append(f"  [2] 3-way retrieval: semantic={len(semantic_results)}, bm25={len(bm25_results)}, graph={len(graph_results)} in {step_duration:.3f}s")

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

            # Step 5: Rerank using selected strategy (use cached rerankers)
            if reranker == "cross-encoder":
                reranker_instance = self._cross_encoder_reranker
                log_buffer.append(f"  [4] Using cross-encoder reranker")
            else:
                reranker_instance = self._heuristic_reranker
                log_buffer.append(f"  [4] Using heuristic reranker")

            # Rerank more candidates than we need (thinking_budget * 2)
            # so token filtering has diverse options to choose from
            rerank_limit = thinking_budget * 2
            results = reranker_instance.rerank(query, results, rerank_limit)

            step_duration = time.time() - step_start
            log_buffer.append(f"  [4] Reranking: {len(results)} candidates scored in {step_duration:.3f}s")

            if tracer:
                tracer.add_reranked(results, merged_candidates)
                tracer.add_phase_metric("reranking", step_duration, {
                    "reranker_type": reranker,
                    "candidates_reranked": len(results)
                })

            # Step 6: Apply MMR (always enabled with λ=0.5)
            step_start = time.time()
            from .search.mmr import apply_mmr

            mmr_lambda = 0.5
            # MMR also uses thinking_budget * 2 to have diverse options for token filtering
            mmr_limit = thinking_budget * 2
            top_results = apply_mmr(results, mmr_limit, mmr_lambda, log_buffer)

            step_duration = time.time() - step_start
            log_buffer.append(f"  [5] MMR diversification (λ={mmr_lambda}): {step_duration:.3f}s")

            if tracer:
                tracer.add_phase_metric("mmr_diversification", step_duration, {
                    "lambda": mmr_lambda,
                    "results_selected": len(top_results)
                })

            # Step 7: Token budget filtering
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

            # Finalize trace if enabled
            if tracer:
                trace = tracer.finalize(top_results)
                return top_results, trace
            return top_results, None

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

    def _apply_mmr(
        self,
        results: List[Dict[str, Any]],
        top_k: int,
        mmr_lambda: float,
        log_buffer: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Apply Maximal Marginal Relevance (MMR) to diversify search results.

        MMR balances relevance with diversity by selecting results that are:
        1. Relevant to the query (high score)
        2. Different from already selected results (low similarity)

        Formula: MMR = λ * relevance - (1-λ) * max_similarity_to_selected

        Args:
            results: Sorted list of all results with embeddings
            top_k: Number of results to select
            mmr_lambda: Balance parameter (0=max diversity, 1=max relevance)
            log_buffer: Logging buffer

        Returns:
            Diversified list of top_k results
        """
        if not results or top_k <= 0:
            return []

        # Normalize weights to [0, 1] for fair comparison with similarity
        max_weight = max(r["weight"] for r in results)
        min_weight = min(r["weight"] for r in results)
        weight_range = max_weight - min_weight if max_weight > min_weight else 1.0

        # Pre-compute normalized relevance scores for all results
        for idx, result in enumerate(results):
            result["original_rank"] = idx + 1
            result["normalized_relevance"] = (result["weight"] - min_weight) / weight_range

        # Extract embeddings as a numpy array for vectorized operations
        # Shape: (num_results, embedding_dim)
        embeddings_list = []
        valid_indices = []
        for idx, result in enumerate(results):
            if result.get("embedding") is not None:
                embeddings_list.append(result["embedding"])
                valid_indices.append(idx)

        if not embeddings_list:
            # No embeddings available, just return top-k by relevance
            return results[:top_k]

        # Stack embeddings into a matrix (num_results, embedding_dim)
        embeddings_matrix = np.array(embeddings_list, dtype=np.float32)

        # Normalize embeddings for faster cosine similarity (just dot product after normalization)
        norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Avoid division by zero
        embeddings_matrix = embeddings_matrix / norms

        selected_indices = []
        remaining_indices = list(range(len(results)))
        diversified_count = 0

        for selection_round in range(min(top_k, len(results))):
            if not remaining_indices:
                break

            best_mmr_score = float('-inf')
            best_remaining_idx = 0

            # Vectorized computation for all remaining candidates
            for remaining_idx, candidate_idx in enumerate(remaining_indices):
                candidate = results[candidate_idx]
                normalized_relevance = candidate["normalized_relevance"]

                # Calculate max similarity to selected results
                max_similarity = 0.0
                if selected_indices and candidate_idx in valid_indices:
                    # Find position in embeddings_matrix
                    embedding_idx = valid_indices.index(candidate_idx)
                    candidate_embedding = embeddings_matrix[embedding_idx]

                    # Vectorized similarity calculation with all selected embeddings
                    if selected_indices:
                        selected_embedding_indices = [valid_indices.index(idx) for idx in selected_indices if idx in valid_indices]
                        if selected_embedding_indices:
                            selected_embeddings = embeddings_matrix[selected_embedding_indices]
                            # Compute cosine similarities in one operation (already normalized, so just dot product)
                            similarities = np.dot(selected_embeddings, candidate_embedding)
                            max_similarity = float(np.max(similarities))

                # MMR score: balance relevance and diversity
                mmr_score = mmr_lambda * normalized_relevance - (1 - mmr_lambda) * max_similarity

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_remaining_idx = remaining_idx
                    best_max_similarity = max_similarity

            # Select the best candidate
            best_candidate_idx = remaining_indices.pop(best_remaining_idx)
            best_candidate = results[best_candidate_idx]

            # Store MMR metadata
            best_candidate["mmr_score"] = best_mmr_score
            best_candidate["mmr_relevance"] = best_candidate["normalized_relevance"]
            best_candidate["mmr_max_similarity"] = best_max_similarity
            best_candidate["mmr_diversified"] = best_remaining_idx > 0

            selected_indices.append(best_candidate_idx)

            if best_remaining_idx > 0:
                diversified_count += 1

        log_buffer.append(f"      MMR: Selected {len(selected_indices)} results, {diversified_count} diversified picks")

        # Return selected results in order
        selected_results = [results[idx] for idx in selected_indices]

        # Remove embeddings from final results (not needed in response)
        for result in selected_results:
            result.pop("embedding", None)
            result.pop("normalized_relevance", None)  # Clean up temp field

        return selected_results

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
        async with pool.acquire() as conn:
            doc = await conn.fetchrow(
                """
                SELECT d.id, d.agent_id, d.original_text, d.content_hash, d.metadata,
                       d.created_at, d.updated_at, COUNT(mu.id) as unit_count
                FROM documents d
                LEFT JOIN memory_units mu ON mu.document_id = d.id
                WHERE d.id = $1 AND d.agent_id = $2
                GROUP BY d.id, d.agent_id, d.original_text, d.content_hash, d.metadata, d.created_at, d.updated_at
                """,
                document_id, agent_id
            )

            if not doc:
                return None

            import json
            return {
                "id": doc["id"],
                "agent_id": doc["agent_id"],
                "original_text": doc["original_text"],
                "content_hash": doc["content_hash"],
                "metadata": json.loads(doc["metadata"]) if doc["metadata"] else {},
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
        async with pool.acquire() as conn:
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
        async with pool.acquire() as conn:
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

    async def delete_agent(self, agent_id: str) -> Dict[str, int]:
        """
        Delete all data for a specific agent (multi-tenant cleanup).

        This is much more efficient than dropping all tables and allows
        multiple agents to coexist in the same database.

        Deletes (with CASCADE):
        - All memory units for this agent
        - All entities for this agent
        - All associated links, unit-entity associations, and co-occurrences

        Args:
            agent_id: Agent ID to delete

        Returns:
            Dictionary with counts of deleted items
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                try:
                    # Count before deletion for reporting
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

    async def list_agents(self) -> List[str]:
        """
        Get list of all agent IDs in the database.

        Returns:
            List of agent IDs
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            # Get distinct agent IDs from memory_units
            agents = await conn.fetch("""
                SELECT DISTINCT agent_id
                FROM memory_units
                WHERE agent_id IS NOT NULL
                ORDER BY agent_id
            """)

            return [row['agent_id'] for row in agents]

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
        async with pool.acquire() as conn:
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
                ORDER BY event_date
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
                SELECT ue.unit_id, e.canonical_name, e.entity_type
                FROM unit_entities ue
                JOIN entities e ON ue.entity_id = e.id
                ORDER BY ue.unit_id
            """)

        # Build entity mapping
        entity_map = {}
        for row in unit_entities:
            unit_id = row['unit_id']
            entity_name = row['canonical_name']
            entity_type = row['entity_type']
            if unit_id not in entity_map:
                entity_map[unit_id] = []
            entity_map[unit_id].append(f"{entity_name} ({entity_type})")

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
            async with pool.acquire() as conn:
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

