"""
Cross-encoder abstraction for reranking.

Provides an interface for reranking with different backends.

Configuration via environment variables - see hindsight_api.config for all env var names.
"""

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

import httpx

from ..config import (
    DEFAULT_LITELLM_API_BASE,
    DEFAULT_RERANKER_COHERE_MODEL,
    DEFAULT_RERANKER_FLASHRANK_CACHE_DIR,
    DEFAULT_RERANKER_FLASHRANK_MODEL,
    DEFAULT_RERANKER_LITELLM_MODEL,
    DEFAULT_RERANKER_LOCAL_FORCE_CPU,
    DEFAULT_RERANKER_LOCAL_MAX_CONCURRENT,
    DEFAULT_RERANKER_LOCAL_MODEL,
    DEFAULT_RERANKER_PROVIDER,
    DEFAULT_RERANKER_TEI_BATCH_SIZE,
    DEFAULT_RERANKER_TEI_MAX_CONCURRENT,
    ENV_COHERE_API_KEY,
    ENV_LITELLM_API_BASE,
    ENV_LITELLM_API_KEY,
    ENV_RERANKER_COHERE_BASE_URL,
    ENV_RERANKER_COHERE_MODEL,
    ENV_RERANKER_FLASHRANK_CACHE_DIR,
    ENV_RERANKER_FLASHRANK_MODEL,
    ENV_RERANKER_LITELLM_MODEL,
    ENV_RERANKER_LOCAL_FORCE_CPU,
    ENV_RERANKER_LOCAL_MAX_CONCURRENT,
    ENV_RERANKER_LOCAL_MODEL,
    ENV_RERANKER_PROVIDER,
    ENV_RERANKER_TEI_BATCH_SIZE,
    ENV_RERANKER_TEI_MAX_CONCURRENT,
    ENV_RERANKER_TEI_URL,
)

logger = logging.getLogger(__name__)


class CrossEncoderModel(ABC):
    """
    Abstract base class for cross-encoder reranking.

    Cross-encoders take query-document pairs and return relevance scores.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return a human-readable name for this provider (e.g., 'local', 'tei')."""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the cross-encoder model asynchronously.

        This should be called during startup to load/connect to the model
        and avoid cold start latency on first predict() call.
        """
        pass

    @abstractmethod
    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """
        Score query-document pairs for relevance.

        Args:
            pairs: List of (query, document) tuples to score

        Returns:
            List of relevance scores (higher = more relevant)
        """
        pass


class LocalSTCrossEncoder(CrossEncoderModel):
    """
    Local cross-encoder implementation using SentenceTransformers.

    Call initialize() during startup to load the model and avoid cold starts.

    Default model is cross-encoder/ms-marco-MiniLM-L-6-v2:
    - Fast inference (~80ms for 100 pairs on CPU)
    - Small model (80MB)
    - Trained for passage re-ranking

    Uses a dedicated thread pool to limit concurrent CPU-bound work.
    """

    # Shared executor across all instances (one model loaded anyway)
    _executor: ThreadPoolExecutor | None = None
    _max_concurrent: int = 4  # Limit concurrent CPU-bound reranking calls

    def __init__(self, model_name: str | None = None, max_concurrent: int = 4, force_cpu: bool = False):
        """
        Initialize local SentenceTransformers cross-encoder.

        Args:
            model_name: Name of the CrossEncoder model to use.
                       Default: cross-encoder/ms-marco-MiniLM-L-6-v2
            max_concurrent: Maximum concurrent reranking calls (default: 2).
                           Higher values may cause CPU thrashing under load.
            force_cpu: Force CPU mode (avoids MPS/XPC issues on macOS in daemon mode).
                      Default: False
        """
        self.model_name = model_name or DEFAULT_RERANKER_LOCAL_MODEL
        self.force_cpu = force_cpu
        self._model = None
        LocalSTCrossEncoder._max_concurrent = max_concurrent

    @property
    def provider_name(self) -> str:
        return "local"

    async def initialize(self) -> None:
        """Load the cross-encoder model and initialize the executor."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for LocalSTCrossEncoder. "
                "Install it with: pip install sentence-transformers"
            )

        logger.info(f"Reranker: initializing local provider with model {self.model_name}")

        # Determine device based on hardware availability.
        # We always set low_cpu_mem_usage=False to prevent lazy loading (meta tensors)
        # which can cause issues when accelerate is installed but no GPU is available.
        # Note: We do NOT use device_map because CrossEncoder internally calls .to(device)
        # after loading, which conflicts with accelerate's device_map handling.
        import torch

        # Force CPU mode if configured (used in daemon mode to avoid MPS/XPC issues on macOS)
        if self.force_cpu:
            device = "cpu"
            logger.info("Reranker: forcing CPU mode (HINDSIGHT_API_RERANKER_LOCAL_FORCE_CPU=1)")
        else:
            # Check for GPU (CUDA) or Apple Silicon (MPS)
            # Wrap in try-except to gracefully handle any device detection issues
            # (e.g., in CI environments or when PyTorch is built without GPU support)
            device = "cpu"  # Default to CPU
            try:
                has_gpu = torch.cuda.is_available() or (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                )
                if has_gpu:
                    device = None  # Let sentence-transformers auto-detect GPU/MPS
            except Exception as e:
                logger.warning(f"Failed to detect GPU/MPS, falling back to CPU: {e}")

        self._model = CrossEncoder(
            self.model_name,
            device=device,
            model_kwargs={"low_cpu_mem_usage": False},
        )

        # Initialize shared executor (limited workers naturally limits concurrency)
        if LocalSTCrossEncoder._executor is None:
            LocalSTCrossEncoder._executor = ThreadPoolExecutor(
                max_workers=LocalSTCrossEncoder._max_concurrent,
                thread_name_prefix="reranker",
            )
            logger.info(f"Reranker: local provider initialized (max_concurrent={LocalSTCrossEncoder._max_concurrent})")
        else:
            logger.info("Reranker: local provider initialized (using existing executor)")

    def _predict_sync(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Synchronous prediction wrapper for thread pool execution."""
        scores = self._model.predict(pairs, show_progress_bar=False)
        return scores.tolist() if hasattr(scores, "tolist") else list(scores)

    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """
        Score query-document pairs for relevance.

        Uses a dedicated thread pool with limited workers to prevent CPU thrashing.

        Args:
            pairs: List of (query, document) tuples to score

        Returns:
            List of relevance scores (raw logits from the model)
        """
        if self._model is None:
            raise RuntimeError("Reranker not initialized. Call initialize() first.")

        # Use dedicated executor - limited workers naturally limits concurrency
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            LocalSTCrossEncoder._executor,
            self._predict_sync,
            pairs,
        )


class RemoteTEICrossEncoder(CrossEncoderModel):
    """
    Remote cross-encoder implementation using HuggingFace Text Embeddings Inference (TEI) HTTP API.

    TEI supports reranking via the /rerank endpoint.
    See: https://github.com/huggingface/text-embeddings-inference

    Note: The TEI server must be running a cross-encoder/reranker model.

    Requests are made in parallel with configurable batch size and max concurrency (backpressure).
    Uses a GLOBAL semaphore to limit concurrent requests across ALL recall operations.
    """

    # Global semaphore shared across all instances and calls to prevent thundering herd
    _global_semaphore: asyncio.Semaphore | None = None
    _global_max_concurrent: int = DEFAULT_RERANKER_TEI_MAX_CONCURRENT

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        batch_size: int = DEFAULT_RERANKER_TEI_BATCH_SIZE,
        max_concurrent: int = DEFAULT_RERANKER_TEI_MAX_CONCURRENT,
        max_retries: int = 3,
        retry_delay: float = 0.5,
    ):
        """
        Initialize remote TEI cross-encoder client.

        Args:
            base_url: Base URL of the TEI server (e.g., "http://localhost:8080")
            timeout: Request timeout in seconds (default: 30.0)
            batch_size: Maximum batch size for rerank requests (default: 128)
            max_concurrent: Maximum concurrent requests for backpressure (default: 8).
                           This is a GLOBAL limit across all parallel recall operations.
            max_retries: Maximum number of retries for failed requests (default: 3)
            retry_delay: Initial delay between retries in seconds, doubles each retry (default: 0.5)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._async_client: httpx.AsyncClient | None = None
        self._model_id: str | None = None

        # Update global semaphore if max_concurrent changed
        if (
            RemoteTEICrossEncoder._global_semaphore is None
            or RemoteTEICrossEncoder._global_max_concurrent != max_concurrent
        ):
            RemoteTEICrossEncoder._global_max_concurrent = max_concurrent
            RemoteTEICrossEncoder._global_semaphore = asyncio.Semaphore(max_concurrent)

    @property
    def provider_name(self) -> str:
        return "tei"

    async def _async_request_with_retry(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        method: str,
        url: str,
        **kwargs,
    ) -> httpx.Response:
        """Make an async HTTP request with automatic retries on transient errors and semaphore for backpressure."""
        last_error = None
        delay = self.retry_delay

        async with semaphore:
            for attempt in range(self.max_retries + 1):
                try:
                    if method == "GET":
                        response = await client.get(url, **kwargs)
                    else:
                        response = await client.post(url, **kwargs)
                    response.raise_for_status()
                    return response
                except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as e:
                    last_error = e
                    if attempt < self.max_retries:
                        logger.warning(
                            f"TEI request failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                            f"Retrying in {delay}s..."
                        )
                        await asyncio.sleep(delay)
                        delay *= 2  # Exponential backoff
                except httpx.HTTPStatusError as e:
                    # Retry on 5xx server errors
                    if e.response.status_code >= 500 and attempt < self.max_retries:
                        last_error = e
                        logger.warning(
                            f"TEI server error (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                            f"Retrying in {delay}s..."
                        )
                        await asyncio.sleep(delay)
                        delay *= 2
                    else:
                        raise

        raise last_error

    async def initialize(self) -> None:
        """Initialize the HTTP client and verify server connectivity."""
        if self._async_client is not None:
            return

        logger.info(
            f"Reranker: initializing TEI provider at {self.base_url} "
            f"(batch_size={self.batch_size}, max_concurrent={self.max_concurrent})"
        )
        self._async_client = httpx.AsyncClient(timeout=self.timeout)

        # Verify server is reachable and get model info
        # Use a temporary semaphore for initialization
        init_semaphore = asyncio.Semaphore(1)
        try:
            response = await self._async_request_with_retry(
                self._async_client, init_semaphore, "GET", f"{self.base_url}/info"
            )
            info = response.json()
            self._model_id = info.get("model_id", "unknown")
            logger.info(f"Reranker: TEI provider initialized (model: {self._model_id})")
        except httpx.HTTPError as e:
            self._async_client = None
            raise RuntimeError(f"Failed to connect to TEI server at {self.base_url}: {e}")

    async def _rerank_query_group(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        query: str,
        texts: list[str],
    ) -> list[tuple[int, float]]:
        """Rerank a single query group and return list of (original_index, score) tuples."""
        try:
            response = await self._async_request_with_retry(
                client,
                semaphore,
                "POST",
                f"{self.base_url}/rerank",
                json={
                    "query": query,
                    "texts": texts,
                    "return_text": False,
                },
            )
            results = response.json()
            # TEI returns results sorted by score descending, with original index
            return [(result["index"], result["score"]) for result in results]
        except httpx.HTTPError as e:
            raise RuntimeError(f"TEI rerank request failed: {e}")

    async def _predict_async(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Async implementation of predict that runs requests in parallel with backpressure."""
        if not pairs:
            return []

        # Group all pairs by query
        query_groups: dict[str, list[tuple[int, str]]] = {}
        for idx, (query, text) in enumerate(pairs):
            if query not in query_groups:
                query_groups[query] = []
            query_groups[query].append((idx, text))

        # Split each query group into batches
        tasks_info: list[tuple[str, list[int], list[str]]] = []  # (query, indices, texts)
        for query, indexed_texts in query_groups.items():
            indices = [idx for idx, _ in indexed_texts]
            texts = [text for _, text in indexed_texts]

            # Split into batches
            for i in range(0, len(texts), self.batch_size):
                batch_indices = indices[i : i + self.batch_size]
                batch_texts = texts[i : i + self.batch_size]
                tasks_info.append((query, batch_indices, batch_texts))

        # Run all requests in parallel with GLOBAL semaphore for backpressure
        # This ensures max_concurrent is respected across ALL parallel recall operations
        all_scores = [0.0] * len(pairs)
        semaphore = RemoteTEICrossEncoder._global_semaphore

        tasks = [
            self._rerank_query_group(self._async_client, semaphore, query, texts) for query, _, texts in tasks_info
        ]
        results = await asyncio.gather(*tasks)

        # Map scores back to original positions
        for (_, indices, _), result_scores in zip(tasks_info, results):
            for original_idx_in_batch, score in result_scores:
                global_idx = indices[original_idx_in_batch]
                all_scores[global_idx] = score

        return all_scores

    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """
        Score query-document pairs using the remote TEI reranker.

        Requests are made in parallel with configurable backpressure.

        Args:
            pairs: List of (query, document) tuples to score

        Returns:
            List of relevance scores
        """
        if self._async_client is None:
            raise RuntimeError("Reranker not initialized. Call initialize() first.")

        return await self._predict_async(pairs)


class CohereCrossEncoder(CrossEncoderModel):
    """
    Cohere cross-encoder implementation using the Cohere Rerank API.

    Supports rerank-english-v3.0 and rerank-multilingual-v3.0 models.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_RERANKER_COHERE_MODEL,
        base_url: str | None = None,
        timeout: float = 60.0,
    ):
        """
        Initialize Cohere cross-encoder client.

        Args:
            api_key: Cohere API key
            model: Cohere rerank model name (default: rerank-english-v3.0)
            base_url: Custom base URL for Cohere-compatible API (e.g., Azure-hosted endpoint)
            timeout: Request timeout in seconds (default: 60.0)
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self._client = None

    @property
    def provider_name(self) -> str:
        return "cohere"

    async def initialize(self) -> None:
        """Initialize the Cohere client."""
        if self._client is not None:
            return

        try:
            import cohere
        except ImportError:
            raise ImportError("cohere is required for CohereCrossEncoder. Install it with: pip install cohere")

        base_url_msg = f" at {self.base_url}" if self.base_url else ""
        logger.info(f"Reranker: initializing Cohere provider with model {self.model}{base_url_msg}")

        # Build client kwargs, only including base_url if set (for Azure or custom endpoints)
        client_kwargs = {"api_key": self.api_key, "timeout": self.timeout}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self._client = cohere.Client(**client_kwargs)
        logger.info("Reranker: Cohere provider initialized")

    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """
        Score query-document pairs using the Cohere Rerank API.

        Args:
            pairs: List of (query, document) tuples to score

        Returns:
            List of relevance scores
        """
        if self._client is None:
            raise RuntimeError("Reranker not initialized. Call initialize() first.")

        if not pairs:
            return []

        # Run sync Cohere API calls in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._predict_sync, pairs)

    def _predict_sync(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Synchronous predict implementation for Cohere API."""
        # Group pairs by query for efficient batching
        # Cohere rerank expects one query with multiple documents
        query_groups: dict[str, list[tuple[int, str]]] = {}
        for idx, (query, text) in enumerate(pairs):
            if query not in query_groups:
                query_groups[query] = []
            query_groups[query].append((idx, text))

        all_scores = [0.0] * len(pairs)

        for query, indexed_texts in query_groups.items():
            texts = [text for _, text in indexed_texts]
            indices = [idx for idx, _ in indexed_texts]

            response = self._client.rerank(
                query=query,
                documents=texts,
                model=self.model,
                return_documents=False,
            )

            # Map scores back to original positions
            for result in response.results:
                original_idx = result.index
                score = result.relevance_score
                all_scores[indices[original_idx]] = score

        return all_scores


class RRFPassthroughCrossEncoder(CrossEncoderModel):
    """
    Passthrough cross-encoder that preserves RRF scores without neural reranking.

    This is useful for:
    - Testing retrieval quality without reranking overhead
    - Deployments where reranking latency is unacceptable
    - Debugging to isolate retrieval vs reranking issues
    """

    def __init__(self):
        """Initialize RRF passthrough cross-encoder."""
        pass

    @property
    def provider_name(self) -> str:
        return "rrf"

    async def initialize(self) -> None:
        """No initialization needed."""
        logger.info("Reranker: RRF passthrough provider initialized (neural reranking disabled)")

    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """
        Return neutral scores - actual ranking uses RRF scores from retrieval.

        Args:
            pairs: List of (query, document) tuples (ignored)

        Returns:
            List of 0.5 scores (neutral, lets RRF scores dominate)
        """
        # Return neutral scores so RRF ranking is preserved
        return [0.5] * len(pairs)


class FlashRankCrossEncoder(CrossEncoderModel):
    """
    FlashRank cross-encoder implementation.

    FlashRank is an ultra-lite reranking library that runs on CPU without
    requiring PyTorch or Transformers. It's ideal for serverless deployments
    with minimal cold-start overhead.

    Available models:
    - ms-marco-TinyBERT-L-2-v2: Fastest, ~4MB
    - ms-marco-MiniLM-L-12-v2: Best quality, ~34MB (default)
    - rank-T5-flan: Best zero-shot, ~110MB
    - ms-marco-MultiBERT-L-12: Multi-lingual, ~150MB
    """

    # Shared executor for CPU-bound reranking
    _executor: ThreadPoolExecutor | None = None
    _max_concurrent: int = 4

    def __init__(
        self,
        model_name: str | None = None,
        cache_dir: str | None = None,
        max_length: int = 512,
        max_concurrent: int = 4,
    ):
        """
        Initialize FlashRank cross-encoder.

        Args:
            model_name: FlashRank model name. Default: ms-marco-MiniLM-L-12-v2
            cache_dir: Directory to cache downloaded models. Default: system cache
            max_length: Maximum sequence length for reranking. Default: 512
            max_concurrent: Maximum concurrent reranking calls. Default: 4
        """
        self.model_name = model_name or DEFAULT_RERANKER_FLASHRANK_MODEL
        self.cache_dir = cache_dir or DEFAULT_RERANKER_FLASHRANK_CACHE_DIR
        self.max_length = max_length
        self._ranker = None
        FlashRankCrossEncoder._max_concurrent = max_concurrent

    @property
    def provider_name(self) -> str:
        return "flashrank"

    async def initialize(self) -> None:
        """Load the FlashRank model."""
        if self._ranker is not None:
            return

        try:
            from flashrank import Ranker
        except ImportError:
            raise ImportError("flashrank is required for FlashRankCrossEncoder. Install it with: pip install flashrank")

        logger.info(f"Reranker: initializing FlashRank provider with model {self.model_name}")

        # Initialize ranker with optional cache directory
        ranker_kwargs = {"model_name": self.model_name, "max_length": self.max_length}
        if self.cache_dir:
            ranker_kwargs["cache_dir"] = self.cache_dir

        self._ranker = Ranker(**ranker_kwargs)

        # Initialize shared executor
        if FlashRankCrossEncoder._executor is None:
            FlashRankCrossEncoder._executor = ThreadPoolExecutor(
                max_workers=FlashRankCrossEncoder._max_concurrent,
                thread_name_prefix="flashrank",
            )
            logger.info(
                f"Reranker: FlashRank provider initialized (max_concurrent={FlashRankCrossEncoder._max_concurrent})"
            )
        else:
            logger.info("Reranker: FlashRank provider initialized (using existing executor)")

    def _predict_sync(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Synchronous predict - processes each query group."""
        from flashrank import RerankRequest

        if not pairs:
            return []

        # Group pairs by query
        query_groups: dict[str, list[tuple[int, str]]] = {}
        for idx, (query, text) in enumerate(pairs):
            if query not in query_groups:
                query_groups[query] = []
            query_groups[query].append((idx, text))

        all_scores = [0.0] * len(pairs)

        for query, indexed_texts in query_groups.items():
            # Build passages list for FlashRank
            passages = [{"id": i, "text": text} for i, (_, text) in enumerate(indexed_texts)]
            global_indices = [idx for idx, _ in indexed_texts]

            # Create rerank request
            request = RerankRequest(query=query, passages=passages)
            results = self._ranker.rerank(request)

            # Map scores back to original positions
            for result in results:
                local_idx = result["id"]
                score = result["score"]
                global_idx = global_indices[local_idx]
                all_scores[global_idx] = score

        return all_scores

    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """
        Score query-document pairs using FlashRank.

        Args:
            pairs: List of (query, document) tuples to score

        Returns:
            List of relevance scores (higher = more relevant)
        """
        if self._ranker is None:
            raise RuntimeError("Reranker not initialized. Call initialize() first.")

        # Run in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(FlashRankCrossEncoder._executor, self._predict_sync, pairs)


class LiteLLMCrossEncoder(CrossEncoderModel):
    """
    LiteLLM cross-encoder implementation using LiteLLM proxy's /rerank endpoint.

    LiteLLM provides a unified interface for multiple reranking providers via
    the Cohere-compatible /rerank endpoint.
    See: https://docs.litellm.ai/docs/rerank

    Supported providers via LiteLLM:
    - Cohere (rerank-english-v3.0, etc.) - prefix with cohere/
    - Together AI - prefix with together_ai/
    - Azure AI - prefix with azure_ai/
    - Jina AI - prefix with jina_ai/
    - AWS Bedrock - prefix with bedrock/
    - Voyage AI - prefix with voyage/
    """

    def __init__(
        self,
        api_base: str = DEFAULT_LITELLM_API_BASE,
        api_key: str | None = None,
        model: str = DEFAULT_RERANKER_LITELLM_MODEL,
        timeout: float = 60.0,
    ):
        """
        Initialize LiteLLM cross-encoder client.

        Args:
            api_base: Base URL of the LiteLLM proxy (default: http://localhost:4000)
            api_key: API key for the LiteLLM proxy (optional, depends on proxy config)
            model: Reranking model name (default: cohere/rerank-english-v3.0)
                   Use provider prefix (e.g., cohere/, together_ai/, voyage/)
            timeout: Request timeout in seconds (default: 60.0)
        """
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._async_client: httpx.AsyncClient | None = None

    @property
    def provider_name(self) -> str:
        return "litellm"

    async def initialize(self) -> None:
        """Initialize the async HTTP client."""
        if self._async_client is not None:
            return

        logger.info(f"Reranker: initializing LiteLLM provider at {self.api_base} with model {self.model}")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        self._async_client = httpx.AsyncClient(timeout=self.timeout, headers=headers)
        logger.info("Reranker: LiteLLM provider initialized")

    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """
        Score query-document pairs using the LiteLLM proxy's /rerank endpoint.

        Args:
            pairs: List of (query, document) tuples to score

        Returns:
            List of relevance scores
        """
        if self._async_client is None:
            raise RuntimeError("Reranker not initialized. Call initialize() first.")

        if not pairs:
            return []

        # Group pairs by query (LiteLLM rerank expects one query with multiple documents)
        query_groups: dict[str, list[tuple[int, str]]] = {}
        for idx, (query, text) in enumerate(pairs):
            if query not in query_groups:
                query_groups[query] = []
            query_groups[query].append((idx, text))

        all_scores = [0.0] * len(pairs)

        for query, indexed_texts in query_groups.items():
            texts = [text for _, text in indexed_texts]
            indices = [idx for idx, _ in indexed_texts]

            # LiteLLM /rerank follows Cohere API format
            response = await self._async_client.post(
                f"{self.api_base}/rerank",
                json={
                    "model": self.model,
                    "query": query,
                    "documents": texts,
                    "top_n": len(texts),  # Return all scores
                },
            )
            response.raise_for_status()
            result = response.json()

            # Map scores back to original positions
            # Response format: {"results": [{"index": 0, "relevance_score": 0.9}, ...]}
            for item in result.get("results", []):
                original_idx = item["index"]
                score = item.get("relevance_score", item.get("score", 0.0))
                all_scores[indices[original_idx]] = score

        return all_scores


def create_cross_encoder_from_env() -> CrossEncoderModel:
    """
    Create a CrossEncoderModel instance based on configuration.

    Reads configuration via get_config() to ensure consistency across the codebase.

    Returns:
        Configured CrossEncoderModel instance
    """
    from ..config import get_config

    config = get_config()
    provider = config.reranker_provider.lower()

    if provider == "tei":
        url = config.reranker_tei_url
        if not url:
            raise ValueError(f"{ENV_RERANKER_TEI_URL} is required when {ENV_RERANKER_PROVIDER} is 'tei'")
        return RemoteTEICrossEncoder(
            base_url=url,
            batch_size=config.reranker_tei_batch_size,
            max_concurrent=config.reranker_tei_max_concurrent,
        )
    elif provider == "local":
        return LocalSTCrossEncoder(
            model_name=config.reranker_local_model,
            max_concurrent=config.reranker_local_max_concurrent,
            force_cpu=config.reranker_local_force_cpu,
        )
    elif provider == "cohere":
        api_key = os.environ.get(ENV_COHERE_API_KEY)
        if not api_key:
            raise ValueError(f"{ENV_COHERE_API_KEY} is required when {ENV_RERANKER_PROVIDER} is 'cohere'")
        model = os.environ.get(ENV_RERANKER_COHERE_MODEL, DEFAULT_RERANKER_COHERE_MODEL)
        base_url = os.environ.get(ENV_RERANKER_COHERE_BASE_URL) or None
        return CohereCrossEncoder(api_key=api_key, model=model, base_url=base_url)
    elif provider == "flashrank":
        model = os.environ.get(ENV_RERANKER_FLASHRANK_MODEL, DEFAULT_RERANKER_FLASHRANK_MODEL)
        cache_dir = os.environ.get(ENV_RERANKER_FLASHRANK_CACHE_DIR, DEFAULT_RERANKER_FLASHRANK_CACHE_DIR)
        return FlashRankCrossEncoder(model_name=model, cache_dir=cache_dir)
    elif provider == "litellm":
        api_base = os.environ.get(ENV_LITELLM_API_BASE, DEFAULT_LITELLM_API_BASE)
        api_key = os.environ.get(ENV_LITELLM_API_KEY)
        model = os.environ.get(ENV_RERANKER_LITELLM_MODEL, DEFAULT_RERANKER_LITELLM_MODEL)
        return LiteLLMCrossEncoder(api_base=api_base, api_key=api_key, model=model)
    elif provider == "rrf":
        return RRFPassthroughCrossEncoder()
    else:
        raise ValueError(
            f"Unknown reranker provider: {provider}. Supported: 'local', 'tei', 'cohere', 'flashrank', 'litellm', 'rrf'"
        )
