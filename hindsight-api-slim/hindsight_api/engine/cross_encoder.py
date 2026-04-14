"""
Cross-encoder abstraction for reranking.

Provides an interface for reranking with different backends.

Configuration via environment variables - see hindsight_api.config for all env var names.
"""

import asyncio
import logging
import os
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

import httpx

from ..config import (
    DEFAULT_LITELLM_API_BASE,
    DEFAULT_RERANKER_COHERE_MODEL,
    DEFAULT_RERANKER_FLASHRANK_CACHE_DIR,
    DEFAULT_RERANKER_FLASHRANK_MODEL,
    DEFAULT_RERANKER_GOOGLE_MODEL,
    DEFAULT_RERANKER_LITELLM_MAX_TOKENS_PER_DOC,
    DEFAULT_RERANKER_LITELLM_MODEL,
    DEFAULT_RERANKER_LITELLM_SDK_MODEL,
    DEFAULT_RERANKER_LOCAL_BATCH_SIZE,
    DEFAULT_RERANKER_LOCAL_FORCE_CPU,
    DEFAULT_RERANKER_LOCAL_MAX_CONCURRENT,
    DEFAULT_RERANKER_LOCAL_MODEL,
    DEFAULT_RERANKER_LOCAL_TRUST_REMOTE_CODE,
    DEFAULT_RERANKER_PROVIDER,
    DEFAULT_RERANKER_SILICONFLOW_BASE_URL,
    DEFAULT_RERANKER_SILICONFLOW_MODEL,
    DEFAULT_RERANKER_TEI_BATCH_SIZE,
    DEFAULT_RERANKER_TEI_MAX_CONCURRENT,
    DEFAULT_RERANKER_ZEROENTROPY_MODEL,
    ENV_RERANKER_COHERE_API_KEY,
    ENV_RERANKER_COHERE_MODEL,
    ENV_RERANKER_FLASHRANK_CACHE_DIR,
    ENV_RERANKER_FLASHRANK_MODEL,
    ENV_RERANKER_GOOGLE_PROJECT_ID,
    ENV_RERANKER_LITELLM_SDK_API_KEY,
    ENV_RERANKER_LOCAL_FORCE_CPU,
    ENV_RERANKER_LOCAL_MAX_CONCURRENT,
    ENV_RERANKER_LOCAL_MODEL,
    ENV_RERANKER_LOCAL_TRUST_REMOTE_CODE,
    ENV_RERANKER_PROVIDER,
    ENV_RERANKER_SILICONFLOW_API_KEY,
    ENV_RERANKER_TEI_BATCH_SIZE,
    ENV_RERANKER_TEI_MAX_CONCURRENT,
    ENV_RERANKER_TEI_URL,
    ENV_RERANKER_ZEROENTROPY_API_KEY,
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

    def __init__(
        self,
        model_name: str | None = None,
        max_concurrent: int = 4,
        force_cpu: bool = False,
        trust_remote_code: bool = False,
        fp16: bool = False,
        bucket_batching: bool = False,
        batch_size: int = DEFAULT_RERANKER_LOCAL_BATCH_SIZE,
    ):
        """
        Initialize local SentenceTransformers cross-encoder.

        Args:
            model_name: Name of the CrossEncoder model to use.
                       Default: cross-encoder/ms-marco-MiniLM-L-6-v2
            max_concurrent: Maximum concurrent reranking calls (default: 2).
                           Higher values may cause CPU thrashing under load.
            force_cpu: Force CPU mode (avoids MPS/XPC issues on macOS in daemon mode).
                      Default: False
            trust_remote_code: Allow loading models with custom code (security risk).
                              Required for some models like jina-reranker-v2-base-multilingual.
                              Default: False (disabled for security)
            fp16: Use FP16 (half precision) inference. Faster on MPS and CUDA,
                  may be slower on CPU. Default: False (opt-in via env var).
            bucket_batching: Sort pairs by token length before batching to reduce
                            padding waste. 36-54% speedup, quality-identical.
                            Default: False (opt-in via env var).
            batch_size: Batch size for predict() calls. Optimal values vary by
                       hardware and model (MPS: 32, CUDA: 128+). Default: 32.
        """
        self.model_name = model_name or DEFAULT_RERANKER_LOCAL_MODEL
        self.force_cpu = force_cpu
        self.trust_remote_code = trust_remote_code
        self.fp16 = fp16
        self.bucket_batching = bucket_batching
        self.batch_size = batch_size
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

        # Patch transformers 5.x compatibility for models using XLM-RoBERTa
        # (e.g., jina-reranker-v2-base-multilingual). transformers 5.x removed
        # create_position_ids_from_input_ids as a module-level function; the custom
        # code in these models still references it. This monkey-patch restores it.
        try:
            import transformers.models.xlm_roberta.modeling_xlm_roberta as xlm_module
            from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaEmbeddings

            if not hasattr(xlm_module, "create_position_ids_from_input_ids"):
                setattr(
                    xlm_module,
                    "create_position_ids_from_input_ids",
                    XLMRobertaEmbeddings.create_position_ids_from_input_ids,
                )
                logger.info("Reranker: applied transformers 5.x compatibility patch for XLM-RoBERTa")
        except Exception:
            pass

        # Suppress verbose transformers warnings during model loading
        # This suppresses the "UNEXPECTED" warnings from CrossEncoder which are harmless
        # but look alarming to users (e.g., "embeddings.position_ids | UNEXPECTED")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*was not found in model state dict.*")
            warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")

            # Also suppress transformers library logging temporarily
            transformers_logger = logging.getLogger("transformers")
            original_level = transformers_logger.level
            transformers_logger.setLevel(logging.ERROR)

            try:
                self._model = CrossEncoder(
                    self.model_name,
                    device=device,
                    model_kwargs={"low_cpu_mem_usage": False},
                    trust_remote_code=self.trust_remote_code,
                )
            finally:
                # Restore original logging level
                transformers_logger.setLevel(original_level)

        # FP16 inference: convert model weights to half precision.
        # Empirically validated: 27-36% faster on MPS, quality-identical (20/20 overlap).
        if self.fp16 and device != "cpu":
            self._model.model.half()
            logger.info("Reranker: FP16 inference enabled")

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
        """Synchronous prediction wrapper for thread pool execution.

        Supports two optimizations (controlled via .env):
        - bucket_batching: sort pairs by token length to reduce padding waste (36-54% speedup)
        - batch_size: explicit batch size for predict() calls (MPS optimal: 32)
        """
        import numpy as np

        if self.bucket_batching and len(pairs) > 1:
            # Sort pairs by approximate token length to create homogeneous batches.
            # This eliminates padding waste — short pairs aren't padded to the length
            # of the longest pair in the batch. Quality-identical by construction.
            lengths = [len(pairs[i][0]) + len(pairs[i][1]) for i in range(len(pairs))]
            sorted_indices = sorted(range(len(pairs)), key=lambda i: lengths[i])
            sorted_pairs = [pairs[i] for i in sorted_indices]

            sorted_scores = self._model.predict(sorted_pairs, batch_size=self.batch_size, show_progress_bar=False)
            sorted_scores = sorted_scores.tolist() if hasattr(sorted_scores, "tolist") else list(sorted_scores)

            # Restore original order
            scores = [0.0] * len(pairs)
            for new_pos, orig_idx in enumerate(sorted_indices):
                scores[orig_idx] = sorted_scores[new_pos]
            return scores

        scores = self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
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


class _CohereCompatibleRerankClient:
    """
    Internal HTTP client for Cohere-compatible /rerank endpoints.

    Shared by all providers that speak the Cohere rerank wire format —
    {model, query, documents[, top_n]} request and
    {results: [{index, relevance_score}, ...]} response. This covers
    SiliconFlow, ZeroEntropy, Jina, Voyage, BGE self-hosted, and Cohere
    itself when reached via a custom base_url (e.g. Azure AI Foundry).

    Not a CrossEncoderModel — providers compose it and expose their own
    provider_name / initialization logging.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        rerank_url: str,
        timeout: float = 60.0,
        include_top_n: bool = True,
    ):
        self.api_key = api_key
        self.model = model
        self.rerank_url = rerank_url
        self.timeout = timeout
        self.include_top_n = include_top_n
        self._async_client: httpx.AsyncClient | None = None

    async def initialize(self) -> None:
        if self._async_client is not None:
            return
        self._async_client = httpx.AsyncClient(
            timeout=self.timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        if self._async_client is None:
            raise RuntimeError("Reranker not initialized. Call initialize() first.")

        if not pairs:
            return []

        query_groups: dict[str, list[tuple[int, str]]] = {}
        for idx, (query, text) in enumerate(pairs):
            query_groups.setdefault(query, []).append((idx, text))

        all_scores = [0.0] * len(pairs)

        for query, indexed_texts in query_groups.items():
            texts = [text for _, text in indexed_texts]
            indices = [idx for idx, _ in indexed_texts]

            body: dict[str, object] = {
                "model": self.model,
                "query": query,
                "documents": texts,
                "return_documents": False,
            }
            if self.include_top_n:
                body["top_n"] = len(texts)

            response = await self._async_client.post(self.rerank_url, json=body)
            response.raise_for_status()
            result = response.json()

            for item in result.get("results", []):
                original_idx = item["index"]
                score = item["relevance_score"]
                all_scores[indices[original_idx]] = score

        return all_scores


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
        # Used when base_url is set (Azure AI Foundry and other Cohere-compatible hosts).
        # Azure endpoints already include the full invoke path, so rerank_url == base_url
        # and top_n is omitted to match the existing Azure contract.
        self._http_client: _CohereCompatibleRerankClient | None = (
            _CohereCompatibleRerankClient(
                api_key=api_key,
                model=model,
                rerank_url=base_url,
                timeout=timeout,
                include_top_n=False,
            )
            if base_url
            else None
        )

    @property
    def provider_name(self) -> str:
        return "cohere"

    async def initialize(self) -> None:
        """Initialize the Cohere client."""
        if self._client is not None or (self._http_client and self._http_client._async_client):
            return

        base_url_msg = f" at {self.base_url}" if self.base_url else ""
        logger.info(f"Reranker: initializing Cohere provider with model {self.model}{base_url_msg}")

        if self._http_client is not None:
            await self._http_client.initialize()
            logger.info("Reranker: Cohere provider initialized (Cohere-compatible HTTP endpoint)")
        else:
            # For native Cohere API, use the official SDK
            try:
                import cohere
            except ImportError:
                raise ImportError("cohere is required for CohereCrossEncoder. Install it with: pip install cohere")

            self._client = cohere.Client(api_key=self.api_key, timeout=self.timeout)
            logger.info("Reranker: Cohere provider initialized")

    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """
        Score query-document pairs using the Cohere Rerank API.

        Args:
            pairs: List of (query, document) tuples to score

        Returns:
            List of relevance scores
        """
        if self._client is None and self._http_client is None:
            raise RuntimeError("Reranker not initialized. Call initialize() first.")

        if not pairs:
            return []

        if self._http_client is not None:
            return await self._http_client.predict(pairs)

        # Run sync Cohere SDK calls in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._predict_sync_sdk, pairs)

    def _predict_sync_sdk(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Synchronous predict using the native Cohere SDK."""
        query_groups: dict[str, list[tuple[int, str]]] = {}
        for idx, (query, text) in enumerate(pairs):
            query_groups.setdefault(query, []).append((idx, text))

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

            for result in response.results:
                original_idx = result.index
                score = result.relevance_score
                all_scores[indices[original_idx]] = score

        return all_scores


class ZeroEntropyCrossEncoder(CrossEncoderModel):
    """
    ZeroEntropy cross-encoder implementation using the ZeroEntropy Rerank API.

    Supports zerank-2 (flagship) and zerank-2-small models.
    See: https://docs.zeroentropy.dev/models
    """

    DEFAULT_BASE_URL = "https://api.zeroentropy.dev"
    RERANK_PATH = "/v1/models/rerank"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_RERANKER_ZEROENTROPY_MODEL,
        base_url: str | None = None,
        timeout: float = 60.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/") if base_url else self.DEFAULT_BASE_URL
        self._client = _CohereCompatibleRerankClient(
            api_key=api_key,
            model=model,
            rerank_url=f"{self.base_url}{self.RERANK_PATH}",
            timeout=timeout,
        )

    @property
    def provider_name(self) -> str:
        return "zeroentropy"

    async def initialize(self) -> None:
        if self._client._async_client is not None:
            return
        logger.info(f"Reranker: initializing ZeroEntropy provider with model {self.model}")
        await self._client.initialize()
        logger.info("Reranker: ZeroEntropy provider initialized")

    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return await self._client.predict(pairs)


class SiliconFlowCrossEncoder(CrossEncoderModel):
    """
    SiliconFlow cross-encoder implementation.

    SiliconFlow (https://siliconflow.cn) exposes a Cohere-compatible /rerank
    endpoint. Shares the HTTP client with ZeroEntropy/Cohere-custom-endpoint
    via _CohereCompatibleRerankClient.
    """

    RERANK_PATH = "/rerank"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_RERANKER_SILICONFLOW_MODEL,
        base_url: str = DEFAULT_RERANKER_SILICONFLOW_BASE_URL,
        timeout: float = 60.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = _CohereCompatibleRerankClient(
            api_key=api_key,
            model=model,
            rerank_url=f"{self.base_url}{self.RERANK_PATH}",
            timeout=timeout,
        )

    @property
    def provider_name(self) -> str:
        return "siliconflow"

    async def initialize(self) -> None:
        if self._client._async_client is not None:
            return
        logger.info(f"Reranker: initializing SiliconFlow provider at {self.base_url} with model {self.model}")
        await self._client.initialize()
        logger.info("Reranker: SiliconFlow provider initialized")

    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return await self._client.predict(pairs)


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


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to at most max_tokens using the shared tiktoken encoder."""
    from .memory_engine import _get_tiktoken_encoding

    enc = _get_tiktoken_encoding()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


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
        max_tokens_per_doc: int | None = DEFAULT_RERANKER_LITELLM_MAX_TOKENS_PER_DOC,
    ):
        """
        Initialize LiteLLM cross-encoder client.

        Args:
            api_base: Base URL of the LiteLLM proxy (default: http://localhost:4000)
            api_key: API key for the LiteLLM proxy (optional, depends on proxy config)
            model: Reranking model name (default: cohere/rerank-english-v3.0)
                   Use provider prefix (e.g., cohere/, together_ai/, voyage/)
            timeout: Request timeout in seconds (default: 60.0)
            max_tokens_per_doc: If set, truncate each document to this many tokens before
                                sending to the reranker (uses tiktoken cl100k_base encoding).
                                Useful for models with small context windows (e.g. 1024 tokens).
        """
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_tokens_per_doc = max_tokens_per_doc
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
            if self.max_tokens_per_doc is not None:
                texts = [_truncate_to_tokens(t, self.max_tokens_per_doc) for t in texts]
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


class LiteLLMSDKCrossEncoder(CrossEncoderModel):
    """
    LiteLLM SDK cross-encoder for direct API integration.

    Supports reranking via LiteLLM SDK without requiring a proxy server.
    Supported providers: Cohere, DeepInfra, Together AI, HuggingFace, Jina AI, Voyage AI, AWS Bedrock.

    Example model names:
    - cohere/rerank-english-v3.0
    - deepinfra/Qwen3-reranker-8B
    - together_ai/Salesforce/Llama-Rank-V1
    - huggingface/BAAI/bge-reranker-v2-m3
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_RERANKER_LITELLM_SDK_MODEL,
        api_base: str | None = None,
        timeout: float = 60.0,
        max_tokens_per_doc: int | None = DEFAULT_RERANKER_LITELLM_MAX_TOKENS_PER_DOC,
    ):
        """
        Initialize LiteLLM SDK cross-encoder client.

        Args:
            api_key: API key for the reranking provider
            model: Model name with provider prefix (e.g., "deepinfra/Qwen3-reranker-8B")
            api_base: Custom base URL for API (optional)
            timeout: Request timeout in seconds (default: 60.0)
            max_tokens_per_doc: If set, truncate each document to this many tokens before
                                sending to the reranker (uses tiktoken cl100k_base encoding).
                                Useful for models with small context windows (e.g. 1024 tokens).
        """
        self.api_key = api_key
        self.model = model
        self.api_base = api_base
        self.timeout = timeout
        self.max_tokens_per_doc = max_tokens_per_doc
        self._initialized = False
        self._litellm = None  # Will be set during initialization

    @property
    def provider_name(self) -> str:
        return "litellm-sdk"

    async def initialize(self) -> None:
        """Initialize the LiteLLM SDK client."""
        if self._initialized:
            return

        try:
            import litellm

            self._litellm = litellm  # Store reference
        except ImportError:
            raise ImportError("litellm is required for LiteLLMSDKCrossEncoder. Install it with: pip install litellm")

        api_base_msg = f" at {self.api_base}" if self.api_base else ""
        logger.info(f"Reranker: initializing LiteLLM SDK provider with model {self.model}{api_base_msg}")

        self._initialized = True
        logger.info("Reranker: LiteLLM SDK provider initialized")

    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """
        Score query-document pairs using the LiteLLM SDK.

        Args:
            pairs: List of (query, document) tuples to score

        Returns:
            List of relevance scores
        """
        if not self._initialized:
            raise RuntimeError("Reranker not initialized. Call initialize() first.")

        if not pairs:
            return []

        # Group pairs by query for efficient batching
        # LiteLLM rerank expects one query with multiple documents
        query_groups: dict[str, list[tuple[int, str]]] = {}
        for idx, (query, text) in enumerate(pairs):
            if query not in query_groups:
                query_groups[query] = []
            query_groups[query].append((idx, text))

        all_scores = [0.0] * len(pairs)

        for query, indexed_texts in query_groups.items():
            texts = [text for _, text in indexed_texts]
            if self.max_tokens_per_doc is not None:
                texts = [_truncate_to_tokens(t, self.max_tokens_per_doc) for t in texts]
            indices = [idx for idx, _ in indexed_texts]

            # Build kwargs for rerank call
            rerank_kwargs = {
                "model": self.model,
                "query": query,
                "documents": texts,
                "api_key": self.api_key,
            }
            if self.api_base:
                rerank_kwargs["api_base"] = self.api_base

            response = await self._litellm.arerank(**rerank_kwargs)

            # Map scores back to original positions
            # Response format: RerankResponse with results list
            # Each result is a TypedDict with "index" and "relevance_score"
            if hasattr(response, "results") and response.results:
                for result in response.results:
                    # Results are TypedDicts, use dict-style access
                    original_idx = result["index"]
                    score = result.get("relevance_score", result.get("score", 0.0))
                    all_scores[indices[original_idx]] = score
            elif isinstance(response, list):
                # Direct list of scores (unlikely but defensive)
                for i, score in enumerate(response):
                    all_scores[indices[i]] = score
            else:
                logger.warning(f"Unexpected response format from LiteLLM rerank: {type(response)}")

        return all_scores


class JinaMLXCrossEncoder(CrossEncoderModel):
    """
    Jina Reranker v3 MLX implementation for Apple Silicon.

    Uses jinaai/jina-reranker-v3-mlx — a 0.6B parameter multilingual listwise reranker
    optimized for Apple Silicon via the MLX framework. No transformers/PyTorch dependency.

    The model is downloaded automatically from HuggingFace Hub on first use.
    Requires: mlx>=0.31.0, mlx-lm>=0.31.1, safetensors>=0.6.2
    """

    HF_REPO_ID = "jinaai/jina-reranker-v3-mlx"

    def __init__(self, model_path: str | None = None):
        """
        Args:
            model_path: Local path to the downloaded model directory.
                        If None, the model is downloaded from HuggingFace Hub.
        """
        self.model_path = model_path
        self._reranker = None

    @property
    def provider_name(self) -> str:
        return "jina-mlx"

    async def initialize(self) -> None:
        if self._reranker is not None:
            return

        # Pre-warm transformers.AutoTokenizer to fully populate the transformers
        # namespace before mlx_lm imports it. transformers 5.x uses _LazyModule,
        # which has an unguarded window where `from transformers import AutoTokenizer`
        # raises ImportError if another thread is concurrently initializing the
        # namespace (e.g. embeddings init in an executor thread).
        # See: https://github.com/vectorize-io/hindsight/issues/994
        import transformers

        _ = transformers.AutoTokenizer

        try:
            import mlx.core  # noqa: F401
            import mlx_lm  # noqa: F401
        except ImportError as exc:
            # Only swallow "package not installed" errors. Anything else (e.g. a
            # transitive import failure inside mlx_lm) must surface verbatim so
            # the real cause is debuggable instead of being masked by a generic
            # "install mlx" message.
            msg = str(exc)
            if "mlx" not in msg and "mlx_lm" not in msg:
                raise
            raise ImportError(
                "mlx and mlx-lm are required for JinaMLXCrossEncoder. "
                "Install with: pip install mlx>=0.31.0 mlx-lm>=0.31.1 safetensors>=0.6.2"
            ) from exc

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model)

    def _load_model(self) -> None:
        """Download (if needed) and load the MLX reranker. Runs in a thread."""
        import os

        from huggingface_hub import snapshot_download

        from .jina_mlx_reranker import MLXReranker

        model_path = self.model_path
        if model_path is None:
            logger.info(f"Reranker: downloading {self.HF_REPO_ID} from HuggingFace Hub...")
            model_path = snapshot_download(repo_id=self.HF_REPO_ID)

        logger.info(f"Reranker: loading jina-reranker-v3-mlx from {model_path}")
        self._reranker = MLXReranker(
            model_path=model_path,
            projector_path=os.path.join(model_path, "projector.safetensors"),
        )
        logger.info("Reranker: jina-mlx provider initialized")

    def _predict_sync(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score pairs grouped by query. Runs in a thread."""
        if not pairs:
            return []

        query_groups: dict[str, list[tuple[int, str]]] = {}
        for idx, (query, doc) in enumerate(pairs):
            query_groups.setdefault(query, []).append((idx, doc))

        all_scores = [0.0] * len(pairs)

        for query, indexed_docs in query_groups.items():
            docs = [doc for _, doc in indexed_docs]
            indices = [idx for idx, _ in indexed_docs]
            results = self._reranker.rerank(query, docs)
            for result in results:
                original_idx = result["index"]
                all_scores[indices[original_idx]] = result["relevance_score"]

        return all_scores

    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        if self._reranker is None:
            raise RuntimeError("Reranker not initialized. Call initialize() first.")

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._predict_sync, pairs)


class GoogleCrossEncoder(CrossEncoderModel):
    """
    Google Discovery Engine cross-encoder using the Ranking REST API.

    Uses httpx + google-auth for lightweight REST calls (no gRPC/protobuf).
    Supports ADC (Application Default Credentials) or service account key file.

    Available models:
    - semantic-ranker-default-004: Best quality, 1024 tokens/record (recommended)
    - semantic-ranker-fast-004: Lower latency, 1024 tokens/record

    Max 200 records per API request. Location is always "global".
    """

    MAX_RECORDS_PER_REQUEST = 200
    API_BASE = "https://discoveryengine.googleapis.com/v1"
    SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

    def __init__(
        self,
        project_id: str,
        model: str = DEFAULT_RERANKER_GOOGLE_MODEL,
        service_account_key: str | None = None,
        location: str = "global",
        timeout: float = 60.0,
    ):
        """
        Initialize Google Discovery Engine cross-encoder.

        Args:
            project_id: Google Cloud project ID
            model: Ranking model name (default: semantic-ranker-default-004)
            service_account_key: Path to service account JSON key file.
                                If None, uses Application Default Credentials (ADC).
            location: API location (default: "global")
            timeout: Request timeout in seconds (default: 60.0)
        """
        self.project_id = project_id
        self.model = model
        self.service_account_key = service_account_key
        self.location = location
        self.timeout = timeout
        self._credentials = None
        self._client: httpx.Client | None = None
        self._rank_url: str | None = None

    @property
    def provider_name(self) -> str:
        return "google"

    def _get_auth_headers(self) -> dict[str, str]:
        """Get Authorization header with a fresh access token."""
        import google.auth.transport.requests

        if not self._credentials.valid:
            self._credentials.refresh(google.auth.transport.requests.Request())
        return {"Authorization": f"Bearer {self._credentials.token}"}

    async def initialize(self) -> None:
        """Initialize credentials and HTTP client."""
        if self._client is not None:
            return

        auth_method = "ADC" if not self.service_account_key else "service_account"
        logger.info(
            f"Reranker: initializing Google Discovery Engine provider "
            f"(project={self.project_id}, model={self.model}, auth={auth_method})"
        )
        if self.service_account_key:
            try:
                from google.oauth2 import service_account
            except ImportError:
                raise ImportError(
                    "google-auth is required for GoogleCrossEncoder. Install it with: pip install google-auth"
                )
            self._credentials = service_account.Credentials.from_service_account_file(
                self.service_account_key,
                scopes=self.SCOPES,
            )
        else:
            try:
                import google.auth
            except ImportError:
                raise ImportError(
                    "google-auth is required for GoogleCrossEncoder. Install it with: pip install google-auth"
                )
            self._credentials, _ = google.auth.default(scopes=self.SCOPES)

        ranking_config = f"projects/{self.project_id}/locations/{self.location}/rankingConfigs/default_ranking_config"
        self._rank_url = f"{self.API_BASE}/{ranking_config}:rank"
        self._client = httpx.Client(timeout=self.timeout)

        logger.info("Reranker: Google Discovery Engine provider initialized")

    def _predict_sync(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Synchronous predict via REST API."""
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
            texts = [text for _, text in indexed_texts]
            indices = [idx for idx, _ in indexed_texts]

            # Process in batches of MAX_RECORDS_PER_REQUEST
            for batch_start in range(0, len(texts), self.MAX_RECORDS_PER_REQUEST):
                batch_texts = texts[batch_start : batch_start + self.MAX_RECORDS_PER_REQUEST]
                batch_indices = indices[batch_start : batch_start + self.MAX_RECORDS_PER_REQUEST]

                records = [{"id": str(i), "content": text} for i, text in enumerate(batch_texts)]

                response = self._client.post(
                    self._rank_url,
                    headers=self._get_auth_headers(),
                    json={
                        "model": self.model,
                        "query": query,
                        "records": records,
                        "topN": len(records),
                    },
                )
                response.raise_for_status()
                result = response.json()

                for record in result.get("records", []):
                    local_idx = int(record["id"])
                    all_scores[batch_indices[local_idx]] = record["score"]

        return all_scores

    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """
        Score query-document pairs using Google Discovery Engine Ranking API.

        Args:
            pairs: List of (query, document) tuples to score

        Returns:
            List of relevance scores (0-1, higher = more relevant)
        """
        if self._client is None:
            raise RuntimeError("Reranker not initialized. Call initialize() first.")

        if not pairs:
            return []

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._predict_sync, pairs)


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
            trust_remote_code=config.reranker_local_trust_remote_code,
            fp16=config.reranker_local_fp16,
            bucket_batching=config.reranker_local_bucket_batching,
            batch_size=config.reranker_local_batch_size,
        )
    elif provider == "cohere":
        api_key = config.reranker_cohere_api_key
        if not api_key:
            raise ValueError(f"{ENV_RERANKER_COHERE_API_KEY} is required when {ENV_RERANKER_PROVIDER} is 'cohere'")
        return CohereCrossEncoder(
            api_key=api_key,
            model=config.reranker_cohere_model,
            base_url=config.reranker_cohere_base_url,
        )
    elif provider == "openrouter":
        api_key = config.reranker_openrouter_api_key
        if not api_key:
            raise ValueError(
                "HINDSIGHT_API_RERANKER_OPENROUTER_API_KEY, HINDSIGHT_API_OPENROUTER_API_KEY, "
                f"or HINDSIGHT_API_LLM_API_KEY is required when {ENV_RERANKER_PROVIDER} is 'openrouter'"
            )
        return CohereCrossEncoder(
            api_key=api_key,
            model=config.reranker_openrouter_model,
            base_url="https://openrouter.ai/api/v1/rerank",
        )
    elif provider == "flashrank":
        model = os.environ.get(ENV_RERANKER_FLASHRANK_MODEL, DEFAULT_RERANKER_FLASHRANK_MODEL)
        cache_dir = os.environ.get(ENV_RERANKER_FLASHRANK_CACHE_DIR, DEFAULT_RERANKER_FLASHRANK_CACHE_DIR)
        return FlashRankCrossEncoder(model_name=model, cache_dir=cache_dir)
    elif provider == "litellm":
        return LiteLLMCrossEncoder(
            api_base=config.reranker_litellm_api_base,
            api_key=config.reranker_litellm_api_key,
            model=config.reranker_litellm_model,
            max_tokens_per_doc=config.reranker_litellm_max_tokens_per_doc,
        )
    elif provider == "litellm-sdk":
        api_key = config.reranker_litellm_sdk_api_key
        if not api_key:
            raise ValueError(
                f"{ENV_RERANKER_LITELLM_SDK_API_KEY} is required when {ENV_RERANKER_PROVIDER} is 'litellm-sdk'"
            )
        return LiteLLMSDKCrossEncoder(
            api_key=api_key,
            model=config.reranker_litellm_sdk_model,
            api_base=config.reranker_litellm_sdk_api_base,
            max_tokens_per_doc=config.reranker_litellm_max_tokens_per_doc,
        )
    elif provider == "zeroentropy":
        api_key = config.reranker_zeroentropy_api_key
        if not api_key:
            raise ValueError(
                f"{ENV_RERANKER_ZEROENTROPY_API_KEY} is required when {ENV_RERANKER_PROVIDER} is 'zeroentropy'"
            )
        return ZeroEntropyCrossEncoder(
            api_key=api_key,
            model=config.reranker_zeroentropy_model,
        )
    elif provider == "siliconflow":
        api_key = config.reranker_siliconflow_api_key
        if not api_key:
            raise ValueError(
                f"{ENV_RERANKER_SILICONFLOW_API_KEY} is required when {ENV_RERANKER_PROVIDER} is 'siliconflow'"
            )
        return SiliconFlowCrossEncoder(
            api_key=api_key,
            model=config.reranker_siliconflow_model,
            base_url=config.reranker_siliconflow_base_url,
        )
    elif provider == "google":
        project_id = config.reranker_google_project_id
        if not project_id:
            raise ValueError(
                f"{ENV_RERANKER_GOOGLE_PROJECT_ID} (or HINDSIGHT_API_LLM_VERTEXAI_PROJECT_ID) "
                f"is required when {ENV_RERANKER_PROVIDER} is 'google'"
            )
        return GoogleCrossEncoder(
            project_id=project_id,
            model=config.reranker_google_model,
            service_account_key=config.reranker_google_service_account_key,
        )
    elif provider == "rrf":
        return RRFPassthroughCrossEncoder()
    elif provider == "jina-mlx":
        return JinaMLXCrossEncoder()
    else:
        raise ValueError(
            f"Unknown reranker provider: {provider}. Supported: 'local', 'tei', 'cohere', 'zeroentropy', 'siliconflow', 'google', 'flashrank', 'litellm', 'litellm-sdk', 'rrf', 'jina-mlx'"
        )
