"""
Cross-encoder abstraction for reranking.

Provides an interface for reranking with different backends.

Configuration via environment variables - see hindsight_api.config for all env var names.
"""

import asyncio
import logging
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

from ..config import (
    DEFAULT_LITELLM_API_BASE,
    DEFAULT_RERANKER_ALIBABA_MODEL,
    DEFAULT_RERANKER_COHERE_MODEL,
    DEFAULT_RERANKER_FLASHRANK_BATCH_SIZE,
    DEFAULT_RERANKER_FLASHRANK_CACHE_DIR,
    DEFAULT_RERANKER_FLASHRANK_MODEL,
    DEFAULT_RERANKER_GOOGLE_MODEL,
    DEFAULT_RERANKER_LITELLM_MAX_TOKENS_PER_DOC,
    DEFAULT_RERANKER_LITELLM_MODEL,
    DEFAULT_RERANKER_LITELLM_SDK_MODEL,
    DEFAULT_RERANKER_LOCAL_BATCH_SIZE,
    DEFAULT_RERANKER_LOCAL_MODEL,
    DEFAULT_RERANKER_SILICONFLOW_BASE_URL,
    DEFAULT_RERANKER_SILICONFLOW_MODEL,
    DEFAULT_RERANKER_TEI_BATCH_SIZE,
    DEFAULT_RERANKER_TEI_MAX_CONCURRENT,
    DEFAULT_RERANKER_ZEROENTROPY_MODEL,
    DEFAULT_ZEROENTROPY_BASE_URL,
    RerankerMemberConfig,
)
from .bank_attribution import reranker_bank_attribution_headers
from .local_device import (
    release_local_inference_memory,
    resolve_model_device_type,
    select_local_device,
)
from .tei_retry import TEI_KEEPALIVE_EXPIRY_SECONDS, is_retryable_tei_transport_error, tei_retry_delay

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

    @property
    def blocking_init(self) -> bool:
        """Whether ``initialize()`` blocks the event loop (loads a model in-process).

        Callers run those in a thread pool. Remote providers leave this False, and
        so does :class:`MultiCrossEncoder` — it offloads its own members.
        """
        return False

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
        allow_mps: bool = False,
    ):
        """
        Initialize local SentenceTransformers cross-encoder.

        Args:
            model_name: Name of the CrossEncoder model to use.
                       Default: cross-encoder/ms-marco-MiniLM-L-6-v2
            max_concurrent: Maximum concurrent reranking calls (default: 2).
                           Higher values may cause CPU thrashing under load.
            force_cpu: Force CPU mode for local inference.
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
            allow_mps: Opt in to the Apple Silicon MPS GPU. Disabled by default
                      because MPS leaks memory under variable-length workloads
                      (see engine/local_device.py). Default: False
        """
        self.model_name = model_name or DEFAULT_RERANKER_LOCAL_MODEL
        self.force_cpu = force_cpu
        self.trust_remote_code = trust_remote_code
        self.fp16 = fp16
        self.bucket_batching = bucket_batching
        self.batch_size = batch_size
        self.allow_mps = allow_mps
        self._model = None
        self._device_type: str = "cpu"
        LocalSTCrossEncoder._max_concurrent = max_concurrent

    @property
    def provider_name(self) -> str:
        return "local"

    @property
    def blocking_init(self) -> bool:
        return True

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

        # Determine device based on hardware availability. We always set
        # low_cpu_mem_usage=False to prevent lazy loading (meta tensors) which can
        # cause issues when accelerate is installed but no GPU is available.
        # Note: We do NOT use device_map because CrossEncoder internally calls .to(device)
        # after loading, which conflicts with accelerate's device_map handling.
        # MPS is opt-in (allow_mps) — see engine/local_device.py for why.
        device = select_local_device(self.force_cpu, self.allow_mps)

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

        self._device_type = resolve_model_device_type(self._model)

        # FP16 inference: convert model weights to half precision.
        # Empirically validated: 27-36% faster on MPS, quality-identical (20/20 overlap).
        if self.fp16 and self._device_type != "cpu":
            self._model.model.half()
            logger.info("Reranker: FP16 inference enabled")

        # Initialize shared executor (limited workers naturally limits concurrency)
        if LocalSTCrossEncoder._executor is None:
            LocalSTCrossEncoder._executor = ThreadPoolExecutor(
                max_workers=LocalSTCrossEncoder._max_concurrent,
                thread_name_prefix="reranker",
            )
            logger.info(
                f"Reranker: local provider initialized "
                f"(device: {self._device_type}, max_concurrent={LocalSTCrossEncoder._max_concurrent})"
            )
        else:
            logger.info(f"Reranker: local provider initialized (device: {self._device_type}, using existing executor)")

    def _predict_sync(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Synchronous prediction wrapper for thread pool execution.

        Supports two optimizations (controlled via .env):
        - bucket_batching: sort pairs by token length to reduce padding waste (36-54% speedup)
        - batch_size: explicit batch size for predict() calls (MPS optimal: 32)
        """

        try:
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
        finally:
            release_local_inference_memory(self._device_type)

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
                except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout) as e:
                    last_error = e
                    if attempt < self.max_retries:
                        logger.warning(
                            f"TEI request failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                            f"Retrying in {delay}s..."
                        )
                        await asyncio.sleep(delay)
                        delay *= 2  # Exponential backoff
                except httpx.RequestError as e:
                    if not is_retryable_tei_transport_error(e):
                        raise
                    last_error = e
                    if attempt < self.max_retries:
                        logger.warning(
                            f"TEI request failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                            f"Retrying in {delay}s..."
                        )
                        await asyncio.sleep(delay)
                        delay *= 2  # Exponential backoff
                except OSError as e:
                    if not is_retryable_tei_transport_error(e):
                        raise
                    last_error = e
                    if attempt < self.max_retries:
                        logger.warning(
                            f"TEI request failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                            f"Retrying in {delay}s..."
                        )
                        await asyncio.sleep(delay)
                        delay *= 2  # Exponential backoff
                except httpx.HTTPStatusError as e:
                    # TEI uses 429 as normal overload backpressure. Retry it with
                    # the same bounded budget as transient server errors.
                    if (e.response.status_code == 429 or e.response.status_code >= 500) and attempt < self.max_retries:
                        last_error = e
                        sleep_delay = tei_retry_delay(
                            e.response,
                            delay,
                            request_timeout=self.timeout,
                        )
                        logger.warning(
                            f"TEI transient error (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                            f"Retrying in {sleep_delay:.2f}s..."
                        )
                        await asyncio.sleep(sleep_delay)
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
        self._async_client = httpx.AsyncClient(
            timeout=self.timeout,
            limits=httpx.Limits(keepalive_expiry=min(self.timeout, TEI_KEEPALIVE_EXPIRY_SECONDS)),
        )

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
                headers=reranker_bank_attribution_headers(),
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
        include_return_documents: bool = False,
    ):
        self.api_key = api_key
        self.model = model
        self.rerank_url = rerank_url
        self.timeout = timeout
        self.include_top_n = include_top_n
        self.include_return_documents = include_return_documents
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

            response = await self._async_client.post(
                self.rerank_url,
                headers=reranker_bank_attribution_headers(),
                json=body,
            )
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

    DEFAULT_BASE_URL = DEFAULT_ZEROENTROPY_BASE_URL
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
        cpu_mem_arena: bool = False,
        batch_size: int = DEFAULT_RERANKER_FLASHRANK_BATCH_SIZE,
    ):
        """
        Initialize FlashRank cross-encoder.

        Args:
            model_name: FlashRank model name. Default: ms-marco-MiniLM-L-12-v2
            cache_dir: Directory to cache downloaded models. Default: system cache
            max_length: Maximum sequence length for reranking. Default: 512
            max_concurrent: Maximum concurrent reranking calls. Default: 4
            cpu_mem_arena: Enable ONNX Runtime CPU memory arena. Default: False.
                          When True, ONNX pre-allocates a memory arena that never
                          shrinks, causing RSS to grow monotonically. False trades
                          slightly slower per-call allocation for bounded RSS.
            batch_size: Passages per forward pass. Default: 32. See
                        ``_predict_sync`` for why this must stay bounded.
        """
        self.model_name = model_name or DEFAULT_RERANKER_FLASHRANK_MODEL
        self.cache_dir = cache_dir or DEFAULT_RERANKER_FLASHRANK_CACHE_DIR
        self.max_length = max_length
        self.cpu_mem_arena = cpu_mem_arena
        # A non-positive size would mean "one pass for everything", which is the
        # unbounded behaviour this batching exists to prevent.
        self.batch_size = max(1, batch_size)
        self._ranker = None
        self._device_type: str = "cpu"  # FlashRank runs on CPU via ONNX Runtime
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

        logger.info(
            f"Reranker: initializing FlashRank provider with model {self.model_name}"
            f" (cpu_mem_arena={self.cpu_mem_arena})"
        )

        # Configure ONNX session options before Ranker creates the session.
        # When cpu_mem_arena=False (default), ONNX won't pre-allocate an arena
        # that grows monotonically, keeping RSS bounded after rerank batches.
        if not self.cpu_mem_arena:
            import onnxruntime as ort

            session_options = ort.SessionOptions()
            session_options.enable_cpu_mem_arena = False
        else:
            session_options = None

        # Initialize ranker with optional cache directory
        ranker_kwargs: dict = {"model_name": self.model_name, "max_length": self.max_length}
        if self.cache_dir:
            ranker_kwargs["cache_dir"] = self.cache_dir

        self._ranker = Ranker(**ranker_kwargs)

        # Patch the ONNX session options if arena is disabled.
        # FlashRank's Ranker doesn't expose SessionOptions in its API,
        # so we replace the session after initialization.
        if session_options is not None and hasattr(self._ranker, "session"):
            import onnxruntime as ort

            model_file = None
            model_dir = getattr(self._ranker, "model_dir", None)
            if model_dir:
                from pathlib import Path

                for candidate in Path(model_dir).glob("*.onnx"):
                    model_file = str(candidate)
                    break
            if model_file:
                self._ranker.session = ort.InferenceSession(model_file, sess_options=session_options)
                logger.info("Reranker: replaced FlashRank ONNX session with cpu_mem_arena=False")

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
        """Synchronous predict — each query group, in bounded batches.

        FlashRank scores every passage of a request in one ONNX forward pass, and
        that pass allocates attention tensors sized ``batch * heads * seq^2``. At
        the default reranker candidate cap that is gigabytes per call, which OOM-
        killed containers on large banks (issue #3355): the burst scales with the
        candidate pool the retrieval arms produce, not with how much work the
        caller asked for. FlashRank also pads a request to its longest passage, so
        one long candidate inflates the sequence length for every other one.

        Splitting into ``batch_size`` chunks bounds the peak the same way the
        local and TEI providers already do. Scores are identical either way —
        passages are scored independently, so batching changes only the
        allocation profile.
        """
        if not pairs:
            return []

        from flashrank import RerankRequest

        try:
            # Group pairs by query
            query_groups: dict[str, list[tuple[int, str]]] = {}
            for idx, (query, text) in enumerate(pairs):
                if query not in query_groups:
                    query_groups[query] = []
                query_groups[query].append((idx, text))

            all_scores = [0.0] * len(pairs)

            for query, indexed_texts in query_groups.items():
                global_indices = [idx for idx, _ in indexed_texts]

                for start in range(0, len(indexed_texts), self.batch_size):
                    batch = indexed_texts[start : start + self.batch_size]

                    # Build passages list for FlashRank. Ids are batch-local, so
                    # `start` shifts them back onto the query group's indices.
                    passages = [{"id": i, "text": text} for i, (_, text) in enumerate(batch)]

                    # Create rerank request
                    request = RerankRequest(query=query, passages=passages)
                    results = self._ranker.rerank(request)

                    # Map scores back to original positions
                    for result in results:
                        local_idx = result["id"]
                        score = result["score"]
                        global_idx = global_indices[start + local_idx]
                        all_scores[global_idx] = score

            return all_scores
        finally:
            release_local_inference_memory(self._device_type)

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
    """Truncate text to at most max_tokens using the shared encoder.

    Reranking truncates every candidate document, and the overwhelming majority
    already fit — so this counts first and only builds the ids for the ones that
    actually need cutting.
    """
    from .token_encoding import truncate_to_tokens

    return truncate_to_tokens(text, max_tokens).text


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
                                sending to the reranker (uses the configured encoding).
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
                headers=reranker_bank_attribution_headers(),
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
        api_key: str | None = None,
        model: str = DEFAULT_RERANKER_LITELLM_SDK_MODEL,
        api_base: str | None = None,
        timeout: float = 60.0,
        max_tokens_per_doc: int | None = DEFAULT_RERANKER_LITELLM_MAX_TOKENS_PER_DOC,
    ):
        """
        Initialize LiteLLM SDK cross-encoder client.

        Args:
            api_key: API key for the reranking provider (optional — omit for
                     providers that use ambient credentials, e.g. AWS Bedrock with IAM)
            model: Model name with provider prefix (e.g., "deepinfra/Qwen3-reranker-8B")
            api_base: Custom base URL for API (optional)
            timeout: Request timeout in seconds (default: 60.0)
            max_tokens_per_doc: If set, truncate each document to this many tokens before
                                sending to the reranker (uses the configured encoding).
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
            rerank_kwargs: dict[str, Any] = {
                "model": self.model,
                "query": query,
                "documents": texts,
                "headers": reranker_bank_attribution_headers(),
            }
            if self.api_key:
                rerank_kwargs["api_key"] = self.api_key
            if self.api_base:
                rerank_kwargs["api_base"] = self.api_base

            response = await self._litellm.arerank(**rerank_kwargs)

            for result in response.results:
                original_idx = result["index"]
                all_scores[indices[original_idx]] = result["relevance_score"]

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
        import threading

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
        # MLX Metal GPU ops are not thread-safe — concurrent calls to
        # Device::end_encoding() crash with SIGSEGV (NULL deref).
        # Serialize all reranker inference through this lock.
        self._mlx_lock = threading.Lock()
        logger.info("Reranker: jina-mlx provider initialized")

    def _predict_sync(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score pairs grouped by query. Runs in a thread."""
        if not pairs:
            return []

        query_groups: dict[str, list[tuple[int, str]]] = {}
        for idx, (query, doc) in enumerate(pairs):
            query_groups.setdefault(query, []).append((idx, doc))

        all_scores = [0.0] * len(pairs)

        with self._mlx_lock:
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


class AlibabaCloudCrossEncoder(CrossEncoderModel):
    """
    Alibaba Cloud DashScope text reranking API.

    Uses the Cohere-compatible /reranks endpoint, which is the standard interface
    for qwen3-rerank. Authentication via HINDSIGHT_API_RERANKER_ALIBABA_API_KEY
    (or DASHSCOPE_API_KEY as a fallback).
    See: https://help.aliyun.com/zh/model-studio/text-rerank-api
    """

    RERANK_URL = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_RERANKER_ALIBABA_MODEL,
        timeout: float = 60.0,
    ):
        self.model = model
        self._client = _CohereCompatibleRerankClient(
            api_key=api_key,
            model=model,
            rerank_url=self.RERANK_URL,
            timeout=timeout,
            include_return_documents=False,
        )

    @property
    def provider_name(self) -> str:
        return "alibaba"

    async def initialize(self) -> None:
        if self._client._async_client is not None:
            return
        logger.info(f"Reranker: initializing Alibaba Cloud provider with model {self.model}")
        await self._client.initialize()
        logger.info("Reranker: Alibaba Cloud provider initialized")

    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return await self._client.predict(pairs)


class MultiCrossEncoder(CrossEncoderModel):
    """Failover across an ordered chain of cross-encoders.

    Member 0 is the primary (the unindexed ``HINDSIGHT_API_RERANKER_*`` config);
    members 1..N are the indexed fallbacks. Each ``predict`` tries members in order
    and returns the first usable set of scores, so an unreachable reranker costs
    ranking quality (whatever the next member gives) instead of the whole recall.
    Put ``rrf`` last to degrade to the fusion order rather than failing.

    Each member keeps its own retry budget, so we only advance after a member has
    exhausted its retries and raised. A member that fails to initialize is not
    fatal — that is the point of the chain — it is retried lazily on the next
    request that reaches it.
    """

    def __init__(self, members: list[CrossEncoderModel]) -> None:
        if len(members) < 2:
            raise ValueError("MultiCrossEncoder requires at least two members")
        self._members = members
        self._ready = [False] * len(members)
        self._locks = [asyncio.Lock() for _ in members]
        self._active = 0

    @property
    def provider_name(self) -> str:
        """The provider of the member that last served a request (primary before any).

        Callers use this to detect a passthrough reranker, so it has to track the
        member actually serving rather than name the chain: a chain that has
        degraded to its ``rrf`` member is passthrough. Concurrent requests share it,
        so a request that fails over can briefly mislabel a neighbour — this only
        tunes downstream scoring, never correctness.
        """
        return self._members[self._active].provider_name

    async def _initialize_member(self, index: int) -> None:
        """Initialize one member, off the event loop when it loads a model in-process."""
        member = self._members[index]
        if member.blocking_init:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: asyncio.run(member.initialize()))
        else:
            await member.initialize()
        self._ready[index] = True

    async def _ensure_member_ready(self, index: int) -> None:
        async with self._locks[index]:
            if not self._ready[index]:
                await self._initialize_member(index)

    async def initialize(self) -> None:
        """Initialize every member, tolerating members that are down.

        Members initialize concurrently so one unreachable member cannot eat the
        startup budget the others need. Failures are logged and retried on use.
        """
        results = await asyncio.gather(
            *(self._ensure_member_ready(i) for i in range(len(self._members))),
            return_exceptions=True,
        )
        for index, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.warning(
                    "Reranker member %d (%s) failed to initialize: %s; it will be retried on use",
                    index,
                    self._members[index].provider_name,
                    result,
                )
        if not any(self._ready):
            logger.error("Reranker: no member of the failover chain initialized; recall will retry them per request")

    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score ``pairs`` with the first member that answers usably."""
        last_exc: BaseException | None = None
        for index, member in enumerate(self._members):
            try:
                if not self._ready[index]:
                    await self._ensure_member_ready(index)
                scores = await member.predict(pairs)
                if len(scores) != len(pairs):
                    raise RuntimeError(f"returned {len(scores)} scores for {len(pairs)} pairs")
            except Exception as e:  # noqa: BLE001 - re-raised below if no member answers
                last_exc = e
                remaining = len(self._members) - index - 1
                logger.warning(
                    "Reranker member %d (%s) failed: %s%s",
                    index,
                    member.provider_name,
                    e,
                    f"; trying next member ({remaining} left)" if remaining else "; no members left",
                )
                continue
            if index != self._active:
                logger.info(
                    "Reranker: now serving from member %d (%s)",
                    index,
                    member.provider_name,
                )
            self._active = index
            return scores
        # All members failed; surface the last error (loop ran at least once).
        assert last_exc is not None
        raise last_exc


def create_cross_encoder(member: RerankerMemberConfig) -> CrossEncoderModel:
    """
    Create a CrossEncoderModel for one member of the reranker chain.

    ``member`` is the primary (index 0, the unindexed ``HINDSIGHT_API_RERANKER_*``
    config) or an indexed fallback. Missing-setting errors name the member's own
    env var, so a chain misconfiguration points at the exact indexed variable.

    Args:
        member: Resolved settings for this member

    Returns:
        Configured CrossEncoderModel instance
    """
    provider = member.provider.lower()

    if provider == "tei":
        url = member.tei_url
        if not url:
            raise ValueError(f"{member.env_name('TEI_URL')} is required when {member.env_name('PROVIDER')} is 'tei'")
        return RemoteTEICrossEncoder(
            base_url=url,
            timeout=member.tei_http_timeout,
            batch_size=member.tei_batch_size,
            max_concurrent=member.tei_max_concurrent,
        )
    elif provider == "local":
        return LocalSTCrossEncoder(
            model_name=member.local_model,
            max_concurrent=member.local_max_concurrent,
            force_cpu=member.local_force_cpu,
            trust_remote_code=member.local_trust_remote_code,
            fp16=member.local_fp16,
            bucket_batching=member.local_bucket_batching,
            batch_size=member.local_batch_size,
            allow_mps=member.local_allow_mps,
        )
    elif provider == "cohere":
        api_key = member.cohere_api_key
        if not api_key:
            raise ValueError(
                f"{member.env_name('COHERE_API_KEY')} is required when {member.env_name('PROVIDER')} is 'cohere'"
            )
        return CohereCrossEncoder(
            api_key=api_key,
            model=member.cohere_model,
            base_url=member.cohere_base_url,
            timeout=member.cohere_timeout,
        )
    elif provider == "openrouter":
        api_key = member.openrouter_api_key
        if not api_key:
            shared = ", HINDSIGHT_API_OPENROUTER_API_KEY, or HINDSIGHT_API_LLM_API_KEY" if member.index == 0 else ""
            raise ValueError(
                f"{member.env_name('OPENROUTER_API_KEY')}{shared} is required "
                f"when {member.env_name('PROVIDER')} is 'openrouter'"
            )
        return CohereCrossEncoder(
            api_key=api_key,
            model=member.openrouter_model,
            base_url=member.openrouter_base_url,
            timeout=member.openrouter_timeout,
        )
    elif provider == "flashrank":
        return FlashRankCrossEncoder(
            model_name=member.flashrank_model,
            cache_dir=member.flashrank_cache_dir,
            cpu_mem_arena=member.flashrank_cpu_mem_arena,
            batch_size=member.flashrank_batch_size,
        )
    elif provider == "litellm":
        return LiteLLMCrossEncoder(
            api_base=member.litellm_api_base,
            api_key=member.litellm_api_key,
            model=member.litellm_model,
            max_tokens_per_doc=member.litellm_max_tokens_per_doc,
            timeout=member.litellm_timeout,
        )
    elif provider == "litellm-sdk":
        return LiteLLMSDKCrossEncoder(
            api_key=member.litellm_sdk_api_key or None,
            model=member.litellm_sdk_model,
            api_base=member.litellm_sdk_api_base,
            max_tokens_per_doc=member.litellm_max_tokens_per_doc,
            timeout=member.litellm_sdk_timeout,
        )
    elif provider == "zeroentropy":
        api_key = member.zeroentropy_api_key
        if not api_key:
            raise ValueError(
                f"{member.env_name('ZEROENTROPY_API_KEY')} is required "
                f"when {member.env_name('PROVIDER')} is 'zeroentropy'"
            )
        return ZeroEntropyCrossEncoder(
            api_key=api_key,
            model=member.zeroentropy_model,
            base_url=member.zeroentropy_base_url,
            timeout=member.zeroentropy_timeout,
        )
    elif provider == "siliconflow":
        api_key = member.siliconflow_api_key
        if not api_key:
            raise ValueError(
                f"{member.env_name('SILICONFLOW_API_KEY')} is required "
                f"when {member.env_name('PROVIDER')} is 'siliconflow'"
            )
        return SiliconFlowCrossEncoder(
            api_key=api_key,
            model=member.siliconflow_model,
            base_url=member.siliconflow_base_url,
            timeout=member.siliconflow_timeout,
        )
    elif provider == "google":
        project_id = member.google_project_id
        if not project_id:
            shared = " (or HINDSIGHT_API_LLM_VERTEXAI_PROJECT_ID)" if member.index == 0 else ""
            raise ValueError(
                f"{member.env_name('GOOGLE_PROJECT_ID')}{shared} "
                f"is required when {member.env_name('PROVIDER')} is 'google'"
            )
        return GoogleCrossEncoder(
            project_id=project_id,
            model=member.google_model,
            service_account_key=member.google_service_account_key,
            timeout=member.google_timeout,
        )
    elif provider == "alibaba":
        api_key = member.alibaba_api_key
        if not api_key:
            raise ValueError(
                f"{member.env_name('ALIBABA_API_KEY')} is required when {member.env_name('PROVIDER')} is 'alibaba'"
            )
        return AlibabaCloudCrossEncoder(
            api_key=api_key,
            model=member.alibaba_model,
            timeout=member.alibaba_timeout,
        )
    elif provider == "rrf":
        return RRFPassthroughCrossEncoder()
    elif provider == "jina-mlx":
        return JinaMLXCrossEncoder()
    else:
        raise ValueError(
            f"Unknown reranker provider: {provider}. Supported: 'local', 'tei', 'cohere', 'zeroentropy', 'siliconflow', 'alibaba', 'google', 'flashrank', 'litellm', 'litellm-sdk', 'rrf', 'jina-mlx'"
        )


def create_cross_encoder_from_env() -> CrossEncoderModel:
    """
    Create the configured reranker, based on configuration.

    Reads configuration via get_config() to ensure consistency across the codebase.
    With no ``HINDSIGHT_API_RERANKER_<n>_*`` members configured (the default) this
    is the single configured reranker; otherwise the chain is wrapped in a
    :class:`MultiCrossEncoder` that fails over across members in order.

    Returns:
        Configured CrossEncoderModel instance
    """
    from ..config import get_config

    chain = get_config().reranker_chain()
    if len(chain) == 1:
        return create_cross_encoder(chain[0])
    return MultiCrossEncoder([create_cross_encoder(member) for member in chain])
