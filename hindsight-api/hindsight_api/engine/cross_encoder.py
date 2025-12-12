"""
Cross-encoder abstraction for reranking.

Provides an interface for reranking with different backends.

Configuration via environment variables - see hindsight_api.config for all env var names.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import logging
import os

import httpx

from ..config import (
    ENV_RERANKER_PROVIDER,
    ENV_RERANKER_LOCAL_MODEL,
    ENV_RERANKER_TEI_URL,
    DEFAULT_RERANKER_PROVIDER,
    DEFAULT_RERANKER_LOCAL_MODEL,
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
    def predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
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
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize local SentenceTransformers cross-encoder.

        Args:
            model_name: Name of the CrossEncoder model to use.
                       Default: cross-encoder/ms-marco-MiniLM-L-6-v2
        """
        self.model_name = model_name or DEFAULT_RERANKER_LOCAL_MODEL
        self._model = None

    @property
    def provider_name(self) -> str:
        return "local"

    async def initialize(self) -> None:
        """Load the cross-encoder model."""
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
        self._model = CrossEncoder(self.model_name)
        logger.info("Reranker: local provider initialized")

    def predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Score query-document pairs for relevance.

        Args:
            pairs: List of (query, document) tuples to score

        Returns:
            List of relevance scores (raw logits from the model)
        """
        if self._model is None:
            raise RuntimeError("Reranker not initialized. Call initialize() first.")
        scores = self._model.predict(pairs, show_progress_bar=False)
        return scores.tolist() if hasattr(scores, 'tolist') else list(scores)


class RemoteTEICrossEncoder(CrossEncoderModel):
    """
    Remote cross-encoder implementation using HuggingFace Text Embeddings Inference (TEI) HTTP API.

    TEI supports reranking via the /rerank endpoint.
    See: https://github.com/huggingface/text-embeddings-inference

    Note: The TEI server must be running a cross-encoder/reranker model.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        batch_size: int = 32,
        max_retries: int = 3,
        retry_delay: float = 0.5,
    ):
        """
        Initialize remote TEI cross-encoder client.

        Args:
            base_url: Base URL of the TEI server (e.g., "http://localhost:8080")
            timeout: Request timeout in seconds (default: 30.0)
            batch_size: Maximum batch size for rerank requests (default: 32)
            max_retries: Maximum number of retries for failed requests (default: 3)
            retry_delay: Initial delay between retries in seconds, doubles each retry (default: 0.5)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client: Optional[httpx.Client] = None
        self._model_id: Optional[str] = None

    @property
    def provider_name(self) -> str:
        return "tei"

    def _request_with_retry(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Make an HTTP request with automatic retries on transient errors."""
        import time
        last_error = None
        delay = self.retry_delay

        for attempt in range(self.max_retries + 1):
            try:
                if method == "GET":
                    response = self._client.get(url, **kwargs)
                else:
                    response = self._client.post(url, **kwargs)
                response.raise_for_status()
                return response
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as e:
                last_error = e
                if attempt < self.max_retries:
                    logger.warning(f"TEI request failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
            except httpx.HTTPStatusError as e:
                # Retry on 5xx server errors
                if e.response.status_code >= 500 and attempt < self.max_retries:
                    last_error = e
                    logger.warning(f"TEI server error (attempt {attempt + 1}/{self.max_retries + 1}): {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise

        raise last_error

    async def initialize(self) -> None:
        """Initialize the HTTP client and verify server connectivity."""
        if self._client is not None:
            return

        logger.info(f"Reranker: initializing TEI provider at {self.base_url}")
        self._client = httpx.Client(timeout=self.timeout)

        # Verify server is reachable and get model info
        try:
            response = self._request_with_retry("GET", f"{self.base_url}/info")
            info = response.json()
            self._model_id = info.get("model_id", "unknown")
            logger.info(f"Reranker: TEI provider initialized (model: {self._model_id})")
        except httpx.HTTPError as e:
            raise RuntimeError(f"Failed to connect to TEI server at {self.base_url}: {e}")

    def predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Score query-document pairs using the remote TEI reranker.

        Args:
            pairs: List of (query, document) tuples to score

        Returns:
            List of relevance scores
        """
        if self._client is None:
            raise RuntimeError("Reranker not initialized. Call initialize() first.")

        if not pairs:
            return []

        all_scores = []

        # Process in batches
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]

            # TEI rerank endpoint expects query and texts separately
            # All pairs in a batch should have the same query for optimal performance
            # but we handle mixed queries by making separate requests per unique query
            query_groups: dict[str, list[tuple[int, str]]] = {}
            for idx, (query, text) in enumerate(batch):
                if query not in query_groups:
                    query_groups[query] = []
                query_groups[query].append((idx, text))

            batch_scores = [0.0] * len(batch)

            for query, indexed_texts in query_groups.items():
                texts = [text for _, text in indexed_texts]
                indices = [idx for idx, _ in indexed_texts]

                try:
                    response = self._request_with_retry(
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
                    for result in results:
                        original_idx = result["index"]
                        score = result["score"]
                        # Map back to batch position
                        batch_scores[indices[original_idx]] = score

                except httpx.HTTPError as e:
                    raise RuntimeError(f"TEI rerank request failed: {e}")

            all_scores.extend(batch_scores)

        return all_scores


def create_cross_encoder_from_env() -> CrossEncoderModel:
    """
    Create a CrossEncoderModel instance based on environment variables.

    See hindsight_api.config for environment variable names and defaults.

    Returns:
        Configured CrossEncoderModel instance
    """
    provider = os.environ.get(ENV_RERANKER_PROVIDER, DEFAULT_RERANKER_PROVIDER).lower()

    if provider == "tei":
        url = os.environ.get(ENV_RERANKER_TEI_URL)
        if not url:
            raise ValueError(
                f"{ENV_RERANKER_TEI_URL} is required when {ENV_RERANKER_PROVIDER} is 'tei'"
            )
        return RemoteTEICrossEncoder(base_url=url)
    elif provider == "local":
        model = os.environ.get(ENV_RERANKER_LOCAL_MODEL)
        model_name = model or DEFAULT_RERANKER_LOCAL_MODEL
        return LocalSTCrossEncoder(model_name=model_name)
    else:
        raise ValueError(
            f"Unknown reranker provider: {provider}. Supported: 'local', 'tei'"
        )
