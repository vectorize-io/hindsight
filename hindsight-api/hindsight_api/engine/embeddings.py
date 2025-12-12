"""
Embeddings abstraction for the memory system.

Provides an interface for generating embeddings with different backends.

IMPORTANT: All embeddings must produce 384-dimensional vectors to match
the database schema (pgvector column defined as vector(384)).

Configuration via environment variables - see hindsight_api.config for all env var names.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
import logging
import os

import httpx

from ..config import (
    ENV_EMBEDDINGS_PROVIDER,
    ENV_EMBEDDINGS_LOCAL_MODEL,
    ENV_EMBEDDINGS_TEI_URL,
    DEFAULT_EMBEDDINGS_PROVIDER,
    DEFAULT_EMBEDDINGS_LOCAL_MODEL,
    EMBEDDING_DIMENSION,
)

logger = logging.getLogger(__name__)


class Embeddings(ABC):
    """
    Abstract base class for embedding generation.

    All implementations MUST generate 384-dimensional embeddings to match
    the database schema.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return a human-readable name for this provider (e.g., 'local', 'tei')."""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the embedding model asynchronously.

        This should be called during startup to load/connect to the model
        and avoid cold start latency on first encode() call.
        """
        pass

    @abstractmethod
    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        Generate 384-dimensional embeddings for a list of texts.

        Args:
            texts: List of text strings to encode

        Returns:
            List of 384-dimensional embedding vectors (each is a list of floats)
        """
        pass


class LocalSTEmbeddings(Embeddings):
    """
    Local embeddings implementation using SentenceTransformers.

    Call initialize() during startup to load the model and avoid cold starts.

    Default model is BAAI/bge-small-en-v1.5 which produces 384-dimensional
    embeddings matching the database schema.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize local SentenceTransformers embeddings.

        Args:
            model_name: Name of the SentenceTransformer model to use.
                       Must produce 384-dimensional embeddings.
                       Default: BAAI/bge-small-en-v1.5
        """
        self.model_name = model_name or DEFAULT_EMBEDDINGS_LOCAL_MODEL
        self._model = None

    @property
    def provider_name(self) -> str:
        return "local"

    async def initialize(self) -> None:
        """Load the embedding model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for LocalSTEmbeddings. "
                "Install it with: pip install sentence-transformers"
            )

        logger.info(f"Embeddings: initializing local provider with model {self.model_name}")
        # Disable lazy loading (meta tensors) which causes issues with newer transformers/accelerate
        # Setting low_cpu_mem_usage=False and device_map=None ensures tensors are fully materialized
        self._model = SentenceTransformer(
            self.model_name,
            model_kwargs={"low_cpu_mem_usage": False, "device_map": None},
        )

        # Validate dimension matches database schema
        model_dim = self._model.get_sentence_embedding_dimension()
        if model_dim != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Model {self.model_name} produces {model_dim}-dimensional embeddings, "
                f"but database schema requires {EMBEDDING_DIMENSION} dimensions. "
                f"Use a model that produces {EMBEDDING_DIMENSION}-dimensional embeddings."
            )

        logger.info(f"Embeddings: local provider initialized (dim: {model_dim})")

    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        Generate 384-dimensional embeddings for a list of texts.

        Args:
            texts: List of text strings to encode

        Returns:
            List of 384-dimensional embedding vectors
        """
        if self._model is None:
            raise RuntimeError("Embeddings not initialized. Call initialize() first.")
        embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]


class RemoteTEIEmbeddings(Embeddings):
    """
    Remote embeddings implementation using HuggingFace Text Embeddings Inference (TEI) HTTP API.

    TEI provides a high-performance inference server for embedding models.
    See: https://github.com/huggingface/text-embeddings-inference

    The server should be running a model that produces 384-dimensional embeddings.
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
        Initialize remote TEI embeddings client.

        Args:
            base_url: Base URL of the TEI server (e.g., "http://localhost:8080")
            timeout: Request timeout in seconds (default: 30.0)
            batch_size: Maximum batch size for embedding requests (default: 32)
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

        logger.info(f"Embeddings: initializing TEI provider at {self.base_url}")
        self._client = httpx.Client(timeout=self.timeout)

        # Verify server is reachable and get model info
        try:
            response = self._request_with_retry("GET", f"{self.base_url}/info")
            info = response.json()
            self._model_id = info.get("model_id", "unknown")
            logger.info(f"Embeddings: TEI provider initialized (model: {self._model_id})")
        except httpx.HTTPError as e:
            raise RuntimeError(f"Failed to connect to TEI server at {self.base_url}: {e}")

    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using the remote TEI server.

        Args:
            texts: List of text strings to encode

        Returns:
            List of embedding vectors
        """
        if self._client is None:
            raise RuntimeError("Embeddings not initialized. Call initialize() first.")

        if not texts:
            return []

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            try:
                response = self._request_with_retry(
                    "POST",
                    f"{self.base_url}/embed",
                    json={"inputs": batch},
                )
                batch_embeddings = response.json()
                all_embeddings.extend(batch_embeddings)
            except httpx.HTTPError as e:
                raise RuntimeError(f"TEI embedding request failed: {e}")

        return all_embeddings


def create_embeddings_from_env() -> Embeddings:
    """
    Create an Embeddings instance based on environment variables.

    See hindsight_api.config for environment variable names and defaults.

    Returns:
        Configured Embeddings instance
    """
    provider = os.environ.get(ENV_EMBEDDINGS_PROVIDER, DEFAULT_EMBEDDINGS_PROVIDER).lower()

    if provider == "tei":
        url = os.environ.get(ENV_EMBEDDINGS_TEI_URL)
        if not url:
            raise ValueError(
                f"{ENV_EMBEDDINGS_TEI_URL} is required when {ENV_EMBEDDINGS_PROVIDER} is 'tei'"
            )
        return RemoteTEIEmbeddings(base_url=url)
    elif provider == "local":
        model = os.environ.get(ENV_EMBEDDINGS_LOCAL_MODEL)
        model_name = model or DEFAULT_EMBEDDINGS_LOCAL_MODEL
        return LocalSTEmbeddings(model_name=model_name)
    else:
        raise ValueError(
            f"Unknown embeddings provider: {provider}. Supported: 'local', 'tei'"
        )
