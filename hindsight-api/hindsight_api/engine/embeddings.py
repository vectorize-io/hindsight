"""
Embeddings abstraction for the memory system.

Provides an interface for generating embeddings with different backends.

The embedding dimension is auto-detected from the model at initialization.
The database schema is automatically adjusted to match the model's dimension.

Configuration via environment variables - see hindsight_api.config for all env var names.
"""

import logging
import os
from abc import ABC, abstractmethod

import httpx

from ..config import (
    DEFAULT_EMBEDDINGS_LOCAL_MODEL,
    DEFAULT_EMBEDDINGS_OPENAI_MODEL,
    DEFAULT_EMBEDDINGS_PROVIDER,
    ENV_EMBEDDINGS_LOCAL_MODEL,
    ENV_EMBEDDINGS_OPENAI_API_KEY,
    ENV_EMBEDDINGS_OPENAI_MODEL,
    ENV_EMBEDDINGS_PROVIDER,
    ENV_EMBEDDINGS_TEI_URL,
    ENV_LLM_API_KEY,
)

logger = logging.getLogger(__name__)


class Embeddings(ABC):
    """
    Abstract base class for embedding generation.

    The embedding dimension is determined by the model and detected at initialization.
    The database schema is automatically adjusted to match the model's dimension.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return a human-readable name for this provider (e.g., 'local', 'tei')."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension produced by this model."""
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
    def encode(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to encode

        Returns:
            List of embedding vectors (each is a list of floats)
        """
        pass


class LocalSTEmbeddings(Embeddings):
    """
    Local embeddings implementation using SentenceTransformers.

    Call initialize() during startup to load the model and avoid cold starts.
    The embedding dimension is auto-detected from the model.
    """

    def __init__(self, model_name: str | None = None):
        """
        Initialize local SentenceTransformers embeddings.

        Args:
            model_name: Name of the SentenceTransformer model to use.
                       Default: BAAI/bge-small-en-v1.5
        """
        self.model_name = model_name or DEFAULT_EMBEDDINGS_LOCAL_MODEL
        self._model = None
        self._dimension: int | None = None

    @property
    def provider_name(self) -> str:
        return "local"

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise RuntimeError("Embeddings not initialized. Call initialize() first.")
        return self._dimension

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

        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(f"Embeddings: local provider initialized (dim: {self._dimension})")

    def encode(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to encode

        Returns:
            List of embedding vectors
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

    The embedding dimension is auto-detected from the server at initialization.
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
        self._client: httpx.Client | None = None
        self._model_id: str | None = None
        self._dimension: int | None = None

    @property
    def provider_name(self) -> str:
        return "tei"

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise RuntimeError("Embeddings not initialized. Call initialize() first.")
        return self._dimension

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
                    logger.warning(
                        f"TEI request failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}. Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
            except httpx.HTTPStatusError as e:
                # Retry on 5xx server errors
                if e.response.status_code >= 500 and attempt < self.max_retries:
                    last_error = e
                    logger.warning(
                        f"TEI server error (attempt {attempt + 1}/{self.max_retries + 1}): {e}. Retrying in {delay}s..."
                    )
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

            # Get dimension from server info or by doing a test embedding
            if "max_input_length" in info and "model_dtype" in info:
                # Try to get dimension from info endpoint (some TEI versions expose it)
                # If not available, do a test embedding
                pass

            # Do a test embedding to detect dimension
            test_response = self._request_with_retry(
                "POST",
                f"{self.base_url}/embed",
                json={"inputs": ["test"]},
            )
            test_embeddings = test_response.json()
            if test_embeddings and len(test_embeddings) > 0:
                self._dimension = len(test_embeddings[0])

            logger.info(f"Embeddings: TEI provider initialized (model: {self._model_id}, dim: {self._dimension})")
        except httpx.HTTPError as e:
            raise RuntimeError(f"Failed to connect to TEI server at {self.base_url}: {e}")

    def encode(self, texts: list[str]) -> list[list[float]]:
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
            batch = texts[i : i + self.batch_size]

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


class OpenAIEmbeddings(Embeddings):
    """
    OpenAI embeddings implementation using the OpenAI API.

    Supports text-embedding-3-small (1536 dims), text-embedding-3-large (3072 dims),
    and text-embedding-ada-002 (1536 dims, legacy).

    The embedding dimension is auto-detected from the model at initialization.
    """

    # Known dimensions for OpenAI embedding models
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_EMBEDDINGS_OPENAI_MODEL,
        batch_size: int = 100,
        max_retries: int = 3,
    ):
        """
        Initialize OpenAI embeddings client.

        Args:
            api_key: OpenAI API key
            model: OpenAI embedding model name (default: text-embedding-3-small)
            batch_size: Maximum batch size for embedding requests (default: 100)
            max_retries: Maximum number of retries for failed requests (default: 3)
        """
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self._client = None
        self._dimension: int | None = None

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise RuntimeError("Embeddings not initialized. Call initialize() first.")
        return self._dimension

    async def initialize(self) -> None:
        """Initialize the OpenAI client and detect dimension."""
        if self._client is not None:
            return

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai is required for OpenAIEmbeddings. Install it with: pip install openai")

        logger.info(f"Embeddings: initializing OpenAI provider with model {self.model}")
        self._client = OpenAI(api_key=self.api_key, max_retries=self.max_retries)

        # Try to get dimension from known models, otherwise do a test embedding
        if self.model in self.MODEL_DIMENSIONS:
            self._dimension = self.MODEL_DIMENSIONS[self.model]
        else:
            # Do a test embedding to detect dimension
            response = self._client.embeddings.create(
                model=self.model,
                input=["test"],
            )
            if response.data:
                self._dimension = len(response.data[0].embedding)

        logger.info(f"Embeddings: OpenAI provider initialized (model: {self.model}, dim: {self._dimension})")

    def encode(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings using the OpenAI API.

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
            batch = texts[i : i + self.batch_size]

            response = self._client.embeddings.create(
                model=self.model,
                input=batch,
            )

            # Sort by index to ensure correct order
            batch_embeddings = sorted(response.data, key=lambda x: x.index)
            all_embeddings.extend([e.embedding for e in batch_embeddings])

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
            raise ValueError(f"{ENV_EMBEDDINGS_TEI_URL} is required when {ENV_EMBEDDINGS_PROVIDER} is 'tei'")
        return RemoteTEIEmbeddings(base_url=url)
    elif provider == "local":
        model = os.environ.get(ENV_EMBEDDINGS_LOCAL_MODEL)
        model_name = model or DEFAULT_EMBEDDINGS_LOCAL_MODEL
        return LocalSTEmbeddings(model_name=model_name)
    elif provider == "openai":
        # Use dedicated embeddings API key, or fall back to LLM API key
        api_key = os.environ.get(ENV_EMBEDDINGS_OPENAI_API_KEY) or os.environ.get(ENV_LLM_API_KEY)
        if not api_key:
            raise ValueError(
                f"{ENV_EMBEDDINGS_OPENAI_API_KEY} or {ENV_LLM_API_KEY} is required "
                f"when {ENV_EMBEDDINGS_PROVIDER} is 'openai'"
            )
        model = os.environ.get(ENV_EMBEDDINGS_OPENAI_MODEL, DEFAULT_EMBEDDINGS_OPENAI_MODEL)
        return OpenAIEmbeddings(api_key=api_key, model=model)
    else:
        raise ValueError(f"Unknown embeddings provider: {provider}. Supported: 'local', 'tei', 'openai'")
