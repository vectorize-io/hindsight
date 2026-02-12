"""
Embeddings abstraction for the memory system.

Provides an interface for generating embeddings with different backends.

The embedding dimension is auto-detected from the model at initialization.
The database schema is automatically adjusted to match the model's dimension.

Configuration via environment variables - see hindsight_api.config for all env var names.
"""

import logging
import os
import warnings
from abc import ABC, abstractmethod

import httpx

from ..config import (
    DEFAULT_EMBEDDINGS_COHERE_MODEL,
    DEFAULT_EMBEDDINGS_LITELLM_MODEL,
    DEFAULT_EMBEDDINGS_LITELLM_SDK_MODEL,
    DEFAULT_EMBEDDINGS_LOCAL_FORCE_CPU,
    DEFAULT_EMBEDDINGS_LOCAL_MODEL,
    DEFAULT_EMBEDDINGS_LOCAL_TRUST_REMOTE_CODE,
    DEFAULT_EMBEDDINGS_OPENAI_MODEL,
    DEFAULT_EMBEDDINGS_PROVIDER,
    DEFAULT_LITELLM_API_BASE,
    ENV_EMBEDDINGS_COHERE_API_KEY,
    ENV_EMBEDDINGS_LITELLM_SDK_API_KEY,
    ENV_EMBEDDINGS_LOCAL_FORCE_CPU,
    ENV_EMBEDDINGS_LOCAL_MODEL,
    ENV_EMBEDDINGS_LOCAL_TRUST_REMOTE_CODE,
    ENV_EMBEDDINGS_OPENAI_API_KEY,
    ENV_EMBEDDINGS_OPENAI_BASE_URL,
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

    def __init__(self, model_name: str | None = None, force_cpu: bool = False, trust_remote_code: bool = False):
        """
        Initialize local SentenceTransformers embeddings.

        Args:
            model_name: Name of the SentenceTransformer model to use.
                       Default: BAAI/bge-small-en-v1.5
            force_cpu: Force CPU mode (avoids MPS/XPC issues on macOS in daemon mode).
                      Default: False
            trust_remote_code: Allow loading models with custom code (security risk).
                              Required for some models with custom architectures.
                              Default: False (disabled for security)
        """
        self.model_name = model_name or DEFAULT_EMBEDDINGS_LOCAL_MODEL
        self.force_cpu = force_cpu
        self.trust_remote_code = trust_remote_code
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

        # Determine device based on hardware availability.
        # We always set low_cpu_mem_usage=False to prevent lazy loading (meta tensors)
        # which can cause issues when accelerate is installed but no GPU is available.
        import torch

        # Force CPU mode if configured (used in daemon mode to avoid MPS/XPC issues on macOS)
        if self.force_cpu:
            device = "cpu"
            logger.info("Embeddings: forcing CPU mode")
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

        # Suppress verbose transformers warnings during model loading
        # This suppresses the "UNEXPECTED" warnings from BertModel which are harmless
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
                self._model = SentenceTransformer(
                    self.model_name,
                    device=device,
                    model_kwargs={"low_cpu_mem_usage": False},
                    trust_remote_code=self.trust_remote_code,
                )
            finally:
                # Restore original logging level
                transformers_logger.setLevel(original_level)

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
        base_url: str | None = None,
        batch_size: int = 100,
        max_retries: int = 3,
    ):
        """
        Initialize OpenAI embeddings client.

        Args:
            api_key: OpenAI API key
            model: OpenAI embedding model name (default: text-embedding-3-small)
            base_url: Custom base URL for OpenAI-compatible API (e.g., Azure OpenAI endpoint)
            batch_size: Maximum batch size for embedding requests (default: 100)
            max_retries: Maximum number of retries for failed requests (default: 3)
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
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

        base_url_msg = f" at {self.base_url}" if self.base_url else ""
        logger.info(f"Embeddings: initializing OpenAI provider with model {self.model}{base_url_msg}")

        # Build client kwargs, only including base_url if set (for Azure or custom endpoints)
        client_kwargs = {"api_key": self.api_key, "max_retries": self.max_retries}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self._client = OpenAI(**client_kwargs)

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


class CohereEmbeddings(Embeddings):
    """
    Cohere embeddings implementation using the Cohere API.

    Supports embed-english-v3.0 (1024 dims) and embed-multilingual-v3.0 (1024 dims).

    The embedding dimension is auto-detected from the model at initialization.
    """

    # Known dimensions for Cohere embedding models
    MODEL_DIMENSIONS = {
        "embed-english-v3.0": 1024,
        "embed-multilingual-v3.0": 1024,
        "embed-english-light-v3.0": 384,
        "embed-multilingual-light-v3.0": 384,
        "embed-english-v2.0": 4096,
        "embed-multilingual-v2.0": 768,
    }

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_EMBEDDINGS_COHERE_MODEL,
        base_url: str | None = None,
        batch_size: int = 96,
        timeout: float = 60.0,
        input_type: str = "search_document",
    ):
        """
        Initialize Cohere embeddings client.

        Args:
            api_key: Cohere API key
            model: Cohere embedding model name (default: embed-english-v3.0)
            base_url: Custom base URL for Cohere-compatible API (e.g., Azure-hosted endpoint)
            batch_size: Maximum batch size for embedding requests (default: 96, Cohere's limit)
            timeout: Request timeout in seconds (default: 60.0)
            input_type: Input type for embeddings (default: search_document).
                       Options: search_document, search_query, classification, clustering
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.batch_size = batch_size
        self.timeout = timeout
        self.input_type = input_type
        self._client = None
        self._dimension: int | None = None

    @property
    def provider_name(self) -> str:
        return "cohere"

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise RuntimeError("Embeddings not initialized. Call initialize() first.")
        return self._dimension

    async def initialize(self) -> None:
        """Initialize the Cohere client and detect dimension."""
        if self._client is not None:
            return

        try:
            import cohere
        except ImportError:
            raise ImportError("cohere is required for CohereEmbeddings. Install it with: pip install cohere")

        base_url_msg = f" at {self.base_url}" if self.base_url else ""
        logger.info(f"Embeddings: initializing Cohere provider with model {self.model}{base_url_msg}")

        # Build client kwargs, only including base_url if set (for Azure or custom endpoints)
        client_kwargs = {"api_key": self.api_key, "timeout": self.timeout}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self._client = cohere.Client(**client_kwargs)

        # Try to get dimension from known models, otherwise do a test embedding
        if self.model in self.MODEL_DIMENSIONS:
            self._dimension = self.MODEL_DIMENSIONS[self.model]
        else:
            # Do a test embedding to detect dimension
            response = self._client.embed(
                texts=["test"],
                model=self.model,
                input_type=self.input_type,
            )
            if response.embeddings and isinstance(response.embeddings, list):
                self._dimension = len(response.embeddings[0])

        logger.info(f"Embeddings: Cohere provider initialized (model: {self.model}, dim: {self._dimension})")

    def encode(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings using the Cohere API.

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

            response = self._client.embed(
                texts=batch,
                model=self.model,
                input_type=self.input_type,
            )

            all_embeddings.extend(response.embeddings)

        return all_embeddings


class LiteLLMEmbeddings(Embeddings):
    """
    LiteLLM embeddings implementation using LiteLLM proxy's /embeddings endpoint.

    LiteLLM provides a unified interface for multiple embedding providers.
    The proxy exposes an OpenAI-compatible /embeddings endpoint.
    See: https://docs.litellm.ai/docs/embedding/supported_embedding

    Supported providers via LiteLLM:
    - OpenAI (text-embedding-3-small, text-embedding-ada-002, etc.)
    - Cohere (embed-english-v3.0, etc.) - prefix with cohere/
    - Vertex AI (textembedding-gecko, etc.) - prefix with vertex_ai/
    - HuggingFace, Mistral, Voyage AI, etc.

    The embedding dimension is auto-detected from the model at initialization.
    """

    def __init__(
        self,
        api_base: str = DEFAULT_LITELLM_API_BASE,
        api_key: str | None = None,
        model: str = DEFAULT_EMBEDDINGS_LITELLM_MODEL,
        batch_size: int = 100,
        timeout: float = 60.0,
    ):
        """
        Initialize LiteLLM embeddings client.

        Args:
            api_base: Base URL of the LiteLLM proxy (default: http://localhost:4000)
            api_key: API key for the LiteLLM proxy (optional, depends on proxy config)
            model: Embedding model name (default: text-embedding-3-small)
                   Use provider prefix for non-OpenAI models (e.g., cohere/embed-english-v3.0)
            batch_size: Maximum batch size for embedding requests (default: 100)
            timeout: Request timeout in seconds (default: 60.0)
        """
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self._client: httpx.Client | None = None
        self._dimension: int | None = None

    @property
    def provider_name(self) -> str:
        return "litellm"

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise RuntimeError("Embeddings not initialized. Call initialize() first.")
        return self._dimension

    async def initialize(self) -> None:
        """Initialize the HTTP client and detect embedding dimension."""
        if self._client is not None:
            return

        logger.info(f"Embeddings: initializing LiteLLM provider at {self.api_base} with model {self.model}")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        self._client = httpx.Client(timeout=self.timeout, headers=headers)

        # Do a test embedding to detect dimension
        try:
            response = self._client.post(
                f"{self.api_base}/embeddings",
                json={"model": self.model, "input": ["test"]},
            )
            response.raise_for_status()
            result = response.json()
            if result.get("data") and len(result["data"]) > 0:
                self._dimension = len(result["data"][0]["embedding"])
            logger.info(f"Embeddings: LiteLLM provider initialized (model: {self.model}, dim: {self._dimension})")
        except httpx.HTTPError as e:
            raise RuntimeError(f"Failed to connect to LiteLLM proxy at {self.api_base}: {e}")

    def encode(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings using the LiteLLM proxy.

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

            response = self._client.post(
                f"{self.api_base}/embeddings",
                json={"model": self.model, "input": batch},
            )
            response.raise_for_status()
            result = response.json()

            # Sort by index to ensure correct order
            batch_embeddings = sorted(result["data"], key=lambda x: x["index"])
            all_embeddings.extend([e["embedding"] for e in batch_embeddings])

        return all_embeddings


class LiteLLMSDKEmbeddings(Embeddings):
    """
    LiteLLM SDK embeddings for direct API integration.

    Supports embeddings via LiteLLM SDK without requiring a proxy server.
    Supported providers: Cohere, OpenAI, Azure OpenAI, HuggingFace, Voyage AI, Together AI, etc.

    Example model names:
    - cohere/embed-english-v3.0
    - openai/text-embedding-3-small
    - together_ai/togethercomputer/m2-bert-80M-8k-retrieval
    - voyage/voyage-2
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_EMBEDDINGS_LITELLM_SDK_MODEL,
        api_base: str | None = None,
        batch_size: int = 100,
        timeout: float = 60.0,
    ):
        """
        Initialize LiteLLM SDK embeddings client.

        Args:
            api_key: API key for the embedding provider
            model: Model name with provider prefix (e.g., "cohere/embed-english-v3.0")
            api_base: Custom base URL for API (optional)
            batch_size: Maximum batch size for embedding requests (default: 100)
            timeout: Request timeout in seconds (default: 60.0)
        """
        self.api_key = api_key
        self.model = model
        self.api_base = api_base
        self.batch_size = batch_size
        self.timeout = timeout
        self._litellm = None  # Will be set during initialization
        self._dimension: int | None = None

    @property
    def provider_name(self) -> str:
        return "litellm-sdk"

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise RuntimeError("Embeddings not initialized. Call initialize() first.")
        return self._dimension

    async def initialize(self) -> None:
        """Initialize the LiteLLM SDK client and detect dimension."""
        if self._litellm is not None:
            return

        try:
            import litellm

            self._litellm = litellm  # Store reference
        except ImportError:
            raise ImportError("litellm is required for LiteLLMSDKEmbeddings. Install it with: pip install litellm")

        api_base_msg = f" at {self.api_base}" if self.api_base else ""
        logger.info(f"Embeddings: initializing LiteLLM SDK provider with model {self.model}{api_base_msg}")

        # Do a test embedding to detect dimension
        try:
            # Build kwargs for embedding call
            embed_kwargs = {
                "model": self.model,
                "input": ["test"],
                "api_key": self.api_key,
            }
            if self.api_base:
                embed_kwargs["api_base"] = self.api_base

            # Use async embedding method (standard in litellm)
            response = await self._litellm.aembedding(**embed_kwargs)

            # Extract dimension from response
            if response.data and len(response.data) > 0:
                self._dimension = len(response.data[0]["embedding"])
            else:
                raise RuntimeError(f"Unable to detect embedding dimension for model {self.model}")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize LiteLLM SDK embeddings: {e}")

        logger.info(f"Embeddings: LiteLLM SDK provider initialized (model: {self.model}, dim: {self._dimension})")

    def encode(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings using the LiteLLM SDK.

        Args:
            texts: List of text strings to encode

        Returns:
            List of embedding vectors (one per input text)
        """
        if self._litellm is None:
            raise RuntimeError("Embeddings not initialized. Call initialize() first.")

        if not texts:
            return []

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            try:
                # Build kwargs for embedding call
                embed_kwargs = {
                    "model": self.model,
                    "input": batch,
                    "api_key": self.api_key,
                }
                if self.api_base:
                    embed_kwargs["api_base"] = self.api_base

                # Use sync embedding (litellm doesn't have async in thread-safe way)
                response = self._litellm.embedding(**embed_kwargs)

                # Extract embeddings from response
                # Sort by index to ensure correct order
                batch_embeddings = sorted(response.data, key=lambda x: x.get("index", 0))
                all_embeddings.extend([e["embedding"] for e in batch_embeddings])

            except Exception as e:
                import traceback

                logger.error(
                    f"Error in LiteLLM embedding for batch starting at index {i}: {e}\n"
                    f"Traceback: {traceback.format_exc()}"
                )
                raise

        return all_embeddings


def create_embeddings_from_env() -> Embeddings:
    """
    Create an Embeddings instance based on configuration.

    Reads configuration via get_config() to ensure consistency across the codebase.

    Returns:
        Configured Embeddings instance
    """
    from ..config import get_config

    config = get_config()
    provider = config.embeddings_provider.lower()

    if provider == "tei":
        url = config.embeddings_tei_url
        if not url:
            raise ValueError(f"{ENV_EMBEDDINGS_TEI_URL} is required when {ENV_EMBEDDINGS_PROVIDER} is 'tei'")
        return RemoteTEIEmbeddings(base_url=url)
    elif provider == "local":
        return LocalSTEmbeddings(
            model_name=config.embeddings_local_model,
            force_cpu=config.embeddings_local_force_cpu,
            trust_remote_code=config.embeddings_local_trust_remote_code,
        )
    elif provider == "openai":
        # Use dedicated embeddings API key, or fall back to LLM API key
        api_key = os.environ.get(ENV_EMBEDDINGS_OPENAI_API_KEY) or os.environ.get(ENV_LLM_API_KEY)
        if not api_key:
            raise ValueError(
                f"{ENV_EMBEDDINGS_OPENAI_API_KEY} or {ENV_LLM_API_KEY} is required "
                f"when {ENV_EMBEDDINGS_PROVIDER} is 'openai'"
            )
        model = os.environ.get(ENV_EMBEDDINGS_OPENAI_MODEL, DEFAULT_EMBEDDINGS_OPENAI_MODEL)
        base_url = os.environ.get(ENV_EMBEDDINGS_OPENAI_BASE_URL) or None
        return OpenAIEmbeddings(api_key=api_key, model=model, base_url=base_url)
    elif provider == "cohere":
        api_key = config.embeddings_cohere_api_key
        if not api_key:
            raise ValueError(f"{ENV_EMBEDDINGS_COHERE_API_KEY} is required when {ENV_EMBEDDINGS_PROVIDER} is 'cohere'")
        return CohereEmbeddings(
            api_key=api_key,
            model=config.embeddings_cohere_model,
            base_url=config.embeddings_cohere_base_url,
        )
    elif provider == "litellm":
        return LiteLLMEmbeddings(
            api_base=config.embeddings_litellm_api_base,
            api_key=config.embeddings_litellm_api_key,
            model=config.embeddings_litellm_model,
        )
    elif provider == "litellm-sdk":
        api_key = config.embeddings_litellm_sdk_api_key
        if not api_key:
            raise ValueError(
                f"{ENV_EMBEDDINGS_LITELLM_SDK_API_KEY} is required when {ENV_EMBEDDINGS_PROVIDER} is 'litellm-sdk'"
            )
        return LiteLLMSDKEmbeddings(
            api_key=api_key,
            model=config.embeddings_litellm_sdk_model,
            api_base=config.embeddings_litellm_sdk_api_base,
        )
    else:
        raise ValueError(
            f"Unknown embeddings provider: {provider}. "
            f"Supported: 'local', 'tei', 'openai', 'cohere', 'litellm', 'litellm-sdk'"
        )
