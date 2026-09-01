"""
Embeddings abstraction for the memory system.

Provides an interface for generating embeddings with different backends.

The embedding dimension is auto-detected from the model at initialization.
The database schema is automatically adjusted to match the model's dimension.

Configuration via environment variables - see hindsight_api.config for all env var names.
"""

import asyncio
import base64
import logging
import os
import struct
import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeVar, cast
from urllib.parse import parse_qs, urlparse, urlunparse

import httpx
from pydantic import BaseModel

from ..config import (
    DEFAULT_EMBEDDINGS_COHERE_MODEL,
    DEFAULT_EMBEDDINGS_GEMINI_MODEL,
    DEFAULT_EMBEDDINGS_INITIAL_BACKOFF,
    DEFAULT_EMBEDDINGS_LITELLM_MODEL,
    DEFAULT_EMBEDDINGS_LITELLM_SDK_MODEL,
    DEFAULT_EMBEDDINGS_LOCAL_MODEL,
    DEFAULT_EMBEDDINGS_MAX_BACKOFF,
    DEFAULT_EMBEDDINGS_MAX_RETRIES,
    DEFAULT_EMBEDDINGS_ONNX_BATCH_SIZE,
    DEFAULT_EMBEDDINGS_ONNX_CPU_MEM_ARENA,
    DEFAULT_EMBEDDINGS_OPENAI_MODEL,
    DEFAULT_EMBEDDINGS_RETRY_BUDGET,
    DEFAULT_EMBEDDINGS_ZEROENTROPY_BATCH_SIZE,
    DEFAULT_EMBEDDINGS_ZEROENTROPY_DIMENSIONS,
    DEFAULT_EMBEDDINGS_ZEROENTROPY_ENCODING_FORMAT,
    DEFAULT_EMBEDDINGS_ZEROENTROPY_LATENCY,
    DEFAULT_EMBEDDINGS_ZEROENTROPY_MODEL,
    DEFAULT_LITELLM_API_BASE,
    DEFAULT_ZEROENTROPY_BASE_URL,
    ENV_EMBEDDINGS_COHERE_API_KEY,
    ENV_EMBEDDINGS_GEMINI_API_KEY,
    ENV_EMBEDDINGS_LITELLM_DIMENSIONS,
    ENV_EMBEDDINGS_OPENAI_API_KEY,
    ENV_EMBEDDINGS_OPENAI_BASE_URL,
    ENV_EMBEDDINGS_OPENAI_MODEL,
    ENV_EMBEDDINGS_PROVIDER,
    ENV_EMBEDDINGS_TEI_URL,
    ENV_EMBEDDINGS_ZEROENTROPY_API_KEY,
    ENV_EMBEDDINGS_ZEROENTROPY_DIMENSIONS,
    ENV_EMBEDDINGS_ZEROENTROPY_ENCODING_FORMAT,
    ENV_LLM_API_KEY,
)
from .bank_attribution import apply_bank_attribution
from .local_device import (
    release_local_inference_memory,
    resolve_model_device_type,
    select_local_device,
)
from .tei_retry import TEI_KEEPALIVE_EXPIRY_SECONDS, is_retryable_tei_transport_error, tei_retry_delay

if TYPE_CHECKING:
    from ..config import HindsightConfig

logger = logging.getLogger(__name__)

T = TypeVar("T")

# 4xx codes that describe a transient condition rather than a bad request.
# Everything else in the 4xx range (auth, validation, not-found) is a client-side
# problem that retrying cannot fix.
_TRANSIENT_STATUS_CODES = frozenset({408, 409, 425, 429})

# Exception class names treated as transient when the exception carries no HTTP
# status code. Matching by name keeps this module free of a hard litellm/openai
# import (litellm is imported lazily, and only by the providers that need it).
_TRANSIENT_EXCEPTION_NAMES = frozenset(
    {
        "APIConnectionError",
        "APIError",
        "APITimeoutError",
        "ConnectionError",
        "InternalServerError",
        "RateLimitError",
        "ServiceUnavailableError",
        "Timeout",
        "TimeoutError",
    }
)


@dataclass(frozen=True)
class EmbeddingRetryPolicy:
    """
    Bounded retry policy for remote embedding APIs.

    Recall embeds the query inline on the request path, so a single upstream 5xx
    would otherwise become a user-visible recall failure. Retries are bounded two
    ways at once:

    * ``max_retries`` caps the number of extra attempts (0 disables retrying).
    * ``budget_seconds`` caps the wall-clock time a single ``encode()`` call may
      *waste* on retries — failed attempts plus backoff sleeps. Successful work
      never counts against it, so a large multi-batch encode is not penalised for
      the batches that worked, while the worst-case added latency stays bounded.

    The pairing matters: attempts alone cannot bound latency (an upstream that
    fails slowly turns 5 attempts into minutes), and a budget alone cannot stop a
    fast-failing upstream from being hammered.
    """

    max_retries: int = DEFAULT_EMBEDDINGS_MAX_RETRIES
    initial_backoff: float = DEFAULT_EMBEDDINGS_INITIAL_BACKOFF
    max_backoff: float = DEFAULT_EMBEDDINGS_MAX_BACKOFF
    budget_seconds: float = DEFAULT_EMBEDDINGS_RETRY_BUDGET

    def new_budget(self) -> "_RetryBudget":
        """Start a fresh retry budget, scoped to one logical embedding call."""
        return _RetryBudget(self.budget_seconds)


class _RetryBudget:
    """Mutable remaining-retry-time counter shared across the batches of one call."""

    __slots__ = ("remaining",)

    def __init__(self, seconds: float):
        self.remaining = max(0.0, seconds)

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0.0

    def spend(self, seconds: float) -> None:
        self.remaining = max(0.0, self.remaining - max(0.0, seconds))


def _status_code_of(exc: BaseException) -> int | None:
    """Best-effort HTTP status extraction across httpx, openai and litellm errors."""
    candidates = (
        getattr(exc, "status_code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    )
    for candidate in candidates:
        if isinstance(candidate, bool):
            continue
        if isinstance(candidate, int):
            return candidate
        if isinstance(candidate, str) and candidate.isdigit():
            return int(candidate)
    return None


def _is_transient_embedding_error(exc: BaseException) -> bool:
    """
    Return True when ``exc`` is worth retrying.

    A status code, when present, is authoritative: 5xx and the transient 4xx set
    are retryable, every other 4xx (401/403 auth, 400/422 validation, 404) is
    permanent and must fail fast. Without a status code we fall back to
    transport-level exception types and known SDK exception names.
    """
    status = _status_code_of(exc)
    if status is not None:
        return status >= 500 or status in _TRANSIENT_STATUS_CODES
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)):
        return True
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    return type(exc).__name__ in _TRANSIENT_EXCEPTION_NAMES


def _retry_delay_for(
    exc: BaseException,
    *,
    attempt: int,
    attempts: int,
    policy: EmbeddingRetryPolicy,
    budget: _RetryBudget,
    provider: str,
) -> float | None:
    """
    Decide whether ``exc`` should be retried and how long to wait first.

    Returns the sleep duration, or None when the exception must propagate. Logs
    the decision and charges the sleep to ``budget``. Upstream error text is
    truncated (matching the LLM providers) so a verbose provider payload cannot
    flood the log.
    """
    if not _is_transient_embedding_error(exc):
        return None

    status = _status_code_of(exc)
    status_label = f"HTTP {status}" if status is not None else type(exc).__name__
    detail = str(exc)[:200]

    if attempt >= attempts - 1:
        logger.error(f"{provider} embedding call failed after {attempts} attempt(s) ({status_label}): {detail}")
        return None

    if budget.exhausted:
        logger.error(
            f"{provider} embedding call failed on attempt {attempt + 1}/{attempts} ({status_label}) and the "
            f"{policy.budget_seconds:.1f}s retry budget is exhausted, giving up: {detail}"
        )
        return None

    backoff = min(policy.initial_backoff * (2**attempt), policy.max_backoff)
    jitter = backoff * 0.2 * (2 * (time.time() % 1) - 1)
    sleep_for = min(max(0.0, backoff + jitter), budget.remaining)
    budget.spend(sleep_for)
    logger.warning(
        f"{provider} embedding call failed (attempt {attempt + 1}/{attempts}, {status_label}), "
        f"retrying in {sleep_for:.2f}s ({budget.remaining:.1f}s of retry budget left): {detail}"
    )
    return sleep_for


def _call_with_retry(
    call: Callable[[], T],
    *,
    policy: EmbeddingRetryPolicy,
    budget: _RetryBudget,
    provider: str,
) -> T:
    """Run a blocking embedding call, retrying transient upstream failures."""
    attempts = policy.max_retries + 1
    for attempt in range(attempts):
        started = time.monotonic()
        try:
            return call()
        except Exception as exc:
            budget.spend(time.monotonic() - started)
            delay = _retry_delay_for(
                exc, attempt=attempt, attempts=attempts, policy=policy, budget=budget, provider=provider
            )
            if delay is None:
                raise
            time.sleep(delay)
    raise RuntimeError("unreachable: retry loop exited without returning or raising")


async def _acall_with_retry(
    call: Callable[[], Awaitable[T]],
    *,
    policy: EmbeddingRetryPolicy,
    budget: _RetryBudget,
    provider: str,
) -> T:
    """Async twin of :func:`_call_with_retry`, sharing its policy and classification."""
    attempts = policy.max_retries + 1
    for attempt in range(attempts):
        started = time.monotonic()
        try:
            return await call()
        except Exception as exc:
            budget.spend(time.monotonic() - started)
            delay = _retry_delay_for(
                exc, attempt=attempt, attempts=attempts, policy=policy, budget=budget, provider=provider
            )
            if delay is None:
                raise
            await asyncio.sleep(delay)
    raise RuntimeError("unreachable: retry loop exited without returning or raising")


ZeroEntropyInputType = Literal["document", "query"]
ZeroEntropyLatency = Literal["fast", "slow"]
ZeroEntropyEncodingFormat = Literal["float", "base64"]


class _ZeroEntropyEmbedRequest(BaseModel):
    """Typed request body for ZeroEntropy's non-OpenAI-compatible embed endpoint."""

    model: str
    input: list[str]
    input_type: ZeroEntropyInputType
    dimensions: int
    encoding_format: ZeroEntropyEncodingFormat = "float"
    latency: ZeroEntropyLatency | None = None


class _ZeroEntropyEmbedResult(BaseModel):
    embedding: list[float] | str


class _ZeroEntropyEmbedResponse(BaseModel):
    results: list[_ZeroEntropyEmbedResult]


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

    # Client-side asymmetric prefixes, empty unless a provider populates them from
    # config. Class-level so providers that never set them are unchanged.
    query_prefix: str = ""
    passage_prefix: str = ""

    def encode_query(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for query text, applying the configured query prefix."""
        return self._encode_prefixed(texts, self.query_prefix)

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for stored document text, applying the configured passage prefix."""
        return self._encode_prefixed(texts, self.passage_prefix)

    def _encode_prefixed(self, texts: list[str], prefix: str) -> list[list[float]]:
        """Prepend an asymmetric model's instruction before handing text to encode().

        Asymmetric models (E5, embeddinggemma, ...) expect a different instruction in
        front of a search than in front of stored text. A provider that is plain
        text-in/vector-out — TEI, LiteLLM, anything behind an OpenAI-compatible
        /embeddings endpoint — has no other channel to carry that distinction, so the
        client has to prepend it. Providers with a native mechanism (SentenceTransformers'
        own prompts, ZeroEntropy's input_type) override encode_query/encode_documents
        instead and never reach this. Empty prefixes leave the text byte-identical.
        """
        if prefix:
            return self.encode([f"{prefix}{text}" for text in texts])
        return self.encode(texts)


class LocalSTEmbeddings(Embeddings):
    """
    Local embeddings implementation using SentenceTransformers.

    Call initialize() during startup to load the model and avoid cold starts.
    The embedding dimension is auto-detected from the model.
    """

    def __init__(
        self,
        model_name: str | None = None,
        force_cpu: bool = False,
        trust_remote_code: bool = False,
        allow_mps: bool = False,
    ):
        """
        Initialize local SentenceTransformers embeddings.

        Args:
            model_name: Name of the SentenceTransformer model to use.
                       Default: BAAI/bge-small-en-v1.5
            force_cpu: Force CPU mode for local inference.
                      Default: False
            trust_remote_code: Allow loading models with custom code (security risk).
                              Required for some models with custom architectures.
                              Default: False (disabled for security)
            allow_mps: Opt in to the Apple Silicon MPS GPU. Disabled by default
                      because MPS leaks memory under variable-length workloads
                      (see engine/local_device.py). Default: False
        """
        self.model_name = model_name or DEFAULT_EMBEDDINGS_LOCAL_MODEL
        self.force_cpu = force_cpu
        self.trust_remote_code = trust_remote_code
        self.allow_mps = allow_mps
        self._model = None
        self._dimension: int | None = None
        self._device_type: str = "cpu"

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

        # Determine device based on hardware availability. We always set
        # low_cpu_mem_usage=False to prevent lazy loading (meta tensors) which can
        # cause issues when accelerate is installed but no GPU is available.
        # MPS is opt-in (allow_mps) — see engine/local_device.py for why.
        device = select_local_device(self.force_cpu, self.allow_mps)

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
        self._device_type = resolve_model_device_type(self._model)
        logger.info(f"Embeddings: local provider initialized (dim: {self._dimension}, device: {self._device_type})")

    def encode(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to encode

        Returns:
            List of embedding vectors
        """
        return self._encode_local(texts)

    def encode_query(self, texts: list[str]) -> list[list[float]]:
        return self._encode_local(texts, input_type="query")

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode_local(texts, input_type="document")

    def _encode_local(
        self, texts: list[str], input_type: Literal["query", "document"] | None = None
    ) -> list[list[float]]:
        if self._model is None:
            raise RuntimeError("Embeddings not initialized. Call initialize() first.")

        try:
            # Delegate to SentenceTransformers' own asymmetric entry points rather than
            # prefixing here: they apply whatever prompts the model ships with (and route
            # the task for models exposing a Router module), so asymmetric models such as
            # Qwen3-Embedding get their configured query prompt without Hindsight carrying
            # per-model prefix config the way the ONNX provider has to. Models that declare
            # no prompts are unaffected — SentenceTransformers defaults them to empty
            # strings and skips prompt handling entirely, so this is byte-identical to
            # encode() for e.g. the default BAAI/bge-small-en-v1.5.
            # encode_query/encode_document exist only in sentence-transformers >= 5.0,
            # which is why local-ml pins that floor.
            if input_type == "query":
                encode = self._model.encode_query
            elif input_type == "document":
                encode = self._model.encode_document
            else:
                encode = self._model.encode
            embeddings = encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return [emb.tolist() for emb in embeddings]
        finally:
            # Only reclaim the GPU allocator pool here, and only when actually on a
            # GPU (opt-in MPS/CUDA/XPU). encode() runs in tight retain loops, so a
            # gc.collect()/malloc_trim on every call is too costly on the CPU default
            # — and unnecessary: refcounting frees the small transient buffers
            # immediately and the allocator reuses them for the next batch. (The
            # reranker keeps its per-batch heap trim for the #1717 CPU case; it runs
            # on the lighter recall path.) See engine/local_device.py.
            if self._device_type != "cpu":
                release_local_inference_memory(self._device_type)


class OnnxEmbeddings(Embeddings):
    """Local ONNX Runtime embeddings provider.

    This provider runs transformer embedding models in-process with ONNX Runtime,
    avoiding a sidecar Ollama/TEI server or a remote embeddings API. It supports
    sentence-transformer style mean pooling and E5-style asymmetric prefixes.
    """

    def __init__(
        self,
        model_id: str,
        model_path: str | None = None,
        tokenizer_name_or_path: str | None = None,
        onnx_file: str = "onnx/model.onnx",
        dimensions: int | None = None,
        max_tokens: int = 512,
        pooling: str = "mean",
        normalize: bool = True,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        output_name: str | None = None,
        batch_size: int = DEFAULT_EMBEDDINGS_ONNX_BATCH_SIZE,
        cpu_mem_arena: bool = DEFAULT_EMBEDDINGS_ONNX_CPU_MEM_ARENA,
    ):
        self.model_id = model_id
        self.model_path = model_path
        if model_path and tokenizer_name_or_path is None:
            logger.warning(
                "Embeddings: ONNX model_path is set without tokenizer_name_or_path; "
                "falling back to tokenizer from model_id %s. Set "
                "HINDSIGHT_API_EMBEDDINGS_ONNX_TOKENIZER_NAME_OR_PATH when using local ONNX artifacts.",
                model_id,
            )
        self.tokenizer_name_or_path = tokenizer_name_or_path or model_id
        self.onnx_file = onnx_file
        self.configured_dimensions = dimensions
        self.max_tokens = max_tokens
        self.pooling = pooling.lower()
        if self.pooling not in {"mean", "cls"}:
            raise ValueError("ONNX embeddings pooling must be 'mean' or 'cls'")
        self.normalize = normalize
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.output_name = output_name
        if batch_size < 1:
            raise ValueError("ONNX embeddings batch_size must be >= 1")
        self.batch_size = batch_size
        self.cpu_mem_arena = cpu_mem_arena
        self._session = None
        self._tokenizer = None
        self._dimension: int | None = dimensions

    @property
    def provider_name(self) -> str:
        return "onnx"

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise RuntimeError("Embeddings not initialized. Call initialize() first.")
        return self._dimension

    async def initialize(self) -> None:
        if self._session is not None and self._tokenizer is not None:
            return

        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "onnxruntime and transformers are required for OnnxEmbeddings. "
                "Install with: pip install 'hindsight-api-slim[local-onnx]'"
            ) from exc

        model_path = self.model_path
        if not model_path:
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise ImportError(
                    "huggingface-hub is required to download ONNX embedding models. "
                    "Set HINDSIGHT_API_EMBEDDINGS_ONNX_MODEL_PATH or install local-onnx."
                ) from exc
            # Some large ONNX exports, for example BAAI/bge-m3, store weights in
            # an external sidecar file next to model.onnx. Download both the
            # requested graph and its conventional *_data sidecar when present.
            snapshot_dir = snapshot_download(
                repo_id=self.model_id,
                allow_patterns=[self.onnx_file, f"{self.onnx_file}_data"],
            )
            model_path = os.path.join(snapshot_dir, self.onnx_file)

        logger.info(
            "Embeddings: initializing ONNX provider with model %s (%s)",
            self.model_id,
            model_path,
        )
        logger.info(
            "Embeddings: ONNX query_prefix=%r passage_prefix=%r pooling=%s normalize=%s batch_size=%s cpu_mem_arena=%s",
            self.query_prefix,
            self.passage_prefix,
            self.pooling,
            self.normalize,
            self.batch_size,
            self.cpu_mem_arena,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)
        # With the arena enabled (ORT's default) freed activation blocks are cached and
        # never returned to the OS, so RSS ratchets up to the largest batch ever run and
        # holds that plateau for the life of the process. The reranker disables it for
        # the same reason (see FlashRankCrossEncoder).
        session_options = None
        if not self.cpu_mem_arena:
            session_options = ort.SessionOptions()
            session_options.enable_cpu_mem_arena = False
        self._session = ort.InferenceSession(
            model_path, sess_options=session_options, providers=["CPUExecutionProvider"]
        )

        detected = len(self.encode(["test"])[0])
        if self.configured_dimensions is not None and detected != self.configured_dimensions:
            raise ValueError(
                f"Configured ONNX embedding dimension {self.configured_dimensions} does not match model output {detected}"
            )
        self._dimension = detected
        logger.info("Embeddings: ONNX provider initialized (dim: %s)", self._dimension)

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` in bounded forward passes.

        Every remote provider slices its input before calling out; this one runs the
        model in-process, so nothing downstream bounds it and a caller that hands over
        a whole bank's worth of text gets a single ``[n_texts x max_seq_len x hidden]``
        float32 activation tensor (issue #3891).
        """
        if self._session is None or self._tokenizer is None:
            raise RuntimeError("Embeddings not initialized. Call initialize() first.")
        if not texts:
            return []
        if len(texts) <= self.batch_size:
            return self._encode_batch(texts)

        # Pack similar-length texts together: tokenization pads every text in a call up
        # to the longest one in that same call, so one long text otherwise inflates the
        # tensor for every short text batched with it. Character length is a cheap
        # stand-in for token count — it only decides grouping, never the output, since
        # both pooling modes mask padding and so cannot see batch composition.
        order = sorted(range(len(texts)), key=lambda index: len(texts[index]), reverse=True)
        embeddings: list[list[float]] = [[] for _ in texts]
        for start in range(0, len(order), self.batch_size):
            window = order[start : start + self.batch_size]
            batch = self._encode_batch([texts[index] for index in window])
            for index, embedding in zip(window, batch, strict=True):
                embeddings[index] = embedding
        return embeddings

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Run one forward pass over at most ``batch_size`` texts."""
        assert self._session is not None and self._tokenizer is not None

        import numpy as np

        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="np",
        )
        input_names = {inp.name for inp in self._session.get_inputs()}
        ort_inputs = {name: value for name, value in encoded.items() if name in input_names}
        if "token_type_ids" in input_names and "token_type_ids" not in ort_inputs:
            ort_inputs["token_type_ids"] = np.zeros_like(encoded["input_ids"])

        outputs = self._session.run([self.output_name] if self.output_name else None, ort_inputs)
        token_embeddings = outputs[0]

        # Some exported models expose a pooled 2-D embedding as their first output.
        if getattr(token_embeddings, "ndim", 0) == 2:
            embeddings = token_embeddings
        elif self.pooling == "cls":
            embeddings = token_embeddings[:, 0]
        else:
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = np.ones(token_embeddings.shape[:2], dtype=np.float32)
            mask = attention_mask[..., None].astype(np.float32)
            summed = (token_embeddings * mask).sum(axis=1)
            counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
            embeddings = summed / counts

        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings = embeddings / norms

        return embeddings.astype(float).tolist()


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
        query_prefix: str = "",
        passage_prefix: str = "",
    ):
        """
        Initialize remote TEI embeddings client.

        Args:
            base_url: Base URL of the TEI server (e.g., "http://localhost:8080")
            timeout: Request timeout in seconds (default: 30.0)
            batch_size: Maximum batch size for embedding requests (default: 32)
            max_retries: Maximum number of retries for failed requests (default: 3)
            retry_delay: Initial delay between retries in seconds, doubles each retry (default: 0.5)
            query_prefix: Prefix prepended to recall/search queries (default: none)
            passage_prefix: Prefix prepended to retained document text (default: none)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
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
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout) as e:
                last_error = e
                if attempt < self.max_retries:
                    logger.warning(
                        f"TEI request failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}. Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
            except httpx.RequestError as e:
                if not is_retryable_tei_transport_error(e):
                    raise
                last_error = e
                if attempt < self.max_retries:
                    logger.warning(
                        f"TEI request failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}. Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
            except OSError as e:
                if not is_retryable_tei_transport_error(e):
                    raise
                last_error = e
                if attempt < self.max_retries:
                    logger.warning(
                        f"TEI request failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}. Retrying in {delay}s..."
                    )
                    time.sleep(delay)
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
                    time.sleep(sleep_delay)
                    delay *= 2
                else:
                    raise

        raise last_error

    async def initialize(self) -> None:
        """Initialize the HTTP client and verify server connectivity."""
        if self._client is not None:
            return

        logger.info(f"Embeddings: initializing TEI provider at {self.base_url}")
        self._client = httpx.Client(
            timeout=self.timeout,
            limits=httpx.Limits(keepalive_expiry=min(self.timeout, TEI_KEEPALIVE_EXPIRY_SECONDS)),
        )

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
        dimensions: int | None = None,
        max_retries: int = 3,
        query_prefix: str = "",
        passage_prefix: str = "",
    ):
        """
        Initialize OpenAI embeddings client.

        Args:
            api_key: OpenAI API key
            model: OpenAI embedding model name (default: text-embedding-3-small)
            base_url: Custom base URL for OpenAI-compatible API (e.g., Azure OpenAI endpoint)
            batch_size: Maximum batch size for embedding requests (default: 100)
            dimensions: Optional requested output dimensions for OpenAI text-embedding-3 models
            max_retries: Maximum number of retries for failed requests (default: 3)
            query_prefix: Prefix prepended to recall/search queries (default: none)
            passage_prefix: Prefix prepended to retained document text (default: none)
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.batch_size = batch_size
        self.dimensions = dimensions
        self.max_retries = max_retries
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
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
        # Parse query parameters from base_url (e.g. ?api-version=xxx for Azure OpenAI)
        # and pass them as default_query so they're included in every request.
        client_kwargs = {"api_key": self.api_key, "max_retries": self.max_retries}
        if self.base_url:
            parsed = urlparse(self.base_url)
            if parsed.query:
                clean_url = urlunparse(parsed._replace(query=""))
                client_kwargs["base_url"] = clean_url
                default_query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
                client_kwargs["default_query"] = default_query
                self.base_url = clean_url
            else:
                client_kwargs["base_url"] = self.base_url
        self._client = OpenAI(**client_kwargs)

        # Try to get dimension from known models, otherwise do a test embedding
        if self.dimensions is not None:
            self._dimension = self.dimensions
        elif self.model in self.MODEL_DIMENSIONS:
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

            request = {
                "model": self.model,
                "input": batch,
            }
            if self.dimensions is not None:
                request["dimensions"] = self.dimensions
            apply_bank_attribution(request)

            response = self._client.embeddings.create(**request)

            # Sort by index to ensure correct order
            batch_embeddings = sorted(response.data, key=lambda x: x.index)
            all_embeddings.extend([e.embedding for e in batch_embeddings])

        return all_embeddings


class CodexOAuthEmbeddings(OpenAIEmbeddings):
    """
    OpenAI embeddings using the Codex/ChatGPT OAuth token from the Codex
    ``auth.json`` (``$CODEX_HOME/auth.json``, or ``~/.codex/auth.json`` when unset).

    Codex OAuth is an LLM-provider auth path in Hindsight, but the same bearer token
    can also authenticate against the standard OpenAI embeddings endpoint. This keeps
    embeddings on the user's existing Codex subscription/OAuth path without requiring
    a separate OpenAI/OpenRouter/Gemini/Cohere API key.

    Token refresh is handled automatically: the manager proactively refreshes the
    access_token before it expires and reactively refreshes on 401 responses from
    the embeddings API.
    """

    def __init__(
        self,
        model: str = DEFAULT_EMBEDDINGS_OPENAI_MODEL,
        batch_size: int = 100,
        dimensions: int | None = None,
        max_retries: int = 3,
        query_prefix: str = "",
        passage_prefix: str = "",
    ):
        from .providers.codex_auth import CodexAuthManager

        self._auth_manager = CodexAuthManager.from_file()
        super().__init__(
            api_key=self._auth_manager.access_token,
            model=model,
            base_url="https://api.openai.com/v1",
            batch_size=batch_size,
            dimensions=dimensions,
            max_retries=max_retries,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
        )

    @property
    def provider_name(self) -> str:
        return "openai-codex"

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings, refreshing the OAuth token if needed.

        Proactively refreshes before the call when the token is near expiry,
        and reactively refreshes once on a 401 from the OpenAI embeddings API.
        """
        from openai import AuthenticationError

        # Proactive refresh — cheap when fresh (JWT exp decode + compare).
        self._auth_manager.ensure_fresh_token()
        if self._auth_manager.access_token != self.api_key:
            self.api_key = self._auth_manager.access_token
            if self._client is not None:
                self._client.api_key = self._auth_manager.access_token

        try:
            return super().encode(texts)
        except AuthenticationError:
            # Reactive refresh — token was valid by the JWT clock but the
            # server rejected it (rotated server-side, race, etc.).
            self._auth_manager.refresh_tokens(
                reason="reactive (401 from embeddings API)",
                force=True,
            )
            self.api_key = self._auth_manager.access_token
            if self._client is not None:
                self._client.api_key = self._auth_manager.access_token
            return super().encode(texts)


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
        output_dimensions: int | None = None,
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
            output_dimensions: Optional output embedding dimensions (for Matryoshka-capable models)
            batch_size: Maximum batch size for embedding requests (default: 96, Cohere's limit)
            timeout: Request timeout in seconds (default: 60.0)
            input_type: Input type for embeddings (default: search_document).
                       Options: search_document, search_query, classification, clustering
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.output_dimensions = output_dimensions
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

        # If output_dimensions is explicitly set, use that as the dimension
        if self.output_dimensions is not None:
            self._dimension = self.output_dimensions
        elif self.model in self.MODEL_DIMENSIONS:
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

            if self.output_dimensions is not None:
                # Use v2 API which supports output_dimension
                response = self._client.v2.embed(
                    texts=batch,
                    model=self.model,
                    input_type=self.input_type,
                    output_dimension=self.output_dimensions,
                    embedding_types=["float"],
                )
                all_embeddings.extend(response.embeddings.float_)
            else:
                response = self._client.embed(
                    texts=batch,
                    model=self.model,
                    input_type=self.input_type,
                )
                all_embeddings.extend(response.embeddings)

        return all_embeddings


class ZeroEntropyEmbeddings(Embeddings):
    """
    ZeroEntropy embeddings implementation using the zembed API.

    ZeroEntropy's embeddings endpoint is not OpenAI-compatible: it lives at
    /v1/models/embed and requires provider-specific parameters such as
    input_type. Hindsight stores document-side vectors and uses query-side
    vectors during recall, so this provider exposes explicit encode_documents()
    and encode_query() helpers while keeping encode() as document-side default.
    """

    VALID_DIMENSIONS = frozenset({2560, 1280, 640, 320, 160, 80, 40})
    VALID_ENCODING_FORMATS = frozenset({"float", "base64"})
    VALID_LATENCIES = frozenset({"fast", "slow"})
    DEFAULT_BASE_URL = DEFAULT_ZEROENTROPY_BASE_URL
    EMBED_PATH = "/v1/models/embed"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_EMBEDDINGS_ZEROENTROPY_MODEL,
        base_url: str | None = None,
        dimensions: int = DEFAULT_EMBEDDINGS_ZEROENTROPY_DIMENSIONS,
        batch_size: int = DEFAULT_EMBEDDINGS_ZEROENTROPY_BATCH_SIZE,
        encoding_format: str = DEFAULT_EMBEDDINGS_ZEROENTROPY_ENCODING_FORMAT,
        latency: str | None = DEFAULT_EMBEDDINGS_ZEROENTROPY_LATENCY,
        timeout: float = 60.0,
    ):
        if dimensions not in self.VALID_DIMENSIONS:
            valid = ", ".join(str(dim) for dim in sorted(self.VALID_DIMENSIONS, reverse=True))
            raise ValueError(f"{ENV_EMBEDDINGS_ZEROENTROPY_DIMENSIONS} must be one of {valid}, got {dimensions}")
        if batch_size < 1:
            raise ValueError("ZeroEntropy embeddings batch_size must be >= 1")
        if encoding_format not in self.VALID_ENCODING_FORMATS:
            valid_formats = ", ".join(sorted(self.VALID_ENCODING_FORMATS))
            raise ValueError(
                f"{ENV_EMBEDDINGS_ZEROENTROPY_ENCODING_FORMAT} must be one of {valid_formats}, got {encoding_format!r}"
            )
        if latency is not None and latency not in self.VALID_LATENCIES:
            valid_latencies = ", ".join(sorted(self.VALID_LATENCIES))
            raise ValueError(f"ZeroEntropy embeddings latency must be one of {valid_latencies}, got {latency!r}")

        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/") if base_url else self.DEFAULT_BASE_URL
        self.embed_url = f"{self.base_url}{self.EMBED_PATH}"
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.encoding_format = cast(ZeroEntropyEncodingFormat, encoding_format)
        self.latency = cast(ZeroEntropyLatency | None, latency)
        self.timeout = timeout
        self._client: httpx.Client | None = None
        self._dimension: int | None = None

    @property
    def provider_name(self) -> str:
        return "zeroentropy"

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise RuntimeError("Embeddings not initialized. Call initialize() first.")
        return self._dimension

    async def initialize(self) -> None:
        """Initialize the ZeroEntropy HTTP client."""
        if self._client is not None:
            return

        logger.info(
            f"Embeddings: initializing ZeroEntropy provider with model {self.model} "
            f"(dim: {self.dimensions}, batch_size={self.batch_size})"
        )
        self._client = httpx.Client(
            timeout=self.timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        # zembed-1 dimensions are explicit Matryoshka truncation steps. Avoid a
        # startup probe so boot does not burn quota or require a throwaway input.
        self._dimension = self.dimensions
        logger.info(f"Embeddings: ZeroEntropy provider initialized (model: {self.model}, dim: {self._dimension})")

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Generate document-side embeddings for backwards-compatible callers."""
        return self.encode_documents(texts)

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate document-side embeddings for retained content."""
        return self._encode_with_input_type(texts, "document")

    def encode_query(self, texts: list[str]) -> list[list[float]]:
        """Generate query-side embeddings for recall/search queries."""
        return self._encode_with_input_type(texts, "query")

    def _encode_with_input_type(self, texts: list[str], input_type: ZeroEntropyInputType) -> list[list[float]]:
        if self._client is None:
            raise RuntimeError("Embeddings not initialized. Call initialize() first.")

        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            request = _ZeroEntropyEmbedRequest(
                model=self.model,
                input=batch,
                input_type=input_type,
                dimensions=self.dimensions,
                encoding_format=self.encoding_format,
                latency=self.latency,
            )

            try:
                response = self._client.post(self.embed_url, json=request.model_dump(exclude_none=True))
                response.raise_for_status()
            except httpx.HTTPError as e:
                raise RuntimeError(f"ZeroEntropy embedding request failed: {e}") from e

            parsed = _ZeroEntropyEmbedResponse.model_validate(response.json())
            if len(parsed.results) != len(batch):
                raise RuntimeError(
                    f"ZeroEntropy returned {len(parsed.results)} embeddings for {len(batch)} input texts; "
                    "expected exact 1:1 alignment"
                )
            all_embeddings.extend(self._parse_embedding(result.embedding) for result in parsed.results)

        return all_embeddings

    @staticmethod
    def _parse_embedding(embedding: list[float] | str) -> list[float]:
        if not isinstance(embedding, str):
            return embedding

        raw = base64.b64decode(embedding)
        if len(raw) % 4 != 0:
            raise RuntimeError("ZeroEntropy returned invalid base64 embedding length")
        return list(struct.unpack(f"<{len(raw) // 4}f", raw))


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

    The embedding dimension is auto-detected from the model at initialization,
    or declared up front via ``dimensions`` to skip that startup probe.
    """

    def __init__(
        self,
        api_base: str = DEFAULT_LITELLM_API_BASE,
        api_key: str | None = None,
        model: str = DEFAULT_EMBEDDINGS_LITELLM_MODEL,
        batch_size: int = 100,
        dimensions: int | None = None,
        timeout: float = 60.0,
        query_prefix: str = "",
        passage_prefix: str = "",
        retry_policy: EmbeddingRetryPolicy | None = None,
    ):
        """
        Initialize LiteLLM embeddings client.

        Args:
            api_base: Base URL of the LiteLLM proxy (default: http://localhost:4000)
            api_key: API key for the LiteLLM proxy (optional, depends on proxy config)
            model: Embedding model name (default: text-embedding-3-small)
                   Use provider prefix for non-OpenAI models (e.g., cohere/embed-english-v3.0)
            batch_size: Maximum batch size for embedding requests (default: 100)
            dimensions: Vector width this proxy/model returns. When set, the startup
                     probe is skipped entirely, so the API boots even while the proxy
                     is still coming up (issue #3695). This *declares* the width
                     rather than requesting it: the value is not forwarded to the
                     proxy, because the backends LiteLLM fronts (vLLM, Cohere, Voyage,
                     HuggingFace) reject an unexpected "dimensions" field. A wrong
                     value is caught on the first encode() instead of silently
                     producing vectors pgvector will reject.
            timeout: Request timeout in seconds (default: 60.0)
            query_prefix: Prefix prepended to recall/search queries (default: none)
            passage_prefix: Prefix prepended to retained document text (default: none)
            retry_policy: Bounded retry policy for transient upstream failures
                (default: EmbeddingRetryPolicy() built-in defaults)
        """
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.dimensions = dimensions
        self.timeout = timeout
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.retry_policy = retry_policy or EmbeddingRetryPolicy()
        self._client: httpx.Client | None = None
        self._dimension: int | None = None
        self._dimension_verified = False

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

        # An explicitly declared width means we never have to reach the proxy to
        # learn it, so the API boots even while the proxy is still starting up.
        if self.dimensions is not None:
            self._dimension = self.dimensions
            logger.info(
                f"Embeddings: LiteLLM provider initialized "
                f"(model: {self.model}, dim: {self._dimension}, declared; startup probe skipped)"
            )
            return

        # Do a test embedding to detect dimension.
        def probe():
            resp = self._client.post(
                f"{self.api_base}/embeddings",
                json={"model": self.model, "input": ["test"]},
            )
            # Inside the retried closure so a 5xx from a proxy that is still
            # coming up is retried rather than crash-looping the daemon (#3695).
            resp.raise_for_status()
            return resp.json()

        try:
            # to_thread + the async retry helper: initialize() is awaited on the
            # event loop, so a blocking post and a time.sleep() backoff would
            # stall the model loads running concurrently with it and freeze the
            # model_init_timeout watchdog that is supposed to bound this.
            result = await _acall_with_retry(
                lambda: asyncio.to_thread(probe),
                policy=self.retry_policy,
                budget=self.retry_policy.new_budget(),
                provider=self.provider_name,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to connect to LiteLLM proxy at {self.api_base}: {e}") from e

        if not result.get("data"):
            raise RuntimeError(
                f"LiteLLM proxy at {self.api_base} returned no embedding data for model "
                f"{self.model}; cannot detect the vector dimension. Set "
                f"{ENV_EMBEDDINGS_LITELLM_DIMENSIONS} to declare it explicitly."
            )
        self._dimension = len(result["data"][0]["embedding"])
        logger.info(f"Embeddings: LiteLLM provider initialized (model: {self.model}, dim: {self._dimension})")

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

        # One retry budget for the whole call: batching must not multiply the
        # worst-case added latency of a single encode().
        budget = self.retry_policy.new_budget()

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            def post_batch(payload_batch=batch):
                response = self._client.post(
                    f"{self.api_base}/embeddings",
                    json={"model": self.model, "input": payload_batch},
                )
                # Inside the retried closure so a 5xx from the proxy is retried
                # rather than raised straight through to the caller.
                response.raise_for_status()
                return response.json()

            result = _call_with_retry(
                post_batch,
                policy=self.retry_policy,
                budget=budget,
                provider=self.provider_name,
            )

            # Sort by index to ensure correct order
            batch_embeddings = sorted(result["data"], key=lambda x: x["index"])
            all_embeddings.extend([e["embedding"] for e in batch_embeddings])

        self._check_declared_dimension(all_embeddings)
        return all_embeddings

    def _check_declared_dimension(self, embeddings: list[list[float]]) -> None:
        """
        Validate a declared dimension against what the proxy actually returns.

        Declaring the width skips the startup probe, which means nothing has
        verified the number until the first real embedding comes back. Left
        unchecked, a wrong value surfaces much later as an opaque pgvector
        "expected N dimensions, not M" on insert, or as a bank whose vector
        column was created at the wrong width. Checked once here, the operator
        gets told which env var to correct.
        """
        if self.dimensions is None or self._dimension_verified or not embeddings:
            return
        actual = len(embeddings[0])
        self._dimension_verified = True
        if actual != self.dimensions:
            raise RuntimeError(
                f"{ENV_EMBEDDINGS_LITELLM_DIMENSIONS} declares {self.dimensions} dimensions but "
                f"model {self.model} on the LiteLLM proxy at {self.api_base} returned {actual}. "
                f"Correct the value, or unset it to let startup detect the dimension."
            )


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
        api_key: str | None = None,
        model: str = DEFAULT_EMBEDDINGS_LITELLM_SDK_MODEL,
        api_base: str | None = None,
        output_dimensions: int | None = None,
        batch_size: int = 100,
        timeout: float = 60.0,
        encoding_format: str | None = "float",
        query_prefix: str = "",
        passage_prefix: str = "",
        retry_policy: EmbeddingRetryPolicy | None = None,
    ):
        """
        Initialize LiteLLM SDK embeddings client.

        Args:
            api_key: API key for the embedding provider (optional — omit for
                     providers that use ambient credentials, e.g. AWS Bedrock with IAM)
            model: Model name with provider prefix (e.g., "cohere/embed-english-v3.0")
            api_base: Custom base URL for API (optional)
            output_dimensions: Optional output embedding dimensions (provider-dependent)
            batch_size: Maximum batch size for embedding requests (default: 100)
            timeout: Request timeout in seconds (default: 60.0)
            encoding_format: Encoding format for embeddings (default: "float").
                Set to None or empty string to omit (needed for Voyage AI, Gemini).
            query_prefix: Prefix prepended to recall/search queries (default: none)
            passage_prefix: Prefix prepended to retained document text (default: none)
            retry_policy: Bounded retry policy for transient upstream failures
                (default: EmbeddingRetryPolicy() built-in defaults)
        """
        self.api_key = api_key
        self.model = model
        self.api_base = api_base
        self.output_dimensions = output_dimensions
        self.batch_size = batch_size
        self.timeout = timeout
        self.encoding_format = encoding_format or None
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.retry_policy = retry_policy or EmbeddingRetryPolicy()
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
                # Without this litellm falls back to its own (much larger) default
                # timeout, so a stalled provider would hang startup for minutes.
                "timeout": self.timeout,
            }
            if self.api_key:
                embed_kwargs["api_key"] = self.api_key
            if self.encoding_format:
                embed_kwargs["encoding_format"] = self.encoding_format
            if self.api_base:
                embed_kwargs["api_base"] = self.api_base
            if self.output_dimensions is not None:
                embed_kwargs["dimensions"] = self.output_dimensions
                if self.model.startswith("openai/"):
                    embed_kwargs["allowed_openai_params"] = ["dimensions"]
            if self.model.startswith("voyage/"):
                embed_kwargs["input_type"] = "document"

            # Use async embedding method (standard in litellm). Retried on transient
            # upstream errors: a flaky provider must not take the whole API down at
            # startup, since dimension detection gates initialization.
            response = await _acall_with_retry(
                lambda: self._litellm.aembedding(**embed_kwargs),
                policy=self.retry_policy,
                budget=self.retry_policy.new_budget(),
                provider=self.provider_name,
            )

            # Extract dimension from response
            if response.data and len(response.data) > 0:
                self._dimension = len(response.data[0]["embedding"])
            else:
                raise RuntimeError(f"Unable to detect embedding dimension for model {self.model}")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize LiteLLM SDK embeddings: {e}")

        logger.info(f"Embeddings: LiteLLM SDK provider initialized (model: {self.model}, dim: {self._dimension})")

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings with provider-default semantics."""
        return self._encode_with_input_type(texts)

    def encode_query(self, texts: list[str]) -> list[list[float]]:
        """Generate query-side embeddings for asymmetric Voyage retrieval."""
        input_type = "query" if self.model.startswith("voyage/") else None
        return self._encode_with_input_type(texts, input_type)

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate document-side embeddings for asymmetric Voyage retrieval."""
        input_type = "document" if self.model.startswith("voyage/") else None
        return self._encode_with_input_type(texts, input_type)

    def _encode_with_input_type(
        self, texts: list[str], input_type: Literal["query", "document"] | None = None
    ) -> list[list[float]]:
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

        # One retry budget for the whole call: batching must not multiply the
        # worst-case added latency of a single encode().
        budget = self.retry_policy.new_budget()

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            try:
                # Build kwargs for embedding call
                embed_kwargs = {
                    "model": self.model,
                    "input": batch,
                    # Without this litellm falls back to its own (much larger)
                    # default timeout, which would let one stalled request hang a
                    # synchronous recall far past the retry budget.
                    "timeout": self.timeout,
                }
                if self.api_key:
                    embed_kwargs["api_key"] = self.api_key
                if self.encoding_format:
                    embed_kwargs["encoding_format"] = self.encoding_format
                if self.api_base:
                    embed_kwargs["api_base"] = self.api_base
                if self.output_dimensions is not None:
                    embed_kwargs["dimensions"] = self.output_dimensions
                    if self.model.startswith("openai/"):
                        embed_kwargs["allowed_openai_params"] = ["dimensions"]
                if input_type is not None:
                    embed_kwargs["input_type"] = input_type

                # Use sync embedding (litellm doesn't have async in thread-safe way).
                # Recall runs this inline, so transient upstream failures are retried
                # here rather than surfacing as a failed recall.
                response = _call_with_retry(
                    lambda kwargs=embed_kwargs: self._litellm.embedding(**kwargs),
                    policy=self.retry_policy,
                    budget=budget,
                    provider=self.provider_name,
                )

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


# Gemini Embedding 2+ multimodal models return a SINGLE aggregated embedding
# for a multi-input request instead of one vector per input (see
# https://ai.google.dev/gemini-api/docs/embeddings#embedding-aggregation). For
# these models we must embed one input per call to preserve the 1:1 input→vector
# alignment the rest of the pipeline relies on. The marker matches preview and GA
# names (e.g. "gemini-embedding-2-preview", "gemini-embedding-2"), with or
# without a "google/" or "models/" prefix.
_GEMINI_AGGREGATING_MODEL_MARKER = "gemini-embedding-2"


def _gemini_model_aggregates_inputs(model: str) -> bool:
    """Whether the model aggregates a multi-input request into one embedding."""
    return _GEMINI_AGGREGATING_MODEL_MARKER in model.lower()


class GeminiEmbeddings(Embeddings):
    """
    Google embeddings via the google.genai SDK.

    Supports both:
    1. Gemini API (api.generativeai.google.com) with API key authentication
    2. Vertex AI with service account or Application Default Credentials (ADC)

    Uses the embed_content API: client.models.embed_content(model, contents)

    Gemini Embedding 2+ multimodal models aggregate a multi-input request into a
    single embedding, so for those the batch size is forced to 1 (one input per
    call) to keep one vector per input.
    """

    def __init__(
        self,
        model: str = DEFAULT_EMBEDDINGS_GEMINI_MODEL,
        api_key: str | None = None,
        vertexai_project_id: str | None = None,
        vertexai_region: str | None = None,
        vertexai_service_account_key: str | None = None,
        output_dimensionality: int | None = None,
        batch_size: int = 100,
        force_ipv4: bool = False,
    ):
        self.model = model
        self.api_key = api_key
        self.vertexai_project_id = vertexai_project_id
        self.vertexai_region = vertexai_region or "us-central1"
        self.vertexai_service_account_key = vertexai_service_account_key
        self.output_dimensionality = output_dimensionality
        self.batch_size = batch_size
        self.force_ipv4 = force_ipv4
        self._client = None
        self._httpx_client = None
        self._dimension: int | None = None
        self._is_vertexai = vertexai_project_id is not None
        self._embed_config = None  # EmbedContentConfig, built during initialize()

    @property
    def provider_name(self) -> str:
        return "google"

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise RuntimeError("Embeddings not initialized. Call initialize() first.")
        return self._dimension

    async def initialize(self) -> None:
        """Initialize the Google genai client and detect embedding dimension."""
        if self._client is not None:
            return

        from google import genai
        from google.genai import types as genai_types

        if self._is_vertexai:
            self._init_vertexai(genai)
        else:
            self._init_gemini(genai, genai_types)

        # Build EmbedContentConfig if output_dimensionality is set
        if self.output_dimensionality is not None:
            self._embed_config = genai_types.EmbedContentConfig(
                output_dimensionality=self.output_dimensionality,
            )

        # Detect dimension via a test embedding (respects output_dimensionality)
        embed_kwargs = {"model": self.model, "contents": ["test"]}
        if self._embed_config is not None:
            embed_kwargs["config"] = self._embed_config

        result = self._client.models.embed_content(**embed_kwargs)  # type: ignore[union-attr]
        if result.embeddings and len(result.embeddings) > 0:
            self._dimension = len(result.embeddings[0].values)

        auth_mode = "vertex_ai" if self._is_vertexai else "api_key"
        logger.info(
            f"Embeddings: google provider initialized (auth: {auth_mode}, model: {self.model}, dim: {self._dimension})"
        )

    def _init_gemini(self, genai, genai_types) -> None:
        """Initialize Gemini API client with API key."""
        if not self.api_key:
            raise ValueError("Gemini embeddings provider requires an API key")

        client_kwargs = {"api_key": self.api_key}
        if self.force_ipv4:
            import httpx

            self._httpx_client = httpx.Client(
                timeout=10,
                transport=httpx.HTTPTransport(local_address="0.0.0.0"),
            )
            client_kwargs["http_options"] = genai_types.HttpOptions(
                timeout=10000,
                httpxClient=self._httpx_client,
            )

        self._client = genai.Client(**client_kwargs)
        logger.info(f"Embeddings: initializing Gemini provider with model {self.model}")

    def _init_vertexai(self, genai) -> None:
        """Initialize Vertex AI client with project, region, and credentials."""
        if not self.vertexai_project_id:
            raise ValueError(
                "HINDSIGHT_API_EMBEDDINGS_VERTEXAI_PROJECT_ID (or HINDSIGHT_API_LLM_VERTEXAI_PROJECT_ID) "
                "is required for Vertex AI embeddings provider."
            )

        auth_method = "ADC"
        credentials = None

        if self.vertexai_service_account_key:
            try:
                from google.oauth2 import service_account
            except ImportError:
                raise ImportError(
                    "Vertex AI service account auth requires 'google-auth' package. "
                    "Install with: pip install google-auth"
                )
            credentials = service_account.Credentials.from_service_account_file(
                self.vertexai_service_account_key,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            auth_method = "service_account"
            logger.info(f"Embeddings: Vertex AI using service account key: {self.vertexai_service_account_key}")

        # Strip google/ prefix from model name — native SDK uses bare names
        if self.model.startswith("google/"):
            self.model = self.model[len("google/") :]

        client_kwargs = {
            "vertexai": True,
            "project": self.vertexai_project_id,
            "location": self.vertexai_region,
        }
        if credentials is not None:
            client_kwargs["credentials"] = credentials

        self._client = genai.Client(**client_kwargs)
        logger.info(
            f"Embeddings: initializing Vertex AI provider "
            f"(project={self.vertexai_project_id}, region={self.vertexai_region}, "
            f"model={self.model}, auth={auth_method})"
        )

    def encode(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings using the Google genai SDK.

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

        # Gemini Embedding 2+ multimodal models return one aggregated vector for a
        # multi-input request, so embed one input per call to keep 1:1 alignment.
        batch_size = 1 if _gemini_model_aggregates_inputs(self.model) else self.batch_size

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            embed_kwargs = {"model": self.model, "contents": batch}
            if self._embed_config is not None:
                embed_kwargs["config"] = self._embed_config

            result = self._client.models.embed_content(**embed_kwargs)

            embeddings = result.embeddings or []
            if len(embeddings) != len(batch):
                raise RuntimeError(
                    f"Gemini embeddings backend returned {len(embeddings)} vectors for "
                    f"{len(batch)} input texts (model {self.model}); expected exact 1:1 alignment"
                )
            all_embeddings.extend([emb.values for emb in embeddings])

        # L2-normalize when output_dimensionality is set — Gemini only returns
        # normalized vectors at full 3072 dims; truncated dims need re-normalization
        # for accurate cosine similarity.
        if self.output_dimensionality is not None:
            import numpy as np

            arr = np.array(all_embeddings)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1
            all_embeddings = (arr / norms).tolist()

        return all_embeddings


def _retry_policy_from_config(config: "HindsightConfig") -> EmbeddingRetryPolicy:
    """Build the embedding retry policy from resolved configuration."""
    return EmbeddingRetryPolicy(
        max_retries=config.embeddings_max_retries,
        initial_backoff=config.embeddings_initial_backoff,
        max_backoff=config.embeddings_max_backoff,
        budget_seconds=config.embeddings_retry_budget,
    )


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

    # Asymmetric prefixes are handed only to the providers that are plain
    # text-in/vector-out. `local` and `zeroentropy` carry the distinction natively
    # (SentenceTransformers prompts / input_type) and `onnx` has its own pair with
    # non-empty E5 defaults, so none of them take these.
    query_prefix = config.embeddings_query_prefix
    passage_prefix = config.embeddings_passage_prefix
    if query_prefix or passage_prefix:
        logger.info(
            "Embeddings: asymmetric prefixes configured (query=%r, passage=%r)",
            query_prefix,
            passage_prefix,
        )

    if provider == "tei":
        url = config.embeddings_tei_url
        if not url:
            raise ValueError(f"{ENV_EMBEDDINGS_TEI_URL} is required when {ENV_EMBEDDINGS_PROVIDER} is 'tei'")
        return RemoteTEIEmbeddings(
            base_url=url,
            batch_size=config.embeddings_tei_batch_size,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
        )
    elif provider == "local":
        return LocalSTEmbeddings(
            model_name=config.embeddings_local_model,
            force_cpu=config.embeddings_local_force_cpu,
            trust_remote_code=config.embeddings_local_trust_remote_code,
            allow_mps=config.embeddings_local_allow_mps,
        )
    elif provider == "onnx":
        return OnnxEmbeddings(
            model_id=config.embeddings_onnx_model_id,
            model_path=config.embeddings_onnx_model_path,
            tokenizer_name_or_path=config.embeddings_onnx_tokenizer_name_or_path,
            onnx_file=config.embeddings_onnx_file,
            dimensions=config.embeddings_onnx_dimensions,
            max_tokens=config.embeddings_onnx_max_tokens,
            pooling=config.embeddings_onnx_pooling,
            normalize=config.embeddings_onnx_normalize,
            query_prefix=config.embeddings_onnx_query_prefix,
            passage_prefix=config.embeddings_onnx_passage_prefix,
            output_name=config.embeddings_onnx_output_name,
            batch_size=config.embeddings_onnx_batch_size,
            cpu_mem_arena=config.embeddings_onnx_cpu_mem_arena,
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
        return OpenAIEmbeddings(
            api_key=api_key,
            model=model,
            base_url=base_url,
            batch_size=config.embeddings_openai_batch_size,
            dimensions=config.embeddings_openai_dimensions,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
        )
    elif provider == "openai-codex":
        model = os.environ.get(ENV_EMBEDDINGS_OPENAI_MODEL, DEFAULT_EMBEDDINGS_OPENAI_MODEL)
        return CodexOAuthEmbeddings(
            model=model,
            batch_size=config.embeddings_openai_batch_size,
            dimensions=config.embeddings_openai_dimensions,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
        )
    elif provider == "openrouter":
        api_key = config.embeddings_openrouter_api_key
        if not api_key:
            raise ValueError(
                "HINDSIGHT_API_EMBEDDINGS_OPENROUTER_API_KEY, HINDSIGHT_API_OPENROUTER_API_KEY, "
                f"or {ENV_LLM_API_KEY} is required when {ENV_EMBEDDINGS_PROVIDER} is 'openrouter'"
            )
        return OpenAIEmbeddings(
            api_key=api_key,
            model=config.embeddings_openrouter_model,
            base_url="https://openrouter.ai/api/v1",
            batch_size=config.embeddings_openai_batch_size,
            dimensions=config.embeddings_openai_dimensions,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
        )
    elif provider == "requesty":
        api_key = config.embeddings_requesty_api_key
        if not api_key:
            raise ValueError(
                "HINDSIGHT_API_EMBEDDINGS_REQUESTY_API_KEY, HINDSIGHT_API_REQUESTY_API_KEY, "
                f"or {ENV_LLM_API_KEY} is required when {ENV_EMBEDDINGS_PROVIDER} is 'requesty'"
            )
        return OpenAIEmbeddings(
            api_key=api_key,
            model=config.embeddings_requesty_model,
            base_url="https://router.requesty.ai/v1",
            batch_size=config.embeddings_openai_batch_size,
            dimensions=config.embeddings_openai_dimensions,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
        )
    elif provider == "zeroentropy":
        api_key = config.embeddings_zeroentropy_api_key
        if not api_key:
            raise ValueError(
                f"{ENV_EMBEDDINGS_ZEROENTROPY_API_KEY} or ZEROENTROPY_API_KEY is required "
                f"when {ENV_EMBEDDINGS_PROVIDER} is 'zeroentropy'"
            )
        return ZeroEntropyEmbeddings(
            api_key=api_key,
            model=config.embeddings_zeroentropy_model,
            base_url=config.embeddings_zeroentropy_base_url,
            dimensions=config.embeddings_zeroentropy_dimensions,
            batch_size=config.embeddings_zeroentropy_batch_size,
            encoding_format=config.embeddings_zeroentropy_encoding_format,
            latency=config.embeddings_zeroentropy_latency,
        )
    elif provider == "cohere":
        api_key = config.embeddings_cohere_api_key
        if not api_key:
            raise ValueError(f"{ENV_EMBEDDINGS_COHERE_API_KEY} is required when {ENV_EMBEDDINGS_PROVIDER} is 'cohere'")
        return CohereEmbeddings(
            api_key=api_key,
            model=config.embeddings_cohere_model,
            base_url=config.embeddings_cohere_base_url,
            output_dimensions=config.embeddings_cohere_output_dimensions,
        )
    elif provider == "litellm":
        return LiteLLMEmbeddings(
            api_base=config.embeddings_litellm_api_base,
            api_key=config.embeddings_litellm_api_key,
            model=config.embeddings_litellm_model,
            dimensions=config.embeddings_litellm_dimensions,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
            retry_policy=_retry_policy_from_config(config),
        )
    elif provider == "litellm-sdk":
        return LiteLLMSDKEmbeddings(
            api_key=config.embeddings_litellm_sdk_api_key or None,
            model=config.embeddings_litellm_sdk_model,
            api_base=config.embeddings_litellm_sdk_api_base,
            output_dimensions=config.embeddings_litellm_sdk_output_dimensions,
            encoding_format=config.embeddings_litellm_sdk_encoding_format,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
            retry_policy=_retry_policy_from_config(config),
        )
    elif provider == "google":
        vertexai_project_id = config.embeddings_vertexai_project_id
        if vertexai_project_id:
            api_key = None  # Vertex AI uses ADC or service account
        else:
            api_key = config.embeddings_gemini_api_key
            if not api_key:
                raise ValueError(
                    f"{ENV_EMBEDDINGS_GEMINI_API_KEY} or {ENV_LLM_API_KEY} is required "
                    f"when {ENV_EMBEDDINGS_PROVIDER} is 'google' (set VERTEXAI_PROJECT_ID for Vertex AI auth instead)"
                )
        return GeminiEmbeddings(
            model=config.embeddings_gemini_model,
            api_key=api_key,
            vertexai_project_id=vertexai_project_id,
            vertexai_region=config.embeddings_vertexai_region,
            vertexai_service_account_key=config.embeddings_vertexai_service_account_key,
            output_dimensionality=config.embeddings_gemini_output_dimensionality,
            force_ipv4=config.embeddings_gemini_force_ipv4,
        )
    else:
        raise ValueError(
            f"Unknown embeddings provider: {provider}. "
            f"Supported: 'local', 'onnx', 'tei', 'openai', 'openai-codex', 'openrouter', 'requesty', 'cohere', 'google', "
            f"'zeroentropy', 'litellm', 'litellm-sdk'"
        )
