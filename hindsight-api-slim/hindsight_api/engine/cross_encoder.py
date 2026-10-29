"""
Cross-encoder abstraction for reranking.

Provides an interface for reranking with different backends.

Configuration via environment variables - see hindsight_api.config for all env var names.
"""

import asyncio
import contextvars
import logging
import os
import threading
import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar, cast

import aiohttp
from pydantic import BaseModel

from .._cross_loop import CrossLoopSemaphore
from .._thread_limits import bound_onnx_session, onnx_intra_op_threads
from ..config import (
    DEFAULT_LITELLM_API_BASE,
    DEFAULT_RERANKER_ALIBABA_MODEL,
    DEFAULT_RERANKER_COHERE_MAX_CONCURRENT,
    DEFAULT_RERANKER_COHERE_MODEL,
    DEFAULT_RERANKER_COHERE_TIMEOUT,
    DEFAULT_RERANKER_FLASHRANK_BATCH_SIZE,
    DEFAULT_RERANKER_FLASHRANK_CACHE_DIR,
    DEFAULT_RERANKER_FLASHRANK_MODEL,
    DEFAULT_RERANKER_GOOGLE_MODEL,
    DEFAULT_RERANKER_HINDSIGHT_MODEL,
    DEFAULT_RERANKER_LITELLM_MODEL,
    DEFAULT_RERANKER_LITELLM_SDK_MODEL,
    DEFAULT_RERANKER_LOCAL_BATCH_SIZE,
    DEFAULT_RERANKER_LOCAL_MODEL,
    DEFAULT_RERANKER_LOCAL_TIMEOUT,
    DEFAULT_RERANKER_NVIDIA_BASE_URL,
    DEFAULT_RERANKER_NVIDIA_BATCH_SIZE,
    DEFAULT_RERANKER_NVIDIA_MAX_CONCURRENT,
    DEFAULT_RERANKER_NVIDIA_MODEL,
    DEFAULT_RERANKER_ONNX_BATCH_SIZE,
    DEFAULT_RERANKER_ONNX_CPU_MEM_ARENA,
    DEFAULT_RERANKER_ONNX_FILE,
    DEFAULT_RERANKER_ONNX_MAX_CONCURRENT,
    DEFAULT_RERANKER_ONNX_MAX_TOKENS,
    DEFAULT_RERANKER_ONNX_THREADS,
    DEFAULT_RERANKER_SILICONFLOW_BASE_URL,
    DEFAULT_RERANKER_SILICONFLOW_MODEL,
    DEFAULT_RERANKER_TEI_BATCH_SIZE,
    DEFAULT_RERANKER_TEI_MAX_CONCURRENT,
    DEFAULT_RERANKER_TYPESAFE_BASE_URL,
    DEFAULT_RERANKER_TYPESAFE_MAX_CONCURRENT,
    DEFAULT_RERANKER_TYPESAFE_MODEL,
    DEFAULT_RERANKER_TYPESAFE_PRUNE_CANDIDATES,
    DEFAULT_RERANKER_TYPESAFE_TIMEOUT,
    DEFAULT_RERANKER_ZEROENTROPY_MODEL,
    DEFAULT_ZEROENTROPY_BASE_URL,
    HINDSIGHT_INFERENCE_URL,
    RerankerMemberConfig,
)
from .aiohttp_session import LoopLocal, LoopLocalSession, UpstreamHTTPError, per_phase_timeout, raise_for_status
from .bank_attribution import reranker_bank_attribution_headers
from .decision_model import overflowed_window, record_decision_call
from .llm_trace import RERANK_SCOPE, trace_model_call
from .local_device import (
    LOCAL_MODEL_BUILD_LOCK,
    align_local_model_weights,
    assert_finite_local_output,
    prefetch_local_model,
    release_local_inference_memory,
    resolve_model_device_type,
    select_local_device,
)
from .remote_retry import RetryPolicy, acall_with_retry
from .tei_retry import TEI_KEEPALIVE_EXPIRY_SECONDS, is_retryable_tei_transport_error, tei_retry_delay
from .token_encoding import count_tokens, truncate_to_tokens

logger = logging.getLogger(__name__)
_T = TypeVar("_T")

# Which member produced the scores for the predict() running in this task.
# MultiCrossEncoder._active is shared by every request on the chain, so reading
# it after await rerank can observe a neighbour's failover. This is set in the
# same task that is about to return those scores, and rerank() copies it onto
# the result before yielding.
_served_provider: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "hindsight_rerank_served_provider", default=None
)

# Providers whose score is a candidate's rank position within one pool, not a
# relevance score for the pair: the top candidate is 1.0 on every recall however
# weak it is, so a fixed floor just keeps a fixed share of the pool and the value
# is not comparable between recalls (#4901). Recall rejects min_scores.reranker
# for these and publishes no reranker score.
# "decision_model" is recall's decision-model reranker (engine/search/decision_reranker.py).
RANK_SCORE_PROVIDERS = frozenset({"typesafe", "decision_model"})

# The recall this rerank belongs to, so a failover warning can be joined to the
# recall's own log lines when several are in flight. Set by recall_async.
recall_log_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("hindsight_recall_log_id", default=None)


class RerankTimeoutError(Exception):
    """An in-process reranker ran out of wall-clock before scoring every pair.

    Carries the partial result: ``scores[i]`` is the score for ``pairs[i]``, or
    ``None`` where the budget ran out first. Callers decide what to do with the
    unscored tail — :class:`~hindsight_api.engine.search.reranking.Reranker`
    keeps those candidates in their pre-rerank (RRF) order behind the scored ones,
    so a mis-sized local model degrades the ordering instead of never returning.
    """

    def __init__(self, scores: list[float | None], timeout: float, model_name: str):
        self.scores = scores
        self.timeout = timeout
        self.model_name = model_name
        scored = sum(1 for score in scores if score is not None)
        super().__init__(
            f"Reranker {model_name!r} exhausted its {timeout:g}s budget after scoring {scored}/{len(scores)} candidates"
        )


async def _run_batches_in_executor(
    executor: ThreadPoolExecutor | None,
    predict_sync: Callable[[list[tuple[str, str]], threading.Event], list[float]],
    pairs: list[tuple[str, str]],
) -> list[float]:
    """Run an in-process reranker's batch loop on ``executor``, stopping it on cancel.

    Cancelling the awaiting task cannot interrupt a worker thread, so without this an
    abandoned rerank kept scoring its remaining batches and the next request queued
    behind them (#5722). ``predict_sync`` gets an event that is set on cancellation and
    must check it before each batch; the batch already running still finishes.
    """
    cancelled = threading.Event()
    try:
        return await asyncio.get_running_loop().run_in_executor(executor, predict_sync, pairs, cancelled)
    except asyncio.CancelledError:
        cancelled.set()
        raise


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

    @property
    def primary_provider_name(self) -> str:
        """The provider that serves requests while nothing has failed over."""
        return self.provider_name

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the cross-encoder model asynchronously.

        This should be called during startup to load/connect to the model
        and avoid cold start latency on first predict() call.
        """
        pass

    # Bounded retry for this member's remote calls, or None to call straight through.
    # Set by create_cross_encoder for the remote providers; the in-process backends,
    # the rrf passthrough, TEI (own retry loop) and MultiCrossEncoder (its members
    # retry individually) leave it None. Living on the base class rather than in each
    # provider is deliberate: retry then applies to the one method every backend has
    # to implement, so a new remote reranker inherits it instead of having to
    # remember — the omission that let a single Cohere 429 fail an entire recall
    # (#4134).
    retry_policy: RetryPolicy | None = None

    # Whether this backend prunes candidates it judges irrelevant, marking each with
    # a score of exactly 0.0 for the caller to leave out. Ordinary rerankers only
    # order candidates — they have no calibrated notion of "not relevant", so their
    # lowest score still means "least bad of these" and must be kept. Leave False
    # unless the score is a real decision.
    prunes_candidates: bool = False

    # Cap each candidate at this many tokens before it reaches the backend, or None to
    # send it whole. Set by create_cross_encoder from the member's config; lives on the
    # base class for the same reason as retry_policy, so every provider honors it.
    max_tokens_per_candidate: int | None = None

    # Whether predict() traces itself as one ``rerank`` row. Off for a backend that makes
    # no model call (rrf), wraps backends that trace themselves (the failover chain), or
    # traces each decision-model request it makes instead.
    traces_calls: bool = True

    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """
        Score query-document pairs for relevance, retrying transient failures.

        Args:
            pairs: List of (query, document) tuples to score

        Returns:
            List of relevance scores (higher = more relevant)
        """
        if self.max_tokens_per_candidate is not None and pairs:
            docs = _truncate_docs_to_tokens([doc for _, doc in pairs], self.max_tokens_per_candidate)
            pairs = [(query, doc) for (query, _), doc in zip(pairs, docs)]
        if not self.traces_calls:
            return await self._predict_with_retry(pairs)
        return await trace_model_call(
            self,
            RERANK_SCOPE,
            len(pairs),
            lambda: self._predict_with_retry(pairs),
            request={"queries": list(dict.fromkeys(q for q, _ in pairs)), "documents": [d for _, d in pairs]},
            output_of=lambda scores: {"scores": [float(score) for score in scores]},
        )

    async def _predict_with_retry(self, pairs: list[tuple[str, str]]) -> list[float]:
        if self.retry_policy is None:
            return await self._predict(pairs)
        # One budget per predict(): rerank is one logical call on the synchronous
        # recall path, so the worst-case added latency has to stay bounded.
        return await acall_with_retry(
            lambda: self._predict(pairs),
            policy=self.retry_policy,
            budget=self.retry_policy.new_budget(),
            provider=self.provider_name,
        )

    async def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score ``pairs`` without retry — what each backend in this module implements.

        Deliberately not ``@abstractmethod``: ``CrossEncoderModel`` is exported from
        ``hindsight_api``, and a subclass that overrides ``predict`` directly (the test
        doubles here, and any downstream implementation) must keep working — it simply
        opts out of the shared retry. Every backend in this module implements
        ``_predict`` instead, which ``TestEveryRemoteRerankerRetries`` enforces so the
        opt-out cannot happen by accident.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _predict() (or override predict() to opt out of retry)"
        )


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
        timeout: float = DEFAULT_RERANKER_LOCAL_TIMEOUT,
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
            fp16: Use FP16 (half precision) inference. Faster on CUDA,
                  may be slower on CPU. Default: False (opt-in via env var).
            bucket_batching: Sort pairs by token length before batching to reduce
                            padding waste. 36-54% speedup, quality-identical.
                            Default: False (opt-in via env var).
            batch_size: Batch size for predict() calls. Optimal values vary by
                       hardware and model (CPU: 32, CUDA: 128+). Default: 32.
            timeout: Wall-clock ceiling for scoring one call's pairs. On expiry
                    predict() raises RerankTimeoutError with the partial scores
                    instead of running to completion. 0 disables. Default: 300.
        """
        self.model_name = model_name or DEFAULT_RERANKER_LOCAL_MODEL
        self.force_cpu = force_cpu
        self.trust_remote_code = trust_remote_code
        self.fp16 = fp16
        self.bucket_batching = bucket_batching
        self.batch_size = batch_size
        self.timeout = timeout
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

        # Imports transformers; see LOCAL_MODEL_BUILD_LOCK.
        with LOCAL_MODEL_BUILD_LOCK:
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
        # cause issues when accelerate is installed but no GPU is available. Only
        # transformers 4.x honours it: 5.x drops the kwarg and always builds on meta.
        # Note: We do NOT use device_map because CrossEncoder internally calls .to(device)
        # after loading, which conflicts with accelerate's device_map handling.
        # MPS is never used — see engine/local_device.py for why.
        device = select_local_device(self.force_cpu)

        # Patch transformers 5.x compatibility for models using XLM-RoBERTa
        # (e.g., jina-reranker-v2-base-multilingual). transformers 5.x removed
        # create_position_ids_from_input_ids as a module-level function; the custom
        # code in these models still references it. This monkey-patch restores it.
        # Imports transformers; see LOCAL_MODEL_BUILD_LOCK.
        with LOCAL_MODEL_BUILD_LOCK:
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

        # Download outside the lock; build and smoke-test under it, because a
        # transformers build flips process-global state (see LOCAL_MODEL_BUILD_LOCK).
        # sentence-transformers caches under SENTENCE_TRANSFORMERS_HOME when it is set.
        prefetch_local_model(self.model_name, cache_dir=os.getenv("SENTENCE_TRANSFORMERS_HOME"))
        with LOCAL_MODEL_BUILD_LOCK:
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

            # Safetensors weights are mapped zero-copy and can land on a byte offset that
            # is not a multiple of the dtype size, which makes torch's vectorized CPU matmul
            # return corrupt scores. See engine/local_device.py. The default reranker
            # (ms-marco-MiniLM-L-6-v2) is one of the affected files.
            align_local_model_weights(self._model.model, label=f"Reranker[{self.model_name}]")

            self._device_type = resolve_model_device_type(self._model)

            # FP16 inference: convert model weights to half precision.
            # Empirically validated: 27-36% faster on MPS, quality-identical (20/20 overlap).
            if self.fp16 and self._device_type != "cpu":
                self._model.model.half()
                logger.info("Reranker: FP16 inference enabled")

            # Smoke-test the fully configured model (after any FP16 conversion). NaN logits
            # are sanitized to 0.0 downstream, so startup is the last point at which a model
            # that computes garbage is still observable.
            assert_finite_local_output(
                self._model.predict([("hindsight startup probe", "hindsight startup probe")]),
                label=f"Reranker[{self.model_name}]",
            )

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

    def _predict_sync(self, pairs: list[tuple[str, str]], cancelled: threading.Event | None = None) -> list[float]:
        """Synchronous prediction wrapper for thread pool execution.

        Scores in explicit batches rather than one `predict()` call so the wall-clock
        budget can be enforced between them — a blocking call already inside torch
        cannot be cancelled from the event loop, and abandoning it with `wait_for`
        would leave it burning the executor's only worker (#4696). ``cancelled`` is
        checked at the same point (see ``_run_batches_in_executor``).

        Supports two optimizations (controlled via .env):
        - bucket_batching: sort pairs by token length to reduce padding waste (36-54% speedup)
        - batch_size: explicit batch size for predict() calls (MPS optimal: 32)
        """

        try:
            order = list(range(len(pairs)))
            if self.bucket_batching and len(pairs) > 1:
                # Sort pairs by approximate token length to create homogeneous batches.
                # This eliminates padding waste — short pairs aren't padded to the length
                # of the longest pair in the batch. Quality-identical by construction.
                order.sort(key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))

            scores: list[float | None] = [None] * len(pairs)
            # ponytail: the deadline is checked between batches, so one pathological
            # batch can overshoot it. Bounded by a single batch, which is the point —
            # cutting mid-batch would mean reaching inside the model's forward pass.
            deadline = time.monotonic() + self.timeout if self.timeout > 0 else None

            for start in range(0, len(order), self.batch_size):
                if cancelled is not None and cancelled.is_set():
                    raise asyncio.CancelledError()
                if deadline is not None and time.monotonic() >= deadline:
                    raise RerankTimeoutError(scores, self.timeout, self.model_name)
                batch = order[start : start + self.batch_size]
                batch_pairs = [pairs[i] for i in batch]
                batch_scores = self._model.predict(batch_pairs, batch_size=self.batch_size, show_progress_bar=False)
                batch_scores = batch_scores.tolist() if hasattr(batch_scores, "tolist") else list(batch_scores)
                for i, score in zip(batch, batch_scores):
                    scores[i] = float(score)

            return [0.0 if score is None else score for score in scores]
        finally:
            release_local_inference_memory(self._device_type)

    async def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
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
        return await _run_batches_in_executor(LocalSTCrossEncoder._executor, self._predict_sync, pairs)


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
    _global_semaphore: CrossLoopSemaphore | None = None
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
        # Idle pooled sockets are dropped before TEI (or a proxy in front of it) can
        # close them under us, which would otherwise surface as a failed request.
        self._session = LoopLocalSession(
            timeout=per_phase_timeout(timeout),
            connector_factory=lambda: aiohttp.TCPConnector(
                keepalive_timeout=min(timeout, TEI_KEEPALIVE_EXPIRY_SECONDS)
            ),
        )
        # Set by a successful initialize(); None means not initialized.
        self._model_id: str | None = None

        # Update global semaphore if max_concurrent changed
        if (
            RemoteTEICrossEncoder._global_semaphore is None
            or RemoteTEICrossEncoder._global_max_concurrent != max_concurrent
        ):
            RemoteTEICrossEncoder._global_max_concurrent = max_concurrent
            # CrossLoopSemaphore, not asyncio.Semaphore: this is a CLASS attribute, so
            # it is shared by every event loop in the process and an asyncio.Semaphore
            # would bind to whichever loop first waits on it.
            RemoteTEICrossEncoder._global_semaphore = CrossLoopSemaphore(max_concurrent)

    @property
    def provider_name(self) -> str:
        return "tei"

    async def _async_request_with_retry(
        self,
        semaphore: asyncio.Semaphore | CrossLoopSemaphore,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> Any:
        """Make an async HTTP request with automatic retries on transient errors and semaphore for backpressure.

        Returns the decoded JSON body.
        """
        last_error: BaseException | None = None
        delay = self.retry_delay
        session = self._session.get()

        async with semaphore:
            for attempt in range(self.max_retries + 1):
                try:
                    async with session.request(method, url, **kwargs) as response:
                        await raise_for_status(response)
                        return await response.json(content_type=None)
                except (aiohttp.ClientConnectorError, aiohttp.ServerTimeoutError, TimeoutError) as e:
                    last_error = e
                    if attempt < self.max_retries:
                        logger.warning(
                            f"TEI request failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                            f"Retrying in {delay}s..."
                        )
                        await asyncio.sleep(delay)
                        delay *= 2  # Exponential backoff
                except aiohttp.ClientError as e:
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
                except UpstreamHTTPError as e:
                    # TEI uses 429 as normal overload backpressure. Retry it with
                    # the same bounded budget as transient server errors.
                    if (e.status_code == 429 or e.status_code >= 500) and attempt < self.max_retries:
                        last_error = e
                        sleep_delay = tei_retry_delay(
                            e.headers,
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

        assert last_error is not None  # the loop ran at least once and only exits via an error
        raise last_error

    async def initialize(self) -> None:
        """Verify server connectivity and fetch the model id.

        The HTTP session itself is created lazily per event loop on first use.
        """
        if self._model_id is not None:
            return

        logger.info(
            f"Reranker: initializing TEI provider at {self.base_url} "
            f"(batch_size={self.batch_size}, max_concurrent={self.max_concurrent})"
        )

        # Verify server is reachable and get model info
        # Use a temporary semaphore for initialization
        init_semaphore = asyncio.Semaphore(1)
        try:
            info = await self._async_request_with_retry(init_semaphore, "GET", f"{self.base_url}/info")
            self._model_id = info.get("model_id", "unknown")
            logger.info(f"Reranker: TEI provider initialized (model: {self._model_id})")
        except (aiohttp.ClientError, UpstreamHTTPError, TimeoutError) as e:
            raise RuntimeError(f"Failed to connect to TEI server at {self.base_url}: {e}")

    async def _rerank_query_group(
        self,
        semaphore: asyncio.Semaphore | CrossLoopSemaphore,
        query: str,
        texts: list[str],
    ) -> list[tuple[int, float]]:
        """Rerank a single query group and return list of (original_index, score) tuples."""
        try:
            results = await self._async_request_with_retry(
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
            # TEI returns results sorted by score descending, with original index
            return [(result["index"], result["score"]) for result in results]
        except (aiohttp.ClientError, UpstreamHTTPError, TimeoutError) as e:
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
        # Built lazily above and non-None from that point on; the class attribute stays Optional
        # because it is unset until the first predict sizes it from config.
        semaphore = cast(CrossLoopSemaphore, RemoteTEICrossEncoder._global_semaphore)

        tasks = [self._rerank_query_group(semaphore, query, texts) for query, _, texts in tasks_info]
        results = await asyncio.gather(*tasks)

        # Map scores back to original positions
        for (_, indices, _), result_scores in zip(tasks_info, results):
            for original_idx_in_batch, score in result_scores:
                global_idx = indices[original_idx_in_batch]
                all_scores[global_idx] = score

        return all_scores

    async def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """
        Score query-document pairs using the remote TEI reranker.

        Requests are made in parallel with configurable backpressure.

        Args:
            pairs: List of (query, document) tuples to score

        Returns:
            List of relevance scores
        """
        if self._model_id is None:
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
        self.initialized = False
        self._session = LoopLocalSession(
            timeout=per_phase_timeout(timeout),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

    async def initialize(self) -> None:
        # Nothing to connect: the session is created lazily per event loop on first use.
        self.initialized = True

    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not self.initialized:
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

            async with self._session.get().post(
                self.rerank_url,
                headers=reranker_bank_attribution_headers(),
                json=body,
            ) as response:
                await raise_for_status(response)
                result = await response.json(content_type=None)

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
        max_concurrent: int = DEFAULT_RERANKER_COHERE_MAX_CONCURRENT,
    ):
        """
        Initialize Cohere cross-encoder client.

        Args:
            api_key: Cohere API key
            model: Cohere rerank model name (default: rerank-english-v3.0)
            base_url: Custom base URL for Cohere-compatible API (e.g., Azure-hosted endpoint)
            timeout: Request timeout in seconds (default: 60.0)
            max_concurrent: Rerank requests in flight at once across every recall in the
                process; the rest wait here before sending.
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        # Held around the request only, so the timeout covers server time, not the wait
        # behind other recalls: without it a burst queues at the server, waits there past
        # the timeout and retries, adding load exactly when it is saturated.
        # CrossLoopSemaphore for the same reason as TypeSafeCrossEncoder's cap.
        self._semaphore = CrossLoopSemaphore(max_concurrent or DEFAULT_RERANKER_COHERE_MAX_CONCURRENT)
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
        if self._client is not None or (self._http_client and self._http_client.initialized):
            return

        base_url_msg = f" at {self.base_url}" if self.base_url else ""
        logger.info(
            f"Reranker: initializing Cohere provider with model {self.model}{base_url_msg} "
            f"(max_concurrent={self.max_concurrent})"
        )

        if self._http_client is not None:
            await self._http_client.initialize()
            logger.info("Reranker: Cohere provider initialized (Cohere-compatible HTTP endpoint)")
        else:
            # For native Cohere API, use the official SDK
            try:
                import cohere
            except ImportError:
                raise ImportError("cohere is required for CohereCrossEncoder. Install it with: pip install cohere")

            # The async client, awaited on the loop: the sync one would need a thread per
            # in-flight rerank. One per loop, because its pooled connections belong to the
            # loop that opened them (the sync client it replaced had no loop affinity).
            self._client = LoopLocal(lambda: cohere.AsyncClient(api_key=self.api_key, timeout=self.timeout))
            logger.info("Reranker: Cohere provider initialized")

    async def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
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

        query_groups: dict[str, list[tuple[int, str]]] = {}
        for idx, (query, text) in enumerate(pairs):
            query_groups.setdefault(query, []).append((idx, text))

        # One request per query, never split: a server may return ranks rather than
        # independent scores (e.g. 1 - rank/n), and those only compare within one request.
        groups = list(query_groups.items())
        results = await asyncio.gather(*(self._rerank_group(query, [t for _, t in items]) for query, items in groups))

        all_scores = [0.0] * len(pairs)
        for (_, items), scores in zip(groups, results):
            for (idx, _), score in zip(items, scores):
                all_scores[idx] = score
        return all_scores

    async def _rerank_group(self, query: str, texts: list[str]) -> list[float]:
        """Score ``texts`` against one query in a single request, within the concurrency cap."""
        async with self._semaphore:
            if self._http_client is not None:
                return await self._http_client.predict([(query, text) for text in texts])
            response = await self._client.get().rerank(
                query=query,
                documents=texts,
                model=self.model,
                return_documents=False,
            )
        scores = [0.0] * len(texts)
        for result in response.results:
            scores[result.index] = result.relevance_score
        return scores


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
        if self._client.initialized:
            return
        logger.info(f"Reranker: initializing ZeroEntropy provider with model {self.model}")
        await self._client.initialize()
        logger.info("Reranker: ZeroEntropy provider initialized")

    async def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
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
        if self._client.initialized:
            return
        logger.info(f"Reranker: initializing SiliconFlow provider at {self.base_url} with model {self.model}")
        await self._client.initialize()
        logger.info("Reranker: SiliconFlow provider initialized")

    async def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return await self._client.predict(pairs)


class _NvidiaText(BaseModel):
    text: str


class _NvidiaRerankRequest(BaseModel):
    model: str
    query: _NvidiaText
    passages: list[_NvidiaText]


class _NvidiaRanking(BaseModel):
    index: int
    logit: float


class _NvidiaRerankResponse(BaseModel):
    rankings: list[_NvidiaRanking]


class NvidiaCrossEncoder(CrossEncoderModel):
    """
    NVIDIA NIM cross-encoder using the native reranking API.

    Not Cohere-compatible, so it cannot reuse _CohereCompatibleRerankClient:
    the path carries the model (/retrieval/<model>/reranking), the query is
    {"text": ...}, each passage is {"text": ...}, and the response is
    rankings[].logit — a raw logit, not a 0..1 relevance_score. Raw logits are
    fine: each is a per-pair score, so they compare across batches and calls, and
    recall's sigmoid normalization maps them into 0..1.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_RERANKER_NVIDIA_MODEL,
        base_url: str = DEFAULT_RERANKER_NVIDIA_BASE_URL,
        timeout: float = 60.0,
        batch_size: int = DEFAULT_RERANKER_NVIDIA_BATCH_SIZE,
        max_concurrent: int = DEFAULT_RERANKER_NVIDIA_MAX_CONCURRENT,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.rerank_url = f"{self.base_url}/retrieval/{model.strip('/')}/reranking"
        self.timeout = timeout
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        # CrossLoopSemaphore, not asyncio.Semaphore: the encoder is built once and
        # reached from every loop in the process — same reasoning as TypeSafeCrossEncoder.
        self._semaphore = CrossLoopSemaphore(max_concurrent)
        self._session = LoopLocalSession(
            timeout=per_phase_timeout(timeout),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )

    @property
    def provider_name(self) -> str:
        return "nvidia"

    async def initialize(self) -> None:
        # Nothing to connect: the session is created lazily per event loop on first use.
        logger.info(
            f"Reranker: initializing NVIDIA provider at {self.rerank_url} "
            f"(batch_size={self.batch_size}, max_concurrent={self.max_concurrent})"
        )

    async def _rerank_batch(self, query: str, texts: list[str]) -> list[float]:
        request = _NvidiaRerankRequest(
            model=self.model, query=_NvidiaText(text=query), passages=[_NvidiaText(text=t) for t in texts]
        )
        async with self._semaphore:
            async with self._session.get().post(
                self.rerank_url,
                headers=reranker_bank_attribution_headers(),
                json=request.model_dump(),
            ) as response:
                await raise_for_status(response)
                result = _NvidiaRerankResponse.model_validate(await response.json(content_type=None))
        scores = [0.0] * len(texts)
        for ranking in result.rankings:
            scores[ranking.index] = ranking.logit
        return scores

    async def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        # One request per distinct query (like the Cohere-compatible client), split
        # into batch_size slices because the API caps passages per request; slices
        # run concurrently, bounded by max_concurrent across all calls.
        query_groups: dict[str, list[int]] = {}
        for idx, (query, _) in enumerate(pairs):
            query_groups.setdefault(query, []).append(idx)

        slices = [
            (query, indices[i : i + self.batch_size])
            for query, indices in query_groups.items()
            for i in range(0, len(indices), self.batch_size)
        ]
        # create_task copies contextvars, so the per-bank attribution header reaches every slice.
        tasks = [
            asyncio.create_task(self._rerank_batch(query, [pairs[idx][1] for idx in indices]))
            for query, indices in slices
        ]
        try:
            # Awaited in order, so the earliest failing slice is the one raised; its
            # siblings are cancelled rather than left running unobserved.
            results = [await task for task in tasks]
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        all_scores = [0.0] * len(pairs)
        for (_, indices), scores in zip(slices, results):
            for idx, score in zip(indices, scores):
                all_scores[idx] = score
        return all_scores


# Single source of truth for prompt templates and overheads
_RANK_INSTRUCTIONS_PREFIX = "Which candidate answers the question: "
_CUT_INSTRUCTIONS = (
    "How far down this ranked list does genuine relevance to the question extend? "
    "Count a candidate as relevant only if it helps answer the question."
)

_SUFFICIENCY_INSTRUCTIONS = (
    "Does the evidence above answer the question? Judge only what the evidence states, "
    "not what could be guessed from it."
)

# The page-impact gate answers with the single most likely option, and anything but
# "no" keeps the memory. With only yes/no, a memory that bears on the topic by
# implication lost to "no": "the payment SDK vendor ends support on 1 September"
# against a page of risks to a 15 September launch got p(yes) 0.10-0.22, so a new
# launch risk never reached the page. Several ways to say "it may matter" split the
# doubt across options that all keep the memory, so "no" wins only when it is the
# model's clear answer: on replayed eval pages, 54/54 memories that should change a
# page kept (48/54 before), unrelated memories let through 13/80 (12/80 before).
_PAGE_IMPACT_INSTRUCTIONS = (
    "Is the new memory about the page's topic, or about anything the topic depends on (a "
    "dependency, vendor, date, person or system the topic relies on)? A memory that could "
    "threaten, block, delay or change what the topic is about counts, even when it never names "
    "the topic. Answer no only when it concerns a different subject altogether."
)
_PAGE_IMPACT_OPTIONS = {
    "yes": "Yes, it concerns the page's topic or something it depends on",
    "maybe": "Possibly, it might affect the topic or something it relies on",
    "unsure": "Unclear whether it bears on the topic",
    "depends": "It concerns something the topic relies on (a vendor, dependency, date, person or system)",
    "no": "No, a different subject altogether: nothing this page covers or relies on",
}

_OPTION_KEY_OVERHEAD = 7  # JSON envelope overhead per choice option: '"c249": "' + comma/newline
_LISTING_ITEM_OVERHEAD = 10  # Formatting overhead per item in cut listing: "[1] ...\n\n"
_MAX_QUERY_TOKENS = 2_000  # Defensive ceiling on query tokens in reranker


class TypeSafeCrossEncoder(CrossEncoderModel):
    """
    TypeSafe reranker (https://typesafe.ai), Jev by default.

    Deprecated as a reranker provider (``HINDSIGHT_API_RERANKER_PROVIDER=typesafe``): set up
    the shared decision model (``HINDSIGHT_API_DECISION_MODEL_*``) instead, which recall
    uses in place of the reranker. The class stays, since reflect's decision model uses it.

    Not a Cohere-compatible /rerank endpoint: TypeSafe evaluates typed *questions*
    against a *state*. This provider asks two of them.

    **Rank — one Choice whose options are the candidates.** A Choice answers with a
    probability for every option, summing to 1. When the candidate pool fits within
    MAX_OPTIONS (250) and token context limits, handing it the whole pool returns the
    ranking in a single call. That beats scoring each candidate on its own: judged together
    the model only has to say which candidate beats which, instead of pinning every
    candidate to an absolute scale it must re-derive each time. On a 200-question LoCoMo
    set the listwise shape scored recall@1 0.94 against 0.87 for one call per candidate,
    at a thirtieth of the calls. Pools exceeding option limits or token budgets are
    partitioned into groups whose winners compete in a finals round.

    **Cut — one Score over the ranked shortlist**, asking how far down the list
    relevance extends. Only asked when ``prune_candidates`` is on. A Score's levels
    are *ordered*, which is what a cut point needs; the obvious alternative — adding
    a "none of these" option to the Choice — does not work, because Choice options
    are unrivalled alternatives rather than a scale, so "none of these" simply wins
    outright on hard queries and returns nothing at all (35 of 200 questions came
    back empty in that shape, against 0 here).

    The scores handed back are positions, not confidences: a Choice probability is a
    share of one pool, so 0.7 means "the best of these" and not "relevant", and two
    pools are not comparable. Candidates below the cut score exactly 0.0, which is
    how :attr:`prunes_candidates` tells the caller to leave them out.
    """

    SYSTEMONE_PATH = "/v1/systemone"

    # Each System One request is traced as a decision call; a rerank row would count them twice.
    traces_calls = False

    # A Choice accepts at most 255 options; stay clear of the edge. A pool larger
    # than this or exceeding single-question token limits is ranked in chunks whose
    # winners are then ranked against each other, because probabilities are normalised
    # within a call and so cannot be compared across two of them.
    MAX_OPTIONS = 250

    # How many of the ranked candidates the cut question is shown. The cut only ever
    # keeps a handful, so a longer list costs tokens to no purpose.
    SHORTLIST = 12

    # The fewest candidates a cut may keep, by rank. 0 for recall, where the cut is
    # the whole point; reflect's decision model raises it, because the cut's levels jump from five
    # to ten and it picked five for a question whose answer was a count over seven.
    MIN_KEEP = 0

    # Context window safety limit for Jev /v1/systemone.
    # Single question context limit is 32k. A defensive safety margin (26k vs 32k)
    # absorbs JSON framing and most cross-tokenizer divergence, but not all of it: on
    # mixed-script text Jev counts twice what we do (#5715). A request Jev still finds
    # too long is rebuilt at half the budget (see _within_window), down to
    # MIN_QUESTION_TOKENS.
    MAX_QUESTION_TOKENS = 26_000
    # Three halvings: 26k -> 3.25k covers an 8x divergence, well past the 2x measured.
    # A request still refused there will never fit, so it fails rather than retrying on.
    MIN_QUESTION_TOKENS = 3_000

    # Ordered depths for the cut. The model picks the level; these are the
    # granularity offered, which is a design choice and not a tuned threshold.
    #
    # There is deliberately no "nothing is relevant" level, so at least one candidate
    # always survives. Recall runs on a pool retrieval already judged plausible, and
    # one weak memory the caller can dismiss beats silence; adding that level cost
    # 7% of queries returning nothing and dropped gold retention from 0.81 to 0.65.
    CUT_LEVELS = [
        "Only the first candidate is relevant",
        "The first two are relevant",
        "The first three are relevant",
        "The first five are relevant",
        "The first ten are relevant",
        "All of the listed candidates are relevant",
    ]
    CUT_DEPTHS = [1, 2, 3, 5, 10, None]  # None keeps the whole shortlist

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_RERANKER_TYPESAFE_MODEL,
        base_url: str = DEFAULT_RERANKER_TYPESAFE_BASE_URL,
        timeout: float = DEFAULT_RERANKER_TYPESAFE_TIMEOUT,
        max_concurrent: int = DEFAULT_RERANKER_TYPESAFE_MAX_CONCURRENT,
        prune_candidates: bool = DEFAULT_RERANKER_TYPESAFE_PRUNE_CANDIDATES,
        shortlist: int | None = None,
        min_keep: int | None = None,
    ):
        if shortlist is not None:
            self.SHORTLIST = shortlist
        if min_keep is not None:
            self.MIN_KEEP = min_keep
        # Tolerate an unset-but-present value ("VAR=" in a compose file, or a config
        # built with every field zeroed) by falling back to the default.
        self.model = model or DEFAULT_RERANKER_TYPESAFE_MODEL
        self.base_url = (base_url or DEFAULT_RERANKER_TYPESAFE_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.prunes_candidates = bool(prune_candidates)
        # CrossLoopSemaphore, not asyncio.Semaphore: one encoder instance is built at
        # startup and reached from every loop in the process (worker threads run their
        # own via asyncio.run), and an asyncio.Semaphore binds to whichever loop first
        # waits on it — the second loop to contend then raises "bound to a different
        # event loop". Same reasoning as RemoteTEICrossEncoder's cap above. The calls
        # it gates are network round trips, well clear of the "short and hot" work
        # _cross_loop warns against guarding this way.
        self._semaphore = CrossLoopSemaphore(max_concurrent or DEFAULT_RERANKER_TYPESAFE_MAX_CONCURRENT)
        self._session = LoopLocalSession(
            timeout=per_phase_timeout(timeout),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )

    @property
    def provider_name(self) -> str:
        return "typesafe"

    async def initialize(self) -> None:
        logger.info(
            f"Reranker: initializing TypeSafe provider at {self.base_url} with model {self.model} "
            f"(prune_candidates={self.prunes_candidates})"
        )

    async def _ask(self, body: dict, scope: str) -> dict:
        """One System One request, traced under ``scope`` like any LLM call."""
        started = time.monotonic()
        result: dict | None = None
        error: BaseException | None = None
        try:
            async with self._semaphore:
                async with self._session.get().post(
                    f"{self.base_url}{self.SYSTEMONE_PATH}",
                    headers=reranker_bank_attribution_headers(),
                    json=body,
                ) as response:
                    await raise_for_status(response)
                    result = await response.json(content_type=None)
                    return result
        except BaseException as e:
            error = e
            raise
        finally:
            usage = result.get("usage") if isinstance(result, dict) else None
            record_decision_call(
                provider=self.provider_name,
                model=self.model,
                scope=scope,
                request=body,
                response=result if isinstance(result, dict) else None,
                input_tokens=(usage.get("input_tokens") or 0) if isinstance(usage, dict) else 0,
                started=started,
                error=error,
            )

    async def _within_window(self, build: Callable[[int], Awaitable[_T]]) -> _T:
        """``build(budget)`` at the full question budget, halved each time Jev finds the
        request longer than its window, since only Jev can count its own tokens."""
        budget = self.MAX_QUESTION_TOKENS
        while True:
            try:
                return await build(budget)
            except Exception as e:
                if not overflowed_window(e) or budget // 2 < self.MIN_QUESTION_TOKENS:
                    raise
            budget //= 2
            logger.warning(f"Reranker: TypeSafe request overflowed Jev's window, retrying at {budget} tokens")

    async def _rank_once(self, query: str, docs: list[str], indices: list[int]) -> list[int]:
        """Rank one group of candidates, returning their indices best first."""
        body = {
            "state": f"Question: {query}",
            "model": self.model,
            "questions": {
                "rank": {
                    "type": "choice",
                    "instructions": f"{_RANK_INSTRUCTIONS_PREFIX}{query}",
                    "criteria": {f"c{position}": docs[index] for position, index in enumerate(indices)},
                }
            },
        }
        result = await self._ask(body, "decision_rank")
        probabilities = result["answers"]["rank"]["probabilities"]
        # Sort the option positions, not the indices themselves: the option key encodes
        # the position, and two candidates can carry the same index-independent text.
        by_probability = sorted(range(len(indices)), key=lambda position: -float(probabilities[f"c{position}"]))
        return [indices[position] for position in by_probability]

    async def _rank(self, query: str, docs: list[str], indices: list[int], budget: int) -> list[int]:
        """Rank a whole pool best first, in rounds when it exceeds option or token limits.

        Each round's probabilities are normalised within its own call, so the winners
        are ranked against each other in a finals round. Candidates that do not make the
        finals maintain their caller input order (initial RRF rank) behind the finalists,
        discarding intra-group model ranks since probabilities across separate rounds
        are not on a shared scale (#4599).
        """
        if not indices:
            return []

        # Available token budget for candidate options in one choice question.
        state_tokens = count_tokens(f"Question: {query}")
        # +30: cushion for the question's JSON framing (type, keys) around the instructions.
        instr_tokens = count_tokens(f"{_RANK_INSTRUCTIONS_PREFIX}{query}") + 30
        net_budget = max(50, budget - state_tokens - instr_tokens)

        # Pre-truncate outlier documents that individually exceed the question budget,
        # and copy docs so we don't mutate caller's list.
        effective_docs = list(docs)
        doc_tokens: dict[int, int] = {}
        for index in indices:
            t = count_tokens(effective_docs[index])
            if t + _OPTION_KEY_OVERHEAD > net_budget:
                cap = max(10, net_budget - _OPTION_KEY_OVERHEAD)
                effective_docs[index] = truncate_to_tokens(effective_docs[index], cap).text
                t = count_tokens(effective_docs[index])
            doc_tokens[index] = t

        # Pack candidates into groups bounded by MAX_OPTIONS and net_budget.
        groups: list[list[int]] = []
        curr_group: list[int] = []
        curr_tokens = 0
        for index in indices:
            item_tokens = doc_tokens[index] + _OPTION_KEY_OVERHEAD
            if curr_group and (len(curr_group) >= self.MAX_OPTIONS or curr_tokens + item_tokens > net_budget):
                groups.append(curr_group)
                curr_group = [index]
                curr_tokens = item_tokens
            else:
                curr_group.append(index)
                curr_tokens += item_tokens
        if curr_group:
            groups.append(curr_group)

        # Fast path: all candidates fit into a single group.
        if len(groups) == 1:
            return await self._rank_once(query, effective_docs, groups[0])

        # Groups with >= 2 candidates are ranked via Jev Choice questions in parallel.
        # Single-candidate groups skip the preliminary round: a 1-candidate Choice
        # is rejected by Jev ("criteria must map 2 or more options") and the winner is trivial.
        multi_indices = [i for i, g in enumerate(groups) if len(g) >= 2]
        ranked_groups: list[list[int]] = [list(g) for g in groups]
        if multi_indices:
            ranked_multi = await asyncio.gather(
                *(self._rank_once(query, effective_docs, groups[i]) for i in multi_indices)
            )
            for i, r in zip(multi_indices, ranked_multi):
                ranked_groups[i] = r

        # Advance the top of each group to the finals; the rest fall back to RRF order (#4599).
        # The finals is one Choice, so the finalists must fit MAX_OPTIONS too: long documents
        # can split a pool into more than MAX_OPTIONS // SHORTLIST groups, so the per-group
        # quota shrinks, and past MAX_OPTIONS groups the lowest-input-ranked winners overflow
        # into the rest.
        quota = max(1, min(self.SHORTLIST, self.MAX_OPTIONS // len(ranked_groups)))
        finalists = [index for group in ranked_groups for index in group[:quota]]
        rest = [index for group in ranked_groups for index in group[quota:]] + finalists[self.MAX_OPTIONS :]
        finalists = finalists[: self.MAX_OPTIONS]
        index_pos = {idx: pos for pos, idx in enumerate(indices)}
        rest.sort(key=lambda idx: index_pos[idx])

        # Finals round: if finalists exceed budget, cap each doc evenly (budget // n).
        finalist_tokens = sum(doc_tokens[idx] + _OPTION_KEY_OVERHEAD for idx in finalists)
        if finalist_tokens > net_budget:
            cap = max(10, (net_budget - len(finalists) * _OPTION_KEY_OVERHEAD) // len(finalists))
            for idx in finalists:
                if doc_tokens[idx] > cap:
                    effective_docs[idx] = truncate_to_tokens(effective_docs[idx], cap).text

        ranked_finalists = await self._rank_once(query, effective_docs, finalists)
        return ranked_finalists + rest

    # Ordered verdicts for the reflect sufficiency question; only the last one ends
    # retrieval. "Partly" is its own level so a half-answer keeps searching.
    SUFFICIENCY_LEVELS = [
        "The evidence does not answer the question",
        "The evidence answers part of the question; something it asks is missing",
        "The evidence fully answers the question",
    ]

    async def evidence_is_sufficient(self, query: str, evidence: str) -> bool:
        """Whether ``evidence`` already answers ``query``: one Score question.

        Used by a decision-orchestrated reflect to skip the LLM turns that would otherwise decide to keep
        searching. A Score, not a yes/no Choice, for the reason the cut uses one: the
        levels are ordered, and "partly" must not be rounded up to "yes".
        """
        overhead = (
            count_tokens(_SUFFICIENCY_INSTRUCTIONS) + sum(count_tokens(level) for level in self.SUFFICIENCY_LEVELS) + 50
        )
        prefix = f"Question: {query}\n\nEvidence:\n"

        async def ask(question_budget: int) -> dict:
            budget = max(100, question_budget - overhead - count_tokens(prefix))
            # ponytail: keeps the head of the evidence; the agent's own results come
            # first (pages, then observations, then facts), so the tail is raw facts.
            fitted = truncate_to_tokens(evidence, budget).text if count_tokens(evidence) > budget else evidence
            return await self._ask(
                {
                    "state": f"{prefix}{fitted}",
                    "model": self.model,
                    "questions": {
                        "sufficient": {
                            "type": "score",
                            "instructions": _SUFFICIENCY_INSTRUCTIONS,
                            "criteria": self.SUFFICIENCY_LEVELS,
                        }
                    },
                },
                "decision_evidence_sufficiency",
            )

        result = await self._within_window(ask)
        answer = result["answers"]["sufficient"]
        # The most likely level, not the rounded expected score: an expected 1.43 is "partly" when
        # rounded, though "fully" was the likeliest verdict at 0.54 (sde-bench boltons-budget), and
        # every such rounding cost two LLM turns. The score is the fallback for a reply without
        # per-level probabilities.
        probabilities = answer.get("probabilities")
        if probabilities:
            level = int(max(probabilities, key=lambda key: float(probabilities[key])))
        else:
            level = round(float(answer["score"]))
        return level >= len(self.SUFFICIENCY_LEVELS) - 1

    async def could_change_page(self, topic: str, page: str, memory: str) -> bool:
        """Whether ``memory`` could change a page about ``topic`` that reads ``page``: one Choice.

        Gates a delta knowledge-page refresh: a memory answered "no" is never shown to
        the LLM. Worded so a doubt is a "yes", because a wrong "no" is silent and
        permanent (the refresh moves past the memory) while a wrong "yes" costs one
        LLM call, which is what every refresh pays without the gate.
        """
        prefix = f"Page topic: {topic}\n\nNew memory: {memory}\n\nThe page as it reads now:\n"
        overhead = (
            count_tokens(_PAGE_IMPACT_INSTRUCTIONS)
            + sum(count_tokens(option) for option in _PAGE_IMPACT_OPTIONS.values())
            + count_tokens(prefix)
            + 50
        )
        policy = self.retry_policy or RetryPolicy()

        async def ask(question_budget: int) -> dict:
            budget = max(100, question_budget - overhead)
            # ponytail: keeps the head of the page (its first sections); a page past
            # ~25k tokens is already far over its own budget.
            fitted = truncate_to_tokens(page, budget).text if count_tokens(page) > budget else page
            body = {
                "state": f"{prefix}{fitted}",
                "model": self.model,
                "questions": {
                    "impact": {
                        "type": "choice",
                        "instructions": _PAGE_IMPACT_INSTRUCTIONS,
                        "criteria": _PAGE_IMPACT_OPTIONS,
                    }
                },
            }
            # A refresh asks this once per new memory, all at once, so a rate limit is the
            # expected failure rather than a rare one: retried, not handed back as "search".
            return await acall_with_retry(
                lambda: self._ask(body, "decision_page_impact"),
                policy=policy,
                budget=policy.new_budget(),
                provider=self.provider_name,
            )

        result = await self._within_window(ask)
        answer = result["answers"]["impact"]
        probabilities = answer.get("probabilities") or {}
        choice = answer.get("choice") or max(probabilities, key=lambda key: float(probabilities[key]), default="yes")
        return choice != "no"

    async def _cut(self, query: str, docs: list[str], order: list[int], budget: int) -> int:
        """How many of the ranked candidates are relevant, as the model sees it."""
        shortlist = order[: self.SHORTLIST]
        if not shortlist:
            return 0

        # Instructions and levels are fixed, plus a cushion for the request's JSON framing.
        cut_overhead = count_tokens(_CUT_INSTRUCTIONS) + sum(count_tokens(c) for c in self.CUT_LEVELS) + 50
        prefix = f"Question: {query}\n\nCandidates, already ranked best first:\n"
        prefix_tokens = count_tokens(prefix)
        available_listing_tokens = max(100, budget - cut_overhead - prefix_tokens)

        # Truncate shortlist documents if they exceed the available listing budget
        shortlist_docs = [docs[index] for index in shortlist]
        shortlist_tokens = [count_tokens(d) for d in shortlist_docs]
        total_shortlist_tokens = sum(shortlist_tokens) + len(shortlist) * _LISTING_ITEM_OVERHEAD
        if total_shortlist_tokens > available_listing_tokens:
            cap = max(10, (available_listing_tokens - len(shortlist) * _LISTING_ITEM_OVERHEAD) // len(shortlist))
            shortlist_docs = [
                truncate_to_tokens(d, cap).text if t > cap else d for d, t in zip(shortlist_docs, shortlist_tokens)
            ]

        listing = "\n\n".join(f"[{position + 1}] {shortlist_docs[position]}" for position in range(len(shortlist)))
        body = {
            "state": f"{prefix}{listing}",
            "model": self.model,
            "questions": {
                "depth": {
                    "type": "score",
                    "instructions": _CUT_INSTRUCTIONS,
                    "criteria": self.CUT_LEVELS,
                }
            },
        }
        result = await self._ask(body, "decision_cut")
        level = round(float(result["answers"]["depth"]["score"]))
        depth = self.CUT_DEPTHS[min(len(self.CUT_DEPTHS) - 1, max(0, level))]
        return len(shortlist) if depth is None else min(depth, len(shortlist))

    async def _rank_group(self, query: str, docs: list[str], indices: list[int], scores: list[float]) -> None:
        if not indices:
            return

        # Query appears in both State and instructions across rank and cut.
        # Defensively bound query tokens so that _rank and _cut share the identical
        # bounded query, preventing context overflow in _cut and eliminating semantic drift.
        max_allowed_query = max(50, (self.MAX_QUESTION_TOKENS - 200) // 2)
        effective_query_cap = min(_MAX_QUERY_TOKENS, max_allowed_query)
        if count_tokens(query) > effective_query_cap:
            query = truncate_to_tokens(query, effective_query_cap).text

        order = await self._within_window(lambda budget: self._rank(query, docs, indices, budget))
        keep = (
            await self._within_window(lambda budget: self._cut(query, docs, order, budget))
            if self.prunes_candidates
            else len(order)
        )
        keep = max(keep, min(self.MIN_KEEP, len(order)))
        # Positions, not confidences — see the class docstring. Descending from 1.0 so
        # the caller's ordering is preserved, and 0.0 for everything past the cut,
        # which is how prunes_candidates marks a candidate to leave out.
        for position, index in enumerate(order):
            scores[index] = (len(order) - position) / len(order) if position < keep else 0.0

    async def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        docs = [doc for _, doc in pairs]
        query_groups: dict[str, list[int]] = {}
        for index, (query, _) in enumerate(pairs):
            query_groups.setdefault(query, []).append(index)

        scores = [0.0] * len(pairs)
        await asyncio.gather(
            *(self._rank_group(query, docs, indices, scores) for query, indices in query_groups.items())
        )
        return scores


class RRFPassthroughCrossEncoder(CrossEncoderModel):
    """
    Passthrough cross-encoder that preserves RRF scores without neural reranking.

    This is useful for:
    - Testing retrieval quality without reranking overhead
    - Deployments where reranking latency is unacceptable
    - Debugging to isolate retrieval vs reranking issues
    """

    traces_calls = False  # no model call to trace

    def __init__(self):
        """Initialize RRF passthrough cross-encoder."""
        pass

    @property
    def provider_name(self) -> str:
        return "rrf"

    async def initialize(self) -> None:
        """No initialization needed."""
        logger.info("Reranker: RRF passthrough provider initialized (neural reranking disabled)")

    async def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
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
        # The session is always rebuilt, not only when the arena is off: FlashRank's
        # own session sizes its pool to every physical core with spinning on, which
        # under a CPU quota throttled each recall for minutes (#5725).
        import onnxruntime as ort

        session_options = ort.SessionOptions()
        session_options.enable_cpu_mem_arena = self.cpu_mem_arena
        bound_onnx_session(session_options)

        # Initialize ranker with optional cache directory
        ranker_kwargs: dict = {"model_name": self.model_name, "max_length": self.max_length}
        if self.cache_dir:
            ranker_kwargs["cache_dir"] = self.cache_dir

        self._ranker = Ranker(**ranker_kwargs)

        # FlashRank's Ranker doesn't expose SessionOptions in its API,
        # so we replace the session after initialization.
        if hasattr(self._ranker, "session"):
            model_file = None
            model_dir = getattr(self._ranker, "model_dir", None)
            if model_dir:
                from pathlib import Path

                for candidate in Path(model_dir).glob("*.onnx"):
                    model_file = str(candidate)
                    break
            if model_file:
                self._ranker.session = ort.InferenceSession(model_file, sess_options=session_options)
                logger.info(
                    "Reranker: replaced FlashRank ONNX session (cpu_mem_arena=%s, threads=%s)",
                    self.cpu_mem_arena,
                    session_options.intra_op_num_threads,
                )

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

    def _predict_sync(self, pairs: list[tuple[str, str]], cancelled: threading.Event | None = None) -> list[float]:
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
                    if cancelled is not None and cancelled.is_set():
                        raise asyncio.CancelledError()
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

    async def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
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
        return await _run_batches_in_executor(FlashRankCrossEncoder._executor, self._predict_sync, pairs)


class OnnxCrossEncoder(CrossEncoderModel):
    """Any cross-encoder exported to ONNX, run in-process with ONNX Runtime.

    The reranking twin of ``OnnxEmbeddings``: an ``onnxruntime`` session for the graph
    and ``AutoTokenizer`` for the tokenizer, so models outside FlashRank's catalog run
    without PyTorch or a TEI sidecar. Scores are the model's raw logits, like the
    ``local`` provider, so switching between the two keeps the ranking and the
    downstream sigmoid unchanged.
    """

    # Shared executor across instances: its worker count is what caps concurrent passes.
    _executor: ThreadPoolExecutor | None = None
    _max_concurrent: int = DEFAULT_RERANKER_ONNX_MAX_CONCURRENT

    def __init__(
        self,
        model_id: str = DEFAULT_RERANKER_LOCAL_MODEL,
        model_path: str | None = None,
        tokenizer_name_or_path: str | None = None,
        onnx_file: str = DEFAULT_RERANKER_ONNX_FILE,
        max_tokens: int = DEFAULT_RERANKER_ONNX_MAX_TOKENS,
        batch_size: int = DEFAULT_RERANKER_ONNX_BATCH_SIZE,
        threads: int = DEFAULT_RERANKER_ONNX_THREADS,
        max_concurrent: int = DEFAULT_RERANKER_ONNX_MAX_CONCURRENT,
        cpu_mem_arena: bool = DEFAULT_RERANKER_ONNX_CPU_MEM_ARENA,
        timeout: float = DEFAULT_RERANKER_LOCAL_TIMEOUT,
    ):
        self.model_id = model_id
        self.model_path = model_path
        if model_path and tokenizer_name_or_path is None:
            logger.warning(
                "Reranker: ONNX model_path is set without tokenizer_name_or_path; falling back to tokenizer "
                "from model_id %s. Set HINDSIGHT_API_RERANKER_ONNX_TOKENIZER_NAME_OR_PATH when using local files.",
                model_id,
            )
        self.tokenizer_name_or_path = tokenizer_name_or_path or model_id
        self.onnx_file = onnx_file
        self.max_tokens = max_tokens
        if batch_size < 1:
            raise ValueError("ONNX reranker batch_size must be >= 1")
        self.batch_size = batch_size
        if threads < 0:
            raise ValueError("ONNX reranker threads must be >= 0 (0 = the native thread cap)")
        self.threads = threads
        self.cpu_mem_arena = cpu_mem_arena
        self.timeout = timeout
        self._session = None
        self._tokenizer = None
        OnnxCrossEncoder._max_concurrent = max(1, max_concurrent)

    @property
    def provider_name(self) -> str:
        return "onnx"

    @property
    def blocking_init(self) -> bool:
        return True

    async def initialize(self) -> None:
        if self._session is not None:
            return

        # Imports transformers; see LOCAL_MODEL_BUILD_LOCK.
        with LOCAL_MODEL_BUILD_LOCK:
            try:
                import onnxruntime as ort
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "onnxruntime and transformers are required for the 'onnx' reranker. "
                    "Install with: pip install 'hindsight-api-slim[local-onnx]'"
                ) from exc

        model_path = self.model_path
        if not model_path:
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise ImportError(
                    "huggingface-hub is required to download ONNX reranker models. "
                    "Set HINDSIGHT_API_RERANKER_ONNX_MODEL_PATH or install local-onnx."
                ) from exc
            # Also fetch the conventional external-weights sidecar of large exports.
            snapshot_dir = snapshot_download(
                repo_id=self.model_id, allow_patterns=[self.onnx_file, f"{self.onnx_file}_data"]
            )
            model_path = os.path.join(snapshot_dir, self.onnx_file)

        logger.info(
            "Reranker: initializing ONNX provider with %s (threads=%s, max_concurrent=%s, batch_size=%s)",
            model_path,
            self.threads or onnx_intra_op_threads(),
            OnnxCrossEncoder._max_concurrent,
            self.batch_size,
        )
        # Imports transformers; see LOCAL_MODEL_BUILD_LOCK.
        with LOCAL_MODEL_BUILD_LOCK:
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)
        session_options = ort.SessionOptions()
        # The arena caches freed activation blocks and never returns them, so RSS would
        # hold the largest batch's high-water mark forever (same as FlashRank/OnnxEmbeddings).
        session_options.enable_cpu_mem_arena = self.cpu_mem_arena
        # Idle worker threads otherwise busy-wait for the next op; measured ~1 core of
        # burn per pass for no latency gain on a 300-candidate rerank.
        bound_onnx_session(session_options, self.threads)
        self._session = ort.InferenceSession(
            model_path, sess_options=session_options, providers=["CPUExecutionProvider"]
        )

        assert_finite_local_output(
            self._predict_sync([("hindsight startup probe", "hindsight startup probe")]),
            label=f"Reranker[{model_path}]",
        )

        if OnnxCrossEncoder._executor is None:
            OnnxCrossEncoder._executor = ThreadPoolExecutor(
                max_workers=OnnxCrossEncoder._max_concurrent, thread_name_prefix="onnx-reranker"
            )
        logger.info("Reranker: ONNX provider initialized")

    def _predict_sync(self, pairs: list[tuple[str, str]], cancelled: threading.Event | None = None) -> list[float]:
        """Score ``pairs`` in bounded forward passes, longest first.

        Each pass pads to its longest pair and allocates ``batch * heads * seq^2``
        attention, so batches stay small (#3355) and similar lengths share a batch.
        Padding is masked, so grouping never changes a score. Shares the ``local``
        provider's wall-clock budget (#4696), checked between passes: a model too big
        for the CPU hands back what it scored instead of holding recall for minutes.
        """
        if self._session is None or self._tokenizer is None:
            raise RuntimeError("Reranker not initialized. Call initialize() first.")
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]), reverse=True)
        scores: list[float | None] = [None] * len(pairs)
        deadline = time.monotonic() + self.timeout if self.timeout > 0 else None
        for start in range(0, len(order), self.batch_size):
            if cancelled is not None and cancelled.is_set():
                raise asyncio.CancelledError()
            if deadline is not None and time.monotonic() >= deadline:
                raise RerankTimeoutError(scores, self.timeout, self.model_path or self.model_id)
            window = order[start : start + self.batch_size]
            batch = self._score_batch([pairs[i] for i in window])
            for index, score in zip(window, batch, strict=True):
                scores[index] = score
        return [0.0 if score is None else score for score in scores]

    def _score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Run one forward pass and reduce the logits to one raw score per pair."""
        assert self._session is not None and self._tokenizer is not None

        import numpy as np

        encoded = self._tokenizer(
            [query for query, _ in pairs],
            [doc for _, doc in pairs],
            padding=True,
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="np",
        )
        input_names = {inp.name for inp in self._session.get_inputs()}
        ort_inputs = {name: value for name, value in encoded.items() if name in input_names}
        if "token_type_ids" in input_names and "token_type_ids" not in ort_inputs:
            ort_inputs["token_type_ids"] = np.zeros_like(encoded["input_ids"])

        logits = np.asarray(self._session.run(None, ort_inputs)[0], dtype=np.float64)
        logits = logits.reshape(len(pairs), -1)
        if logits.shape[1] == 1:
            return logits[:, 0].tolist()
        if logits.shape[1] == 2:
            # Binary [irrelevant, relevant] heads: the log-odds of "relevant", whose
            # sigmoid (applied downstream) is exactly the model's softmax probability.
            return (logits[:, 1] - logits[:, 0]).tolist()
        raise ValueError(
            f"ONNX reranker {self.model_path or self.model_id} returned {logits.shape[1]} logits per pair; expected 1 or 2"
        )

    async def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        if self._session is None:
            raise RuntimeError("Reranker not initialized. Call initialize() first.")
        if not pairs:
            return []
        return await _run_batches_in_executor(OnnxCrossEncoder._executor, self._predict_sync, pairs)


def _truncate_docs_to_tokens(texts: list[str], max_tokens: int) -> list[str]:
    """Truncate every candidate document to at most ``max_tokens``.

    Batched rather than looped: reranking truncates the whole candidate list on
    every call, and the shared tokenizer cuts a list in Rust with the GIL released.
    The overwhelming majority already fit, and those come back untouched.
    """
    from .token_encoding import truncate_many_to_tokens

    return [result.text for result in truncate_many_to_tokens(texts, max_tokens)]


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
        self._initialized = False
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self._session = LoopLocalSession(timeout=per_phase_timeout(timeout), headers=headers)

    @property
    def provider_name(self) -> str:
        return "litellm"

    async def initialize(self) -> None:
        """Mark the provider ready; the HTTP session is created lazily per event loop."""
        if self._initialized:
            return

        logger.info(f"Reranker: initializing LiteLLM provider at {self.api_base} with model {self.model}")
        self._initialized = True
        logger.info("Reranker: LiteLLM provider initialized")

    async def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """
        Score query-document pairs using the LiteLLM proxy's /rerank endpoint.

        Args:
            pairs: List of (query, document) tuples to score

        Returns:
            List of relevance scores
        """
        if not self._initialized:
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
            async with self._session.get().post(
                f"{self.api_base}/rerank",
                headers=reranker_bank_attribution_headers(),
                json={
                    "model": self.model,
                    "query": query,
                    "documents": texts,
                    "top_n": len(texts),  # Return all scores
                },
            ) as response:
                await raise_for_status(response)
                result = await response.json(content_type=None)

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
    ):
        """
        Initialize LiteLLM SDK cross-encoder client.

        Args:
            api_key: API key for the reranking provider (optional — omit for
                     providers that use ambient credentials, e.g. AWS Bedrock with IAM)
            model: Model name with provider prefix (e.g., "deepinfra/Qwen3-reranker-8B")
            api_base: Custom base URL for API (optional)
            timeout: Request timeout in seconds (default: 60.0)
        """
        self.api_key = api_key
        self.model = model
        self.api_base = api_base
        self.timeout = timeout
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

    async def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
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
        # See: https://github.com/vectorize-io/hindsight/issues/994 and #6069.
        with LOCAL_MODEL_BUILD_LOCK:
            import transformers

            _ = transformers.AutoTokenizer

        try:
            import mlx.core  # noqa: F401  # ty: ignore[unresolved-import]
            import mlx_lm  # noqa: F401  # ty: ignore[unresolved-import]
        except ImportError as exc:
            # Only swallow "package not installed" errors. Anything else (e.g. a
            # transitive import failure inside mlx_lm) must surface verbatim so
            # the real cause is debuggable instead of being masked by a generic
            # "install mlx" message.
            msg = str(exc)
            if "mlx" not in msg and "mlx_lm" not in msg:
                raise
            # mlx is Apple's Metal/unified-memory framework, so the local-ml extra
            # only installs it on macOS arm64 (see pyproject.toml). Missing here
            # therefore usually means "wrong platform", not "forgot the extra" —
            # say both, and name the way out.
            raise ImportError(
                "mlx and mlx-lm are required for the 'jina-mlx' reranker, and are only "
                "installed on Apple Silicon — mlx is Apple's Metal framework, and its "
                "Linux build is CPU-only and slower than the 'local' provider. Set "
                "HINDSIGHT_API_RERANKER_PROVIDER=local (or a hosted provider), or install "
                "them yourself: pip install mlx>=0.31.0 mlx-lm>=0.31.1 safetensors>=0.6.2"
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
        # Picked up by release_local_inference_memory() below, which routes "mlx" to
        # mx.clear_cache(). Without it MLX holds every freed Metal buffer for the life
        # of the process.
        self._device_type = "mlx"
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
            try:
                for query, indexed_docs in query_groups.items():
                    docs = [doc for _, doc in indexed_docs]
                    indices = [idx for idx, _ in indexed_docs]
                    results = self._reranker.rerank(query, docs)
                    for result in results:
                        original_idx = result["index"]
                        all_scores[indices[original_idx]] = result["relevance_score"]
            finally:
                # Inside the lock: MLX Metal ops are not thread-safe, which is why
                # #1113 introduced _mlx_lock in the first place.
                release_local_inference_memory(self._device_type)

        return all_scores

    async def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        if self._reranker is None:
            raise RuntimeError("Reranker not initialized. Call initialize() first.")

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._predict_sync, pairs)


class GoogleCrossEncoder(CrossEncoderModel):
    """
    Google Discovery Engine cross-encoder using the Ranking REST API.

    Uses aiohttp + google-auth for lightweight REST calls (no gRPC/protobuf).
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
        self._credentials: Any = None
        self._session = LoopLocalSession(timeout=per_phase_timeout(timeout))
        # Set by initialize(); None means not initialized.
        self._rank_url: str | None = None

    @property
    def provider_name(self) -> str:
        return "google"

    async def _get_auth_headers(self) -> dict[str, str]:
        """Get Authorization header with a fresh access token."""
        import google.auth.transport.requests

        if not self._credentials.valid:
            # google-auth has no public async refresh API: its refresh is a blocking
            # token-endpoint round trip, so it runs in a thread rather than on the loop.
            await asyncio.to_thread(self._credentials.refresh, google.auth.transport.requests.Request())
        return {"Authorization": f"Bearer {self._credentials.token}"}

    async def initialize(self) -> None:
        """Initialize credentials; the HTTP session is created lazily per event loop."""
        if self._rank_url is not None:
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
            # ADC discovery can probe the GCE metadata server with a blocking request, and
            # google-auth has no async equivalent, so it runs in a thread.
            self._credentials, _ = await asyncio.to_thread(google.auth.default, scopes=self.SCOPES)

        ranking_config = f"projects/{self.project_id}/locations/{self.location}/rankingConfigs/default_ranking_config"
        self._rank_url = f"{self.API_BASE}/{ranking_config}:rank"

        logger.info("Reranker: Google Discovery Engine provider initialized")

    async def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """
        Score query-document pairs using Google Discovery Engine Ranking API.

        Args:
            pairs: List of (query, document) tuples to score

        Returns:
            List of relevance scores (0-1, higher = more relevant)
        """
        if self._rank_url is None:
            raise RuntimeError("Reranker not initialized. Call initialize() first.")

        if not pairs:
            return []

        session = self._session.get()

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

                async with session.post(
                    self._rank_url,
                    headers=await self._get_auth_headers(),
                    json={
                        "model": self.model,
                        "query": query,
                        "records": records,
                        "topN": len(records),
                    },
                ) as response:
                    await raise_for_status(response)
                    result = await response.json(content_type=None)

                for record in result.get("records", []):
                    local_idx = int(record["id"])
                    all_scores[batch_indices[local_idx]] = record["score"]

        return all_scores


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
        if self._client.initialized:
            return
        logger.info(f"Reranker: initializing Alibaba Cloud provider with model {self.model}")
        await self._client.initialize()
        logger.info("Reranker: Alibaba Cloud provider initialized")

    async def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return await self._client.predict(pairs)


class MultiCrossEncoder(CrossEncoderModel):
    """Failover across an ordered chain of cross-encoders.

    Member 0 is the primary (the unindexed ``HINDSIGHT_API_RERANKER_*`` config);
    members 1..N are the indexed fallbacks. Each ``predict`` tries members in order
    and returns the first usable set of scores, so an unreachable reranker costs
    ranking quality (whatever the next member gives) instead of the whole recall.
    Put ``rrf`` last to degrade to the fusion order rather than failing.

    Each member keeps its own retry budget (``create_cross_encoder`` gives every
    remote member a ``retry_policy``), so we only advance after a member has
    exhausted its retries and raised — an ordinary quota blip does not burn a
    fallback. This chain holds no policy of its own: retrying here would retry the
    whole failover sequence rather than the member that stumbled. A member that fails to initialize is not fatal — that is the point of
    the chain — it is retried lazily once its cooldown (below) is over.

    A member that fails is benched for ``MEMBER_COOLDOWN_SECONDS``: requests try
    it after the healthy members instead of first, so an outage costs one slow
    request per cooldown rather than every recall paying the dead member's
    retries. The first request after the cooldown probes it again.
    """

    # Same idea as MultiLLMProvider's cooldown, minus its config and doubling: a
    # reranker outage is a host being down, not an hours-long quota to back off from.
    # ponytail: fixed cooldown, make it a setting if operators ask to tune it
    MEMBER_COOLDOWN_SECONDS = 30.0

    traces_calls = False  # each member's predict() traces its own attempt

    def __init__(self, members: list[CrossEncoderModel]) -> None:
        if len(members) < 2:
            raise ValueError("MultiCrossEncoder requires at least two members")
        self._members = members
        self._ready = [False] * len(members)
        self._locks = [asyncio.Lock() for _ in members]
        self._active = 0
        self._benched_until = [0.0] * len(members)

    @property
    def provider_name(self) -> str:
        """The provider of the member that last served a request (primary before any).

        This shared cursor is for diagnostics only: concurrent requests can move
        it after another request received its scores. Recall uses the provider
        captured on RerankResult to decide passthrough scoring and response metadata.
        """
        return self._members[self._active].provider_name

    @property
    def primary_provider_name(self) -> str:
        return self._members[0].provider_name

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
                self._benched_until[index] = time.monotonic() + self.MEMBER_COOLDOWN_SECONDS
                logger.warning(
                    "Reranker member %d (%s) failed to initialize: %s; it will be retried on use after %.0fs",
                    index,
                    self._members[index].provider_name,
                    result,
                    self.MEMBER_COOLDOWN_SECONDS,
                )
        if not any(self._ready):
            logger.error("Reranker: no member of the failover chain initialized; recall will retry them per request")

    async def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score ``pairs`` with the first member that answers usably."""
        last_exc: BaseException | None = None
        now = time.monotonic()
        # Benched members go last, not away: if every member is down, still try them.
        order = sorted(range(len(self._members)), key=lambda i: self._benched_until[i] > now)
        for position, index in enumerate(order):
            member = self._members[index]
            try:
                if not self._ready[index]:
                    await self._ensure_member_ready(index)
                scores = await member.predict(pairs)
                if len(scores) != len(pairs):
                    raise RuntimeError(f"returned {len(scores)} scores for {len(pairs)} pairs")
            except Exception as e:  # noqa: BLE001 - re-raised below if no member answers
                last_exc = e
                self._benched_until[index] = time.monotonic() + self.MEMBER_COOLDOWN_SECONDS
                remaining = len(order) - position - 1
                logger.warning(
                    "[RECALL %s] Reranker member %d (%s) failed: %s%s",
                    recall_log_id.get() or "-",
                    index,
                    member.provider_name,
                    e,
                    f"; trying next member ({remaining} left)" if remaining else "; no members left",
                )
                continue
            self._benched_until[index] = 0.0
            if index != self._active:
                logger.info(
                    "Reranker: now serving from member %d (%s)",
                    index,
                    member.provider_name,
                )
            self._active = index
            # Record the member for this task before returning. A later read of
            # provider_name follows _active and can name a different request.
            _served_provider.set(member.provider_name)
            return scores
        # All members failed; surface the last error (loop ran at least once).
        assert last_exc is not None
        raise last_exc


# Reranker providers that must NOT be given a retry policy, and why. Everything
# else is a remote API whose transient failures are worth retrying.
_RERANKER_PROVIDERS_WITHOUT_RETRY = {
    # In-process: no round trip to fail transiently, and a local failure (a bad
    # tensor, an OOM) is not fixed by trying again.
    "local": "in-process model",
    "flashrank": "in-process model",
    "onnx": "in-process model",
    "jina-mlx": "in-process model",
    "rrf": "passthrough, does no work",
    # Already retries internally, including TEI's 429-as-backpressure handling.
    # A second layer would multiply its attempts by ours.
    "tei": "own _async_request_with_retry (see tei_retry.py)",
}


def _reranker_retry_policy() -> RetryPolicy:
    """Build the reranker retry policy from resolved configuration."""
    from ..config import get_config

    config = get_config()
    return RetryPolicy(
        max_retries=config.reranker_max_retries,
        initial_backoff=config.reranker_initial_backoff,
        max_backoff=config.reranker_max_backoff,
        budget_seconds=config.reranker_retry_budget,
    )


def create_cross_encoder(member: RerankerMemberConfig) -> CrossEncoderModel:
    """
    Create a CrossEncoderModel for one member of the reranker chain.

    ``member`` is the primary (index 0, the unindexed ``HINDSIGHT_API_RERANKER_*``
    config) or an indexed fallback. Missing-setting errors name the member's own
    env var, so a chain misconfiguration points at the exact indexed variable.

    Every member carries its ``max_tokens_per_candidate`` cap, and remote members a
    ``retry_policy``; the base class's ``predict`` applies both; see ``_RERANKER_PROVIDERS_WITHOUT_RETRY`` for the ones that
    deliberately do not get one.

    Args:
        member: Resolved settings for this member

    Returns:
        Configured CrossEncoderModel instance
    """
    encoder = _create_cross_encoder_backend(member)
    encoder.max_tokens_per_candidate = member.max_tokens_per_candidate
    if member.provider.lower() not in _RERANKER_PROVIDERS_WITHOUT_RETRY:
        encoder.retry_policy = _reranker_retry_policy()
    return encoder


def _create_cross_encoder_backend(member: RerankerMemberConfig) -> CrossEncoderModel:
    """Construct the provider backend itself, before any retry policy is attached."""
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
            timeout=member.local_timeout,
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
            max_concurrent=member.cohere_max_concurrent,
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
    elif provider == "onnx":
        onnx = member.onnx
        return OnnxCrossEncoder(
            model_id=onnx.model_id,
            model_path=onnx.model_path,
            tokenizer_name_or_path=onnx.tokenizer_name_or_path,
            onnx_file=onnx.file,
            max_tokens=onnx.max_tokens,
            batch_size=onnx.batch_size,
            threads=onnx.threads,
            max_concurrent=onnx.max_concurrent,
            cpu_mem_arena=onnx.cpu_mem_arena,
            timeout=onnx.timeout,
        )
    elif provider == "litellm":
        return LiteLLMCrossEncoder(
            api_base=member.litellm_api_base,
            api_key=member.litellm_api_key,
            model=member.litellm_model,
            timeout=member.litellm_timeout,
        )
    elif provider == "litellm-sdk":
        return LiteLLMSDKCrossEncoder(
            api_key=member.litellm_sdk_api_key or None,
            model=member.litellm_sdk_model,
            api_base=member.litellm_sdk_api_base,
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
    elif provider == "nvidia":
        api_key = member.nvidia_api_key
        if not api_key:
            raise ValueError(
                f"{member.env_name('NVIDIA_API_KEY')} is required when {member.env_name('PROVIDER')} is 'nvidia'"
            )
        return NvidiaCrossEncoder(
            api_key=api_key,
            model=member.nvidia_model,
            base_url=member.nvidia_base_url,
            timeout=member.nvidia_timeout,
            batch_size=member.nvidia_batch_size,
            max_concurrent=member.nvidia_max_concurrent,
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
    elif provider == "typesafe":
        logger.warning(
            "Reranker provider 'typesafe' is deprecated and will be removed in a future release. "
            "Set up the shared decision model instead (HINDSIGHT_API_DECISION_MODEL_PROVIDER=typesafe "
            "and HINDSIGHT_API_DECISION_MODEL_API_KEY): recall then ranks with it in place of the reranker."
        )
        api_key = member.typesafe_api_key
        if not api_key:
            raise ValueError(
                f"{member.env_name('TYPESAFE_API_KEY')} is required when {member.env_name('PROVIDER')} is 'typesafe'"
            )
        return TypeSafeCrossEncoder(
            api_key=api_key,
            model=member.typesafe_model,
            base_url=member.typesafe_base_url,
            timeout=member.typesafe_timeout,
            max_concurrent=member.typesafe_max_concurrent,
            prune_candidates=member.typesafe_prune_candidates,
        )
    elif provider == "hindsight":
        # Imported here: cloud_auth pulls in aiohttp's server, which only this provider needs.
        from .providers.cloud_auth import saved_api_key

        api_key = member.hindsight_api_key or saved_api_key()
        if not api_key:
            raise ValueError(
                f"{member.env_name('HINDSIGHT_API_KEY')} is required when {member.env_name('PROVIDER')} is "
                "'hindsight' (or run `hindsight-admin cloud login`)"
            )
        # Hindsight Inference speaks Cohere's /v1/rerank, so the Cohere-compatible client serves it.
        return CohereCrossEncoder(
            api_key=api_key,
            model=DEFAULT_RERANKER_HINDSIGHT_MODEL,
            base_url=f"{HINDSIGHT_INFERENCE_URL}/v1/rerank",
            timeout=DEFAULT_RERANKER_COHERE_TIMEOUT,
        )
    elif provider == "rrf":
        return RRFPassthroughCrossEncoder()
    elif provider == "jina-mlx":
        return JinaMLXCrossEncoder()
    else:
        raise ValueError(
            f"Unknown reranker provider: {provider}. Supported: 'local', 'tei', 'cohere', 'zeroentropy', 'siliconflow', 'nvidia', 'typesafe', 'hindsight', 'alibaba', 'google', 'flashrank', 'onnx', 'litellm', 'litellm-sdk', 'rrf', 'jina-mlx'"
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
