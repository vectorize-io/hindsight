"""
Cross-encoder abstraction for reranking.

Provides an interface for reranking with different backends.

Configuration via environment variables - see hindsight_api.config for all env var names.
"""

import asyncio
import logging
import threading
import time
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
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
from .tei_retry import tei_retry_delay

logger = logging.getLogger(__name__)

# Fallback only when the loaded tokenizer does not expose a usable max_length.
# MiniLM-L-6-v2 reports 512; other CE models may report 256..131072.
_DEFAULT_CE_MAX_LENGTH = 512
_CE_MAX_LENGTH_FLOOR = 8
_CE_MAX_LENGTH_CEILING = 131072
_OVERFLOW_VALUEERROR_MARKERS = (
    "Unable to create tensor",
    "expected sequence of length",
)
_QUERY_TRUNCATION_WARN_COOLDOWN_S = 60.0
_query_truncation_last_warn_at = 0.0
_bind_tokenizer_lock = threading.Lock()


def _is_overflow_value_error(exc: BaseException) -> bool:
    """True for the known convert_to_tensors / ragged-sequence overflow class."""
    message = str(exc)
    return any(marker in message for marker in _OVERFLOW_VALUEERROR_MARKERS)


def resolve_ce_max_length(model: Any, default: int = _DEFAULT_CE_MAX_LENGTH) -> int:
    """Read max_length from the loaded model / tokenizer. Do not hard-code 512."""
    tok = getattr(model, "tokenizer", None)
    inner = getattr(tok, "_inner", None)
    candidates = (
        getattr(model, "max_length", None),
        getattr(tok, "model_max_length", None),
        getattr(tok, "_max_length", None),
        getattr(inner, "model_max_length", None),
    )
    for val in candidates:
        if isinstance(val, bool):
            continue
        if not isinstance(val, (int, float)):
            continue
        n = int(val)
        if n < _CE_MAX_LENGTH_FLOOR:
            logger.warning(
                "Reranker: model_max_length=%s below floor %s; trying next candidate",
                n,
                _CE_MAX_LENGTH_FLOOR,
            )
            continue
        if n > _CE_MAX_LENGTH_CEILING:
            logger.warning(
                "Reranker: model_max_length=%s exceeds ceiling %s; clamping",
                n,
                _CE_MAX_LENGTH_CEILING,
            )
            return _CE_MAX_LENGTH_CEILING
        return n
    return default


def query_exceeds_max_length(tokenizer: Any, query: str, max_length: int) -> bool:
    """True when tokenising the query without truncation exceeds max_length."""
    if tokenizer is None or not query:
        return False
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        return False
    try:
        ids = encode(query, add_special_tokens=False, truncation=False)
    except TypeError:
        try:
            ids = encode(query)
        except Exception:
            return False
    except Exception:
        return False
    try:
        return len(ids) > int(max_length)
    except TypeError:
        return False


class TokenizerMaxLengthWrapper:
    """Force padding, truncation, and max_length on every tokenizer() call.

    sentence-transformers 5.2.0 CrossEncoder.predict already passes
    padding=True and truncation=True but omits max_length. A file-sized
    query then dies in convert_to_tensors (ValueError: Unable to create tensor).
    """

    def __init__(self, inner: Any, max_length: int) -> None:
        self._inner = inner
        self._max_length = int(max_length)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("padding", True)
        kwargs.setdefault("truncation", True)
        kwargs.setdefault("max_length", self._max_length)
        return self._inner(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _bind_tokenizer_max_length(model: Any, max_length: int) -> None:
    tok = getattr(model, "tokenizer", None)
    if tok is None or isinstance(tok, TokenizerMaxLengthWrapper):
        return
    with _bind_tokenizer_lock:
        tok = getattr(model, "tokenizer", None)
        if tok is None or isinstance(tok, TokenizerMaxLengthWrapper):
            return
        try:
            model.tokenizer = TokenizerMaxLengthWrapper(tok, max_length)
        except Exception as exc:
            logger.warning("Reranker: could not bind tokenizer wrapper: %s", exc)


def _warn_query_truncated(max_length: int) -> None:
    """Warn on oversized queries, then DEBUG for 60s so a busy reranker cannot flood logs."""
    global _query_truncation_last_warn_at
    now = time.monotonic()
    if now - _query_truncation_last_warn_at < _QUERY_TRUNCATION_WARN_COOLDOWN_S:
        logger.debug("Reranker: query truncated to tokenizer max_length=%s", max_length)
        return
    _query_truncation_last_warn_at = now
    logger.warning("Reranker: query truncated to tokenizer max_length=%s", max_length)


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

    Uses a dedicated thread pool to limit concurrent CPU-bound work. Each
    executor thread owns its own CrossEncoder (tokenizer + weights). Sharing
    one instance across threads races the HuggingFace fast tokenizer
    (set_truncation_and_padding + encode_batch) and was observed live as
    ``Unable to create tensor`` wrapping ``'int' object is not callable``.
    """

    # Shared executor across LocalSTCrossEncoder wrappers. The executor is the
    # only shared mutable object - not the CrossEncoder / tokenizer.
    _executor: ThreadPoolExecutor | None = None
    _max_concurrent: int = 4  # Limit concurrent CPU-bound reranking calls
    # Serializes first-use CrossEncoder() construction so concurrent worker
    # threads do not race the HuggingFace cache / torch load path.
    _load_lock = threading.Lock()

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
        # Kept for test doubles that inject a stub; production predict never
        # reads this. A load failure must not fall back to a shared instance.
        self._model = None
        self._device: str | None = None
        self._device_type: str = "cpu"
        self._initialized = False
        self._thread_models = threading.local()
        LocalSTCrossEncoder._max_concurrent = max_concurrent

    @property
    def provider_name(self) -> str:
        return "local"

    @property
    def blocking_init(self) -> bool:
        return True

    def _apply_xlm_compat_patch(self) -> None:
        """Restore create_position_ids_from_input_ids for transformers 5.x + XLM-R.

        transformers 5.x removed that helper as a module-level function; custom
        code in models such as jina-reranker-v2-base-multilingual still imports
        it. Best-effort: a missing transformers install is not an init failure.
        """
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

    def _ensure_executor(self) -> ThreadPoolExecutor:
        """Create the class-level pool once. Does not resize an existing pool.

        max_workers stays whatever the first initialize() observed - changing
        RERANKER_LOCAL_MAX_CONCURRENT after the executor exists is a process
        restart, same as before this per-thread change.
        """
        if LocalSTCrossEncoder._executor is None:
            LocalSTCrossEncoder._executor = ThreadPoolExecutor(
                max_workers=LocalSTCrossEncoder._max_concurrent,
                thread_name_prefix="reranker",
            )
        return LocalSTCrossEncoder._executor

    def _pin_eval_and_device(self, model: Any) -> None:
        """Run to(device)/eval once at load so predict() is not the first writer.

        Sentence-Transformers CrossEncoder.predict still calls these on every
        batch; after this they are idempotent writes on a thread-private model.
        """
        inner = getattr(model, "model", None)
        target = inner if inner is not None else model
        device = getattr(model, "device", None)
        if device is None:
            device = self._device
        to_fn = getattr(target, "to", None)
        if callable(to_fn) and device is not None:
            to_fn(device)
        eval_fn = getattr(target, "eval", None)
        if callable(eval_fn):
            eval_fn()

    def _load_model_instance(self) -> Any:
        """Construct one CrossEncoder for the calling thread. Never returns a shared instance."""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for LocalSTCrossEncoder. "
                "Install it with: pip install sentence-transformers"
            )

        device = self._device
        if device is None:
            device = select_local_device(self.force_cpu, self.allow_mps)
            self._device = device

        # Suppress verbose transformers warnings during model loading.
        # CrossEncoder emits harmless "UNEXPECTED" / missing-key UserWarnings
        # (e.g. "embeddings.position_ids | UNEXPECTED") that look like failures.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*was not found in model state dict.*")
            warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")

            transformers_logger = logging.getLogger("transformers")
            original_level = transformers_logger.level
            transformers_logger.setLevel(logging.ERROR)

            try:
                # low_cpu_mem_usage=False avoids accelerate meta tensors when no
                # GPU is present. We do not pass device_map: CrossEncoder calls
                # .to(device) after load, which conflicts with accelerate.
                model = CrossEncoder(
                    self.model_name,
                    device=device,
                    model_kwargs={"low_cpu_mem_usage": False},
                    trust_remote_code=self.trust_remote_code,
                )
            finally:
                transformers_logger.setLevel(original_level)

        if model is None:
            raise RuntimeError(
                f"CrossEncoder({self.model_name!r}) returned None; refusing to share another thread's model"
            )

        max_len = resolve_ce_max_length(model)
        try:
            model.max_length = max_len
        except Exception:
            pass
        _bind_tokenizer_max_length(model, max_len)

        self._device_type = resolve_model_device_type(model)
        if self.fp16 and self._device_type != "cpu":
            model.model.half()
            logger.info("Reranker: FP16 inference enabled")
        self._pin_eval_and_device(model)
        return model

    def _get_thread_model(self) -> Any:
        """Return the CrossEncoder owned by this executor thread, loading if needed.

        Load failures raise. This method never returns another thread's instance
        and never falls back to ``self._model``.
        """
        model = getattr(self._thread_models, "model", None)
        if model is not None:
            return model

        thread_name = threading.current_thread().name
        with LocalSTCrossEncoder._load_lock:
            model = getattr(self._thread_models, "model", None)
            if model is not None:
                return model
            try:
                model = self._load_model_instance()
            except ImportError:
                raise
            except Exception as exc:
                raise RuntimeError(
                    "Failed to create a per-thread LocalSTCrossEncoder instance "
                    f"for thread {thread_name!r} (model={self.model_name!r}). "
                    "Refusing to share another thread's model."
                ) from exc
            if model is None:
                raise RuntimeError(
                    "Failed to create a per-thread LocalSTCrossEncoder instance "
                    f"for thread {thread_name!r}: loader returned None. "
                    "Refusing to share another thread's model."
                )
            self._thread_models.model = model
            return model

    def _warmup_executor_threads(self) -> None:
        """Force every pool thread to exist and load its own instance.

        ThreadPoolExecutor only creates a worker when a task runs on it.
        Submitting N tasks without a barrier can run them all on one worker
        (N loads, 1 thread-local). The barrier makes all N workers start
        before any load, so initialize() fails fast if any instance cannot
        be created and first recall does not pay 4 cold loads.
        """
        executor = LocalSTCrossEncoder._executor
        if executor is None:
            raise RuntimeError("Reranker executor missing during per-thread warmup")
        # The class-level _max_concurrent is last-writer-wins and does not
        # resize an existing pool (same as before). Warming N>max_workers
        # tasks on a smaller pool deadlocks the barrier.
        n = getattr(executor, "_max_workers", LocalSTCrossEncoder._max_concurrent)
        barrier = threading.Barrier(n)

        def _warmup() -> None:
            # Bound so a dead worker cannot hang API startup forever.
            barrier.wait(timeout=600)
            self._get_thread_model()

        futures: list[Future[None]] = [executor.submit(_warmup) for _ in range(n)]
        for fut in futures:
            fut.result()

    async def initialize(self) -> None:
        """Prepare the executor and load one CrossEncoder per executor thread."""
        if self._initialized:
            return

        logger.info(f"Reranker: initializing local provider with model {self.model_name}")
        # Device is chosen once; each per-thread CrossEncoder is constructed on it.
        # MPS is opt-in (allow_mps) - see engine/local_device.py for why.
        self._device = select_local_device(self.force_cpu, self.allow_mps)
        self._apply_xlm_compat_patch()
        created_executor = LocalSTCrossEncoder._executor is None
        self._ensure_executor()
        self._warmup_executor_threads()
        self._initialized = True
        reused = "" if created_executor else ", using existing executor"
        logger.info(
            "Reranker: local provider initialized "
            f"(max_concurrent={LocalSTCrossEncoder._max_concurrent}, "
            f"per-thread model instances{reused})"
        )

    def _scores_from_model(self, model: Any, pairs: list[tuple[str, str]]) -> list[float]:
        """Score pairs with tokenizer truncation.

        Known convert_to_tensors overflow ValueErrors retry one-at-a-time; other
        ValueErrors (shape, dtype, config) propagate. CUDA OOM / TypeError still
        raise. Failed pairs score ``-inf`` so they cannot outrank a real score.
        """
        if not pairs:
            return []
        max_len = resolve_ce_max_length(model)
        try:
            if getattr(model, "max_length", None) in (None, 0):
                model.max_length = max_len
        except Exception:
            pass
        _bind_tokenizer_max_length(model, max_len)
        query = pairs[0][0] if isinstance(pairs[0], (list, tuple)) and pairs[0] else ""
        if query_exceeds_max_length(getattr(model, "tokenizer", None), str(query), max_len):
            # First oversized query in a cooldown window is WARNING; later ones DEBUG.
            _warn_query_truncated(max_len)
        try:
            scores = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        except ValueError as exc:
            if not _is_overflow_value_error(exc):
                raise
            # Overflow / convert_to_tensors. Do not log the HF message
            # (operators grep "Unable to create tensor" as a crash class).
            logger.warning(
                "Reranker: batched predict failed; scoring %d pairs one-at-a-time",
                len(pairs),
            )
            scores = []
            for pair in pairs:
                try:
                    one = model.predict([pair], batch_size=1, show_progress_bar=False)
                except ValueError as one_exc:
                    if not _is_overflow_value_error(one_exc):
                        raise
                    scores.append(float("-inf"))
                    continue
                one_list = one.tolist() if hasattr(one, "tolist") else list(one)
                scores.extend(one_list)
            return list(scores)
        return scores.tolist() if hasattr(scores, "tolist") else list(scores)

    def _predict_sync(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Synchronous prediction wrapper for thread pool execution.

        Both the bucket-batching and plain arms score through the calling
        thread's private CrossEncoder.

        Supports two optimizations (controlled via .env):
        - bucket_batching: sort pairs by token length to reduce padding waste (36-54% speedup)
        - batch_size: explicit batch size for predict() calls (MPS optimal: 32)
        """
        model = self._get_thread_model()
        try:
            if self.bucket_batching and len(pairs) > 1:
                # Sort pairs by approximate token length to create homogeneous batches.
                # This eliminates padding waste - short pairs aren't padded to the length
                # of the longest pair in the batch. Quality-identical by construction.
                lengths = [len(pairs[i][0]) + len(pairs[i][1]) for i in range(len(pairs))]
                sorted_indices = sorted(range(len(pairs)), key=lambda i: lengths[i])
                sorted_pairs = [pairs[i] for i in sorted_indices]

                sorted_scores = self._scores_from_model(model, sorted_pairs)

                # Restore original order
                scores = [0.0] * len(pairs)
                for new_pos, orig_idx in enumerate(sorted_indices):
                    scores[orig_idx] = sorted_scores[new_pos]
                return scores

            return self._scores_from_model(model, pairs)
        finally:
            release_local_inference_memory(self._device_type)

    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """
        Score query-document pairs for relevance.

        Uses a dedicated thread pool with limited workers to prevent CPU thrashing.
        Forwards overlap across banks because each worker has its own model;
        a process-wide predict lock would sum per-bank CE times.

        Args:
            pairs: List of (query, document) tuples to score

        Returns:
            List of relevance scores (raw logits from the model)
        """
        if not self._initialized:
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
