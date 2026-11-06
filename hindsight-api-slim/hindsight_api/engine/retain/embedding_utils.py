"""
Embedding generation utilities for memory units.
"""

import logging
import math
import time
import weakref
from typing import Literal, Protocol

from ...config import ENV_EMBEDDINGS_MAX_INPUT_TOKENS, get_config
from ..llm_interface import ProviderRateLimitResetError
from ..llm_trace import EMBEDDING_SCOPE_PREFIX, trace_model_call
from ..remote_retry import is_transient_remote_error
from ..token_encoding import count_tokens, truncate_many_to_tokens

logger = logging.getLogger(__name__)

EmbeddingInputType = Literal["document", "query"]


class EmbeddingsBackend(Protocol):
    """Minimal duck-typed surface used by retain/recall — the concrete `Embeddings`
    ABC supplies default implementations that delegate to `encode()`."""

    @property
    def dimension(self) -> int: ...

    async def encode_query(self, texts: list[str]) -> list[list[float]]: ...

    async def encode_documents(self, texts: list[str]) -> list[list[float]]: ...


def _tokenizer_encoding(backend: EmbeddingsBackend) -> str | None:
    """The backend's own tokenizer vocabulary, when it declares one.

    `Embeddings.tokenizer_encoding` is an optional capability of the concrete ABC and
    not part of this duck-typed Protocol, so getattr can hand back anything; only a
    real string selects an encoding. See #5234 — counting the cap with the configured
    encoding instead of the model's overshoots OpenAI's 8192 by a handful of tokens.
    """
    encoding = getattr(backend, "tokenizer_encoding", None)
    return encoding if isinstance(encoding, str) and encoding else None


def _prefix_tokens(backend: EmbeddingsBackend, input_type: EmbeddingInputType) -> int:
    """Tokens the backend will prepend to every text after this cap is applied.

    Asymmetric models (E5, embeddinggemma, ...) carry their instruction as a literal
    prefix that `Embeddings._encode_prefixed` glues on *after* truncation, so a text cut
    to exactly the model's limit still arrives a few tokens over it. Charge the prefix
    against the budget here. Providers with a native mechanism (SentenceTransformers'
    prompts, ZeroEntropy's input_type) carry no prefix and cost nothing.
    """
    # `EmbeddingsBackend` is a duck-typed Protocol and does not declare the prefixes —
    # they are an optional capability of the concrete `Embeddings` ABC — so getattr can
    # hand back anything at all. Only a real, non-empty string costs budget.
    attr = "query_prefix" if input_type == "query" else "passage_prefix"
    prefix = getattr(backend, attr, "")
    if not (isinstance(prefix, str) and prefix):
        return 0
    return count_tokens(prefix, _tokenizer_encoding(backend))


def _truncate_inputs(
    texts: list[str], max_input_tokens: int, backend: EmbeddingsBackend, input_type: EmbeddingInputType
) -> list[str]:
    """Cap each input at ``max_input_tokens`` tokens before it reaches the provider.

    Remote providers with a fixed input-token limit (e.g. Bedrock Titan V2's hard 8192
    cap, or a llama.cpp ``/v1/embeddings`` server) reject an oversized text with a
    permanent 4xx rather than truncating server-side the way SentenceTransformers does.
    One oversized memory then fails the whole retain/recall batch. This is provider-
    agnostic on purpose: the cap is applied here, once, before any backend's `encode()`.

    The budget covers the *whole* string that will go over the wire, not just the
    caller's payload: an asymmetric model's prefix is subtracted up front (see
    :func:`_prefix_tokens`), and callers that decorate their text — a knowledge page
    embeds ``f"{name} {content}"`` — pass the joined string, so the join is inside the
    cap rather than one token past it (#4165).
    """
    budget = max(max_input_tokens - _prefix_tokens(backend, input_type), 0)
    results = truncate_many_to_tokens(texts, budget, _tokenizer_encoding(backend))
    truncated = [result.text for result in results]
    original_token_counts = [r.original_tokens for r in results if r.original_tokens > budget]
    if original_token_counts:
        logger.warning(
            "Embeddings: truncated %d of %d input(s) to %d tokens for provider %s "
            "(largest was ~%d tokens); embedded content is incomplete. Raise or disable "
            "%s if this is unexpected.",
            len(original_token_counts),
            len(texts),
            budget,
            getattr(backend, "provider_name", "?"),
            max(original_token_counts),
            ENV_EMBEDDINGS_MAX_INPUT_TOKENS,
        )
    return truncated


def _validate_embedding_vector(vector: list[float], *, index: int, expected_dimension: int) -> list[float]:
    actual_dimension = len(vector)
    if actual_dimension == 0:
        raise RuntimeError(f"embedding {index} has dimension 0; expected {expected_dimension}")
    if actual_dimension != expected_dimension:
        raise RuntimeError(f"embedding {index} has dimension {actual_dimension}; expected {expected_dimension}")
    # A NaN/Inf vector otherwise travels to the pgvector bind and fails there as an
    # opaque "NaN not allowed in vector" that names neither the backend nor the vector
    # (issue #6255). Fail here, loudly, with the index. Left retryable: the observed
    # case was a transient backend glitch that clean re-embeds of the same text fixed.
    non_finite = sum(1 for x in vector if not math.isfinite(x))
    if non_finite:
        raise RuntimeError(
            f"embedding {index} contains {non_finite} non-finite value(s) (NaN/Inf) of {actual_dimension}; "
            "the embedding backend produced an unusable vector"
        )
    return vector


async def _encode_with_input_type(
    embeddings_backend: EmbeddingsBackend, texts: list[str], input_type: EmbeddingInputType
) -> list[list[float]]:
    if input_type == "query":
        return await embeddings_backend.encode_query(texts)
    return await embeddings_backend.encode_documents(texts)


async def generate_embeddings_batch(
    embeddings_backend: EmbeddingsBackend, texts: list[str], input_type: EmbeddingInputType = "document"
) -> list[list[float]]:
    """
    Generate embeddings for multiple texts using the provided embeddings backend.

    Remote backends await their HTTP calls on the loop; in-process backends move the
    CPU-bound model call to a worker thread themselves.

    Args:
        embeddings_backend: Embeddings instance to use for encoding
        texts: List of texts to embed
        input_type: Whether texts are retained documents or recall/search queries.

    Returns:
        List of embeddings in same order as input texts
    """
    # Cap oversized inputs once, here, before they reach any provider's encode(). Kept
    # out of the individual backends so every provider (and every call path — retain,
    # recall queries, consolidation, import) gets identical, model-agnostic truncation.
    max_input_tokens = get_config().embeddings_max_input_tokens
    if max_input_tokens is not None and texts:
        texts = _truncate_inputs(texts, max_input_tokens, embeddings_backend, input_type)

    try:
        embeddings = await trace_model_call(
            embeddings_backend,
            f"{EMBEDDING_SCOPE_PREFIX}{input_type}",
            len(texts),
            lambda: _encode_with_input_type(embeddings_backend, texts, input_type),
            request={"texts": texts},
        )
    except ProviderRateLimitResetError:
        # A control signal, not a failure: the worker parks the operation until then.
        raise
    except Exception as e:
        if _is_transient_chain(e):
            _mark_down(embeddings_backend)
        raise Exception(f"Failed to generate batch embeddings: {str(e)}") from e
    _down_since.pop(embeddings_backend, None)

    # Guarantee 1:1 alignment with input texts. A silent length mismatch here
    # propagates downstream as zip() drops items, eventually surfacing as an
    # IndexError in retain mapping (see issue #1037).
    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Embeddings backend returned {len(embeddings)} vectors for {len(texts)} input texts; "
            "expected exact 1:1 alignment"
        )

    return [
        _validate_embedding_vector(embedding, index=index, expected_dimension=embeddings_backend.dimension)
        for index, embedding in enumerate(embeddings)
    ]


class EmbeddingsUnavailableError(Exception):
    """The embeddings backend is failing right now; raised before a retain spends LLM tokens."""


# How long a failed probe is trusted before the next retain probes again.
EMBEDDINGS_PROBE_INTERVAL_SECONDS = 30.0

# In-memory health note per backend (issue #5999). An entry means "a real embedding call
# failed with a transient error and nothing has succeeded since"; the value is when the
# backend was last seen failing. Set and cleared by the real calls in
# generate_embeddings_batch, so a healthy backend costs no extra requests.
_down_since: "weakref.WeakKeyDictionary[object, float]" = weakref.WeakKeyDictionary()


def _is_transient_chain(exc: BaseException) -> bool:
    """True when ``exc`` or anything it wraps is a transient remote error.

    Providers re-raise transport failures as a RuntimeError carrying the message only
    (TEI's "TEI embedding request failed: ..."), so the original is on the chain.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if is_transient_remote_error(current):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def _mark_down(backend: object) -> None:
    try:
        _down_since[backend] = time.monotonic()
    except TypeError:
        pass  # not weak-referenceable: no health note, behave as before


async def ensure_embeddings_reachable(embeddings_backend: EmbeddingsBackend) -> None:
    """Fail fast when the embeddings backend is known to be down, before any LLM spend.

    No-op unless a real embedding call recently failed with a transient error. Then one
    one-word probe decides: success clears the note and the retain proceeds, failure
    raises EmbeddingsUnavailableError. A failed probe is trusted for
    EMBEDDINGS_PROBE_INTERVAL_SECONDS, so an outage costs at most one tiny embed per
    process per interval. A single blip therefore never blocks a retain: it only makes
    the next one check with a real call.
    """
    last_failure = _down_since.get(embeddings_backend)
    if last_failure is None:
        return
    if time.monotonic() - last_failure < EMBEDDINGS_PROBE_INTERVAL_SECONDS:
        raise EmbeddingsUnavailableError("embeddings backend is unreachable (recent request failed)")
    try:
        await generate_embeddings_batch(embeddings_backend, ["ping"])
    except Exception as e:
        if _is_transient_chain(e):
            raise EmbeddingsUnavailableError(f"embeddings backend is unreachable: {e}") from e
        # The backend answered, just not well (bad key, dimension mismatch): not an
        # outage. Let the retain run and surface that error the usual way.
        _down_since.pop(embeddings_backend, None)
