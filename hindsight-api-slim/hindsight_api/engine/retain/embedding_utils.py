"""
Embedding generation utilities for memory units.
"""

import asyncio
import contextvars
import logging
from typing import Literal, Protocol

from ...config import ENV_EMBEDDINGS_MAX_INPUT_TOKENS, get_config
from ..token_encoding import truncate_to_tokens

logger = logging.getLogger(__name__)

EmbeddingInputType = Literal["document", "query"]


class EmbeddingsBackend(Protocol):
    """Minimal duck-typed surface used by retain/recall — the concrete `Embeddings`
    ABC supplies default implementations that delegate to `encode()`."""

    @property
    def dimension(self) -> int: ...

    def encode_query(self, texts: list[str]) -> list[list[float]]: ...

    def encode_documents(self, texts: list[str]) -> list[list[float]]: ...


def _truncate_inputs(texts: list[str], max_input_tokens: int, backend: EmbeddingsBackend) -> list[str]:
    """Cap each input at ``max_input_tokens`` tiktoken tokens before it reaches the provider.

    Remote providers with a fixed input-token limit (e.g. Bedrock Titan V2's hard 8192
    cap, or a llama.cpp ``/v1/embeddings`` server) reject an oversized text with a
    permanent 4xx rather than truncating server-side the way SentenceTransformers does.
    One oversized memory then fails the whole retain/recall batch. This is provider-
    agnostic on purpose: the cap is applied here, once, before any backend's `encode()`.
    """
    truncated: list[str] = []
    original_token_counts: list[int] = []
    for text in texts:
        result = truncate_to_tokens(text, max_input_tokens)
        truncated.append(result.text)
        if result.original_tokens > max_input_tokens:
            original_token_counts.append(result.original_tokens)
    if original_token_counts:
        logger.warning(
            "Embeddings: truncated %d of %d input(s) to %d tokens for provider %s "
            "(largest was ~%d tokens); embedded content is incomplete. Raise or unset "
            "%s if this is unexpected.",
            len(original_token_counts),
            len(texts),
            max_input_tokens,
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
    return vector


def _encode_with_input_type(
    embeddings_backend: EmbeddingsBackend, texts: list[str], input_type: EmbeddingInputType
) -> list[list[float]]:
    if input_type == "query":
        return embeddings_backend.encode_query(texts)
    return embeddings_backend.encode_documents(texts)


async def generate_embeddings_batch(
    embeddings_backend: EmbeddingsBackend, texts: list[str], input_type: EmbeddingInputType = "document"
) -> list[list[float]]:
    """
    Generate embeddings for multiple texts using the provided embeddings backend.

    Runs the embedding generation in a thread pool to avoid blocking the event loop
    for CPU-bound operations.

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
        texts = _truncate_inputs(texts, max_input_tokens, embeddings_backend)

    try:
        loop = asyncio.get_event_loop()
        # run_in_executor runs the encode in a worker thread, which does NOT inherit
        # the caller's contextvars. Capture the current context and run the encode
        # inside it so context-dependent behavior (e.g. per-bank `user` attribution
        # read via get_current_bank_id()) survives the thread hop.
        ctx = contextvars.copy_context()
        embeddings = await loop.run_in_executor(
            None, lambda: ctx.run(_encode_with_input_type, embeddings_backend, texts, input_type)
        )
    except Exception as e:
        raise Exception(f"Failed to generate batch embeddings: {str(e)}")

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
