"""
LLM Client Pool with LRU eviction.

Pools LLM provider instances to avoid creating new clients for every operation
while supporting per-bank configuration.
"""

import hashlib
import logging
from collections import OrderedDict
from typing import Tuple

from hindsight_api.engine.llm_wrapper import LLMProvider

logger = logging.getLogger(__name__)

# LRU cache for LLM clients
# Structure: OrderedDict{(provider, model, api_key_hash): LLMProvider}
# Max 100 entries, oldest evicted when limit exceeded
_CLIENT_CACHE: OrderedDict[Tuple[str, str, str], LLMProvider] = OrderedDict()
_CACHE_MAX_SIZE = 100


class LLMClientPool:
    """
    Pool of LLM clients with LRU eviction.

    Caches LLMProvider instances by (provider, model, api_key_hash) to avoid
    creating new clients for every operation while supporting per-bank configuration.
    """

    @staticmethod
    def get_or_create(
        provider: str,
        api_key: str | None,
        base_url: str,
        model: str,
        reasoning_effort: str = "low",
        groq_service_tier: str | None = None,
    ) -> LLMProvider:
        """
        Get cached LLM client or create new one.

        Args:
            provider: Provider name (openai, groq, anthropic, etc.)
            api_key: API key (may be None for local providers)
            base_url: Base URL for the API
            model: Model name
            reasoning_effort: Reasoning effort level for supported providers
            groq_service_tier: Groq service tier (for Groq provider)

        Returns:
            LLMProvider instance (cached or newly created)
        """
        # Create cache key
        api_key_hash = _hash_api_key(api_key or "")
        cache_key = (provider.lower(), model, api_key_hash)

        # Check cache
        if cache_key in _CLIENT_CACHE:
            logger.debug(f"LLM client pool hit for {provider}/{model}")
            # Move to end (mark as recently used)
            _CLIENT_CACHE.move_to_end(cache_key)
            return _CLIENT_CACHE[cache_key]

        logger.debug(f"LLM client pool miss for {provider}/{model}, creating new client")

        # Create new client
        client = LLMProvider(
            provider=provider,
            api_key=api_key or "",
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
            groq_service_tier=groq_service_tier,
        )

        # Add to cache
        _CLIENT_CACHE[cache_key] = client

        # Evict oldest if over max size
        if len(_CLIENT_CACHE) > _CACHE_MAX_SIZE:
            evicted_key = next(iter(_CLIENT_CACHE))  # First item (oldest)
            del _CLIENT_CACHE[evicted_key]
            logger.debug(
                f"LLM client pool full, evicted oldest client: {evicted_key[0]}/{evicted_key[1]} "
                f"(pool size: {len(_CLIENT_CACHE)})"
            )

        return client

    @staticmethod
    def clear() -> None:
        """Clear the entire client pool. Useful for testing."""
        global _CLIENT_CACHE
        _CLIENT_CACHE.clear()
        logger.info("Cleared LLM client pool")


def _hash_api_key(api_key: str) -> str:
    """
    Hash API key for cache key generation.

    Args:
        api_key: API key to hash

    Returns:
        First 16 characters of SHA256 hash
    """
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]
