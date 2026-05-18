"""Token counting helpers for reflect prompts and agent control flow."""

from functools import lru_cache

import tiktoken


@lru_cache(maxsize=1)
def _get_cl100k_base_encoding() -> tiktoken.Encoding:
    # tiktoken downloads this encoding on first lookup when it is not cached.
    # Keep the lookup lazy so importing hindsight_api does not depend on network access.
    return tiktoken.get_encoding("cl100k_base")


def count_cl100k_tokens(text: str) -> int:
    """Return the number of cl100k_base tokens in text."""
    return len(_get_cl100k_base_encoding().encode(text))
