"""Token counting helpers for reflect prompts and agent control flow."""

from ..token_encoding import count_tokens as _count_tokens


def count_cl100k_tokens(text: str) -> int:
    """Return the number of cl100k_base tokens in text."""
    return _count_tokens(text)
