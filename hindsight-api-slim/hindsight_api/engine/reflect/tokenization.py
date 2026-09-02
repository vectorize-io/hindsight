"""Token counting helpers for reflect prompts and agent control flow."""

from ..token_encoding import count_tokens as _count_tokens


def count_prompt_tokens(text: str) -> int:
    """Return the number of tokens in text under the configured encoding."""
    return _count_tokens(text)
