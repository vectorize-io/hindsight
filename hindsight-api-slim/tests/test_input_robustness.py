"""Input-robustness regression tests.

Covers the "server 500s on unusual-but-valid input" class:
- #1883: content containing tiktoken special-token literals (e.g. ``<|endoftext|>``).
- #1875: queries/content containing an unpaired UTF-16 surrogate (e.g. a half-emoji).
"""

from datetime import datetime, timezone

import pytest

from hindsight_api.engine.llm_wrapper import sanitize_llm_output, sanitize_text
from hindsight_api.engine.reflect.tokenization import count_cl100k_tokens
from hindsight_api.engine.token_encoding import count_tokens, get_token_encoding

# A lone high surrogate — valid in a Python str, but rejected by the Rust
# tokenizers behind the local embedder / cross-encoder and uncodable to UTF-8.
LONE_SURROGATE = "deploy the \ud83d service"
SPECIAL_TOKEN_TEXT = "The fix was to sanitize the <|endoftext|> token before sending."


# --- Prong A: tiktoken tolerates special-token literals (#1883) ------------------


def test_count_tokens_handles_special_token_literal():
    # Default tiktoken disallowed_special="all" would raise ValueError here.
    assert count_tokens(SPECIAL_TOKEN_TEXT) > 0
    assert count_cl100k_tokens(SPECIAL_TOKEN_TEXT) > 0


def test_encode_decode_roundtrip_with_special_token():
    enc = get_token_encoding()
    tokens = enc.encode(SPECIAL_TOKEN_TEXT)
    assert enc.decode(tokens) == SPECIAL_TOKEN_TEXT


def test_special_token_counted_as_ordinary_text():
    # The literal is split into ordinary tokens, not collapsed into one special id.
    assert count_tokens("<|endoftext|>") > 1


# --- Prong B: surrogate / control-char sanitization (#1875) ----------------------


def test_sanitize_strips_lone_surrogate():
    cleaned = sanitize_text(LONE_SURROGATE)
    assert cleaned == "deploy the  service"
    assert cleaned.encode("utf-8")  # no longer raises


def test_sanitize_preserves_valid_text_and_paired_emoji():
    text = "café 🎉\tindented\nnewline"
    assert sanitize_text(text) == text


def test_sanitize_strips_control_chars_but_keeps_whitespace():
    assert sanitize_text("a\x00b\x07c") == "abc"
    assert sanitize_text("a\tb\nc\rd") == "a\tb\nc\rd"


def test_sanitize_none_and_empty():
    assert sanitize_text(None) is None
    assert sanitize_text("") == ""


def test_sanitize_llm_output_is_alias():
    assert sanitize_llm_output is sanitize_text


# --- Integration: full pipeline survives both inputs -----------------------------


@pytest.mark.asyncio
async def test_retain_with_special_token_literal(memory, request_context):
    """Retaining content that mentions ``<|endoftext|>`` must not 500 (#1883)."""
    bank_id = f"test_special_token_{datetime.now(timezone.utc).timestamp()}"
    unit_ids = await memory.retain_async(
        bank_id=bank_id,
        content=SPECIAL_TOKEN_TEXT,
        context="debugging tokenizers",
        request_context=request_context,
    )
    assert isinstance(unit_ids, list)


@pytest.mark.asyncio
async def test_recall_with_lone_surrogate_query(memory, request_context):
    """A recall query with an unpaired surrogate must not crash the embedder (#1875)."""
    bank_id = f"test_surrogate_{datetime.now(timezone.utc).timestamp()}"
    await memory.retain_async(
        bank_id=bank_id,
        content="The deploy service ships releases.",
        request_context=request_context,
    )
    # Without ingress sanitization the local ST embedder raises TextEncodeInput.
    result = await memory.recall_async(
        bank_id=bank_id,
        query=LONE_SURROGATE,
        request_context=request_context,
    )
    assert result is not None


@pytest.mark.asyncio
async def test_retain_with_lone_surrogate_content(memory, request_context):
    """Retaining content with an unpaired surrogate must not 500 (#1875)."""
    bank_id = f"test_surrogate_retain_{datetime.now(timezone.utc).timestamp()}"
    unit_ids = await memory.retain_async(
        bank_id=bank_id,
        content="A half emoji \ud83d slipped into the transcript.",
        request_context=request_context,
    )
    assert isinstance(unit_ids, list)
