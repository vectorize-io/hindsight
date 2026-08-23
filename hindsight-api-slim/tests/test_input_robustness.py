"""Input-robustness regression tests.

Covers the "server 500s on unusual-but-valid input" class:
- #1883: content containing tokenizer special-token literals (e.g. ``<|endoftext|>``).
- #1875: queries/content containing an unpaired UTF-16 surrogate (e.g. a half-emoji).
- #3729: structured LLM fact output containing an unpaired surrogate.
"""

import dataclasses
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.llm_wrapper import sanitize_llm_output, sanitize_text
from hindsight_api.engine.reflect.tokenization import count_prompt_tokens
from hindsight_api.engine.response_models import TokenUsage
from hindsight_api.engine.token_encoding import count_tokens, get_token_encoding

# A lone high surrogate — valid in a Python str, but rejected by the Rust
# tokenizers behind the local embedder / cross-encoder and uncodable to UTF-8.
LONE_SURROGATE = "deploy the \ud83d service"
SPECIAL_TOKEN_TEXT = "The fix was to sanitize the <|endoftext|> token before sending."


# --- Prong A: the tokenizer tolerates special-token literals (#1883) ------------


def test_count_tokens_handles_special_token_literal():
    # The tokenizer's default disallowed_special="all" would raise ValueError here.
    assert count_tokens(SPECIAL_TOKEN_TEXT) > 0
    assert count_prompt_tokens(SPECIAL_TOKEN_TEXT) > 0


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


@pytest.mark.asyncio
async def test_fact_extraction_sanitizes_surrogates_generated_by_llm():
    """Valid source can still produce a lone surrogate in structured LLM output."""
    from hindsight_api.config import _get_raw_config
    from hindsight_api.engine.llm_wrapper import LLMProvider
    from hindsight_api.engine.retain.fact_extraction import _extract_facts_from_chunk

    config = dataclasses.replace(
        _get_raw_config(),
        retain_llm_max_retries=0,
        retain_extraction_mode="concise",
        retain_extract_causal_links=False,
        retain_mission=None,
        llm_temperature_retain=0.1,
        llm_strict_schema_retain=False,
        entity_labels=None,
        entities_allow_free_form=True,
    )
    llm = MagicMock(spec=LLMProvider)
    llm.provider = "mock"
    llm.call = AsyncMock(
        return_value=(
            {
                "facts": [
                    {
                        "what": "Alex laughed 😂",
                        "when": "N/A",
                        "where": "N/A",
                        "who": "Alex",
                        "why": "The joke was funny \ude02",
                        "fact_type": "world",
                        "fact_kind": "conversation",
                    }
                ]
            },
            TokenUsage(),
        )
    )

    with patch(
        "hindsight_api.engine.retain.fact_extraction._build_extraction_prompt_and_schema",
        return_value=("system prompt", MagicMock()),
    ):
        facts, _usage = await _extract_facts_from_chunk(
            chunk="Alex laughed at the joke.",
            chunk_index=0,
            total_chunks=1,
            event_date=datetime(2026, 8, 22, tzinfo=timezone.utc),
            context="",
            llm_config=llm,
            config=config,
        )

    assert facts[0].fact == "Alex laughed 😂 | Involving: Alex | The joke was funny "
    assert facts[0].fact.encode("utf-8")


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
