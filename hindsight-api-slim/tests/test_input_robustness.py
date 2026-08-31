"""Input-robustness regression tests.

Covers the "server 500s on unusual-but-valid input" class:
- #1883: content containing tokenizer special-token literals (e.g. ``<|endoftext|>``).
- #1875: queries/content containing an unpaired UTF-16 surrogate (e.g. a half-emoji).
- #3729: structured LLM fact output containing an unpaired surrogate.
"""

import dataclasses
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.llm_wrapper import (
    LLMProvider,
    parse_llm_json,
    sanitize_llm_output,
    sanitize_llm_value,
    sanitize_text,
)
from hindsight_api.engine.reflect.tokenization import count_prompt_tokens
from hindsight_api.engine.response_models import LLMToolCall, LLMToolCallResult, TokenUsage
from hindsight_api.engine.retain.fact_extraction import Fact
from hindsight_api.engine.token_encoding import count_tokens, get_token_encoding

# A lone high surrogate — valid in a Python str, but rejected by the Rust
# tokenizers behind the local embedder / cross-encoder and uncodable to UTF-8.
HIGH_SURROGATE = "\ud83d"
LONE_SURROGATE = f"deploy the {HIGH_SURROGATE} service"
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


# --- Prong C: every LLM response is scrubbed at one boundary (#3729) ------------


def test_sanitize_llm_value_returns_clean_input_unchanged():
    """The common case must not copy: identity is preserved all the way down."""
    payload = {"facts": [{"what": "café 🎉", "n": 3}], "ok": None, "flag": True, "score": 1.5}
    assert sanitize_llm_value(payload) is payload
    text = "plain text"
    assert sanitize_llm_value(text) is text


def test_sanitize_llm_value_scrubs_nested_strings_only():
    payload = {"facts": [{"what": f"a{HIGH_SURROGATE}b", "count": 3, "ratio": 0.5, "missing": None}]}
    cleaned = sanitize_llm_value(payload)
    assert cleaned["facts"][0]["what"] == "ab"
    # Non-text fields keep their type and value.
    assert cleaned["facts"][0]["count"] == 3
    assert cleaned["facts"][0]["ratio"] == 0.5
    assert cleaned["facts"][0]["missing"] is None


def test_sanitize_llm_value_scrubs_dict_keys():
    cleaned = sanitize_llm_value({f"na{HIGH_SURROGATE}me": "Alex"})
    assert cleaned == {"name": "Alex"}


def test_sanitize_llm_value_scrubs_pydantic_models():
    result = LLMToolCallResult(
        content=f"answer{HIGH_SURROGATE}",
        tool_calls=[LLMToolCall(id="call_1", name="recall", arguments={"query": f"q{HIGH_SURROGATE}"})],
        output_tokens=42,
    )
    cleaned = sanitize_llm_value(result)
    assert isinstance(cleaned, LLMToolCallResult)
    assert cleaned.content == "answer"
    assert cleaned.tool_calls[0].arguments == {"query": "q"}
    # Numeric fields survive the copy.
    assert cleaned.output_tokens == 42


def test_sanitize_llm_value_preserves_return_usage_tuple_shape():
    """``return_usage=True`` hands back ``(result, TokenUsage)`` — both must survive."""
    usage = TokenUsage()
    cleaned = sanitize_llm_value(({"text": f"x{HIGH_SURROGATE}"}, usage))
    assert isinstance(cleaned, tuple) and len(cleaned) == 2
    assert cleaned[0] == {"text": "x"}
    assert cleaned[1] is usage


def test_parse_llm_json_scrubs_surrogates_born_at_decode():
    """The exact mechanism behind #3729.

    A model writes the six ASCII characters ``\\ud83d``; nothing is wrong with the
    raw text and scrubbing it there is a no-op. ``json.loads`` is what turns them
    into a lone surrogate no downstream stage can UTF-8 encode, so the scrub has
    to happen on the parsed object.
    """
    raw = '{"facts": [{"what": "Alex laughed \\ud83d", "entities": ["Al\\ud83dex"], "n": 3}]}'
    assert sanitize_text(raw) == raw  # the escape is invisible before decoding

    parsed = parse_llm_json(raw)

    assert parsed["facts"][0]["what"] == "Alex laughed "
    assert parsed["facts"][0]["entities"] == ["Alex"]
    assert parsed["facts"][0]["n"] == 3
    assert json.dumps(parsed).encode("utf-8")


def test_parse_llm_json_keeps_valid_surrogate_pairs():
    """Only *lone* surrogates go. A pair is how JSON spells an astral codepoint."""
    raw = '{"what": "caf\\u00e9 \\ud83c\\udf89", "n": 1, "ok": true}'

    assert parse_llm_json(raw) == {"what": "caf\u00e9 \U0001f389", "n": 1, "ok": True}


@pytest.mark.asyncio
async def test_llm_provider_call_sanitizes_response():
    """Every structured/text LLM response is scrubbed inside ``LLMProvider.call``."""
    llm = LLMProvider(provider="mock", api_key="", base_url="", model="mock-model")
    llm.set_mock_response({"facts": [{"what": f"Alex laughed 😂{HIGH_SURROGATE}"}]})

    result = await llm.call(messages=[{"role": "user", "content": "hi"}], skip_validation=True)

    assert result["facts"][0]["what"] == "Alex laughed 😂"
    assert result["facts"][0]["what"].encode("utf-8")


@pytest.mark.asyncio
async def test_llm_provider_call_with_tools_sanitizes_content_and_arguments():
    """The reflect/agent path is model-authored too, arguments included."""
    llm = LLMProvider(provider="mock", api_key="", base_url="", model="mock-model")
    llm.set_mock_response(
        LLMToolCallResult(
            content=f"thinking{HIGH_SURROGATE}",
            tool_calls=[LLMToolCall(id="call_1", name="recall", arguments={"query": f"deploy{HIGH_SURROGATE}"})],
        )
    )

    result = await llm.call_with_tools(messages=[{"role": "user", "content": "hi"}], tools=[])

    assert result.content == "thinking"
    assert result.tool_calls[0].arguments["query"] == "deploy"
    assert result.tool_calls[0].arguments["query"].encode("utf-8")


# --- Prong D: entity names are text too (#3729) ---------------------------------


def test_fact_model_sanitizes_entity_names():
    """Entity names join the embedded string and the BM25 ``text_signals`` column."""
    fact = Fact(fact="Alex shipped it", fact_type="world", entities=[f"Al{HIGH_SURROGATE}ex", "Kubernetes"])
    assert fact.entities == ["Alex", "Kubernetes"]


def test_fact_model_drops_entity_names_that_sanitize_away():
    """A name made only of hostile characters is dropped, not stored blank."""
    fact = Fact(fact="Alex shipped it", fact_type="world", entities=[HIGH_SURROGATE, "Alex"])
    assert fact.entities == ["Alex"]


def test_fact_model_leaves_valid_entity_names_alone():
    fact = Fact(fact="Alex shipped it", fact_type="world", entities=["key:value", "café 🎉"])
    assert fact.entities == ["key:value", "café 🎉"]


def test_embedding_text_for_sanitized_fact_is_utf8_encodable():
    """The end the crash actually happened at: the string handed to the embedder.

    ``augment_texts_with_dates`` splices entity names into the fact text, so a
    surrogate in either field reaches the tokenizer as one un-encodable string.
    """
    from hindsight_api.engine.retain import embedding_processing
    from hindsight_api.engine.retain.types import ExtractedFact

    fact = Fact(
        fact=f"Alex laughed 😂{HIGH_SURROGATE}",
        fact_type="world",
        entities=[f"Al{HIGH_SURROGATE}ex"],
    )
    shim = ExtractedFact(fact_text=fact.fact, fact_type=fact.fact_type, entities=list(fact.entities or []))
    (augmented,) = embedding_processing.augment_texts_with_dates([shim], lambda d: "today")

    assert augmented.encode("utf-8")  # raised UnicodeEncodeError before the fix
    assert augmented == "Alex laughed 😂 [Alex]"


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


@pytest.mark.asyncio
async def test_retain_with_lone_surrogate_entity_name(memory, request_context):
    """A client-supplied entity name with an unpaired surrogate must not 500 (#3729).

    Entity names are spliced into the string handed to the embedder and joined
    into the ``text_signals`` column, so they crash in the same two places the
    fact text does — but nothing sanitized them at ingress.
    """
    bank_id = f"test_surrogate_entity_{datetime.now(timezone.utc).timestamp()}"
    unit_ids = await memory.retain_batch_async(
        bank_id=bank_id,
        contents=[
            {
                "content": "The deploy service ships releases.",
                "entities": [{"text": f"depl{HIGH_SURROGATE}oy"}, {"text": HIGH_SURROGATE}],
            }
        ],
        request_context=request_context,
    )
    assert isinstance(unit_ids, list)
