"""Deterministic regression tests for consolidation source-language guarding."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hindsight_api.engine.consolidation.consolidator import (
    _consolidate_batch_with_llm,
    _ConsolidationBatchResponse,
    _CreateAction,
    _UpdateAction,
)
from hindsight_api.engine.consolidation.prompts import build_consolidation_system_prompt
from hindsight_api.engine.language_detection import detect_dominant_language, detect_language, languages_match


def _config(**overrides):
    values = {
        "llm_output_language": None,
        "consolidation_language_validation": True,
        "consolidation_language_validation_failure_policy": "fail_batch",
        "observations_mission": None,
        "llm_supports_max_items": True,
        "consolidation_max_attempts": 3,
        "consolidation_llm_max_retries": None,
        "consolidation_max_completion_tokens": None,
        "llm_strict_schema_consolidation": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _create_response(text: str) -> _ConsolidationBatchResponse:
    return _ConsolidationBatchResponse(
        creates=[_CreateAction(text=text, source_fact_ids=["fact-1"])],
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("我叫张三，住在深圳。", "zh"),
        ("Alice lives in Shenzhen and owns a cat.", "en"),
        ("I am a software engineer.", "en"),
        ("Bonjour, je suis à Paris.", "fr"),
        ("Hola, vivo en Madrid.", "es"),
        ("日本語の文章です。", "ja"),
    ],
)
def test_detect_language_for_common_source_scripts(text: str, expected: str):
    assert detect_language(text).language == expected


def test_mixed_language_dominance_uses_first_fact_on_tie():
    chinese = "张三丰"
    english = "the"
    assert detect_dominant_language([chinese, english]).language == "zh"
    assert detect_dominant_language([english, chinese]).language == "en"


def test_ambiguous_latin_text_does_not_claim_to_be_english():
    ambiguous_latin = detect_language("Juan reside Madrid")
    assert ambiguous_latin.language == "latin"
    assert languages_match(detect_language("Bonjour, je suis à Paris."), ambiguous_latin)
    assert not languages_match(detect_language("用户住在深圳。"), ambiguous_latin)


@pytest.mark.asyncio
async def test_mixed_batch_validates_each_action_against_its_referenced_facts():
    llm = SimpleNamespace(
        call=AsyncMock(
            return_value=_ConsolidationBatchResponse(
                creates=[
                    _CreateAction(text="用户住在深圳。", source_fact_ids=["fact-zh"]),
                    _CreateAction(text="User works at Google.", source_fact_ids=["fact-en"]),
                ]
            )
        )
    )

    result = await _consolidate_batch_with_llm(
        llm_config=llm,
        memories=[
            {"id": "fact-zh", "text": "用户住在深圳。"},
            {"id": "fact-en", "text": "User works at Google."},
        ],
        union_observations=[],
        union_source_facts={},
        config=_config(),
    )

    assert llm.call.call_count == 1
    assert result.failed is False
    assert len(result.creates) == 2


@pytest.mark.asyncio
async def test_language_validation_can_be_disabled():
    llm = SimpleNamespace(call=AsyncMock(return_value=_create_response("User lives in Shenzhen.")))

    result = await _consolidate_batch_with_llm(
        llm_config=llm,
        memories=[{"id": "fact-1", "text": "用户住在深圳。"}],
        union_observations=[],
        union_source_facts={},
        config=_config(consolidation_language_validation=False),
    )

    assert llm.call.call_count == 1
    assert result.failed is False
    assert result.creates[0].text == "User lives in Shenzhen."


@pytest.mark.asyncio
async def test_language_validation_allows_insufficient_source_evidence():
    llm = SimpleNamespace(call=AsyncMock(return_value=_create_response("TLS 1.3 is enabled.")))

    result = await _consolidate_batch_with_llm(
        llm_config=llm,
        memories=[{"id": "fact-1", "text": "TLS 1.3"}],
        union_observations=[],
        union_source_facts={},
        config=_config(),
    )

    assert llm.call.call_count == 1
    assert result.failed is False


@pytest.mark.asyncio
async def test_language_validation_ignores_action_without_valid_source_fact():
    response = _ConsolidationBatchResponse(
        creates=[_CreateAction(text="English output", source_fact_ids=["not-in-batch"])],
    )
    llm = SimpleNamespace(call=AsyncMock(return_value=response))

    result = await _consolidate_batch_with_llm(
        llm_config=llm,
        memories=[{"id": "fact-1", "text": "用户住在深圳。"}],
        union_observations=[],
        union_source_facts={},
        config=_config(),
    )

    assert llm.call.call_count == 1
    assert result.failed is False


def test_default_consolidation_prompt_documents_source_language_rule():
    prompt = build_consolidation_system_prompt()
    assert "dominant language" in prompt
    assert "mixed-language" in prompt
    assert "first referenced fact" in prompt
    assert "existing observation uses another language" in prompt


@pytest.mark.asyncio
async def test_wrong_language_response_gets_one_corrective_retry(caplog):
    llm = SimpleNamespace(provider="openai", model="qwen-test", call=AsyncMock())
    llm.call.side_effect = [
        _create_response("User lives in Shenzhen."),
        _create_response("用户住在深圳。"),
    ]

    result = await _consolidate_batch_with_llm(
        llm_config=llm,
        memories=[{"id": "fact-1", "text": "用户住在深圳。"}],
        union_observations=[],
        union_source_facts={},
        config=_config(),
        bank_id="language-bank",
    )

    assert llm.call.call_count == 2
    assert result.failed is False
    assert result.creates[0].text == "用户住在深圳。"
    assert result.language_validation_failures == 1
    assert result.language_correction_retries == 1
    second_user_message = llm.call.call_args_list[1].kwargs["messages"][1]["content"]
    assert "LANGUAGE CORRECTION" in second_user_message
    assert "Chinese" in second_user_message
    assert llm.call.call_args_list[0].kwargs["response_format"] is llm.call.call_args_list[1].kwargs["response_format"]
    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "bank=language-bank" in log_text
    assert "provider=openai" in log_text
    assert "model=qwen-test" in log_text
    assert "source_language=zh" in log_text
    assert "output_language=en" in log_text
    assert "retry=True" in log_text
    assert "用户住在深圳" not in log_text


@pytest.mark.asyncio
async def test_second_wrong_language_response_fails_batch_by_default():
    llm = SimpleNamespace(call=AsyncMock())
    llm.call.side_effect = [
        _create_response("User lives in Shenzhen."),
        _create_response("User lives in Shenzhen."),
    ]

    result = await _consolidate_batch_with_llm(
        llm_config=llm,
        memories=[{"id": "fact-1", "text": "用户住在深圳。"}],
        union_observations=[],
        union_source_facts={},
        config=_config(),
    )

    assert llm.call.call_count == 2
    assert result.failed is True
    assert result.creates == []
    assert result.language_validation_failures == 2


@pytest.mark.asyncio
async def test_fail_open_policy_keeps_second_wrong_response():
    llm = SimpleNamespace(call=AsyncMock())
    wrong = _create_response("User lives in Shenzhen.")
    llm.call.side_effect = [wrong, wrong]

    result = await _consolidate_batch_with_llm(
        llm_config=llm,
        memories=[{"id": "fact-1", "text": "用户住在深圳。"}],
        union_observations=[],
        union_source_facts={},
        config=_config(consolidation_language_validation_failure_policy="fail_open"),
    )

    assert result.failed is False
    assert result.creates[0].text == "User lives in Shenzhen."


@pytest.mark.asyncio
async def test_explicit_output_language_bypasses_source_validation():
    llm = SimpleNamespace(call=AsyncMock(return_value=_create_response("User lives in Shenzhen.")))

    result = await _consolidate_batch_with_llm(
        llm_config=llm,
        memories=[{"id": "fact-1", "text": "用户住在深圳。"}],
        union_observations=[],
        union_source_facts={},
        config=_config(llm_output_language="English"),
    )

    assert llm.call.call_count == 1
    assert result.failed is False
    assert result.creates[0].text == "User lives in Shenzhen."


@pytest.mark.asyncio
async def test_updates_are_validated_the_same_as_creates():
    llm = SimpleNamespace(
        call=AsyncMock(
            side_effect=[
                _ConsolidationBatchResponse(
                    updates=[
                        _UpdateAction(
                            text="User lives in Shenzhen.",
                            observation_id="observation-1",
                            source_fact_ids=["fact-1"],
                        )
                    ]
                ),
                _ConsolidationBatchResponse(
                    updates=[
                        _UpdateAction(
                            text="用户住在深圳。",
                            observation_id="observation-1",
                            source_fact_ids=["fact-1"],
                        )
                    ]
                ),
            ]
        )
    )

    result = await _consolidate_batch_with_llm(
        llm_config=llm,
        memories=[{"id": "fact-1", "text": "用户住在深圳。"}],
        union_observations=[],
        union_source_facts={},
        config=_config(),
    )

    assert llm.call.call_count == 2
    assert result.failed is False
    assert result.updates[0].text == "用户住在深圳。"


@pytest.mark.asyncio
@pytest.mark.hs_llm_core
async def test_real_model_preserves_chinese_source_language(memory_real_llm):
    """The real provider should follow the default source-language prompt rule."""
    from dataclasses import replace

    from hindsight_api.config import _get_raw_config
    from tests.llm_judge import assert_meets_criteria

    config = replace(
        _get_raw_config(),
        llm_output_language=None,
        consolidation_language_validation=True,
        consolidation_language_validation_failure_policy="fail_batch",
    )
    result = await _consolidate_batch_with_llm(
        llm_config=memory_real_llm._consolidation_llm_config,
        memories=[
            {
                "id": "11111111-1111-1111-1111-111111111111",
                "text": "张三住在深圳，养了一只名叫豆豆的猫，每周末带豆豆去深圳湾公园散步。",
            }
        ],
        union_observations=[],
        union_source_facts={},
        config=config,
        bank_id="language-judge",
    )

    assert result.failed is False
    observation_text = "\n".join(action.text for action in [*result.creates, *result.updates])
    assert observation_text, "The real model should produce at least one observation action"
    await assert_meets_criteria(
        response=observation_text,
        criteria=(
            "Every observation is written predominantly in Chinese, matching the language of the source fact. "
            "English instructions, names, and technical terms may remain unchanged, but the model must not "
            "translate the observation into English."
        ),
        context=(
            "The only new source fact is Chinese: 张三住在深圳，养了一只名叫豆豆的猫，每周末带豆豆去深圳湾公园散步。 "
            "No explicit output language was configured."
        ),
        msg=f"Consolidation should preserve Chinese source language: {observation_text}",
    )
