"""A configured output language actually reaches retain's facts and reflect's answer (#3776).

``HINDSIGHT_API_LLM_OUTPUT_LANGUAGE`` was a silent no-op for both: each prompt carried its
own "never translate the source language" rule AND the appended translate-into-X directive,
and the model resolved the contradiction in favour of the rule — it is phrased far more
forcefully and comes first. Nothing errored; the facts were simply stored in the source
language while ``/config`` reported the setting as active.

That failure is invisible to the deterministic tests. Which rule text ends up in the prompt
for a given config is pinned in ``test_multilingual_bm25.py``, and those assertions passed
happily on the broken code — the old retain tests only ever asserted the directive was
*present*, never that the contradicting rule was *absent*, which is exactly how this
shipped. Whether the model then *obeys* the prompt is not something MockLLM or a string
match can show, so these drive the real call and judge the output. Same split, and the same
shape, as ``test_consolidation_output_language.py``.

Scope: the **configured** half of the mutual exclusion, which nothing else covers for retain
or reflect. The unset half — that the source-language rule is still there and still works,
per #181 — is owned by ``test_multilingual.py`` behaviourally and by
``test_retain_unset_requires_source_language`` / ``test_reflect_unset_requires_source_language``
deterministically.

Chinese source material, because it is unambiguous to the judge: a sentence left in the
source language is a different script, not a vocabulary guess. Translating *into* the
prompt's own language is also the easy direction for any multilingual model — a failure here
is the prompt contradicting itself again, not the model being weak. The language judgement
is the judge's alone: a script gate on the output would reject the proper nouns
(张伟 / 腾讯 / 阿里云) the criteria deliberately allow verbatim, and flake.
"""

import dataclasses
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from hindsight_api import LLMConfig
from hindsight_api.config import _get_raw_config
from hindsight_api.engine.reflect.agent import run_reflect_agent
from hindsight_api.engine.reflect.prompts import build_final_prompt, build_final_system_prompt
from hindsight_api.engine.retain.fact_extraction import extract_facts_from_text
from tests.llm_judge import assert_meets_criteria

pytestmark = pytest.mark.hs_llm_core


# The shape of the production report in #3776: source content in one language, the operator
# expecting every stored artifact in another. Chinese rather than the issue's Spanish so a
# sentence left untranslated is unmistakable to the judge.
_CHINESE_SOURCE = """
张伟是一位资深软件工程师，在腾讯工作了五年。他专门研究分布式系统，并领导了公司微服务架构的开发。
团队使用Kubernetes进行容器编排，并部署到阿里云。他们遵循敏捷方法论，采用两周冲刺周期。
合并前必须进行代码审查。
"""


@pytest.mark.asyncio
async def test_retain_configured_output_language_overrides_source_language():
    """Chinese input + ``llm_output_language="English"`` → English facts.

    The bug: the extraction prompt's own rule ("STRICTLY FORBIDDEN from translating")
    outranked the appended directive, so the facts came back in Chinese and the setting
    did nothing. Dropping the rule when a language is configured is what makes this pass.
    """
    config = dataclasses.replace(_get_raw_config(), llm_output_language="English")

    facts, _, _ = await extract_facts_from_text(
        text=_CHINESE_SOURCE,
        event_date=datetime(2024, 1, 15, tzinfo=UTC),
        context="团队概述",
        llm_config=LLMConfig.from_env(),
        agent_name="TestUser",
        config=config,
    )

    assert facts, "Should extract at least one fact from the Chinese source"
    extracted = "\n".join(f"- {fact.fact}" for fact in facts)

    await assert_meets_criteria(
        response=extracted,
        criteria=(
            "Every fact is written in English, even though the source material was Chinese. "
            "Proper nouns may be transliterated (e.g. 'Zhang Wei', 'Tencent') or left in their "
            "original script; the sentences themselves must be English."
        ),
        context=(
            "Fact extraction was given Chinese source text about a senior software engineer at "
            "Tencent who works on distributed systems, and about a team using Kubernetes on "
            "Alibaba Cloud with two-week sprints and mandatory code review. The output language "
            "was configured to English, which must override the source language."
        ),
        msg="HINDSIGHT_API_LLM_OUTPUT_LANGUAGE must override the source language for retain facts",
    )


_REFLECT_BANK_PROFILE = {"name": "工程团队记忆"}
_REFLECT_QUERY = "张伟负责什么工作？团队用什么部署？"
_REFLECT_MEMORIES = [
    "张伟是资深软件工程师，专门研究分布式系统，并领导微服务架构的开发。",
    "团队使用Kubernetes进行容器编排，并部署到阿里云。",
]


async def _assert_reflect_answer_is_english(answer: str) -> None:
    assert answer.strip(), "Reflect should produce an answer"
    await assert_meets_criteria(
        response=answer,
        criteria=(
            "The answer is written in English, even though the question and the retrieved data "
            "were both in Chinese. Proper nouns may be transliterated or left in their original "
            "script; the sentences themselves must be English."
        ),
        context=(
            "Reflect was asked, in Chinese, what Zhang Wei works on and what the team deploys "
            "with. The retrieved facts were also in Chinese (distributed systems, microservice "
            "architecture, Kubernetes, Alibaba Cloud). The output language was configured to "
            "English, which must override the question's language."
        ),
        msg="HINDSIGHT_API_LLM_OUTPUT_LANGUAGE must override the question language for reflect answers",
    )


@pytest.mark.asyncio
async def test_reflect_done_path_configured_output_language_overrides_question_language(llm_config):
    """Chinese question and Chinese memories + configured English → English ``done()`` answer.

    This is the path that produces most reflect answers: the tool-calling loop under
    ``build_system_prompt_for_tools``, ending in a ``done()`` call. The first cut of this fix
    only reached ``build_final_system_prompt``, whose sole caller is the forced-synthesis
    fallback, so a normal run still answered in Chinese — and the forced-path test below,
    which assembles its prompts by hand, could not see it. This one drives the real agent
    and asserts the answer did not come from the fallback.
    """
    recall_fn = AsyncMock(
        return_value={
            "memories": [
                {"id": f"mem-{idx}", "text": text, "mentioned_at": "2024-01-15T00:00:00Z"}
                for idx, text in enumerate(_REFLECT_MEMORIES, 1)
            ]
        }
    )
    result = await run_reflect_agent(
        llm_config=llm_config,
        bank_id="test-bank",
        query=_REFLECT_QUERY,
        bank_profile=_REFLECT_BANK_PROFILE,
        search_mental_models_fn=AsyncMock(return_value={"mental_models": []}),
        search_observations_fn=AsyncMock(return_value={"observations": []}),
        recall_fn=recall_fn,
        expand_fn=AsyncMock(return_value={"memories": []}),
        has_mental_models=False,
        include_observations=False,
        budget="low",
        llm_output_language="English",
    )

    assert not any(call.scope == "final" for call in result.llm_trace), (
        "Answer came from forced synthesis, so the done() path was not exercised: "
        f"{[call.scope for call in result.llm_trace]}"
    )
    await _assert_reflect_answer_is_english(result.text)


@pytest.mark.asyncio
async def test_reflect_forced_synthesis_configured_output_language_overrides_question_language(llm_config):
    """Chinese question and Chinese retrieved data + configured English → English answer.

    Reflect needed BOTH halves of the fix, and this test is what proved the second one was
    missing. Dropping the contradicting source-language rule is not enough on its own: with
    the rule gone and only the directive left, the answer still came back in Chinese 12/12
    on gemini-2.5-flash-lite and 5/5 on gemini-2.5-flash, because the directive sat at the
    end of the SYSTEM prompt and the question and retrieved data both arrive after it. The
    directive now rides at the end of the user message, which measured 12/12 and 5/5 English.

    Drives the real synthesis call the way ``_forced_final_synthesis`` does: the same system
    prompt from ``build_final_system_prompt`` and user prompt from ``build_final_prompt``, one
    tool-less call, ``scope="reflect"``.
    """
    context_history = [{"tool": "recall", "output": {"results": [{"text": text} for text in _REFLECT_MEMORIES]}}]

    system_prompt = build_final_system_prompt(_REFLECT_BANK_PROFILE.get("mission"), "English", None)
    prompt = build_final_prompt(_REFLECT_QUERY, context_history, _REFLECT_BANK_PROFILE, llm_output_language="English")

    answer = await llm_config.call(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        scope="reflect",
    )

    await _assert_reflect_answer_is_english(answer)
