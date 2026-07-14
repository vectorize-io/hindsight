"""Real-LLM check that the minify directive is FORMATTING-ONLY.

Enabling ``HINDSIGHT_API_MINIFY_LLM_JSON_OUTPUT`` must change only the JSON
whitespace the model emits, not WHICH facts it extracts. This is an
instruction-following property MockLLM cannot simulate (it echoes input), so the
module is marked ``hs_llm_core`` and runs in the single-provider quality CI job.
The semantic equivalence is judged (paraphrase-tolerant), not string-matched.
"""

from datetime import datetime

import pytest

from hindsight_api import LLMConfig
from hindsight_api.config import _get_raw_config, clear_config_cache
from hindsight_api.engine.retain.fact_extraction import extract_facts_from_text
from tests.llm_judge import assert_meets_criteria

pytestmark = pytest.mark.hs_llm_core

_TEXT = (
    "Alice Chen joined Acme Corp in March 2021 as a senior software engineer in "
    "Austin, reporting to Bob Ruiz. She previously worked at Globex for three years "
    "on payments infrastructure, and holds a master's degree from UT Austin."
)


async def _extract() -> list:
    facts, _, _ = await extract_facts_from_text(
        text=_TEXT,
        event_date=datetime(2024, 11, 13),
        context="Profile note",
        llm_config=LLMConfig.from_env(),
        agent_name="TestUser",
        config=_get_raw_config(),
    )
    return facts


@pytest.mark.asyncio
async def test_minify_directive_is_formatting_only(monkeypatch):
    # Baseline: minify OFF (the default).
    monkeypatch.delenv("HINDSIGHT_API_MINIFY_LLM_JSON_OUTPUT", raising=False)
    clear_config_cache()
    facts_off = await _extract()

    # Minify ON.
    monkeypatch.setenv("HINDSIGHT_API_MINIFY_LLM_JSON_OUTPUT", "true")
    clear_config_cache()
    facts_on = await _extract()

    clear_config_cache()  # restore default for following tests in this worker

    assert len(facts_off) > 0, "baseline extraction produced no facts"
    assert len(facts_on) > 0, "minified extraction produced no facts"

    off_text = "\n".join(f.fact for f in facts_off)
    on_text = "\n".join(f.fact for f in facts_on)

    # Format-only: enabling minify must not drop or alter the information extracted.
    await assert_meets_criteria(
        response=on_text,
        criteria=(
            "These facts capture the same core information as the reference facts, with no "
            "key fact from the reference missing: Alice Chen joining Acme Corp in 2021 as a "
            "senior software engineer in Austin, reporting to Bob Ruiz; her prior three years "
            "at Globex on payments infrastructure; and her UT Austin master's degree. "
            "Paraphrases are fine — only the presence of the information matters."
        ),
        context=f"Reference facts (minify OFF):\n{off_text}\n\nSource text:\n{_TEXT}",
    )
