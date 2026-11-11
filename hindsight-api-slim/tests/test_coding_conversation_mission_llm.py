"""The coding-agents plugin's conversation mission keeps the method behind a checked result.

A Claude Code session found that `prc` holds cents and `prc_usd` is NULL for most rows, answered
with `prc/100.0`, and was told it was right. Extraction under the plugin's mission kept "prc is
cents, prc_usd is dollars" and dropped both the method and the NULL caveat, so a later session
picked `prc_usd` and failed (CL-Bench database_exploration, Q15). The mission now asks for the
method behind a checked outcome and the caveats found. What the model extracts is model
behaviour, so a judge decides it.

The mission is read from the plugin's source, its single home, so this test follows its edits.
"""

import dataclasses
import json
import re
from datetime import datetime
from pathlib import Path

import pytest

from hindsight_api import LLMConfig
from hindsight_api.config import _get_raw_config
from hindsight_api.engine.retain.fact_extraction import extract_facts_from_text
from tests.llm_judge import assert_meets_criteria

pytestmark = pytest.mark.hs_llm_core

_MISSIONS_TS = Path(__file__).resolve().parents[2] / "hindsight-integrations/coding-agents/src/core/missions.ts"


def _conversation_mission() -> str:
    source = _MISSIONS_TS.read_text()
    body = re.search(r"export const CONVERSATION_MISSION =(.*?);\n", source, re.S).group(1)
    # The body is string literals joined by +, in either quote style ('... "Edit x.py" ...' too).
    literals = re.findall(r'"((?:[^"\\]|\\.)*)"|\'((?:[^\'\\]|\\.)*)\'', body)
    return "".join(json.loads(f'"{dq}"') if dq or not sq else sq.replace("\\'", "'") for dq, sq in literals)


_SESSION = "\n".join(
    [
        '{"role":"user","content":"Question 5/15. What is the median price of electronics products '
        '(in dollars)? Only consider products with a non-zero price."}',
        '{"role":"assistant","content":"{\\"action\\":\\"QUERY\\",\\"content\\":\\"SELECT COUNT(*), '
        'SUM(prc_usd IS NULL), MIN(prc), MAX(prc) FROM items_g2\\"}"}',
        '{"role":"user","content":"Query result: 20000 | 13654 | 99 | 2688800"}',
        '{"role":"assistant","content":"{\\"action\\":\\"ANSWER\\",\\"content\\":\\"21.99\\"}"}',
        '{"role":"user","content":"Question 5: CORRECT! Your answer: 21.99"}',
        '{"role":"assistant","content":"Question 5 was marked correct: the median is $21.99. The prc '
        "column in items_g2 holds prices in cents, and prc_usd is NULL for 13,654 of the 20,000 rows, "
        'so I took the median of prc/100.0 over the rows with prc > 0."}',
    ]
)


@pytest.mark.asyncio
async def test_a_checked_result_keeps_its_method_and_the_data_caveat():
    # concise, as the plugin writes its sessions (DEFAULT_RETAIN_EXTRACTION_MODE)
    config = dataclasses.replace(
        _get_raw_config(), retain_mission=_conversation_mission(), retain_extraction_mode="concise"
    )

    facts, _, _ = await extract_facts_from_text(
        text=_SESSION,
        event_date=datetime(2026, 11, 11),
        context="conversation between the user and you (the coding agent)",
        llm_config=LLMConfig.from_env(),
        config=config,
    )

    await assert_meets_criteria(
        response="\n".join(f"- {f.fact}" for f in facts),
        criteria=(
            "The facts record HOW the accepted answer was computed — the median of prc/100.0 (prc in cents) — "
            "and the caveat that prc_usd is NULL for most rows (13,654 of 20,000)."
        ),
        context="Facts extracted from a coding-agent session whose answer was judged correct.",
    )
