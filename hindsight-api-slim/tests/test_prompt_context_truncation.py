"""A memory's context is capped, or left out, in every internal prompt that lists memories.

Callers can store a huge context on each fact; repeated per memory it would
bloat the think prompt. These pin the cap there, and that consolidation leaves
the context out entirely (#5996).
"""

import json

from hindsight_api.engine.consolidation.consolidator import _build_observations_for_llm
from hindsight_api.engine.prompt_utils import PROMPT_CONTEXT_MAX_CHARS, truncate_context_for_prompt
from hindsight_api.engine.response_models import MemoryFact
from hindsight_api.engine.search.think_utils import format_facts_for_prompt

LONG = "x" * (PROMPT_CONTEXT_MAX_CHARS + 50)
CUT = "x" * PROMPT_CONTEXT_MAX_CHARS + "..."


def test_truncate_context_for_prompt():
    assert truncate_context_for_prompt(LONG) == CUT
    assert truncate_context_for_prompt("x" * PROMPT_CONTEXT_MAX_CHARS) == "x" * PROMPT_CONTEXT_MAX_CHARS
    assert truncate_context_for_prompt("chat") == "chat"
    assert truncate_context_for_prompt(None) is None


def test_consolidation_source_memory_carries_no_context():
    obs = MemoryFact(id="o1", text="obs", fact_type="observation", source_fact_ids=["f1"])
    src = MemoryFact(id="f1", text="fact", fact_type="world", context=LONG)
    [obs_data] = _build_observations_for_llm([obs], {"f1": src})
    assert obs_data["source_memories"] == [{"text": "fact"}]


def test_think_prompt_context_is_capped():
    fact = MemoryFact(id="f1", text="fact", fact_type="world", context=LONG)
    assert json.loads(format_facts_for_prompt([fact]))[0]["context"] == CUT
