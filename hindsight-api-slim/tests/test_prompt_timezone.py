"""#6441: every time a model reads is in the bank's zone, and the prompt says which zone.

Stored times carry no offset of their own, so the bank's ``default_timezone`` is the local time.
Shown as UTC, a fact at 07:10 on 8 Nov in Shanghai reached consolidation as 23:10 on 7 Nov and the
observation text said "7 November". Whether the model then writes the local date is graded by
``test_consolidation_timezone_llm.py``; this pins what it is given.
"""

from datetime import UTC, datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from hindsight_api.engine.consolidation.consolidator import _build_observations_for_llm, _prompt_time
from hindsight_api.engine.consolidation.prompts import build_consolidation_input
from hindsight_api.engine.observation_tree import TreeObservation, build_observation_tree, iter_files
from hindsight_api.engine.reflect import prompts as reflect_prompts
from hindsight_api.engine.reflect.agent import _short_date
from hindsight_api.engine.reflect.presentation import ToolResultPresenter, compact_timestamp
from hindsight_api.engine.response_models import MemoryFact
from hindsight_api.engine.stored_dates import bank_timezone

SHANGHAI = ZoneInfo("Asia/Shanghai")
# 07:10 on 8 Nov in Shanghai: still 7 Nov in UTC.
MORNING = datetime(2026, 11, 7, 23, 10, tzinfo=UTC)


def test_bank_timezone_defaults_to_utc():
    assert bank_timezone(SimpleNamespace(default_timezone=None)) is UTC
    assert bank_timezone(SimpleNamespace(default_timezone="Asia/Shanghai")) == SHANGHAI


def test_consolidation_shows_times_in_the_bank_zone():
    assert _prompt_time(MORNING, SHANGHAI) == "2026-11-08T07:10+08:00"
    # A UTC bank reads exactly as before.
    assert _prompt_time(MORNING) == "2026-11-07T23:10Z"
    # Observations and their sources use the same zone as the facts: one format per prompt (#5938).
    obs = MemoryFact(id="o", text="t", fact_type="observation", mentioned_at=MORNING.isoformat())
    [shown] = _build_observations_for_llm([obs], {}, None, SHANGHAI)
    assert shown["mentioned_at"] == "2026-11-08T07:10+08:00"


def test_consolidation_prompt_names_the_zone():
    prompt = build_consolidation_input("[F1] x", "[]", timezone_name="Asia/Shanghai")
    assert "## TIMEZONE" in prompt and "local time in Asia/Shanghai" in prompt
    assert "local time in UTC" in build_consolidation_input("[F1] x", "[]")


def test_reflect_shows_memory_times_in_the_bank_zone():
    assert compact_timestamp("2026-11-07T23:10:00+00:00", SHANGHAI) == "2026-11-08 07:10"
    assert compact_timestamp("2026-11-07T23:10:00+00:00") == "2026-11-07 23:10"
    # Any offset is converted, not only UTC: the prompt names one zone for all of them.
    assert compact_timestamp("2026-11-08T01:10:00+02:00", SHANGHAI) == "2026-11-08 07:10"
    assert compact_timestamp("2026-11-08", SHANGHAI) == "2026-11-08"
    out = ToolResultPresenter(SHANGHAI).present(
        {"memories": [{"id": "x", "text": "t", "mentioned_at": MORNING.isoformat()}]}
    )
    assert out["memories"][0]["mentioned_at"] == "2026-11-08 07:10"
    chunks = ToolResultPresenter(SHANGHAI).present(
        {"chunks": {"c": {"chunk_text": "t", "mentioned_at": MORNING.isoformat()}}}
    )
    assert chunks["chunks"]["c1"]["mentioned_at"] == "2026-11-08 07:10"
    assert _short_date(MORNING.isoformat(), SHANGHAI) == "2026-11-08 07:10"


def test_reflect_names_the_zone_beside_now():
    text = reflect_prompts._current_datetime(SHANGHAI)
    assert text.endswith("Asia/Shanghai (UTC+08:00)")
    assert reflect_prompts._current_datetime(UTC).endswith(" UTC")
    assert "Asia/Shanghai" in reflect_prompts.build_final_system_prompt(tz=SHANGHAI)
    assert "Asia/Shanghai" in reflect_prompts.build_system_prompt_for_tools({"name": "B"}, tz=SHANGHAI)


def test_observation_tree_dates_and_months_follow_the_bank_zone():
    obs = [TreeObservation(id="o", text="download at 57%", proof_count=1, entities=[], mentioned_at=MORNING)]
    [topic] = [f for f in iter_files(build_observation_tree(obs, tz=SHANGHAI)) if f.kind == "topic"]
    assert "2026-11-08 07:10" in topic.content
    assert "Times are Asia/Shanghai." in topic.content
    [utc_topic] = [f for f in iter_files(build_observation_tree(obs)) if f.kind == "topic"]
    assert "2026-11-07 23:10" in utc_topic.content and "Times are UTC." in utc_topic.content
