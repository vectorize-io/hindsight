"""Tag globs in observation_scopes and the bank's default_observation_scopes (#5666)."""

import uuid
from unittest.mock import MagicMock

import pytest

from hindsight_api.config import (
    ENV_DEFAULT_OBSERVATION_SCOPES,
    HindsightConfig,
    _get_raw_config,
    validate_observation_scopes,
)
from hindsight_api.engine.consolidation.consolidator import (
    _batch_scope_signature,
    _consolidation_batch_key,
    _resolve_obs_tags_list,
    _resolve_write_scopes,
)
from hindsight_api.engine.memories import get_memories
from hindsight_api.engine.retain.fact_extraction import _apply_default_observation_scopes
from hindsight_api.engine.retain.types import ExtractedFact
from hindsight_api.engine.schema import fq_table

# Two subjects, written by different agents and sessions: every tag set is unique.
TAG_SETS = [
    ["topic:billing", "source:agent-a", "session:1"],
    ["topic:billing", "source:agent-b", "session:2"],
    ["topic:billing", "source:agent-a", "session:3", "certainty:high"],
    ["topic:billing", "source:agent-c"],
    ["topic:auth", "source:agent-a", "session:4"],
    ["topic:auth", "source:agent-b", "session:5"],
    ["topic:auth", "source:agent-c", "session:6", "origin:chat"],
]


def _batch_keys(scopes):
    memories = [{"tags": t, "observation_scopes": scopes} for t in TAG_SETS]
    keys = {_consolidation_batch_key(m) for m in memories}
    # Each group must stay scope-homogeneous, or the dispatcher's guard splits it again.
    for key in keys:
        group = [m for m in memories if _consolidation_batch_key(m) == key]
        assert len({_batch_scope_signature(m) for m in group}) == 1
    return keys


def test_combined_fragments_one_batch_per_memory():
    assert len(_batch_keys(None)) == 7


def test_glob_scope_batches_by_subject():
    assert _batch_keys([["topic:*"]]) == {("scope", "topic:billing"), ("scope", "topic:auth")}


def _memory(scopes, tags=("topic:billing", "user:alice", "source:xx", "session:7")):
    return {"tags": list(tags), "observation_scopes": scopes}


@pytest.mark.parametrize(
    ("scopes", "expected"),
    [
        ([["topic:*"]], [["topic:billing"]]),
        ([["topic:*", "user:*"]], [["topic:billing", "user:alice"]]),
        ([["topic:*"], ["user:*"]], [["topic:billing"], ["user:alice"]]),
        ([["topic:*", "team:x"]], [["team:x", "topic:billing"]]),
        # A glob matching nothing drops only its own scope...
        ([["topic:*"], ["project:*"]], [["topic:billing"]]),
        # ...and a literal next to it keeps the scope alive.
        ([["project:*", "team:x"]], [["team:x"]]),
        # A scope written empty on purpose is still the untagged one.
        ([[]], [[]]),
    ],
)
def test_glob_resolution(scopes, expected):
    assert _resolve_obs_tags_list(_memory(scopes)) == expected
    assert sorted(_resolve_write_scopes(_memory(scopes)), key=sorted) == sorted(
        (frozenset(s) for s in expected), key=sorted
    )


def test_globs_matching_nothing_fall_back_to_own_tags_not_untagged():
    memory = _memory([["project:*"]])
    assert _consolidation_batch_key(memory) == _consolidation_batch_key(_memory(None))
    assert _resolve_write_scopes(memory) == [frozenset(memory["tags"])]


def test_glob_resolution_from_json_string():
    assert _resolve_obs_tags_list(_memory('[["topic:*"]]')) == [["topic:billing"]]


def _fact(scopes=None):
    return ExtractedFact(fact_text="x", fact_type="world", tags=["topic:billing"], observation_scopes=scopes)


def test_default_applies_only_when_nothing_declared():
    config = MagicMock(default_observation_scopes=[["topic:*"]])
    undeclared, explicit, empty = _fact(), _fact("per_tag"), _fact([])
    _apply_default_observation_scopes([undeclared, explicit, empty], config)
    assert undeclared.observation_scopes == [["topic:*"]]
    assert explicit.observation_scopes == "per_tag"
    assert empty.observation_scopes == []


def test_no_default_keeps_historical_behaviour():
    fact = _fact()
    _apply_default_observation_scopes([fact], MagicMock(default_observation_scopes=None))
    assert fact.observation_scopes is None


@pytest.mark.parametrize("value", ["combined", "shared", [["topic:*"]], [["a", "b"], ["c"]]])
def test_default_validation_accepts_retain_values(value):
    validate_observation_scopes(value, "default_observation_scopes")


@pytest.mark.parametrize("value", ["by_topic", ["topic:*"], [[1]], {"scope": "x"}])
def test_default_validation_rejects_other_values(value):
    with pytest.raises(ValueError, match="default_observation_scopes"):
        validate_observation_scopes(value, "default_observation_scopes")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("", None), ("shared", "shared"), ('[["topic:*", "user:*"]]', [["topic:*", "user:*"]])],
)
def test_env_default_is_parsed_at_startup(monkeypatch, raw, expected):
    monkeypatch.setenv(ENV_DEFAULT_OBSERVATION_SCOPES, raw)
    assert HindsightConfig.from_env().default_observation_scopes == expected


@pytest.mark.parametrize("raw", ["by_topic", '[["topic:*"]', '["topic:*"]'])
def test_env_default_typo_stops_startup(monkeypatch, raw):
    monkeypatch.setenv(ENV_DEFAULT_OBSERVATION_SCOPES, raw)
    with pytest.raises(ValueError, match=ENV_DEFAULT_OBSERVATION_SCOPES):
        HindsightConfig.from_env()


# ─── Through the engine: bank default → retain → consolidation ────────────────


@pytest.fixture
def enable_observations():
    config = _get_raw_config()
    original = config.enable_observations
    config.enable_observations = True
    yield
    config.enable_observations = original


async def _scan(memory, bank_id, fact_types):
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        page = await get_memories().scan_memories(
            conn=conn, fq_table=fq_table, bank_id=bank_id, fact_types=fact_types, limit=1_000_000
        )
    return list(page.memories)


@pytest.mark.asyncio
async def test_bank_default_scopes_observations_by_glob(memory, request_context, enable_observations):
    bank_id = f"test_default_scopes_{uuid.uuid4().hex[:8]}"
    try:
        await memory.update_bank_config(
            bank_id,
            {"default_observation_scopes": [["topic:*"]], "enable_auto_consolidation": False},
            request_context=request_context,
        )
        for session, scopes in (("a", None), ("b", None), ("c", "combined")):
            item = {
                "content": f"Alice said in session {session} she prefers dark roast coffee every morning.",
                "document_id": f"session-{session}",
                "tags": ["topic:coffee", f"session:{session}"],
            }
            if scopes:
                item["observation_scopes"] = scopes
            await memory.retain_batch_async(bank_id=bank_id, contents=[item], request_context=request_context)

        facts = await _scan(memory, bank_id, ["experience", "world"])
        by_session = {next(t for t in f.tags if t.startswith("session:")): f.observation_scopes for f in facts}
        assert by_session == {"session:a": [["topic:*"]], "session:b": [["topic:*"]], "session:c": "combined"}

        await memory.run_consolidation(bank_id=bank_id, request_context=request_context)
        observation_tags = {tuple(sorted(o.tags or [])) for o in await _scan(memory, bank_id, ["observation"])}
        assert ("topic:coffee",) in observation_tags, "Sessions a and b consolidate under their topic alone"
        assert observation_tags <= {("topic:coffee",), ("session:c", "topic:coffee")}, "An explicit value wins"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_bank_config_rejects_a_malformed_default(memory, request_context):
    bank_id = f"test_default_scopes_bad_{uuid.uuid4().hex[:8]}"
    try:
        with pytest.raises(ValueError, match="default_observation_scopes"):
            await memory.update_bank_config(
                bank_id, {"default_observation_scopes": ["topic:*"]}, request_context=request_context
            )
        with pytest.raises(ValueError, match="default_observation_scopes"):
            await memory.update_bank_config(
                bank_id,
                {"retain_strategies": {"support": {"default_observation_scopes": "by_topic"}}},
                request_context=request_context,
            )
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
