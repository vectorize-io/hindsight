"""Test observation tracking for a sequence of horse-related memories.

This test retains a series of facts about horses on a farm and inspects
how observations track the evolving state over time, with full prompt debugging.
"""

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

import pytest

from hindsight_api.config import _get_raw_config
from hindsight_api.engine.consolidation import consolidator as consolidator_mod
from hindsight_api.engine.memory_engine import MemoryEngine


@pytest.fixture(autouse=True)
def enable_observations():
    """Enable observations for all tests in this module."""
    config = _get_raw_config()
    original_value = config.enable_observations
    config.enable_observations = True
    yield
    config.enable_observations = original_value


@dataclass
class _ActionLog:
    text: str
    source_fact_ids: list[str] = field(default_factory=list)
    observation_id: str = ""


@dataclass
class _ConsolidationResponse:
    creates: list[_ActionLog] = field(default_factory=list)
    updates: list[_ActionLog] = field(default_factory=list)
    deletes: list[_ActionLog] = field(default_factory=list)


@dataclass
class _ConsolidationDebugEntry:
    facts: str
    observations_text: str
    response: _ConsolidationResponse


# Store prompts/responses for debugging
_debug_log: list[_ConsolidationDebugEntry] = []


def _fact_line(m: dict[str, Any]) -> str:
    text = f"[{m['id']}] {m['text']}"
    temporal_parts = []
    if m.get("occurred_start"):
        temporal_parts.append(f"occurred_start={m['occurred_start']}")
    if m.get("occurred_end"):
        temporal_parts.append(f"occurred_end={m['occurred_end']}")
    if m.get("mentioned_at"):
        temporal_parts.append(f"mentioned_at={m['mentioned_at']}")
    if temporal_parts:
        text += f" ({', '.join(temporal_parts)})"
    return text


async def _instrumented_consolidate(
    original_fn: Any,
    *,
    llm_config: Any,
    memories: list[dict[str, Any]],
    union_observations: Any,
    union_source_facts: Any,
    config: Any = None,
    remaining_observation_slots: int | None = None,
    max_observations_per_scope: int = -1,
) -> Any:
    """Wrapper that captures the prompt and response for debugging."""
    if union_observations:
        obs_list = consolidator_mod._build_observations_for_llm(union_observations, union_source_facts)
        observations_text = json.dumps(obs_list, indent=2)
    else:
        observations_text = "[]"

    facts_lines = "\n".join(_fact_line(m) for m in memories)

    result = await original_fn(
        llm_config=llm_config,
        memories=memories,
        union_observations=union_observations,
        union_source_facts=union_source_facts,
        config=config,
        remaining_observation_slots=remaining_observation_slots,
        max_observations_per_scope=max_observations_per_scope,
    )

    _debug_log.append(_ConsolidationDebugEntry(
        facts=facts_lines,
        observations_text=observations_text,
        response=_ConsolidationResponse(
            creates=[_ActionLog(text=c.text, source_fact_ids=c.source_fact_ids) for c in result.creates],
            updates=[
                _ActionLog(text=u.text, observation_id=u.observation_id, source_fact_ids=u.source_fact_ids)
                for u in result.updates
            ],
            deletes=[_ActionLog(text="", observation_id=d.observation_id) for d in result.deletes],
        ),
    ))

    return result


def _print_consolidation_debug(entry: _ConsolidationDebugEntry, index: int) -> None:
    """Print a single consolidation LLM call for debugging."""
    print(f"\n  --- LLM Call #{index} ---")
    print("  FACTS sent to LLM:")
    for line in entry.facts.split("\n"):
        print(f"    {line}")
    print("\n  EXISTING OBSERVATIONS sent to LLM:")
    obs_data = json.loads(entry.observations_text)
    if obs_data:
        for obs in obs_data:
            src_summary = ""
            if obs.get("source_memories"):
                src_texts = [sm["text"] for sm in obs["source_memories"]]
                src_summary = f" (sources: {src_texts})"
            print(f"    [{obs['id'][:8]}..] proof={obs.get('proof_count', '?')}: {obs['text']}{src_summary}")
    else:
        print("    (none)")

    resp = entry.response
    print("\n  LLM RESPONSE:")
    if resp.creates:
        for c in resp.creates:
            print(f"    CREATE: \"{c.text}\" (from facts: {[fid[:8] + '..' for fid in c.source_fact_ids]})")
    if resp.updates:
        for u in resp.updates:
            print(
                f"    UPDATE [{u.observation_id[:8]}..]: \"{u.text}\""
                f" (from facts: {[fid[:8] + '..' for fid in u.source_fact_ids]})"
            )
    if resp.deletes:
        for d in resp.deletes:
            print(f"    DELETE [{d.observation_id[:8]}..]")
    if not resp.creates and not resp.updates and not resp.deletes:
        print("    (no actions)")


def _parse_history(hist: Any) -> list[str]:
    """Parse observation history from DB (may be list of dicts or JSON strings)."""
    if not hist:
        return []
    parsed = hist if isinstance(hist, list) else json.loads(hist)
    prev_texts = []
    for h in parsed:
        if isinstance(h, str):
            h = json.loads(h)
        prev_texts.append(h.get("previous_text", "?"))
    return prev_texts


@pytest.mark.asyncio
async def test_horse_farm_observation_history(memory: MemoryEngine, request_context: Any) -> None:
    """Retain a sequence of horse facts and inspect how observations evolve."""
    bank_id = f"test-horses-{uuid.uuid4().hex[:8]}"

    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

    messages = [
        "I have a farm.",
        "I have 2 horses.",
        "I have a horse named Daisy.",
        "I have a horse named Buttercup.",
        "I sold Buttercup.",
        "I now have 1 horse.",
        "I have 5 horses on my farm.",
        "I have a horse named Midnight.",
        "I have horses named Midnight and Shadow.",
        "I have horses named Shadow and Twister.",
        "I am sad to report that Shadow has died.",
    ]

    # Monkey-patch to intercept consolidation LLM calls
    _original_consolidate = consolidator_mod._consolidate_batch_with_llm

    async def _patched(**kwargs: Any) -> Any:
        return await _instrumented_consolidate(_original_consolidate, **kwargs)

    consolidator_mod._consolidate_batch_with_llm = _patched
    _debug_log.clear()

    try:
        for i, content in enumerate(messages):
            print(f"\n{'='*80}")
            print(f"RETAIN #{i+1}: {content}")
            print(f"{'='*80}")

            log_start = len(_debug_log)

            await memory.retain_async(
                bank_id=bank_id,
                content=content,
                request_context=request_context,
            )
            await memory.wait_for_background_tasks()

            for j, entry in enumerate(_debug_log[log_start:]):
                _print_consolidation_debug(entry, j + 1)

            # Dump current observations
            pool = await memory._get_pool()
            async with pool.acquire() as conn:
                observations = await conn.fetch(
                    """
                    SELECT id, text, proof_count, source_memory_ids, history
                    FROM memory_units
                    WHERE bank_id = $1 AND fact_type = 'observation'
                    ORDER BY created_at
                    """,
                    bank_id,
                )

                print(f"\n  CURRENT OBSERVATIONS ({len(observations)}):")
                for obs in observations:
                    prev_texts = _parse_history(obs["history"])
                    hist_str = f" (was: {' -> '.join(prev_texts)})" if prev_texts else ""
                    print(f"    [{str(obs['id'])[:8]}..] proof={obs['proof_count']}: {obs['text']}{hist_str}")
    finally:
        consolidator_mod._consolidate_batch_with_llm = _original_consolidate

    # Final summary
    print(f"\n{'='*80}")
    print("FINAL STATE")
    print(f"{'='*80}")
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        observations = await conn.fetch(
            """
            SELECT id, text, proof_count, source_memory_ids, history
            FROM memory_units
            WHERE bank_id = $1 AND fact_type = 'observation'
            ORDER BY created_at
            """,
            bank_id,
        )
        print(f"\nFinal observations ({len(observations)}):")
        for obs in observations:
            prev_texts = _parse_history(obs["history"])
            if prev_texts:
                chain = prev_texts + [obs["text"]]
                print(f"  - [proof={obs['proof_count']}] {obs['text']}")
                print(f"    evolution: {' -> '.join(chain)}")
            else:
                print(f"  - [proof={obs['proof_count']}] {obs['text']}")

    # Create a mental model to synthesize the observations
    print(f"\n{'='*80}")
    print("MENTAL MODEL")
    print(f"{'='*80}")

    # Patch reflect _execute_tool to log tool inputs/outputs
    from hindsight_api.engine.reflect import agent as reflect_agent_mod

    _original_execute = reflect_agent_mod._execute_tool

    async def _logging_execute(tool_name: str, args: dict[str, Any], *a: Any, **kw: Any) -> dict[str, Any]:
        result = await _original_execute(tool_name, args, *a, **kw)
        normalized = reflect_agent_mod._normalize_tool_name(tool_name)
        print(f"\n  [REFLECT TOOL] {normalized}(args={args})")
        if isinstance(result, dict):
            if "observations" in result:
                print(f"    Observations returned ({result.get('count', '?')}, freshness={result.get('freshness', '?')}):")
                for obs in result.get("observations", []):
                    print(f"      - [proof={obs.get('proof_count', '?')}] {obs.get('text', '?')}")
            if "memories" in result:
                print(f"    Memories returned ({result.get('count', '?')}):")
                for mem in result.get("memories", []):
                    chunk = mem.get("chunk_text", "")
                    chunk_preview = f" | chunk: {chunk[:80]}..." if chunk else ""
                    print(f"      - [{mem.get('fact_type', '?')}] {mem.get('text', '?')}{chunk_preview}")
            if "mental_models" in result:
                print(f"    Mental models returned ({result.get('count', '?')}):")
                for mm_item in result.get("mental_models", []):
                    print(f"      - {mm_item.get('name', '?')}: {str(mm_item.get('content', '?'))[:120]}")
            if "error" in result:
                print(f"    ERROR: {result['error']}")
        return result

    reflect_agent_mod._execute_tool = _logging_execute

    source_query = (
        "Produce a structured summary of all animals on the farm. Include:\n"
        "1. A chronological timeline of events (acquisitions, sales, deaths) with dates\n"
        "2. The list of all known horse names and their current status (alive, sold, died)\n"
        "3. The current number of horses on the farm, accounting for all events\n"
        "Reason step by step from the facts. If a horse died or was sold, subtract from the count."
    )

    try:
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Farm Animals",
            source_query=source_query,
            content="(initial — awaiting refresh)",
            request_context=request_context,
        )

        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id,
            mental_model_id=mm["id"],
            request_context=request_context,
        )
        content = refreshed["content"]
    finally:
        reflect_agent_mod._execute_tool = _original_execute
    print(f"\nMental model content:\n{content}")

    reflect_resp = refreshed.get("reflect_response")
    if reflect_resp and isinstance(reflect_resp, str) and reflect_resp.strip():
        try:
            reflect_resp = json.loads(reflect_resp)
        except json.JSONDecodeError:
            reflect_resp = None
    if isinstance(reflect_resp, dict):
        based_on = reflect_resp.get("based_on", [])
        if based_on:
            print("\nBased on:")
            for item in based_on:
                if isinstance(item, str):
                    try:
                        item = json.loads(item)
                    except json.JSONDecodeError:
                        continue
                print(f"  - [{item.get('fact_type', '?')}] {item.get('text', '?')}")

    # Verify the mental model captures key facts
    content_lower = content.lower()

    for name in ["daisy", "buttercup", "midnight", "shadow", "twister"]:
        assert name in content_lower, f"Mental model should mention {name}. Got:\n{content}"

    assert "sold" in content_lower or "sale" in content_lower, (
        f"Mental model should mention Buttercup was sold. Got:\n{content}"
    )

    assert "died" in content_lower or "passed" in content_lower or "death" in content_lower, (
        f"Mental model should mention Shadow's death. Got:\n{content}"
    )

    # Cleanup
    await memory.delete_bank(bank_id, request_context=request_context)
