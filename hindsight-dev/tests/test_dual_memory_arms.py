"""Unit tests for the dual-memory arm dispatch (FederationArm fallback policy).

The arms module is otherwise exercised end-to-end by the orchestrator; the
unit-test surface is the policy decision the four-arm comparison hinges on:
"does the federation arm consult the world graph only when private recall is
empty?". The other arms are simple wrappers whose behaviour is covered by
the orchestrator smoke test.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchmarks.dual_memory.arms import FederationArm
from benchmarks.dual_memory.taskset import SessionSplit, Task


def _split() -> SessionSplit:
    return SessionSplit(
        conv_id="conv-1",
        agent_a="Caroline",
        agent_b="Melanie",
        a_sessions=frozenset({1, 3, 5}),
        b_sessions=frozenset({2, 4}),
    )


def _task() -> Task:
    return Task(
        conv_id="conv-1",
        question="What did Melanie do?",
        gold_answer="went running",
        category="blind",
        asker="b",
        evidence_sessions=(2,),
        locomo_category=1,
    )


def _private_payload(*hits: dict) -> str:
    return json.dumps({"private_memory": list(hits)})


def _federation_arm() -> tuple[FederationArm, MagicMock, MagicMock]:
    """Build a FederationArm wired to a MagicMock HindsightArm + MemoryEngine."""
    hindsight = MagicMock()
    hindsight.bank_id.return_value = "dm-run-conv-1-b"
    hindsight.retrieve = AsyncMock(return_value=_private_payload({"text": "a fact"}))
    hindsight.ingest = AsyncMock()
    memory = MagicMock()
    return FederationArm(hindsight, memory, run_id="run"), hindsight, memory


@pytest.mark.asyncio
async def test_federation_ingest_delegates_to_hindsight_arm():
    arm, hindsight, _memory = _federation_arm()
    item = {"sample_id": "conv-1"}
    split = _split()
    await arm.ingest(item, split)
    hindsight.ingest.assert_awaited_once_with(item, split)


@pytest.mark.asyncio
async def test_federation_returns_private_only_when_private_has_hits():
    """No world-graph call when private recall has any results — the
    federation arm's whole point is to be a no-op on private tasks."""
    arm, hindsight, _memory = _federation_arm()
    hindsight.retrieve = AsyncMock(return_value=_private_payload({"text": "a fact"}, {"text": "another fact"}))
    with patch(
        "hindsight_api.engine.reflect.tools.tool_search_world_graph",
        new=AsyncMock(),
    ) as tool:
        payload = json.loads(await arm.retrieve(_task()))
    tool.assert_not_awaited()
    assert "world_graph" not in payload
    assert len(payload["private_memory"]) == 2


@pytest.mark.asyncio
async def test_federation_falls_through_to_world_graph_on_private_empty():
    """Empty private recall → call the C3 tool with the conversation's group id."""
    arm, hindsight, memory = _federation_arm()
    hindsight.retrieve = AsyncMock(return_value=_private_payload())
    tool_result = {"facts": [{"fact": "world fact 1", "valid_at": "2023-05-01", "invalid_at": None}]}
    with patch(
        "hindsight_api.engine.reflect.tools.tool_search_world_graph",
        new=AsyncMock(return_value=tool_result),
    ) as tool:
        payload = json.loads(await arm.retrieve(_task()))
    tool.assert_awaited_once()
    kwargs = tool.await_args.kwargs
    assert kwargs["group_id"] == "dm-run-conv-1"
    assert kwargs["bank_id"] == "dm-run-conv-1-b"
    assert kwargs["query"] == "What did Melanie do?"
    assert payload["world_graph"] == tool_result["facts"]


@pytest.mark.asyncio
async def test_federation_handles_tool_error_with_empty_world_graph():
    """Tool unavailable / circuit-open / no group_id must not poison the
    answer prompt — the federation arm returns the private envelope with
    an empty ``world_graph`` so downstream code paths stay byte-identical."""
    arm, hindsight, _memory = _federation_arm()
    hindsight.retrieve = AsyncMock(return_value=_private_payload())
    with patch(
        "hindsight_api.engine.reflect.tools.tool_search_world_graph",
        new=AsyncMock(return_value={"error": "world graph unavailable: HINDSIGHT_API_GRAPHITI_BASE_URL not set"}),
    ):
        payload = json.loads(await arm.retrieve(_task()))
    assert payload["world_graph"] == []
    assert payload["private_memory"] == []


@pytest.mark.asyncio
async def test_federation_group_id_matches_graphiti_arm_convention():
    """The federation arm and the standalone graphiti arm must read from
    the same shared graph. Verify the convention by direct call."""
    arm, _hindsight, _memory = _federation_arm()
    assert arm.group_id("conv-7") == "dm-run-conv-7"
    # Same format the GraphitiArm class uses (hindsight-dev/benchmarks/
    # dual_memory/arms.py GraphitiArm.group_id). If the convention
    # drifts the federation arm will not see the same world graph
    # the graphiti arm populated, silently breaking the 4-arm
    # comparison.
    from benchmarks.dual_memory.arms import GraphitiArm

    assert arm.group_id("conv-7") == GraphitiArm(run_id="run").group_id("conv-7")
