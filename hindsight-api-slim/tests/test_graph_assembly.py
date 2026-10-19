"""The graph view's edge assembly stays bounded on a densely-connected bank (#5703)."""

import uuid
from datetime import UTC, datetime

from hindsight_api.api.http import GraphDataResponse
from hindsight_api.engine.memories.base import GRAPH_MAX_EDGES
from hindsight_api.engine.memory_engine import _assemble_graph_data


def _unit(fact_type: str, source_memory_ids: list[uuid.UUID] | None = None) -> dict:
    now = datetime.now(UTC)
    return {
        "id": uuid.uuid4(),
        "text": "an observation",
        "event_date": now,
        "context": None,
        "occurred_start": None,
        "occurred_end": None,
        "mentioned_at": now,
        "document_id": None,
        "chunk_id": None,
        "fact_type": fact_type,
        "tags": [],
        "created_at": now,
        "proof_count": None,
        "source_memory_ids": source_memory_ids,
    }


def test_derived_edges_stop_at_the_cap_and_stored_links_come_first() -> None:
    # 3000 observations over shared sources that all mention the same 20 entities: uncapped,
    # this derives ~600k entity edges plus every pair of observations sharing a source.
    sources = [uuid.uuid4() for _ in range(300)]
    observations = [_unit("observation", [sources[i % 300], sources[(i * 7) % 300]]) for i in range(3000)]
    entity_rows = [
        {"unit_id": s, "entity_id": uuid.uuid4(), "canonical_name": f"entity {e}"} for s in sources for e in range(20)
    ]
    a, b = observations[0]["id"], observations[1]["id"]
    stored = {"from_unit_id": a, "to_unit_id": b, "link_type": "temporal", "weight": 0.5, "entity_name": None}

    data = _assemble_graph_data(observations, [stored], entity_rows, total_count=3000, limit=3000)

    # The stored link, then a full budget of derived edges
    assert len(data["edges"]) == 1 + GRAPH_MAX_EDGES
    assert data["edges"][0]["data"]["linkType"] == "temporal"
    assert len(data["nodes"]) == len(data["table_rows"]) == 3000
    # The route encodes this dict as is, so it must already be the response model's shape
    GraphDataResponse.model_validate(data)


def test_small_graph_keeps_every_edge() -> None:
    source = uuid.uuid4()
    obs_a, obs_b = _unit("observation", [source]), _unit("observation", [source])
    entity_rows = [{"unit_id": source, "entity_id": uuid.uuid4(), "canonical_name": "Alice"}]

    data = _assemble_graph_data([obs_a, obs_b], [], entity_rows, total_count=2, limit=10)

    edge_types = sorted(e["data"]["linkType"] for e in data["edges"])
    assert edge_types == ["entity", "semantic"]
    assert all(n["data"]["entities"] == "Alice" for n in data["nodes"])


def test_stored_links_filling_their_budget_leave_room_for_entity_edges() -> None:
    facts = [_unit("world") for _ in range(200)]
    stored = [
        {"from_unit_id": a["id"], "to_unit_id": b["id"], "link_type": "semantic", "weight": 0.9, "entity_name": None}
        for a in facts
        for b in facts
        if a is not b
    ][:GRAPH_MAX_EDGES]
    entity_rows = [{"unit_id": f["id"], "entity_id": uuid.uuid4(), "canonical_name": "Acme"} for f in facts]

    data = _assemble_graph_data(facts, stored, entity_rows, total_count=200, limit=200)

    link_types = [e["data"]["linkType"] for e in data["edges"]]
    assert link_types.count("semantic") == GRAPH_MAX_EDGES
    assert link_types.count("entity") > 0
