"""The list/graph rows are typed models, and typing them changed no wire payload.

Regression for #4218: ``list_documents``, ``list_memories``, ``get_graph`` and
``get_entity_graph`` declared their rows as ``dict[str, Any]``, so every generated SDK
handed callers untyped dicts while the single-fetch siblings returned real models.

The rows are now models, but they stay **open**: each declares ``extra="allow"`` so a key
the server emits and the model does not declare is still carried through to the client
verbatim. These tests pin exactly that — the rows the engine builds today survive
validation with every key intact, including keys no field describes.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hindsight_api.api.http import (
    ExcludeNoneRoute,
    DocumentListItem,
    EntityGraphResponse,
    GraphDataResponse,
    ListDocumentsResponse,
    ListMemoryUnitsResponse,
    MemoryUnitListItem,
)

# The exact row `memory_engine.list_documents` builds (SQL branch).
_DOCUMENT_ROW = {
    "id": "session_1",
    "bank_id": "user123",
    "content_hash": "abc123",
    "created_at": "2024-01-15T10:30:00Z",
    "updated_at": "2024-01-16T09:00:00Z",
    "text_length": 5420,
    "memory_unit_count": 15,
    "retain_params": {"context": "Team meeting notes"},
    "document_metadata": {"source": "slack"},
    "tags": ["user_a", "session_123"],
}

# The exact row `memories/pg/curation.list_memory_units` builds.
_MEMORY_ROW = {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "text": "Alice works at Google",
    "context": "Work conversation",
    "date": "2024-01-15T10:30:00Z",
    "fact_type": "world",
    "document_id": "session_1",
    "mentioned_at": "2024-01-15T10:30:00Z",
    "occurred_start": None,
    "occurred_end": None,
    "entities": "Alice, Google",
    "chunk_id": None,
    "proof_count": 2,
    "tags": ["user:alice"],
    "metadata": {"source": "slack"},
    "consolidated_at": None,
    "consolidation_failed_at": None,
    "state": "valid",
    "invalidation_reason": None,
    "invalidated_at": None,
    "edited_at": None,
    "updated_at": "2024-01-15T10:30:00Z",
    "source_memory_ids": [],
}

# The exact payload `memory_engine.get_graph_data` builds.
_GRAPH_DATA = {
    "nodes": [
        {
            "data": {
                "id": "u1",
                "label": "Alice works at Goo...",
                "text": "Alice works at Google",
                "date": "2024-01-15T10:30:00Z",
                "context": "Work conversation",
                "entities": "Alice, Google",
                "color": "#42a5f5",
            }
        }
    ],
    "edges": [
        {
            "data": {
                "id": "u1-u2-entity",
                "source": "u1",
                "target": "u2",
                "linkType": "entity",
                "weight": 1.0,
                "entityName": "Alice",
                "color": "#ffd700",
                "lineStyle": "solid",
            }
        }
    ],
    "table_rows": [
        {
            "id": "u1",
            "text": "Alice works at Google",
            "context": "Work conversation",
            "occurred_start": None,
            "occurred_end": None,
            "mentioned_at": "2024-01-15T10:30:00Z",
            "date": "2024-01-15 10:30",
            "entities": "Alice, Google",
            "document_id": "session_1",
            "chunk_id": None,
            "fact_type": "world",
            "tags": ["user:alice"],
            "created_at": "2024-01-15T10:30:00Z",
            "proof_count": 2,
        }
    ],
    "total_units": 1,
    "limit": 1000,
}

# The exact payload `memory_engine.get_entity_graph` builds.
_ENTITY_GRAPH = {
    "nodes": [{"data": {"id": "e1", "label": "Alice", "mentionCount": 12, "color": "#42a5f5"}}],
    "edges": [
        {
            "data": {
                "id": "e1-e2",
                "source": "e1",
                "target": "e2",
                "linkType": "cooccurrence",
                "weight": 5,
                "color": "#ffd700",
                "lineStyle": "solid",
                "lastCooccurred": "2024-02-01T14:00:00Z",
            }
        }
    ],
    "total_entities": 2,
    "total_edges": 1,
    "limit": 1000,
}


def _dump(model):
    return model.model_dump(mode="json")


class TestRowsAreTypedWithoutChangingTheWire:
    """Each engine payload validates, and serializes back byte-for-key identical."""

    def test_document_rows_round_trip_unchanged(self):
        payload = {"items": [_DOCUMENT_ROW], "total": 1, "limit": 100, "offset": 0}
        assert _dump(ListDocumentsResponse.model_validate(payload)) == payload

    def test_memory_rows_round_trip_unchanged(self):
        payload = {"items": [_MEMORY_ROW], "total": 1, "limit": 100, "offset": 0}
        assert _dump(ListMemoryUnitsResponse.model_validate(payload)) == payload

    def test_graph_payload_round_trips_unchanged(self):
        assert _dump(GraphDataResponse.model_validate(_GRAPH_DATA)) == _GRAPH_DATA

    def test_entity_graph_payload_round_trips_unchanged(self):
        assert _dump(EntityGraphResponse.model_validate(_ENTITY_GRAPH)) == _ENTITY_GRAPH

    def test_rows_are_attribute_addressable(self):
        # The point of the issue: `items[0].id` instead of `items[0]["id"]`.
        listing = ListDocumentsResponse.model_validate(
            {"items": [_DOCUMENT_ROW], "total": 1, "limit": 100, "offset": 0}
        )
        assert listing.items[0].id == "session_1"
        assert listing.items[0].memory_unit_count == 15
        graph = GraphDataResponse.model_validate(_GRAPH_DATA)
        assert graph.nodes[0].data.label.startswith("Alice works at")
        assert graph.edges[0].data.linkType == "entity"
        assert graph.table_rows[0].fact_type == "world"


class TestRowsStayOpen:
    """A key the model does not declare must still reach the client."""

    def test_unknown_document_key_is_preserved(self):
        row = {**_DOCUMENT_ROW, "some_future_field": {"nested": 1}}
        payload = {"items": [row], "total": 1, "limit": 100, "offset": 0}
        assert _dump(ListDocumentsResponse.model_validate(payload))["items"][0] == row

    def test_unknown_memory_key_is_preserved(self):
        row = {**_MEMORY_ROW, "some_future_field": "kept"}
        payload = {"items": [row], "total": 1, "limit": 100, "offset": 0}
        assert _dump(ListMemoryUnitsResponse.model_validate(payload))["items"][0] == row

    def test_unknown_graph_node_key_is_preserved(self):
        graph = {
            **_GRAPH_DATA,
            "nodes": [{"data": {**_GRAPH_DATA["nodes"][0]["data"], "shape": "hexagon"}}],
        }
        assert _dump(GraphDataResponse.model_validate(graph)) == graph

    def test_unknown_entity_graph_edge_key_is_preserved(self):
        graph = {
            **_ENTITY_GRAPH,
            "edges": [{"data": {**_ENTITY_GRAPH["edges"][0]["data"], "curvature": 0.3}}],
        }
        assert _dump(EntityGraphResponse.model_validate(graph)) == graph


class TestSparseRowsStillValidate:
    """A store-backed row that omits optional keys must not 500 the endpoint.

    Documents and memories can come from a store that owns its own registry, whose rows
    need not carry every column the SQL branch selects. Only the identity is required.
    """

    def test_document_row_needs_only_an_id(self):
        item = DocumentListItem.model_validate({"id": "d1"})
        assert item.tags == []
        assert item.memory_unit_count == 0
        assert item.retain_params is None

    def test_memory_row_needs_only_an_id(self):
        item = MemoryUnitListItem.model_validate({"id": "u1"})
        assert item.state == "valid"
        assert item.proof_count == 1
        assert item.source_memory_ids == []


class TestNullsStayOnTheWire:
    """``exclude_none`` must not reach these rows.

    ``ExcludeNoneRoute`` turns on ``response_model_exclude_none`` for most routes, but it
    never reached inside a ``dict[str, Any]`` row, so these endpoints always emitted the
    row's nulls. Typing the rows would silently start dropping those keys — a ``KeyError``
    for anyone indexing a row — unless the route keeps them.
    """

    def test_a_route_serves_the_row_nulls(self):
        # End-to-end through the route class the app actually installs.
        app = FastAPI()
        app.router.route_class = ExcludeNoneRoute

        @app.get("/rows", response_model=ListMemoryUnitsResponse)
        async def rows():
            return {"items": [_MEMORY_ROW], "total": 1, "limit": 100, "offset": 0}

        body = TestClient(app).get("/rows").json()
        assert body["items"][0] == _MEMORY_ROW


class TestSchemaStaysGeneratorSafe:
    """The published schema must not carry ``additionalProperties`` for these rows.

    openapi-generator 7.10.0's Python generator crashes on a schema that pairs
    ``additionalProperties`` with a nullable ``anyOf`` property, and every row here has one.
    The runtime passthrough (``TestRowsStayOpen``) is what matters; the spec keyword is not.
    """

    def test_row_schemas_declare_no_additional_properties(self):
        for model in (DocumentListItem, MemoryUnitListItem):
            assert "additionalProperties" not in model.model_json_schema(), model.__name__

    def test_nested_row_schemas_declare_no_additional_properties(self):
        defs = GraphDataResponse.model_json_schema()["$defs"]
        for name, schema in defs.items():
            assert "additionalProperties" not in schema, name
