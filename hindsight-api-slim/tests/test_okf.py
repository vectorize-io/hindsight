"""Pure unit tests for the OKF (Open Knowledge Format) serializer.

These exercise hindsight_api/api/okf.py with plain dicts — no DB, no LLM — so
they pin the OKF contract (frontmatter projection, type-from-tag, shared-tag
graph) deterministically and fast.
"""

from hindsight_api.api import okf


def _mm(**overrides):
    base = {
        "id": "orders",
        "name": "Orders",
        "source_query": "What are the order facts?",
        "content": "# Orders\n\nOne row per order.",
        "tags": ["type:runbook", "sales", "revenue"],
        "last_refreshed_at": "2026-01-02T00:00:00Z",
        "created_at": "2026-01-01T00:00:00Z",
    }
    base.update(overrides)
    return base


class TestPageType:
    def test_lifts_type_from_tag_and_drops_it(self):
        pt = okf.page_type(["type:runbook", "sales", "revenue"])
        assert pt.type == "runbook"
        assert pt.display_tags == ["sales", "revenue"]

    def test_defaults_when_no_type_tag(self):
        pt = okf.page_type(["sales"])
        assert pt.type == okf.DEFAULT_PAGE_TYPE
        assert pt.display_tags == ["sales"]

    def test_handles_none_and_empty(self):
        assert okf.page_type(None).type == okf.DEFAULT_PAGE_TYPE
        assert okf.page_type(None).display_tags == []

    def test_blank_type_suffix_falls_back(self):
        pt = okf.page_type(["type:", "sales"])
        assert pt.type == okf.DEFAULT_PAGE_TYPE
        # the (blank) type tag is still stripped from display tags
        assert pt.display_tags == ["sales"]

    def test_first_type_tag_wins(self):
        pt = okf.page_type(["type:runbook", "type:guide"])
        assert pt.type == "runbook"
        assert pt.display_tags == []


class TestFrontmatter:
    def test_projects_expected_fields(self):
        fm = okf.frontmatter(_mm())
        assert fm["id"] == "orders"
        assert fm["type"] == "runbook"
        assert fm["title"] == "Orders"
        assert fm["description"] == "What are the order facts?"
        assert fm["tags"] == ["sales", "revenue"]
        assert fm["timestamp"] == "2026-01-02T00:00:00Z"

    def test_timestamp_falls_back_to_created_at(self):
        fm = okf.frontmatter(_mm(last_refreshed_at=None))
        assert fm["timestamp"] == "2026-01-01T00:00:00Z"

    def test_render_omits_none_and_empty(self):
        rendered = okf.render_frontmatter({"type": "x", "title": None, "tags": []})
        assert "title" not in rendered
        assert "tags" not in rendered
        assert 'type: "x"' in rendered

    def test_render_quotes_and_escapes(self):
        # A name that looks like a YAML bool / contains a quote must stay a string.
        rendered = okf.render_frontmatter({"title": 'true "x"'})
        assert 'title: "true \\"x\\""' in rendered


class TestRenderDocument:
    def test_includes_frontmatter_and_body(self):
        doc = okf.render_document(_mm())
        assert doc.startswith("---\n")
        assert 'type: "runbook"' in doc
        assert "One row per order." in doc

    def test_empty_body(self):
        doc = okf.render_document(_mm(content=""))
        assert doc.count("---") == 2
        assert doc.rstrip().endswith("---")


class TestKnowledgeGraph:
    def test_edge_from_shared_tag(self):
        pages = [
            _mm(id="orders", tags=["type:runbook", "sales", "revenue"]),
            _mm(id="customers", tags=["sales", "crm"]),
            _mm(id="lonely", tags=[]),
        ]
        graph = okf.knowledge_graph(pages)
        assert len(graph.nodes) == 3
        assert len(graph.edges) == 1
        edge = graph.edges[0]["data"]
        assert {edge["source"], edge["target"]} == {"orders", "customers"}
        assert edge["sharedTags"] == ["sales"]
        assert edge["weight"] == 1

    def test_type_tag_does_not_create_edges(self):
        # Two pages sharing only a type: tag must NOT be linked.
        pages = [
            _mm(id="a", tags=["type:runbook"]),
            _mm(id="b", tags=["type:runbook"]),
        ]
        graph = okf.knowledge_graph(pages)
        assert graph.edges == []

    def test_node_carries_type_and_color(self):
        graph = okf.knowledge_graph([_mm(id="orders", tags=["type:runbook", "sales"])])
        node = graph.nodes[0]["data"]
        assert node["type"] == "runbook"
        assert node["label"] == "Orders"
        assert node["tagCount"] == 1
        assert node["color"].startswith("#")

    def test_weight_counts_shared_tags(self):
        pages = [
            _mm(id="a", tags=["sales", "revenue", "x"]),
            _mm(id="b", tags=["sales", "revenue", "y"]),
        ]
        graph = okf.knowledge_graph(pages)
        assert graph.edges[0]["data"]["weight"] == 2
        assert graph.edges[0]["data"]["sharedTags"] == ["revenue", "sales"]


class TestReservedFiles:
    def test_index_links_each_page(self):
        index = okf.render_index([_mm(id="orders", name="Orders", source_query="q?")])
        assert "[Orders](./orders.md)" in index
        assert "q?" in index
        assert 'type: "index"' in index

    def test_index_empty(self):
        assert "No knowledge pages yet" in okf.render_index([])

    def test_index_nests_folders(self):
        nodes = [
            {"id": "f1", "kind": "folder", "name": "Runbooks", "parent_id": None},
            {"id": "p1", "kind": "page", "name": "Orders", "parent_id": "f1", "source_query": "q?"},
            {"id": "p2", "kind": "page", "name": "Loose", "parent_id": None},
        ]
        idx = okf.render_index(nodes)
        assert "**Runbooks/**" in idx
        # the page nested in the folder is indented and links to its file
        assert "  - [Orders](./p1.md) — q?" in idx
        assert "- [Loose](./p2.md)" in idx

    def test_log_renders_history_newest_first(self):
        history = [
            {"previous_content": "v2", "changed_at": "2026-01-02T00:00:00Z"},
            {"previous_content": "v1", "changed_at": "2026-01-01T00:00:00Z"},
        ]
        log = okf.render_log(_mm(), history)
        assert 'type: "log"' in log
        assert log.index("2026-01-02") < log.index("2026-01-01")
        assert "v2" in log and "v1" in log

    def test_log_empty(self):
        assert "No refresh history" in okf.render_log(_mm(), [])
