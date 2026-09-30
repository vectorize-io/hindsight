"""Pure unit tests for the markdown (Open Knowledge Format) serializer.

These exercise hindsight_api/api/page_markdown.py with plain dicts — no DB, no LLM — so
they pin the markdown contract (frontmatter projection, type-from-tag, shared-tag
graph) deterministically and fast.
"""

from hindsight_api.api import page_markdown


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
        pt = page_markdown.page_type(["type:runbook", "sales", "revenue"])
        assert pt.type == "runbook"
        assert pt.display_tags == ["sales", "revenue"]

    def test_defaults_when_no_type_tag(self):
        pt = page_markdown.page_type(["sales"])
        assert pt.type == page_markdown.DEFAULT_PAGE_TYPE
        assert pt.display_tags == ["sales"]

    def test_handles_none_and_empty(self):
        assert page_markdown.page_type(None).type == page_markdown.DEFAULT_PAGE_TYPE
        assert page_markdown.page_type(None).display_tags == []

    def test_blank_type_suffix_falls_back(self):
        pt = page_markdown.page_type(["type:", "sales"])
        assert pt.type == page_markdown.DEFAULT_PAGE_TYPE
        # the (blank) type tag is still stripped from display tags
        assert pt.display_tags == ["sales"]

    def test_first_type_tag_wins(self):
        pt = page_markdown.page_type(["type:runbook", "type:guide"])
        assert pt.type == "runbook"
        assert pt.display_tags == []


class TestFrontmatter:
    def test_projects_expected_fields(self):
        fm = page_markdown.frontmatter(_mm())
        assert fm["id"] == "orders"
        assert fm["type"] == "runbook"
        assert fm["title"] == "Orders"
        assert fm["description"] == "What are the order facts?"
        assert fm["tags"] == ["sales", "revenue"]
        assert fm["timestamp"] == "2026-01-02T00:00:00Z"

    def test_timestamp_falls_back_to_created_at(self):
        fm = page_markdown.frontmatter(_mm(last_refreshed_at=None))
        assert fm["timestamp"] == "2026-01-01T00:00:00Z"

    def test_render_omits_none_and_empty(self):
        rendered = page_markdown.render_frontmatter({"type": "x", "title": None, "tags": []})
        assert "title" not in rendered
        assert "tags" not in rendered
        assert 'type: "x"' in rendered

    def test_render_quotes_and_escapes(self):
        # A name that looks like a YAML bool / contains a quote must stay a string.
        rendered = page_markdown.render_frontmatter({"title": 'true "x"'})
        assert 'title: "true \\"x\\""' in rendered


class TestRenderDocument:
    def test_includes_frontmatter_and_body(self):
        doc = page_markdown.render_document(_mm())
        assert doc.startswith("---\n")
        assert 'type: "runbook"' in doc
        assert "One row per order." in doc

    def test_empty_body(self):
        doc = page_markdown.render_document(_mm(content=""))
        assert doc.count("---") == 2
        assert doc.rstrip().endswith("---")

    def test_empty_body_can_say_so_for_a_reader(self):
        """The read surfaces opt in; the export bundle does not.

        Frontmatter and nothing else reads as a page that failed to render rather
        than one nobody has written, which is what sends an agent off to create a
        second page for the topic. It stays off by default because this same
        renderer writes the export bundle, where the notice would come back as
        ordinary content the next time the bundle is read.
        """
        doc = page_markdown.render_document(_mm(content=""), notice_when_empty=True)
        assert page_markdown.EMPTY_PAGE_NOTICE in doc

        exported = page_markdown.render_document(_mm(content=""))
        assert page_markdown.EMPTY_PAGE_NOTICE not in exported

    def test_a_real_body_is_never_replaced_by_the_notice(self):
        doc = page_markdown.render_document(_mm(), notice_when_empty=True)
        assert "One row per order." in doc
        assert page_markdown.EMPTY_PAGE_NOTICE not in doc


class TestReservedFiles:
    def test_index_links_each_page(self):
        index = page_markdown.render_index([_mm(id="orders", name="Orders", source_query="q?")])
        assert "[Orders](./orders.md)" in index
        assert "q?" in index
        assert 'type: "index"' in index

    def test_index_empty(self):
        assert "No knowledge pages yet" in page_markdown.render_index([])

    def test_index_nests_folders(self):
        nodes = [
            {"id": "f1", "kind": "folder", "name": "Runbooks", "parent_id": None},
            {"id": "p1", "kind": "page", "name": "Orders", "parent_id": "f1", "source_query": "q?"},
            {"id": "p2", "kind": "page", "name": "Loose", "parent_id": None},
        ]
        idx = page_markdown.render_index(nodes)
        assert "**Runbooks/**" in idx
        # the page nested in the folder is indented and links to its file
        assert "  - [Orders](./p1.md) — q?" in idx
        assert "- [Loose](./p2.md)" in idx

    def test_log_renders_history_newest_first(self):
        history = [
            {"previous_content": "v2", "changed_at": "2026-01-02T00:00:00Z"},
            {"previous_content": "v1", "changed_at": "2026-01-01T00:00:00Z"},
        ]
        log = page_markdown.render_log(_mm(), history)
        assert 'type: "log"' in log
        assert log.index("2026-01-02") < log.index("2026-01-01")
        assert "v2" in log and "v1" in log

    def test_log_empty(self):
        assert "No refresh history" in page_markdown.render_log(_mm(), [])
