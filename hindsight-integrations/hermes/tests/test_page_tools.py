"""Page tools (#5289): which tools are offered, what each sends, and how short its output stays."""

from __future__ import annotations

import json
from types import SimpleNamespace

import hindsight_hermes as plugin
from conftest import FakeClient


def _node(id: str, name: str, kind: str = "page", children=()):
    return SimpleNamespace(id=id, name=name, kind=kind, children=list(children))


class PagesClient(FakeClient):
    """The current client's page methods."""

    def __init__(self):
        super().__init__()
        self.page_calls: list[tuple[str, dict]] = []

    async def asearch_pages(self, **kwargs):
        self.page_calls.append(("search", kwargs))
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    id="p1",
                    name="Renovation",
                    snippet="# Renovation\n## Kitchen\nKitchen  done.\nBathroom " + "x" * 300,
                ),
                SimpleNamespace(id="p2", name="Empty", snippet=""),
            ]
        )

    async def aget_page(self, **kwargs):
        self.page_calls.append(("get", kwargs))
        return SimpleNamespace(name="Renovation", body="Kitchen is done.\n")

    async def aget_page_tree(self, **kwargs):
        self.page_calls.append(("tree", kwargs))
        return SimpleNamespace(roots=[_node("f1", "Home", "folder", [_node("p1", "Renovation")]), _node("p2", "Car")])

    async def acreate_page(self, **kwargs):
        self.page_calls.append(("create", kwargs))
        return SimpleNamespace(page_id="p9", mental_model_id="m9", operation_id="op")


class LegacyPagesClient(FakeClient):
    """hindsight-client before 1.2.1: the same calls under their "knowledge" names."""

    def __init__(self):
        super().__init__()
        self.page_calls: list[str] = []

    async def asearch_knowledge_base(self, **kwargs):
        self.page_calls.append("search")
        return SimpleNamespace(results=[])

    async def aget_knowledge_page(self, **kwargs):
        self.page_calls.append("get")
        return SimpleNamespace(name="Old", body="")

    async def aget_knowledge_base_tree(self, **kwargs):
        self.page_calls.append("tree")
        return SimpleNamespace(roots=[])

    async def acreate_knowledge_page(self, **kwargs):
        self.page_calls.append("create")
        return SimpleNamespace(page_id="p1", mental_model_id="m1", operation_id=None)


def _call(instance, tool: str, **args) -> str:
    out = instance.handle_tool_call(tool, args)
    return json.loads(out)["result"] if out.startswith("{") else out  # errors are tool_error() text


def _tool_names(instance) -> list[str]:
    return [s["name"] for s in instance.get_tool_schemas()]


def test_read_tools_on_by_default_create_off(provider):
    instance, _ = provider()
    names = _tool_names(instance)
    assert {"hindsight_search_pages", "hindsight_read_page", "hindsight_list_pages"} <= set(names)
    assert "hindsight_create_page" not in names
    assert "hindsight_search_pages" in instance.system_prompt_block()


def test_page_tools_off_removes_tools_and_guidance(provider):
    instance, _ = provider({"page_tools": False})
    assert not any("page" in n for n in _tool_names(instance))
    assert "hindsight_search_pages" not in instance.system_prompt_block()


def test_context_mode_has_no_page_tools_or_guidance(provider):
    instance, _ = provider({"memory_mode": "context"})
    assert instance.get_tool_schemas() == []
    assert "hindsight_search_pages" not in instance.system_prompt_block()


def test_validation_errors_are_joined():
    exc = plugin_api_error(422, '{"detail": [{"msg": "Field required", "loc": ["q"]}, {"msg": "Too long"}]}')
    assert plugin._api_error_detail(exc) == "Field required; Too long"
    assert plugin._api_error_detail(plugin_api_error(500, "not json")) == "HTTP 500"


def test_create_tool_only_when_enabled_and_writes_allowed(provider, monkeypatch):
    instance, _ = provider({"page_create_tool": True})
    assert "hindsight_create_page" in _tool_names(instance)
    monkeypatch.setattr(instance, "_writes_disabled", True)  # a read-only kanban worker
    assert "hindsight_create_page" not in _tool_names(instance)


def test_create_refused_when_disabled_even_if_called(provider):
    client = PagesClient()
    instance, _ = provider(client=client)
    assert "disabled" in _call(instance, "hindsight_create_page", name="X", question="Q?")
    assert client.page_calls == []


def test_search_sends_query_and_clamped_limit_and_prints_one_short_line_per_page(provider):
    client = PagesClient()
    instance, _ = provider(client=client)
    out = _call(instance, "hindsight_search_pages", query="renovation", limit=99)
    assert client.page_calls == [("search", {"bank_id": instance._bank_id, "q": "renovation", "limit": 20})]
    first, second = out.split("\n")
    assert first.startswith("- Renovation (p1): Renovation Kitchen Kitchen done. Bathroom xxx") and first.endswith("…")
    assert len(first) < 240
    assert second == "- Empty (p2)"
    _call(instance, "hindsight_search_pages", query="q", limit="abc")
    assert client.page_calls[-1][1]["limit"] == 5


def test_read_list_and_create_output(provider):
    client = PagesClient()
    instance, _ = provider({"page_create_tool": True}, client=client)
    assert _call(instance, "hindsight_read_page", page_id="p1") == "Kitchen is done."
    assert _call(instance, "hindsight_list_pages") == "- Home/\n  - Renovation (p1)\n- Car (p2)"
    out = _call(instance, "hindsight_create_page", name="Car", question="What is the state of the car?")
    assert out.startswith("Created page Car (p9).")
    assert client.page_calls[-1] == (
        "create",
        {"bank_id": instance._bank_id, "name": "Car", "source_query": "What is the state of the car?"},
    )


def test_list_is_capped(provider, monkeypatch):
    client = PagesClient()

    async def big_tree(**kwargs):
        return SimpleNamespace(roots=[_node(f"p{i}", f"Page {i}") for i in range(130)])

    client.aget_page_tree = big_tree
    instance, _ = provider(client=client)
    lines = _call(instance, "hindsight_list_pages").split("\n")
    assert len(lines) == 101 and lines[-1].startswith("… 30 more")


def test_legacy_client_names_are_used(provider):
    client = LegacyPagesClient()
    instance, _ = provider({"page_create_tool": True}, client=client)
    assert _call(instance, "hindsight_search_pages", query="q") == "No pages match."
    assert _call(instance, "hindsight_read_page", page_id="p1") == "Page Old is not built yet."
    assert _call(instance, "hindsight_list_pages") == "No pages yet."
    assert _call(instance, "hindsight_create_page", name="N", question="Q?").startswith("Created page N (p1)")
    assert client.page_calls == ["search", "get", "tree", "create"]


def test_missing_args_and_server_detail(provider):
    client = PagesClient()

    async def not_found(**kwargs):
        raise plugin_api_error(404, '{"detail": "Page not found"}')

    client.aget_page = not_found
    instance, _ = provider({"page_create_tool": True}, client=client)
    assert "Missing required parameter: question" in _call(instance, "hindsight_create_page", name="X")
    assert _call(instance, "hindsight_read_page", page_id="nope") == "ERROR: Failed to read page: Page not found"


def plugin_api_error(status: int, body: str) -> Exception:
    exc = RuntimeError(f"({status})\nReason: Not Found\nHTTP response headers: <lots of headers>")
    exc.status, exc.body = status, body
    return exc


def test_read_only_children_get_the_page_reads():
    names = plugin.HindsightMemoryProvider.read_only_tool_names(None)
    assert {"hindsight_search_pages", "hindsight_read_page", "hindsight_list_pages"} <= set(names)
    assert "hindsight_create_page" not in names
