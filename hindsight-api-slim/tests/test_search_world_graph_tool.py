"""Tests for the C3 ``search_world_graph`` reflect tool.

The tool sits between the reflect agent loop and the Graphiti world-graph
service. It does three things on top of a raw HTTP call:

* Validates inputs (group_id, base_url, query) and degrades failures to a
  graceful ``{"error": ...}`` shape that the agent loop treats as "the world
  graph said nothing" rather than a hard failure.
* Caps the requested fact count by a derived token budget so a large
  ``max_facts`` can't blow past the LLM context.
* Renders each fact as a one-line ledger-annotated string and stops adding
  lines once the running token count would exceed ``max_tokens``.

Tests below cover the pure rendering, the budget calculation, the HTTP error
paths, and the gating logic in memory_engine.py that decides whether the
tool is registered at all.
"""

from __future__ import annotations

from uuid import UUID

import httpx
import pytest

from hindsight_api.config import clear_config_cache
from hindsight_api.engine.federation.graphiti_client import (
    FactResult,
    GraphitiClient,
    GraphitiClientError,
)
from hindsight_api.engine.reflect.tools import tool_search_world_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeMemoryEngine:
    """Minimal stand-in. tool_search_world_graph does not call into the engine
    itself, but the signature requires one. The body of the test never
    exercises any engine method.
    """


def _mocked_client(handler, *, base_url: str = "http://graphiti.test") -> GraphitiClient:
    """Build a GraphitiClient whose transport is the supplied handler."""
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url=base_url)
    return GraphitiClient(base_url=base_url, api_key=None, client=http)


# ---------------------------------------------------------------------------
# Pure-input validation (no HTTP at all)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_returns_error_when_group_id_empty(monkeypatch):
    """Empty group_id must short-circuit before any HTTP call."""
    monkeypatch.setenv("HINDSIGHT_API_GRAPHITI_BASE_URL", "http://graphiti.test")
    clear_config_cache()
    try:
        result = await tool_search_world_graph(
            memory_engine=_FakeMemoryEngine(),  # type: ignore[arg-type]
            bank_id="bank-A",
            group_id="",
            query="what does Alice do?",
        )
        assert "error" in result
        assert "graphiti_group_id" in result["error"]
    finally:
        clear_config_cache()


@pytest.mark.asyncio
async def test_tool_returns_error_when_query_empty():
    """An empty query is a programmer error, not a graph error."""
    result = await tool_search_world_graph(
        memory_engine=_FakeMemoryEngine(),  # type: ignore[arg-type]
        bank_id="bank-A",
        group_id="agent-shared-001",
        query="",
    )
    assert "error" in result
    assert "query" in result["error"]


@pytest.mark.asyncio
async def test_tool_returns_error_when_base_url_unset(monkeypatch):
    """Defense in depth: even if the caller passed a group_id, a missing
    HINDSIGHT_API_GRAPHITI_BASE_URL must degrade gracefully. This guards
    against the memory_engine gate being bypassed in tests.
    """
    monkeypatch.delenv("HINDSIGHT_API_GRAPHITI_BASE_URL", raising=False)
    clear_config_cache()
    try:
        result = await tool_search_world_graph(
            memory_engine=_FakeMemoryEngine(),  # type: ignore[arg-type]
            bank_id="bank-A",
            group_id="agent-shared-001",
            query="what does Alice do?",
        )
        assert "error" in result
        assert "BASE_URL" in result["error"] or "unavailable" in result["error"]
    finally:
        clear_config_cache()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_returns_facts_and_renders(monkeypatch):
    """The happy path: Graphiti returns two facts, both come back in the
    response, both appear in the rendered block.
    """
    monkeypatch.setenv("HINDSIGHT_API_GRAPHITI_BASE_URL", "http://graphiti.test")
    clear_config_cache()
    try:
        from hindsight_api.engine.federation.graphiti_client import GraphitiClient as _GC

        async def fake_search(self, group_ids, query, max_facts):
            assert group_ids == ["agent-shared-001"]
            assert query == "what does Alice do?"
            return [
                FactResult(
                    uuid=UUID("55555555-5555-5555-5555-555555555555"),
                    name="WORKS_AT",
                    fact="Alice works at Acme",
                    valid_at="2024-01-01T00:00:00Z",
                    invalid_at=None,
                    created_at=None,
                    expired_at=None,
                ),
                FactResult(
                    uuid=UUID("66666666-6666-6666-6666-666666666666"),
                    name="LIVES_IN",
                    fact="Alice lives in Paris",
                    valid_at="2024-01-01T00:00:00Z",
                    invalid_at=None,
                    created_at=None,
                    expired_at=None,
                ),
            ]

        monkeypatch.setattr(_GC, "search", fake_search)

        result = await tool_search_world_graph(
            memory_engine=_FakeMemoryEngine(),  # type: ignore[arg-type]
            bank_id="bank-A",
            group_id="agent-shared-001",
            query="what does Alice do?",
            max_facts=10,
            max_tokens=1024,
        )
        assert "error" not in result
        assert len(result["facts"]) == 2
        assert "55555555-5555-5555-5555-555555555555" in result["rendered"]
        assert "66666666-6666-6666-6666-666666666666" in result["rendered"]
        assert result["truncated"] is False
    finally:
        clear_config_cache()


@pytest.mark.asyncio
async def test_tool_renders_superseded_annotaion(monkeypatch):
    """A fact with ``invalid_at`` set must surface ``superseded <date>`` in
    the rendered line. This is the C3 disposition hook: high-skepticism
    banks must see the ledger timeline in the text the LLM reads.
    """
    monkeypatch.setenv("HINDSIGHT_API_GRAPHITI_BASE_URL", "http://graphiti.test")
    clear_config_cache()
    try:
        from hindsight_api.engine.federation.graphiti_client import GraphitiClient as _GC

        seen: dict[str, Any] = {}

        async def fake_search(self, group_ids, query, max_facts):
            seen["group_ids"] = group_ids
            seen["query"] = query
            seen["max_facts"] = max_facts
            return [
                FactResult(
                    uuid=UUID("11111111-1111-1111-1111-111111111111"),
                    name="WORKS_AT",
                    fact="Alice works at Acme",
                    valid_at="2024-01-01T00:00:00Z",
                    invalid_at="2025-06-01T00:00:00Z",
                    created_at=None,
                    expired_at=None,
                ),
            ]

        # Patch the bound method so any GraphitiClient we build goes through it
        monkeypatch.setattr(_GC, "search", fake_search)

        result = await tool_search_world_graph(
            memory_engine=_FakeMemoryEngine(),  # type: ignore[arg-type]
            bank_id="bank-A",
            group_id="agent-shared-001",
            query="what does Alice do?",
        )
        assert "error" not in result
        assert "11111111-1111-1111-1111-111111111111" in result["rendered"]
        assert "WORKS_AT" in result["rendered"]
        assert "valid since 2024-01-01" in result["rendered"]
        assert "superseded 2025-06-01" in result["rendered"]
    finally:
        clear_config_cache()


@pytest.mark.asyncio
async def test_tool_renders_only_valid_part_when_no_supersede(monkeypatch):
    """An active fact (no ``invalid_at``) must NOT carry a superseded
    annotation — only the valid-since part. (The disposition hint comes
    from the *absence* of supersede, not from spelling it out.)
    """
    monkeypatch.setenv("HINDSIGHT_API_GRAPHITI_BASE_URL", "http://graphiti.test")
    clear_config_cache()
    try:
        from hindsight_api.engine.federation.graphiti_client import GraphitiClient as _GC

        async def fake_search(self, group_ids, query, max_facts):
            return [
                FactResult(
                    uuid=UUID("22222222-2222-2222-2222-222222222222"),
                    name="WORKS_AT",
                    fact="Alice works at Acme",
                    valid_at="2024-01-01T00:00:00Z",
                    invalid_at=None,
                    created_at=None,
                    expired_at=None,
                ),
            ]

        monkeypatch.setattr(_GC, "search", fake_search)

        result = await tool_search_world_graph(
            memory_engine=_FakeMemoryEngine(),  # type: ignore[arg-type]
            bank_id="bank-A",
            group_id="agent-shared-001",
            query="what does Alice do?",
        )
        assert "error" not in result
        assert "valid since 2024-01-01" in result["rendered"]
        assert "superseded" not in result["rendered"]
    finally:
        clear_config_cache()


# ---------------------------------------------------------------------------
# Budget / truncation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_caps_fact_count_by_token_budget(monkeypatch):
    """A caller asking for max_facts=1000 with max_tokens=100 must still
    cap the request to ~2 facts (100/50). This is what protects the
    reflect context from a runaway Graphiti response.
    """
    monkeypatch.setenv("HINDSIGHT_API_GRAPHITI_BASE_URL", "http://graphiti.test")
    clear_config_cache()
    try:
        from hindsight_api.engine.federation.graphiti_client import GraphitiClient as _GC

        seen_max_facts: list[int] = []

        async def fake_search(self, group_ids, query, max_facts):
            seen_max_facts.append(max_facts)
            return []

        monkeypatch.setattr(_GC, "search", fake_search)

        result = await tool_search_world_graph(
            memory_engine=_FakeMemoryEngine(),  # type: ignore[arg-type]
            bank_id="bank-A",
            group_id="agent-shared-001",
            query="q",
            max_facts=1000,
            max_tokens=100,
        )
        assert seen_max_facts and seen_max_facts[0] <= 5, (
            f"expected fact_cap to clamp to ~token_budget/50, got {seen_max_facts[0]}"
        )
        assert result["facts"] == []
        assert result["truncated"] is False  # zero facts in == zero facts kept
    finally:
        clear_config_cache()


@pytest.mark.asyncio
async def test_tool_truncates_rendered_lines_when_over_budget(monkeypatch):
    """When the running token count would exceed the budget, the tool must
    stop appending rendered lines. The discarded facts are still returned
    in ``facts`` so the LLM can see the count and reason about coverage.
    """
    monkeypatch.setenv("HINDSIGHT_API_GRAPHITI_BASE_URL", "http://graphiti.test")
    clear_config_cache()
    try:
        from hindsight_api.engine.federation.graphiti_client import GraphitiClient as _GC

        async def fake_search(self, group_ids, query, max_facts):
            # Five long facts — would normally blow past any small budget.
            return [
                FactResult(
                    uuid=UUID(f"33333333-3333-3333-3333-{i:012d}"),
                    name="WORKS_AT",
                    fact=("Alice works at Acme " * 20).strip(),
                    valid_at="2024-01-01T00:00:00Z",
                    invalid_at=None,
                    created_at=None,
                    expired_at=None,
                )
                for i in range(5)
            ]

        monkeypatch.setattr(_GC, "search", fake_search)

        result = await tool_search_world_graph(
            memory_engine=_FakeMemoryEngine(),  # type: ignore[arg-type]
            bank_id="bank-A",
            group_id="agent-shared-001",
            query="q",
            max_facts=10,
            max_tokens=120,  # tight enough to break after the first fact
        )
        assert result["truncated"] is True
        assert len(result["rendered"].splitlines()) < 5
        # The token counter is reported and never exceeds the budget.
        assert result["rendered_tokens"] <= 120
    finally:
        clear_config_cache()


@pytest.mark.asyncio
async def test_tool_minimum_max_tokens_floor(monkeypatch):
    """``max_tokens`` < 100 must still be honored as a floor, not a
    hard cap that disables rendering. The tool uses ``max(int, 100)``
    so a typo like max_tokens=1 still gets a sensible fact_cap.
    """
    monkeypatch.setenv("HINDSIGHT_API_GRAPHITI_BASE_URL", "http://graphiti.test")
    clear_config_cache()
    try:
        from hindsight_api.engine.federation.graphiti_client import GraphitiClient as _GC

        seen_max_facts: list[int] = []

        async def fake_search(self, group_ids, query, max_facts):
            seen_max_facts.append(max_facts)
            return []

        monkeypatch.setattr(_GC, "search", fake_search)

        await tool_search_world_graph(
            memory_engine=_FakeMemoryEngine(),  # type: ignore[arg-type]
            bank_id="bank-A",
            group_id="agent-shared-001",
            query="q",
            max_facts=10,
            max_tokens=1,
        )
        # floor is 100 tokens / 50 = 2 facts; we never request more than the cap
        assert seen_max_facts and 1 <= seen_max_facts[0] <= 2
    finally:
        clear_config_cache()


# ---------------------------------------------------------------------------
# HTTP error paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_returns_error_on_http_500(monkeypatch):
    """A 5xx from Graphiti must degrade to ``{"error": ...}`` rather than
    raising — the reflect loop must be able to fall back to private memory.
    """
    monkeypatch.setenv("HINDSIGHT_API_GRAPHITI_BASE_URL", "http://graphiti.test")
    clear_config_cache()
    try:

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        # Patch the GraphitiClient constructor to return our mock-transport
        # client, so the tool's ``GraphitiClient(base_url=...)`` picks it up.
        from hindsight_api.engine.federation import graphiti_client as gc_mod

        original_init = gc_mod.GraphitiClient.__init__

        def patched_init(self, base_url, api_key=None, **kwargs):
            transport = httpx.MockTransport(handler)
            http = httpx.AsyncClient(transport=transport, base_url=base_url)
            original_init(self, base_url, api_key=api_key, client=http, **kwargs)

        monkeypatch.setattr(gc_mod.GraphitiClient, "__init__", patched_init)

        result = await tool_search_world_graph(
            memory_engine=_FakeMemoryEngine(),  # type: ignore[arg-type]
            bank_id="bank-A",
            group_id="agent-shared-001",
            query="q",
        )
        assert "error" in result
        assert "world graph unavailable" in result["error"]
    finally:
        clear_config_cache()


@pytest.mark.asyncio
async def test_tool_returns_error_on_parse_failure(monkeypatch):
    """A 200 OK with a body that fails UUID parsing must also degrade
    gracefully — the contract is the tool never raises, only the
    GraphitiClient does. (``{"oops": ...}`` is *not* a parse failure
    because the dict branch in ``_parse_fact_results`` tolerates a
    missing ``facts`` key and yields ``[]``.)
    """
    monkeypatch.setenv("HINDSIGHT_API_GRAPHITI_BASE_URL", "http://graphiti.test")
    clear_config_cache()
    try:
        from hindsight_api.engine.federation import graphiti_client as gc_mod

        original_init = gc_mod.GraphitiClient.__init__

        def patched_init(self, base_url, api_key=None, **kwargs):
            def handler(request: httpx.Request) -> httpx.Response:
                # A list with a non-UUID ``uuid`` field makes
                # ``UUID(it["uuid"])`` raise, which the client catches
                # and re-raises as ``GraphitiClientError``.
                return httpx.Response(200, json=[{"uuid": "not-a-uuid"}])

            transport = httpx.MockTransport(handler)
            http = httpx.AsyncClient(transport=transport, base_url=base_url)
            original_init(self, base_url, api_key=api_key, client=http, **kwargs)

        monkeypatch.setattr(gc_mod.GraphitiClient, "__init__", patched_init)

        result = await tool_search_world_graph(
            memory_engine=_FakeMemoryEngine(),  # type: ignore[arg-type]
            bank_id="bank-A",
            group_id="agent-shared-001",
            query="q",
        )
        assert "error" in result
        assert "world graph unavailable" in result["error"]
    finally:
        clear_config_cache()


# ---------------------------------------------------------------------------
# Output shape — done() validation contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_output_items_have_uuid_and_id_fields(monkeypatch):
    """The reflect agent's available_world_fact_ids tracker reads either
    ``uuid`` or ``id`` off the result. The tool must emit both so the
    tracker never sees a None and drops a valid citation.
    """
    monkeypatch.setenv("HINDSIGHT_API_GRAPHITI_BASE_URL", "http://graphiti.test")
    clear_config_cache()
    try:
        from hindsight_api.engine.federation.graphiti_client import GraphitiClient as _GC

        async def fake_search(self, group_ids, query, max_facts):
            return [
                FactResult(
                    uuid=UUID("44444444-4444-4444-4444-444444444444"),
                    name="WORKS_AT",
                    fact="Alice works at Acme",
                    valid_at="2024-01-01T00:00:00Z",
                    invalid_at=None,
                    created_at=None,
                    expired_at=None,
                ),
            ]

        monkeypatch.setattr(_GC, "search", fake_search)

        result = await tool_search_world_graph(
            memory_engine=_FakeMemoryEngine(),  # type: ignore[arg-type]
            bank_id="bank-A",
            group_id="agent-shared-001",
            query="q",
        )
        assert result["facts"][0]["uuid"] == "44444444-4444-4444-4444-444444444444"
        assert result["facts"][0]["id"] == "44444444-4444-4444-4444-444444444444"
    finally:
        clear_config_cache()


# ---------------------------------------------------------------------------
# graphiti_client.search error surface
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graphiti_client_search_raises_on_5xx():
    """Sanity: the underlying client raises ``GraphitiClientError`` on
    5xx. The reflect tool then catches it and emits ``{"error": ...}``.
    This test documents the boundary so a future refactor of one side
    cannot silently change the contract.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="overloaded")

    client = _mocked_client(handler)
    try:
        with pytest.raises(GraphitiClientError):
            await client.search(group_ids=["g1"], query="q", max_facts=5)
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# get_reflect_tools gating
# ---------------------------------------------------------------------------


def test_get_reflect_tools_omits_world_graph_by_default():
    """When ``include_world_graph`` is False (default), the schema must
    not advertise the tool — exposing it would let the LLM attempt to
    call a function the dispatcher will reject.
    """
    from hindsight_api.engine.reflect.tools_schema import get_reflect_tools

    tools = get_reflect_tools()
    tool_names = {t["function"]["name"] for t in tools}
    assert "search_world_graph" not in tool_names


def test_get_reflect_tools_includes_world_graph_when_enabled():
    """When the bank is federated, the schema must advertise the tool.
    The done tool must also surface the ``world_fact_ids`` field so the
    LLM knows it can cite world-graph UUIDs.
    """
    from hindsight_api.engine.reflect.tools_schema import get_reflect_tools

    tools = get_reflect_tools(include_world_graph=True)
    tool_names = {t["function"]["name"] for t in tools}
    assert "search_world_graph" in tool_names

    # Find the done tool and verify world_fact_ids is part of its params.
    done_tool = next(t for t in tools if t["function"]["name"] == "done")
    params = done_tool["function"]["parameters"]
    assert "world_fact_ids" in params["properties"]
