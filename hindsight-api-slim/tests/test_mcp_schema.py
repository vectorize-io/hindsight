"""Tests for portable MCP tool schemas (``hindsight_api.api.mcp_schema``)."""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastmcp import FastMCP

from hindsight_api.api.mcp import _make_tool_schemas_portable
from hindsight_api.api.mcp_schema import ANY_SCHEMA, MAX_REF_DEPTH, dereference_refs
from hindsight_api.mcp_tools import MentalModelTriggerInput


def _has_ref_cycle(schema: Any) -> bool:
    """Whether ``schema`` contains a ``$ref`` that is reachable from itself."""
    graph: dict[str, set[str]] = {}
    definitions: dict[str, Any] = {}
    definitions.update(schema.get("$defs") or {})
    definitions.update(schema.get("definitions") or {})

    def build(node: Any, current: str | None = None) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "$ref" and isinstance(value, str):
                    if current is not None:
                        graph.setdefault(current, set()).add(value)
                else:
                    build(value, current)
        elif isinstance(node, list):
            for item in node:
                build(item, current)

    for name, node in definitions.items():
        ref = f"#/$defs/{name}"
        graph.setdefault(ref, set())
        build(node, ref)

    for start in graph:
        stack = [(start, {start})]
        while stack:
            node, seen = stack.pop()
            for neighbour in graph.get(node, ()):
                if neighbour in seen:
                    return True
                stack.append((neighbour, seen | {neighbour}))
    return False


class TestDereferenceRefs:
    """Unit tests for the pure ``$ref`` dereferencer."""

    def test_schema_without_refs_is_returned_unchanged(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}},
            "required": ["name"],
        }

        assert dereference_refs(schema) == schema

    def test_acyclic_ref_is_inlined_and_defs_dropped(self):
        schema = {
            "type": "object",
            "properties": {"leaf": {"$ref": "#/$defs/Leaf"}},
            "$defs": {"Leaf": {"type": "object", "properties": {"tags": {"type": "array"}}}},
        }

        result = dereference_refs(schema)

        assert "$defs" not in result
        assert result["properties"]["leaf"] == {"type": "object", "properties": {"tags": {"type": "array"}}}

    def test_recursive_ref_is_cut_and_output_is_acyclic(self):
        # A cons-list: `Node` contains a `Node`, the shape that broke the Meta API.
        schema = {
            "type": "object",
            "properties": {"root": {"$ref": "#/$defs/Node"}},
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "next": {"anyOf": [{"$ref": "#/$defs/Node"}, {"type": "null"}]},
                    },
                }
            },
        }

        result = dereference_refs(schema)

        assert not _has_ref_cycle(result)
        assert "$ref" not in json.dumps(result)
        assert "$defs" not in result
        # The first level keeps its shape; the self-reference is cut to a placeholder.
        assert result["properties"]["root"]["properties"]["value"] == {"type": "string"}
        assert result["properties"]["root"]["properties"]["next"]["anyOf"][0] == ANY_SCHEMA

    def test_ref_sibling_keywords_are_preserved(self):
        # Pydantic emits `description` beside `$ref`; JSON Schema 2020-12 allows it.
        schema = {
            "type": "object",
            "properties": {
                "leaf": {"$ref": "#/$defs/Leaf", "description": "the leaf"},
            },
            "$defs": {"Leaf": {"type": "object", "properties": {"tags": {"type": "array"}}}},
        }

        result = dereference_refs(schema)

        assert result["properties"]["leaf"]["description"] == "the leaf"
        assert result["properties"]["leaf"]["properties"] == {"tags": {"type": "array"}}

    def test_property_literally_named_ref_is_not_followed(self):
        schema = {
            "type": "object",
            "properties": {"$ref": {"type": "string", "description": "a literal field name"}},
        }

        assert dereference_refs(schema) == schema

    def test_remote_or_unknown_ref_becomes_permissive_placeholder(self):
        schema = {
            "type": "object",
            "properties": {
                "remote": {"$ref": "https://example.com/schema.json#/Thing"},
                "missing": {"$ref": "#/$defs/Nope"},
            },
        }

        result = dereference_refs(schema)

        assert result["properties"]["remote"] == ANY_SCHEMA
        assert result["properties"]["missing"] == ANY_SCHEMA

    def test_expansion_depth_is_bounded(self):
        # A non-cyclic chain deeper than the bound must still terminate acyclic.
        defs = {
            f"D{i}": {"type": "object", "properties": {"next": {"$ref": f"#/$defs/D{i + 1}"}}}
            for i in range(MAX_REF_DEPTH + 5)
        }
        defs[f"D{MAX_REF_DEPTH + 5}"] = {"type": "string"}
        schema = {"type": "object", "properties": {"root": {"$ref": "#/$defs/D0"}}, "$defs": defs}

        result = dereference_refs(schema)

        assert "$ref" not in json.dumps(result)
        assert not _has_ref_cycle(result)


def _mental_model_trigger_tool(trigger: MentalModelTriggerInput | None = None) -> str:
    """A tool mirroring the mental-model tools whose schema carried the cycle."""
    return "ok"


class TestToolSchemasPortable:
    """Regression tests: the real recursive model must not reach the wire."""

    def _server(self) -> FastMCP:
        mcp = FastMCP("test")
        mcp.tool(_mental_model_trigger_tool)
        return mcp

    def _parameters(self, mcp: FastMCP, name: str) -> dict[str, Any]:
        tools = mcp._local_provider._components
        tool = next(value for key, value in tools.items() if key.startswith("tool:") and name in key)
        return tool.parameters

    def test_recursive_model_is_acyclic_before_and_after(self):
        mcp = self._server()
        before = self._parameters(mcp, "mental_model_trigger_tool")
        assert _has_ref_cycle(before), "fixture must reproduce the upstream cycle"

        _make_tool_schemas_portable(mcp)

        after = self._parameters(mcp, "mental_model_trigger_tool")
        assert not _has_ref_cycle(after)
        assert "$ref" not in json.dumps(after)
        assert "$defs" not in after

    def test_missing_tool_manager_is_reported_not_raised(self):
        _make_tool_schemas_portable(MagicMock(spec=[]))  # must not raise

    @pytest.mark.parametrize("tool_name", ["_mental_model_trigger_tool"])
    def test_every_tool_schema_is_acyclic(self, tool_name: str):
        mcp = self._server()

        _make_tool_schemas_portable(mcp)

        assert not _has_ref_cycle(self._parameters(mcp, tool_name))

    def test_create_mcp_server_emits_only_acyclic_schemas(self):
        """The assembled server — the wire contract agents actually read — is clean."""
        from hindsight_api.api.mcp import _get_mcp_tools, create_mcp_server

        memory = MagicMock()
        memory.resolve_bank_alias = AsyncMock(side_effect=lambda bank_id, **_: bank_id)
        tools = _get_mcp_tools(create_mcp_server(memory))

        assert "create_mental_model" in tools, "regression fixture must expose the recursive tool"
        for name, tool in tools.items():
            assert not _has_ref_cycle(tool.parameters), f"{name} still carries a recursive $ref"
            assert "$ref" not in json.dumps(tool.parameters), f"{name} still carries a $ref"
