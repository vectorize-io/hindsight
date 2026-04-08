"""Tests for MCP tool argument string-to-JSON coercion (issue #849)."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api.api.mcp import (
    _coerce_string_json,
    _collect_coercible_types,
    _get_mcp_tools,
    _make_tools_tolerant,
)


# ---------------------------------------------------------------------------
# _collect_coercible_types — schema type detection
# ---------------------------------------------------------------------------


class TestCollectCoercibleTypes:
    """Tests for _collect_coercible_types schema detection."""

    def _run(self, schema: dict, param_name: str = "p") -> tuple[set[str], set[str]]:
        array_params: set[str] = set()
        object_params: set[str] = set()
        _collect_coercible_types(schema, param_name, array_params, object_params)
        return array_params, object_params

    # --- array types ---

    def test_direct_array_type(self):
        arrays, objects = self._run({"type": "array", "items": {"type": "string"}})
        assert "p" in arrays and not objects

    def test_anyof_nullable_array(self):
        """list[str] | None → anyOf with array and null."""
        arrays, objects = self._run(
            {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]}
        )
        assert "p" in arrays

    def test_oneof_nullable_array(self):
        """oneOf variant."""
        arrays, objects = self._run(
            {"oneOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]}
        )
        assert "p" in arrays

    # --- object types ---

    def test_direct_object_type(self):
        arrays, objects = self._run({"type": "object"})
        assert "p" in objects and not arrays

    def test_anyof_nullable_object(self):
        """dict[str, str] | None → anyOf with object and null."""
        arrays, objects = self._run({"anyOf": [{"type": "object"}, {"type": "null"}]})
        assert "p" in objects

    def test_oneof_nullable_object(self):
        arrays, objects = self._run({"oneOf": [{"type": "object"}, {"type": "null"}]})
        assert "p" in objects

    # --- non-coercible types (should be ignored) ---

    def test_string_type_ignored(self):
        arrays, objects = self._run({"type": "string"})
        assert not arrays and not objects

    def test_integer_type_ignored(self):
        arrays, objects = self._run({"type": "integer"})
        assert not arrays and not objects

    def test_number_type_ignored(self):
        arrays, objects = self._run({"type": "number"})
        assert not arrays and not objects

    def test_boolean_type_ignored(self):
        arrays, objects = self._run({"type": "boolean"})
        assert not arrays and not objects

    def test_null_type_ignored(self):
        arrays, objects = self._run({"type": "null"})
        assert not arrays and not objects

    def test_anyof_string_or_null_ignored(self):
        """str | None should not be collected."""
        arrays, objects = self._run({"anyOf": [{"type": "string"}, {"type": "null"}]})
        assert not arrays and not objects

    def test_anyof_integer_or_null_ignored(self):
        arrays, objects = self._run({"anyOf": [{"type": "integer"}, {"type": "null"}]})
        assert not arrays and not objects


# ---------------------------------------------------------------------------
# _coerce_string_json — value coercion
# ---------------------------------------------------------------------------


class TestCoerceStringJson:
    """Tests for _coerce_string_json argument coercion."""

    # --- list coercion ---

    def test_coerce_string_to_list(self):
        result = _coerce_string_json(
            {"tags": '["tag1", "tag2"]', "query": "hello"},
            array_params={"tags"},
            object_params=set(),
        )
        assert result["tags"] == ["tag1", "tag2"]
        assert result["query"] == "hello"

    def test_coerce_empty_list_string(self):
        result = _coerce_string_json({"tags": "[]"}, array_params={"tags"}, object_params=set())
        assert result["tags"] == []

    def test_native_list_passthrough(self):
        result = _coerce_string_json({"tags": ["a", "b"]}, array_params={"tags"}, object_params=set())
        assert result["tags"] == ["a", "b"]

    # --- dict coercion ---

    def test_coerce_string_to_dict(self):
        result = _coerce_string_json(
            {"metadata": '{"key": "value"}'},
            array_params=set(),
            object_params={"metadata"},
        )
        assert result["metadata"] == {"key": "value"}

    def test_coerce_empty_dict_string(self):
        result = _coerce_string_json({"metadata": "{}"}, array_params=set(), object_params={"metadata"})
        assert result["metadata"] == {}

    def test_native_dict_passthrough(self):
        result = _coerce_string_json(
            {"metadata": {"key": "value"}}, array_params=set(), object_params={"metadata"}
        )
        assert result["metadata"] == {"key": "value"}

    # --- non-coercible values left untouched ---

    def test_none_passthrough(self):
        result = _coerce_string_json({"tags": None}, array_params={"tags"}, object_params=set())
        assert result["tags"] is None

    def test_invalid_json_string_passthrough(self):
        result = _coerce_string_json({"tags": "not-json"}, array_params={"tags"}, object_params=set())
        assert result["tags"] == "not-json"

    def test_wrong_json_type_not_coerced_list(self):
        """String that parses to a dict should NOT be coerced for an array param."""
        result = _coerce_string_json(
            {"tags": '{"key": "value"}'}, array_params={"tags"}, object_params=set()
        )
        assert result["tags"] == '{"key": "value"}'

    def test_wrong_json_type_not_coerced_dict(self):
        """String that parses to a list should NOT be coerced for an object param."""
        result = _coerce_string_json(
            {"metadata": '["a", "b"]'}, array_params=set(), object_params={"metadata"}
        )
        assert result["metadata"] == '["a", "b"]'

    def test_string_param_not_touched(self):
        """Strings not in array_params/object_params are never modified."""
        result = _coerce_string_json(
            {"query": '["looks", "like", "json"]'},
            array_params=set(),
            object_params=set(),
        )
        assert result["query"] == '["looks", "like", "json"]'

    def test_integer_param_not_touched(self):
        result = _coerce_string_json(
            {"max_tokens": 4096}, array_params=set(), object_params=set()
        )
        assert result["max_tokens"] == 4096

    def test_boolean_param_not_touched(self):
        result = _coerce_string_json(
            {"verbose": True}, array_params=set(), object_params=set()
        )
        assert result["verbose"] is True

    def test_missing_param_no_error(self):
        result = _coerce_string_json(
            {"query": "hello"},
            array_params={"tags"},
            object_params={"metadata"},
        )
        assert result == {"query": "hello"}

    # --- multiple params coerced at once ---

    def test_multiple_params_coerced(self):
        result = _coerce_string_json(
            {
                "tags": '["a", "b"]',
                "types": '["world"]',
                "metadata": '{"source": "test"}',
                "query": "hello",
                "max_tokens": 4096,
            },
            array_params={"tags", "types"},
            object_params={"metadata"},
        )
        assert result["tags"] == ["a", "b"]
        assert result["types"] == ["world"]
        assert result["metadata"] == {"source": "test"}
        assert result["query"] == "hello"
        assert result["max_tokens"] == 4096


# ---------------------------------------------------------------------------
# _make_tools_tolerant — integration test with a real FastMCP tool
# ---------------------------------------------------------------------------


class TestMakeToolsTolerantIntegration:
    """Test that _make_tools_tolerant correctly wraps real FastMCP tool functions."""

    def _create_mcp_with_tool(self):
        """Create a FastMCP instance with a tool that uses various parameter types."""
        from fastmcp import FastMCP

        mcp = FastMCP("test")
        captured = {}

        @mcp.tool(description="test tool with diverse param types")
        async def test_tool(
            query: str,
            max_tokens: int = 100,
            verbose: bool = False,
            tags: list[str] | None = None,
            metadata: dict[str, str] | None = None,
        ) -> dict:
            """Test tool.

            Args:
                query: a string param
                max_tokens: an integer param
                verbose: a boolean param
                tags: an array param
                metadata: an object param
            """
            captured["query"] = query
            captured["max_tokens"] = max_tokens
            captured["verbose"] = verbose
            captured["tags"] = tags
            captured["metadata"] = metadata
            return {"ok": True}

        return mcp, captured

    @pytest.mark.asyncio
    async def test_coerces_string_encoded_list(self):
        mcp, captured = self._create_mcp_with_tool()
        _make_tools_tolerant(mcp)
        tool = _get_mcp_tools(mcp)["test_tool"]
        await tool.run({"query": "hi", "tags": '["a", "b"]'})
        assert captured["tags"] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_coerces_string_encoded_dict(self):
        mcp, captured = self._create_mcp_with_tool()
        _make_tools_tolerant(mcp)
        tool = _get_mcp_tools(mcp)["test_tool"]
        await tool.run({"query": "hi", "metadata": '{"k": "v"}'})
        assert captured["metadata"] == {"k": "v"}

    @pytest.mark.asyncio
    async def test_native_types_pass_through(self):
        mcp, captured = self._create_mcp_with_tool()
        _make_tools_tolerant(mcp)
        tool = _get_mcp_tools(mcp)["test_tool"]
        await tool.run({
            "query": "hi",
            "max_tokens": 200,
            "verbose": True,
            "tags": ["x"],
            "metadata": {"a": "b"},
        })
        assert captured["query"] == "hi"
        assert captured["max_tokens"] == 200
        assert captured["verbose"] is True
        assert captured["tags"] == ["x"]
        assert captured["metadata"] == {"a": "b"}

    @pytest.mark.asyncio
    async def test_strips_extra_args_and_coerces(self):
        """Both extra-arg stripping and coercion work together."""
        mcp, captured = self._create_mcp_with_tool()
        _make_tools_tolerant(mcp)
        tool = _get_mcp_tools(mcp)["test_tool"]
        await tool.run({
            "query": "hi",
            "tags": '["x"]',
            "explanation": "LLM added this",
        })
        assert captured["tags"] == ["x"]
        assert "explanation" not in captured

    @pytest.mark.asyncio
    async def test_string_param_not_coerced(self):
        """A string param whose value happens to look like JSON is NOT coerced."""
        mcp, captured = self._create_mcp_with_tool()
        _make_tools_tolerant(mcp)
        tool = _get_mcp_tools(mcp)["test_tool"]
        await tool.run({"query": '["this", "is", "a", "string"]'})
        assert captured["query"] == '["this", "is", "a", "string"]'

    @pytest.mark.asyncio
    async def test_integer_param_not_coerced(self):
        mcp, captured = self._create_mcp_with_tool()
        _make_tools_tolerant(mcp)
        tool = _get_mcp_tools(mcp)["test_tool"]
        await tool.run({"query": "hi", "max_tokens": 50})
        assert captured["max_tokens"] == 50

    @pytest.mark.asyncio
    async def test_boolean_param_not_coerced(self):
        mcp, captured = self._create_mcp_with_tool()
        _make_tools_tolerant(mcp)
        tool = _get_mcp_tools(mcp)["test_tool"]
        await tool.run({"query": "hi", "verbose": True})
        assert captured["verbose"] is True
