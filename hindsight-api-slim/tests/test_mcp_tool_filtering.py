"""Tests for filter_mcp_tools on OperationValidatorExtension."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api.api.mcp import (
    _current_api_key,
    _current_api_key_id,
    _current_bank_id,
    _current_mcp_authenticated,
    _current_tenant_id,
    create_mcp_server,
)
from hindsight_api.extensions.operation_validator import OperationValidatorExtension, ValidationResult
from hindsight_api.models import RequestContext


class MinimalValidator(OperationValidatorExtension):
    """Minimal concrete subclass — only implements abstract methods."""

    async def validate_retain(self, ctx):
        return ValidationResult.accept()

    async def validate_recall(self, ctx):
        return ValidationResult.accept()

    async def validate_reflect(self, ctx):
        return ValidationResult.accept()


class FilteringValidator(OperationValidatorExtension):
    """Validator that removes retain from the tool set."""

    async def validate_retain(self, ctx):
        return ValidationResult.accept()

    async def validate_recall(self, ctx):
        return ValidationResult.accept()

    async def validate_reflect(self, ctx):
        return ValidationResult.accept()

    async def filter_mcp_tools(self, bank_id, request_context, tools):
        return tools - {"retain"}


@pytest.mark.asyncio
async def test_filter_mcp_tools_default_returns_all():
    """Default implementation returns the input unchanged."""
    validator = MinimalValidator({})
    tools = frozenset({"retain", "recall", "reflect", "list_memories"})
    ctx = RequestContext()

    result = await validator.filter_mcp_tools("test-bank", ctx, tools)

    assert result == tools
    assert isinstance(result, frozenset)


@pytest.mark.asyncio
async def test_filter_mcp_tools_subclass_removes_tools():
    """Subclass can remove tools from the set."""
    validator = FilteringValidator({})
    tools = frozenset({"retain", "recall", "reflect"})
    ctx = RequestContext()

    result = await validator.filter_mcp_tools("test-bank", ctx, tools)

    assert result == frozenset({"recall", "reflect"})
    assert "retain" not in result


@pytest.mark.asyncio
async def test_filter_mcp_tools_returns_empty_set():
    """Validator can return empty set — no tools visible."""

    class DenyAllValidator(OperationValidatorExtension):
        async def validate_retain(self, ctx):
            return ValidationResult.accept()

        async def validate_recall(self, ctx):
            return ValidationResult.accept()

        async def validate_reflect(self, ctx):
            return ValidationResult.accept()

        async def filter_mcp_tools(self, bank_id, request_context, tools):
            return frozenset()

    validator = DenyAllValidator({})
    tools = frozenset({"retain", "recall", "reflect"})
    ctx = RequestContext()

    result = await validator.filter_mcp_tools("test-bank", ctx, tools)

    assert result == frozenset()
    assert len(result) == 0


@pytest.mark.asyncio
async def test_validator_filters_tools_list():
    """Validator filter is applied during tools/list via _get_enabled_tools."""
    mock_memory = MagicMock()
    mock_memory._tenant_extension = MagicMock()
    mock_memory._tenant_extension.authenticate_mcp = AsyncMock()
    mock_memory.retain_batch_async = AsyncMock()
    mock_memory.submit_async_retain = AsyncMock()
    mock_memory.recall_async = AsyncMock()
    mock_memory.reflect_async = AsyncMock()
    mock_memory.list_banks = AsyncMock(return_value=[])

    validator = FilteringValidator({})
    mock_memory._operation_validator = validator

    mock_config = {"mcp_enabled_tools": None}
    mock_memory._config_resolver = MagicMock()
    mock_memory._config_resolver.get_bank_config = AsyncMock(return_value=mock_config)

    mcp_server = create_mcp_server(mock_memory, multi_bank=False)

    bank_token = _current_bank_id.set("test-bank")
    api_key_token = _current_api_key.set("hsk_test_key")
    tenant_token = _current_tenant_id.set("alice")
    key_id_token = _current_api_key_id.set("key-uuid")
    mcp_auth_token = _current_mcp_authenticated.set(False)
    try:
        if hasattr(mcp_server, "list_tools"):
            tools = await mcp_server.list_tools()
            tool_names = {t.name for t in tools}
        else:
            tools = await mcp_server._tool_manager.get_tools()
            tool_names = set(tools.keys())

        assert "recall" in tool_names
        assert "reflect" in tool_names
        assert "retain" not in tool_names
    finally:
        _current_bank_id.reset(bank_token)
        _current_api_key.reset(api_key_token)
        _current_tenant_id.reset(tenant_token)
        _current_api_key_id.reset(key_id_token)
        _current_mcp_authenticated.reset(mcp_auth_token)


@pytest.mark.asyncio
async def test_bank_config_and_validator_compose():
    """Bank config sets ceiling, validator narrows further."""
    mock_memory = MagicMock()
    mock_memory._tenant_extension = MagicMock()
    mock_memory._tenant_extension.authenticate_mcp = AsyncMock()
    mock_memory.retain_batch_async = AsyncMock()
    mock_memory.submit_async_retain = AsyncMock()
    mock_memory.recall_async = AsyncMock()
    mock_memory.reflect_async = AsyncMock()
    mock_memory.list_banks = AsyncMock(return_value=[])

    mock_memory._operation_validator = FilteringValidator({})

    mock_config = {"mcp_enabled_tools": ["recall", "retain", "reflect"]}
    mock_memory._config_resolver = MagicMock()
    mock_memory._config_resolver.get_bank_config = AsyncMock(return_value=mock_config)

    mcp_server = create_mcp_server(mock_memory, multi_bank=False)

    bank_token = _current_bank_id.set("test-bank")
    api_key_token = _current_api_key.set("hsk_test")
    tenant_token = _current_tenant_id.set("alice")
    key_id_token = _current_api_key_id.set("key-1")
    mcp_auth_token = _current_mcp_authenticated.set(False)
    try:
        if hasattr(mcp_server, "list_tools"):
            tools = await mcp_server.list_tools()
            tool_names = {t.name for t in tools}
        else:
            tools = await mcp_server._tool_manager.get_tools()
            tool_names = set(tools.keys())

        assert tool_names == {"recall", "reflect"}
    finally:
        _current_bank_id.reset(bank_token)
        _current_api_key.reset(api_key_token)
        _current_tenant_id.reset(tenant_token)
        _current_api_key_id.reset(key_id_token)
        _current_mcp_authenticated.reset(mcp_auth_token)


@pytest.mark.asyncio
async def test_validator_cannot_add_tools_beyond_bank_config():
    """Validator returning tools not in bank config doesn't expand the set."""

    class PermissiveValidator(OperationValidatorExtension):
        async def validate_retain(self, ctx):
            return ValidationResult.accept()

        async def validate_recall(self, ctx):
            return ValidationResult.accept()

        async def validate_reflect(self, ctx):
            return ValidationResult.accept()

        async def filter_mcp_tools(self, bank_id, request_context, tools):
            return tools | {"retain", "delete_bank"}

    mock_memory = MagicMock()
    mock_memory._tenant_extension = MagicMock()
    mock_memory._tenant_extension.authenticate_mcp = AsyncMock()
    mock_memory.retain_batch_async = AsyncMock()
    mock_memory.submit_async_retain = AsyncMock()
    mock_memory.recall_async = AsyncMock()
    mock_memory.reflect_async = AsyncMock()
    mock_memory.list_banks = AsyncMock(return_value=[])

    mock_memory._operation_validator = PermissiveValidator({})

    mock_config = {"mcp_enabled_tools": ["recall"]}
    mock_memory._config_resolver = MagicMock()
    mock_memory._config_resolver.get_bank_config = AsyncMock(return_value=mock_config)

    mcp_server = create_mcp_server(mock_memory, multi_bank=False)

    bank_token = _current_bank_id.set("test-bank")
    api_key_token = _current_api_key.set("hsk_test")
    tenant_token = _current_tenant_id.set("alice")
    key_id_token = _current_api_key_id.set("key-1")
    mcp_auth_token = _current_mcp_authenticated.set(False)
    try:
        if hasattr(mcp_server, "list_tools"):
            tools = await mcp_server.list_tools()
            tool_names = {t.name for t in tools}
        else:
            tools = await mcp_server._tool_manager.get_tools()
            tool_names = set(tools.keys())

        assert "recall" in tool_names
        assert "retain" not in tool_names
        assert "delete_bank" not in tool_names
    finally:
        _current_bank_id.reset(bank_token)
        _current_api_key.reset(api_key_token)
        _current_tenant_id.reset(tenant_token)
        _current_api_key_id.reset(key_id_token)
        _current_mcp_authenticated.reset(mcp_auth_token)


@pytest.mark.asyncio
async def test_validator_exception_fails_open(caplog):
    """If filter_mcp_tools raises, all tools remain visible and warning is logged."""
    import logging

    caplog.set_level(logging.WARNING)

    class BrokenValidator(OperationValidatorExtension):
        async def validate_retain(self, ctx):
            return ValidationResult.accept()

        async def validate_recall(self, ctx):
            return ValidationResult.accept()

        async def validate_reflect(self, ctx):
            return ValidationResult.accept()

        async def filter_mcp_tools(self, bank_id, request_context, tools):
            raise RuntimeError("Policy backend unreachable")

    mock_memory = MagicMock()
    mock_memory._tenant_extension = MagicMock()
    mock_memory._tenant_extension.authenticate_mcp = AsyncMock()
    mock_memory.retain_batch_async = AsyncMock()
    mock_memory.submit_async_retain = AsyncMock()
    mock_memory.recall_async = AsyncMock()
    mock_memory.reflect_async = AsyncMock()
    mock_memory.list_banks = AsyncMock(return_value=[])

    mock_memory._operation_validator = BrokenValidator({})
    mock_config = {"mcp_enabled_tools": None}
    mock_memory._config_resolver = MagicMock()
    mock_memory._config_resolver.get_bank_config = AsyncMock(return_value=mock_config)

    mcp_server = create_mcp_server(mock_memory, multi_bank=False)

    bank_token = _current_bank_id.set("test-bank")
    api_key_token = _current_api_key.set("hsk_test")
    tenant_token = _current_tenant_id.set("alice")
    key_id_token = _current_api_key_id.set("key-1")
    mcp_auth_token = _current_mcp_authenticated.set(False)
    try:
        if hasattr(mcp_server, "list_tools"):
            tools = await mcp_server.list_tools()
            tool_names = {t.name for t in tools}
        else:
            tools = await mcp_server._tool_manager.get_tools()
            tool_names = set(tools.keys())

        assert "retain" in tool_names
        assert "recall" in tool_names
        assert "reflect" in tool_names

        assert any("filter_mcp_tools raised" in r.message for r in caplog.records)
    finally:
        _current_bank_id.reset(bank_token)
        _current_api_key.reset(api_key_token)
        _current_tenant_id.reset(tenant_token)
        _current_api_key_id.reset(key_id_token)
        _current_mcp_authenticated.reset(mcp_auth_token)


@pytest.mark.asyncio
async def test_no_validator_returns_unfiltered():
    """Without an operation validator, tools/list returns all tools."""
    mock_memory = MagicMock()
    mock_memory._tenant_extension = MagicMock()
    mock_memory._tenant_extension.authenticate_mcp = AsyncMock()
    mock_memory.retain_batch_async = AsyncMock()
    mock_memory.submit_async_retain = AsyncMock()
    mock_memory.recall_async = AsyncMock()
    mock_memory.reflect_async = AsyncMock()
    mock_memory.list_banks = AsyncMock(return_value=[])

    mock_memory._operation_validator = None
    mock_config = {"mcp_enabled_tools": None}
    mock_memory._config_resolver = MagicMock()
    mock_memory._config_resolver.get_bank_config = AsyncMock(return_value=mock_config)

    mcp_server = create_mcp_server(mock_memory, multi_bank=False)

    bank_token = _current_bank_id.set("test-bank")
    api_key_token = _current_api_key.set("hsk_test")
    tenant_token = _current_tenant_id.set("alice")
    key_id_token = _current_api_key_id.set("key-1")
    mcp_auth_token = _current_mcp_authenticated.set(False)
    try:
        if hasattr(mcp_server, "list_tools"):
            tools = await mcp_server.list_tools()
            tool_names = {t.name for t in tools}
        else:
            tools = await mcp_server._tool_manager.get_tools()
            tool_names = set(tools.keys())

        assert "retain" in tool_names
        assert "recall" in tool_names
        assert "reflect" in tool_names
    finally:
        _current_bank_id.reset(bank_token)
        _current_api_key.reset(api_key_token)
        _current_tenant_id.reset(tenant_token)
        _current_api_key_id.reset(key_id_token)
        _current_mcp_authenticated.reset(mcp_auth_token)
