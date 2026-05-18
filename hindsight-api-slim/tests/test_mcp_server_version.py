"""Tests for MCP server identity reported via serverInfo."""

from unittest.mock import MagicMock, patch

from hindsight_api import __version__ as HINDSIGHT_VERSION
from hindsight_api.api.mcp import create_mcp_server


def test_mcp_server_reports_hindsight_version():
    """serverInfo.version should be Hindsight's version, not the FastMCP library version."""
    memory = MagicMock()
    server = create_mcp_server(memory, multi_bank=True)
    assert server.version == HINDSIGHT_VERSION


def test_mcp_server_constructor_does_not_receive_http_transport_options():
    """FastMCP 3.x rejects stateless_http on the constructor; pass it only to http_app()."""
    from hindsight_api.api import mcp as mcp_module

    server = MagicMock()

    with (
        patch.object(mcp_module, "FastMCP", return_value=server) as fastmcp_cls,
        patch.object(mcp_module, "_get_raw_config") as raw_config,
        patch.object(mcp_module, "register_mcp_tools"),
        patch.object(mcp_module, "load_extension", return_value=None),
        patch.object(mcp_module, "_make_tools_tolerant"),
    ):
        raw_config.return_value.mcp_enabled_tools = None

        assert create_mcp_server(MagicMock(), multi_bank=True) is server

    fastmcp_cls.assert_called_once_with("hindsight-mcp-server", version=HINDSIGHT_VERSION)
    assert "stateless_http" not in fastmcp_cls.call_args.kwargs
