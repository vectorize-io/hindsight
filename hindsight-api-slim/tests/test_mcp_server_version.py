"""Tests for MCP server identity reported via serverInfo."""

from unittest.mock import MagicMock

from hindsight_api import __version__ as HINDSIGHT_VERSION
from hindsight_api.api.mcp import create_mcp_server


def test_mcp_server_reports_hindsight_version():
    """serverInfo.version should be Hindsight's version, not the FastMCP library version."""
    memory = MagicMock()
    server = create_mcp_server(memory, multi_bank=True)
    assert server.version == HINDSIGHT_VERSION
