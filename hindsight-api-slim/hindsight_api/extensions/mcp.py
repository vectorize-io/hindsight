"""MCP Extension for registering additional MCP tools.

This extension allows external packages (like hindsight-cloud) to register
additional MCP tools on the Hindsight MCP server.

Example:
    HINDSIGHT_API_MCP_EXTENSION=hindsight_cloud.extensions:CloudMCPExtension
"""

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING

from hindsight_api.extensions.base import Extension

if TYPE_CHECKING:
    # Type-only: fastmcp loads when the MCP server starts, so a broken MCP
    # install can't stop a server that doesn't use it (#5453).
    from fastmcp import FastMCP

    from hindsight_api import MemoryEngine

logger = logging.getLogger(__name__)


class MCPExtension(Extension):
    """Base class for MCP extensions that register additional tools.

    Subclass this to add MCP tools in extension packages.

    Example:
        class CloudMCPExtension(MCPExtension):
            def register_tools(self, mcp: "FastMCP", memory: "MemoryEngine") -> None:
                @mcp.tool()
                async def my_custom_tool(query: str) -> str:
                    return "result"
    """

    @abstractmethod
<<<<<<< HEAD
    def register_tools(self, mcp: "FastMCP", memory: MemoryEngine) -> None:
=======
    def register_tools(self, mcp: "FastMCP", memory: "MemoryEngine") -> None:
>>>>>>> 3419560fe (refactor(engine): stop the engine importing the API layer and untangle import cycles (#5964))
        """Register additional MCP tools.

        Args:
            mcp: FastMCP server instance to register tools on
            memory: MemoryEngine instance for accessing memory operations
        """
        pass
