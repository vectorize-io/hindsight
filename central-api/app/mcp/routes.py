"""MCP routes — tool registry and gateway endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app.mcp.tools import ALLOWED_TOOLS

router = APIRouter(prefix="/mcp", tags=["mcp"])


@router.get("/tools")
async def list_mcp_tools() -> dict:
    """List all MCP tools exposed through the gateway (no auth required for tool discovery)."""
    tools = [
        {
            "name": "list_workspaces",
            "method": "POST",
            "api_path": "/api/mcp/tools/list_workspaces",
            "description": "List workspaces the actor has access to",
        },
        {
            "name": "list_connected_sources",
            "method": "POST",
            "api_path": "/api/mcp/tools/list_connected_sources",
            "description": "List connected sources for a workspace",
        },
        {
            "name": "list_source_documents",
            "method": "POST",
            "api_path": "/api/mcp/tools/list_source_documents",
            "description": "List documents from connected sources",
        },
        {
            "name": "sync_source",
            "method": "POST",
            "api_path": "/api/mcp/tools/sync_source",
            "description": "Trigger a source sync for a workspace",
        },
        {
            "name": "get_source_audit",
            "method": "POST",
            "api_path": "/api/mcp/tools/get_source_audit",
            "description": "Get audit trail for source operations",
        },
        {
            "name": "list_ingestion_jobs",
            "method": "POST",
            "api_path": "/api/mcp/tools/list_ingestion_jobs",
            "description": "List document ingestion jobs",
        },
        {
            "name": "search_governed_documents",
            "method": "POST",
            "api_path": "/api/mcp/tools/search_governed_documents",
            "description": "Search documents with governance policy enforcement",
        },
    ]
    return {"tools": [t for t in tools if t["name"] in ALLOWED_TOOLS]}
