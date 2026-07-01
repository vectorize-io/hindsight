"""MCP tool surface. Tools call the Central API service/repository layer — never
Google Drive, the vector DB, the database, or raw tokens directly. Every tool
call is audited (mcp_tool_called) and subject to the same governance as the API.
"""
