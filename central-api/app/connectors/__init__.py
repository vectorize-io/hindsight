"""Source connectors. Connectors are backends — only the Central API may drive
them. Agents, MCP tools, and the GUI never call a provider (e.g. Google Drive)
directly; they call the Central API, which calls the connector.
"""
