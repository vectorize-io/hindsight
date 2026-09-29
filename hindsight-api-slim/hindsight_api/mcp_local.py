"""
Local MCP server entry point for use with Claude Code (HTTP transport).

This is a thin wrapper around the main hindsight-api server that pre-configures
sensible defaults for local use (embedded PostgreSQL via pg0, warning log level).

The full API runs on localhost:8888. Configure Claude Code's MCP settings:
    claude mcp add --transport http hindsight http://localhost:8888/mcp/

Or pinned to a specific bank (single-bank mode):
    claude mcp add --transport http hindsight http://localhost:8888/mcp/default/

Run with:
    hindsight-local-mcp

Or with uvx:
    uvx hindsight-api@latest hindsight-local-mcp

Environment variables:
    HINDSIGHT_API_LLM_API_KEY: Required. API key for LLM provider.
    HINDSIGHT_API_LLM_PROVIDER: Optional. LLM provider (default: "openai").
    HINDSIGHT_API_LLM_MODEL: Optional. LLM model (default: "gpt-4o-mini").
    HINDSIGHT_API_DATABASE_URL: Optional. Override database URL (default: pg0://hindsight-mcp).
"""

import os


def main() -> None:
    """Start the Hindsight API server with local defaults."""
    # Seeds the environment that HindsightConfig then parses — a write, not a read
    # that bypasses the config. It has to happen before the config is first built,
    # so it cannot be expressed as a config field.
    from hindsight_api.config import ENV_DATABASE_URL

    os.environ.setdefault(ENV_DATABASE_URL, "pg0://hindsight-mcp")

    from hindsight_api.main import main as api_main

    api_main()


if __name__ == "__main__":
    main()
