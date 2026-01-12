"""
FastAPI server for Hindsight API.

This module provides the ASGI app for uvicorn import string usage:
    uvicorn hindsight_api.server:app

For CLI usage, use the hindsight-api command instead.
"""

import os
import warnings

# Filter deprecation warnings from third-party libraries
warnings.filterwarnings("ignore", message="websockets.legacy is deprecated")
warnings.filterwarnings("ignore", message="websockets.server.WebSocketServerProtocol is deprecated")

from hindsight_api import MemoryEngine
from hindsight_api.api import create_app
from hindsight_api.config import get_config

# Disable tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load configuration and configure logging
config = get_config()
config.configure_logging()

# Create app at module level (required for uvicorn import string)
# MemoryEngine reads configuration from environment variables automatically
# Note: run_migrations=True by default, but migrations are idempotent so safe with workers
_memory = MemoryEngine(run_migrations=config.run_migrations_on_startup)

# Create unified app with both HTTP and optionally MCP
app = create_app(
    memory=_memory,
    http_api_enabled=True,
    mcp_api_enabled=config.mcp_enabled,
    mcp_mount_path="/mcp",
    initialize_memory=True,
)


if __name__ == "__main__":
    # When run directly, delegate to the CLI
    from hindsight_api.main import main

    main()
