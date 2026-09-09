"""
FastAPI server for Hindsight API.

This module provides the ASGI app for uvicorn import string usage:
    uvicorn hindsight_api.server:app

For CLI usage, use the hindsight-api command instead.
"""

import logging
import os
import warnings

# Filter deprecation warnings from third-party libraries
warnings.filterwarnings("ignore", message="websockets.legacy is deprecated")
warnings.filterwarnings("ignore", message="websockets.server.WebSocketServerProtocol is deprecated")

# This module IS an entry point: it is the ASGI app targeted by
# `uvicorn hindsight_api.server:app`, and the import string uvicorn re-imports
# in each worker process when `hindsight-api` runs with --workers/--reload. Load
# .env here (like the CLI entry points) so those worker processes see the same
# configuration. Importing hindsight_api as a library never reaches this module.
# Note: Must be called BEFORE importing MemoryEngine or create_app, because
# engine submodules (e.g. llm_wrapper) instantiate semaphores at import time.
from hindsight_api.config import get_config, load_dotenv_for_entrypoint

load_dotenv_for_entrypoint()

# Arm profiling in THIS process, which is where the requests are served.
#
# `main()` arms it too, but with `--workers N` uvicorn's supervisor spawns children that re-import
# this module and never run `main()` — so the only profiler was the one in the supervisor, and its
# report was `keep_subprocess_alive`/`ping` at 0.01 cores while the workers did all the work. A
# profile that cannot see the request path is worse than none: it looks like an answer.
#
# Safe to call twice: cProfile is a process-wide tool since 3.12, and `install()` is a no-op when
# HINDSIGHT_API_PROFILE is unset and idempotent within a process.
from hindsight_api.profiling import install as _install_profiling

_install_profiling()

# Disable tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from hindsight_api import MemoryEngine
from hindsight_api.api import create_app
from hindsight_api.extensions import (
    DefaultExtensionContext,
    OperationValidatorExtension,
    TenantExtension,
    load_extension,
)

# Load configuration and configure logging
config = get_config()
config.configure_logging()

# Load operation validator extension if configured
operation_validator = load_extension("OPERATION_VALIDATOR", OperationValidatorExtension)
if operation_validator:
    logging.info(f"Loaded operation validator: {operation_validator.__class__.__name__}")

# Load tenant extension if configured
tenant_extension = load_extension("TENANT", TenantExtension)
if tenant_extension:
    logging.info(f"Loaded tenant extension: {tenant_extension.__class__.__name__}")

# Create app at module level (required for uvicorn import string)
# MemoryEngine reads configuration from environment variables automatically
# Note: run_migrations=True by default, but migrations are idempotent so safe with workers
_memory = MemoryEngine(
    operation_validator=operation_validator,
    tenant_extension=tenant_extension,
    run_migrations=config.run_migrations_on_startup,
)

# Set extension context on tenant extension (needed for schema provisioning)
if tenant_extension:
    extension_context = DefaultExtensionContext(
        database_url=config.database_url,
        memory_engine=_memory,
    )
    tenant_extension.set_context(extension_context)
    logging.info("Extension context set on tenant extension")

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
