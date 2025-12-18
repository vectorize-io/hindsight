"""
Command-line interface for Hindsight API.

Run the server with:
    hindsight-api

Stop with Ctrl+C.
"""

import argparse
import asyncio
import atexit
import os
import signal
import sys
import warnings

import uvicorn

from . import MemoryEngine
from .api import create_app
from .banner import print_banner
from .config import HindsightConfig, get_config

print()
print_banner()

# Filter deprecation warnings from third-party libraries
warnings.filterwarnings("ignore", message="websockets.legacy is deprecated")
warnings.filterwarnings("ignore", message="websockets.server.WebSocketServerProtocol is deprecated")

# Disable tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global reference for cleanup
_memory: MemoryEngine | None = None


def _cleanup():
    """Synchronous cleanup function to stop resources on exit."""
    global _memory
    if _memory is not None and _memory._pg0 is not None:
        try:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(_memory._pg0.stop())
            loop.close()
            print("\npg0 stopped.")
        except Exception as e:
            print(f"\nError stopping pg0: {e}")


def _signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM to ensure cleanup."""
    print(f"\nReceived signal {signum}, shutting down...")
    _cleanup()
    sys.exit(0)


def main():
    """Main entry point for the CLI."""
    global _memory

    # Load configuration from environment (for CLI args defaults)
    config = get_config()

    parser = argparse.ArgumentParser(
        prog="hindsight-api",
        description="Hindsight API Server",
    )

    # Server options
    parser.add_argument(
        "--host", default=config.host, help=f"Host to bind to (default: {config.host}, env: HINDSIGHT_API_HOST)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=config.port,
        help=f"Port to bind to (default: {config.port}, env: HINDSIGHT_API_PORT)",
    )
    parser.add_argument(
        "--log-level",
        default=config.log_level,
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help=f"Log level (default: {config.log_level}, env: HINDSIGHT_API_LOG_LEVEL)",
    )

    # Development options
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes (development only)")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes (default: 1)")

    # Access log options
    parser.add_argument("--access-log", action="store_true", help="Enable access log")
    parser.add_argument("--no-access-log", dest="access_log", action="store_false", help="Disable access log (default)")
    parser.set_defaults(access_log=False)

    # Proxy options
    parser.add_argument(
        "--proxy-headers", action="store_true", help="Enable X-Forwarded-Proto, X-Forwarded-For headers"
    )
    parser.add_argument(
        "--forwarded-allow-ips", default=None, help="Comma separated list of IPs to trust with proxy headers"
    )

    # SSL options
    parser.add_argument("--ssl-keyfile", default=None, help="SSL key file")
    parser.add_argument("--ssl-certfile", default=None, help="SSL certificate file")

    args = parser.parse_args()

    # Configure Python logging based on log level
    # Update config with CLI override if provided
    if args.log_level != config.log_level:
        config = HindsightConfig(
            database_url=config.database_url,
            llm_provider=config.llm_provider,
            llm_api_key=config.llm_api_key,
            llm_model=config.llm_model,
            llm_base_url=config.llm_base_url,
            embeddings_provider=config.embeddings_provider,
            embeddings_local_model=config.embeddings_local_model,
            embeddings_tei_url=config.embeddings_tei_url,
            reranker_provider=config.reranker_provider,
            reranker_local_model=config.reranker_local_model,
            reranker_tei_url=config.reranker_tei_url,
            host=args.host,
            port=args.port,
            log_level=args.log_level,
            mcp_enabled=config.mcp_enabled,
            graph_retriever=config.graph_retriever,
        )
    config.configure_logging()
    config.log_config()

    # Register cleanup handlers
    atexit.register(_cleanup)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Create MemoryEngine (reads configuration from environment)
    _memory = MemoryEngine()

    # Create FastAPI app
    app = create_app(
        memory=_memory,
        http_api_enabled=True,
        mcp_api_enabled=config.mcp_enabled,
        mcp_mount_path="/mcp",
        initialize_memory=True,
    )

    # Prepare uvicorn config
    uvicorn_config = {
        "app": app,
        "host": args.host,
        "port": args.port,
        "log_level": args.log_level,
        "access_log": args.access_log,
        "proxy_headers": args.proxy_headers,
        "ws": "wsproto",  # Use wsproto instead of websockets to avoid deprecation warnings
    }

    # Add optional parameters if provided
    if args.reload:
        uvicorn_config["reload"] = True
    if args.workers > 1:
        uvicorn_config["workers"] = args.workers
    if args.forwarded_allow_ips:
        uvicorn_config["forwarded_allow_ips"] = args.forwarded_allow_ips
    if args.ssl_keyfile:
        uvicorn_config["ssl_keyfile"] = args.ssl_keyfile
    if args.ssl_certfile:
        uvicorn_config["ssl_certfile"] = args.ssl_certfile

    from .banner import print_startup_info

    print_startup_info(
        host=args.host,
        port=args.port,
        database_url=config.database_url,
        llm_provider=config.llm_provider,
        llm_model=config.llm_model,
        embeddings_provider=config.embeddings_provider,
        reranker_provider=config.reranker_provider,
        mcp_enabled=config.mcp_enabled,
    )

    uvicorn.run(**uvicorn_config)  # type: ignore[invalid-argument-type] - dict kwargs


if __name__ == "__main__":
    main()
