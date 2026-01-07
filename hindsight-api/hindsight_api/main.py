"""
Command-line interface for Hindsight API.

Run the server with:
    hindsight-api

Run as background daemon:
    hindsight-api --daemon

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
from .daemon import (
    DEFAULT_DAEMON_PORT,
    DEFAULT_IDLE_TIMEOUT,
    DaemonLock,
    IdleTimeoutMiddleware,
    daemonize,
)
from .extensions import DefaultExtensionContext, OperationValidatorExtension, TenantExtension, load_extension

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

    # Daemon mode options
    parser.add_argument(
        "--daemon",
        action="store_true",
        help=f"Run as background daemon (uses port {DEFAULT_DAEMON_PORT}, auto-exits after idle)",
    )
    parser.add_argument(
        "--idle-timeout",
        type=int,
        default=DEFAULT_IDLE_TIMEOUT,
        help=f"Idle timeout in seconds before auto-exit in daemon mode (default: {DEFAULT_IDLE_TIMEOUT})",
    )

    args = parser.parse_args()

    # Daemon mode handling
    if args.daemon:
        # Use fixed daemon port
        args.port = DEFAULT_DAEMON_PORT
        args.host = "127.0.0.1"  # Only bind to localhost for security

        # Check if another daemon is already running
        daemon_lock = DaemonLock()
        if not daemon_lock.acquire():
            print(f"Daemon already running (PID: {daemon_lock.get_pid()})", file=sys.stderr)
            sys.exit(1)

        # Fork into background
        daemonize()

        # Re-acquire lock in child process
        daemon_lock = DaemonLock()
        if not daemon_lock.acquire():
            sys.exit(1)

        # Register cleanup to release lock
        def release_lock():
            daemon_lock.release()

        atexit.register(release_lock)

    # Print banner (not in daemon mode)
    if not args.daemon:
        print()
        print_banner()

    # Configure Python logging based on log level
    # Update config with CLI override if provided
    if args.log_level != config.log_level:
        config = HindsightConfig(
            database_url=config.database_url,
            llm_provider=config.llm_provider,
            llm_api_key=config.llm_api_key,
            llm_model=config.llm_model,
            llm_base_url=config.llm_base_url,
            llm_max_concurrent=config.llm_max_concurrent,
            llm_timeout=config.llm_timeout,
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
            observation_min_facts=config.observation_min_facts,
            observation_top_entities=config.observation_top_entities,
            retain_max_completion_tokens=config.retain_max_completion_tokens,
            skip_llm_verification=config.skip_llm_verification,
            lazy_reranker=config.lazy_reranker,
            run_migrations_on_startup=config.run_migrations_on_startup,
        )
    config.configure_logging()
    if not args.daemon:
        config.log_config()

    # Register cleanup handlers
    atexit.register(_cleanup)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Load operation validator extension if configured
    operation_validator = load_extension("OPERATION_VALIDATOR", OperationValidatorExtension)
    if operation_validator:
        import logging

        logging.info(f"Loaded operation validator: {operation_validator.__class__.__name__}")

    # Load tenant extension if configured
    tenant_extension = load_extension("TENANT", TenantExtension)
    if tenant_extension:
        import logging

        logging.info(f"Loaded tenant extension: {tenant_extension.__class__.__name__}")

    # Create MemoryEngine (reads configuration from environment)
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

    # Create FastAPI app
    app = create_app(
        memory=_memory,
        http_api_enabled=True,
        mcp_api_enabled=config.mcp_enabled,
        mcp_mount_path="/mcp",
        initialize_memory=True,
    )

    # Wrap with idle timeout middleware in daemon mode
    idle_middleware = None
    if args.daemon:
        idle_middleware = IdleTimeoutMiddleware(app, idle_timeout=args.idle_timeout)
        app = idle_middleware

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

    # Print startup info (not in daemon mode)
    if not args.daemon:
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

    # Start idle checker in daemon mode
    if idle_middleware is not None:
        # Start the idle checker in a background thread with its own event loop
        import threading

        def run_idle_checker():
            import time

            time.sleep(2)  # Wait for uvicorn to start
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(idle_middleware._check_idle())
            except Exception:
                pass

        threading.Thread(target=run_idle_checker, daemon=True).start()

    uvicorn.run(**uvicorn_config)  # type: ignore[invalid-argument-type] - dict kwargs


if __name__ == "__main__":
    main()
