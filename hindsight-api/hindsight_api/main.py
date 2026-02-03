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

from . import MemoryEngine, __version__
from .api import create_app
from .banner import print_banner
from .config import DEFAULT_WORKERS, ENV_WORKERS, HindsightConfig, get_config
from .daemon import (
    DEFAULT_DAEMON_PORT,
    DEFAULT_IDLE_TIMEOUT,
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
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv(ENV_WORKERS, str(DEFAULT_WORKERS))),
        help=f"Number of worker processes (env: {ENV_WORKERS}, default: {DEFAULT_WORKERS})",
    )

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
        # Use port from args (may be custom for profiles)
        if args.port == config.port:  # No custom port specified
            args.port = DEFAULT_DAEMON_PORT
        args.host = "127.0.0.1"  # Only bind to localhost for security

        # Fork into background
        # No lockfile needed - port binding prevents duplicate daemons
        daemonize()

    # Print banner (not in daemon mode)
    if not args.daemon:
        print()
        print_banner()

    # Configure Python logging based on log level
    # Update config with CLI override if provided
    if args.log_level != config.log_level:
        config = HindsightConfig(
            database_url=config.database_url,
            database_schema=config.database_schema,
            llm_provider=config.llm_provider,
            llm_api_key=config.llm_api_key,
            llm_model=config.llm_model,
            llm_base_url=config.llm_base_url,
            llm_max_concurrent=config.llm_max_concurrent,
            llm_max_retries=config.llm_max_retries,
            llm_initial_backoff=config.llm_initial_backoff,
            llm_max_backoff=config.llm_max_backoff,
            llm_timeout=config.llm_timeout,
            llm_vertexai_project_id=config.llm_vertexai_project_id,
            llm_vertexai_region=config.llm_vertexai_region,
            llm_vertexai_service_account_key=config.llm_vertexai_service_account_key,
            retain_llm_provider=config.retain_llm_provider,
            retain_llm_api_key=config.retain_llm_api_key,
            retain_llm_model=config.retain_llm_model,
            retain_llm_base_url=config.retain_llm_base_url,
            retain_llm_max_concurrent=config.retain_llm_max_concurrent,
            retain_llm_max_retries=config.retain_llm_max_retries,
            retain_llm_initial_backoff=config.retain_llm_initial_backoff,
            retain_llm_max_backoff=config.retain_llm_max_backoff,
            retain_llm_timeout=config.retain_llm_timeout,
            reflect_llm_provider=config.reflect_llm_provider,
            reflect_llm_api_key=config.reflect_llm_api_key,
            reflect_llm_model=config.reflect_llm_model,
            reflect_llm_base_url=config.reflect_llm_base_url,
            reflect_llm_max_concurrent=config.reflect_llm_max_concurrent,
            reflect_llm_max_retries=config.reflect_llm_max_retries,
            reflect_llm_initial_backoff=config.reflect_llm_initial_backoff,
            reflect_llm_max_backoff=config.reflect_llm_max_backoff,
            reflect_llm_timeout=config.reflect_llm_timeout,
            consolidation_llm_provider=config.consolidation_llm_provider,
            consolidation_llm_api_key=config.consolidation_llm_api_key,
            consolidation_llm_model=config.consolidation_llm_model,
            consolidation_llm_base_url=config.consolidation_llm_base_url,
            consolidation_llm_max_concurrent=config.consolidation_llm_max_concurrent,
            consolidation_llm_max_retries=config.consolidation_llm_max_retries,
            consolidation_llm_initial_backoff=config.consolidation_llm_initial_backoff,
            consolidation_llm_max_backoff=config.consolidation_llm_max_backoff,
            consolidation_llm_timeout=config.consolidation_llm_timeout,
            embeddings_provider=config.embeddings_provider,
            embeddings_local_model=config.embeddings_local_model,
            embeddings_local_force_cpu=config.embeddings_local_force_cpu,
            embeddings_tei_url=config.embeddings_tei_url,
            embeddings_openai_base_url=config.embeddings_openai_base_url,
            embeddings_cohere_base_url=config.embeddings_cohere_base_url,
            reranker_provider=config.reranker_provider,
            reranker_local_model=config.reranker_local_model,
            reranker_local_force_cpu=config.reranker_local_force_cpu,
            reranker_local_max_concurrent=config.reranker_local_max_concurrent,
            reranker_tei_url=config.reranker_tei_url,
            reranker_tei_batch_size=config.reranker_tei_batch_size,
            reranker_tei_max_concurrent=config.reranker_tei_max_concurrent,
            reranker_max_candidates=config.reranker_max_candidates,
            reranker_cohere_base_url=config.reranker_cohere_base_url,
            host=args.host,
            port=args.port,
            log_level=args.log_level,
            log_format=config.log_format,
            mcp_enabled=config.mcp_enabled,
            graph_retriever=config.graph_retriever,
            mpfp_top_k_neighbors=config.mpfp_top_k_neighbors,
            recall_max_concurrent=config.recall_max_concurrent,
            recall_connection_budget=config.recall_connection_budget,
            retain_max_completion_tokens=config.retain_max_completion_tokens,
            retain_chunk_size=config.retain_chunk_size,
            retain_extract_causal_links=config.retain_extract_causal_links,
            retain_extraction_mode=config.retain_extraction_mode,
            retain_custom_instructions=config.retain_custom_instructions,
            enable_observations=config.enable_observations,
            consolidation_batch_size=config.consolidation_batch_size,
            consolidation_max_tokens=config.consolidation_max_tokens,
            skip_llm_verification=config.skip_llm_verification,
            lazy_reranker=config.lazy_reranker,
            run_migrations_on_startup=config.run_migrations_on_startup,
            db_pool_min_size=config.db_pool_min_size,
            db_pool_max_size=config.db_pool_max_size,
            db_command_timeout=config.db_command_timeout,
            db_acquire_timeout=config.db_acquire_timeout,
            worker_enabled=config.worker_enabled,
            worker_id=config.worker_id,
            worker_poll_interval_ms=config.worker_poll_interval_ms,
            worker_max_retries=config.worker_max_retries,
            worker_http_port=config.worker_http_port,
            worker_max_slots=config.worker_max_slots,
            worker_consolidation_max_slots=config.worker_consolidation_max_slots,
            reflect_max_iterations=config.reflect_max_iterations,
            mental_model_refresh_concurrency=config.mental_model_refresh_concurrency,
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
    # When using workers or reload, we must use import string so each worker can import the app
    use_import_string = args.workers > 1 or args.reload
    # Check for uvloop availability
    try:
        import uvloop  # noqa: F401

        loop_impl = "uvloop"
        print("uvloop available, will use for event loop")
    except ImportError:
        loop_impl = "asyncio"
        print("uvloop not installed, using default asyncio event loop")

    uvicorn_config = {
        "app": "hindsight_api.server:app" if use_import_string else app,
        "host": args.host,
        "port": args.port,
        "log_level": args.log_level,
        "access_log": args.access_log,
        "proxy_headers": args.proxy_headers,
        "ws": "wsproto",  # Use wsproto instead of websockets to avoid deprecation warnings
        "loop": loop_impl,  # Explicitly set event loop implementation
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
            version=__version__,
        )

    # Start idle checker in daemon mode
    if idle_middleware is not None:
        # Start the idle checker in a background thread with its own event loop
        import logging
        import threading

        def run_idle_checker():
            import time

            time.sleep(2)  # Wait for uvicorn to start
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(idle_middleware._check_idle())
            except Exception as e:
                logging.error(f"Idle checker error: {e}", exc_info=True)

        threading.Thread(target=run_idle_checker, daemon=True).start()

    uvicorn.run(**uvicorn_config)


if __name__ == "__main__":
    main()
