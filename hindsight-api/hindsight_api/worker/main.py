"""
Command-line interface for Hindsight Worker.

Run the worker with:
    hindsight-worker

Stop with Ctrl+C (graceful shutdown).
"""

import argparse
import asyncio
import atexit
import logging
import os
import signal
import socket
import sys
import warnings

from ..config import get_config
from ..engine.task_backend import SyncTaskBackend
from .poller import WorkerPoller

# Filter deprecation warnings from third-party libraries
warnings.filterwarnings("ignore", message="websockets.legacy is deprecated")
warnings.filterwarnings("ignore", message="websockets.server.WebSocketServerProtocol is deprecated")

# Disable tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


def create_worker_app(poller: WorkerPoller, memory):
    """Create a minimal FastAPI app for worker metrics and health."""
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, Response
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    from ..metrics import create_metrics_collector, get_metrics_collector, initialize_metrics

    app = FastAPI(
        title="Hindsight Worker",
        description="Worker process for distributed task execution",
    )

    # Initialize OpenTelemetry metrics
    try:
        prometheus_reader = initialize_metrics(service_name="hindsight-worker", service_version="1.0.0")
        create_metrics_collector()
        app.state.prometheus_reader = prometheus_reader
        logger.info("Metrics initialized - available at /metrics endpoint")
    except Exception as e:
        logger.warning(f"Failed to initialize metrics: {e}. Metrics will be disabled.")
        app.state.prometheus_reader = None

    # Set up DB pool metrics if available
    metrics_collector = get_metrics_collector()
    if memory._pool is not None and hasattr(metrics_collector, "set_db_pool"):
        metrics_collector.set_db_pool(memory._pool)
        logger.info("DB pool metrics configured")

    @app.get(
        "/health",
        summary="Health check endpoint",
        description="Returns worker health status including database connectivity",
        tags=["Monitoring"],
    )
    async def health_endpoint():
        """Health check endpoint."""
        health = await memory.health_check()
        health["worker_id"] = poller.worker_id
        health["is_shutdown"] = poller.is_shutdown
        status_code = 200 if health.get("status") == "healthy" else 503
        return JSONResponse(content=health, status_code=status_code)

    @app.get(
        "/metrics",
        summary="Prometheus metrics endpoint",
        description="Exports metrics in Prometheus format for scraping",
        tags=["Monitoring"],
    )
    async def metrics_endpoint():
        """Return Prometheus metrics."""
        metrics_data = generate_latest()
        return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)

    @app.get(
        "/",
        summary="Worker info",
        description="Basic worker information",
        tags=["Info"],
    )
    async def root():
        """Return basic worker info."""
        return {
            "service": "hindsight-worker",
            "worker_id": poller.worker_id,
            "is_shutdown": poller.is_shutdown,
        }

    return app


def main():
    """Main entry point for the hindsight-worker CLI."""
    # Load configuration from environment
    config = get_config()

    parser = argparse.ArgumentParser(
        prog="hindsight-worker",
        description="Hindsight Worker - distributed task processor",
    )

    # Worker options
    parser.add_argument(
        "--worker-id",
        default=config.worker_id or socket.gethostname(),
        help="Worker identifier (default: hostname, env: HINDSIGHT_API_WORKER_ID)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=config.worker_poll_interval_ms,
        help=f"Poll interval in milliseconds (default: {config.worker_poll_interval_ms}, env: HINDSIGHT_API_WORKER_POLL_INTERVAL_MS)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=config.worker_max_retries,
        help=f"Max retries before marking failed (default: {config.worker_max_retries}, env: HINDSIGHT_API_WORKER_MAX_RETRIES)",
    )

    # HTTP server options
    parser.add_argument(
        "--http-port",
        type=int,
        default=config.worker_http_port,
        help=f"HTTP port for metrics/health endpoints (default: {config.worker_http_port}, env: HINDSIGHT_API_WORKER_HTTP_PORT)",
    )
    parser.add_argument(
        "--http-host",
        default="0.0.0.0",
        help="HTTP host to bind (default: 0.0.0.0)",
    )

    # Logging options
    parser.add_argument(
        "--log-level",
        default=config.log_level,
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help=f"Log level (default: {config.log_level}, env: HINDSIGHT_API_LOG_LEVEL)",
    )

    args = parser.parse_args()

    # Configure logging
    config.configure_logging()

    # Import MemoryEngine here to avoid circular imports
    from .. import MemoryEngine

    print(f"Starting Hindsight Worker: {args.worker_id}")
    print(f"  Poll interval: {args.poll_interval}ms")
    print(f"  Max retries: {args.max_retries}")
    print(f"  Max slots: {config.worker_max_slots}")
    print(f"  Consolidation max slots: {config.worker_consolidation_max_slots}")
    print(f"  HTTP server: {args.http_host}:{args.http_port}")
    print()

    # Global references for cleanup
    memory = None
    poller = None

    async def run():
        nonlocal memory, poller
        import uvicorn

        from ..extensions import OperationValidatorExtension, TenantExtension, load_extension

        # Load tenant extension BEFORE creating MemoryEngine so it can
        # set correct schema context during task execution. Without this,
        # _authenticate_tenant sees no extension and resets schema to "public",
        # causing worker writes to land in the wrong schema.
        tenant_extension = load_extension("TENANT", TenantExtension)

        # Load operation validator so workers can record usage metering
        # for async operations (e.g. refresh_mental_model after consolidation)
        operation_validator = load_extension("OPERATION_VALIDATOR", OperationValidatorExtension)
        if operation_validator:
            logger.info(f"Loaded operation validator: {operation_validator.__class__.__name__}")

        # Initialize MemoryEngine
        # Workers use SyncTaskBackend because they execute tasks directly,
        # they don't need to store tasks (they poll from DB)
        memory = MemoryEngine(
            run_migrations=False,  # Workers don't run migrations
            task_backend=SyncTaskBackend(),
            tenant_extension=tenant_extension,
            operation_validator=operation_validator,
        )

        await memory.initialize()

        print(f"Database connected: {config.database_url}")

        if tenant_extension:
            print("Tenant extension loaded - schemas will be discovered dynamically on each poll")
        else:
            print(f"No tenant extension configured, using schema: {config.database_schema}")

        # Create a single poller that handles all schemas dynamically
        # Convert default schema to None for SQL compatibility (no schema prefix)
        from hindsight_api.config import DEFAULT_DATABASE_SCHEMA

        schema = None if config.database_schema == DEFAULT_DATABASE_SCHEMA else config.database_schema
        poller = WorkerPoller(
            pool=memory._pool,
            worker_id=args.worker_id,
            executor=memory.execute_task,
            poll_interval_ms=args.poll_interval,
            max_retries=args.max_retries,
            schema=schema,
            tenant_extension=tenant_extension,
            max_slots=config.worker_max_slots,
            consolidation_max_slots=config.worker_consolidation_max_slots,
        )

        # Create the HTTP app for metrics/health
        app = create_worker_app(poller, memory)

        # Setup signal handlers for graceful shutdown using asyncio
        shutdown_requested = asyncio.Event()
        force_exit = False

        loop = asyncio.get_event_loop()

        def signal_handler():
            nonlocal force_exit
            if shutdown_requested.is_set():
                # Second signal = force exit
                print("\nReceived second signal, forcing immediate exit...")
                force_exit = True
                # Restore default handler so third signal kills process
                loop.remove_signal_handler(signal.SIGINT)
                loop.remove_signal_handler(signal.SIGTERM)
                sys.exit(1)
            else:
                print("\nReceived shutdown signal, initiating graceful shutdown...")
                print("(Press Ctrl+C again to force immediate exit)")
                shutdown_requested.set()

        # Use asyncio's signal handlers which work properly with the event loop
        loop.add_signal_handler(signal.SIGINT, signal_handler)
        loop.add_signal_handler(signal.SIGTERM, signal_handler)

        # Create uvicorn config and server
        uvicorn_config = uvicorn.Config(
            app,
            host=args.http_host,
            port=args.http_port,
            log_level="info",  # Reduce uvicorn noise
            access_log=False,
        )
        server = uvicorn.Server(uvicorn_config)

        # Run the poller and HTTP server concurrently
        poller_task = asyncio.create_task(poller.run())
        http_task = asyncio.create_task(server.serve())

        print(f"Worker started. Metrics available at http://{args.http_host}:{args.http_port}/metrics")

        # Wait for shutdown signal
        try:
            await shutdown_requested.wait()
        except KeyboardInterrupt:
            print("\nReceived interrupt, initiating graceful shutdown...")

        # Graceful shutdown
        print("Shutting down HTTP server...")
        server.should_exit = True

        print("Waiting for poller to finish...")
        await poller.shutdown_graceful(timeout=30.0)
        poller_task.cancel()
        try:
            await poller_task
        except asyncio.CancelledError:
            pass

        # Wait for HTTP server to finish
        try:
            await asyncio.wait_for(http_task, timeout=5.0)
        except asyncio.TimeoutError:
            http_task.cancel()
            try:
                await http_task
            except asyncio.CancelledError:
                pass

        # Close memory engine
        await memory.close()
        print("Worker shutdown complete")

    def cleanup():
        """Synchronous cleanup for atexit."""
        if memory is not None and memory._pg0 is not None:
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(memory._pg0.stop())
                loop.close()
                print("\npg0 stopped.")
            except Exception as e:
                print(f"\nError stopping pg0: {e}")

    atexit.register(cleanup)

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nWorker interrupted")
        sys.exit(0)


if __name__ == "__main__":
    main()
