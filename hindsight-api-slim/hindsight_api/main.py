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
import dataclasses
import os
import signal
import sys
import warnings

import uvicorn

from . import __version__
from .banner import print_banner
from .config import (
    DEFAULT_ACCESS_LOG,
    DEFAULT_WORKERS,
    ENV_ACCESS_LOG,
    ENV_HOST,
    ENV_WORKERS,
    HindsightConfig,
    _get_raw_config,
    load_dotenv_for_entrypoint,
)
from .daemon import (
    DEFAULT_DAEMON_PORT,
    ENV_DAEMON_CHILD,
    daemonize,
)

# `create_app`, `MemoryEngine` and the extension machinery are NOT imported at module level, and
# that is load-bearing rather than tidiness. uvicorn's multiprocess supervisor uses spawn, so every
# worker rebuilds `__main__` by re-running `sys.argv[0]` — pip's console-script wrapper — whose top
# line is `from hindsight_api.main import main`. Anything this module pulls in at import time is
# therefore paid by EVERY spawned worker before uvicorn's child bootstrap begins; a worker still
# importing when the supervisor's 5 s healthcheck arrives is SIGKILLed and respawned, forever, with
# no traceback. Measured: `.api` alone is ~6.2 s to import and `.extensions` ~2.6 s, and the whole
# line the console script runs went 6578 ms -> 312 ms by moving them here.
#
# They resolve through the module `__getattr__` below on first USE, which keeps them ordinary
# module attributes: `main()` refers to them as plain globals, and `patch("hindsight_api.main.
# MemoryEngine")` still finds and replaces them. Importing them inside `main()` instead would do
# neither — the name would be invisible to `patch`, and a local import would shadow any patch that
# did land. See docs/plans/recall-latency.md.
_LAZY_IMPORTS: "dict[str, tuple[str, str]]" = {
    "MemoryEngine": (".", "MemoryEngine"),
    "create_app": (".api", "create_app"),
    "DefaultExtensionContext": (".extensions", "DefaultExtensionContext"),
    "OperationValidatorExtension": (".extensions", "OperationValidatorExtension"),
    "TenantExtension": (".extensions", "TenantExtension"),
    "load_extension": (".extensions", "load_extension"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _LAZY_IMPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    from importlib import import_module

    value = getattr(import_module(module_name, __package__), attribute)
    globals()[name] = value
    return value


# Filter deprecation warnings from third-party libraries
warnings.filterwarnings("ignore", message="websockets.legacy is deprecated")
warnings.filterwarnings("ignore", message="websockets.server.WebSocketServerProtocol is deprecated")

# Disable tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global reference for cleanup
_memory: "MemoryEngine | None" = None


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


@dataclasses.dataclass(frozen=True)
class ResolvedDaemonHostPort:
    host: str
    port: int


def resolve_daemon_host_port(
    *,
    args_host: str,
    args_port: int,
    explicit_host: bool,
    explicit_port: bool,
) -> ResolvedDaemonHostPort:
    """Resolve host/port for daemon mode.

    Defaults to 127.0.0.1 for security, but honors explicit user overrides
    via --host flag or HINDSIGHT_API_HOST env var. Uses DEFAULT_DAEMON_PORT
    unless the user specified a custom port.
    """
    port = args_port if explicit_port else DEFAULT_DAEMON_PORT
    # Only force localhost if the user didn't explicitly set a host
    if explicit_host or os.environ.get(ENV_HOST):
        host = args_host
    else:
        host = "127.0.0.1"
    return ResolvedDaemonHostPort(host=host, port=port)


@dataclasses.dataclass(frozen=True)
class ParsedCliArgs:
    args: argparse.Namespace
    explicit_host: bool
    explicit_port: bool


def _parse_cli_args(argv: list[str], config: HindsightConfig) -> ParsedCliArgs:
    parser = argparse.ArgumentParser(
        prog="hindsight-api",
        description="Hindsight API Server",
    )

    # Server options
    parser.add_argument(
        "--host",
        default=argparse.SUPPRESS,
        help=f"Host to bind to (default: {config.host}, env: HINDSIGHT_API_HOST)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=argparse.SUPPRESS,
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
    parser.add_argument(
        "--access-log",
        action="store_true",
        default=os.getenv(ENV_ACCESS_LOG, "").lower() in ("1", "true", "yes", "on") or DEFAULT_ACCESS_LOG,
        help=f"Enable access log (env: {ENV_ACCESS_LOG}, default: {DEFAULT_ACCESS_LOG})",
    )
    parser.add_argument(
        "--no-access-log",
        dest="access_log",
        action="store_false",
        help="Disable access log (overrides env and default)",
    )

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
        help=f"Run as background daemon (uses port {DEFAULT_DAEMON_PORT})",
    )
    parser.add_argument(
        "--idle-timeout",
        type=int,
        default=0,
        help="Deprecated and ignored: the daemon no longer auto-exits when idle (accepted for "
        "backward compatibility with existing launchers).",
    )

    args = parser.parse_args(argv)

    explicit_host = hasattr(args, "host")
    explicit_port = hasattr(args, "port")
    if not explicit_host:
        args.host = config.host
    if not explicit_port:
        args.port = config.port

    return ParsedCliArgs(args=args, explicit_host=explicit_host, explicit_port=explicit_port)


def main():
    """Main entry point for the CLI."""
    global _memory

    load_dotenv_for_entrypoint()

    # Load configuration from environment (for CLI args defaults)
    config = _get_raw_config()

    parsed_cli_args = _parse_cli_args(sys.argv[1:], config)
    args = parsed_cli_args.args

    # Daemon mode handling.
    # is_daemon_child is True when we are the re-exec'd child spawned by
    # daemonize() or by hindsight-embed's DaemonEmbedManager.  The child
    # does not have --daemon in its argv, but must still behave as a daemon
    # (resolve host/port, suppress banner, etc.).
    is_daemon_child = os.environ.get(ENV_DAEMON_CHILD) == "1"
    is_daemon = args.daemon or is_daemon_child

    if args.idle_timeout:
        # Kept parseable so older launchers (hindsight-embed, the coding-agent
        # integrations) still start, but deliberately inert — see daemon.py.
        print(
            f"--idle-timeout {args.idle_timeout} is ignored: the daemon no longer auto-exits when idle.",
            file=sys.stderr,
        )

    if is_daemon:
        resolved_daemon_host_port = resolve_daemon_host_port(
            args_host=args.host,
            args_port=args.port,
            explicit_host=parsed_cli_args.explicit_host,
            explicit_port=parsed_cli_args.explicit_port,
        )
        args.host = resolved_daemon_host_port.host
        args.port = resolved_daemon_host_port.port

        # Detach into background (parent re-execs and exits; child redirects
        # stdio to log file).  No lockfile needed — port binding prevents
        # duplicate daemons.
        daemonize()

    # Print banner (not in daemon mode)
    if not is_daemon:
        print()
        print_banner()

    # Configure Python logging based on log level
    # Update config with CLI override if provided
    if args.log_level != config.log_level:
        config = dataclasses.replace(config, host=args.host, port=args.port, log_level=args.log_level)
    config.configure_logging()
    if not is_daemon:
        config.log_config()

    # Register cleanup handlers
    atexit.register(_cleanup)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Bind the lazily-resolved names through the MODULE, not as bare globals: a module
    # `__getattr__` (PEP 562) is consulted for `module.X` access, but NOT for a plain global lookup
    # inside this module's own functions — that raises NameError. Reading them off the module object
    # both triggers the lazy import and picks up anything a test has patched onto the module, which
    # a local `from .x import y` would silently shadow.
    _this = sys.modules[__name__]
    MemoryEngine = _this.MemoryEngine
    create_app = _this.create_app
    load_extension = _this.load_extension
    OperationValidatorExtension = _this.OperationValidatorExtension
    TenantExtension = _this.TenantExtension
    DefaultExtensionContext = _this.DefaultExtensionContext

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

    # When using workers or reload, we must use import string so each worker can import the app
    use_import_string = args.workers > 1 or args.reload

    # ...and in THAT mode the parent does not need to build the application at all: it hands
    # uvicorn an import string, and every worker imports `hindsight_api.server:app` for itself, so
    # the object built here was constructed and then thrown away — about ten seconds of work, a
    # MemoryEngine and a whole FastAPI app, for nothing.
    #
    # This is a cleanup, NOT a fix for the worker respawn loop. It was first committed as that fix,
    # on the theory that children inherited the parent's pools and locks across fork; uvicorn's
    # multiprocess uses spawn, not fork, so nothing is inherited, and deploying this to dev left
    # the loop exactly as it was. The real cause is that a spawn child rebuilds `__main__` by
    # re-running `sys.argv[0]` — pip's console-script wrapper — whose top-level
    # `from hindsight_api.main import main` pulls this package's `__init__` and the entire engine
    # with it, before uvicorn's child bootstrap even starts. See docs/plans/recall-latency.md.
    _memory = None
    app = None

    if not use_import_string:
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

    # Check for uvloop/winloop availability
    loop_impl = "asyncio"
    if sys.platform == "win32":
        try:
            import winloop

            winloop.install()  # Patches asyncio globally — uvicorn uses "asyncio" but gets winloop
            loop_impl = "asyncio"  # Tell uvicorn "asyncio" — it's now winloop underneath
            print("winloop installed as asyncio event loop policy (Windows uvloop port)")
        except ImportError:
            print("winloop not installed, using default asyncio event loop")
    else:
        try:
            import uvloop  # noqa: F401

            loop_impl = "uvloop"
            print("uvloop available, will use for event loop")
        except ImportError:
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
        "timeout_keep_alive": 30,  # Exceed aiohttp's 15s client timeout so the client always closes first
        "timeout_graceful_shutdown": 5,  # Cap graceful shutdown at 5s; also enables force-kill on second Ctrl+C
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
    if not is_daemon:
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
            vector_extension=config.vector_extension,
            text_search_extension=config.text_search_extension,
        )

    uvicorn.run(**uvicorn_config)


if __name__ == "__main__":
    main()
