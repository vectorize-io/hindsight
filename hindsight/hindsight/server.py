"""
Server module for running Hindsight in a background thread.

Provides a simple way to start and stop the Hindsight HTTP API server
without blocking the main thread.
"""
import asyncio
import logging
import socket
import threading
import time
from typing import Optional

import uvicorn
from uvicorn import Config

from hindsight_api import MemoryEngine
from hindsight_api.api import create_app

logger = logging.getLogger(__name__)


def _find_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


class Server:
    """
    Hindsight server that runs in a background thread.

    Example:
        ```python
        from hindsight import Server

        server = Server(
            db_url="pg0",
            llm_provider="groq",
            llm_api_key="your-api-key",
            llm_model="openai/gpt-oss-120b"
        )
        server.start()

        print(f"Server running at {server.url}")

        # Use the server...

        server.stop()
        ```
    """

    def __init__(
        self,
        db_url: str = "pg0",
        llm_provider: str = "groq",
        llm_api_key: str = "",
        llm_model: str = "openai/gpt-oss-120b",
        llm_base_url: Optional[str] = None,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        mcp_enabled: bool = False,
        log_level: str = "info",
    ):
        """
        Initialize the Hindsight server.

        Args:
            db_url: Database URL. Use "pg0" for embedded PostgreSQL.
            llm_provider: LLM provider ("groq", "openai", "ollama")
            llm_api_key: API key for the LLM provider
            llm_model: Model name to use
            llm_base_url: Optional custom base URL for LLM API
            host: Host to bind to (default: 127.0.0.1)
            port: Port to bind to (default: auto-select free port)
            mcp_enabled: Whether to enable MCP server
            log_level: Uvicorn log level (default: warning)
        """
        self.db_url = db_url
        self.llm_provider = llm_provider
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.host = host
        self.port = port or _find_free_port()
        self.mcp_enabled = mcp_enabled
        self.log_level = log_level

        self._memory: Optional[MemoryEngine] = None
        self._server: Optional[uvicorn.Server] = None
        self._thread: Optional[threading.Thread] = None
        self._started = threading.Event()
        self._stopped = threading.Event()

    @property
    def url(self) -> str:
        """Get the server URL."""
        return f"http://{self.host}:{self.port}"

    def _run_server(self):
        """Run the server in a background thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Create MemoryEngine
            self._memory = MemoryEngine(
                db_url=self.db_url,
                memory_llm_provider=self.llm_provider,
                memory_llm_api_key=self.llm_api_key,
                memory_llm_model=self.llm_model,
                memory_llm_base_url=self.llm_base_url,
            )

            # Create FastAPI app
            app = create_app(
                memory=self._memory,
                mcp_api_enabled=self.mcp_enabled,
                initialize_memory=True,
            )

            # Create uvicorn config and server
            config = Config(
                app=app,
                host=self.host,
                port=self.port,
                log_level=self.log_level,
                loop="asyncio",
            )
            self._server = uvicorn.Server(config)

            # Signal that we're starting
            self._started.set()

            # Run the server
            loop.run_until_complete(self._server.serve())

        except Exception as e:
            logger.error(f"Server error: {e}")
            raise
        finally:
            # Cleanup
            if self._memory:
                loop.run_until_complete(self._memory.close())
            loop.close()
            self._stopped.set()

    def start(self, timeout: float = 30.0) -> "Server":
        """
        Start the server in a background thread.

        Args:
            timeout: Maximum time to wait for server to start (seconds)

        Returns:
            self (for chaining)

        Raises:
            RuntimeError: If server fails to start within timeout
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Server is already running")

        self._started.clear()
        self._stopped.clear()

        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()

        # Wait for server to start
        self._started.wait(timeout=timeout)

        # Give uvicorn a moment to actually bind to the port
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with socket.create_connection((self.host, self.port), timeout=1):
                    logger.info(f"Hindsight server started at {self.url}")
                    return self
            except (ConnectionRefusedError, socket.timeout, OSError):
                time.sleep(0.1)

        raise RuntimeError(f"Server failed to start within {timeout} seconds")

    def stop(self, timeout: float = 10.0) -> None:
        """
        Stop the server.

        Args:
            timeout: Maximum time to wait for server to stop (seconds)
        """
        if self._server is None:
            return

        # Signal uvicorn to shutdown
        self._server.should_exit = True

        # Wait for thread to finish
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("Server thread did not stop cleanly")

        self._server = None
        self._thread = None
        logger.info("Hindsight server stopped")

    def __enter__(self) -> "Server":
        """Context manager entry."""
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


def start_server(
    db_url: str = "pg0",
    llm_provider: str = "groq",
    llm_api_key: str = "",
    llm_model: str = "openai/gpt-oss-120b",
    llm_base_url: Optional[str] = None,
    host: str = "127.0.0.1",
    port: Optional[int] = None,
    mcp_enabled: bool = False,
    log_level: str = "warning",
    timeout: float = 30.0,
) -> Server:
    """
    Start a Hindsight server in a background thread.

    This is a convenience function that creates and starts a Server instance.

    Args:
        db_url: Database URL. Use "pg0" for embedded PostgreSQL.
        llm_provider: LLM provider ("groq", "openai", "ollama")
        llm_api_key: API key for the LLM provider
        llm_model: Model name to use
        llm_base_url: Optional custom base URL for LLM API
        host: Host to bind to (default: 127.0.0.1)
        port: Port to bind to (default: auto-select free port)
        mcp_enabled: Whether to enable MCP server
        log_level: Uvicorn log level (default: warning)
        timeout: Maximum time to wait for server to start (seconds)

    Returns:
        Running Server instance

    Example:
        ```python
        from hindsight import start_server, Client

        server = start_server(
            db_url="pg0",
            llm_provider="groq",
            llm_api_key="your-api-key",
            llm_model="openai/gpt-oss-120b"
        )

        client = Client(base_url=server.url)
        client.put(agent_id="assistant", content="User likes Python")

        server.stop()
        ```
    """
    server = Server(
        db_url=db_url,
        llm_provider=llm_provider,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        host=host,
        port=port,
        mcp_enabled=mcp_enabled,
        log_level=log_level,
    )
    return server.start(timeout=timeout)
