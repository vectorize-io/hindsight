"""
Embedded Hindsight client with automatic daemon lifecycle management.

This module provides HindsightEmbedded, a client that uses the same daemon
management interface as hindsight-embed CLI, ensuring full compatibility.

Example:
    ```python
    from hindsight import HindsightEmbedded

    # Daemon starts automatically on first use
    client = HindsightEmbedded(
        profile="myapp",
        llm_provider="groq",
        llm_api_key="your-api-key",
    )

    # Use just like HindsightClient
    client.retain(bank_id="alice", content="Alice loves AI")
    results = client.recall(bank_id="alice", query="What does Alice like?")

    # Optional cleanup
    client.close()
    ```

Using context manager:
    ```python
    from hindsight import HindsightEmbedded

    with HindsightEmbedded(profile="myapp") as client:
        client.retain(bank_id="alice", content="Alice loves AI")
        # Daemon managed automatically
    ```
"""

import logging
import os
import threading
from typing import Optional

from hindsight_client import Hindsight
from hindsight_embed import get_embed_manager

from .api_namespaces import BanksAPI, DirectivesAPI, MemoriesAPI, MentalModelsAPI

logger = logging.getLogger(__name__)


class HindsightEmbedded:
    """
    Hindsight client with automatic daemon lifecycle management.

    This client uses the same daemon management interface as hindsight-embed CLI,
    ensuring full compatibility and shared profiles. The daemon is started automatically
    on first use and manages profile-specific databases.

    Profile data is stored in: ~/.pg0/instances/hindsight-embed-{profile}/

    All methods from HindsightClient are available:
    - retain(), retain_batch()
    - recall()
    - reflect()
    - create_bank(), set_mission(), delete_bank()
    - create_mental_model(), list_mental_models(), etc.
    - create_directive(), list_directives(), etc.
    - And all async variants (aretain, arecall, areflect, etc.)

    Args:
        profile: Profile name for data isolation (default: "default")
        llm_provider: LLM provider ("groq", "openai", "ollama", "gemini", "anthropic", "lmstudio")
        llm_api_key: API key for the LLM provider
        llm_model: Model name to use
        llm_base_url: Optional custom base URL for LLM API
        database_url: Optional database URL override (default: profile-specific pg0)
        idle_timeout: Seconds before daemon auto-exits when idle (default: 300)
        log_level: Daemon log level (default: "info")
    """

    def __init__(
        self,
        profile: str = "default",
        llm_provider: str = "groq",
        llm_api_key: str = "",
        llm_model: str = "openai/gpt-oss-120b",
        llm_base_url: Optional[str] = None,
        database_url: Optional[str] = None,
        idle_timeout: int = 300,
        log_level: str = "info",
    ):
        """
        Initialize the embedded client (daemon starts on first use).

        Args:
            profile: Profile name for data isolation
            llm_provider: LLM provider
            llm_api_key: API key for the LLM provider
            llm_model: Model name to use
            llm_base_url: Optional custom base URL for LLM API
            database_url: Optional database URL override
            idle_timeout: Seconds before daemon auto-exits when idle
            log_level: Daemon log level
        """
        self.profile = profile

        # Build config dict for daemon (matches CLI format)
        self.config = {
            "HINDSIGHT_API_LLM_PROVIDER": llm_provider,
            "HINDSIGHT_API_LLM_API_KEY": llm_api_key,
            "HINDSIGHT_API_LLM_MODEL": llm_model,
            "HINDSIGHT_API_LOG_LEVEL": log_level,
            "HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT": str(idle_timeout),
        }

        if llm_base_url:
            self.config["HINDSIGHT_API_LLM_BASE_URL"] = llm_base_url

        if database_url:
            self.config["HINDSIGHT_EMBED_API_DATABASE_URL"] = database_url

        self._client: Optional[Hindsight] = None
        self._lock = threading.Lock()
        self._started = False
        self._closed = False
        self._manager = get_embed_manager()

        # API namespaces (initialized once, lazily)
        self._banks_api: Optional[BanksAPI] = None
        self._mental_models_api: Optional[MentalModelsAPI] = None
        self._directives_api: Optional[DirectivesAPI] = None
        self._memories_api: Optional[MemoriesAPI] = None

    def _ensure_started(self):
        """Ensure daemon is running (thread-safe)."""
        if self._started and self._client is not None:
            return

        with self._lock:
            # Double-check after acquiring lock
            if self._started and self._client is not None:
                return

            if self._closed:
                raise RuntimeError("Cannot use HindsightEmbedded after it has been closed")

            # Use embed manager interface for daemon management
            logger.info(f"Ensuring daemon is running for profile '{self.profile}'...")
            success = self._manager.ensure_running(self.config, self.profile)
            if not success:
                raise RuntimeError(f"Failed to start daemon for profile '{self.profile}'")

            # Get daemon URL and create client
            daemon_url = self._manager.get_url(self.profile)
            self._client = Hindsight(base_url=daemon_url)
            self._started = True
            logger.info(f"Connected to daemon at {daemon_url}")

    def _cleanup(self, stop_daemon_on_close: bool = False):
        """
        Cleanup client resources (idempotent).

        Args:
            stop_daemon_on_close: If True, stops the daemon. Otherwise, daemon continues
                running (it will auto-stop after idle timeout).
        """
        if self._closed:
            return

        with self._lock:
            if self._closed:
                return

            if self._client is not None:
                self._client.close()
                self._client = None

            # Optionally stop daemon (daemon has idle timeout, so not required)
            if stop_daemon_on_close and self._started:
                logger.info(f"Stopping daemon for profile '{self.profile}'...")
                self._manager.stop(self.profile)

            self._closed = True

    def close(self, stop_daemon: bool = False):
        """
        Explicitly close the client.

        Args:
            stop_daemon: If True, stops the daemon. Otherwise, daemon continues running
                and will auto-stop after idle timeout (default: False).

        Note:
            The daemon may be shared with other clients or the CLI, so stopping it
            might affect other users. By default, we rely on the daemon's idle timeout.
        """
        self._cleanup(stop_daemon_on_close=stop_daemon)

    def __getattr__(self, name: str):
        """
        Proxy all method calls to the underlying Hindsight client.

        This allows HindsightEmbedded to expose all HindsightClient methods
        without manually wrapping each one.
        """
        # Ensure server is started before proxying
        self._ensure_started()

        # Get the attribute from the underlying client
        attr = getattr(self._client, name)

        # If it's a callable, wrap it to ensure server is started
        # (shouldn't be needed since _ensure_started already called, but defensive)
        if callable(attr):

            def wrapper(*args, **kwargs):
                self._ensure_started()
                return attr(*args, **kwargs)

            return wrapper

        return attr

    def __enter__(self):
        """Context manager entry - ensures server is started."""
        self._ensure_started()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stops the server."""
        self.close()

    def __del__(self):
        """Cleanup on garbage collection."""
        self._cleanup()

    @property
    def banks(self) -> BanksAPI:
        """
        Access bank management operations.

        Each method call ensures the daemon is running before executing.

        Example:
            ```python
            from hindsight import HindsightEmbedded

            embedded = HindsightEmbedded(profile="myapp", ...)

            # Create a bank
            embedded.banks.create(bank_id="test", name="Test Bank")

            # Set mission
            embedded.banks.set_mission(bank_id="test", mission="Help users")
            ```
        """
        if self._banks_api is None:
            self._banks_api = BanksAPI(self)
        return self._banks_api

    @property
    def mental_models(self) -> MentalModelsAPI:
        """
        Access mental model operations.

        Each method call ensures the daemon is running before executing.

        Example:
            ```python
            from hindsight import HindsightEmbedded

            embedded = HindsightEmbedded(profile="myapp", ...)

            # Create a mental model
            embedded.mental_models.create(
                bank_id="test",
                name="User Preferences",
                content="User prefers dark mode"
            )

            # List mental models
            models = embedded.mental_models.list(bank_id="test")
            ```
        """
        if self._mental_models_api is None:
            self._mental_models_api = MentalModelsAPI(self)
        return self._mental_models_api

    @property
    def directives(self) -> DirectivesAPI:
        """
        Access directive operations.

        Each method call ensures the daemon is running before executing.

        Example:
            ```python
            from hindsight import HindsightEmbedded

            embedded = HindsightEmbedded(profile="myapp", ...)

            # Create a directive
            embedded.directives.create(
                bank_id="test",
                name="Response Style",
                content="Always be concise and friendly"
            )

            # List directives
            directives = embedded.directives.list(bank_id="test")
            ```
        """
        if self._directives_api is None:
            self._directives_api = DirectivesAPI(self)
        return self._directives_api

    @property
    def memories(self) -> MemoriesAPI:
        """
        Access memory listing operations.

        Each method call ensures the daemon is running before executing.

        Example:
            ```python
            from hindsight import HindsightEmbedded

            embedded = HindsightEmbedded(profile="myapp", ...)

            # List memories
            memories = embedded.memories.list(
                bank_id="test",
                type="world",
                limit=50
            )
            ```
        """
        if self._memories_api is None:
            self._memories_api = MemoriesAPI(self)
        return self._memories_api

    @property
    def client(self) -> Hindsight:
        """
        Get the underlying Hindsight client for direct access.

        WARNING: Using this property directly means daemon restarts won't be
        handled automatically. Prefer using the API namespaces (banks, mental_models,
        directives, memories) or direct method calls on HindsightEmbedded instead.

        Ensures daemon is started before returning the client.

        Returns:
            Hindsight: The underlying client instance

        Example:
            ```python
            from hindsight import HindsightEmbedded

            embedded = HindsightEmbedded(profile="myapp", ...)

            # Direct access (not recommended - daemon crashes won't be handled)
            client = embedded.client
            banks = client.list_banks()  # If daemon crashes, this will fail
            ```
        """
        self._ensure_started()
        return self._client

    @property
    def url(self) -> str:
        """Get the daemon URL (starts daemon if needed)."""
        self._ensure_started()
        return self._manager.get_url(self.profile)

    @property
    def is_running(self) -> bool:
        """Check if the client is initialized."""
        return self._started and not self._closed and self._client is not None
