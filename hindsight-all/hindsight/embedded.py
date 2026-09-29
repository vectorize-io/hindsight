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

    Only the settings you pass explicitly are forwarded to the daemon. Anything
    left at its default is resolved by the daemon instead, in this order: the
    profile's .env file, then the parent process environment, then the daemon's
    own default. That is what lets a client constructed without credentials run
    against a profile (or a shell) that already has them configured, rather than
    overwriting them with placeholders (#3253).

    Args:
        profile: Profile name for data isolation (default: "default")
        llm_provider: LLM provider ("groq", "openai", "ollama", "gemini", "anthropic",
            "lmstudio", "github-copilot"). Omit to inherit; the server default is "openai".
        llm_api_key: API key for the LLM provider. Omit to inherit; pass "" to
            explicitly run without a key (local services that need no auth).
        llm_model: Model name to use. Omit to inherit; the server picks a default
            for the resolved provider.
        llm_base_url: Optional custom base URL for LLM API
        database_url: Optional database URL override (default: profile-specific pg0)
        idle_timeout: Deprecated and ignored — the daemon no longer auto-exits
            when idle. Still accepted so existing callers keep working.
        log_level: Daemon log level. Omit to inherit (daemon default: "info").
        ui: Whether to start the control plane web UI alongside the daemon (default: False)
        ui_port: Port for the UI. Defaults to daemon_port + 10000.
        ui_hostname: Hostname to bind the UI to. Defaults to "0.0.0.0".
    """

    def __init__(
        self,
        profile: str = "default",
        llm_provider: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        database_url: Optional[str] = None,
        idle_timeout: Optional[int] = None,
        log_level: Optional[str] = None,
        ui: bool = False,
        ui_port: Optional[int] = None,
        ui_hostname: str = "0.0.0.0",
    ):
        """
        Initialize the embedded client (daemon starts on first use).

        Every LLM/daemon setting left as None is omitted from the daemon config so
        the daemon resolves it from the profile .env, then the parent environment,
        then its own default.

        Args:
            profile: Profile name for data isolation
            llm_provider: LLM provider. Omit to inherit.
            llm_api_key: API key for the LLM provider. Omit to inherit; pass "" to
                explicitly run without a key.
            llm_model: Model name to use. Omit to inherit.
            llm_base_url: Optional custom base URL for LLM API
            database_url: Optional database URL override
            idle_timeout: Deprecated and ignored (the daemon no longer auto-exits).
            log_level: Daemon log level. Omit to inherit.
            ui: Whether to start the control plane web UI alongside the daemon
            ui_port: Port for the UI (defaults to daemon_port + 10000)
            ui_hostname: Hostname to bind the UI to (defaults to "0.0.0.0")
        """
        self.profile = profile

        # Build the config dict for the daemon (matches CLI format), omitting
        # every setting the caller did not specify. An omitted key is inherited
        # by the daemon from the profile .env / parent environment; sending a
        # placeholder instead would overwrite it, and _register_profile would
        # then persist that placeholder into the profile's .env file (#3253).
        # An explicit "" is still an override — that is how a local LLM service
        # with no authentication clears an inherited API key.
        self.config: dict[str, str] = {}

        if llm_provider is not None:
            self.config["HINDSIGHT_API_LLM_PROVIDER"] = llm_provider

        if llm_api_key is not None:
            self.config["HINDSIGHT_API_LLM_API_KEY"] = llm_api_key

        if llm_model is not None:
            self.config["HINDSIGHT_API_LLM_MODEL"] = llm_model

        if log_level is not None:
            self.config["HINDSIGHT_API_LOG_LEVEL"] = log_level

        if idle_timeout is not None:
            self.config["HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT"] = str(idle_timeout)

        if llm_base_url:
            self.config["HINDSIGHT_API_LLM_BASE_URL"] = llm_base_url

        if database_url:
            self.config["HINDSIGHT_EMBED_API_DATABASE_URL"] = database_url

        self._ui = ui
        self._ui_port = ui_port
        self._ui_hostname = ui_hostname

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
        """Ensure daemon is running (thread-safe), restarting if crashed."""
        if self._started and self._client is not None:
            if self._manager.is_running(self.profile):
                return
            # Daemon crashed — reset state and fall through to restart
            logger.warning(
                "Daemon for profile '%s' is no longer responsive, restarting...",
                self.profile,
            )
            try:
                self._client.close()
            except Exception:
                logger.debug("Error closing stale client", exc_info=True)
            self._client = None
            self._started = False

        with self._lock:
            # Double-check after acquiring lock
            if self._started and self._client is not None:
                if self._manager.is_running(self.profile):
                    return
                logger.warning(
                    "Daemon for profile '%s' is no longer responsive (lock path), restarting...",
                    self.profile,
                )
                try:
                    self._client.close()
                except Exception:
                    logger.debug("Error closing stale client", exc_info=True)
                self._client = None
                self._started = False

            if self._closed:
                raise RuntimeError(
                    "Cannot use HindsightEmbedded after it has been closed"
                )

            # Use embed manager interface for daemon management
            logger.info(f"Ensuring daemon is running for profile '{self.profile}'...")
            success = self._manager.ensure_running(self.config, self.profile)
            if not success:
                raise RuntimeError(
                    f"Failed to start daemon for profile '{self.profile}'"
                )

            # Get daemon URL and create client
            daemon_url = self._manager.get_url(self.profile)
            self._client = Hindsight(base_url=daemon_url)
            self._started = True
            logger.info(f"Connected to daemon at {daemon_url}")

            # Start UI if requested
            if self._ui:
                logger.info(f"Starting UI for profile '{self.profile}'...")
                ui_started = self._manager.start_ui(
                    self.profile, self._ui_port, self._ui_hostname
                )
                if not ui_started:
                    logger.warning(f"Failed to start UI for profile '{self.profile}'")

    def _cleanup(self, stop_daemon_on_close: bool = False):
        """
        Cleanup client resources (idempotent).

        Args:
            stop_daemon_on_close: If True, stops the daemon. Otherwise, daemon continues
                running (it will auto-stop after idle timeout).
        """
        if self._closed:
            return

        acquired = self._lock.acquire(timeout=5.0)
        if not acquired:
            # Lock is held by another thread (e.g. _ensure_started).
            # Mark closed to prevent new operations but skip shared-state
            # teardown — the daemon's idle timeout handles the rest.
            logger.warning(
                "Cleanup lock acquisition timed out for profile '%s'; "
                "marking closed, daemon will idle-stop on its own",
                self.profile,
            )
            self._closed = True
            return

        try:
            if self._closed:
                return

            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    logger.debug(
                        "Error closing client for profile '%s'",
                        self.profile,
                        exc_info=True,
                    )
                self._client = None

            # Stop UI if it was started
            if self._ui and self._started:
                logger.info(f"Stopping UI for profile '{self.profile}'...")
                self._manager.stop_ui(self.profile, self._ui_port)

            # Optionally stop daemon (daemon has idle timeout, so not required)
            if stop_daemon_on_close and self._started:
                logger.info(f"Stopping daemon for profile '{self.profile}'...")
                self._manager.stop(self.profile)

            self._closed = True
        finally:
            self._lock.release()

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
        # Ensure server is started (and restart if crashed) before proxying
        self._ensure_started()

        return getattr(self._client, name)

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

        Ensures daemon is started (and restarts it if it has crashed) before
        returning the client.

        Returns:
            Hindsight: The underlying client instance

        Example:
            ```python
            from hindsight import HindsightEmbedded

            embedded = HindsightEmbedded(profile="myapp", ...)

            client = embedded.client
            banks = client.list_banks()
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
        """Check if the client is initialized and the daemon is responsive."""
        return (
            self._started
            and not self._closed
            and self._client is not None
            and self._manager.is_running(self.profile)
        )

    @property
    def ui_url(self) -> str:
        """Get the UI URL for this profile."""
        return self._manager.get_ui_url(self.profile)
