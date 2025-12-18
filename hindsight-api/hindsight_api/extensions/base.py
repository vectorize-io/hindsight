"""Base Extension class for all Hindsight extensions."""

from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hindsight_api.extensions.context import ExtensionContext


class Extension(ABC):
    """
    Base class for all Hindsight extensions.

    Extensions are loaded via environment variables and receive configuration
    from prefixed environment variables.

    Example:
        HINDSIGHT_API_MY_EXTENSION=mypackage.ext:MyExtension
        HINDSIGHT_API_MY_SOME_CONFIG=value

        The extension receives: {"some_config": "value"}

    Extensions also receive an ExtensionContext that provides a controlled API
    for interacting with the system (e.g., running migrations for tenant schemas).
    """

    def __init__(self, config: dict[str, str]):
        """
        Initialize the extension with configuration.

        Args:
            config: Dictionary of configuration values from environment variables.
                    Keys are lowercased with the prefix stripped.
        """
        self.config = config
        self._context: "ExtensionContext | None" = None

    def set_context(self, context: "ExtensionContext") -> None:
        """
        Set the extension context.

        Called by the extension loader after instantiation.
        Extensions should not call this directly.

        Args:
            context: The ExtensionContext providing system APIs.
        """
        self._context = context

    @property
    def context(self) -> "ExtensionContext":
        """
        Get the extension context.

        Returns:
            The ExtensionContext providing system APIs.

        Raises:
            RuntimeError: If context has not been set yet.
        """
        if self._context is None:
            raise RuntimeError(
                "Extension context not set. Context is available after the extension is loaded by the system."
            )
        return self._context

    async def on_startup(self) -> None:
        """
        Called when the application starts.

        Override to perform initialization tasks like connecting to external services.
        """
        pass

    async def on_shutdown(self) -> None:
        """
        Called when the application shuts down.

        Override to perform cleanup tasks like closing connections.
        """
        pass
