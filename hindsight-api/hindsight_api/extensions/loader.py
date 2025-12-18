"""Extension loader utilities."""

import importlib
import logging
import os
from typing import TYPE_CHECKING, TypeVar

from hindsight_api.extensions.base import Extension

if TYPE_CHECKING:
    from hindsight_api.extensions.context import ExtensionContext

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Extension)


class ExtensionLoadError(Exception):
    """Raised when an extension fails to load."""

    pass


def load_extension(
    prefix: str,
    base_class: type[T],
    env_prefix: str = "HINDSIGHT_API",
    context: "ExtensionContext | None" = None,
) -> T | None:
    """
    Load an extension from environment variable configuration.

    The extension class is specified via {env_prefix}_{prefix}_EXTENSION environment
    variable in the format "module.path:ClassName".

    Configuration for the extension is collected from all environment variables
    matching {env_prefix}_{prefix}_* (excluding the EXTENSION variable itself).

    Args:
        prefix: The extension prefix (e.g., "OPERATION_VALIDATOR").
        base_class: The base class that the extension must inherit from.
        env_prefix: The environment variable prefix (default: "HINDSIGHT_API").
        context: Optional ExtensionContext to provide system APIs to the extension.

    Returns:
        An instance of the extension, or None if not configured.

    Raises:
        ExtensionLoadError: If the extension fails to load or validate.

    Example:
        HINDSIGHT_API_OPERATION_VALIDATOR_EXTENSION=mypackage.validators:MyValidator
        HINDSIGHT_API_OPERATION_VALIDATOR_MAX_REQUESTS=100

        ext = load_extension("OPERATION_VALIDATOR", OperationValidatorExtension)
        # ext.config == {"max_requests": "100"}
    """
    env_var = f"{env_prefix}_{prefix}_EXTENSION"
    ext_path = os.getenv(env_var)

    if not ext_path:
        logger.debug(f"No extension configured for {env_var}")
        return None

    logger.info(f"Loading extension from {env_var}={ext_path}")

    # Parse "module.path:ClassName"
    if ":" not in ext_path:
        raise ExtensionLoadError(f"Invalid extension path '{ext_path}'. Expected format: 'module.path:ClassName'")

    module_path, class_name = ext_path.rsplit(":", 1)

    # Import the module
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ExtensionLoadError(f"Failed to import extension module '{module_path}': {e}") from e

    # Get the class
    try:
        ext_class = getattr(module, class_name)
    except AttributeError as e:
        raise ExtensionLoadError(f"Extension class '{class_name}' not found in module '{module_path}'") from e

    # Validate inheritance
    if not isinstance(ext_class, type) or not issubclass(ext_class, base_class):
        raise ExtensionLoadError(f"Extension class '{ext_class.__name__}' must inherit from '{base_class.__name__}'")

    # Collect configuration from environment variables
    config = _collect_config(env_prefix, prefix)

    logger.info(f"Loaded extension {ext_class.__name__} with config keys: {list(config.keys())}")

    # Instantiate the extension
    try:
        extension = ext_class(config)
    except Exception as e:
        raise ExtensionLoadError(f"Failed to instantiate extension '{ext_class.__name__}': {e}") from e

    # Set the context if provided
    if context is not None:
        extension.set_context(context)
        logger.debug(f"Set context on extension {ext_class.__name__}")

    return extension


def _collect_config(env_prefix: str, prefix: str) -> dict[str, str]:
    """
    Collect configuration from environment variables.

    Collects all variables matching {env_prefix}_{prefix}_* except for
    {env_prefix}_{prefix}_EXTENSION, strips the prefix, and lowercases keys.
    """
    config = {}
    full_prefix = f"{env_prefix}_{prefix}_"
    extension_var = f"{full_prefix}EXTENSION"

    for key, value in os.environ.items():
        if key.startswith(full_prefix) and key != extension_var:
            # Strip prefix and lowercase the key
            config_key = key[len(full_prefix) :].lower()
            config[config_key] = value

    return config
