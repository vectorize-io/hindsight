"""Extension loader utilities."""

import importlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from hindsight_api.extensions.base import Extension

if TYPE_CHECKING:
    from hindsight_api.extensions.context import ExtensionContext

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Extension)


class ExtensionLoadError(Exception):
    """Raised when an extension fails to load."""

    pass


def _resolve_extension_class(ext_path: str, base_class: type[T]) -> type[T]:
    """Import and validate the class named by ``module.path:ClassName``, no instance.

    Split out so a caller that only needs the class does not have to construct one:
    a constructor may have process-wide side effects, and running it to answer a
    question about the package is how those get run more times than their author
    expected.
    """
    if ":" not in ext_path:
        raise ExtensionLoadError(f"Invalid extension path '{ext_path}'. Expected format: 'module.path:ClassName'")

    module_path, class_name = ext_path.rsplit(":", 1)

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ExtensionLoadError(f"Failed to import extension module '{module_path}': {e}") from e

    try:
        ext_class = getattr(module, class_name)
    except AttributeError as e:
        raise ExtensionLoadError(f"Extension class '{class_name}' not found in module '{module_path}'") from e

    if not isinstance(ext_class, type) or not issubclass(ext_class, base_class):
        raise ExtensionLoadError(f"Extension class '{ext_class.__name__}' must inherit from '{base_class.__name__}'")

    return ext_class


def resolve_configured_extension_class(
    prefix: str,
    base_class: type[T],
    env_prefix: str = "HINDSIGHT_API",
) -> "type[T] | None":
    """The configured extension CLASS for ``prefix``, without instantiating it.

    ``None`` when nothing is configured, exactly as :func:`load_extension` returns
    ``None``. Use this for anything that asks an extension about itself rather than
    asking it to do work.
    """
    ext_path = os.getenv(f"{env_prefix}_{prefix}_EXTENSION")
    if not ext_path:
        return None
    return _resolve_extension_class(ext_path, base_class)


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

    ext_class = _resolve_extension_class(ext_path, base_class)

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


@dataclass(frozen=True)
class ExtensionKind:
    """One kind of extension: its env prefix and the base class it must inherit from."""

    prefix: str
    module_path: str
    class_name: str


#: Every extension kind. Used to ask each configured extension for the Alembic
#: revisions it owns.
#:
#: Listed here rather than discovered, because the set is small, fixed, and the
#: cost of missing one is silent: an extension whose revisions are never collected
#: simply never migrates, and nothing fails. `test_extension_kinds_covers_every_base`
#: is the guard against that — it fails if a new base class is added and not listed.
EXTENSION_KINDS: tuple[ExtensionKind, ...] = (
    ExtensionKind("TENANT", "hindsight_api.extensions.tenant", "TenantExtension"),
    ExtensionKind("MEMORIES", "hindsight_api.engine.memories.base", "MemoriesExtension"),
    ExtensionKind("OPERATION_VALIDATOR", "hindsight_api.extensions.operation_validator", "OperationValidatorExtension"),
    ExtensionKind("MEMORY_DEFENSE", "hindsight_api.extensions.memory_defense", "MemoryDefenseExtension"),
    ExtensionKind("HTTP", "hindsight_api.extensions.http", "HttpExtension"),
    ExtensionKind("MCP", "hindsight_api.extensions.mcp", "MCPExtension"),
)


def collect_alembic_version_locations(env_prefix: str = "HINDSIGHT_API") -> list[str]:
    """Every Alembic version directory the configured extensions own.

    Asked once per migration run and passed to Alembic as ``version_locations``
    alongside core's own, so an extension's revisions are applied on the same
    lifecycle, under the same advisory lock, and recorded in the same
    ``alembic_version`` table as core's.

    Loading an extension here must never break migrations. An extension that
    cannot be imported, or that raises while answering, is skipped with a warning:
    the alternative is that a misconfigured extension makes the database
    unmigratable, which is a far worse failure than that extension's own state
    being out of date. A path that does not exist is skipped for the same reason —
    Alembic treats a missing version location as a hard error.

    Paths are de-duplicated in order, since two extension kinds may be served by
    one class (a package that provides both a tenant and a memories extension
    would otherwise contribute its tree twice, and Alembic rejects a duplicate
    version location).
    """
    found: list[str] = []
    for kind in EXTENSION_KINDS:
        try:
            base = getattr(importlib.import_module(kind.module_path), kind.class_name)
            # The CLASS, never an instance. Constructing an extension to ask which
            # directories it ships runs its `__init__`, and a constructor may publish
            # process-wide state written on the assumption the application builds it
            # once; a second construction here would leave the instance serving
            # requests and that global pointing at different objects.
            ext_class = resolve_configured_extension_class(kind.prefix, base, env_prefix=env_prefix)
        except Exception as e:  # noqa: BLE001 - a broken extension must not block migrations
            logger.warning("Could not load %s extension to collect migrations: %s", kind.prefix, e)
            continue
        if ext_class is None:
            continue
        try:
            locations = ext_class.alembic_version_locations()
        except Exception as e:  # noqa: BLE001 - same reason
            logger.warning("%s extension failed to report its migrations: %s", kind.prefix, e)
            continue
        # `or []` is load-bearing: this loop is outside the try above, so a third-party
        # extension returning None would otherwise raise and fail the whole migration.
        for loc in locations or []:
            if not Path(loc).is_dir():
                logger.warning(
                    "%s extension names Alembic version location %r, which is not a directory; skipping.",
                    kind.prefix,
                    loc,
                )
                continue
            found.append(str(Path(loc).resolve()))
            logger.info("%s extension contributes Alembic revisions from %s", kind.prefix, loc)
    return list(dict.fromkeys(found))
