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

    @classmethod
    def alembic_version_locations(cls) -> list[str]:
        """Directories of Alembic revisions this extension owns.

        A **classmethod**, and that is load-bearing rather than tidiness. The
        directories an extension ships are a static property of its package, so
        answering needs no instance — and the migration runner must not build one.
        Constructing an extension runs its ``__init__``, and a constructor is
        entitled to process-wide side effects: publishing a cache as a module-level
        singleton, registering a client, claiming a lock. Such a constructor is
        written assuming the application builds the extension ONCE, and an extra
        construction during migrations silently breaks that assumption — the
        instance serving requests keeps the object it built with, while the global
        now points at a newer one, so the two halves of a subsystem stop agreeing.
        Asking the class costs nothing and cannot do that.

        An extension that keeps state of its own — tables, or data in a store it
        fronts — needs that state migrated on the same lifecycle as core's, and
        with the same guarantees: ordered, applied once, recorded. Returning a
        path here puts the extension's revisions into the migration run, rather
        than leaving it to a bespoke command somebody has to remember.

        Each returned path is an Alembic **version directory** (the folder holding
        revision files), not an Alembic environment: there is one ``env.py``, core's,
        and it is what configures the schema and the connection. The extension
        supplies revisions only.

        The tree is **independent of core's**. It has its own base (``down_revision
        = None``), its own head, and its own row in ``alembic_version``. Core's DAG
        checks are unaffected, and an extension head can never block core's.

        **Alembic does not order independent branches relative to each other.** The
        migration runs ``upgrade heads``, which applies every branch, but the
        interleaving between unrelated trees is not defined and must not be relied
        on. A revision that needs a core table to exist first says so explicitly::

            depends_on = ("a1b2c3d4e5f6",)   # a core revision

        which orders it without making core its parent — so the extension keeps its
        own independent history.

        Give the tree's base revision a ``branch_labels`` naming the extension, so
        an operator can address it (``alembic upgrade <label>@head``) and so a
        stray head is attributable to whoever shipped it.

        Empty by default: an extension with no state of its own migrates nothing.
        """
        return []

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
