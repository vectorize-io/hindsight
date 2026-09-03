"""Keep a free-threaded interpreter actually free-threaded.

On a ``Py_GIL_DISABLED`` build, CPython re-enables the GIL for the whole process the
moment it imports a C extension that has not declared ``Py_MOD_GIL_NOT_USED``. The only
signal is a ``RuntimeWarning``. Nothing crashes, nothing degrades visibly -- the server
starts, serves traffic, and passes its tests, having quietly reverted to single-threaded
execution. One new module-scope import of the wrong dependency is enough.

So this module makes that failure loud. It has to run *before* the offending import, which
is why ``hindsight_api/__init__.py`` calls it eagerly, next to
``apply_default_thread_limits()`` and for the same class of reason: both configure how the
process executes, and both are worthless once the libraries they govern have loaded.

Controlled by ``HINDSIGHT_API_FREE_THREADING``:

    strict  (default on a free-threaded build)
            The re-enable warning becomes an exception, so the import that caused it
            raises with the module and full import chain in the traceback. A GIL that is
            somehow already on raises at startup.
    warn    Log it and continue. For bringing up a deployment whose dependencies are not
            all free-threading-ready yet.
    off     No guard at all. Used by the migration child process, which imports psycopg2
            on purpose (see ``migrations.run_migrations``).

On a normal GIL build every mode is a no-op, so this is inert on Python 3.11.
"""

import logging
import os
import sys
import sysconfig
import warnings

logger = logging.getLogger(__name__)

ENV_FREE_THREADING = "HINDSIGHT_API_FREE_THREADING"

#: Matches the start of CPython's own message; scoped to the message on purpose so that
#: unrelated RuntimeWarnings from any library are left alone.
_GIL_WARNING_PREFIX = "The global interpreter lock"


class GilReenabledError(RuntimeError):
    """The GIL is enabled on a build that was supposed to be free-threaded."""


def is_free_threaded_build() -> bool:
    """True on a ``python3.14t``-style interpreter, regardless of the GIL's current state."""
    return bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def gil_enabled() -> bool:
    """Whether the GIL is currently active. Always True on a normal build."""
    return getattr(sys, "_is_gil_enabled", lambda: True)()


def mode() -> str:
    """Resolved guard mode: ``strict``, ``warn`` or ``off``."""
    raw = os.environ.get(ENV_FREE_THREADING, "").strip().lower()
    if raw in ("strict", "warn", "off"):
        return raw
    if raw:
        logger.warning("Ignoring unknown %s=%r; using the default.", ENV_FREE_THREADING, raw)
    return "strict" if is_free_threaded_build() else "off"


def enforce() -> None:
    """Install the guard. Safe to call more than once; a no-op on a GIL build."""
    if not is_free_threaded_build():
        return

    current = mode()
    if current == "off":
        return

    if current == "strict":
        # Turn the re-enable warning into an exception. This is what makes the failure
        # point at the offending import rather than at some later, unrelated symptom.
        warnings.filterwarnings("error", message=_GIL_WARNING_PREFIX, category=RuntimeWarning)
    else:
        warnings.filterwarnings("always", message=_GIL_WARNING_PREFIX, category=RuntimeWarning)

    if not gil_enabled():
        return

    # Already lost before the guard was installed -- something imported ahead of us.
    message = (
        "The GIL is enabled on a free-threaded build. A C extension without "
        "free-threading support was imported before the guard was installed, so this "
        "process is running single-threaded. Re-run with "
        'PYTHONWARNINGS="error:The global interpreter lock:RuntimeWarning" to get a '
        f"traceback naming the import, or set {ENV_FREE_THREADING}=warn to continue anyway."
    )
    if current == "strict":
        raise GilReenabledError(message)
    logger.warning(message)
