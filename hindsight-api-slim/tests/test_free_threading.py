"""The API surface must import without re-enabling the GIL on a free-threaded build.

A free-threaded CPython silently turns the GIL back ON the moment it imports a C
extension that has not declared ``Py_MOD_GIL_NOT_USED`` -- the only signal is a
RuntimeWarning. A single new module-scope import of the wrong dependency therefore
reverts the whole server to single-threaded execution while every test still passes
and the process still serves traffic.

This test is the guard for that. It runs in a subprocess because the GIL can only be
re-enabled once per interpreter, so an import that already happened in the pytest
parent would mask the regression here.

Skipped on a normal (GIL-enabled) build, so it is inert on the 3.11 CI matrix and
only bites on a ``python3.14t`` job.
"""

import os
import subprocess
import sys
import sysconfig

import pytest

FREE_THREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))

pytestmark = pytest.mark.skipif(
    not FREE_THREADED,
    reason="only meaningful on a free-threaded (Py_GIL_DISABLED) interpreter",
)

# Importing this pulls the whole engine: MemoryEngine, the retain pipeline, every
# provider module and the parser registry -- i.e. everything the server imports before
# it serves a request.
_PROBE = """
import sys
import hindsight_api.api.http  # noqa: F401
print("GIL_ENABLED" if sys._is_gil_enabled() else "GIL_DISABLED")
"""


def _run(env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, **(env_extra or {})}
    return subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, env=env)


def test_importing_the_api_does_not_re_enable_the_gil():
    """The plain import must leave free-threading intact."""
    result = _run()
    assert result.returncode == 0, f"probe failed:\n{result.stderr}"
    assert "GIL_DISABLED" in result.stdout, (
        "Importing hindsight_api.api.http re-enabled the GIL. A dependency without a "
        "free-threaded build was imported at module scope; stderr names it:\n"
        f"{result.stderr}"
    )


def test_gil_warning_is_promotable_to_an_error():
    """``PYTHONWARNINGS`` turns the re-enable warning fatal, naming the exact import.

    This is the form to use in CI and locally: it fails at the offending import with a
    full traceback, instead of at some later point where the cause is no longer visible.
    Unrelated RuntimeWarnings are left alone by the message-scoped filter.
    """
    result = _run({"PYTHONWARNINGS": "error:The global interpreter lock:RuntimeWarning"})
    assert result.returncode == 0, (
        f"A dependency re-enabled the GIL during import; the traceback names it:\n{result.stderr}"
    )
    assert "GIL_DISABLED" in result.stdout
