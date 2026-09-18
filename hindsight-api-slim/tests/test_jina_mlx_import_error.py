"""
Regression test for the JinaMLXCrossEncoder import-error handling.

See: https://github.com/vectorize-io/hindsight/issues/994

Before the fix, the bare `except ImportError` around `import mlx_lm` masked
*any* ImportError raised transitively during mlx_lm's own initialization
(e.g. transformers 5.x's _LazyModule race producing
`ImportError: cannot import name 'AutoTokenizer' from 'transformers'`),
replacing it with a misleading "install mlx" message.

These tests verify:
1. Anywhere that is not macOS arm64, initialize refuses at startup (mlx is an
   Apple Silicon framework and is no longer installed elsewhere — issue #4499).
2. A transitive ImportError raised from inside mlx_lm surfaces verbatim.
3. A genuine "package not installed" ImportError still produces the install hint.
"""

import contextlib
import sys
import types
from unittest.mock import patch

import pytest

# These stub mlx itself, but the code path under test still reaches transformers for
# the tokenizer — so without the local-ml extra the assertion sees
# "No module named 'transformers'" rather than the message it is checking.
pytest.importorskip("transformers", reason="the mlx path loads a tokenizer; needs the local-ml extra")

from hindsight_api.engine.cross_encoder import JinaMLXCrossEncoder


def _stub_mlx_modules() -> dict[str, types.ModuleType]:
    """Stub mlx + mlx.core so `import mlx.core` succeeds even without mlx installed."""
    import importlib.machinery

    mlx = types.ModuleType("mlx")
    mlx.__spec__ = importlib.machinery.ModuleSpec("mlx", loader=None)
    mlx_core = types.ModuleType("mlx.core")
    mlx_core.__spec__ = importlib.machinery.ModuleSpec("mlx.core", loader=None)
    mlx.core = mlx_core
    return {"mlx": mlx, "mlx.core": mlx_core}


@contextlib.contextmanager
def _pretend_platform(sys_platform: str, machine: str):
    """Run the block as if we were on the given platform.

    The import-error tests below exercise paths only reached on macOS arm64, so
    they have to claim to be there regardless of which runner they execute on.
    """
    with patch("sys.platform", sys_platform), patch("platform.machine", return_value=machine):
        yield


@pytest.mark.asyncio
async def test_initialize_refuses_to_start_off_apple_silicon():
    """The provider only exists on macOS arm64; anywhere else it must fail at startup."""
    encoder = JinaMLXCrossEncoder()

    with _pretend_platform("linux", "x86_64"):
        with pytest.raises(RuntimeError, match="requires Apple Silicon"):
            await encoder.initialize()


@pytest.mark.asyncio
async def test_initialize_surfaces_transitive_import_error():
    """A transformers-lazy-load-style failure must propagate, not be masked."""
    encoder = JinaMLXCrossEncoder()

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "mlx_lm" or name.startswith("mlx_lm."):
            raise ImportError("cannot import name 'AutoTokenizer' from 'transformers'")
        return real_import(name, *args, **kwargs)

    sys.modules.pop("mlx_lm", None)

    with _pretend_platform("darwin", "arm64"):
        with patch.dict(sys.modules, _stub_mlx_modules()):
            with patch("builtins.__import__", side_effect=fake_import):
                with pytest.raises(ImportError, match="AutoTokenizer"):
                    await encoder.initialize()


@pytest.mark.asyncio
async def test_initialize_reports_install_hint_when_mlx_missing():
    """A genuine 'package not installed' error still gets the friendly install hint."""
    encoder = JinaMLXCrossEncoder()

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "mlx_lm" or name.startswith("mlx_lm."):
            raise ImportError("No module named 'mlx_lm'")
        if name == "mlx" or name.startswith("mlx."):
            raise ImportError("No module named 'mlx'")
        return real_import(name, *args, **kwargs)

    sys.modules.pop("mlx_lm", None)
    sys.modules.pop("mlx", None)
    sys.modules.pop("mlx.core", None)

    with _pretend_platform("darwin", "arm64"):
        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(ImportError, match="mlx and mlx-lm are required"):
                await encoder.initialize()
