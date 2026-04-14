"""
Regression test for the JinaMLXCrossEncoder import-error handling.

See: https://github.com/vectorize-io/hindsight/issues/994

Before the fix, the bare `except ImportError` around `import mlx_lm` masked
*any* ImportError raised transitively during mlx_lm's own initialization
(e.g. transformers 5.x's _LazyModule race producing
`ImportError: cannot import name 'AutoTokenizer' from 'transformers'`),
replacing it with a misleading "install mlx" message.

These tests verify:
1. A transitive ImportError raised from inside mlx_lm surfaces verbatim.
2. A genuine "package not installed" ImportError still produces the install hint.
"""

import sys
import types
from unittest.mock import patch

import pytest

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

    with patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(ImportError, match="mlx and mlx-lm are required"):
            await encoder.initialize()
