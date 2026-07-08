"""
Tests for the server-level default mental-model refresh mode.

``HINDSIGHT_API_DEFAULT_MENTAL_MODEL_REFRESH_MODE`` lets a deployment choose
whether mental models that don't pin a mode in their trigger refresh in
``full`` (regenerate from scratch — the historical default) or ``delta``
(surgical edit against an existing baseline) mode. An explicit ``trigger.mode``
always wins over this default; that resolution lives in the engine and is
covered by the engine tests.
"""

import os

import pytest

from hindsight_api.config import (
    DEFAULT_MENTAL_MODEL_REFRESH_MODE,
    HindsightConfig,
)

_ENV = "HINDSIGHT_API_DEFAULT_MENTAL_MODEL_REFRESH_MODE"


@pytest.fixture(autouse=True)
def _restore_env():
    """Isolate each test from the process-level value of the env var."""
    original = os.environ.get(_ENV)
    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    yield
    if original is None:
        os.environ.pop(_ENV, None)
    else:
        os.environ[_ENV] = original


def _resolve(value: str | None) -> str:
    if value is None:
        os.environ.pop(_ENV, None)
    else:
        os.environ[_ENV] = value
    return HindsightConfig.from_env().default_mental_model_refresh_mode


def test_default_is_full():
    """Unset env preserves historical behaviour."""
    assert _resolve(None) == "full"
    assert DEFAULT_MENTAL_MODEL_REFRESH_MODE == "full"


def test_delta_opt_in():
    assert _resolve("delta") == "delta"


def test_full_explicit():
    assert _resolve("full") == "full"


@pytest.mark.parametrize("raw", ["DELTA", "Delta", " delta ", "\tdelta\n"])
def test_case_and_whitespace_insensitive(raw):
    assert _resolve(raw) == "delta"


@pytest.mark.parametrize("raw", ["", "bogus", "incremental", "none", "0"])
def test_invalid_falls_back_to_full(raw):
    """A typo must never silently flip every model's refresh behaviour."""
    assert _resolve(raw) == "full"
