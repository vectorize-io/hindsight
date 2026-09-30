"""The single-read env helpers in config.py: unset and empty both mean "not set"."""

import pytest

from hindsight_api.config import _env_float, _env_int, _env_int_or, _env_str_list

VAR = "HINDSIGHT_API_TEST_ENV_READER"


@pytest.mark.parametrize("raw", [None, ""])
def test_unset_or_empty_falls_back(monkeypatch, raw):
    if raw is None:
        monkeypatch.delenv(VAR, raising=False)
    else:
        monkeypatch.setenv(VAR, raw)
    assert _env_int(VAR) is None
    assert _env_float(VAR) is None
    assert _env_str_list(VAR) is None
    assert _env_int_or(VAR, 7) == 7
    assert _env_int_or(VAR, None) is None


def test_set_values_are_converted(monkeypatch):
    monkeypatch.setenv(VAR, "3")
    assert _env_int(VAR) == 3
    assert _env_int_or(VAR, 7) == 3
    assert _env_float(VAR) == 3.0
    monkeypatch.setenv(VAR, "a, b")
    assert _env_str_list(VAR) == ["a", "b"]


def test_bad_value_raises_instead_of_falling_back(monkeypatch):
    monkeypatch.setenv(VAR, "not-a-number")
    with pytest.raises(ValueError):
        _env_int(VAR)
