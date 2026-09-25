"""The local_embedded self-install of ``hindsight-all``.

Hermes core installs that package through a special case keyed on the provider name
(``memory_setup._provider_pip_dependencies``), which this plugin cannot carry out of the Hermes
tree. These cover the guards that decide whether we install it ourselves.
"""

import sys
from types import SimpleNamespace

import pytest

from hindsight_hermes import embedded


class _Recorder:
    """Stands in for ``tools.lazy_deps.install_specs``; records what was asked for."""

    def __init__(self, ok=True):
        self.calls = []
        self.ok = ok

    def __call__(self, specs, **kwargs):
        self.calls.append(list(specs))
        return SimpleNamespace(ok=self.ok, reason="blocked by policy", stderr="")


@pytest.fixture(autouse=True)
def _reset_attempt_flag():
    embedded._local_runtime_install_attempted = False
    yield
    embedded._local_runtime_install_attempted = False


def _patch(monkeypatch, *, probe_results, active="hindsight", installer=None):
    """Wire the probe to yield ``probe_results`` in order, plus the active provider + installer."""
    results = iter(probe_results)
    monkeypatch.setattr(embedded, "_check_local_runtime", lambda: next(results))
    # conftest registers these as synthetic sys.modules entries, so patch the module objects
    # directly — dotted-path monkeypatching walks attributes from the parent package.
    monkeypatch.setattr(sys.modules["plugins.memory"], "_get_active_memory_provider", lambda: active)
    recorder = installer or _Recorder()
    monkeypatch.setattr(sys.modules["tools.lazy_deps"], "install_specs", recorder)
    return recorder


def test_installs_hindsight_all_when_the_package_is_missing(monkeypatch):
    recorder = _patch(
        monkeypatch,
        probe_results=[(False, "No module named 'hindsight'"), (True, None)],
    )
    assert embedded._ensure_local_runtime() == (True, None)
    assert recorder.calls == [["hindsight-all"]]


def test_working_runtime_installs_nothing(monkeypatch):
    recorder = _patch(monkeypatch, probe_results=[(True, None)])
    assert embedded._ensure_local_runtime() == (True, None)
    assert recorder.calls == []


def test_unrelated_import_failure_installs_nothing(monkeypatch):
    """An old CPU raising inside NumPy is not something reinstalling fixes."""
    reason = "numpy: this CPU lacks AVX support"
    recorder = _patch(monkeypatch, probe_results=[(False, reason)])
    assert embedded._ensure_local_runtime() == (False, reason)
    assert recorder.calls == []


def test_inactive_provider_installs_nothing(monkeypatch):
    """A stale local_embedded config.json must not make a dashboard probe pull the ML stack down."""
    reason = "No module named 'hindsight'"
    recorder = _patch(monkeypatch, probe_results=[(False, reason)], active="mem0")
    assert embedded._ensure_local_runtime() == (False, reason)
    assert recorder.calls == []


def test_install_is_attempted_once_per_process(monkeypatch):
    reason = "No module named 'hindsight'"
    recorder = _patch(monkeypatch, probe_results=[(False, reason)] * 4, installer=_Recorder(ok=False))
    assert embedded._ensure_local_runtime() == (False, reason)
    assert embedded._ensure_local_runtime() == (False, reason)
    assert recorder.calls == [["hindsight-all"]]


def test_blocked_install_reports_the_original_reason(monkeypatch):
    """security.allow_lazy_installs=false: the caller still gets the hint, not a crash."""
    reason = "No module named 'hindsight'"
    _patch(monkeypatch, probe_results=[(False, reason)], installer=_Recorder(ok=False))
    available, got = embedded._ensure_local_runtime()
    assert (available, got) == (False, reason)
    assert "hindsight-all" in embedded._local_runtime_hint(got)
