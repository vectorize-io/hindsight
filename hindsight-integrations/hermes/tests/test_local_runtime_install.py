"""The local_embedded self-install of ``hindsight-all``.

Hermes core used to install that package through a special case keyed on the provider name
(``memory_setup._provider_pip_dependencies``); that special case is gone in current builds, so the
plugin's own guarded install is the only automatic path. Hermes >= 0.21.5 additionally replaced
``tools.lazy_deps`` with a relaunch shim whose ``install_specs`` exits the process, so these cover
both the guards that decide whether we install, and the guard that refuses a shim-shaped installer.
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


def test_relaunch_shim_is_never_called(monkeypatch):
    """A build whose ``tools.lazy_deps`` is the update-relaunch shim must not be called at all.

    Hermes 0.21.5 ships ``install_specs`` annotated ``-> NoReturn``: it calls
    ``_old_updater.stop_for_relaunch()``, which re-runs the entire update handoff and exits the
    calling process. From an availability probe that converts "optional runtime missing" (a
    degraded provider) into "the session dies", so the plugin must refuse it and degrade.
    """
    from typing import NoReturn

    calls = []

    def shim(specs, **kwargs) -> NoReturn:          # the exact shape Hermes 0.21.5 ships
        calls.append(list(specs))
        raise SystemExit(1)

    monkeypatch.setitem(sys.modules, "tools.lazy_deps",
                        SimpleNamespace(install_specs=shim))
    reason = "No module named 'hindsight'"
    monkeypatch.setattr(embedded, "_check_local_runtime", lambda: (False, reason))
    monkeypatch.setattr(sys.modules["plugins.memory"], "_get_active_memory_provider",
                        lambda: "hindsight")

    assert embedded._ensure_local_runtime() == (False, reason)
    assert calls == [], "the relaunch shim must never be invoked"


def test_missing_installer_degrades_with_the_manual_hint(monkeypatch):
    """No importable installer (module removed/renamed) → stay a degraded provider, never exit."""
    reason = "No module named 'hindsight'"
    monkeypatch.setitem(sys.modules, "tools.lazy_deps", None)   # import raises ImportError
    monkeypatch.setattr(embedded, "_check_local_runtime", lambda: (False, reason))
    monkeypatch.setattr(sys.modules["plugins.memory"], "_get_active_memory_provider",
                        lambda: "hindsight")

    assert embedded._ensure_local_runtime() == (False, reason)
    assert "hindsight-all" in embedded._local_runtime_hint(reason)


def test_installer_without_a_no_return_annotation_is_used(monkeypatch):
    """The guard keys on the DECLARATION, not on the module name — a real installer still installs."""
    recorder = _Recorder()
    monkeypatch.setitem(sys.modules, "tools.lazy_deps",
                        SimpleNamespace(install_specs=recorder))
    results = iter([(False, "No module named 'hindsight'"), (True, None)])
    monkeypatch.setattr(embedded, "_check_local_runtime", lambda: next(results))
    monkeypatch.setattr(sys.modules["plugins.memory"], "_get_active_memory_provider",
                        lambda: "hindsight")

    assert embedded._ensure_local_runtime() == (True, None)
    assert recorder.calls == [["hindsight-all"]]


def test_shim_is_detected_by_its_structural_fingerprint(monkeypatch):
    """Annotations can be stripped; the shim also imports the relaunch helper — either mark is enough."""
    calls = []

    def unannotated(specs, **kwargs):           # no NoReturn annotation at all
        calls.append(list(specs))
        return SimpleNamespace(ok=True, reason="", stderr="")

    monkeypatch.setitem(sys.modules, "tools.lazy_deps", SimpleNamespace(
        install_specs=unannotated, stop_for_relaunch=lambda **kwargs: None))
    reason = "No module named 'hindsight'"
    monkeypatch.setattr(embedded, "_check_local_runtime", lambda: (False, reason))
    monkeypatch.setattr(sys.modules["plugins.memory"], "_get_active_memory_provider",
                        lambda: "hindsight")

    assert embedded._ensure_local_runtime() == (False, reason)
    assert calls == []
