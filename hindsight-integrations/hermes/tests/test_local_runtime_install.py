"""``embedded._ensure_local_runtime`` no longer self-installs ``hindsight-all``.

It used to call Hermes's ``tools.lazy_deps.install_specs`` to self-install the missing package.
That helper has since been reduced, on the Hermes side, to an old-updater relaunch stub that
raises ``SystemExit`` instead of installing anything (NousResearch/hermes-agent's
``tools/lazy_deps.py``, guarded by ``scripts/ci/check_lazy_deps_imports.py``). Calling it from
here killed the whole Hermes gateway on every session/cron turn that touched memory while the
package was missing — each relaunch redoing a full desktop/TUI/web rebuild — producing a
crash-restart loop instead of ever installing the dependency. These cover the graceful-
degradation contract that replaced the self-install attempt.
"""

import sys

import pytest
from hindsight_hermes import embedded


@pytest.fixture(autouse=True)
def _reset_attempt_flag():
    embedded._local_runtime_install_attempted = False
    yield
    embedded._local_runtime_install_attempted = False


def _patch(monkeypatch, *, probe_results, active="hindsight"):
    """Wire the probe to yield ``probe_results`` in order, plus the active provider."""
    results = iter(probe_results)
    monkeypatch.setattr(embedded, "_check_local_runtime", lambda: next(results))
    # conftest registers this as a synthetic sys.modules entry, so patch the module object
    # directly — dotted-path monkeypatching walks attributes from the parent package.
    monkeypatch.setattr(sys.modules["plugins.memory"], "_get_active_memory_provider", lambda: active)


def test_working_runtime_reports_available(monkeypatch):
    _patch(monkeypatch, probe_results=[(True, None)])
    assert embedded._ensure_local_runtime() == (True, None)


def test_missing_package_degrades_gracefully_instead_of_crashing(monkeypatch, caplog):
    """The old behaviour called into Hermes's dead relaunch stub here, which raised SystemExit
    and killed the gateway. It must now just report unavailable and log the install hint."""
    reason = "No module named 'hindsight'"
    _patch(monkeypatch, probe_results=[(False, reason)])
    with caplog.at_level("WARNING"):
        assert embedded._ensure_local_runtime() == (False, reason)
    assert "hindsight-all" in caplog.text


def test_unrelated_import_failure_logs_nothing(monkeypatch, caplog):
    """An old CPU raising inside NumPy is not something reinstalling fixes, so it gets no hint
    and no warning here (agent_init's own unavailable-provider warning still covers it)."""
    reason = "numpy: this CPU lacks AVX support"
    _patch(monkeypatch, probe_results=[(False, reason)])
    with caplog.at_level("WARNING"):
        assert embedded._ensure_local_runtime() == (False, reason)
    assert caplog.text == ""


def test_inactive_provider_logs_nothing(monkeypatch, caplog):
    """A stale local_embedded config.json must not make a dashboard availability probe log this."""
    reason = "No module named 'hindsight'"
    _patch(monkeypatch, probe_results=[(False, reason)], active="mem0")
    with caplog.at_level("WARNING"):
        assert embedded._ensure_local_runtime() == (False, reason)
    assert caplog.text == ""


def test_warning_logged_once_per_process(monkeypatch, caplog):
    reason = "No module named 'hindsight'"
    _patch(monkeypatch, probe_results=[(False, reason)] * 4)
    with caplog.at_level("WARNING"):
        assert embedded._ensure_local_runtime() == (False, reason)
        logged_after_first = len(caplog.records)
        assert embedded._ensure_local_runtime() == (False, reason)
    assert len(caplog.records) == logged_after_first  # nothing new logged on the second call


def test_unavailable_reason_still_surfaces_the_install_hint(monkeypatch):
    """``_local_runtime_hint`` (used by ``HindsightMemoryProvider.unavailable_reason``) still
    gives the manual-install instruction even though nothing here acts on it automatically."""
    reason = "No module named 'hindsight'"
    assert "hindsight-all" in embedded._local_runtime_hint(reason)
