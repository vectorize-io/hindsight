"""local_embedded's in-process requirements and the out-of-process daemon handoff.

The server used to be imported into Hermes' own venv (``from hindsight import
HindsightEmbedded``, provided by ``hindsight-all``). It cannot be installed there —
hindsight-api-slim needs protobuf>=7.35.1 against mem0ai's and modal's protobuf<7.0, and
otel-semconv>=0.65b0 against mistralai's <0.61 — so the daemon runs as a separate process and
this venv keeps only a client plus the daemon manager.
"""

import sys
from types import SimpleNamespace

import pytest

from hindsight_hermes import embedded
from pathlib import Path


def test_the_probe_requires_only_the_client_and_the_daemon_manager(monkeypatch):
    """Probing for in-process ``hindsight`` / ``sentence_transformers`` is what dragged the whole
    server into Hermes' venv. Both are gone; neither may come back by accident."""
    asked = []
    monkeypatch.setattr(embedded.importlib, "import_module", lambda name: asked.append(name))

    assert embedded._check_local_runtime() == embedded.LocalRuntimeStatus(available=True)
    assert asked == ["hindsight_client", "hindsight_embed.daemon_embed_manager"]


def test_a_missing_plugin_package_reports_it_with_the_repair_hint(monkeypatch):
    def _boom(name):
        raise ModuleNotFoundError("No module named 'hindsight_embed'")

    monkeypatch.setattr(embedded.importlib, "import_module", _boom)
    status = embedded._check_local_runtime()
    assert status.available is False and "hindsight_embed" in status.reason
    hint = embedded._local_runtime_hint(status.reason)
    assert "hermes pm repair" in hint
    assert "hindsight-all" not in hint


def test_an_unrelated_import_failure_gets_no_install_hint(monkeypatch):
    """An old CPU raising inside NumPy is not a missing-package problem."""
    monkeypatch.setattr(
        embedded.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(RuntimeError("numpy: this CPU lacks AVX support")),
    )
    status = embedded._check_local_runtime()
    assert embedded._local_runtime_hint(status.reason) == ""


def _fake_embed_module(monkeypatch, *, running=True, url="http://127.0.0.1:54321", scrubs=True):
    """Stand in for hindsight_embed. *scrubs* says whether the installed release drops our
    PYTHONPATH from the daemon child itself (the release after 0.10.2 does)."""
    calls = {}

    class _Manager:
        def ensure_running(self, config, profile):
            calls["ensure_running"] = (config, profile)
            return running

        def get_url(self, profile):
            calls["get_url"] = profile
            return url

        def is_running(self, profile):
            return running

        def stop(self, profile):
            calls["stop"] = profile
            return True

    manager_module = SimpleNamespace()
    if scrubs:
        manager_module._strip_parent_interpreter_env = lambda env: env
    monkeypatch.setitem(
        sys.modules,
        "hindsight_embed",
        SimpleNamespace(get_embed_manager=lambda: _Manager(), daemon_embed_manager=manager_module),
    )
    monkeypatch.setitem(sys.modules, "hindsight_embed.daemon_embed_manager", manager_module)
    return calls


def test_starting_the_daemon_passes_the_config_through_and_returns_its_url(monkeypatch):
    calls = _fake_embed_module(monkeypatch)
    config = {"HINDSIGHT_API_LLM_PROVIDER": "ollama", "HINDSIGHT_API_LLM_MODEL": "gemma3:12b"}

    url = embedded._start_daemon(config, "hermes")

    assert url == "http://127.0.0.1:54321"
    assert calls["ensure_running"] == (config, "hermes")
    assert calls["get_url"] == "hermes"


def test_a_daemon_that_will_not_start_raises_naming_the_profile(monkeypatch):
    _fake_embed_module(monkeypatch, running=False)
    with pytest.raises(RuntimeError, match="hermes"):
        embedded._start_daemon({}, "hermes")


def test_daemon_status_and_stop_never_raise(monkeypatch):
    calls = _fake_embed_module(monkeypatch)
    assert embedded._daemon_is_running("hermes") is True
    assert embedded._stop_daemon("hermes") is True
    assert calls["stop"] == "hermes"

    monkeypatch.setitem(sys.modules, "hindsight_embed", SimpleNamespace())  # no get_embed_manager
    assert embedded._daemon_is_running("hermes") is False
    assert embedded._stop_daemon("hermes") is False


def test_the_plugin_never_depends_on_hindsight_all_again():
    """The package that cannot be installed next to Hermes must not reappear as a dependency, and
    the retired lazy installer must not be imported again. Checked on the parsed AST, so the
    docstrings explaining this history do not count as usage."""
    import ast
    import tomllib
    from pathlib import Path

    root = Path(embedded.__file__).resolve().parent
    declared = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert not [s for s in declared["project"]["dependencies"] if "hindsight-all" in s]

    # Hermes' runtime installers: the retired lazy_deps shim and PM's ensure_import. Dependencies
    # reach the environment through PM admission (pyproject.toml), never from plugin code.
    banned_modules = {"hindsight", "tools.lazy_deps", "pm"}

    def banned(module: str) -> bool:
        return module in banned_modules or module.startswith("pm.")

    for name in ("__init__.py", "embedded.py", "setup.py"):
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not banned(node.module or ""), f"{name}:{node.lineno} imports from {node.module}"
                if node.module == "tools":
                    assert all(a.name != "lazy_deps" for a in node.names), f"{name}:{node.lineno} imports lazy_deps"
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not banned(alias.name), f"{name}:{node.lineno} imports {alias.name}"


def test_the_start_worker_reconciles_the_profile_env_before_the_daemon_boots(monkeypatch, tmp_path):
    """Ordering is load-bearing: the daemon reads the profile .env at boot, so a drifted file must
    be rewritten (and a running daemon stopped) BEFORE the client is built — building it is what
    starts the daemon now. Booting first would pin the stale values for the life of the process.
    """
    from hindsight_hermes import HindsightMemoryProvider

    order = []
    provider = HindsightMemoryProvider()
    provider._config = {"profile": "orderingtest", "llm_provider": "ollama"}

    monkeypatch.setattr("hindsight_hermes._load_simple_env", lambda path: {"STALE": "1"})
    monkeypatch.setattr("hindsight_hermes._build_embedded_profile_env", lambda cfg: {"FRESH": "1"})
    monkeypatch.setattr("hindsight_hermes._may_rewrite_profile_env", lambda cfg: True)
    monkeypatch.setattr("hindsight_hermes._embedded_profile_env_path", lambda cfg: tmp_path / "p.env")

    def _fake_write(path, content):
        order.append("wrote env")
        Path(path).write_text(content)

    monkeypatch.setattr("hindsight_hermes._secure_write_profile_env", _fake_write)
    monkeypatch.setattr("hindsight_hermes._validate_profile_env_permissions", lambda path: None)
    monkeypatch.setattr("hindsight_hermes._daemon_is_running", lambda profile: True)
    monkeypatch.setattr("hindsight_hermes._stop_daemon", lambda profile: order.append("stopped daemon"))
    monkeypatch.setattr(type(provider), "_get_client", lambda self: order.append("built client"))

    provider._daemon_start_worker()

    assert order == ["wrote env", "stopped daemon", "built client"]
    written = (tmp_path / "p.env").read_text()
    assert "STALE=1" in written, "operator keys must be carried forward"
    assert "FRESH=1" in written, "builder keys must win"


def test_an_old_embed_starts_the_daemon_from_a_child_without_our_pythonpath(monkeypatch):
    """hindsight-embed <= 0.10.2 copies os.environ into the daemon, so Hermes' PYTHONPATH (its own
    3.14 generation) reaches a server that uvx may run on another Python — which then imports
    Hermes' pydantic and dies. Start it from a child that never had those variables."""
    calls = _fake_embed_module(monkeypatch, running=False, scrubs=False)
    monkeypatch.setenv("PYTHONPATH", "/hermes/venv/lib/python3.14/site-packages")
    monkeypatch.setenv("VIRTUAL_ENV", "/hermes/venv")
    monkeypatch.setenv("PATH", "/usr/bin")
    recorded = {}

    def _run(cmd, **kwargs):
        recorded["cmd"] = cmd
        recorded.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(embedded.subprocess, "run", _run)

    assert embedded._start_daemon({"HINDSIGHT_API_LLM_API_KEY": "sk-secret"}, "hermes") == "http://127.0.0.1:54321"
    assert "ensure_running" not in calls  # started by the child, not in this process
    assert "PYTHONPATH" not in recorded["env"] and "VIRTUAL_ENV" not in recorded["env"]
    assert recorded["env"]["PATH"] == "/usr/bin"  # everything else is inherited
    # the key travels on stdin; argv is visible to every user on the box
    assert "sk-secret" not in " ".join(recorded["cmd"])
    assert "sk-secret" in recorded["input"]


def test_a_new_embed_is_left_to_scrub_the_env_itself(monkeypatch):
    calls = _fake_embed_module(monkeypatch, scrubs=True)
    monkeypatch.setattr(
        embedded.subprocess, "run", lambda *a, **k: pytest.fail("must not spawn a helper when embed scrubs")
    )

    assert embedded._start_daemon({}, "hermes") == "http://127.0.0.1:54321"
    assert calls["ensure_running"] == ({}, "hermes")


def test_a_running_daemon_is_reused_instead_of_respawning_the_helper(monkeypatch):
    _fake_embed_module(monkeypatch, running=True, scrubs=False)
    monkeypatch.setattr(
        embedded.subprocess, "run", lambda *a, **k: pytest.fail("must not spawn a helper for a live daemon")
    )

    assert embedded._start_daemon({}, "hermes") == "http://127.0.0.1:54321"


def test_a_failing_helper_raises_naming_the_profile(monkeypatch):
    _fake_embed_module(monkeypatch, running=False, scrubs=False)
    monkeypatch.setattr(
        embedded.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="ModuleNotFoundError: pydantic_core"),
    )

    with pytest.raises(RuntimeError, match="hermes"):
        embedded._start_daemon({}, "hermes")


def test_an_unreadable_embed_defaults_to_the_safe_path(monkeypatch):
    """If we cannot tell whether the installed embed scrubs the env, assume it does not: the child
    is correct either way, while skipping it on a release that needs it breaks the daemon."""
    monkeypatch.setitem(sys.modules, "hindsight_embed", SimpleNamespace())  # no daemon_embed_manager
    assert embedded._embed_scrubs_parent_env() is False


def test_the_binary_probe_checks_the_scripts_dir_the_manager_uses(monkeypatch):
    """No hindsight-api installed means the daemon comes down through uvx first. The probe must
    look where the manager looks — the sysconfig scripts path — not only beside the interpreter."""
    monkeypatch.setattr(embedded.shutil, "which", lambda name, path=None: None)
    assert embedded._installed_api_binary_exists() is False

    scripts = embedded.sysconfig.get_path("scripts")
    monkeypatch.setattr(
        embedded.shutil, "which", lambda name, path=None: "/s/hindsight-api" if path == scripts else None
    )
    assert embedded._installed_api_binary_exists() is True
