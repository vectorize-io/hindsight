"""local_embedded's in-process requirements and the out-of-process daemon handoff.

The server used to be imported into Hermes' own venv (``from hindsight import
HindsightEmbedded``, provided by ``hindsight-all``). It cannot be installed there —
hindsight-api-slim needs protobuf>=7.35.1 against mem0ai's and modal's protobuf<7.0, and
otel-semconv>=0.65b0 against mistralai's <0.61 — so the daemon runs as a separate process and
this venv keeps only a client plus the daemon manager.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hindsight_hermes import embedded


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

    monkeypatch.setattr("hindsight_hermes._profile_env_drifted", lambda cfg: True)
    monkeypatch.setattr("hindsight_hermes._may_rewrite_profile_env", lambda cfg: True)
    monkeypatch.setattr("hindsight_hermes._embedded_profile_env_path", lambda cfg: tmp_path / "p.env")
    monkeypatch.setattr("hindsight_hermes._materialize_embedded_profile_env", lambda cfg: order.append("rewrote env"))
    monkeypatch.setattr("hindsight_hermes._daemon_is_running", lambda profile: True)
    monkeypatch.setattr("hindsight_hermes._stop_daemon", lambda profile: order.append("stopped daemon"))
    monkeypatch.setattr(type(provider), "_get_client", lambda self: order.append("built client"))

    provider._daemon_start_worker()

    assert order == ["rewrote env", "stopped daemon", "built client"]


def test_an_old_embed_starts_the_daemon_from_a_child_without_our_pythonpath(monkeypatch):
    """hindsight-embed <= 0.10.2 copies os.environ into the daemon, so Hermes' PYTHONPATH (its own
    3.14 generation) reaches a server that uvx may run on another Python — which then imports
    Hermes' pydantic and dies. Start it from a child that drops those variables before the spawn."""
    calls = _fake_embed_module(monkeypatch, running=False, scrubs=False)
    monkeypatch.setenv("PYTHONPATH", "/hermes/venv/lib/python3.14/site-packages")
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setattr(embedded, "_managed_uvx_dir", lambda path: None)
    recorded = {}

    def _run(cmd, **kwargs):
        recorded["cmd"] = cmd
        recorded.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(embedded.subprocess, "run", _run)

    assert embedded._start_daemon({"HINDSIGHT_API_LLM_API_KEY": "sk-secret"}, "hermes") == "http://127.0.0.1:54321"
    assert "ensure_running" not in calls  # started by the child, not in this process
    assert recorded["env"]["PATH"] == "/usr/bin"  # everything else is inherited
    assert set(recorded["cmd"][4:]) == embedded._PARENT_INTERPRETER_ENV  # the helper drops these
    # the key travels on stdin; argv is visible to every user on the box
    assert "sk-secret" not in " ".join(recorded["cmd"])
    assert "sk-secret" in recorded["input"]


def test_the_helper_imports_embed_through_pythonpath_and_hides_it_from_the_daemon(tmp_path, monkeypatch):
    """#5006: where PYTHONPATH is the only way the plugin's packages are found, the helper must keep
    it for its own imports, then drop it before hindsight-embed copies os.environ into the daemon."""
    package = tmp_path / "hindsight_embed"
    package.mkdir()
    seen = tmp_path / "daemon_env.json"
    (package / "__init__.py").write_text(
        "import json, os\n"
        "class _Manager:\n"
        "    def ensure_running(self, config, profile):\n"
        f"        open({str(seen)!r}, 'w').write(json.dumps([profile, config, 'PYTHONPATH' in os.environ]))\n"
        "        return True\n"
        "def get_embed_manager():\n"
        "    return _Manager()\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))

    assert embedded._start_daemon_in_clean_child({"HINDSIGHT_API_LLM_MODEL": "m"}, "hermes") is True
    assert json.loads(seen.read_text()) == ["hermes", {"HINDSIGHT_API_LLM_MODEL": "m"}, False]


def test_the_helper_finds_the_uvx_hermes_installed_when_path_has_none(tmp_path, monkeypatch):
    """#5192: a package-manager Hermes keeps uvx under ~/.hermes/tools/uv-*/, off PATH."""
    uv_dir = tmp_path / ".hermes" / "tools" / "uv-0.12.3-darwin-arm64"
    uv_dir.mkdir(parents=True)
    (uv_dir / "uvx").write_text("#!/bin/sh\n")
    (uv_dir / "uvx").chmod(0o755)
    monkeypatch.setattr(embedded.Path, "home", lambda: tmp_path)

    assert embedded._managed_uvx_dir(str(tmp_path / "empty")) == str(uv_dir)
    assert embedded._managed_uvx_dir(str(uv_dir)) is None  # already on PATH: leave it


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


def _profile_env(text: str) -> Path:
    path = embedded._embedded_profile_env_path({"profile": "hermes"})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


_CFG = {"profile": "hermes", "llm_provider": "openai", "llm_model": "gpt", "llmApiKey": "sk"}
_OWNED = "HINDSIGHT_API_LLM_PROVIDER=openai\nHINDSIGHT_API_LLM_API_KEY=sk\nHINDSIGHT_API_LLM_MODEL=gpt\nHINDSIGHT_API_LOG_LEVEL=info\n"


def test_keys_the_plugin_does_not_own_are_not_drift(hermes_env):
    """#5222: the port hindsight-embed adds at daemon start must not restart the daemon."""
    _profile_env(_OWNED + "HINDSIGHT_API_PORT=9077\nHINDSIGHT_API_TENANT_API_KEY=t\n")
    assert embedded._profile_env_drifted(_CFG) is False


def test_a_changed_or_removed_plugin_key_is_drift(hermes_env):
    _profile_env(_OWNED.replace("gpt", "old-model"))
    assert embedded._profile_env_drifted(_CFG) is True
    _profile_env(_OWNED + "HINDSIGHT_API_LLM_BASE_URL=http://gone\n")  # no longer configured
    assert embedded._profile_env_drifted(_CFG) is True


def test_a_rewrite_keeps_the_keys_the_plugin_does_not_own(hermes_env):
    """#5252: the rewrite replaced the whole file, dropping the daemon's port and operator keys."""
    path = _profile_env(
        _OWNED.replace("gpt", "old-model") + "HINDSIGHT_API_PORT=9077\nHINDSIGHT_API_TENANT_API_KEY=t\n"
    )

    embedded._materialize_embedded_profile_env(_CFG)

    assert embedded._load_simple_env(path) == {
        "HINDSIGHT_API_LLM_PROVIDER": "openai",
        "HINDSIGHT_API_LLM_API_KEY": "sk",
        "HINDSIGHT_API_LLM_MODEL": "gpt",
        "HINDSIGHT_API_LOG_LEVEL": "info",
        "HINDSIGHT_API_PORT": "9077",
        "HINDSIGHT_API_TENANT_API_KEY": "t",
    }
    assert embedded._profile_env_drifted(_CFG) is False
