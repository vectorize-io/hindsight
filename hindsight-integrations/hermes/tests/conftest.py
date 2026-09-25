"""Loads the plugin the way Hermes does: stub Hermes core modules into ``sys.modules``,
then import this directory as a package so its relative imports resolve.

Hermes Agent is not on PyPI, so its interfaces (``MemoryProvider``, the secret scope,
``cfg_get``, ...) are simulated here. The stubs are deliberately thin — they exist so the
tests can assert what the plugin *does* with them, not to reimplement Hermes.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parent.parent

# --- fake Hermes core ---------------------------------------------------------

SECRETS: dict[str, str] = {}


class UnscopedSecretError(RuntimeError):
    """Raised by Hermes when a secret is read with no profile scope installed."""


def _get_secret(name: str, default: str = "") -> str:
    return SECRETS.get(name, default)


class MemoryProvider:  # the Hermes base class, reduced to what the plugin overrides
    pass


@dataclass
class RecallStatus:
    provider_label: str
    count: int
    glyph: str


def _spawn_context_thread(target, name: str = ""):
    import threading

    return threading.Thread(target=target, name=name, daemon=True)


def _safe_schedule_threadsafe(coro, loop):
    import asyncio

    return asyncio.run_coroutine_threadsafe(coro, loop)


def _cfg_get(cfg: dict, *keys, default=None):
    node = cfg
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _read_json_or_empty(path) -> dict:
    import json

    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _atomic_json_write(path, data, mode=0o600) -> None:
    import json
    import os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    os.chmod(path, mode)


def _install_hermes_stubs(hermes_home: Path) -> None:
    def module(name: str, **attrs) -> None:
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        sys.modules[name] = mod

    module("agent")
    module(
        "agent.memory_provider",
        MemoryProvider=MemoryProvider,
        RecallStatus=RecallStatus,
        spawn_context_thread=_spawn_context_thread,
    )
    module("agent.secret_scope", get_secret=_get_secret, UnscopedSecretError=UnscopedSecretError)
    module("agent.async_utils", safe_schedule_threadsafe=_safe_schedule_threadsafe)
    module("hermes_cli")
    module("hermes_cli.config", cfg_get=_cfg_get, save_config=lambda *a, **k: None)
    module("hermes_cli.secret_prompt", masked_secret_prompt=lambda label="": "")
    module(
        "hermes_cli.memory_setup",
        _CANCELLED=-1,
        _curses_select=lambda *a, **k: 0,
        _print_cancelled_setup=lambda: None,
        _prompt=lambda *a, **k: "",
    )
    module("hermes_constants", get_hermes_home=lambda: hermes_home)
    module("hermes_time", now=lambda: datetime.now(timezone.utc))
    module("tools")
    module("tools.registry", tool_error=lambda msg: f"ERROR: {msg}")
    module("tools.lazy_deps", install_specs=lambda *a, **k: None)
    module("utils", read_json_or_empty=_read_json_or_empty, atomic_json_write=_atomic_json_write)
    # plugins.memory.config_schema stays Hermes-owned (the desktop panel needs core's type).
    module("plugins")
    module("plugins.memory", _get_active_memory_provider=lambda: "hindsight")
    module(
        "plugins.memory.config_schema",
        KIND_SECRET="secret",
        KIND_SELECT="select",
        KIND_TEXT="text",
        ProviderConfigSchema=lambda **kw: kw,
        ProviderField=lambda **kw: kw,
        ProviderFieldOption=lambda *a: a,
    )


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    """Fresh Hermes home, empty secret store, plugin module loaded against them."""
    SECRETS.clear()
    # The plugin binds `get_hermes_home` at import time, so patch it where it is used.
    monkeypatch.setattr(sys.modules["hindsight_hermes"], "get_hermes_home", lambda: tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    for var in list(__import__("os").environ):
        if var.startswith("HINDSIGHT_"):
            monkeypatch.delenv(var, raising=False)
    yield tmp_path
    SECRETS.clear()


_install_hermes_stubs(Path.home())

_spec = importlib.util.spec_from_file_location(
    "hindsight_hermes",
    PLUGIN_ROOT / "__init__.py",
    submodule_search_locations=[str(PLUGIN_ROOT)],
)
plugin = importlib.util.module_from_spec(_spec)
sys.modules["hindsight_hermes"] = plugin
_spec.loader.exec_module(plugin)


class FakeResult:
    def __init__(self, text: str):
        self.text = text


class FakeRecallResponse:
    def __init__(self, texts):
        self.results = [FakeResult(t) for t in texts]


class FakeReflectResponse:
    def __init__(self, text):
        self.text = text


class FakeClient:
    """Records the calls the plugin makes against the Hindsight client."""

    def __init__(self, recall_texts=(), reflect_text=""):
        self.retains: list[dict] = []
        self.recalls: list[dict] = []
        self.reflects: list[dict] = []
        self._recall_texts = list(recall_texts)
        self._reflect_text = reflect_text

    async def aretain_batch(self, **kwargs):
        self.retains.append(kwargs)
        return types.SimpleNamespace(operations=[])

    async def arecall(self, **kwargs):
        self.recalls.append(kwargs)
        return FakeRecallResponse(self._recall_texts)

    async def areflect(self, **kwargs):
        self.reflects.append(kwargs)
        return FakeReflectResponse(self._reflect_text)

    async def aclose(self):
        pass


@pytest.fixture
def provider(hermes_env, monkeypatch):
    """An initialized cloud-mode provider wired to a FakeClient."""

    def _make(config: dict | None = None, client: FakeClient | None = None, **init_kwargs):
        import json

        config_path = hermes_env / "hindsight" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps({"mode": "cloud", "apiKey": "test-key", **(config or {})}))
        fake = client or FakeClient()
        instance = plugin.HindsightMemoryProvider()
        monkeypatch.setattr(instance, "_new_cloud_client", lambda: fake)
        # The /version probe is a live HTTP call; pin the capability instead.
        monkeypatch.setattr(plugin, "_check_api_supports_update_mode_append", lambda *a, **k: True)
        instance.initialize(init_kwargs.pop("session_id", "session-1"), **init_kwargs)
        return instance, fake

    return _make
