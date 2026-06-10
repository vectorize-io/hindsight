"""Shared fixtures for the Hindsight Cline integration tests."""

import json
import os
import sys
from types import SimpleNamespace

import pytest

# The hook scripts do `sys.path.insert(0, <hooks dir>)` so `lib.*` resolves.
# Mirror that here (the hook payload now lives under the package as data).
HOOKS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "hindsight_cline", "hooks"))
if HOOKS_DIR not in sys.path:
    sys.path.insert(0, HOOKS_DIR)

from lib.cline_io import HookInput  # noqa: E402
from lib.config import HindsightClineConfig, camel_to_snake  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Point HOME at a tmp dir so state and user config never touch the real ~."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    for key in list(os.environ):
        if key.startswith("HINDSIGHT_"):
            monkeypatch.delenv(key, raising=False)
    return tmp_path


def base_config(**overrides) -> HindsightClineConfig:
    """A config with a fixed external API URL and no mission PATCH by default.

    Overrides may be passed in camelCase (the settings.json form) for parity
    with how the plugin is configured; they're mapped to the dataclass fields.
    """
    cfg = HindsightClineConfig(hindsight_api_url="https://api.test", bank_mission="", retain_mission=None)
    for key, value in overrides.items():
        setattr(cfg, camel_to_snake(key), value)
    return cfg


def make_hook(hook_name="UserPromptSubmit", prompt="", task="", task_id="t1", workspace="/home/user/proj"):
    return HookInput(
        hook_name=hook_name,
        task_id=task_id,
        prompt=prompt,
        task=task,
        workspace_roots=[workspace] if workspace else [],
    )


def make_memory(text, mem_type="experience", mentioned_at="2026-01-15"):
    return {"text": text, "type": mem_type, "mentioned_at": mentioned_at}


class _FakeHTTPResponse:
    def __init__(self, data: dict, status: int = 200):
        self.status = status
        self._data = json.dumps(data).encode()

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class HttpMock:
    """Records requests and routes them to canned responses."""

    def __init__(self):
        self.calls = []
        self.results = []  # what /recall returns
        self.fail = False  # when True, urlopen raises (simulates server down)

    def _urlopen(self, req, timeout=None):
        if self.fail:
            raise OSError("connection refused")
        url = req.full_url
        body = json.loads(req.data.decode()) if req.data else None
        self.calls.append(SimpleNamespace(url=url, method=req.get_method(), body=body))
        if "/memories/recall" in url:
            return _FakeHTTPResponse({"results": self.results})
        return _FakeHTTPResponse({"status": "ok"})

    def retain_calls(self):
        return [c for c in self.calls if c.method == "POST" and c.url.endswith("/memories")]


@pytest.fixture
def http(monkeypatch):
    mock = HttpMock()
    monkeypatch.setattr("urllib.request.urlopen", mock._urlopen)
    return mock
