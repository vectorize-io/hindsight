"""Regression coverage for llama.cpp server startup failures (issue #3733).

The published Docker image omits `llama-cpp-python`, so `provider=llamacpp`
there can only fail. It has to fail *legibly*: before the ~3.5 GB model
download, and without leaving a dead server behind that turns every later call
into an unexplained connection error.
"""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from hindsight_api.engine.providers import llamacpp_llm
from hindsight_api.engine.providers.llamacpp_llm import LlamaCppLLM


@pytest.fixture(autouse=True)
def _reset_shared_server():
    """The provider caches one server per process; keep tests independent."""
    llamacpp_llm._shared_server = None
    yield
    llamacpp_llm._shared_server = None


def _provider() -> LlamaCppLLM:
    return LlamaCppLLM(provider="llamacpp", api_key="", base_url="", model="test-model")


@pytest.mark.asyncio
async def test_missing_llama_cpp_fails_before_downloading_the_model(monkeypatch):
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("model resolution must not run without llama_cpp installed")

    real_find_spec = llamacpp_llm.importlib.util.find_spec
    monkeypatch.setattr(
        llamacpp_llm.importlib.util,
        "find_spec",
        lambda name, *args, **kwargs: None if name == "llama_cpp" else real_find_spec(name, *args, **kwargs),
    )
    monkeypatch.setattr(llamacpp_llm, "_resolve_model_path", _fail_if_called)

    with pytest.raises(RuntimeError) as exc_info:
        await _provider()._ensure_initialized()

    message = str(exc_info.value)
    # Both supported ways out of this state have to be spelled out: the extra
    # for local installs, the sidecar for the published image.
    assert "local-llm" in message
    assert "docker/docker-compose/local-llm" in message


@pytest.mark.asyncio
async def test_failed_start_leaves_no_dead_server_behind(monkeypatch):
    starts: list[str] = []

    stops: list[str] = []

    class FakeServer:
        def __init__(self, **kwargs):
            self.port = kwargs["port"]

        @property
        def base_url(self) -> str:
            return f"http://127.0.0.1:{self.port}/v1"

        async def start(self) -> None:
            starts.append("start")
            if len(starts) == 1:
                raise RuntimeError("llama.cpp server exited with code 1")

        async def stop(self) -> None:
            stops.append("stop")

    monkeypatch.setattr(llamacpp_llm, "_require_llama_cpp", lambda: None)
    monkeypatch.setattr(llamacpp_llm, "_resolve_model_path", lambda path: Path("/models/test.gguf"))
    monkeypatch.setattr(llamacpp_llm, "LlamaCppServer", FakeServer)

    provider = _provider()
    with pytest.raises(RuntimeError, match="exited with code 1"):
        await provider._ensure_initialized()

    # A server that never started must not be published as the shared one, and
    # a subprocess that outlived a timed-out start must be reaped.
    assert llamacpp_llm._shared_server is None
    assert provider._delegate is None
    assert stops == ["stop"]

    # The next attempt genuinely retries startup instead of building a client
    # against a port nothing listens on.
    await provider._ensure_initialized()
    assert starts == ["start", "start"]
    assert provider._delegate is not None
    assert provider._delegate.base_url == llamacpp_llm._shared_server.base_url


@pytest.mark.asyncio
async def test_started_server_is_shared_across_providers(monkeypatch):
    started = AsyncMock()

    class FakeServer:
        def __init__(self, **kwargs):
            self.base_url = "http://127.0.0.1:1234/v1"
            self.start = started

    monkeypatch.setattr(llamacpp_llm, "_require_llama_cpp", lambda: None)
    monkeypatch.setattr(llamacpp_llm, "_resolve_model_path", lambda path: Path("/models/test.gguf"))
    monkeypatch.setattr(llamacpp_llm, "LlamaCppServer", FakeServer)

    await _provider()._ensure_initialized()
    await _provider()._ensure_initialized()

    assert started.await_count == 1
