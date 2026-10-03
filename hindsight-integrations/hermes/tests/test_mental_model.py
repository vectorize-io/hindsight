"""Mental-model injection: the configured model is read once at initialize() and appended
to system_prompt_block(), which must then stay byte-identical for the session."""

import asyncio
import logging
import time
import types

import hindsight_embed
import hindsight_hermes as plugin
import pytest
from conftest import FakeClient

PROFILE = "## Profile\nPrefers concise answers."


def _client(content) -> FakeClient:
    fake = FakeClient()
    fake.mental_model = content
    return fake


class EmbeddedShapedClient(FakeClient):
    """HindsightEmbedded's mental_models namespace is its own wrapper (create/list/get/...),
    not the generated API, so only the top-level client methods reach the real client."""

    mental_models = types.SimpleNamespace(get=lambda bank_id, mental_model_id: None)


@pytest.fixture
def embedded_provider(hermes_env, monkeypatch):
    """An initialized local_embedded provider: runtime present, daemon start stubbed."""

    def _make(config: dict, client: FakeClient, *, running: bool, start=lambda instance: None):
        import json

        path = hermes_env / "hindsight" / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"mode": "local_embedded", **config}))
        monkeypatch.setattr(plugin, "_check_local_runtime", lambda: (True, None))
        monkeypatch.setattr(plugin, "_check_api_supports_update_mode_append", lambda *a, **k: True)
        instance = plugin.HindsightMemoryProvider()
        monkeypatch.setattr(instance, "_new_embedded_client", lambda: client)
        monkeypatch.setattr(instance, "_start_embedded_daemon", lambda: start(instance))
        monkeypatch.setattr(instance, "_embedded_daemon_running", lambda: running)
        instance.initialize("session-1")
        return instance

    return _make


def test_configured_model_is_fetched_once_and_injected(provider):
    fake = _client(PROFILE)
    instance, _ = provider({"bank_id": "team", "mental_model_id": "user-profile"}, client=fake)

    assert fake.mental_model_gets == [{"bank_id": "team", "mental_model_id": "user-profile", "detail": "content"}]
    block = instance.system_prompt_block()
    assert block.startswith("# Hindsight Memory\nActive. Bank: team")
    assert block.endswith(
        f"<memory-context>\n# Hindsight Mental Model (synthesized cross-session context)\nID: user-profile\n\n{PROFILE}\n</memory-context>"
    )
    instance.shutdown()


def test_block_is_byte_identical_for_the_session(provider):
    fake = _client(PROFILE)
    instance, _ = provider({"mental_model_id": "user-profile"}, client=fake)

    first = instance.system_prompt_block()
    fake.mental_model = "changed on the server"
    assert instance.system_prompt_block() == first
    assert len(fake.mental_model_gets) == 1
    instance.shutdown()


@pytest.mark.parametrize("model_id", ["", "   ", None])
def test_unset_id_makes_no_request(provider, model_id):
    fake = _client(PROFILE)
    instance, _ = provider({"mental_model_id": model_id}, client=fake)

    assert fake.mental_model_gets == []
    assert "<memory-context>" not in instance.system_prompt_block()
    instance.shutdown()


def test_fetch_failure_warns_and_leaves_the_block_unchanged(provider, caplog):
    fake = _client(RuntimeError("404 mental model not found"))
    with caplog.at_level(logging.WARNING, logger=plugin.__name__):
        instance, _ = provider({"mental_model_id": "missing"}, client=fake)

    assert "<memory-context>" not in instance.system_prompt_block()
    assert "404 mental model not found" in caplog.text
    instance.shutdown()


@pytest.mark.parametrize("content", [None, ""])
def test_empty_content_warns_and_is_not_injected(provider, caplog, content):
    with caplog.at_level(logging.WARNING, logger=plugin.__name__):
        instance, _ = provider({"mental_model_id": "empty"}, client=_client(content))

    assert "<memory-context>" not in instance.system_prompt_block()
    assert "has no content" in caplog.text
    instance.shutdown()


def test_slow_server_is_bounded_and_not_injected(provider, monkeypatch):
    monkeypatch.setattr(plugin, "_MENTAL_MODEL_FETCH_TIMEOUT", 0.05)

    class SlowClient(FakeClient):
        async def aget_mental_model(self, bank_id, mental_model_id, detail=None):
            await asyncio.sleep(5)

    started = time.monotonic()
    instance, _ = provider({"mental_model_id": "user-profile"}, client=SlowClient())

    assert time.monotonic() - started < 2
    assert "<memory-context>" not in instance.system_prompt_block()
    instance.shutdown()


@pytest.mark.parametrize("memory_mode", ["context", "tools", "hybrid"])
def test_injected_in_every_memory_mode_after_the_mode_tail(provider, memory_mode):
    instance, _ = provider({"mental_model_id": "user-profile", "memory_mode": memory_mode}, client=_client(PROFILE))

    block = instance.system_prompt_block()
    tail = plugin._SYSTEM_PROMPT_TAILS[memory_mode]
    assert PROFILE in block and block.index(tail) < block.index("<memory-context>")
    instance.shutdown()


def test_embedded_running_daemon_fetches_through_the_top_level_client(embedded_provider):
    client = EmbeddedShapedClient()
    client.mental_model = PROFILE
    instance = embedded_provider({"mental_model_id": "user-profile"}, client, running=True)

    assert len(client.mental_model_gets) == 1
    assert PROFILE in instance.system_prompt_block()
    instance.shutdown()


def test_embedded_daemon_not_running_skips_without_touching_the_client(embedded_provider, caplog):
    client = EmbeddedShapedClient()
    client.mental_model = PROFILE
    with caplog.at_level(logging.INFO, logger=plugin.__name__):
        instance = embedded_provider({"mental_model_id": "user-profile"}, client, running=False)

    assert client.mental_model_gets == []
    assert instance._client is None  # never built on the caller thread
    assert "<memory-context>" not in instance.system_prompt_block()
    assert "embedded daemon is not running yet" in caplog.text
    instance.shutdown()


def test_embedded_disabled_at_daemon_start_skips_the_fetch(embedded_provider):
    """The root guard in _start_embedded_daemon() disables the provider synchronously."""
    client = _client(PROFILE)

    def disable(instance):
        instance._mode = "disabled"

    instance = embedded_provider({"mental_model_id": "user-profile"}, client, running=True, start=disable)

    assert instance._mode == "disabled"
    assert client.mental_model_gets == []
    instance.shutdown()


@pytest.mark.parametrize("probe, expected", [(lambda p: True, True), (lambda p: False, False)])
def test_liveness_check_asks_the_manager_for_the_configured_profile(provider, monkeypatch, probe, expected):
    instance, _ = provider({"profile": "work"})
    seen = []
    manager = types.SimpleNamespace(is_running=lambda profile: seen.append(profile) or probe(profile))
    monkeypatch.setattr(hindsight_embed, "get_embed_manager", lambda: manager)

    assert instance._embedded_daemon_running() is expected
    assert seen == ["work"]
    instance.shutdown()


def test_liveness_check_failure_counts_as_not_running(provider, monkeypatch):
    instance, _ = provider({})

    def boom():
        raise RuntimeError("no profile")

    monkeypatch.setattr(hindsight_embed, "get_embed_manager", boom)
    assert instance._embedded_daemon_running() is False
    instance.shutdown()
