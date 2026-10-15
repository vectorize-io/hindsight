"""The provider driven through the Hermes MemoryProvider interface, asserting what it
sends to Hindsight (a recording fake client stands in for the real SDK)."""

import asyncio
import json
import sys
import threading
from types import SimpleNamespace

import pytest

import hindsight_hermes as plugin
from conftest import FakeClient, FakeRecallResponse
from hindsight_client_api.exceptions import NotFoundException


def _retain_item(fake: FakeClient, index: int = 0) -> dict:
    return fake.retains[index]["items"][0]


def _messages_of(fake: FakeClient, index: int = 0) -> list[str]:
    """Message texts in one retain. Content is one flat JSON array of messages (#5262):
    a list of per-turn lists would keep the server from chunking by turn."""
    return [m["content"] for m in json.loads(_retain_item(fake, index)["content"])]


def test_sync_turn_retains_the_turn(provider):
    instance, fake = provider({"bank_id": "team", "retain_tags": "hermes"})
    instance.sync_turn("what is my name?", "Ada.")
    instance.shutdown()

    assert len(fake.retains) == 1
    call = fake.retains[0]
    assert call["bank_id"] == "team"
    assert call["document_id"] == "session-1"  # stable id + append on a capable API
    item = _retain_item(fake)
    assert item["update_mode"] == "append"
    assert "hermes" in item["tags"] and "session:session-1" in item["tags"]
    assert _messages_of(fake) == ["User: what is my name?", "Assistant: Ada."]


def test_retain_every_n_turns_buffers_then_ships_the_batch(provider):
    instance, fake = provider({"retain_every_n_turns": 2})
    instance.sync_turn("one", "1")
    assert fake.retains == []
    instance.sync_turn("two", "2")
    instance.shutdown()

    assert len(fake.retains) == 1
    assert _retain_item(fake)["metadata"]["message_count"] == "4"


def test_shutdown_flushes_turns_below_the_retain_cadence(provider):
    """#5493: turns waiting for the next every-N retain must not die with the process."""
    instance, fake = provider({"retain_every_n_turns": 3})
    instance.sync_turn("one", "1")
    instance.sync_turn("two", "2")
    instance.shutdown()

    assert len(fake.retains) == 1
    assert _messages_of(fake) == ["User: one", "Assistant: 1", "User: two", "Assistant: 2"]


def test_turn_timestamp_is_taken_when_queued_not_when_written(provider, monkeypatch):
    """#5189: a busy writer must not move the turn's event time to when it finally ran."""
    instance, fake = provider({})
    release = threading.Event()
    instance._enqueue_retain(release.wait)  # hold the writer until the clock has moved on
    monkeypatch.setattr(plugin, "_event_timestamp", lambda: "2026-01-01T00:00:00Z")
    instance.sync_turn("one", "1")
    monkeypatch.setattr(plugin, "_event_timestamp", lambda: "2026-01-01T00:05:00Z")
    release.set()
    instance.shutdown()

    assert _retain_item(fake)["timestamp"] == "2026-01-01T00:00:00Z"


def test_auto_retain_off_stores_nothing(provider):
    instance, fake = provider({"auto_retain": False})
    instance.sync_turn("hello", "hi")
    instance.shutdown()
    assert fake.retains == []


def test_recall_tool_queries_the_bank_and_formats_results(provider):
    instance, fake = provider(
        {"bank_id": "team", "recall_budget": "high"}, client=FakeClient(recall_texts=["fact one", "fact two"])
    )
    result = json.loads(instance.handle_tool_call("hindsight_recall", {"query": "who am I?"}))

    assert fake.recalls[0]["bank_id"] == "team"
    assert fake.recalls[0]["budget"] == "high"
    assert fake.recalls[0]["types"] == ["observation"]  # observation-only default
    assert result["result"] == "1. fact one\n2. fact two"
    instance.shutdown()


def test_recall_sends_no_score_floor_by_default(provider):
    instance, fake = provider({}, client=FakeClient(recall_texts=["fact one"]))
    instance.handle_tool_call("hindsight_recall", {"query": "who am I?"})
    assert "min_scores" not in fake.recalls[0]
    instance.shutdown()


def test_recall_min_scores_reaches_the_tool_and_the_prefetch(provider):
    instance, fake = provider(
        {"recall_sync": True, "recall_min_scores": {"reranker": 0.25}}, client=FakeClient(recall_texts=["fact one"])
    )
    instance.handle_tool_call("hindsight_recall", {"query": "who am I?"})
    instance.prefetch("what do you know?")
    assert [call["min_scores"] for call in fake.recalls] == [{"reranker": 0.25}] * 2
    instance.shutdown()


def test_recall_min_scores_accepts_the_json_string_the_setup_wizard_writes(provider):
    instance, fake = provider({"recall_min_scores": '{"reranker": 0.25}'}, client=FakeClient(recall_texts=["x"]))
    instance.handle_tool_call("hindsight_recall", {"query": "q"})
    assert fake.recalls[0]["min_scores"] == {"reranker": 0.25}
    instance.shutdown()


def test_recall_with_an_empty_answer_stays_an_empty_block(provider):
    # The server drops what falls under a reranker/final floor; an empty answer must stay empty.
    instance, fake = provider({"recall_sync": True, "recall_min_scores": {"reranker": 0.9}}, client=FakeClient())
    assert instance.prefetch("something off topic") == ""
    assert fake.recalls[0]["min_scores"] == {"reranker": 0.9}
    assert json.loads(instance.handle_tool_call("hindsight_recall", {"query": "q"}))["result"] == (
        "No relevant memories found."
    )
    instance.shutdown()


def test_recall_min_scores_is_enforced_on_each_result_the_server_returns(provider):
    # The server prunes only the retrieval arm a `semantic` floor names, so weak results (and ones
    # another arm found, with no semantic score) can still come back; the plugin drops them so the
    # floor really abstains.
    results = [("kept", {"semantic": 0.7}), ("weak", {"semantic": 0.3}), ("other arm", {"semantic": None})]
    instance, fake = provider({"recall_min_scores": {"semantic": 0.5}}, client=FakeClient(recall_texts=results))
    result = json.loads(instance.handle_tool_call("hindsight_recall", {"query": "q"}))
    assert result["result"] == "1. kept"
    assert fake.recalls[0]["min_scores"] == {"semantic": 0.5}
    instance.shutdown()


def test_recall_min_scores_drops_the_whole_block_when_nothing_clears_it(provider):
    results = [("weak", {"semantic": 0.3}), ("other arm", {"semantic": None})]
    instance, _ = provider(
        {"recall_sync": True, "recall_min_scores": {"semantic": 0.5}}, client=FakeClient(recall_texts=results)
    )
    assert instance.prefetch("something off topic") == ""
    assert json.loads(instance.handle_tool_call("hindsight_recall", {"query": "q"}))["result"] == (
        "No relevant memories found."
    )
    instance.shutdown()


def test_reflect_tool_uses_reflect(provider):
    instance, fake = provider({}, client=FakeClient(reflect_text="You are Ada."))
    result = json.loads(instance.handle_tool_call("hindsight_reflect", {"query": "who am I?"}))
    assert fake.reflects[0]["query"] == "who am I?"
    assert result["result"] == "You are Ada."
    instance.shutdown()


def test_retain_tool_stores_content_with_per_call_tags(provider):
    instance, fake = provider({"retain_tags": "base"})
    instance.handle_tool_call("hindsight_retain", {"content": "Ada likes tea", "tags": ["drink"]})
    item = _retain_item(fake)
    assert item["content"] == "Ada likes tea"
    assert item["tags"] == ["base", "drink"]
    instance.shutdown()


def test_builtin_memory_adds_and_replaces_are_retained(provider):
    instance, fake = provider({"bank_id": "team", "retain_tags": "base"})
    instance.on_memory_write("add", "memory", "Deploys go through Fly.io")
    instance.on_memory_write("replace", "user", "Ada prefers tea", metadata={"previous_content": "Ada likes tea"})
    instance.on_memory_write("remove", "memory", "", metadata={"previous_content": "Deploys go through Fly.io"})
    instance.shutdown()

    assert [call["bank_id"] for call in fake.retains] == ["team", "team"]
    first, second = _retain_item(fake, 0), _retain_item(fake, 1)
    assert first["content"] == "Deploys go through Fly.io"
    assert first["tags"] == ["base", "builtin-memory", "builtin-target:memory", "builtin-action:add"]
    assert second["content"] == "Ada prefers tea"
    assert second["tags"] == ["base", "builtin-memory", "builtin-target:user", "builtin-action:replace"]


def test_tool_call_errors_are_reported_not_raised(provider):
    instance, _ = provider({})
    assert instance.handle_tool_call("hindsight_recall", {}).startswith("ERROR:")
    assert instance.handle_tool_call("nope", {"query": "x"}).startswith("ERROR:")
    instance.shutdown()


class _EmptyServerClient(FakeClient):
    async def arecall(self, **kwargs):
        raise NotFoundException(status=404, reason="Not Found")

    async def areflect(self, **kwargs):
        raise NotFoundException(status=404, reason="Not Found")


def test_searching_a_bank_nothing_was_saved_to_finds_nothing(provider):
    instance, _ = provider({}, client=_EmptyServerClient())
    for tool in ("hindsight_recall", "hindsight_reflect"):
        result = json.loads(instance.handle_tool_call(tool, {"query": "who am I?"}))
        assert result == {"result": "No relevant memories found."}
    instance.shutdown()


def test_prefetch_injects_recalled_memories(provider):
    instance, fake = provider({"recall_sync": True}, client=FakeClient(recall_texts=["fact one"]))
    block = instance.prefetch("what do you know?")
    assert "- fact one" in block
    status = instance.recall_status()
    assert status.count == 1 and status.provider_label == "Hindsight"
    instance.shutdown()


def test_context_mode_hides_tools_tools_mode_skips_recall(provider):
    context_only, _ = provider({"memory_mode": "context"})
    assert context_only.get_tool_schemas() == []
    context_only.shutdown()

    tools_only, fake = provider({"memory_mode": "tools", "recall_sync": True})
    assert [t["name"] for t in tools_only.get_tool_schemas()] == [
        "hindsight_retain",
        "hindsight_recall",
        "hindsight_reflect",
    ]
    assert tools_only.prefetch("anything") == ""
    assert fake.recalls == []
    tools_only.shutdown()


def test_session_switch_starts_a_new_document(provider):
    instance, fake = provider({})
    instance.sync_turn("one", "1")
    instance.on_session_switch("session-2", reset=True)
    instance.sync_turn("two", "2")
    instance.shutdown()

    # The switch flushes the old session's buffer under the old document id first,
    # so the new session's turn can never land in the previous document. In append
    # mode the buffer is already empty here (sync_turn shipped and dropped the turn),
    # so there is nothing left to flush — previously this re-shipped the retained
    # turn under session-1 a second time, duplicating it in the document.
    assert [call["document_id"] for call in fake.retains] == ["session-1", "session-2"]


def test_session_switch_follows_a_session_bank_template(provider):
    """#4824: a {session} bank template names one bank per session, so a switch moves
    writes to the new session's bank."""
    instance, fake = provider({"bank_id_template": "hermes-{session}"})
    instance.sync_turn("one", "1")
    instance.on_session_switch("session-2")
    instance.sync_turn("two", "2")
    instance.shutdown()

    assert [call["bank_id"] for call in fake.retains] == ["hermes-session-1", "hermes-session-2"]


def test_a_session_named_as_its_own_parent_gets_no_parent_tag(provider):
    """#5422: in-place compaction reports the session as its own parent; tagging it would
    flip the document's tag set and re-consolidate it from scratch."""
    instance, fake = provider({}, parent_session_id="session-1")
    instance.sync_turn("one", "1")
    instance.shutdown()

    assert _retain_item(fake)["tags"] == ["session:session-1"]


def test_register_exposes_the_provider_to_hermes():
    registered = []
    plugin.register(type("Ctx", (), {"register_memory_provider": lambda _self, p: registered.append(p)})())
    assert registered and registered[0].name == "hindsight"


def test_append_mode_drops_retained_turns_from_the_buffer(provider):
    """Append retains ship a delta, so keeping every turn would pin the whole session
    in memory on a long-running gateway (hermes-agent #62950).

    Append mode comes from the API capability probe, which the fixture pins on — it is
    not a config key.
    """
    instance, fake = provider({})
    instance.sync_turn("one", "1")
    instance.sync_turn("two", "2")

    # Buffer state is read before shutdown(); retains only land once the writer drains.
    assert instance._session_turns == []
    assert instance._last_retained_turn_count == 0
    instance.shutdown()

    # Each retain still carries only its own un-retained tail, never a replay.
    assert _messages_of(fake, 0) == ["User: one", "Assistant: 1"]
    assert _messages_of(fake, 1) == ["User: two", "Assistant: 2"]


def test_overwrite_mode_keeps_every_turn(provider, monkeypatch):
    """Overwrite resends the full session on each retain, so its buffer must NOT be
    cleared — only the append path drops shipped turns."""
    instance, fake = provider({})
    # An API without update_mode='append' support: the fixture pins the probe on, so
    # turn it back off to exercise the overwrite path.
    monkeypatch.setattr(plugin, "_check_api_supports_update_mode_append", lambda *a, **k: False)
    instance.sync_turn("one", "1")
    instance.sync_turn("two", "2")

    assert len(instance._session_turns) == 2  # one buffered entry per turn
    instance.shutdown()

    # The second retain resends the whole session, which is what overwrite means.
    assert _messages_of(fake, 1) == ["User: one", "Assistant: 1", "User: two", "Assistant: 2"]
    # Shutdown on a cadence boundary sends nothing more: every turn already went out.
    assert len(fake.retains) == 2


def test_root_warning_goes_through_the_hosts_warning_callback(provider, monkeypatch):
    """The 'cannot run as root' notice is an automatic startup diagnostic: hosts that
    wire a gated sink must receive it there, not on stderr (hermes-agent cd3de040ab9)."""
    seen = []
    instance, _ = provider({}, warning_callback=seen.append, platform="telegram")
    assert instance._platform == "telegram"

    monkeypatch.setattr(plugin.os, "geteuid", lambda: 0, raising=False)
    instance._mode = "local_embedded"
    instance._start_embedded_daemon()

    assert len(seen) == 1 and "cannot run as root" in seen[0]
    assert instance._mode == "disabled"
    instance.shutdown()


def test_warning_sink_defaults_exist_without_initialize():
    """_start_embedded_daemon reads these directly, and availability probes construct a
    provider without ever calling initialize() — so __init__ must supply both."""
    bare = plugin.HindsightMemoryProvider()
    assert bare._warning_callback is None
    assert bare._platform == "cli"


def test_system_prompt_guides_tool_choice_only_when_tools_exist(provider):
    blocks = {}
    for mode in ("context", "tools", "hybrid"):
        instance, _ = provider({"memory_mode": mode})
        blocks[mode] = instance.system_prompt_block()
        instance.shutdown()

    assert "session_search" not in blocks["context"]
    assert "automatically injected" in blocks["context"]
    for mode in ("tools", "hybrid"):
        assert "prefer hindsight_recall over session_search" in blocks[mode]
        assert "hindsight_reflect" in blocks[mode] and "hindsight_retain" in blocks[mode]
    assert "automatically injected" in blocks["hybrid"]
    assert "automatically injected" not in blocks["tools"]


def test_the_first_run_download_is_announced_through_the_warning_sink(provider, monkeypatch):
    """A first embedded start fetches the server through uvx, which took minutes with nothing on
    screen (hermes-agent#4936: a 6m23s reply that retained nothing and printed no error). The
    notice goes to the same gated sink as the root-refusal warning."""
    seen = []
    instance, _ = provider({}, warning_callback=seen.append, platform="telegram")
    monkeypatch.setattr(plugin, "_daemon_is_running", lambda profile: False)
    monkeypatch.setattr(plugin, "_installed_api_binary_exists", lambda: False)

    instance._announce_slow_first_start("hermes")

    assert len(seen) == 1 and "downloading its local memory server" in seen[0]
    instance.shutdown()


def test_no_announcement_when_the_server_is_already_there(provider, monkeypatch):
    seen = []
    instance, _ = provider({}, warning_callback=seen.append)
    monkeypatch.setattr(plugin, "_daemon_is_running", lambda profile: False)
    monkeypatch.setattr(plugin, "_installed_api_binary_exists", lambda: True)
    instance._announce_slow_first_start("hermes")

    monkeypatch.setattr(plugin, "_daemon_is_running", lambda profile: True)
    monkeypatch.setattr(plugin, "_installed_api_binary_exists", lambda: False)
    instance._announce_slow_first_start("hermes")

    assert seen == []
    instance.shutdown()


def test_concurrent_callers_start_the_daemon_and_build_the_client_once(provider, monkeypatch):
    """The start worker and the first memory operation both reach _get_client. Unguarded, each
    started a daemon and built a client, and the loser's client was dropped without being closed.

    No barrier inside the build: with the lock in place only one caller ever gets there, so the
    contention window is opened with a sleep instead.
    """
    import threading
    import time as _time

    instance, _ = provider({})
    instance._mode = "local_embedded"
    built = []

    def _slow_build(self):
        _time.sleep(0.2)  # as wide as a real daemon start, in miniature
        built.append(object())
        return built[-1]

    monkeypatch.setattr(type(instance), "_new_embedded_client", _slow_build)

    results = []
    threads = [threading.Thread(target=lambda: results.append(instance._get_client())) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)

    assert len(built) == 1, f"client built {len(built)} times"
    assert len({id(r) for r in results}) == 1  # every caller got the same client
    instance.shutdown()


def test_building_the_embedded_client_announces_before_it_waits(provider, monkeypatch):
    """The notice has to fire from the path that actually blocks — asserting the helper in
    isolation would keep passing if nothing called it."""
    seen = []
    # Mode is switched after init: configuring local_embedded would also start the background
    # daemon worker, which builds a client too and made this test announce twice at random.
    instance, _ = provider({"profile": "hermes"}, warning_callback=seen.append)
    instance._mode = "local_embedded"
    order = []
    from hindsight_hermes.embedded import LocalRuntimeStatus

    monkeypatch.setattr(plugin, "_check_local_runtime", lambda: LocalRuntimeStatus(available=True))
    monkeypatch.setattr(plugin, "_daemon_is_running", lambda profile: False)
    monkeypatch.setattr(plugin, "_installed_api_binary_exists", lambda: False)
    monkeypatch.setattr(plugin, "_build_embedded_profile_env", lambda cfg: {})
    monkeypatch.setattr(
        plugin, "_start_daemon", lambda config, profile: order.append("started") or "http://127.0.0.1:1"
    )
    monkeypatch.setattr(plugin, "Hindsight", lambda **kw: object(), raising=False)
    instance._warning_callback = lambda m: order.append("announced")

    instance._new_embedded_client()

    assert order == ["announced", "started"], order
    instance.shutdown()


def test_the_embedded_client_sends_the_daemons_tenant_key(provider, monkeypatch):
    """#5023: a daemon running ApiKeyTenantExtension answers 401 to a keyless client."""
    instance, _ = provider({"profile": "hermes"})
    instance._mode = "local_embedded"
    from hindsight_hermes.embedded import LocalRuntimeStatus, _embedded_profile_env_path

    profile_env = _embedded_profile_env_path({"profile": "hermes"})
    profile_env.parent.mkdir(parents=True, exist_ok=True)
    profile_env.write_text("HINDSIGHT_API_TENANT_API_KEY=tenant-secret\n")
    built = {}

    class _Client:
        def __init__(self, **kwargs):
            built.update(kwargs)

    monkeypatch.setattr(plugin, "_check_local_runtime", lambda: LocalRuntimeStatus(available=True))
    monkeypatch.setattr(plugin, "_start_daemon", lambda config, profile: "http://127.0.0.1:1")
    monkeypatch.setattr(plugin, "_installed_api_binary_exists", lambda: True)
    monkeypatch.setitem(sys.modules, "hindsight_client", SimpleNamespace(Hindsight=_Client))

    instance._new_embedded_client()

    assert built == {"base_url": "http://127.0.0.1:1", "api_key": "tenant-secret"}
    instance.shutdown()


def test_retain_omits_strategy_by_default(provider):
    """No strategy configured means no key on the item, so the bank keeps deciding."""
    instance, fake = provider()
    instance.sync_turn("hello", "hi")
    instance.shutdown()

    assert fake.retains
    for call in fake.retains:
        for item in call["items"]:
            assert "strategy" not in item


def test_retain_sends_the_configured_strategy(provider):
    """A configured strategy rides on every stored item."""
    instance, fake = provider({"retain_strategy": "agent-session"})
    instance.sync_turn("hello", "hi")
    instance.shutdown()

    assert fake.retains
    for call in fake.retains:
        for item in call["items"]:
            assert item["strategy"] == "agent-session"


def test_retain_strategy_is_exposed_as_a_setting(provider):
    """Operators must be able to set it without editing code."""
    instance, _ = provider()
    keys = {option["key"] for option in instance.get_config_schema()}
    assert "retain_strategy" in keys


def test_version_probe_sends_plugin_user_agent(monkeypatch):
    import urllib.request

    sent = {}

    def fake_urlopen(req, timeout):
        sent.update(req.headers)
        raise OSError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(plugin._VersionProbeUnreachable):
        plugin._fetch_hindsight_api_version("http://hs.test")
    assert sent["User-agent"] == plugin._USER_AGENT
    assert plugin._USER_AGENT != "hindsight-hermes/0.0.0"


class _VersionServer:
    """urlopen stand-in: maps a path to a JSON body, an HTTP status, or an exception."""

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def __call__(self, req, timeout):
        import io
        import urllib.error

        path = req.full_url.split("hs.test", 1)[1]
        self.calls.append(path)
        answer = self.routes.get(path, 404)
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, int):
            raise urllib.error.HTTPError(req.full_url, answer, "", {}, io.BytesIO(b""))
        return io.BytesIO(json.dumps(answer).encode())


@pytest.fixture
def version_probe(monkeypatch):
    import urllib.request

    monkeypatch.setitem(sys.modules, "agent.credential_persistence", SimpleNamespace(fingerprint_secret_value=str))
    monkeypatch.setattr(plugin, "_append_capability_cache", {})
    monkeypatch.setattr(plugin, "_append_probe_unreachable_at", {})

    def _serve(routes):
        server = _VersionServer(routes)
        monkeypatch.setattr(urllib.request, "urlopen", server)
        return server

    return _serve


def test_an_unreachable_server_is_asked_again_not_cached(version_probe, monkeypatch):
    """#5221: one failed probe at startup used to disable append for the whole process."""
    down = version_probe({"/v1/version": OSError("refused"), "/version": OSError("refused")})
    assert plugin._check_api_supports_update_mode_append("http://hs.test") is False
    # Inside the retry window a hung server is not asked again on every turn.
    assert plugin._check_api_supports_update_mode_append("http://hs.test") is False
    assert down.calls == ["/v1/version", "/version"]

    monkeypatch.setattr(plugin, "_APPEND_PROBE_RETRY_S", 0.0)
    up = version_probe({"/v1/version": {"api_version": "0.10.3"}})
    assert plugin._check_api_supports_update_mode_append("http://hs.test") is True
    assert up.calls == ["/v1/version"]


def test_a_server_error_counts_as_unreachable(version_probe, monkeypatch):
    version_probe({"/v1/version": 503, "/version": 502})
    assert plugin._check_api_supports_update_mode_append("http://hs.test") is False
    monkeypatch.setattr(plugin, "_APPEND_PROBE_RETRY_S", 0.0)
    version_probe({"/v1/version": {"api_version": "0.10.3"}})
    assert plugin._check_api_supports_update_mode_append("http://hs.test") is True


def test_one_failed_path_keeps_the_answer_unknown(version_probe, monkeypatch):
    """A 404 on /v1/version and a timeout on /version is not proof of an old server."""
    version_probe({"/v1/version": 404, "/version": TimeoutError()})
    assert plugin._check_api_supports_update_mode_append("http://hs.test") is False
    monkeypatch.setattr(plugin, "_APPEND_PROBE_RETRY_S", 0.0)
    version_probe({"/v1/version": 404, "/version": {"version": "0.10.3"}})
    assert plugin._check_api_supports_update_mode_append("http://hs.test") is True


def test_an_old_server_answer_is_cached(version_probe):
    """A server that answers without a version is old, not down: that answer is kept."""
    old = version_probe({"/v1/version": 404, "/version": 404})
    assert plugin._check_api_supports_update_mode_append("http://hs.test") is False
    assert plugin._check_api_supports_update_mode_append("http://hs.test") is False
    assert old.calls == ["/v1/version", "/version"]


def test_version_falls_back_to_the_bare_path(version_probe):
    """Servers that predate the /v1 alias still answer /version (coding-agents order, #5286)."""
    server = version_probe({"/v1/version": 404, "/version": {"version": "0.5.0"}})
    assert plugin._check_api_supports_update_mode_append("http://hs.test") is True
    assert server.calls == ["/v1/version", "/version"]


def test_retain_tool_honours_retain_async(provider):
    """#4662: the tool blocked on a sync retain whatever retain_async said."""
    instance, fake = provider({})
    result = json.loads(instance.handle_tool_call("hindsight_retain", {"content": "Ada likes tea"}))
    assert fake.retains[0]["retain_async"] is True
    assert result["result"] == "Memory queued for storage."

    instance, fake = provider({"retain_async": False})
    result = json.loads(instance.handle_tool_call("hindsight_retain", {"content": "Ada likes tea"}))
    assert fake.retains[0]["retain_async"] is False
    assert result["result"] == "Memory stored successfully."


def test_a_tool_error_with_no_message_still_says_what_failed(provider):
    """#4662: a timeout's str() is empty, which left the agent a bare 'Failed to search memory: '."""

    class TimingOut(FakeClient):
        async def arecall(self, **kwargs):
            raise TimeoutError()

    instance, _ = provider({}, client=TimingOut())
    result = instance.handle_tool_call("hindsight_recall", {"query": "q"})
    assert result.endswith("Failed to search memory: TimeoutError")


def test_a_slow_prefetch_never_overwrites_a_newer_one(provider):
    """#4662: an older background recall that finishes last must not replace the newer result."""
    release_first = threading.Event()

    class SlowFirst(FakeClient):
        async def arecall(self, **kwargs):
            if kwargs["query"] == "old":
                while not release_first.is_set():
                    await asyncio.sleep(0.01)
                return FakeRecallResponse(["stale memory"])
            return FakeRecallResponse(["fresh memory"])

    instance, _ = provider({"prefetch_waits_for_retain": False}, client=SlowFirst())
    instance.queue_prefetch("old")
    old_worker = instance._prefetch_thread
    instance.queue_prefetch("new")
    instance._prefetch_thread.join(5)
    release_first.set()
    old_worker.join(5)

    assert "fresh memory" in instance.prefetch("next")
    instance.shutdown()


def test_recall_tags_accept_the_documented_comma_separated_form(provider):
    """#4662: the string went to the client as-is and the API rejected it (422)."""
    instance, fake = provider({"recall_tags": "team, project-x"})
    instance.handle_tool_call("hindsight_recall", {"query": "q"})
    assert fake.recalls[0]["tags"] == ["team", "project-x"]


def test_a_mistyped_number_in_config_falls_back_to_its_default(provider):
    """#4662: a plain int()/float() on these crashed initialize, so memory was off entirely."""
    instance, _ = provider(
        {
            "retain_every_n_turns": "two",
            "recall_max_tokens": "lots",
            "recall_max_input_chars": "",
            "prefetch_retain_drain_timeout": "soon",
        }
    )
    assert instance._retain_every_n_turns == 1
    assert instance._recall_max_tokens == 4096
    assert instance._recall_max_input_chars == 800
    assert instance._prefetch_retain_drain_timeout == 10.0


def test_prompt_does_not_promise_injected_memories_when_auto_recall_is_off(provider):
    """#4662: hybrid mode told the model memories were injected even with auto_recall off."""
    on, _ = provider({})
    off, _ = provider({"auto_recall": False})
    assert plugin._CONTEXT_NOTE in on.system_prompt_block()
    assert plugin._CONTEXT_NOTE not in off.system_prompt_block()
    assert plugin._TOOL_GUIDANCE in off.system_prompt_block()


def test_reconnect_closes_the_client_it_replaces(provider, monkeypatch):
    """#4662: the retry after a dead embedded daemon dropped the old client unclosed."""

    class Dead(FakeClient):
        closed = False

        async def arecall(self, **kwargs):
            raise RuntimeError("Cannot connect to host 127.0.0.1:8888")

        async def aclose(self):
            Dead.closed = True

    dead, fresh = Dead(), FakeClient(recall_texts=["back"])
    instance, _ = provider({}, client=dead)
    instance._mode, instance._client = "local_embedded", dead
    monkeypatch.setattr(instance, "_new_embedded_client", lambda: fresh)

    result = json.loads(instance.handle_tool_call("hindsight_recall", {"query": "q"}))
    assert result["result"] == "1. back"
    assert Dead.closed


def test_shutdown_drops_its_atexit_hook(provider, monkeypatch):
    """#4662: the hook kept every provider alive until exit; the gateway makes one per chat session.
    Hooks are recorded rather than counted: atexit._ncallbacks() does not drop on unregister on
    every Python the plugin supports."""
    hooks: list = []
    monkeypatch.setattr(plugin.atexit, "register", hooks.append)
    monkeypatch.setattr(plugin.atexit, "unregister", hooks.remove)

    instance, _ = provider({})
    instance.sync_turn("hi", "hello")
    assert hooks == [instance._atexit_shutdown]
    instance.shutdown()
    assert hooks == []
    assert not instance._atexit_registered


def test_availability_failure_is_logged(hermes_env, caplog):
    """#4662: is_available swallowed the error, so a broken config showed no reason anywhere."""
    (hermes_env / "hindsight").mkdir(parents=True, exist_ok=True)
    (hermes_env / "hindsight" / "config.json").write_text(json.dumps({"mode": ["cloud"]}))
    with caplog.at_level("DEBUG"):
        assert plugin.HindsightMemoryProvider().is_available() is False
    assert "availability check failed" in caplog.text


def test_desktop_panel_offers_every_mode_the_plugin_runs():
    """#4662: without local_embedded the panel could not hold that mode and saving rewrote it."""
    from hindsight_hermes.config_schema import CONFIG_SCHEMA

    mode = next(field for field in CONFIG_SCHEMA["fields"] if field["key"] == "mode")
    assert {option[0] for option in mode["options"]} == {"cloud", "local_embedded", "local_external"}


def test_cron_runs_are_not_auto_retained(provider):
    """#4945: Hermes passes agent_context="cron" for scheduled jobs; the contract is to skip writes."""
    cron, fake = provider({}, agent_context="cron")
    cron.sync_turn("daily digest", "here is the news")
    cron.handle_tool_call("hindsight_retain", {"content": "an explicit note"})  # the tool still works
    cron.shutdown()
    assert [call["items"][0]["content"] for call in fake.retains] == ["an explicit note"]


def test_retain_cron_opts_cron_runs_back_in(provider):
    cron, fake = provider({"retain_cron": True}, agent_context="cron")
    cron.sync_turn("daily digest", "here is the news")
    cron.shutdown()
    assert len(fake.retains) == 1


def test_primary_runs_are_auto_retained(provider):
    primary, fake = provider({}, agent_context="primary")
    primary.sync_turn("hello", "hi")
    primary.shutdown()
    assert len(fake.retains) == 1


def test_bank_id_env_wins_over_config_json(provider):
    """#5171: HINDSIGHT_BANK_ID is documented as an override; config.json used to shadow it."""
    from conftest import SECRETS

    SECRETS["HINDSIGHT_BANK_ID"] = "env-bank"
    instance, fake = provider({"bank_id": "config-bank"})
    instance.sync_turn("hello", "hi")
    instance.shutdown()
    assert fake.retains[0]["bank_id"] == "env-bank"


def test_config_bank_is_kept_without_the_env_var(provider):
    instance, fake = provider({"bank_id": "config-bank"})
    instance.sync_turn("hello", "hi")
    instance.shutdown()
    assert fake.retains[0]["bank_id"] == "config-bank"


def test_reflect_tags_scope_reflect(provider):
    """#5170: reflect gets its own opt-in scope; recall_tags never leak into it."""
    instance, fake = provider(
        {"reflect_tags": "team, project-x", "reflect_tags_match": "all", "recall_tags": "other"},
        client=FakeClient(reflect_text="synthesis"),
    )
    instance.handle_tool_call("hindsight_reflect", {"query": "q"})
    assert fake.reflects[0]["tags"] == ["team", "project-x"]
    assert fake.reflects[0]["tags_match"] == "all"


def test_reflect_without_reflect_tags_uses_the_whole_bank(provider):
    instance, fake = provider({"recall_tags": "team"}, client=FakeClient(reflect_text="synthesis"))
    instance.handle_tool_call("hindsight_reflect", {"query": "q"})
    assert "tags" not in fake.reflects[0]


def test_observations_default_to_shared_scope(provider):
    instance, fake = provider({})
    instance.sync_turn("hi", "hello")
    instance.shutdown()
    assert _retain_item(fake)["observation_scopes"] == "shared"


def test_explicit_observation_scopes_win_over_the_default(provider):
    instance, fake = provider({"observation_scopes": "combined"})
    instance.sync_turn("hi", "hello")
    instance.shutdown()
    assert _retain_item(fake)["observation_scopes"] == "combined"
