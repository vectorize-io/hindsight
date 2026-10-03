"""The provider driven through the Hermes MemoryProvider interface, asserting what it
sends to Hindsight (a recording fake client stands in for the real SDK)."""

import json
from types import SimpleNamespace

import hindsight_hermes as plugin
import pytest
from hindsight_hermes import embedded
from hindsight_hermes.embedded import LocalRuntimeStatus, _build_embedded_profile_env, _embedded_tenant_api_key
from conftest import SECRETS, FakeClient


def _retain_item(fake: FakeClient, index: int = 0) -> dict:
    return fake.retains[index]["items"][0]


def _turns_of(fake: FakeClient, index: int = 0) -> list[list[str]]:
    """Message texts per turn in one retain. Content is ``"[" + ",".join(turns) + "]"``
    where each turn is itself a JSON array, so the whole payload is a list of turns."""
    return [[m["content"] for m in turn] for turn in json.loads(_retain_item(fake, index)["content"])]


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
    messages = json.loads(item["content"][1:-1])
    assert [m["content"] for m in messages] == ["User: what is my name?", "Assistant: Ada."]


def test_retain_every_n_turns_buffers_then_ships_the_batch(provider):
    instance, fake = provider({"retain_every_n_turns": 2})
    instance.sync_turn("one", "1")
    assert fake.retains == []
    instance.sync_turn("two", "2")
    instance.shutdown()

    assert len(fake.retains) == 1
    assert _retain_item(fake)["metadata"]["message_count"] == "4"


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


def test_tool_call_errors_are_reported_not_raised(provider):
    instance, _ = provider({})
    assert instance.handle_tool_call("hindsight_recall", {}).startswith("ERROR:")
    assert instance.handle_tool_call("nope", {"query": "x"}).startswith("ERROR:")
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
    assert _turns_of(fake, 0) == [["User: one", "Assistant: 1"]]
    assert _turns_of(fake, 1) == [["User: two", "Assistant: 2"]]


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
    assert _turns_of(fake, 1) == [["User: one", "Assistant: 1"], ["User: two", "Assistant: 2"]]


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
    instance, _ = provider({"mode": "local_embedded", "profile": "hermes"}, warning_callback=seen.append)
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


def _embedded_client_kwargs(provider, monkeypatch, cfg):
    """kwargs the embedded client is constructed with, daemon start stubbed out."""
    import hindsight_client

    instance, _ = provider(cfg)
    instance._mode = "local_embedded"

    built = {}
    monkeypatch.setattr(plugin, "_check_local_runtime", lambda: LocalRuntimeStatus(available=True))
    monkeypatch.setattr(plugin, "_build_embedded_profile_env", lambda cfg: {})
    monkeypatch.setattr(plugin, "_start_daemon", lambda config, profile: "http://127.0.0.1:1")
    monkeypatch.setattr(hindsight_client, "Hindsight", lambda **kw: built.update(kw))

    instance._new_embedded_client()
    instance.shutdown()
    return built


def test_embedded_client_sends_the_tenant_api_key(provider, monkeypatch):
    """A daemon running a tenant extension answers 401 to a keyless client (#5023)."""
    cfg = {"mode": "local_embedded", "tenant_api_key": "tenant-key"}

    kwargs = _embedded_client_kwargs(provider, monkeypatch, cfg)

    assert kwargs == {"base_url": "http://127.0.0.1:1", "api_key": "tenant-key"}


def test_embedded_client_sends_the_tenant_api_key_from_the_secret_scope(provider, monkeypatch):
    """~/.hermes/.env feeds the plugin through the secret scope: the documented way to set it."""
    SECRETS["HINDSIGHT_API_TENANT_API_KEY"] = "scoped-key"

    kwargs = _embedded_client_kwargs(provider, monkeypatch, {"mode": "local_embedded"})

    assert kwargs == {"base_url": "http://127.0.0.1:1", "api_key": "scoped-key"}


def test_embedded_client_never_sends_the_cloud_api_key(provider, monkeypatch):
    """The Cloud credential is not the daemon's tenant key; it must not leave for the daemon."""
    kwargs = _embedded_client_kwargs(provider, monkeypatch, {"mode": "local_embedded", "apiKey": "cloud-key"})

    assert kwargs == {"base_url": "http://127.0.0.1:1"}


def test_the_tenant_api_key_survives_profile_env_reconciliation(hermes_env):
    """The profile env is rewritten from config on every change; the tenant settings must be in it
    or a rewrite strips them and the daemon boots unauthenticated (#5023)."""
    env = _build_embedded_profile_env({"tenant_api_key": "tenant-key"}, llm_api_key="sk")

    assert env["HINDSIGHT_API_TENANT_API_KEY"] == "tenant-key"
    assert env["HINDSIGHT_API_TENANT_EXTENSION"] == "hindsight_api.extensions.builtin.tenant:ApiKeyTenantExtension"


def test_no_tenant_settings_are_written_without_a_tenant_api_key(hermes_env):
    env = _build_embedded_profile_env({}, llm_api_key="sk")

    assert not [key for key in env if "TENANT" in key]


def test_the_tenant_api_key_accepts_the_camel_case_alias(hermes_env):
    assert _embedded_tenant_api_key({"tenantApiKey": "camel"}) == "camel"


def test_a_padded_tenant_api_key_is_stripped(hermes_env):
    """The profile file is read back stripped; an unstripped build never equals it and every
    start would rewrite the file and restart the daemon."""
    assert _embedded_tenant_api_key({"tenant_api_key": "  padded \n"}) == "padded"
    assert _embedded_tenant_api_key({"tenant_api_key": "   "}) == ""


def test_a_tenant_api_key_with_a_newline_is_rejected(hermes_env):
    """The profile env is KEY=value lines: a newline would inject settings of its own."""
    with pytest.raises(ValueError, match="line break"):
        _embedded_tenant_api_key({"tenant_api_key": "key\nHINDSIGHT_API_TENANT_EXTENSION=evil"})


@pytest.mark.parametrize("separator", ["\n", "\r", "\x0b", "\x85", "\u2028"])
def test_every_line_break_the_profile_reader_splits_on_is_rejected(hermes_env, separator):
    """The file is read back with splitlines(), which breaks on more than \\n."""
    with pytest.raises(ValueError, match="line break"):
        _embedded_tenant_api_key({"tenant_api_key": f"key{separator}HINDSIGHT_API_TENANT_EXTENSION=evil"})


def test_an_explicit_tenant_api_key_is_validated_too(hermes_env):
    """The wizard hands the builder a key directly; it must not bypass the checks."""
    with pytest.raises(ValueError, match="line break"):
        _build_embedded_profile_env({}, llm_api_key="sk", tenant_api_key="a\nB=c")
    assert (
        _build_embedded_profile_env({}, llm_api_key="sk", tenant_api_key=" k ")["HINDSIGHT_API_TENANT_API_KEY"] == "k"
    )


def _write_profile_env(tmp_path, text):
    path = tmp_path / ".hindsight" / "profiles" / "hermes.env"
    path.parent.mkdir(parents=True)
    path.write_text(text)
    return path


def _scopeless(monkeypatch):
    def _no_scope(name, default=""):
        raise embedded.UnscopedSecretError(name)

    monkeypatch.setattr(embedded, "get_secret", _no_scope)


def test_a_scopeless_thread_keeps_the_tenant_api_key_from_the_profile_file(hermes_env, monkeypatch):
    """The daemon-start worker has no secret scope. Building "" there would rewrite the file
    without the key and reboot the daemon unauthenticated."""
    _write_profile_env(hermes_env, "HINDSIGHT_API_TENANT_API_KEY=disk-key\n")
    _scopeless(monkeypatch)

    assert _embedded_tenant_api_key({"profile": "hermes"}) == "disk-key"
    assert _build_embedded_profile_env({"profile": "hermes"})["HINDSIGHT_API_TENANT_API_KEY"] == "disk-key"


def test_removing_the_tenant_api_key_where_a_scope_is_visible_turns_auth_off(hermes_env):
    """The disk fallback is for scopeless threads only, or auth could never be removed."""
    _write_profile_env(hermes_env, "HINDSIGHT_API_TENANT_API_KEY=old-key\n")

    assert _embedded_tenant_api_key({"profile": "hermes"}) == ""


def _run_start_worker(monkeypatch, config, *, stop_returns=True, survives=False, cached_client=None):
    """Run the daemon-start worker against the real profile file.

    ``stop_returns`` is what _stop_daemon reports; ``survives`` is whether the daemon is still up
    afterwards (the two can disagree: is_running swallows errors and reports False)."""
    import hindsight_embed.daemon_embed_manager as dem
    from hindsight_hermes import HindsightMemoryProvider

    # The worker swaps this module global for a console on a log file; undo it at teardown.
    monkeypatch.setattr(dem, "console", getattr(dem, "console", None), raising=False)
    stopped, built = [], []
    daemon_up = {"running": True}

    def _stop(profile):
        stopped.append(profile)
        daemon_up["running"] = survives
        return stop_returns

    provider = HindsightMemoryProvider()
    provider._config = config
    provider._client = cached_client
    monkeypatch.setattr(plugin, "_daemon_is_running", lambda profile: daemon_up["running"])
    monkeypatch.setattr(plugin, "_stop_daemon", _stop)
    real_get_client = type(provider)._get_client
    monkeypatch.setattr(type(provider), "_get_client", lambda self: built.append(True))
    provider._daemon_start_worker()
    return SimpleNamespace(
        stopped=stopped, built=built, provider=provider, get_client=lambda: real_get_client(provider)
    )


def test_the_start_worker_does_not_strip_the_tenant_key_without_a_scope(hermes_env, monkeypatch):
    """The bug in #5023 problem 2: reconciliation rewrote the profile without the tenant settings
    and restarted the daemon, which came back with no authentication."""
    config = {"profile": "hermes", "llm_provider": "ollama", "llm_model": "m"}
    SECRETS["HINDSIGHT_API_TENANT_API_KEY"] = "tenant-key"
    profile_env = embedded._materialize_embedded_profile_env(config)
    before = profile_env.read_text()
    assert "HINDSIGHT_API_TENANT_API_KEY=tenant-key" in before
    _scopeless(monkeypatch)

    run = _run_start_worker(monkeypatch, config)

    assert run.built == [True]  # the worker ran to the end, so the assertions below mean something
    assert profile_env.read_text() == before
    assert run.stopped == []


def test_the_start_worker_applies_a_rotated_tenant_key_and_restarts(hermes_env, monkeypatch):
    config = {"profile": "hermes", "llm_provider": "ollama", "llm_model": "m"}
    SECRETS["HINDSIGHT_API_TENANT_API_KEY"] = "old-key"
    profile_env = embedded._materialize_embedded_profile_env(config)
    SECRETS["HINDSIGHT_API_TENANT_API_KEY"] = "new-key"

    run = _run_start_worker(monkeypatch, config)

    assert "HINDSIGHT_API_TENANT_API_KEY=new-key" in profile_env.read_text()
    assert run.stopped == ["hermes"]
    assert run.built == [True]


def _rotate_the_tenant_key(config):
    SECRETS["HINDSIGHT_API_TENANT_API_KEY"] = "old-key"
    profile_env = embedded._materialize_embedded_profile_env(config)
    before = profile_env.read_text()
    SECRETS["HINDSIGHT_API_TENANT_API_KEY"] = "new-key"
    return profile_env, before


@pytest.mark.parametrize(
    ("stop_returns", "survives"),
    [(False, True), (False, False), (True, True)],
    ids=["stop failed, daemon up", "stop failed, is_running errored", "stop reported ok, daemon up"],
)
def test_a_daemon_that_may_have_survived_a_tenant_key_change_is_never_reused(
    hermes_env, monkeypatch, stop_returns, survives
):
    """A daemon that ignores the stop keeps its old auth. Reusing it would let a newly added key
    sit on a daemon that still accepts anything. The worker's own exception is only logged, so the
    refusal has to outlive it: every later client build refuses too."""
    config = {"profile": "hermes", "llm_provider": "ollama", "llm_model": "m"}
    profile_env, before = _rotate_the_tenant_key(config)

    run = _run_start_worker(monkeypatch, config, stop_returns=stop_returns, survives=survives)

    assert run.stopped == ["hermes"]
    assert run.built == []
    with pytest.raises(RuntimeError, match="refusing to reuse"):
        run.provider._new_embedded_client()
    # The old file is back, so the next start still sees the drift and retries the restart.
    assert profile_env.read_text() == before


def test_a_client_cached_before_the_refusal_is_dropped_too(hermes_env, monkeypatch):
    """The worker reconciles while the first memory operation may already have built and cached a
    client against the old daemon. Refusing only new builds would leave that one talking to it."""
    config = {"profile": "hermes", "llm_provider": "ollama", "llm_model": "m"}
    _rotate_the_tenant_key(config)
    cached = FakeClient()

    run = _run_start_worker(monkeypatch, config, stop_returns=False, survives=True, cached_client=cached)

    assert run.provider._client is None
    assert cached.closed  # dropped AND closed: an unclosed client leaks its aiohttp session
    with pytest.raises(RuntimeError, match="refusing to reuse"):
        run.get_client()


def test_a_client_cached_after_the_refusal_is_not_handed_out(hermes_env, monkeypatch):
    """A client can also land in the cache after the flag is set (retry path, other threads)."""
    from hindsight_hermes import HindsightMemoryProvider

    provider = HindsightMemoryProvider()
    cached = provider._client = FakeClient()
    provider._embedded_auth_stale = "refusing to reuse a daemon with the old auth"

    with pytest.raises(RuntimeError, match="refusing to reuse"):
        provider._get_client()
    assert provider._client is None
    assert cached.closed


def test_the_restore_removes_the_new_profile_when_there_was_none_before(hermes_env, monkeypatch):
    """No file to put back: leaving the new one would make the next start see no drift and reuse
    the daemon that never stopped."""
    config = {"profile": "hermes", "llm_provider": "ollama", "llm_model": "m"}
    SECRETS["HINDSIGHT_API_TENANT_API_KEY"] = "new-key"
    profile_env = hermes_env / ".hindsight" / "profiles" / "hermes.env"
    assert not profile_env.exists()

    run = _run_start_worker(monkeypatch, config, stop_returns=False, survives=True)

    assert run.stopped == ["hermes"]
    assert run.built == []
    assert not profile_env.exists()


def test_a_failing_restore_does_not_leave_the_new_profile_behind(hermes_env, monkeypatch):
    profile_env = _write_profile_env(hermes_env, "HINDSIGHT_API_TENANT_API_KEY=old-key\n")

    def _boom(path, text):
        raise OSError("disk full")

    monkeypatch.setattr(embedded, "_secure_write_profile_env", _boom)
    profile_env.write_text("HINDSIGHT_API_TENANT_API_KEY=new-key\n")

    embedded._restore_profile_env(profile_env, "HINDSIGHT_API_TENANT_API_KEY=old-key\n")

    assert not profile_env.exists()


def test_a_failed_stop_for_non_tenant_drift_behaves_as_it_always_did(hermes_env, monkeypatch):
    """Only a changed tenant key makes the stop mandatory."""
    config = {"profile": "hermes", "llm_provider": "ollama", "llm_model": "m"}
    embedded._materialize_embedded_profile_env(config)

    run = _run_start_worker(monkeypatch, {**config, "llm_model": "other"}, stop_returns=False, survives=True)

    assert run.stopped == ["hermes"]
    assert run.built == [True]
    assert run.provider._embedded_auth_stale == ""


def test_the_start_worker_warns_when_it_drops_a_hand_written_tenant_key(hermes_env, monkeypatch, caplog):
    """The reporter of #5023 set the key in the profile file by hand; the rewrite removes it, and
    that must not happen silently."""
    config = {"profile": "hermes", "llm_provider": "ollama", "llm_model": "m"}
    _write_profile_env(hermes_env, "HINDSIGHT_API_TENANT_API_KEY=hand-written\n")

    with caplog.at_level("WARNING"):
        _run_start_worker(monkeypatch, config)

    assert "tenant auth is being disabled" in caplog.text
    assert "hand-written" not in caplog.text


# --- the setup wizard's profile-env write -------------------------------------------------------


def _wizard(monkeypatch, hermes_env, *, config=None, dotenv="", running=False, stop_returns=True, survives=False):
    """Run the wizard's embedded step; return the stops it issued."""
    from hindsight_hermes import setup as wizard

    stopped = []
    state = {"running": running}

    def _stop(profile):
        stopped.append(profile)
        state["running"] = survives
        return stop_returns

    monkeypatch.setattr(wizard, "_daemon_is_running", lambda profile: state["running"])
    monkeypatch.setattr(wizard, "_stop_daemon", _stop)
    env_path = hermes_env / ".env"
    env_path.write_text(dotenv)
    base = {"profile": "hermes", "llm_provider": "ollama", "llm_model": "m"}
    wizard._apply_embedded_profile_env({**base, **(config or {})}, str(hermes_env), env_path, {})
    return stopped


def _profile_env_text(hermes_env):
    return (hermes_env / ".hindsight" / "profiles" / "hermes.env").read_text()


def test_the_wizard_restarts_a_running_daemon_when_the_tenant_key_changed(hermes_env, monkeypatch):
    _wizard(monkeypatch, hermes_env, dotenv="HINDSIGHT_API_TENANT_API_KEY=old-key\n")

    stopped = _wizard(monkeypatch, hermes_env, dotenv="HINDSIGHT_API_TENANT_API_KEY=new-key\n", running=True)

    assert stopped == ["hermes"]
    assert "HINDSIGHT_API_TENANT_API_KEY=new-key" in _profile_env_text(hermes_env)


def test_the_wizard_leaves_a_running_daemon_alone_when_the_tenant_key_is_unchanged(hermes_env, monkeypatch):
    dotenv = "HINDSIGHT_API_TENANT_API_KEY=same-key\n"
    _wizard(monkeypatch, hermes_env, dotenv=dotenv)

    assert _wizard(monkeypatch, hermes_env, dotenv=dotenv, running=True) == []


def test_the_wizard_restores_the_old_profile_when_the_daemon_will_not_stop(hermes_env, monkeypatch, capsys):
    """Leaving the new file would make the next start see no drift and reuse the stale daemon."""
    _wizard(monkeypatch, hermes_env, dotenv="HINDSIGHT_API_TENANT_API_KEY=old-key\n")
    before = _profile_env_text(hermes_env)

    _wizard(
        monkeypatch,
        hermes_env,
        dotenv="HINDSIGHT_API_TENANT_API_KEY=new-key\n",
        running=True,
        stop_returns=False,
        survives=True,
    )

    assert _profile_env_text(hermes_env) == before
    assert "Could not restart" in capsys.readouterr().out


def test_the_wizard_prefers_the_config_key_like_the_runtime_does(hermes_env, monkeypatch):
    """Else the wizard writes one key and the next start rewrites the file with the other."""
    _wizard(
        monkeypatch, hermes_env, config={"tenant_api_key": "cfg-key"}, dotenv="HINDSIGHT_API_TENANT_API_KEY=env-key\n"
    )

    assert "HINDSIGHT_API_TENANT_API_KEY=cfg-key" in _profile_env_text(hermes_env)


def test_the_wizard_reports_an_invalid_tenant_key_instead_of_crashing(hermes_env, monkeypatch, capsys):
    _wizard(monkeypatch, hermes_env, config={"tenant_api_key": "a\nB=c"})

    assert "not updated" in capsys.readouterr().out
    assert not (hermes_env / ".hindsight" / "profiles" / "hermes.env").exists()


def test_the_wizard_removes_the_new_profile_when_there_was_none_before_and_the_daemon_will_not_stop(
    hermes_env, monkeypatch
):
    _wizard(
        monkeypatch,
        hermes_env,
        dotenv="HINDSIGHT_API_TENANT_API_KEY=new-key\n",
        running=True,
        stop_returns=False,
        survives=True,
    )

    assert not (hermes_env / ".hindsight" / "profiles" / "hermes.env").exists()
