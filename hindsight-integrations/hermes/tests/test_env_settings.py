"""Settings a profile can give through its environment (``.env``) when it has no
``hindsight/config.json``: the retain label, both status-line indicators and synchronous recall."""

import hindsight_hermes as plugin
from conftest import SECRETS, FakeClient


def _provider_from_env(monkeypatch, **env) -> plugin.HindsightMemoryProvider:
    SECRETS.update({"HINDSIGHT_MODE": "cloud", "HINDSIGHT_API_KEY": "test-key", **env})
    instance = plugin.HindsightMemoryProvider()
    monkeypatch.setattr(instance, "_new_cloud_client", lambda: FakeClient())
    monkeypatch.setattr(plugin, "_check_api_supports_update_mode_append", lambda *a, **k: True)
    instance.initialize("session-1")
    return instance


def test_the_environment_sets_the_retain_label_the_indicators_and_sync_recall(hermes_env, monkeypatch):
    instance = _provider_from_env(
        monkeypatch,
        HINDSIGHT_RETAIN_CONTEXT="conversation between the support bot and a customer",
        HINDSIGHT_RETAIN_INDICATOR="false",
        HINDSIGHT_RECALL_INDICATOR="0",
        HINDSIGHT_RECALL_SYNC="true",
    )
    assert instance._retain_context == "conversation between the support bot and a customer"
    assert instance._retain_indicator is False
    assert instance._recall_indicator is False
    assert instance._recall_sync is True
    instance.shutdown()


def test_unset_variables_keep_the_defaults(hermes_env, monkeypatch):
    instance = _provider_from_env(monkeypatch)
    assert instance._retain_context == plugin._RETAIN_CONTEXT_DEFAULT
    assert instance._retain_indicator is True
    assert instance._recall_indicator is True
    assert instance._recall_sync is False
    instance.shutdown()


def test_flags_read_the_usual_spellings():
    for value, expected in {
        "true": True,
        "TRUE": True,
        "1": True,
        "yes": True,
        "on": True,
        "false": False,
        "0": False,
        "no": False,
        "off": False,
    }.items():
        SECRETS["HINDSIGHT_RECALL_SYNC"] = value
        assert plugin._scoped_flag("HINDSIGHT_RECALL_SYNC", not expected) is expected
    SECRETS.pop("HINDSIGHT_RECALL_SYNC")
    assert plugin._scoped_flag("HINDSIGHT_RECALL_SYNC", True) is True


def test_a_config_file_still_takes_precedence(provider):
    SECRETS.update({"HINDSIGHT_RECALL_SYNC": "true", "HINDSIGHT_RETAIN_INDICATOR": "false"})
    instance, _ = provider({"recall_sync": False})
    assert instance._recall_sync is False
    assert instance._retain_indicator is True
    instance.shutdown()
