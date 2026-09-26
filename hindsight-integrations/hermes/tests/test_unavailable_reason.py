"""``unavailable_reason()``: the hint the host appends to its "provider unavailable" warning.

``is_available()`` is a side-effect-free hot-path check that cannot log, and ``initialize()`` never
runs when it returns False, so this hint is the only thing the user ever sees. It has to name the
variables to set — for cloud mode the host's warning text explains *why* they can be missing (a
gateway/systemd unit does not inherit ``~/.hermes/.env``) but not *which* ones.
NousResearch/hermes-agent#86078.
"""

import json
from pathlib import Path

import hindsight_hermes as plugin
from conftest import SECRETS


def _config_path(hermes_home: Path) -> Path:
    return hermes_home / "hindsight" / "config.json"


def _write_config(hermes_home: Path, config: dict) -> None:
    path = _config_path(hermes_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config))


def _provider() -> plugin.HindsightMemoryProvider:
    return plugin.HindsightMemoryProvider()


def test_cloud_mode_without_endpoint_or_key_names_both_variables(hermes_env):
    _write_config(hermes_env, {"mode": "cloud"})

    instance = _provider()
    assert instance.is_available() is False
    reason = instance.unavailable_reason()
    assert "HINDSIGHT_API_URL" in reason
    assert "HINDSIGHT_API_KEY" in reason
    assert str(_config_path(hermes_env)) in reason


def test_cloud_mode_with_a_config_key_has_no_hint(hermes_env):
    _write_config(hermes_env, {"mode": "cloud", "apiKey": "cloud-key"})

    instance = _provider()
    assert instance.is_available() is True
    assert instance.unavailable_reason() == ""


def test_cloud_mode_with_a_config_endpoint_has_no_hint(hermes_env):
    _write_config(hermes_env, {"mode": "cloud", "api_url": "https://cloud.hindsight.example"})

    instance = _provider()
    assert instance.is_available() is True
    assert instance.unavailable_reason() == ""


def test_cloud_mode_with_the_key_in_the_secret_scope_has_no_hint(hermes_env):
    _write_config(hermes_env, {"mode": "cloud"})
    SECRETS["HINDSIGHT_API_KEY"] = "scoped-key"

    instance = _provider()
    assert instance.is_available() is True
    assert instance.unavailable_reason() == ""


def test_local_external_keeps_its_own_availability_rule(hermes_env):
    """local_external is available without cloud credentials, so it must stay hint-free."""
    _write_config(hermes_env, {"mode": "local_external"})

    instance = _provider()
    assert instance.is_available() is True
    assert instance.unavailable_reason() == ""


def test_local_embedded_keeps_the_install_hint(hermes_env, monkeypatch):
    _write_config(hermes_env, {"mode": "local_embedded"})
    monkeypatch.setattr(
        plugin, "_check_local_runtime", lambda: (False, "No module named 'hindsight'")
    )

    reason = _provider().unavailable_reason()
    assert "hindsight-all" in reason


def test_hint_is_stripped_because_the_host_prefixes_it(hermes_env):
    """agent_init renders ``f" {reason}"``, so a leading space would double up."""
    _write_config(hermes_env, {"mode": "cloud"})

    reason = _provider().unavailable_reason()
    assert reason
    assert reason == reason.strip()
