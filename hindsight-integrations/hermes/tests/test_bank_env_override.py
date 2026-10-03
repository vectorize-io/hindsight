"""Documented bank overrides exercised through the Hermes provider interface."""

import json

import hindsight_hermes as plugin
import pytest
from conftest import SECRETS, UnscopedSecretError


def test_scoped_bank_override_reaches_recall_with_file_config(provider):
    SECRETS["HINDSIGHT_BANK_ID"] = "campaign-bank"
    instance, client = provider({"bank_id": "file-bank"})
    try:
        instance.handle_tool_call("hindsight_recall", {"query": "remember"})
        assert client.recalls[-1]["bank_id"] == "campaign-bank"
    finally:
        instance.shutdown()


def test_empty_bank_override_preserves_file_bank(provider):
    SECRETS["HINDSIGHT_BANK_ID"] = ""
    instance, client = provider({"banks": {"hermes": {"bankId": "legacy-file-bank"}}})
    try:
        instance.handle_tool_call("hindsight_recall", {"query": "remember"})
        assert client.recalls[-1]["bank_id"] == "legacy-file-bank"
    finally:
        instance.shutdown()


def test_explicit_bank_template_keeps_priority_over_static_override(provider):
    SECRETS["HINDSIGHT_BANK_ID"] = "static-override"
    instance, client = provider({"bank_id_template": "team-{user}"}, user_id="Ada")
    try:
        instance.handle_tool_call("hindsight_recall", {"query": "remember"})
        assert client.recalls[-1]["bank_id"] == "team-Ada"
    finally:
        instance.shutdown()


def test_secondary_profile_never_inherits_default_profile_process_bank(provider, monkeypatch):
    monkeypatch.setenv("HINDSIGHT_BANK_ID", "default-profile-bank")
    instance, client = provider({"bank_id": "secondary-bank"})
    try:
        instance.handle_tool_call("hindsight_recall", {"query": "remember"})
        assert client.recalls[-1]["bank_id"] == "secondary-bank"
    finally:
        instance.shutdown()


def test_unscoped_bank_override_fails_closed_with_file_config(hermes_env, monkeypatch):
    path = hermes_env / "hindsight" / "config.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"mode": "cloud", "apiKey": "test-key", "bank_id": "file-bank"}))
    real_get_secret = plugin.get_secret

    def scoped_read(name, default=""):
        if name == "HINDSIGHT_BANK_ID":
            raise UnscopedSecretError("missing profile scope")
        return real_get_secret(name, default)

    monkeypatch.setattr(plugin, "get_secret", scoped_read)
    instance = plugin.HindsightMemoryProvider()
    with pytest.raises(UnscopedSecretError, match="missing profile scope"):
        instance.initialize("session-1")
