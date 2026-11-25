"""Per-chat config: a ``chats`` entry keyed ``platform[/chat_id[/thread_id]]`` laid over the top level."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from conftest import PLUGIN_ROOT, SECRETS, FakeClient, plugin

from hindsight_hermes.settings import ChatScope, resolve_chat_config

TOPIC = {"platform": "telegram", "chat_id": "-100123", "thread_id": "42"}


def test_candidates_go_from_the_thread_to_the_platform():
    assert ChatScope("telegram", "-100123", "42").candidates() == [
        "telegram/-100123/42",
        "telegram/-100123",
        "telegram",
    ]
    assert ChatScope("cli").candidates() == ["cli"]
    assert ChatScope("telegram", "-100123", "42").key == "telegram/-100123/42"


def test_an_id_containing_a_slash_never_matches_a_shorter_key():
    resolved = resolve_chat_config({"chats": {"signal/ab": {"auto_recall": False}}}, ChatScope("signal", "ab/cd"))
    assert resolved.matched is None
    assert resolve_chat_config({"chats": {"signal/ab/cd": {"x": 1}}}, ChatScope("signal", "ab/cd")).matched


def test_only_the_most_specific_entry_applies():
    raw = {
        "auto_recall": True,
        "recall_budget": "mid",
        "chats": {"telegram": {"recall_budget": "low"}, "telegram/-100123": {"auto_recall": False}},
    }
    resolved = resolve_chat_config(raw, ChatScope("telegram", "-100123", "42"))
    assert resolved.matched == "telegram/-100123"
    # No stacking: the platform entry's budget is not inherited by the chat entry.
    assert resolved.values == {"auto_recall": False, "recall_budget": "mid"}


def test_no_match_or_no_chats_is_the_top_level_without_the_map():
    assert resolve_chat_config({"bank_id": "b"}, ChatScope("cli")).values == {"bank_id": "b"}
    resolved = resolve_chat_config({"bank_id": "b", "chats": {"discord": {}}}, ChatScope("telegram", "1"))
    assert resolved.values == {"bank_id": "b"} and resolved.matched is None


@pytest.mark.parametrize("chats", ["telegram", ["telegram"], {"telegram": "off"}])
def test_a_malformed_map_or_entry_is_ignored(chats):
    resolved = resolve_chat_config({"bank_id": "b", "chats": chats}, ChatScope("telegram"))
    assert resolved.values == {"bank_id": "b"} and resolved.overrides is None


def test_an_entry_replaces_the_setting_under_every_name():
    """``[]`` in the entry must clear a list the top level set under its other name."""
    raw = {"recallAdditionalBanks": ["vault"], "budget": "high", "chats": {"telegram": {"recall_additional_banks": []}}}
    assert resolve_chat_config(raw, ChatScope("telegram")).values == {"budget": "high", "recall_additional_banks": []}


def test_nested_chats_are_dropped():
    raw = {"chats": {"telegram": {"auto_recall": False, "chats": {"telegram/1": {}}}}}
    assert resolve_chat_config(raw, ChatScope("telegram", "1")).values == {"auto_recall": False}


# -- through initialize ------------------------------------------------------------------------


def test_a_platform_entry_turns_auto_recall_off_there_only(provider):
    """#5490: no auto-recall on a capture channel, kept on the DM of the same profile."""
    config = {"auto_recall": True, "chats": {"pocket": {"auto_recall": False}}}
    pocket, _ = provider(config, platform="pocket", chat_id="capture-1")
    discord, _ = provider(config, platform="discord", chat_id="555")
    assert pocket._auto_recall is False and discord._auto_recall is True


def test_a_chat_bank_beats_the_env_bank_and_the_top_level_template(provider):
    """#5723: the chat names its bank; neither HINDSIGHT_BANK_ID nor bank_id_template overrides it."""
    SECRETS["HINDSIGHT_BANK_ID"] = "env-bank"
    config = {"bank_id_template": "hermes-{platform}", "chats": {"telegram/-100123/42": {"bank_id": "home"}}}
    instance, fake = provider(config, **TOPIC)
    assert (instance._bank_id, instance._bank_source) == ("home", "chats['telegram/-100123/42']")
    instance.handle_tool_call("hindsight_retain", {"content": "the NAS runs TrueNAS"})
    assert [r["bank_id"] for r in fake.retains] == ["home"]
    instance.shutdown()


def test_a_chat_template_names_a_bank_per_thread(provider):
    config = {"bank_id": "personal", "chats": {"telegram/-100123": {"bank_id_template": "project-a-{thread}"}}}
    topic, _ = provider(config, **TOPIC)
    no_thread, _ = provider(config, platform="telegram", chat_id="-100123")
    elsewhere, _ = provider(config, platform="telegram", chat_id="-100999")
    assert topic._bank_id == "project-a-42"
    assert no_thread._bank_id == "project-a"
    assert elsewhere._bank_id == "personal"


def test_recall_only_banks_for_one_chat(provider):
    """#6599: a business chat also reads that business's bank; other chats don't."""
    config = {
        "bank_id": "fleet",
        "recall_sync": True,
        "chats": {"telegram/-100555": {"recall_additional_banks": ["peptides"]}},
    }
    shop, shop_client = provider(config, client=FakeClient(recall_texts=["x"]), platform="telegram", chat_id="-100555")
    other, other_client = provider(
        config, client=FakeClient(recall_texts=["x"]), platform="telegram", chat_id="-100777"
    )
    shop.handle_tool_call("hindsight_recall", {"query": "shipping"})
    other.handle_tool_call("hindsight_recall", {"query": "shipping"})
    assert sorted(r["bank_id"] for r in shop_client.recalls) == ["fleet", "peptides"]
    assert [r["bank_id"] for r in other_client.recalls] == ["fleet"]
    assert shop._write_bank_ids == ["fleet"]  # read, never written
    shop.shutdown()
    other.shutdown()


def test_tag_placeholders_follow_the_chat(provider):
    """#5073: one bank, soft-partitioned by topic."""
    config = {
        "retain_tags": ["hermes", "topic:{thread}", "chat:{chat}"],
        "recall_tags": ["topic:{thread}"],
        "reflect_tags": ["topic:{thread}", "{nope}"],
    }
    topic, _ = provider(config, **TOPIC)
    assert topic._retain_tags == ["hermes", "topic:42", "chat:-100123"]
    assert topic._recall_tags == ["topic:42"]
    assert topic._reflect_tags == ["topic:42", "{nope}"]  # unknown placeholder: kept as written
    dm, _ = provider(config, platform="discord", chat_id="555")
    assert dm._retain_tags == ["hermes", "chat:555"]  # empty {thread}: that tag is dropped
    assert dm._recall_tags is None


def test_the_log_line_names_the_chat_key_to_copy(provider, caplog):
    caplog.set_level("INFO")
    provider({"chats": {"telegram/-100123": {"bank_id": "a"}}}, **TOPIC)
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("Hindsight initialized"))
    assert "chat=telegram/-100123/42 (chats['telegram/-100123'])" in line
    assert "bank=a (from chats['telegram/-100123'])" in line


# -- by design: every setting, including ones added later, can be set per chat ----------------


def _functions_calling(name: str) -> set[str]:
    callers: set[str] = set()
    for path in PLUGIN_ROOT.glob("*.py"):
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for node in ast.walk(fn):
                    if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", "")) == name:
                        callers.add(f"{path.stem}.{fn.name}")
    return callers


def test_only_the_resolver_and_the_file_level_checks_read_the_config_file():
    """Settings come from ``self._config``, which ``initialize`` builds with the chat entry merged
    in before any setting is read. A new setting therefore works per chat with no extra code, as
    long as nothing reads the file around that path; this pins the places that may."""
    assert _functions_calling("_load_config") == {
        "__init__.initialize",  # -> resolve_chat_config; everything else reads its result
        "__init__.is_available",  # whether the plugin is configured at all: the top level decides
        "__init__.unavailable_reason",
        "setup.run_setup",  # edits the file, so it must see the file
    }


def test_an_entry_can_point_one_chat_at_another_server(provider):
    config = {"chats": {"slack": {"mode": "local_external", "api_url": "http://team-hindsight:8888"}}}
    slack, _ = provider(config, platform="slack", chat_id="C1")
    cli, _ = provider(config)
    assert (slack._mode, slack._api_url) == ("local_external", "http://team-hindsight:8888")
    assert cli._mode == "cloud"


def test_any_key_reaches_the_provider_config(provider):
    instance, _ = provider({"chats": {"telegram": {"a_setting_added_next_year": 7}}}, platform="telegram")
    assert instance._config["a_setting_added_next_year"] == 7
    assert instance._config is instance._chat_route.values
    assert plugin.HindsightMemoryProvider()._chat_route.values == {}
