"""Pure config normalizers — no Hermes, no network."""

from datetime import datetime, timezone
from types import SimpleNamespace

from hindsight_hermes.settings import (
    _memory_date_prefix,
    _normalize_observation_scopes,
    _normalize_retain_tags,
    _parse_int_setting,
    _resolve_bank_id_template,
)


def test_retain_tags_accepts_csv_json_and_lists():
    assert _normalize_retain_tags("a, b ,a") == ["a", "b"]
    assert _normalize_retain_tags('["a", "b"]') == ["a", "b"]
    assert _normalize_retain_tags(["a", "a", "b"]) == ["a", "b"]
    assert _normalize_retain_tags(None) == []


def test_parse_int_setting_falls_back_on_garbage():
    assert _parse_int_setting("30", 120) == 30
    assert _parse_int_setting("", 120) == 120
    assert _parse_int_setting("nope", 120) == 120
    assert _parse_int_setting(0, 120) == 0


def test_bank_id_template_sanitizes_and_collapses_empty_placeholders():
    assert _resolve_bank_id_template("hermes-{profile}", "hermes", profile="My Bot") == "hermes-My-Bot"
    assert _resolve_bank_id_template("hermes-{user}", "hermes", user="") == "hermes"
    assert _resolve_bank_id_template("", "hermes", user="x") == "hermes"
    # An unknown placeholder must not blow up the session — fall back to the static bank.
    assert _resolve_bank_id_template("hermes-{nope}", "hermes") == "hermes"


def test_memory_date_prefix_prefers_mentioned_at_then_occurred_start():
    both = SimpleNamespace(mentioned_at="2026-09-22T09:00:00Z", occurred_start="2026-08-15T00:00:00Z")
    assert _memory_date_prefix(both) == "[2026-09-22] "
    assert _memory_date_prefix(SimpleNamespace(mentioned_at=None, occurred_start="2026-08-15")) == "[2026-08-15] "
    # A parsed datetime keeps working should the client stop returning strings.
    parsed = SimpleNamespace(mentioned_at=datetime(2026, 6, 3, 14, 22, tzinfo=timezone.utc))
    assert _memory_date_prefix(parsed) == "[2026-06-03] "


def test_memory_date_prefix_is_empty_without_a_usable_date():
    assert _memory_date_prefix(SimpleNamespace(text="undated")) == ""
    assert _memory_date_prefix(SimpleNamespace(mentioned_at=None, occurred_start=None)) == ""
    # Too short to hold a date: no prefix rather than a garbled one.
    assert _memory_date_prefix(SimpleNamespace(mentioned_at="2026", occurred_start="")) == ""


def test_observation_scopes_normalization():
    assert _normalize_observation_scopes("per_tag") == "per_tag"
    assert _normalize_observation_scopes(["a", "b"]) == [["a", "b"]]
    assert _normalize_observation_scopes([["a"], ["b"]]) == [["a"], ["b"]]
    assert _normalize_observation_scopes("garbage") is None
