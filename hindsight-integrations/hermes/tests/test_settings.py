"""Pure config normalizers — no Hermes, no network."""

from hindsight_hermes.settings import (
    _normalize_observation_scopes,
    _normalize_retain_tags,
    _parse_int_setting,
    _resolve_bank_id_for_user,
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


def test_observation_scopes_normalization():
    assert _normalize_observation_scopes("per_tag") == "per_tag"
    assert _normalize_observation_scopes(["a", "b"]) == [["a", "b"]]
    assert _normalize_observation_scopes([["a"], ["b"]]) == [["a"], ["b"]]
    assert _normalize_observation_scopes("garbage") is None


def test_per_user_bank_override_matches_exact_platform_user():
    assert _resolve_bank_id_for_user(
        "education", {"dingtalk:user-a": "jiayin_learning"}, "dingtalk", "user-a"
    ) == "jiayin_learning"


def test_per_user_bank_override_does_not_cross_platforms():
    mapping = {"dingtalk:user-a": "jiayin_learning"}
    assert _resolve_bank_id_for_user("education", mapping, "slack", "user-a") == "education"


def test_per_user_bank_override_raw_key_only_without_platform():
    mapping = {"user-a": "jiayin_learning"}
    assert _resolve_bank_id_for_user("education", mapping, "", "user-a") == "jiayin_learning"
    assert _resolve_bank_id_for_user("education", mapping, "dingtalk", "user-a") == "education"


def test_per_user_bank_override_malformed_or_unknown_falls_back():
    assert _resolve_bank_id_for_user("education", None, "dingtalk", "user-a") == "education"
    assert _resolve_bank_id_for_user("education", [], "dingtalk", "user-a") == "education"
    assert _resolve_bank_id_for_user("education", {"dingtalk:user-a": ""}, "dingtalk", "user-a") == "education"
    assert _resolve_bank_id_for_user("education", {"dingtalk:user-a": 7}, "dingtalk", "user-a") == "education"
    assert _resolve_bank_id_for_user("education", {"dingtalk:user-b": "other"}, "dingtalk", "user-a") == "education"


def test_per_user_bank_override_accepts_json_text_and_preserves_bank_id():
    mapping = '{"dingtalk:user-a": "tenant:alice"}'
    assert _resolve_bank_id_for_user("education", mapping, "dingtalk", "user-a") == "tenant:alice"
    assert _resolve_bank_id_for_user("education", mapping, "slack", "user-a") == "education"


def test_per_user_bank_override_never_collapses_distinct_explicit_bank_ids():
    assert _resolve_bank_id_for_user("education", {"dingtalk:a": "tenant:alice"}, "dingtalk", "a") == "tenant:alice"
    assert _resolve_bank_id_for_user("education", {"dingtalk:b": "tenant-alice"}, "dingtalk", "b") == "tenant-alice"


def test_per_user_bank_override_preserves_explicit_bank_ids_without_collision():
    mapping = {
        "dingtalk:user-a": "tenant:alice",
        "dingtalk:user-b": "tenant-alice",
    }
    assert _resolve_bank_id_for_user("education", mapping, "dingtalk", "user-a") == "tenant:alice"
    assert _resolve_bank_id_for_user("education", mapping, "dingtalk", "user-b") == "tenant-alice"


def test_per_user_bank_override_accepts_json_text_for_older_config_ui():
    mapping = '{"dingtalk:user-a": "tenant:alice"}'
    assert _resolve_bank_id_for_user("education", mapping, "dingtalk", "user-a") == "tenant:alice"
    assert _resolve_bank_id_for_user("education", "not-json", "dingtalk", "user-a") == "education"
