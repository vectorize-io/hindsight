"""Pure config normalizers — no Hermes, no network."""

import pytest

from hindsight_hermes.settings import (
    _normalize_observation_scopes,
    _normalize_retain_tags,
    _parse_float_setting,
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


def test_observation_scopes_normalization():
    assert _normalize_observation_scopes("per_tag") == "per_tag"
    assert _normalize_observation_scopes(["a", "b"]) == [["a", "b"]]
    assert _normalize_observation_scopes([["a"], ["b"]]) == [["a"], ["b"]]
    assert _normalize_observation_scopes("garbage") is None


def test_parse_float_setting_accepts_valid_values():
    assert _parse_float_setting(3.5, 5.0, 60.0) == 3.5
    assert _parse_float_setting("2.5", 5.0, 60.0) == 2.5
    assert _parse_float_setting(3, 5.0, 60.0) == 3.0
    # 0 is a valid timeout (non-blocking join), not an "unset" sentinel.
    assert _parse_float_setting(0, 5.0, 60.0) == 0.0


def test_parse_float_setting_falls_back_on_invalid():
    assert _parse_float_setting(None, 5.0, 60.0) == 5.0
    assert _parse_float_setting("", 5.0, 60.0) == 5.0
    assert _parse_float_setting("not-a-number", 5.0, 60.0) == 5.0
    # bool is an int subclass; a flag must not become a 1s timeout.
    assert _parse_float_setting(True, 5.0, 60.0) == 5.0


def test_parse_float_setting_falls_back_on_out_of_range():
    assert _parse_float_setting(-1.0, 5.0, 60.0) == 5.0
    assert _parse_float_setting("-3", 5.0, 60.0) == 5.0
    assert _parse_float_setting(100.0, 5.0, 60.0) == 5.0
    # 1e20 parses to a finite float but is far beyond any sane join timeout;
    # the range check must reject it.
    assert _parse_float_setting("1e20", 5.0, 60.0) == 5.0


@pytest.mark.parametrize("value", ["nan", "NaN", "inf", "Infinity", "-inf", "1e309"])
def test_parse_float_setting_rejects_non_finite_strings(value):
    # nan/non-finite (including float-parse spellings and 1e309 overflow) must
    # not reach the join timeout -- an inf wait is an unbounded wait.
    assert _parse_float_setting(value, 5.0, 60.0) == 5.0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_parse_float_setting_rejects_non_finite_floats(value):
    assert _parse_float_setting(value, 5.0, 60.0) == 5.0


def test_parse_float_setting_clamps_default_and_keeps_max_boundary():
    assert _parse_float_setting(None, 100.0, 60.0) == 60.0  # default clamped into range
    assert _parse_float_setting(60.0, 5.0, 60.0) == 60.0  # max is inclusive
