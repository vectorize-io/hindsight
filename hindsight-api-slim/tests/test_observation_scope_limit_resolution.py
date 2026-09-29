"""Unit tests for per-scope observation-limit resolution.

These cover the three pure helpers behind the ``observation_scope_limits``
config field, which lets a bank cap observations differently per consolidation
scope (e.g. one tag's scope unlimited, while scopes that also carry a wildcard
tag are capped):

- ``_scope_matches_globs``  — exact-cover match between a glob pattern and a
  concrete tag set (the crux: ``{a}`` and ``{run_1, a}`` must resolve to
  *different* rules even though both contain ``a``).
- ``_parse_scope_limit_rules`` — defensive parsing of the raw JSON config.
- ``_effective_scope_limit`` — first-match-wins resolution with fallback to the
  bank-wide ``max_observations_per_scope``.

All deterministic — direct asserts, no LLM.
"""

from types import SimpleNamespace

import pytest

from hindsight_api.engine.consolidation.consolidator import (
    _effective_scope_limit,
    _parse_scope_limit_rules,
    _scope_matches_globs,
    _ScopeLimitRule,
)


def _config(scope_limits, default=50):
    """A minimal stand-in for the resolved HindsightConfig fields we read."""
    return SimpleNamespace(
        observation_scope_limits=scope_limits,
        max_observations_per_scope=default,
    )


# ---------------------------------------------------------------------------
# _scope_matches_globs — exact cover (every tag covered, every glob used)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "globs,tags,expected",
    [
        # Literal single-tag scope: matches only the exact set.
        (("shared",), ["shared"], True),
        (("shared",), ["run_1", "shared"], False),  # run_1 uncovered
        (("shared",), [], False),  # untagged never matches
        # Wildcard + literal combined scope.
        (("run_*", "shared"), ["run_1", "shared"], True),
        (("run_*", "shared"), ["shared"], False),  # run_* glob is vacuous
        (("run_*", "shared"), ["run_1"], False),  # shared glob is vacuous
        (("run_*", "shared"), ["run_1", "shared", "extra"], False),  # extra uncovered
        # One glob may cover several tags (still exact cover).
        (("run_*", "shared"), ["run_1", "run_2", "shared"], True),
        # Catch-all single glob matches any tagged scope but not the untagged one.
        (("*",), ["anything"], True),
        (("*",), ["a", "b"], True),
        (("*",), [], False),
        # Matching is case-sensitive.
        (("SHARED",), ["shared"], False),
    ],
)
def test_scope_matches_globs_exact_cover(globs, tags, expected):
    assert _scope_matches_globs(globs, tags) is expected


def test_scope_matches_globs_is_order_independent():
    # Tags are a set; pattern order must not change the verdict.
    assert _scope_matches_globs(("run_*", "shared"), ["shared", "run_9"]) is True
    assert _scope_matches_globs(("shared", "run_*"), ["run_9", "shared"]) is True


# ---------------------------------------------------------------------------
# _parse_scope_limit_rules — defensive, order-preserving
# ---------------------------------------------------------------------------


def test_parse_rules_happy_path_preserves_order():
    raw = [
        {"scope": ["shared"], "limit": -1},
        {"scope": ["run_*", "shared"], "limit": 1},
    ]
    rules = _parse_scope_limit_rules(raw)
    assert rules == [
        _ScopeLimitRule(globs=("shared",), limit=-1),
        _ScopeLimitRule(globs=("run_*", "shared"), limit=1),
    ]


@pytest.mark.parametrize("raw", [None, "not-a-list", 42, {}, {"scope": ["shared"], "limit": 1}])
def test_parse_rules_non_list_yields_empty(raw):
    assert _parse_scope_limit_rules(raw) == []


@pytest.mark.parametrize(
    "entry",
    [
        "string-entry",  # not a dict
        {"limit": 1},  # missing scope
        {"scope": ["a"]},  # missing limit
        {"scope": [], "limit": 1},  # empty scope
        {"scope": "a", "limit": 1},  # scope not a list
        {"scope": ["a", 7], "limit": 1},  # non-str glob
        {"scope": ["a", ""], "limit": 1},  # empty glob string
        {"scope": ["a"], "limit": "1"},  # limit not an int
        {"scope": ["a"], "limit": True},  # bool masquerading as int
    ],
)
def test_parse_rules_skips_malformed_entries(entry):
    # Malformed entries are dropped; a following valid entry still parses.
    raw = [entry, {"scope": ["ok"], "limit": 3}]
    assert _parse_scope_limit_rules(raw) == [_ScopeLimitRule(globs=("ok",), limit=3)]


# ---------------------------------------------------------------------------
# _effective_scope_limit — first match wins, else bank default
# ---------------------------------------------------------------------------


def test_effective_limit_literal_vs_wildcard_scope():
    """Literal scope unlimited, wildcard+literal scope capped, everything else default."""
    config = _config(
        [
            {"scope": ["shared"], "limit": -1},
            {"scope": ["run_*", "shared"], "limit": 1},
        ],
        default=50,
    )
    assert _effective_scope_limit(config, ["shared"]) == -1
    assert _effective_scope_limit(config, ["run_42", "shared"]) == 1
    assert _effective_scope_limit(config, ["some_other_tag"]) == 50  # fallback
    assert _effective_scope_limit(config, []) == 50  # untagged → fallback (no rule matches)


def test_effective_limit_first_match_wins():
    # A broad catch-all placed first shadows a more specific later rule.
    config = _config(
        [
            {"scope": ["*"], "limit": 5},
            {"scope": ["shared"], "limit": -1},
        ],
        default=50,
    )
    assert _effective_scope_limit(config, ["shared"]) == 5


def test_effective_limit_falls_back_when_no_rules():
    assert _effective_scope_limit(_config(None, default=7), ["shared"]) == 7
    assert _effective_scope_limit(_config([], default=7), ["shared"]) == 7


def test_effective_limit_none_config_is_unlimited():
    # Mirrors the old `config is None` branch at the call site.
    assert _effective_scope_limit(None, ["shared"]) == -1
