"""Unit tests for per-scope consolidation strategies.

``consolidation_strategies`` lets one bank be federated across user / team /
company tag scopes where each scope consolidates under its *own* settings — the
company-wide scope can be told to record only generalized trends (and to keep
fewer observations) while the per-user scope keeps the specifics.

A strategy lists rules (``scopes``: ``{"tags": [...], "tags_match": ...}``) and
claims a concrete scope when *any* rule matches it: under ``"all"`` (default) the
scope must carry every tag in the rule and may carry others; under ``"exact"``
it must carry exactly those. When several strategies claim the same scope, the
first in the list wins, whole — see "Two strategies claiming one scope" below.
It supersedes the deprecated ``observation_scope_limits``, which is still
honoured but consulted only afterwards; the "exact" matcher is shared with it
(see ``test_observation_scope_limit_resolution.py`` for ``_scope_matches_globs``).

What is tested here is the parsing, multi-rule claiming, the two match modes,
which strategy wins a contested scope, and the precedence over the deprecated
field.

All deterministic — direct asserts, no LLM.
"""

from types import SimpleNamespace

import pytest

from hindsight_api.engine.consolidation.consolidator import (
    _ConsolidationStrategy,
    _ScopePattern,
    _config_for_scope,
    _effective_scope_limit,
    _parse_consolidation_strategies,
)

GENERIC = "Record only industry-level trends. Name no specific company."
BANK_MISSION = "Record what the user worked on."


BANK_SOURCE_FACTS = 4096
BANK_SOURCE_FACTS_PER_OBS = 256


def _config(strategies, *, scope_limits=None, mission=BANK_MISSION, max_obs=50):
    """A minimal stand-in for the resolved HindsightConfig fields we read."""
    return SimpleNamespace(
        consolidation_strategies=strategies,
        observation_scope_limits=scope_limits,
        observations_mission=mission,
        max_observations_per_scope=max_obs,
        consolidation_source_facts_max_tokens=BANK_SOURCE_FACTS,
        consolidation_source_facts_max_tokens_per_observation=BANK_SOURCE_FACTS_PER_OBS,
    )


def _mission(config, tags):
    return _config_for_scope(config, tags).observations_mission


# ---------------------------------------------------------------------------
# _parse_consolidation_strategies — defensive parsing of the raw JSON config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "not-a-list",
        {},
        [None],
        ["string-entry"],
        [{"scopes": [], "observations_mission": GENERIC}],  # claims nothing
        [{"scopes": [{"tags": []}], "observations_mission": GENERIC}],  # the one scope is empty
        [{"scopes": ["company:*"], "observations_mission": GENERIC}],  # patterns must be objects
        [{"scopes": [["company:*"]], "observations_mission": GENERIC}],  # the old list-of-lists form
        [{"scopes": [{"tag": ["company:*"]}], "observations_mission": GENERIC}],  # no "tags" key
        [{"scopes": "company:*", "observations_mission": GENERIC}],  # scopes not a list
        [{"scopes": [{"tags": ["company:*", 3]}], "observations_mission": GENERIC}],  # non-string glob
        [{"observations_mission": GENERIC}],  # no scopes at all
        [{"scopes": [{"tags": ["company:*"]}]}],  # overrides nothing
        [{"scopes": [{"tags": ["company:*"]}], "observations_mission": ""}],  # mission empty
        [{"scopes": [{"tags": ["company:*"]}], "observations_mission": "   "}],  # mission blank
        [{"scopes": [{"tags": ["company:*"]}], "observations_mission": 42}],  # mission not a string
        [{"scopes": [{"tags": ["company:*"]}], "max_observations_per_scope": "20"}],  # cap not an int
        [{"scopes": [{"tags": ["company:*"]}], "max_observations_per_scope": True}],  # bool is not a cap
        [{"scopes": [{"tags": ["company:*"]}], "consolidation_source_facts_max_tokens": "2048"}],  # not an int
        [{"scopes": [{"tags": ["company:*"]}], "consolidation_source_facts_max_tokens_per_observation": False}],
        [{"scopes": [{"tags": ["company:*"], "tags_match": "exact"}]}],  # a mode alone overrides nothing
    ],
)
def test_malformed_entries_are_skipped_not_raised(raw):
    """Config round-trips as JSON through env and the bank-config API, so a bad
    entry must never take consolidation down."""
    assert _parse_consolidation_strategies(raw) == []


def test_a_strategy_may_carry_either_setting_or_both():
    raw = [
        {"scopes": [{"tags": ["company:*"]}], "observations_mission": GENERIC},
        {"scopes": [{"tags": ["team:*"]}], "max_observations_per_scope": 20},
        {
            "scopes": [{"tags": ["user:*"]}],
            "observations_mission": "User brief.",
            "max_observations_per_scope": 5,
        },
        {"scopes": [{"tags": ["bad"]}]},  # overrides nothing, skipped without shifting the rest
    ]
    assert _parse_consolidation_strategies(raw) == [
        _ConsolidationStrategy((_ScopePattern(("company:*",)),), GENERIC, None),
        _ConsolidationStrategy((_ScopePattern(("team:*",)),), None, 20),
        _ConsolidationStrategy((_ScopePattern(("user:*",)),), "User brief.", 5),
    ]


def test_one_bad_setting_does_not_discard_the_good_one():
    raw = [
        {
            "scopes": [{"tags": ["company:*"]}],
            "observations_mission": GENERIC,
            "max_observations_per_scope": "20",
        }
    ]
    assert _parse_consolidation_strategies(raw) == [
        _ConsolidationStrategy((_ScopePattern(("company:*",)),), GENERIC, None)
    ]


def test_an_unusable_scope_is_dropped_but_its_siblings_survive():
    """A strategy listing several scopes stays usable when only one is malformed
    — losing the whole strategy would silently un-govern the good scopes too."""
    raw = [
        {
            "scopes": [{"tags": ["company:*"]}, {"tags": []}, {"tags": ["org:*", "shared"]}],
            "observations_mission": GENERIC,
        }
    ]
    assert _parse_consolidation_strategies(raw) == [
        _ConsolidationStrategy((_ScopePattern(("company:*",)), _ScopePattern(("org:*", "shared"))), GENERIC, None)
    ]


# ---------------------------------------------------------------------------
# Multi-scope claiming — one strategy, several scopes
# ---------------------------------------------------------------------------


def test_one_strategy_claims_every_scope_it_lists():
    """The reason a strategy holds a list of rules: one brief, written once,
    applied to several unrelated scope shapes."""
    config = _config(
        [{"scopes": [{"tags": ["company:*"]}, {"tags": ["org:*", "shared"]}], "observations_mission": GENERIC}]
    )

    assert _mission(config, ["company:acme"]) == GENERIC
    assert _mission(config, ["org:eng", "shared"]) == GENERIC
    assert _mission(config, ["user:dana"]) == BANK_MISSION
    assert _mission(config, ["org:eng"]) == BANK_MISSION, (
        "every tag in a rule must be on the scope: 'shared' is missing here"
    )


# ---------------------------------------------------------------------------
# Mission — first claiming strategy carrying a mission wins
# ---------------------------------------------------------------------------


def test_a_claimed_scope_overrides_the_bank_mission():
    config = _config([{"scopes": [{"tags": ["company:*"]}], "observations_mission": GENERIC}])
    assert _mission(config, ["company:acme"]) == GENERIC


def test_an_unclaimed_scope_falls_back_to_the_bank_mission():
    config = _config([{"scopes": [{"tags": ["company:*"]}], "observations_mission": GENERIC}])
    assert _mission(config, ["user:dana"]) == BANK_MISSION
    assert _mission(config, []) == BANK_MISSION


def test_the_first_claiming_strategy_wins():
    config = _config(
        [
            {"scopes": [{"tags": ["company:acme"]}], "observations_mission": "Acme-specific brief."},
            {"scopes": [{"tags": ["company:*"]}], "observations_mission": GENERIC},
        ]
    )
    assert _mission(config, ["company:acme"]) == "Acme-specific brief."
    assert _mission(config, ["company:other"]) == GENERIC


def test_federation_scopes_resolve_independently():
    """The whole point: one fan-out document, three scopes, three briefs."""
    config = _config(
        [
            {"scopes": [{"tags": ["company:*"]}], "observations_mission": GENERIC},
            {"scopes": [{"tags": ["team:*"]}], "observations_mission": "Team brief."},
        ]
    )
    assert _mission(config, ["company:acme"]) == GENERIC
    assert _mission(config, ["team:exec"]) == "Team brief."
    assert _mission(config, ["user:dana"]) == BANK_MISSION


def test_no_strategies_and_no_config_are_both_safe():
    assert _mission(_config(None), ["company:acme"]) == BANK_MISSION
    assert _config_for_scope(None, ["company:acme"]) is None


# ---------------------------------------------------------------------------
# _effective_scope_limit — strategies win over the deprecated scope limits
# ---------------------------------------------------------------------------


def test_strategies_take_precedence_over_the_deprecated_scope_limits():
    config = _config(
        [{"scopes": [{"tags": ["company:*"]}], "max_observations_per_scope": 20}],
        scope_limits=[{"scope": ["company:*"], "limit": 99}],
    )
    assert _effective_scope_limit(config, ["company:acme"]) == 20


def test_deprecated_scope_limits_still_apply_where_no_strategy_claims():
    config = _config(
        [{"scopes": [{"tags": ["company:*"]}], "max_observations_per_scope": 20}],
        scope_limits=[{"scope": ["run_*"], "limit": 1}],
    )
    assert _effective_scope_limit(config, ["run_7"]) == 1
    assert _effective_scope_limit(config, ["user:dana"]) == 50  # bank-wide fallback


def test_a_mission_only_strategy_leaves_the_cap_to_the_deprecated_field():
    config = _config(
        [{"scopes": [{"tags": ["company:*"]}], "observations_mission": GENERIC}],
        scope_limits=[{"scope": ["company:*"], "limit": 99}],
    )
    assert _mission(config, ["company:acme"]) == GENERIC
    assert _effective_scope_limit(config, ["company:acme"]) == 99


# ---------------------------------------------------------------------------
# Source-facts token limits — the evidence each consolidation call is shown
# ---------------------------------------------------------------------------


def test_a_strategy_overrides_the_source_facts_limits_for_its_scopes():
    config = _config(
        [
            {
                "scopes": [{"tags": ["company:*"]}],
                "consolidation_source_facts_max_tokens": 1024,
                "consolidation_source_facts_max_tokens_per_observation": 64,
            }
        ]
    )

    company = _config_for_scope(config, ["company:acme"])
    assert company.consolidation_source_facts_max_tokens == 1024
    assert company.consolidation_source_facts_max_tokens_per_observation == 64

    user = _config_for_scope(config, ["user:dana"])
    assert user.consolidation_source_facts_max_tokens == BANK_SOURCE_FACTS
    assert user.consolidation_source_facts_max_tokens_per_observation == BANK_SOURCE_FACTS_PER_OBS


def test_each_setting_falls_back_on_its_own():
    """Setting one limit must leave the other at the bank-wide value, not at some
    empty default — the strategy only replaces what it names."""
    config = _config([{"scopes": [{"tags": ["company:*"]}], "consolidation_source_facts_max_tokens": 1024}])

    company = _config_for_scope(config, ["company:acme"])
    assert company.consolidation_source_facts_max_tokens == 1024
    assert company.consolidation_source_facts_max_tokens_per_observation == BANK_SOURCE_FACTS_PER_OBS
    assert company.observations_mission == BANK_MISSION
    assert company.max_observations_per_scope == 50


def test_resolving_a_scope_never_mutates_the_bank_config():
    """The bank config is shared by every scope in the pass loop; a strategy that
    leaked into it would apply to the scopes after it."""
    config = _config(
        [
            {
                "scopes": [{"tags": ["company:*"]}],
                "observations_mission": GENERIC,
                "max_observations_per_scope": 5,
                "consolidation_source_facts_max_tokens": 1024,
            }
        ]
    )

    _config_for_scope(config, ["company:acme"])

    assert config.observations_mission == BANK_MISSION
    assert config.max_observations_per_scope == 50
    assert config.consolidation_source_facts_max_tokens == BANK_SOURCE_FACTS


# ---------------------------------------------------------------------------
# Two strategies claiming one scope — the first wins, whole
# ---------------------------------------------------------------------------

ACME_CAP_ONLY = {"scopes": [{"tags": ["company:acme"]}], "max_observations_per_scope": 5}
ALL_COMPANIES = {
    "scopes": [{"tags": ["company:*"]}],
    "observations_mission": GENERIC,
    "max_observations_per_scope": 20,
    "consolidation_source_facts_max_tokens": 1024,
}


def test_the_earlier_strategy_wins_a_scope_both_claim():
    config = _config([ACME_CAP_ONLY, ALL_COMPANIES])

    assert _config_for_scope(config, ["company:acme"]).max_observations_per_scope == 5


def test_a_later_strategy_never_fills_the_winners_gaps():
    """The case that makes the rule matter. The winner sets only a cap; the later
    strategy also claims the scope and sets a mission and a token limit. Those
    must NOT leak in — the gaps come from the bank-wide values, so the scope runs
    on exactly one strategy plus Default, never on a blend of two."""
    config = _config([ACME_CAP_ONLY, ALL_COMPANIES])

    acme = _config_for_scope(config, ["company:acme"])

    assert acme.observations_mission == BANK_MISSION
    assert acme.consolidation_source_facts_max_tokens == BANK_SOURCE_FACTS


def test_the_losing_strategy_still_governs_the_scopes_it_alone_claims():
    config = _config([ACME_CAP_ONLY, ALL_COMPANIES])

    other = _config_for_scope(config, ["company:globex"])

    assert other.observations_mission == GENERIC
    assert other.max_observations_per_scope == 20
    assert other.consolidation_source_facts_max_tokens == 1024


def test_list_order_is_the_priority_order():
    """Swapping the two strategies swaps the winner — order is the only tie rule."""
    config = _config([ALL_COMPANIES, ACME_CAP_ONLY])

    acme = _config_for_scope(config, ["company:acme"])

    assert acme.max_observations_per_scope == 20
    assert acme.observations_mission == GENERIC


def test_a_strategy_that_sets_nothing_does_not_block_a_later_one():
    """An empty strategy is dropped at parse time, so it cannot win a scope and
    leave it running on nothing but Default."""
    config = _config([{"scopes": [{"tags": ["company:acme"]}]}, ALL_COMPANIES])

    assert _config_for_scope(config, ["company:acme"]).observations_mission == GENERIC


def test_the_winners_missing_cap_falls_back_past_later_strategies():
    """Same rule for the cap, which has one extra fallback: the deprecated
    observation_scope_limits sits between the winning strategy and the bank."""
    config = _config(
        [{"scopes": [{"tags": ["company:acme"]}], "observations_mission": "Acme brief."}, ALL_COMPANIES],
        scope_limits=[{"scope": ["company:*"], "limit": 7}],
    )

    assert _effective_scope_limit(config, ["company:acme"]) == 7


# ---------------------------------------------------------------------------
# tags_match — per pattern: "all" (default: extra tags allowed) vs "exact"
# ---------------------------------------------------------------------------

COMPANY_AND_TEAM = {"scopes": [{"tags": ["company:*", "team:*"]}], "observations_mission": GENERIC}


def _with_mode(mode):
    return {"scopes": [{"tags": ["company:*", "team:*"], "tags_match": mode}], "observations_mission": GENERIC}


@pytest.mark.parametrize(
    "tags,claimed",
    [
        (["company:acme", "team:exec"], True),
        # The case the default exists for: a scope retained with all of a memory's
        # tags together. It has a company and a team, plus a user.
        (["user:dana", "team:exec", "company:acme"], True),
        (["company:acme"], False),  # no team
        (["team:exec"], False),  # no company
        ([], False),  # the untagged scope never matches
    ],
)
def test_all_is_the_default_and_allows_extra_tags(tags, claimed):
    config = _config([COMPANY_AND_TEAM])

    assert (_mission(config, tags) == GENERIC) is claimed


@pytest.mark.parametrize(
    "tags,claimed",
    [
        (["company:acme", "team:exec"], True),
        (["user:dana", "team:exec", "company:acme"], False),  # user:dana is extra
        (["company:acme"], False),
    ],
)
def test_exact_requires_the_scope_to_have_nothing_else(tags, claimed):
    config = _config([_with_mode("exact")])

    assert (_mission(config, tags) == GENERIC) is claimed


def test_any_of_several_tags_is_written_as_one_pattern_per_tag():
    """There is no "any" mode: patterns are already alternatives."""
    config = _config([{"scopes": [{"tags": ["company:*"]}, {"tags": ["team:*"]}], "observations_mission": GENERIC}])

    assert _mission(config, ["company:acme"]) == GENERIC
    assert _mission(config, ["team:exec"]) == GENERIC
    assert _mission(config, ["user:dana"]) == BANK_MISSION


def test_an_unknown_mode_falls_back_to_all_instead_of_dropping_the_pattern():
    config = _config([_with_mode("exat")])

    assert _mission(config, ["user:dana", "team:exec", "company:acme"]) == GENERIC


def test_each_pattern_has_its_own_mode():
    """The reason tags_match is per pattern: one strategy can say "exactly
    company:*" OR "team:*, other tags allowed" without a second strategy."""
    config = _config(
        [
            {
                "scopes": [
                    {"tags": ["company:*"], "tags_match": "exact"},
                    {"tags": ["team:*"]},
                ],
                "observations_mission": GENERIC,
            }
        ]
    )

    assert _mission(config, ["company:acme"]) == GENERIC
    assert _mission(config, ["company:acme", "user:dana"]) == BANK_MISSION, "exact pattern, extra tag"
    assert _mission(config, ["team:exec", "user:dana"]) == GENERIC, "all pattern, extra tag allowed"
