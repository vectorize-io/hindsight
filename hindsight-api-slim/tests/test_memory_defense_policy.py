"""Memory Defense policy parsing."""

import pytest

from hindsight_api.extensions.memory_defense import (
    DefenseAction,
    parse_policy,
)


def test_parse_minimal_policy() -> None:
    policy = parse_policy({"enabled": True})
    assert policy.enabled is True
    assert policy.rules == ()


def test_parse_policy_with_rule() -> None:
    policy = parse_policy(
        {
            "enabled": True,
            "rules": [
                {"on": "sensitive_data", "action": "redact"},
            ],
        }
    )
    assert {r.on for r in policy.rules} == {"sensitive_data"}
    assert policy.rules[0].action is DefenseAction.REDACT


def test_invalid_action_raises() -> None:
    # Use a valid ``on`` so the parser progresses to action validation.
    with pytest.raises(ValueError, match="action"):
        parse_policy({"enabled": True, "rules": [{"on": "sensitive_data", "action": "lol"}]})


def test_parse_policy_rejects_unknown_detector() -> None:
    with pytest.raises(ValueError, match="invalid on"):
        parse_policy({"enabled": True, "rules": [{"on": "nope", "action": "block"}]})


def test_parse_policy_accepts_block_action_on_sensitive_data() -> None:
    policy = parse_policy({"enabled": True, "rules": [{"on": "sensitive_data", "action": "block"}]})
    assert policy.rules[0].action is DefenseAction.BLOCK


def test_disabled_policy_is_inert() -> None:
    policy = parse_policy({"enabled": False, "rules": [{"on": "sensitive_data", "action": "redact"}]})
    assert policy.enabled is False


def test_defense_action_string_round_trip() -> None:
    assert DefenseAction("redact") is DefenseAction.REDACT
    assert DefenseAction.BLOCK.value == "block"
