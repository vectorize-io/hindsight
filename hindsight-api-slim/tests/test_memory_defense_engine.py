"""Unit tests for MemoryDefenseRegexExtension — the default open-source engine."""

import pytest

from hindsight_api.extensions.builtin.memory_defense_regex import MemoryDefenseRegexExtension
from hindsight_api.extensions.memory_defense import (
    DefenseAction,
    parse_policy,
)


@pytest.fixture
def regex_defense() -> MemoryDefenseRegexExtension:
    return MemoryDefenseRegexExtension({})


@pytest.fixture
def redact_policy() -> dict:
    return {
        "enabled": True,
        "rules": [
            {"on": "sensitive_data", "action": "redact"},
        ],
    }


@pytest.mark.asyncio
async def test_engine_allows_innocuous_content(regex_defense: MemoryDefenseRegexExtension, redact_policy: dict) -> None:
    policy = parse_policy(redact_policy)
    decision = await regex_defense.screen(
        policy=policy,
        bank_id="b1",
        document_id="d1",
        content="The Q3 roadmap meeting is on Friday.",
        tags=["session:abc"],
    )
    assert decision.action is DefenseAction.ALLOW


@pytest.mark.asyncio
async def test_engine_redacts_secrets(regex_defense: MemoryDefenseRegexExtension, redact_policy: dict) -> None:
    policy = parse_policy(redact_policy)
    decision = await regex_defense.screen(
        policy=policy,
        bank_id="b1",
        document_id="d1",
        content="The GitHub token is ghp_" + "A" * 36,
        tags=[],
    )
    assert decision.action is DefenseAction.REDACT
    assert "[REDACTED:" in (decision.redacted_content or "")


@pytest.mark.asyncio
async def test_engine_disabled_policy_always_allows(regex_defense: MemoryDefenseRegexExtension) -> None:
    policy = parse_policy({"enabled": False})
    decision = await regex_defense.screen(
        policy=policy,
        bank_id="b1",
        document_id="d1",
        content="Here is a secret ghp_" + "A" * 36,
        tags=[],
    )
    assert decision.action is DefenseAction.ALLOW


@pytest.mark.asyncio
async def test_decision_reports_matched_types(regex_defense: MemoryDefenseRegexExtension, redact_policy: dict) -> None:
    """The decision carries which redaction patterns fired (for the webhook payload)."""
    decision = await regex_defense.screen(
        policy=parse_policy(redact_policy),
        bank_id="b1",
        document_id="d1",
        content="The token is sk-ant-" + "B" * 40,
        tags=["session:abc"],
    )
    assert decision.action is DefenseAction.REDACT
    assert "anthropic_key" in decision.matched_types
