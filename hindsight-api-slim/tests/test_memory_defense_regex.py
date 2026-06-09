import pytest

from hindsight_api.extensions.builtin.memory_defense_regex import MemoryDefenseRegexExtension
from hindsight_api.extensions.memory_defense import DefenseAction, parse_policy


@pytest.fixture
def regex_defense() -> MemoryDefenseRegexExtension:
    return MemoryDefenseRegexExtension(config={})


@pytest.fixture
def redact_policy() -> dict:
    return {
        "enabled": True,
        "rules": [{"on": "sensitive_data", "action": "redact"}],
    }


@pytest.mark.asyncio
async def test_allows_innocuous_content(regex_defense, redact_policy) -> None:
    decision = await regex_defense.screen(
        policy=parse_policy(redact_policy),
        bank_id="b1",
        document_id="d1",
        content="The Q3 roadmap meeting is on Friday.",
        tags=[],
    )
    assert decision.action is DefenseAction.ALLOW


@pytest.mark.asyncio
async def test_redacts_github_token(regex_defense, redact_policy) -> None:
    secret = "ghp_" + "A" * 36
    decision = await regex_defense.screen(
        policy=parse_policy(redact_policy),
        bank_id="b1",
        document_id="d1",
        content=f"rotate this token: {secret}",
        tags=[],
    )
    assert decision.action is DefenseAction.REDACT
    assert decision.redacted_content is not None
    assert secret not in decision.redacted_content
    assert "[REDACTED:github_token]" in decision.redacted_content
    assert "github_token" in decision.matched_types


@pytest.mark.asyncio
async def test_block_action_blocks(regex_defense) -> None:
    """A sensitive_data rule with action=block returns BLOCK (no redacted content)."""
    policy = parse_policy({"enabled": True, "rules": [{"on": "sensitive_data", "action": "block"}]})
    secret = "AKIA" + "A" * 16
    decision = await regex_defense.screen(
        policy=policy,
        bank_id="b1",
        document_id="d1",
        content=f"key={secret}",
        tags=[],
    )
    assert decision.action is DefenseAction.BLOCK
    assert decision.redacted_content is None
    assert "aws_access_key" in decision.matched_types


@pytest.mark.asyncio
async def test_allows_when_policy_has_no_sensitive_data_rule(regex_defense) -> None:
    """If the policy is enabled but lists no ``sensitive_data`` rule, there is
    nothing to enforce — content passes through."""
    policy = parse_policy({"enabled": True, "rules": []})
    decision = await regex_defense.screen(
        policy=policy,
        bank_id="b1",
        document_id="d1",
        content="ghp_" + "Z" * 36,
        tags=[],
    )
    assert decision.action is DefenseAction.ALLOW


@pytest.mark.asyncio
async def test_disabled_policy_is_inert(regex_defense) -> None:
    policy = parse_policy({"enabled": False, "rules": [{"on": "sensitive_data", "action": "redact"}]})
    decision = await regex_defense.screen(
        policy=policy,
        bank_id="b1",
        document_id="d1",
        content="ghp_" + "Z" * 36,
        tags=[],
    )
    assert decision.action is DefenseAction.ALLOW
