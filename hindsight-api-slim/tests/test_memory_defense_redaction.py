"""Redaction benchmark for the Memory Defense regex engine.

If the redaction pattern set silently degrades, this fails CI. Run in standard
pytest — NOT gated behind a marker, so regressions are loud.
"""

import pytest

from hindsight_api.extensions.builtin.memory_defense_regex import MemoryDefenseRegexExtension
from hindsight_api.extensions.memory_defense import (
    DefenseAction,
    parse_policy,
)

REDACT_POLICY = parse_policy(
    {
        "enabled": True,
        "rules": [
            {"on": "sensitive_data", "action": "redact"},
        ],
    }
)


@pytest.fixture
def regex_defense() -> MemoryDefenseRegexExtension:
    return MemoryDefenseRegexExtension({})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        "ghp_" + "A" * 36,
        "sk-ant-" + "B" * 40,
        "sk-" + "C" * 30,
        "AKIA" + "D" * 16,
    ],
)
async def test_redacts_known_secret_patterns(payload: str, regex_defense: MemoryDefenseRegexExtension) -> None:
    d = await regex_defense.screen(
        policy=REDACT_POLICY,
        bank_id="b",
        document_id="d",
        content=f"my key is {payload}",
        tags=[],
    )
    assert d.action is DefenseAction.REDACT, f"expected redact for {payload!r}, got {d.action}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        "The roadmap meeting is on Friday",
        "Product launch planning notes",
        "Reminder about Tuesday",
    ],
)
async def test_allows_benign_payloads(payload: str, regex_defense: MemoryDefenseRegexExtension) -> None:
    """Benign payloads have no secret to scrub, so they're ALLOWed."""
    d = await regex_defense.screen(
        policy=REDACT_POLICY,
        bank_id="b",
        document_id="d",
        content=payload,
        tags=[],
    )
    assert d.action is DefenseAction.ALLOW
