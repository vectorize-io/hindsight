"""The guard that refuses an update aimed at a lookalike of the fact's subject.

Cases are the ones the consolidation eval met on gemini-3.1-flash-lite (``ledger-sync`` vs
``ledger-sink``, ``invoice-api`` vs ``invoice-sync``, Marco vs Marta), plus the legitimate
updates the guard must leave alone — every change scenario of that suite.
"""

from __future__ import annotations

import pytest

from hindsight_api.engine.consolidation.consolidator import _aimed_at_a_lookalike


@pytest.mark.parametrize(
    ("fact", "observation"),
    [
        ("The ledger-sink service listens on port 8444.", "The ledger-sync service listens on port 8443."),
        ("The invoice-api service is owned by team Vega.", "The invoice-sync service is owned by team Vega."),
        ("Marta is allergic to shellfish.", "Marco is allergic to peanuts and is vegetarian."),
    ],
)
def test_a_lookalike_target_is_refused(fact: str, observation: str) -> None:
    assert _aimed_at_a_lookalike(fact, observation)


@pytest.mark.parametrize(
    ("fact", "observation"),
    [
        ("The ledger-sync service moved from port 8443 to port 9443.", "The ledger-sync service listens on port 8443."),
        (
            "Ownership of the billing-gateway service moved from team Atlas to team Nimbus.",
            "The billing-gateway service is written in Kotlin, owned by team Atlas.",
        ),
        ("Alice left Google in April 2026 and now works at Microsoft.", "Alice works at Google as a senior engineer."),
        ("The user reported that Shadow has died.", "The user's horses include Shadow and Twister."),
        ("Corepack was removed from the project setup.", "Corepack is enabled to pin the pnpm version."),
        ("Lena grew up in Gothenburg, Sweden.", "Lena plans to move back to her home country next spring."),
        ("Ben left the payments on-call rotation.", "The payments on-call rotation is Ana, Ben, Chen and Dmitri."),
        ("The project switched from fnm to nub.", "The project uses fnm to manage Node.js versions."),
        ("The service was deployed on Monday.", "The service deploys nightly."),
    ],
)
def test_an_update_about_the_same_thing_is_kept(fact: str, observation: str) -> None:
    assert not _aimed_at_a_lookalike(fact, observation)


def test_a_count_is_found_by_its_noun() -> None:
    """A count names a kind of thing, not an entity: "has 5 horses" must reach the stale
    "has 1 horse, which is named Daisy" that a run wrote into Daisy's observation."""
    from hindsight_api.engine.consolidation.consolidator import _counted_nouns

    assert _counted_nouns("The user has 5 horses on their farm.") == {"horse"}
    assert _counted_nouns("The user has 1 horse, which is named Daisy.") == {"horse"}
    assert _counted_nouns("The ledger-sync service listens on port 8443.") == set()


@pytest.mark.parametrize(
    ("fact", "old", "new", "changes"),
    [
        ("The user has 5 horses on their farm.", "The user has 1 horse.", "The user has 5 horses.", True),
        ("Ownership of billing-gateway moved to team Nimbus.", "billing-gateway is owned by Atlas.", "x", True),
        (
            "The payout-worker service is written in Kotlin.",
            "payout-worker is owned by Atlas.",
            "payout-worker is owned by Atlas and written in Kotlin.",
            False,
        ),
        (
            "Pages for billing-api go to #oncall-atlas.",
            "billing-api listens on 8211.",
            "Pages for billing-api go to #oncall-atlas.",
            True,
        ),
    ],
)
def test_only_a_state_change_asks_for_a_second_look(fact: str, old: str, new: str, changes: bool) -> None:
    """Adding a facet leaves the other observations true; a change, a new count, or a rewrite
    that dropped what the observation said does not."""
    from hindsight_api.engine.consolidation.consolidator import _signals_change

    assert _signals_change(fact, old, new) is changes
