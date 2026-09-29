"""One bank, federated across user / team / company — each scope under its own brief.

`observation_scopes` (test 24) partitions one memory's evidence into several
independently-maintained observation sets. But every scope was consolidated under
the same bank-wide `observations_mission`, so the company-wide scope came out as
the plain union of everyone's observations: the same detail, the same names, just
visible to more people. That is the wrong shape for the case the scopes exist to
serve — a CEO's confidential 1:1 belongs in their own scope in full, and in the
all-hands scope only as a generality, if at all.

`consolidation_strategies` closes that: a strategy names the scopes it claims
and carries their mission and observation cap, resolved per consolidation pass. It is safe
precisely because of how the fan-out already works — each resolved scope gets its
*own* LLM call (test 24's isolation property), so each call can be given a
different brief with no way for one scope's instructions to reach another's.

These tests assert what the server assembled, not what a model chose to write:
the mission text is a prompt input, and whether an LLM then obeys it is the
model's business, not the seam's. A pass is identified by the mission it was
given, or — where the strategy under test carries no mission — by counting how many
of the three passes received the capacity note.

Two strategies can claim the same scope. Exactly one applies — the first in the
list — and whatever it leaves unset comes from the bank-wide values, never from
the later strategy. The last two stories pin that, because a silent blend of two
strategies is the failure a user cannot see from the config alone.

The deprecated `observation_scope_limits` is pinned here too, because the
interesting case is the precedence *between* the two fields — the composition
nothing else tests.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import Consolidation, Observation, extracted, fact, fact_ids_in

pytestmark = pytest.mark.asyncio

USER = "user:dana"
TEAM = "team:exec"
COMPANY = "company:acme"

BANK_MISSION = "Record what this person worked on, in full detail."
COMPANY_MISSION = "Record only generalized industry trends. Name no specific company."
TEAM_MISSION = "Record decisions the exec team must act on."

# The note the server assembles when a scope's cap is already spent. It is the
# only thing that actually constrains the model, so it is what the cap tests pin.
CAP_REACHED = "OBSERVATION LIMIT REACHED"

CONTENT = "Dana met the founders of Northwind and they are moving off frontier models."
SCOPES = [[USER], [TEAM], [COMPANY]]


def _observe(text: str):
    """One observation per consolidation pass, over whatever facts it was shown.

    Except where the pass was told its scope is full: the response schema caps
    creates at the remaining slots, so a create there is rejected, the pass fails,
    and the loop abandons every scope after it. A model that reads the capacity
    note does not do that, and neither does this stub — otherwise a cap on one
    scope would silently stop the others from being consolidated at all.
    """

    def build(request) -> Consolidation:
        if CAP_REACHED in request.all_text:
            return Consolidation()
        return Consolidation(
            creates=[Observation(text=text, source_fact_ids=fact_ids_in(request.all_text), reason="system test")]
        )

    return build


@pytest.fixture(autouse=True)
def _extraction(llm):
    llm.on_step("extract_facts").returns(
        extracted(fact("Northwind is moving off frontier models", who="Dana", entities=["Dana", "Northwind"]))
    )
    llm.on_step("consolidate").answers_with(_observe("Northwind is moving off frontier models"))


async def _retain(client, bank: str, settled) -> None:
    await client.aretain_batch(
        bank_id=bank,
        items=[{"content": CONTENT, "tags": [USER, TEAM, COMPANY], "observation_scopes": SCOPES}],
    )
    await settled(bank)


def _passes(llm) -> list[str]:
    """The three consolidation prompts — one per resolved scope.

    Asserted here rather than in each test: every assertion below counts how many
    of the passes carry something, which only means anything if all three ran.
    """
    prompts = llm.prompts_for("consolidate")
    assert len(prompts) == 3, f"expected one consolidation pass per scope, got {len(prompts)}"
    return prompts


def _passes_containing(llm, needle: str) -> int:
    return sum(1 for prompt in _passes(llm) if needle in prompt)


async def _configure(client, bank: str, **updates) -> None:
    await client.banks.update_bank_config(bank, {"updates": updates})


async def test_each_scope_is_consolidated_under_its_own_mission(client, bank_id, settled, llm):
    """The point of the feature: three passes over the same facts, three briefs.

    Each mission reaches exactly one pass. That count is the isolation property
    restated for missions — at two it would mean one scope's brief had leaked
    into another's call, which is the failure the whole design has to prevent.
    """
    await _configure(
        client,
        bank_id,
        observations_mission=BANK_MISSION,
        consolidation_strategies=[
            {"scopes": [{"tags": [COMPANY]}], "observations_mission": COMPANY_MISSION},
            {"scopes": [{"tags": [TEAM]}], "observations_mission": TEAM_MISSION},
        ],
    )

    await _retain(client, bank_id, settled)

    assert _passes_containing(llm, COMPANY_MISSION) == 1
    assert _passes_containing(llm, TEAM_MISSION) == 1
    assert _passes_containing(llm, BANK_MISSION) == 1, (
        "the scope no strategy claims keeps the bank-wide mission, and only that scope"
    )


async def test_a_strategy_may_set_the_cap_without_touching_the_mission(client, bank_id, settled, llm):
    """Each setting is independent. A cap-only strategy leaves the brief alone — so an
    operator can throttle a noisy shared scope without also rewriting what it is
    for, and the cap still lands on that scope alone.
    """
    await _configure(
        client,
        bank_id,
        observations_mission=BANK_MISSION,
        consolidation_strategies=[{"scopes": [{"tags": [COMPANY]}], "max_observations_per_scope": 0}],
    )

    await _retain(client, bank_id, settled)

    assert _passes_containing(llm, BANK_MISSION) == 3, "a cap-only strategy must not disturb the mission"
    assert _passes_containing(llm, CAP_REACHED) == 1, "the cap applies to the matching scope only"


async def test_strategies_win_over_the_deprecated_scope_limits(client, bank_id, settled, llm):
    """`observation_scope_limits` still works, but it is consulted second.

    A bank carrying both must not have its new strategy silently shadowed by the old
    one it was written to replace — this is the upgrade path, and the only place
    the two fields meet.
    """
    await _configure(
        client,
        bank_id,
        observation_scope_limits=[{"scope": [COMPANY], "limit": 0}],
        consolidation_strategies=[{"scopes": [{"tags": [COMPANY]}], "max_observations_per_scope": 50}],
    )

    await _retain(client, bank_id, settled)

    assert _passes_containing(llm, CAP_REACHED) == 0, "the strategy's cap of 50 must win over the deprecated 0"


async def test_the_deprecated_field_still_applies_where_no_strategy_claims(client, bank_id, settled, llm):
    """The other half of the upgrade path: adding a strategy for one scope must not
    quietly disable the old rules governing the others."""
    await _configure(
        client,
        bank_id,
        observation_scope_limits=[{"scope": [TEAM], "limit": 0}],
        consolidation_strategies=[{"scopes": [{"tags": [COMPANY]}], "max_observations_per_scope": 50}],
    )

    await _retain(client, bank_id, settled)

    assert _passes_containing(llm, CAP_REACHED) == 1, "the deprecated rule still governs the scope no strategy claims"


async def test_a_malformed_strategy_is_ignored_rather_than_breaking_consolidation(client, bank_id, settled, llm):
    """The config round-trips as JSON through the bank-config API, so a bad entry
    must degrade to the bank-wide default rather than take consolidation down with
    it — a typo in one strategy must not stop every scope in the bank, nor shift the
    strategies written after it.
    """
    await _configure(
        client,
        bank_id,
        observations_mission=BANK_MISSION,
        consolidation_strategies=[
            {"scopes": [{"tags": [COMPANY]}], "observations_mission": ""},
            {"scopes": [{"tags": [TEAM]}], "observations_mission": TEAM_MISSION},
        ],
    )

    await _retain(client, bank_id, settled)

    assert _passes_containing(llm, TEAM_MISSION) == 1, "a bad strategy must not shift the ones after it"
    assert _passes_containing(llm, BANK_MISSION) == 2, "the scope whose strategy was dropped falls back to the bank"


async def test_one_strategy_can_claim_several_scopes(client, bank_id, settled, llm):
    """`scopes` is a list of scopes so one brief covers several without being
    written out per scope — the team and company scopes here share a strategy, and
    the user scope, claimed by nobody, keeps the bank's own mission."""
    await _configure(
        client,
        bank_id,
        observations_mission=BANK_MISSION,
        consolidation_strategies=[
            {"scopes": [{"tags": [COMPANY]}, {"tags": [TEAM]}], "observations_mission": COMPANY_MISSION}
        ],
    )

    await _retain(client, bank_id, settled)

    assert _passes_containing(llm, COMPANY_MISSION) == 2
    assert _passes_containing(llm, BANK_MISSION) == 1


async def test_when_two_strategies_claim_a_scope_the_first_one_wins(client, bank_id, settled, llm):
    """The company scope is claimed by both strategies; the team scope only by the
    second. The first wins the company scope, the second still gets the team."""
    await _configure(
        client,
        bank_id,
        observations_mission=BANK_MISSION,
        consolidation_strategies=[
            {"scopes": [{"tags": [COMPANY]}], "observations_mission": COMPANY_MISSION},
            {"scopes": [{"tags": [COMPANY]}, {"tags": [TEAM]}], "observations_mission": TEAM_MISSION},
        ],
    )

    await _retain(client, bank_id, settled)

    assert _passes_containing(llm, COMPANY_MISSION) == 1
    assert _passes_containing(llm, TEAM_MISSION) == 1, "the losing strategy must not also reach the company pass"
    assert _passes_containing(llm, BANK_MISSION) == 1


async def test_a_later_strategy_never_fills_the_winners_gaps(client, bank_id, settled, llm):
    """The first strategy wins the company scope but sets only a cap. The second
    also claims it and sets a mission — which must not reach the company pass:
    the gap is filled from the bank, so the scope runs on one strategy, not two."""
    await _configure(
        client,
        bank_id,
        observations_mission=BANK_MISSION,
        consolidation_strategies=[
            {"scopes": [{"tags": [COMPANY]}], "max_observations_per_scope": 0},
            {"scopes": [{"tags": [COMPANY]}], "observations_mission": COMPANY_MISSION},
        ],
    )

    await _retain(client, bank_id, settled)

    assert _passes_containing(llm, COMPANY_MISSION) == 0, "the later strategy's mission leaked into a scope it lost"
    assert _passes_containing(llm, BANK_MISSION) == 3
    assert _passes_containing(llm, CAP_REACHED) == 1, "the winner's own setting still applies"


async def _retain_combined(client, bank: str, settled) -> None:
    """Retain with the default scope — all of the memory's tags in one scope —
    which consolidates as a single pass."""
    await client.aretain_batch(bank_id=bank, items=[{"content": CONTENT, "tags": [USER, TEAM, COMPANY]}])
    await settled(bank)


def _single_pass(llm) -> str:
    prompts = llm.prompts_for("consolidate")
    assert len(prompts) == 1, f"a combined scope is one pass, got {len(prompts)}"
    return prompts[0]


async def test_tags_match_all_claims_a_scope_that_also_has_other_tags(client, bank_id, settled, llm):
    """The default. "Has a company and a team" must claim the scope a memory gets
    by default — {user, team, company} — even though it carries a user tag too.
    Under an exact match it would not, which is why "all" is the default."""
    await _configure(
        client,
        bank_id,
        observations_mission=BANK_MISSION,
        consolidation_strategies=[
            {"scopes": [{"tags": ["company:*", "team:*"]}], "observations_mission": COMPANY_MISSION}
        ],
    )

    await _retain_combined(client, bank_id, settled)

    assert COMPANY_MISSION in _single_pass(llm)


async def test_tags_match_exact_refuses_a_scope_with_extra_tags(client, bank_id, settled, llm):
    await _configure(
        client,
        bank_id,
        observations_mission=BANK_MISSION,
        consolidation_strategies=[
            {
                "scopes": [{"tags": ["company:*", "team:*"], "tags_match": "exact"}],
                "observations_mission": COMPANY_MISSION,
            }
        ],
    )

    await _retain_combined(client, bank_id, settled)

    prompt = _single_pass(llm)
    assert COMPANY_MISSION not in prompt, "the user tag is extra, so an exact strategy must not claim this scope"
    assert BANK_MISSION in prompt


async def test_the_preview_shows_which_existing_scopes_each_rule_would_take(client, bank_id, settled):
    """The editor's "matches N scopes · M handled by an earlier strategy" comes from
    this endpoint, computed with consolidation's own matching and first-strategy-wins
    rule. Asserted whole: every count, every sample and who handles it. A draft is
    previewed, never saved."""
    from hindsight_client_api.models.consolidation_strategies_preview_request import (
        ConsolidationStrategiesPreviewRequest,
    )

    await _retain(client, bank_id, settled)  # -> scopes {user:dana}, {team:exec}, {company:acme}

    draft = [
        {"scopes": [{"tags": ["company:*"]}], "observations_mission": COMPANY_MISSION},
        {"scopes": [{"tags": ["team:*"]}, {"tags": ["company:*"]}], "observations_mission": TEAM_MISSION},
    ]
    preview = await client.memory.preview_consolidation_strategies(
        bank_id, ConsolidationStrategiesPreviewRequest(strategies=draft, sample_limit=5)
    )

    def scope(tags, handled_by):
        return {"tags": tags, "count": 1, "handled_by": handled_by}

    assert preview.to_dict() == {
        "strategies": [
            {
                "active": True,
                "claimed_count": 1,
                "rules": [
                    {
                        "match_count": 1,
                        "taken_count": 0,
                        "observation_count": 1,
                        "samples": [scope([COMPANY], 0)],
                    }
                ],
            },
            {
                "active": True,
                "claimed_count": 1,
                "rules": [
                    {
                        "match_count": 1,
                        "taken_count": 0,
                        "observation_count": 1,
                        "samples": [scope([TEAM], 1)],
                    },
                    {
                        # Same scope as strategy 0's rule, which comes first and wins it.
                        "match_count": 1,
                        "taken_count": 1,
                        "observation_count": 1,
                        "samples": [scope([COMPANY], 0)],
                    },
                ],
            },
        ],
        "default": {"match_count": 1, "observation_count": 1, "samples": [scope([USER], None)]},
        "scopes_scanned": 3,
        "complete": True,
    }

    overrides = (await client.banks.get_bank_config(bank_id)).overrides
    assert "consolidation_strategies" not in overrides, "a preview must not save the draft"
