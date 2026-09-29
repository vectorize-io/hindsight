"""Preview of consolidation strategies against a bank's existing scopes.

The control plane shows, per rule, which existing scopes it matches and which of
those an earlier strategy takes. That answer must be the consolidator's own —
same parsing, matching and first-strategy-wins rule — which is why the preview is
computed on the server (``preview_consolidation_strategies``) instead of by a
second, client-side implementation of the glob matching.

The pure function is tested directly (deterministic, no I/O); one HTTP test
checks the endpoint is wired through the engine for a real bank.
"""

from datetime import datetime

import httpx
import pytest
import pytest_asyncio

from hindsight_api.api import create_app
from hindsight_api.engine.consolidation.consolidator import preview_consolidation_strategies

# (tags, observation_count), most populous first — the shape the scope listing returns.
SCOPES = [
    (["company:acme"], 8),
    (["team:exec"], 4),
    (["company:acme", "team:exec", "user:erin"], 1),
    (["user:dana"], 3),
]

COMPANY = {"scopes": [{"tags": ["company:*"]}], "observations_mission": "Company brief."}
TEAM = {"scopes": [{"tags": ["team:*"]}], "observations_mission": "Team brief."}


def _preview(strategies, *, sample_limit=5, complete=True):
    return preview_consolidation_strategies(strategies, SCOPES, sample_limit=sample_limit, complete=complete)


def test_each_rule_reports_its_matches_and_the_scopes_it_loses():
    preview = _preview([COMPANY, TEAM])

    company_rule = preview.strategies[0].rules[0]
    assert company_rule.match_count == 2  # company:acme, and erin's combined scope ("all" allows extras)
    assert company_rule.taken_count == 0
    assert company_rule.observation_count == 9

    team_rule = preview.strategies[1].rules[0]
    assert team_rule.match_count == 2  # team:exec, and erin's combined scope...
    assert team_rule.taken_count == 1  # ...which the company strategy, earlier, wins


def test_claimed_counts_and_default_follow_first_strategy_wins():
    preview = _preview([COMPANY, TEAM])

    assert [s.claimed_count for s in preview.strategies] == [2, 1]
    assert preview.default.match_count == 1
    assert [scope.tags for scope in preview.default.samples] == [["user:dana"]]
    assert preview.default.samples[0].handled_by is None


def test_samples_name_the_strategy_that_actually_applies():
    preview = _preview([COMPANY, TEAM])

    handled = {tuple(s.tags): s.handled_by for s in preview.strategies[1].rules[0].samples}
    assert handled == {("team:exec",): 1, ("company:acme", "team:exec", "user:erin"): 0}


def test_a_strategy_the_server_would_ignore_keeps_its_slot():
    """The editor shows the list as typed; a dropped strategy must not shift the
    indices of the ones after it, or every "handled by #N" would be wrong."""
    empty = {"scopes": [{"tags": ["company:*"]}]}  # sets nothing -> ignored by consolidation
    preview = _preview([empty, TEAM])

    assert [s.active for s in preview.strategies] == [False, True]
    assert preview.strategies[0].claimed_count == 0
    assert preview.strategies[0].rules[0].match_count == 2  # still reports what the rule matches
    assert preview.strategies[0].rules[0].taken_count == 2  # but none of it is applied
    assert preview.strategies[1].claimed_count == 2


def test_a_malformed_rule_matches_nothing_without_failing_the_preview():
    preview = _preview([{"scopes": [{"tags": []}, {"tags": ["team:*"]}], "observations_mission": "x"}])

    assert [r.match_count for r in preview.strategies[0].rules] == [0, 2]


def test_sample_limit_and_completeness_are_reported():
    preview = _preview([COMPANY], sample_limit=1, complete=False)

    assert len(preview.strategies[0].rules[0].samples) == 1
    assert preview.strategies[0].rules[0].match_count == 2
    assert preview.complete is False
    assert preview.scopes_scanned == len(SCOPES)


@pytest_asyncio.fixture
async def api_client(memory):
    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_endpoint_previews_a_draft_for_a_real_bank(api_client, memory, request_context):
    bank_id = f"strategy_preview_{datetime.now().timestamp()}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        resp = await api_client.post(
            f"/v1/default/banks/{bank_id}/consolidation-strategies/preview",
            json={"strategies": [COMPANY], "sample_limit": 3},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        # A fresh bank has no observation scopes yet: nothing to match, but complete.
        assert body["complete"] is True
        assert body["scopes_scanned"] == 0
        assert body["strategies"][0]["active"] is True
        assert body["strategies"][0]["rules"][0]["match_count"] == 0
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_endpoint_404s_for_an_unknown_bank(api_client):
    resp = await api_client.post(
        "/v1/default/banks/no-such-bank-for-preview/consolidation-strategies/preview",
        json={"strategies": []},
    )
    assert resp.status_code == 404
