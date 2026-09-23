"""How many tokens a knowledge-page refresh spends, and on what.

A baseline, not a gate. Every other suite here grades what a refresh wrote; this
one records what writing it cost — each LLM call's input, cached and output
tokens, and the full prompt it sent — so a change to how reflect budgets its
tools can be judged against numbers instead of a feeling.

Behind it: refreshes on real project banks sent 30-90k-token synthesis prompts,
because a tool's ``max_tokens`` counts fact text only and the JSON around each
fact is three to four times larger (#4566). That one fact explains a 500 on the
30s call deadline (#4568), a context guard ending the search before ``recall``
ran (#4563), a small-window model answering "I don't have information" (#4561),
and a background loop billing a metered key all night (#4532).

The bank is a frozen archive (see ``refresh_cost``), so two runs refresh the same
facts, observations and pages. The model is not frozen, so numbers still move
between runs: compare medians over repeats, not single runs.

Two refreshes per page:

* ``full`` — the page rebuilt from the whole bank: the reflect loop at its widest.
* ``delta`` — a small wave of new facts, then the page's own default refresh:
  what a page pays every time it keeps itself current.

Run ``--cost-output DIR`` to keep the report and every prompt for comparison.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import uuid
from pathlib import Path

import pytest
from hindsight_client import Hindsight

from hindsight_system_evals import provider_environment
from hindsight_system_evals.pages import SettleFn
from hindsight_system_evals.refresh_cost import (
    FIXTURE,
    PAGES,
    PRICES,
    CostReport,
    RefreshCost,
    measure_refresh,
    new_wave,
    page_trigger,
    summary_table,
)

log = logging.getLogger(__name__)

#: One page per run in minimum acceptance: enough to see a regression in the
#: token count, cheap enough to run daily. ``--full`` measures all of them.
MINIMUM_PAGES = ("Known bugs and fixes",)


async def _import_fixture(client: Hindsight, host_bank: str, settled: SettleFn) -> str:
    """Restore the frozen bank into a fresh one and wait for the re-embedding."""
    if not FIXTURE.exists():
        pytest.fail(f"missing {FIXTURE} — build it with `uv run python -m hindsight_system_evals.refresh_cost build`")
    # The import operation is recorded against an existing bank; the target must
    # not exist yet.
    await client.aupdate_bank_config(host_bank, enable_auto_consolidation=False)
    target = f"{host_bank}-cost"
    await client.aimport_bank(host_bank, FIXTURE.read_bytes(), target_bank_id=target)
    await settled(host_bank)
    await settled(target)
    # A new wave is retained as written; extraction would add calls outside any
    # refresh, and consolidation would too.
    await client.aupdate_bank_config(
        target, retain_extraction_mode="chunks", enable_auto_consolidation=False, enable_observations=False
    )
    return target


async def _page_ids(client: Hindsight, bank: str) -> dict[str, str]:
    models = await client.alist_mental_models(bank_id=bank, detail="metadata")
    return {m.name: m.id for m in models.items}


async def _measure(client: Hindsight, bank: str, settled: SettleFn, pages: tuple[str, ...]) -> list[RefreshCost]:
    ids = await _page_ids(client, bank)
    missing = [p for p in pages if p not in ids]
    assert not missing, f"the fixture has no page named {missing}; it has {sorted(ids)}"

    results: list[RefreshCost] = []
    for page in pages:
        await client.aupdate_mental_model(bank_id=bank, mental_model_id=ids[page], trigger=page_trigger("full"))
        await settled(bank)
        results.append(await measure_refresh(client, bank, ids[page], page=page, mode="full"))

    # The delta pass after every full one, so each delta edits a page built from
    # the same bank, and the wave is the only thing new to all of them.
    await client.aretain_batch(bank_id=bank, items=new_wave())
    await settled(bank)
    for page in pages:
        await client.aupdate_mental_model(bank_id=bank, mental_model_id=ids[page], trigger=page_trigger("delta"))
        await settled(bank)
        results.append(await measure_refresh(client, bank, ids[page], page=page, mode="delta"))
    return results


def _report(request: pytest.FixtureRequest, results: list[RefreshCost]) -> None:
    model = provider_environment()["HINDSIGHT_API_LLM_MODEL"] if not request.config.getoption("--api-url") else "remote"
    price = PRICES.get(model)
    # The server's own default unless this run overrode it.
    explicit = os.getenv("HINDSIGHT_EVAL_SET_REFLECT_PROMPT_CACHE_ENABLED", "true").lower() != "false"
    table = summary_table(results, price, explicit)
    log.info("refresh cost baseline:\n%s", table)
    out = request.config.getoption("--cost-output")
    if not out:
        return
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    report = CostReport(
        timestamp=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        model=provider_environment()["HINDSIGHT_API_LLM_MODEL"]
        if not request.config.getoption("--api-url")
        else "remote",
        fixture=FIXTURE.name,
        explicit_cache=explicit,
        refreshes=results,
    )
    (directory / "refresh-cost.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    (directory / "refresh-cost.txt").write_text(table + "\n", encoding="utf-8")
    # The synthesis prompt of every refresh, readable on its own: the thing to
    # open when a number moves.
    for r in results:
        if r.final_call is not None:
            name = f"{r.page.lower().replace(' ', '-')}-{r.mode}-final-prompt.json"
            (directory / name).write_text(
                json.dumps(r.final_call.prompt, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
            )
    log.info("refresh cost report written to %s", directory)


def _assert_measured(results: list[RefreshCost]) -> None:
    # The only hard claims: every refresh reached the model, and the trace saw
    # it. A refresh with no traced calls would record a baseline of zero.
    for r in results:
        # No synthesis call is not a failure: a refresh that answers through the
        # ``done`` tool writes its page from the tool loop alone.
        assert r.calls, f"{r.page} ({r.mode}): no LLM calls traced — is HINDSIGHT_API_LLM_TRACE_ENABLED off?"


async def test_refresh_cost_minimum(request: pytest.FixtureRequest, client: Hindsight, settled: SettleFn) -> None:
    bank = await _import_fixture(client, f"syseval-{uuid.uuid4().hex[:10]}", settled)
    results = await _measure(client, bank, settled, MINIMUM_PAGES)
    _report(request, results)
    _assert_measured(results)


@pytest.mark.full
async def test_refresh_cost_full(request: pytest.FixtureRequest, client: Hindsight, settled: SettleFn) -> None:
    bank = await _import_fixture(client, f"syseval-{uuid.uuid4().hex[:10]}", settled)
    results = await _measure(client, bank, settled, tuple(PAGES))
    _report(request, results)
    _assert_measured(results)
