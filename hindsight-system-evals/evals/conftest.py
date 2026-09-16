"""Fixtures for the eval suite.

Same shape as the system tests: one server for the session, a client, and a
settle helper. The differences are the ones a real model forces —
credentials are required, and the judge is checked against the server's model so
a run cannot quietly grade itself.
"""

from __future__ import annotations

import contextlib
import os
import uuid
import warnings
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from hindsight_client import Hindsight

from hindsight_system_evals import (
    Target,
    eval_target,
    judge_model,
    provider_environment,
    wait_until_settled,
)
from hindsight_system_evals.pages import SettleFn
from hindsight_system_evals.target import ENV_API_KEY, ENV_API_URL
from hindsight_system_evals.report import RECORDED, ModelConfig, ModelRef, RunReport, summarise

BANK_PREFIX = "syseval-"


@pytest.fixture(scope="session")
def target(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Target]:
    log_path: Path = tmp_path_factory.mktemp("hindsight-eval-server") / "server.log"
    with eval_target(request.config.getoption("--api-url"), log_path=log_path) as resolved:
        print(f"system-evals target: {resolved.describe()}")
        yield resolved


@pytest.fixture(scope="session", autouse=True)
def _judge_is_independent(target: Target) -> None:
    """Warn when the judge and the model under test are the same.

    Not an error, because a single-provider setup is a legitimate way to run
    locally — but a model grading its own output agrees with itself, and a run
    that does so silently is worth less than it appears.

    Only answerable for a server this run configured. A remote target's model is
    its own business, and reading it back here would be one HTTP call to produce
    a warning.
    """
    if target.is_remote:
        return
    server_model = provider_environment()["HINDSIGHT_API_LLM_MODEL"]
    if judge_model() == server_model:
        warnings.warn(
            f"The judge and the model under test are both {server_model!r}. "
            "Set HINDSIGHT_EVAL_JUDGE_MODEL to something else — self-grading inflates the score.",
            stacklevel=1,
        )


@pytest.fixture
async def client(target: Target) -> AsyncIterator[Hindsight]:
    client = Hindsight(base_url=target.url, api_key=target.api_key)
    yield client
    await client.aclose()


@pytest.fixture
async def bank_id(request: pytest.FixtureRequest, target: Target, client: Hindsight) -> AsyncIterator[str]:
    # One bank per eval. Sharing would let every page's refresh reflect over
    # every other question's corpus, which is not the scenario and makes a
    # failure impossible to attribute.
    bank = f"{BANK_PREFIX}{uuid.uuid4().hex[:10]}"
    yield bank
    # On pg0 the bank is thrown away with the server, and leaving it costs
    # nothing — it is how a failure gets inspected in the control plane. A shared
    # remote tenant is the opposite: every run would leave banks behind forever.
    if target.is_remote and not request.config.getoption("--keep-banks"):
        with contextlib.suppress(Exception):
            await client.adelete_bank(bank)


@pytest.fixture
def settled(client: Hindsight) -> SettleFn:
    async def _settle(bank: str) -> None:
        await wait_until_settled(client, bank)

    return _settle


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "full: the complete set; deselected in minimum-acceptance runs")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Minimum acceptance is the default; ``--full`` opts into everything.

    CI runs the minimum set, so the gate stays inside a sensible wall-clock
    budget and its threshold can be set against measured variance. The full set
    is for humans changing a prompt, where breadth matters more than duration.
    """
    if config.getoption("--full"):
        return
    skip_full = pytest.mark.skip(reason="full-only; pass --full to include")
    for item in items:
        if "full" in item.keywords:
            item.add_marker(skip_full)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--full", action="store_true", default=False, help="run the complete eval set")
    parser.addoption(
        "--output",
        default=None,
        help="write every page outcome as JSON here — what the perf dashboard publishes",
    )
    parser.addoption(
        "--api-url",
        default=None,
        help=(
            "run against a server that is already up (cloud dev, a colleague's box) instead of "
            f"starting one; ${ENV_API_URL} does the same, and ${ENV_API_KEY} carries its key"
        ),
    )
    parser.addoption(
        "--keep-banks",
        action="store_true",
        default=False,
        help="do not delete the banks a remote run creates — for inspecting a failure",
    )


def _hindsight_model(config: pytest.Config) -> ModelRef:
    """Which model produced these answers.

    Locally it is the one this run configured. Against a remote target it is read
    back from the server, because that server's model is not ours to assert, and
    a report naming the model we merely wished for is worse than one naming none.
    """
    if not (config.getoption("--api-url") or os.getenv(ENV_API_URL)):
        env = provider_environment()
        return ModelRef(provider=env["HINDSIGHT_API_LLM_PROVIDER"], model=env["HINDSIGHT_API_LLM_MODEL"])

    with contextlib.suppress(Exception):
        url = config.getoption("--api-url") or os.environ[ENV_API_URL]
        cfg = Hindsight(base_url=url, api_key=os.getenv(ENV_API_KEY)).get_bank_config(RECORDED[0].bank_id)
        return ModelRef(provider=cfg.get("llm_provider"), model=cfg.get("llm_model") or "unknown")
    return ModelRef(provider="remote", model="unknown")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    output = session.config.getoption("--output")
    if not output:
        return
    import datetime

    overall = summarise(RECORDED)
    report = RunReport(
        timestamp=datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        suite="system-evals",
        mode="full" if session.config.getoption("--full") else "minimum-acceptance",
        llm_config=ModelConfig(
            hindsight=_hindsight_model(session.config),
            judge=ModelRef(model=judge_model()),
        ),
        total=overall.total,
        correct=overall.correct,
        correct_rate=(overall.correct / overall.total) if overall.total else None,
        trap_count=overall.trap_count,
        by_kind={kind: summarise([r for r in RECORDED if r.kind == kind]) for kind in sorted({r.kind for r in RECORDED})},
        items=RECORDED,
    )
    Path(output).write_text(report.to_json(), encoding="utf-8")
