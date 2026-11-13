"""Knowledge banks: retrieval, end to end through the API, one category per kind of content.

Each case writes a public corpus into a knowledge bank, waits for ingestion, searches every
query and scores the result with the leaderboard's own metric (nDCG@10, recall@k). Scores,
ingest rate and search latency go to ``--output`` under ``knowledge`` and print with
``eval-table``. No LLM: the model under test is the server's knowledge embedder. See the
README's "Knowledge banks" section for the leaderboards and the numbers to beat.

Opt-in: these need ``uv sync --extra knowledge`` and download datasets into
``$HINDSIGHT_EVAL_KNOWLEDGE_CACHE``; without the extra every case skips.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator

import pytest

for _module in ("pyarrow", "pypdfium2", "PIL", "huggingface_hub"):
    pytest.importorskip(_module, reason="knowledge suites need `uv sync --extra knowledge`")

from hindsight_system_evals.knowledge import KnowledgeApi, embedder_media_types, embedder_name  # noqa: E402
from hindsight_system_evals.knowledge_cases import CASES, MINIMUM, Bank, Case, Context, run_case  # noqa: E402
from hindsight_system_evals.report import KNOWLEDGE, case_params  # noqa: E402
from hindsight_system_evals.target import Target  # noqa: E402

pytestmark = pytest.mark.timeout(6 * 3600)


@pytest.fixture(scope="session")
def knowledge_banks(request: pytest.FixtureRequest, target: Target) -> Iterator[dict[str, Bank]]:
    """Banks by corpus, shared by every case on that corpus for the whole session."""
    banks: dict[str, Bank] = {}
    yield banks
    # pg0 banks go with the server; a remote tenant keeps them forever unless deleted.
    if target.is_remote and not request.config.getoption("--keep-banks"):
        import asyncio

        async def drop() -> None:
            api = KnowledgeApi(target.url, target.api_key)
            for bank in banks.values():
                with contextlib.suppress(Exception):
                    await api.delete_bank(bank.bank_id)
            await api.aclose()

        asyncio.run(drop())


@pytest.mark.parametrize("case", case_params(CASES, MINIMUM))
async def test_knowledge(case: Case, request: pytest.FixtureRequest, target: Target, knowledge_banks: dict[str, Bank]):
    client = KnowledgeApi(target.url, target.api_key)
    media = embedder_media_types()
    if case.needs not in media:
        pytest.skip(f"the server's knowledge embedder does not write {case.needs} (it writes {media})")
    ctx = Context(
        client=client,
        full=request.config.getoption("--full"),
        embedder=embedder_name(),
        banks=knowledge_banks,
    )
    try:
        outcome = await run_case(ctx, case)
    finally:
        await client.aclose()
    KNOWLEDGE.append(outcome)
    print(f"\n{case.category}/{case.id} [{outcome.embedder}] {outcome.corpus} docs, {outcome.queries} queries")
    print(json.dumps(outcome.model_dump(exclude={"category", "case", "embedder"}), indent=1))
