"""Knowledge banks take odd, broken and big files: each is processed, or refused with a reason.

No model call and no judge. The server's own embedder (local EmbeddingGemma 2 by default) embeds,
so the suite runs with ``HINDSIGHT_EVAL_LLM_PROVIDER=none``.

* ``test_edge_case`` — the committed sample (``fixtures/kb-edge-cases/manifest.json``): one file per
  issue a corpus sweep (``python -m hindsight_system_evals.kb_edge_cases harvest``) found. Each must
  end as the manifest says (``ready``, or ``failed`` / ``rejected`` with its reason), and a ready one
  must come back from a search for its query, among every other case in the same bank.
* ``test_heavy_file`` (``--full``) — common uploads, only big (a minute of 4K video, a 108 MP photo, an hour
  of audio, a 1500-page PDF, a 50 MB CSV), built at run time with ffmpeg; and a 3 GB file over the size
  limit, refused without the server's memory growing by it. Each common one must become ready and be
  found; what it cost the server (peak memory, CPU seconds, wall time) is printed, and appended as
  JSON lines to ``HINDSIGHT_EVAL_KB_USAGE`` when set, so two runs can be compared.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
import pytest

from hindsight_system_evals import Target
from hindsight_system_evals.kb_edge_cases import (
    FIXTURES,
    HEAVY_CASES,
    KB,
    EdgeCase,
    FileOutcome,
    build_heavy,
    create_bank,
    ingest_file,
    load_manifest,
    rank_of,
    server_alive,
)

#: Under the worker's first retry delay (60s), so a file failing late was not retried.
FAIL_FAST_SECONDS = 45
#: The stored passage text of every case with ``text_contains``, read back after its ingest.
TEXTS: dict[str, str] = {}
MANIFEST = load_manifest()
HEAVY_DIR = Path(os.getenv("HINDSIGHT_EVAL_KB_HEAVY_DIR", Path.home() / ".cache" / "hindsight-evals" / "kb-heavy"))


@dataclass(frozen=True)
class CaseResult:
    outcome: FileOutcome
    #: Where the document came back for its query (1-based), or None.
    rank: int | None


def _record(case_id: str, outcome: FileOutcome, rank: int | None) -> None:
    usage = outcome.usage
    print(
        f"\n{case_id}: {outcome.status} in {outcome.seconds}s, {outcome.passage_count} passages, rank {rank}"
        + (f", memory +{usage.growth_mb}MB (peak {usage.peak_rss_mb}MB), {usage.cpu_seconds} cpu-s" if usage else "")
    )
    if sink := os.getenv("HINDSIGHT_EVAL_KB_USAGE"):
        with open(sink, "a") as f:
            f.write(json.dumps({"case": case_id, "rank": rank, **asdict(outcome)}) + "\n")


@pytest.fixture(scope="module")
def sample(target: Target) -> dict[str, CaseResult]:
    """Every committed case ingested into ONE bank, so each search has the others as distractors."""

    async def run() -> dict[str, CaseResult]:
        async with httpx.AsyncClient(base_url=target.url, timeout=120, headers=_auth(target)) as http:
            bank = await create_bank(http)
            outcomes = {}
            for case in MANIFEST:
                path = FIXTURES / case.file
                outcomes[case.id] = await ingest_file(http, bank, path, document_id=case.id, pid=target.pid)
                assert server_alive(target.pid), f"the server died ingesting {case.file}"
            ranks = {
                case.id: await rank_of(http, bank, case.query, case.id)
                for case in MANIFEST
                if case.query and outcomes[case.id].status == "ready"
            }
            for case in MANIFEST:
                if case.text_contains and outcomes[case.id].status == "ready":
                    detail = (await http.get(f"{KB}/{bank}/documents/{case.id}")).json()
                    TEXTS[case.id] = "\n".join(p.get("text") or "" for p in detail["passages"])
            return {cid: CaseResult(o, ranks.get(cid)) for cid, o in outcomes.items()}

    return asyncio.run(run())


def _auth(target: Target) -> dict[str, str]:
    return {"Authorization": f"Bearer {target.api_key}"} if target.api_key else {}


@pytest.mark.parametrize("case", MANIFEST, ids=[f"{c.category}/{c.id}" for c in MANIFEST])
def test_edge_case(case: EdgeCase, sample: dict[str, CaseResult]) -> None:
    outcome, rank = sample[case.id].outcome, sample[case.id].rank
    _record(case.id, outcome, rank)
    assert outcome.handled, f"{case.file}: {outcome.status} ({outcome.http_status}) {outcome.error}"
    assert outcome.status == case.expect, f"{case.file}: expected {case.expect}, got {outcome}"
    if case.expect == "ready":
        assert outcome.passage_count > 0, outcome
        if case.query:
            assert rank is not None, f"{case.file} is ready but not in the top 10 for {case.query!r}"
        if case.text_contains:
            # Found is not enough: garbled text (UTF-16 read as UTF-8) still ranked among a few files.
            assert case.text_contains in TEXTS.get(case.id, ""), TEXTS.get(case.id, "")[:300]
    else:
        # For good, at once: a file that cannot be read fails the same way every time, and an error
        # the worker took for a transient one was retried a minute apart, three times.
        assert outcome.seconds < FAIL_FAST_SECONDS, f"{case.file} took {outcome.seconds}s to fail: retried?"
        if case.error_contains:
            assert case.error_contains in (outcome.error or ""), outcome.error


@pytest.fixture(scope="module")
def warm(target: Target) -> None:
    """The embedder loaded before any heavy file is measured: its first use costs gigabytes on its own."""
    tiny = HEAVY_DIR / "warm-up.txt"
    HEAVY_DIR.mkdir(parents=True, exist_ok=True)
    tiny.write_text("A warm-up passage, so the model is loaded before anything is measured.")

    async def run() -> None:
        async with httpx.AsyncClient(base_url=target.url, timeout=120, headers=_auth(target)) as http:
            await ingest_file(http, await create_bank(http), tiny, document_id="warm-up", pid=target.pid)

    asyncio.run(run())


@pytest.mark.full
@pytest.mark.parametrize("case", HEAVY_CASES, ids=[f"heavy/{c.id}" for c in HEAVY_CASES])
def test_heavy_file(case, target: Target, warm: None) -> None:
    if case.ffmpeg is not None and not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is not on PATH")
    HEAVY_DIR.mkdir(parents=True, exist_ok=True)
    path = build_heavy(case, HEAVY_DIR)

    async def run() -> CaseResult:
        async with httpx.AsyncClient(base_url=target.url, timeout=600, headers=_auth(target)) as http:
            bank = await create_bank(http)
            outcome = await ingest_file(http, bank, path, document_id=case.id, pid=target.pid, timeout=3600)
            ready = outcome.status == "ready" and case.query
            return CaseResult(outcome, await rank_of(http, bank, case.query, case.id) if ready else None)

    result = asyncio.run(run())
    outcome, rank = result.outcome, result.rank
    _record(case.id, outcome, rank)
    assert server_alive(target.pid), f"the server died ingesting {path.name}"
    assert outcome.status == case.expect, outcome
    if case.expect == "ready":
        assert rank is not None, f"{path.name} is ready but not found for {case.query!r}"
    if case.max_growth_mb is not None and outcome.usage is not None:
        # The file is never held whole: a 3 GB upload refused for its size grew the server by 3 GB first.
        assert outcome.usage.growth_mb < case.max_growth_mb, (
            f"{path.name} ({outcome.size_bytes / 1048576:.0f} MB) grew the server by {outcome.usage.growth_mb} MB"
        )
