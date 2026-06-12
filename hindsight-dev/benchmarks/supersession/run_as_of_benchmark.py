"""as_of recall benchmark — end-to-end runner for the synthesizer's pairs.

This is the actual scoring harness the D-experiment can use for the
as_of contract. The synthesizer produces (question, gold_answer, as_of,
scenario) quadruples; the runner ingests a LoCoMo conversation into a
real MemoryEngine and, for each pair, calls ``recall(as_of=...)`` and
checks whether the gold answer appears in the recalled context.

Scoring rules:

* ``after`` scenario  — gold answer MUST be reachable from the recall
  output (token-overlap check on the rendered context, not an LLM
  verdict; LLM judge is reserved for downstream answer-generation
  accuracy, not for the recall index contract).
* ``before`` scenario — gold answer MUST NOT be reachable. This is the
  false-positive guard: if as_of is being ignored (i.e. the index
  returns everything regardless of validity), every "before" pair
  would score wrong and the failure rate would be ~100%.
* ``spanning`` scenario — informational only; the contract here is
  "the recall should be a subset of the full answer set", which we
  don't try to score precisely.

Two metrics are reported:

  * ``true_positive_rate`` — fraction of ``after`` pairs that hit.
    High = recall index returns the answer at the as_of date.
  * ``true_negative_rate`` — fraction of ``before`` pairs that miss.
    High = as_of is being respected; the index actually filters by
    validity window.

A perfect system scores 1.0 / 1.0. A system that ignores as_of scores
1.0 / 0.0 (always hits). A broken system scores 0.0 / 1.0 (always
empty). The product is the headline number; either rate dropping is
informative.

Run from the project root:

    uv run python -m benchmarks.supersession.run_as_of_benchmark \\
        --conversations 2 --per-scenario 10

Or with mock LLM (plumbing only, no real recall — useful to verify the
harness runs without an API key):

    uv run python -m benchmarks.supersession.run_as_of_benchmark --mock
"""

import argparse
import asyncio
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from benchmarks.common.benchmark_runner import create_memory_engine
from benchmarks.locomo.locomo_benchmark import LoComoDataset

from .as_of_synthesizer import AsOfPair, AsOfScenario, build_as_of_pairs, cap_per_scenario

_DATASET_PATH = Path(__file__).parent.parent / "locomo" / "datasets" / "locomo10.json"

_ANSWER_PROMPT = """You answer questions using ONLY the recalled memory context.
Reply with a short phrase or 'I don't know' if the context does not contain the answer."""


@dataclass
class AsOfResult:
    pair: AsOfPair
    recalled_text: str
    gold_present: bool


@dataclass
class AsOfReport:
    """Per-scenario scoring summary."""

    true_positive_rate: float
    true_negative_rate: float
    spanning_skipped: int
    counts_by_scenario: dict[str, int] = field(default_factory=dict)
    hits_by_scenario: dict[str, int] = field(default_factory=dict)


def _gold_present(gold: str, recalled_text: str) -> bool:
    """Cheap deterministic check: does the gold answer (loosely) appear in
    the recalled context? Token overlap — splits on whitespace, lowercases,
    requires the FIRST gold token and at least 50% of the rest to appear.

    This is a recall-index contract test, not a generation-accuracy test.
    The check is intentionally lenient (LLM-free, fast) so it can run in
    CI. Downstream answer-accuracy uses the LLM judge (see run_dual_memory).
    """
    gold_tokens = [t for t in gold.lower().split() if len(t) > 1]
    if not gold_tokens:
        return False
    recalled_lower = recalled_text.lower()
    first = gold_tokens[0]
    if first not in recalled_lower:
        return False
    rest = gold_tokens[1:]
    if not rest:
        return True
    hits = sum(1 for t in rest if t in recalled_lower)
    return hits / len(rest) >= 0.5


async def _recall_with_as_of(memory, bank_id: str, question: str, as_of: datetime):
    """Run a single recall with the as_of date set.

    Returns the rendered text of the top memories (concatenated) so the
    gold-answer presence check works on the same string the LLM
    generator would see. Keeps the harness LLM-free.
    """
    from hindsight_api.engine.memory_engine import Budget
    from hindsight_api.models import RequestContext

    result = await memory.recall_async(
        bank_id=bank_id,
        query=question,
        budget=Budget.MID,
        max_tokens=2048,
        fact_type=["world", "experience"],
        as_of=as_of,
        request_context=RequestContext(),
        _quiet=True,
    )
    return "\n".join(r.text for r in result.results)


async def _run_one_pair(memory, bank_id: str, pair: AsOfPair) -> AsOfResult:
    recalled = await _recall_with_as_of(memory, bank_id, pair.question, pair.as_of)
    return AsOfResult(pair=pair, recalled_text=recalled, gold_present=_gold_present(pair.gold_answer, recalled))


def _score(results: list[AsOfResult]) -> AsOfReport:
    by_scenario: dict[AsOfScenario, list[AsOfResult]] = {"before": [], "after": [], "spanning": []}
    for r in results:
        by_scenario[r.pair.scenario].append(r)

    counts = {s: len(rs) for s, rs in by_scenario.items()}
    hits = {s: sum(1 for r in rs if r.gold_present) for s, rs in by_scenario.items()}

    after = by_scenario["after"]
    before = by_scenario["before"]
    tp = (hits["after"] / counts["after"]) if counts["after"] else 1.0
    tn = ((counts["before"] - hits["before"]) / counts["before"]) if counts["before"] else 1.0
    return AsOfReport(
        true_positive_rate=round(tp, 4),
        true_negative_rate=round(tn, 4),
        spanning_skipped=counts["spanning"],
        counts_by_scenario=counts,
        hits_by_scenario=hits,
    )


def _markdown(report: AsOfReport, meta: dict) -> str:
    return "\n".join(
        [
            "# as_of recall benchmark",
            "",
            f"- date: {meta['date']}  conversations: {meta['conversations']}  per-scenario: {meta['per_scenario']}",
            f"- model: {meta['model']}  elapsed: {meta['elapsed_s']}s",
            "",
            "| scenario | count | hit | rate |",
            "|---|---|---|---|",
            *(
                f"| {s} | {report.counts_by_scenario.get(s, 0)} | "
                f"{report.hits_by_scenario.get(s, 0)} | "
                f"{(report.hits_by_scenario.get(s, 0) / report.counts_by_scenario[s] * 100):.0f}% |"
                for s in ("after", "before", "spanning")
                if report.counts_by_scenario.get(s, 0)
            ),
            "",
            "## Headline",
            "",
            f"- true_positive_rate (after): {report.true_positive_rate:.0%} "
            f"(system finds the answer at the as_of date)",
            f"- true_negative_rate (before): {report.true_negative_rate:.0%} "
            f"(system does NOT find the answer before the session is ingested)",
            "",
            f"spanning scenarios ({report.spanning_skipped}) are reported but not "
            "scored — their contract is 'subset of full answer set' which needs an LLM "
            "judge to score precisely.",
            "",
        ]
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversations", type=int, default=2)
    parser.add_argument("--per-scenario", type=int, default=10)
    parser.add_argument("--mock", action="store_true", help="plumbing smoke test (no real recall)")
    parser.add_argument("--output", default=None, help="output prefix (default results/as_of_<ts>)")
    args = parser.parse_args()

    if args.mock:
        os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
        os.environ.setdefault("HINDSIGHT_API_LLM_API_KEY", "")
        os.environ.setdefault("HINDSIGHT_API_LLM_MODEL", "mock")
        print("mock mode: skipping real recall and scoring")
        return

    items = LoComoDataset().load(_DATASET_PATH, max_items=args.conversations)
    all_pairs: list[AsOfPair] = []
    for item in items:
        all_pairs.extend(cap_per_scenario(build_as_of_pairs(item), args.per_scenario))
    print(f"conversations: {len(items)}, pairs: {len(all_pairs)}")

    memory = await create_memory_engine()
    bank_id = f"as-of-bench-{uuid.uuid4().hex[:8]}"
    started = time.time()

    # Ingest all sessions from all items into the same bank. The as_of
    # filter is the only thing differentiating "before" and "after"
    # pairs, so the bank content is fixed for the whole run.
    from hindsight_api.models import RequestContext

    dataset = LoComoDataset()
    for item in items:
        for session in dataset.prepare_sessions_for_ingestion(item):
            await memory.retain_async(
                bank_id=bank_id,
                content=session["content"],
                event_date=session["event_date"],
                request_context=RequestContext(),
            )

    # Sequential recall — the runner is intentionally serial so the
    # reported elapsed_s is interpretable (parallel recall would
    # confound the latency signal the benchmark also reports).
    results: list[AsOfResult] = []
    for pair in all_pairs:
        result = await _run_one_pair(memory, bank_id, pair)
        results.append(result)

    await memory.close()

    report = _score(results)
    meta = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "conversations": len(items),
        "per_scenario": args.per_scenario,
        "model": os.getenv("HINDSIGHT_API_LLM_MODEL", "?"),
        "elapsed_s": round(time.time() - started, 1),
    }
    md = _markdown(report, meta)
    print("\n" + md)

    prefix = args.output or f"results/as_of_{datetime.now(UTC):%Y%m%d_%H%M%S}"
    out = Path(prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    serializable = [
        {
            "conv_id": r.pair.conv_id,
            "question": r.pair.question,
            "gold_answer": r.pair.gold_answer,
            "as_of": r.pair.as_of.isoformat(),
            "scenario": r.pair.scenario,
            "evidence_sessions": list(r.pair.evidence_sessions),
            "gold_present": r.gold_present,
        }
        for r in results
    ]
    out.with_suffix(".json").write_text(
        json.dumps({"meta": meta, "report": asdict(report), "results": serializable}, indent=2, ensure_ascii=False)
    )
    out.with_suffix(".md").write_text(md)
    print(f"saved: {out.with_suffix('.json')} / {out.with_suffix('.md')}")


if __name__ == "__main__":
    asyncio.run(main())
