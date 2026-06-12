"""D-experiment orchestrator: three arms × three task categories.

Usage (from hindsight-dev/):
    uv run python -m benchmarks.dual_memory.run_dual_memory \\
        --arms hindsight,graphiti,dual --conversations 3 --per-category 15

Outputs per-(arm, category) accuracy plus the pre-registered decision rules
from HINDSIGHT_GRAPHITI_EVAL_4WAY.md §3/§5:
  * shared gain  = dual − hindsight on blind+spanning  (≥ +10pp → C track)
  * private regression = hindsight − dual on private   (> 5pp → routing rework)

``--mock`` runs the Hindsight arm with the mock LLM and skips answering and
judging — a plumbing smoke test that needs no API key.
"""

import argparse
import asyncio
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from benchmarks.common.benchmark_runner import LLMAnswerEvaluator, create_memory_engine
from benchmarks.locomo.locomo_benchmark import LoComoDataset

from .arms import DualArm, GraphitiArm, HindsightArm, MemoryArm
from .taskset import Task, build_tasks, cap_per_category, split_sessions

_DATASET_PATH = Path(__file__).parent.parent / "locomo" / "datasets" / "locomo10.json"

_ANSWER_PROMPT = """You answer questions using ONLY the provided memory context.
Be concise (a short phrase or sentence). If the context does not contain the
answer, reply exactly: I don't know."""


@dataclass
class TaskResult:
    arm: str
    category: str
    question: str
    gold_answer: str
    predicted: str
    correct: bool
    judge_reasoning: str


@dataclass
class ArmReport:
    arm: str
    accuracy_by_category: dict[str, float] = field(default_factory=dict)
    counts_by_category: dict[str, int] = field(default_factory=dict)


async def _generate_answer(llm_config, context: str, question: str) -> str:
    raw, _usage = await llm_config.call(
        messages=[
            {"role": "system", "content": _ANSWER_PROMPT},
            {"role": "user", "content": f"MEMORY CONTEXT:\n{context}\n\nQUESTION: {question}"},
        ],
        response_format=None,
        scope="dual_memory_answer",
        temperature=0.0,
        skip_validation=True,
        return_usage=True,
    )
    return str(raw).strip()


async def _run_arm(
    arm: MemoryArm,
    items: list[dict],
    tasks: list[Task],
    evaluator: "LLMAnswerEvaluator | None",
    answer_llm,
    concurrency: int = 4,
) -> list[TaskResult]:
    await arm.setup()
    for item in items:
        split = split_sessions(item)
        print(f"  [{arm.name}] ingesting {split.conv_id} ...")
        await arm.ingest(item, split)

    if evaluator is None:  # mock mode: plumbing only
        sample = await arm.retrieve(tasks[0])
        print(f"  [{arm.name}] retrieve smoke OK ({len(sample)} chars)")
        return []

    semaphore = asyncio.Semaphore(concurrency)
    results: list[TaskResult] = []

    async def _one(task: Task) -> TaskResult:
        async with semaphore:
            context = await arm.retrieve(task)
            predicted = await _generate_answer(answer_llm, context, task.question)
        correct, reasoning = await evaluator.judge_answer(
            question=task.question,
            correct_answer=task.gold_answer,
            predicted_answer=predicted,
            semaphore=semaphore,
        )
        return TaskResult(
            arm=arm.name,
            category=task.category,
            question=task.question,
            gold_answer=task.gold_answer,
            predicted=predicted,
            correct=correct,
            judge_reasoning=reasoning,
        )

    for coro in asyncio.as_completed([_one(t) for t in tasks]):
        result = await coro
        results.append(result)
        mark = "Y" if result.correct else "N"
        print(f"  [{arm.name}][{result.category}][{mark}] {result.question[:60]}")
    await arm.close()
    return results


def _aggregate(results: list[TaskResult]) -> dict[str, ArmReport]:
    reports: dict[str, ArmReport] = {}
    for r in results:
        report = reports.setdefault(r.arm, ArmReport(arm=r.arm))
        report.counts_by_category[r.category] = report.counts_by_category.get(r.category, 0) + 1
        report.accuracy_by_category[r.category] = report.accuracy_by_category.get(r.category, 0.0) + (
            1.0 if r.correct else 0.0
        )
    for report in reports.values():
        for cat, total in report.counts_by_category.items():
            report.accuracy_by_category[cat] = round(report.accuracy_by_category[cat] / total, 4)
    return reports


def _pooled(report: ArmReport, categories: tuple[str, ...]) -> float | None:
    total = sum(report.counts_by_category.get(c, 0) for c in categories)
    if total == 0:
        return None
    hits = sum(report.accuracy_by_category.get(c, 0.0) * report.counts_by_category.get(c, 0) for c in categories)
    return hits / total


def _decision_rules(reports: dict[str, ArmReport]) -> list[str]:
    lines: list[str] = []
    hindsight, dual = reports.get("hindsight"), reports.get("dual")
    if hindsight and dual:
        shared_h = _pooled(hindsight, ("blind", "spanning"))
        shared_d = _pooled(dual, ("blind", "spanning"))
        if shared_h is not None and shared_d is not None:
            gain = (shared_d - shared_h) * 100
            verdict = "C-track signal CONFIRMED" if gain >= 10 else "below the +10pp bar"
            lines.append(f"shared gain (dual − hindsight on blind+spanning): {gain:+.1f}pp — {verdict}")
        private_h = _pooled(hindsight, ("private",))
        private_d = _pooled(dual, ("private",))
        if private_h is not None and private_d is not None:
            regression = (private_h - private_d) * 100
            verdict = "tool routing needs rework" if regression > 5 else "within tolerance"
            lines.append(f"private regression (hindsight − dual on private): {regression:+.1f}pp — {verdict}")
    if not lines:
        lines.append("decision rules need both 'hindsight' and 'dual' arms")
    return lines


def _markdown(reports: dict[str, ArmReport], rules: list[str], meta: dict) -> str:
    cats = ("private", "blind", "spanning")
    lines = [
        "# Dual-memory D experiment",
        "",
        f"- date: {meta['date']}  conversations: {meta['conversations']}  tasks: {meta['tasks']}",
        f"- answer/judge model: {meta['model']}",
        "",
        "| arm | " + " | ".join(cats) + " |",
        "|---|" + "---|" * len(cats),
    ]
    for arm, report in sorted(reports.items()):
        cells = [
            f"{report.accuracy_by_category.get(c, float('nan')):.0%} (n={report.counts_by_category.get(c, 0)})"
            if report.counts_by_category.get(c)
            else "—"
            for c in cats
        ]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines += ["", "## Pre-registered decision rules", ""]
    lines += [f"- {rule}" for rule in rules]
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", default="hindsight,graphiti,dual")
    parser.add_argument("--conversations", type=int, default=2)
    parser.add_argument("--per-category", type=int, default=10, help="task cap per category per conversation")
    parser.add_argument("--mock", action="store_true", help="mock-LLM plumbing smoke test (hindsight arm only)")
    parser.add_argument("--output", default=None, help="output path prefix (default results/dual_memory_<ts>)")
    args = parser.parse_args()

    items = LoComoDataset().load(_DATASET_PATH, max_items=args.conversations)
    all_tasks: list[Task] = []
    for item in items:
        split = split_sessions(item)
        all_tasks.extend(cap_per_category(build_tasks(item, split), args.per_category))
    counts = {c: sum(1 for t in all_tasks if t.category == c) for c in ("private", "blind", "spanning")}
    print(f"conversations: {len(items)}, tasks: {len(all_tasks)}, mix: {counts}")

    run_id = uuid.uuid4().hex[:8]
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]

    if args.mock:
        import os

        os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
        os.environ.setdefault("HINDSIGHT_API_LLM_API_KEY", "")
        os.environ.setdefault("HINDSIGHT_API_LLM_MODEL", "mock")
        arm_names = ["hindsight"]

    memory = await create_memory_engine() if {"hindsight", "dual"} & set(arm_names) else None
    hindsight_arm = HindsightArm(memory, run_id) if memory else None
    graphiti_arm = GraphitiArm(run_id) if {"graphiti", "dual"} & set(arm_names) else None
    arms: list[MemoryArm] = []
    for name in arm_names:
        if name == "hindsight":
            arms.append(hindsight_arm)
        elif name == "graphiti":
            arms.append(graphiti_arm)
        elif name == "dual":
            arms.append(DualArm(hindsight_arm, graphiti_arm))
        else:
            raise SystemExit(f"unknown arm: {name}")

    evaluator = None if args.mock else LLMAnswerEvaluator()
    answer_llm = None if args.mock else LLMAnswerEvaluator().llm_config

    started = time.time()
    results: list[TaskResult] = []
    for arm in arms:
        print(f"\n=== arm: {arm.name} ===")
        try:
            results.extend(await _run_arm(arm, items, all_tasks, evaluator, answer_llm))
        except Exception:
            # One arm's failure must not discard the others' finished results.
            import traceback

            traceback.print_exc()
            print(f"  [{arm.name}] ARM FAILED — continuing with remaining arms")

    if memory is not None:
        await memory.close()
    if args.mock:
        print("\nmock smoke test complete — plumbing OK")
        return

    reports = _aggregate(results)
    rules = _decision_rules(reports)
    meta = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "conversations": len(items),
        "tasks": len(all_tasks),
        "model": evaluator.model,
        "elapsed_s": round(time.time() - started, 1),
    }
    markdown = _markdown(reports, rules, meta)
    print("\n" + markdown)

    prefix = args.output or f"results/dual_memory_{datetime.now(UTC):%Y%m%d_%H%M%S}"
    out = Path(prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(
        json.dumps({"meta": meta, "results": [asdict(r) for r in results]}, ensure_ascii=False, indent=2)
    )
    out.with_suffix(".md").write_text(markdown)
    print(f"\nsaved: {out.with_suffix('.json')} / {out.with_suffix('.md')}")


if __name__ == "__main__":
    asyncio.run(main())
