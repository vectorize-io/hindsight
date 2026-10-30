"""Reflect abstain benchmark: does the decision model say when a memory watcher should speak?

A coding agent's memory watcher calls ``reflect`` with ``abstain`` at each moment of the agent's
work (its request, a plan, a command, an edit). Before any LLM call, the decision model routes the
moment one of three ways: **silent** (no LLM call), **writer** (the memories settle it, a writer only
words the note), or **check** (the reflect LLM reads the memories and decides). This benchmark
replays labeled moments through that routing and counts the two mistakes that matter:

- a **missed note**: the moment should speak, and it was routed silent;
- a **false note**: the moment should stay silent, and it went to the writer, which has no veto.

``check`` is never wrong, only slower: it costs an LLM call.

Two modes, on whichever decision model the server is set up with
(``HINDSIGHT_API_DECISION_MODEL_*``, ``HINDSIGHT_API_REFLECT_DECISION_MODEL_*``):

- ``pick`` (default): reflect's own code, ``pick_answering`` and the cut-offs in
  ``engine/reflect/agent.py``. This is what ships.
- ``verdict``: one 8-option question about the moment, routed on the summed probability of its two
  "speak" options. A candidate replacement for ``pick``, kept here so both can be measured on any model.

Usage:
    cd hindsight-dev
    uv run python -m benchmarks.reflect_abstain.reflect_abstain_benchmark
    uv run python -m benchmarks.reflect_abstain.reflect_abstain_benchmark --mode verdict --split heldout
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from hindsight_api.config import _get_raw_config
from hindsight_api.engine.aiohttp_session import close_loop_sessions
from hindsight_api.engine.cross_encoder import TypeSafeCrossEncoder
from hindsight_api.engine.decision_model import DecisionModel, DecisionQuestion
from hindsight_api.engine.reflect.agent import _ABSTAIN_AT, _ANSWER_AT
from rich.console import Console
from rich.table import Table

DATASET = Path(__file__).parent / "datasets" / "watcher_moments.jsonl"

Route = Literal["silent", "check", "writer"]

# The verdict question. Each option names one concrete reason to speak or stay quiet: worded
# vaguely ("a memory is related, it depends on details"), the middle option was never chosen, and
# Jev split every unsure moment between "speak" and "none". The quiet reasons are the ones the
# tuned split showed going wrong: a rule the agent was already told, a memory about how to run
# tests (Jev ignored the query's own "running tests is never a conflict"), and two memories that
# disagree (one describing the old code as the decision).
VERDICT_SUFFIX = " Compare the memories above with my situation and ALREADY TOLD ME. Which fits best?"
VERDICT_OPTIONS: dict[str, str] = {
    "breaks": "breaks: my latest step goes against a rule, even if I was warned about it before I made the step",
    "untold": "untold: a memory gives a rule for the code I am changing that ALREADY TOLD ME does not have",
    "disagree": "disagree: a memory says something different from ALREADY TOLD ME about this code",
    "unsure": "unsure: a memory gives a rule for this code, and I cannot tell yet whether my step follows it",
    "follows": "follows: my step does what ALREADY TOLD ME says",
    "told": "told: ALREADY TOLD ME already reacted to this same step of mine and pointed out its problem",
    "tests": "tests: my step only runs, installs, finds or checks tests, or the only memory that fits is about that",
    "unrelated": "unrelated: no memory is about the code I am working on",
}
VERDICT_SPEAK = frozenset({"breaks", "untold"})
# Measured on jev-latest over three runs of the tuned split: every speak moment summed >= 0.68,
# every silent one <= 0.65. The bands are a model's, not the question's: another model needs its own.
VERDICT_WRITE_AT = 0.66
VERDICT_CHECK_AT = 0.5


@dataclass
class Moment:
    id: str
    split: str
    task: str
    trigger: str
    query: str
    context: str
    memories: list[str]
    label: Literal["speak", "silent"]
    label_source: str
    note: str = ""

    def state(self) -> str:
        """The state the verdict question reads: the same layout ``pick_answering`` builds."""
        listing = "\n".join(f"[{i + 1}] {m}" for i, m in enumerate(self.memories))
        return f"{self.context}\n\nMEMORY:\n{listing}"


@dataclass
class Outcome:
    id: str
    split: str
    label: str
    route: Route
    #: How strongly the decision model leans to speaking: 1 - p(none) for ``pick``, the summed
    #: probability of the speak options for ``verdict``. Higher should mean "speak".
    speak_score: float
    seconds: float
    probabilities: dict[str, float] = field(default_factory=dict)

    @property
    def wrong(self) -> str | None:
        if self.label == "speak" and self.route == "silent":
            return "missed note"
        if self.label == "silent" and self.route == "writer":
            return "false note"
        return None


def load(split: str) -> list[Moment]:
    rows = [Moment(**json.loads(line)) for line in DATASET.read_text().splitlines() if line.strip()]
    return [m for m in rows if split == "all" or m.split == split]


def route_pick(none: float) -> Route:
    # The same comparisons as reflect's _pick_round and the settled-answer check after it.
    if none >= _ABSTAIN_AT:
        return "silent"
    return "writer" if none < _ANSWER_AT else "check"


def route_verdict(speak: float, write_at: float, check_at: float) -> Route:
    if speak >= write_at:
        return "writer"
    return "check" if speak >= check_at else "silent"


def separation(outcomes: list[Outcome]) -> float | None:
    """How often a speak moment scores above a silent one (AUC): 1.0 means some cut-off separates
    them perfectly, 0.5 means the score says nothing."""
    pos = [o.speak_score for o in outcomes if o.label == "speak"]
    neg = [o.speak_score for o in outcomes if o.label == "silent"]
    if not pos or not neg:
        return None
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


async def run(args: argparse.Namespace) -> list[Outcome]:
    setup = _get_raw_config().decision_model_config("reflect")
    if setup is None:
        raise SystemExit("No decision model is set up: set HINDSIGHT_API_DECISION_MODEL_PROVIDER and its key.")
    picker = TypeSafeCrossEncoder(
        api_key=setup.api_key,
        model=setup.model,
        base_url=setup.base_url,
        timeout=setup.timeout,
        max_concurrent=setup.max_concurrent,
    )
    asker = DecisionModel.from_config(setup)
    limit = asyncio.Semaphore(args.concurrency)

    async def one(m: Moment) -> Outcome:
        async with limit:
            t0 = time.perf_counter()
            if args.mode == "pick":
                pick = await picker.pick_answering(m.query, m.context, m.memories)
                probs = {"none": pick.none, **{f"memory {i + 1}": p for i, p in enumerate(pick.candidates)}}
                speak, route = 1 - pick.none, route_pick(pick.none)
            else:
                question = DecisionQuestion(
                    type="choice", instructions=m.query + VERDICT_SUFFIX, criteria=dict(VERDICT_OPTIONS)
                )
                answer = (
                    await asker.ask(m.state(), {"verdict": question}, scope="decision_abstain_benchmark")
                ).answers["verdict"]
                probs = answer.probabilities
                speak = sum(p for k, p in probs.items() if k in VERDICT_SPEAK)
                route = route_verdict(speak, args.write_at, args.check_at)
            return Outcome(m.id, m.split, m.label, route, speak, time.perf_counter() - t0, probs)

    moments = load(args.split)
    print(f"{len(moments)} moments ({args.split}), mode {args.mode}, {setup.provider} {setup.model} @ {setup.base_url}")
    try:
        return list(await asyncio.gather(*(one(m) for m in moments)))
    finally:
        await close_loop_sessions()


def report(outcomes: list[Outcome]) -> None:
    table = Table(title="Reflect abstain routing")
    for col in ("split", "moments", "speak", "writer", "check", "silent", "missed", "false", "AUC", "median s"):
        table.add_column(col, justify="right")
    for split in sorted({o.split for o in outcomes}) + ["all"]:
        rows = [o for o in outcomes if split == "all" or o.split == split]
        routes = Counter(o.route for o in rows)
        wrong = Counter(o.wrong for o in rows if o.wrong)
        auc = separation(rows)
        table.add_row(
            split,
            str(len(rows)),
            str(sum(o.label == "speak" for o in rows)),
            str(routes["writer"]),
            str(routes["check"]),
            str(routes["silent"]),
            str(wrong["missed note"]),
            str(wrong["false note"]),
            "-" if auc is None else f"{auc:.2f}",
            f"{statistics.median(o.seconds for o in rows):.2f}",
        )
    console = Console()
    console.print(table)
    for o in outcomes:
        if o.wrong:
            console.print(f"  {o.wrong}: {o.id} (speak score {o.speak_score:.2f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["pick", "verdict"], default="pick")
    ap.add_argument("--split", choices=["tuned", "heldout", "all"], default="all")
    ap.add_argument("--write-at", type=float, default=VERDICT_WRITE_AT, help="verdict: route to the writer at or above")
    ap.add_argument(
        "--check-at", type=float, default=VERDICT_CHECK_AT, help="verdict: route to the LLM check at or above"
    )
    # pick_answering does not retry a 429, so a small endpoint (one replica) needs a low value.
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", type=Path, help="write one JSON line per moment (route, scores, probabilities)")
    args = ap.parse_args()
    outcomes = asyncio.run(run(args))
    report(outcomes)
    if args.out:
        args.out.write_text("".join(json.dumps(asdict(o)) + "\n" for o in outcomes))


if __name__ == "__main__":
    main()
