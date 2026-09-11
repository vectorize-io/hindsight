"""Ask reflect a corpus question over a fully seeded bank, and keep what it did.

The knowledge-page suite measures what a page converges to across waves. This
one measures the one-shot answer: the whole corpus is in the bank, reflect gets
the question, and the judge reads what it wrote.

Retrieval is a necessary condition, not a sufficient one — reflect can retrieve
every gold fact and still answer from the wrong one — so the ANSWER is what gets
graded. The tool trace is kept only to say which half failed, because the two
have opposite fixes: evidence that never arrived is a query or ranking problem,
evidence that arrived and was misused is a prompt or model problem.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from hindsight_client import Hindsight

from hindsight_system_evals.corpus import HardFact, HardQuestion
from hindsight_system_evals.pages import SettleFn, prepare_bank

#: ``low`` is what most callers send, and the budget the forced prelude and its
#: short-circuit are tuned for. Override to measure another one.
REFLECT_BUDGET = os.getenv("HINDSIGHT_EVAL_REFLECT_BUDGET", "low")


@dataclass
class AnswerOutcome:
    """One reflect answer, graded, with enough of its trace to attribute a failure."""

    question_id: str
    category: str
    bank_id: str
    answer: str = ""
    queries: list[str] = field(default_factory=list)
    gold_retrieved: int = 0
    gold_total: int = 0
    correct: bool = False
    correct_reason: str = ""
    hit_trap: bool = False
    trap_reason: str = ""

    @property
    def blame(self) -> str:
        if self.correct and not self.hit_trap:
            return "ok"
        if self.gold_total and self.gold_retrieved < self.gold_total:
            return f"retrieval ({self.gold_retrieved}/{self.gold_total} gold facts reached the model)"
        return "reasoning (every gold fact reached the model)"


async def seed_bank(client: Hindsight, bank_id: str, facts: list[HardFact], settle: SettleFn) -> None:
    """Store the whole corpus as written, with no model calls — see ``prepare_bank``."""
    await prepare_bank(client, bank_id)
    await client.aretain_batch(bank_id=bank_id, items=[{"content": fact.text} for fact in facts])
    await settle(bank_id)


async def ask(client: Hindsight, bank_id: str, question: HardQuestion, facts: list[HardFact]) -> AnswerOutcome:
    """Reflect once and record which gold facts its tools actually returned."""
    outcome = AnswerOutcome(question_id=question.id, category=question.category, bank_id=bank_id)
    response = await client.areflect(
        bank_id=bank_id, query=question.question, budget=REFLECT_BUDGET, include_tool_calls=True
    )
    outcome.answer = (response.text or "").strip()

    # Matched by text, not id: ``chunks`` retain stores each fact verbatim, and
    # the server's ids are its own. Walks the tool OUTPUTS rather than
    # ``based_on``, which is what the model declared it used — downstream of the
    # very thing being diagnosed.
    gold_ids = set(question.gold)
    gold_texts = {fact.text for fact in facts if fact.id in gold_ids}
    outcome.gold_total = len(gold_texts)
    seen: set[str] = set()
    for call in (response.trace.tool_calls if response.trace else None) or []:
        if query := (call.input or {}).get("query"):
            outcome.queries.append(f"{call.tool}({query})")
        payload = json.dumps(call.output or {}, ensure_ascii=False, default=str)
        seen.update(text for text in gold_texts if text in payload)
    outcome.gold_retrieved = len(seen)
    return outcome
