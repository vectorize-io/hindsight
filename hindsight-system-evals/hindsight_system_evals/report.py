"""The JSON a run publishes to the continuous performance monitor.

The field names are the dashboard's contract (``system-evals.html`` and
``publish-system-evals-results.sh`` read them), so they live in one typed place
rather than as dict literals spread across the test modules and conftest.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EvalKind = Literal["knowledge_page", "reflect"]


class EvalRecord(BaseModel):
    kind: EvalKind
    question_id: str
    category: str
    correct: bool
    hit_trap: bool
    bank_id: str
    reason: str
    page_id: str = ""
    #: The page's size after each wave; empty for a one-shot reflect answer.
    sizes: list[int] = Field(default_factory=list)


class KindSummary(BaseModel):
    total: int
    correct: int
    trap_count: int


class ModelRef(BaseModel):
    provider: str | None = None
    model: str


class ModelConfig(BaseModel):
    hindsight: ModelRef
    judge: ModelRef


class RunReport(BaseModel):
    timestamp: str
    suite: str
    mode: str
    # ``model_config`` is reserved on pydantic models; the dashboard reads that
    # key, so the field is renamed only on the way out.
    llm_config: ModelConfig = Field(serialization_alias="model_config")
    total: int
    correct: int
    #: The headline pair. The correct rate can dip on an incomplete answer; a trap
    #: is a stored falsehood, so it is reported on its own and never averaged in.
    correct_rate: float | None
    trap_count: int
    #: The same counts per suite, keyed by ``EvalKind``: a page and a one-shot
    #: answer fail for different reasons and should not hide inside one rate.
    by_kind: dict[str, KindSummary]
    items: list[EvalRecord]

    def to_json(self) -> str:
        return self.model_dump_json(by_alias=True, indent=2)


def summarise(records: list[EvalRecord]) -> KindSummary:
    return KindSummary(
        total=len(records),
        correct=sum(1 for r in records if r.correct),
        trap_count=sum(1 for r in records if r.hit_trap),
    )


#: Filled by the evals as they finish; written once at session end. A failed
#: assertion still records its outcome first, so a red run publishes what it saw.
RECORDED: list[EvalRecord] = []
