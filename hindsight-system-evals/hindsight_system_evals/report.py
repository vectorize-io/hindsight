"""The JSON a run publishes to the continuous performance monitor.

The field names are the dashboard's contract (``system-evals.html`` and
``publish-system-evals-results.sh`` read them), so they live in one typed place
rather than as dict literals spread across the test modules and conftest.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from typing import Literal, Protocol

import pytest
from pydantic import BaseModel, Field

EvalKind = Literal["knowledge_page", "delta_update", "reflect", "retain", "source_priority", "consolidation", "recall"]
#: The operation an eval grades: one test file each, ``evals/<operation>/test_<operation>.py``.
Operation = Literal["retain", "recall", "consolidation", "reflect", "refresh_pages"]

#: ``kind`` predates operations and is the dashboard's contract, so it stays; this maps it.
_OPERATION: dict[str, Operation] = {
    "retain": "retain",
    "recall": "recall",
    "consolidation": "consolidation",
    "reflect": "reflect",
    "source_priority": "reflect",
    "knowledge_page": "refresh_pages",
    "delta_update": "refresh_pages",
}


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

    @property
    def operation(self) -> Operation:
        return _OPERATION[self.kind]


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


class KnowledgeIngest(BaseModel):
    """Writing the corpus into a knowledge bank, upload to every document ready or failed."""

    documents: int
    passages: int
    failed: int
    upload_s: float
    wall_s: float
    documents_per_s: float
    passages_per_s: float


class KnowledgeLatency(BaseModel):
    searches: int
    #: Searches in flight at once. Above 1 the client numbers include queueing.
    concurrency: int
    client_p50_ms: float
    client_p95_ms: float


class AgentLatency(BaseModel):
    """An agent suite's time per question: the whole answer (every LLM and tool call), not one
    search, so it is kept apart from ``KnowledgeLatency``."""

    questions: int
    #: Questions in flight at once. Above 1 the numbers include queueing.
    concurrency: int
    answer_p50_ms: float
    answer_p95_ms: float


class KnowledgeResult(BaseModel):
    """One knowledge-bank retrieval suite: scores, not a pass/fail.

    Kept apart from ``EvalRecord`` on purpose: an nDCG is a number to compare across
    embedders, and folding it into a correct/total count would throw that away.
    """

    category: str
    case: str
    #: ``<provider>:<model>`` as the run names it (see ``knowledge.embedder_name``).
    embedder: str
    corpus: int
    queries: int
    #: Percentages (x100), as the write-ups report them.
    metrics: dict[str, float]
    ingest: KnowledgeIngest | None = None
    #: Retrieval suites: one search's time. Agent suites report ``agent_latency`` instead.
    latency: KnowledgeLatency | None = None
    agent_latency: AgentLatency | None = None


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
    #: Operation -> category -> counts: the table a model comparison reads (``eval-table``).
    by_category: dict[str, dict[str, KindSummary]] = Field(default_factory=dict)
    items: list[EvalRecord]
    knowledge: list[KnowledgeResult] = Field(default_factory=list)

    def to_json(self) -> str:
        return self.model_dump_json(by_alias=True, indent=2)


def summarise(records: list[EvalRecord]) -> KindSummary:
    return KindSummary(
        total=len(records),
        correct=sum(1 for r in records if r.correct),
        trap_count=sum(1 for r in records if r.hit_trap),
    )


def by_category(records: list[EvalRecord]) -> dict[str, dict[str, KindSummary]]:
    table: dict[str, dict[str, list[EvalRecord]]] = {}
    for record in records:
        table.setdefault(record.operation, {}).setdefault(record.category, []).append(record)
    return {op: {cat: summarise(rs) for cat, rs in sorted(cats.items())} for op, cats in sorted(table.items())}


class Categorised(Protocol):
    @property
    def id(self) -> str: ...
    @property
    def category(self) -> str: ...


def case_params(cases: Iterable[Categorised], minimum: Collection[str]) -> list[object]:
    """The pytest params of an operation's cases, each id'd ``<category>/<case id>``.

    The category in the id is what ``-k`` selects on (``-k "temporal/"``) and what a
    per-category table groups by; a case outside ``minimum`` is marked ``full``.
    """
    return [
        pytest.param(case, id=f"{case.category}/{case.id}", marks=() if case.id in minimum else pytest.mark.full)
        for case in cases
    ]


#: Filled by the evals as they finish; written once at session end. A failed
#: assertion still records its outcome first, so a red run publishes what it saw.
RECORDED: list[EvalRecord] = []
#: The knowledge-bank suites' scores, written beside ``items`` at session end.
KNOWLEDGE: list[KnowledgeResult] = []
