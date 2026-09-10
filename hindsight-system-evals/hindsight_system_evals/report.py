"""The JSON a run publishes to the continuous performance monitor.

The field names are the dashboard's contract (``system-evals.html`` and
``publish-system-evals-results.sh`` read them), so they live in one typed place
rather than as dict literals spread across the test module and conftest.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PageRecord(BaseModel):
    question_id: str
    category: str
    correct: bool
    hit_trap: bool
    sizes: list[int]
    bank_id: str
    page_id: str
    reason: str


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
    #: The headline pair. The correct rate can dip on an incomplete page; a trap
    #: is a stored falsehood, so it is reported on its own and never averaged in.
    correct_rate: float | None
    trap_count: int
    items: list[PageRecord]

    def to_json(self) -> str:
        return self.model_dump_json(by_alias=True, indent=2)


#: Filled by the evals as they finish; written once at session end. A failed
#: assertion still records its outcome first, so a red run publishes what it saw.
RECORDED: list[PageRecord] = []
