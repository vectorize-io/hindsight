"""Client for a decision model: a System One endpoint (TypeSafe's ``/v1/systemone``).

A decision model answers typed questions (pick one of these, how likely is this, where on
this scale) about a state, in well under a second. Operations hand it the in-between
judgments an LLM call would otherwise make, and keep the LLM for writing. Every operation
reaches it through this one client, built from ``HindsightConfig.decision_model_config``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .._cross_loop import CrossLoopSemaphore
from ..config import DecisionModelConfig
from .aiohttp_session import LoopLocalSession, per_phase_timeout, raise_for_status


class DecisionAnswer(BaseModel):
    """One answer. Which field is set depends on the question type: a choice fills
    ``choice`` and ``probabilities``, a noul fills ``noul``, a score fills ``score``."""

    choice: str | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)
    noul: float | None = None
    score: float | None = None
    confidence: float | None = None


class DecisionUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class DecisionResponse(BaseModel):
    #: Keyed by the question ids the caller sent.
    answers: dict[str, DecisionAnswer]
    usage: DecisionUsage = Field(default_factory=DecisionUsage)


class DecisionModel:
    SYSTEMONE_PATH = "/v1/systemone"

    def __init__(self, api_key: str, model: str, base_url: str, timeout: float, max_concurrent: int):
        self.model = model
        self.base_url = base_url.rstrip("/")
        # CrossLoopSemaphore, not asyncio.Semaphore: one client is built at startup and
        # reached from every loop in the process (worker threads run their own via
        # asyncio.run), and an asyncio.Semaphore binds to the first loop that waits on it.
        self._semaphore = CrossLoopSemaphore(max_concurrent)
        self._session = LoopLocalSession(
            timeout=per_phase_timeout(timeout),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )

    @classmethod
    def from_config(cls, config: DecisionModelConfig) -> DecisionModel:
        return cls(
            api_key=config.api_key,
            model=config.model,
            base_url=config.base_url,
            timeout=config.timeout,
            max_concurrent=config.max_concurrent,
        )

    async def ask(self, state: Any, questions: dict[str, Any]) -> DecisionResponse:
        """Ask ``questions`` (keyed by caller-chosen ids) about ``state`` in one request."""
        body = {"model": self.model, "state": state, "questions": questions}
        async with self._semaphore:
            async with self._session.get().post(f"{self.base_url}{self.SYSTEMONE_PATH}", json=body) as response:
                await raise_for_status(response)
                return DecisionResponse.model_validate(await response.json(content_type=None))
