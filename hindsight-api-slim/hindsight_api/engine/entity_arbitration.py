"""Gray-band LLM arbitration for entity resolution (B2 tier 3).

The deterministic scorer accepts candidates above 0.6 and creates new entities
below it. Two kinds of cases deserve a second opinion before defaulting to
"create": blended scores just under the threshold (the gray band), and
low-entropy names whose name signal the entropy gate suppressed but which have
plausible candidates with context support.

One small-LLM call arbitrates ALL deferred cases of a retain batch at once
(contiguous indices — the same anti-hallucination pattern as relation
extraction and supersession arbitration). The model may only pick one of the
listed candidates or none (-1): it can never invent entities or trigger merges
of existing rows. Out-of-range answers are dropped defensively, falling back
to the deterministic outcome (create).

Disabled by default (``entity_llm_arbitration``): it adds LLM cost to every
retain batch that produces gray-band mentions. Resolution runs inside the
batch_retain worker task, so the latency is worker time, not user-facing.
"""

import logging
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GrayBandCandidate:
    """One existing entity offered to the arbiter."""

    entity_id: str
    canonical_name: str
    cooccurring_names: tuple[str, ...]


@dataclass(frozen=True)
class GrayBandCase:
    """One unresolved mention deferred to the arbiter."""

    mention: str
    nearby_names: tuple[str, ...]
    candidates: tuple[GrayBandCandidate, ...]


class GrayBandChoice(BaseModel):
    case_index: int = Field(description="Index of the CASE being decided (0-based).")
    candidate_index: int = Field(
        description="Index of the chosen candidate within that case's CANDIDATES list, or -1 for none."
    )


class GrayBandVerdict(BaseModel):
    """Arbiter output: at most one choice per case; omitted cases mean 'none'."""

    choices: list[GrayBandChoice] = Field(default_factory=list)


_ARBITRATION_PROMPT = """You disambiguate entity mentions from a memory system.

For each numbered CASE you get a mention, the entities it appeared alongside
(NEARBY), and numbered CANDIDATE entities already in the store with the names
they typically co-occur with.

For each case decide: is the mention the SAME real-world entity as one of the
candidates? Answer with candidate_index, or -1 if none clearly matches.

Rules:
- Pick a candidate only when the names are compatible AND the context
  (nearby vs co-occurring names) supports identity. A shared social circle
  with an INCOMPATIBLE name (e.g. "Bob" vs "Rob") is NOT a match.
- Nicknames, partial names, and spelling variants of the same person/thing
  ARE compatible ("Bob" / "Bob Smith", "Ann" / "Ann Lee").
- When unsure, answer -1: a wrong merge corrupts both entities' histories;
  a missed merge only leaves a duplicate.
"""


def render_cases(cases: list[GrayBandCase]) -> str:
    """Render the deferred cases as the arbiter's user message."""
    blocks: list[str] = []
    for i, case in enumerate(cases):
        nearby = ", ".join(case.nearby_names) if case.nearby_names else "(none)"
        lines = [f"CASE [{i}] mention: {case.mention!r}  NEARBY: {nearby}"]
        for j, cand in enumerate(case.candidates):
            cooc = ", ".join(cand.cooccurring_names) if cand.cooccurring_names else "(none)"
            lines.append(f"  CANDIDATE [{j}] {cand.canonical_name!r}  co-occurs with: {cooc}")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def apply_verdict(verdict: GrayBandVerdict, cases: list[GrayBandCase]) -> dict[int, str]:
    """Map case index -> chosen entity_id, dropping malformed answers.

    Defensive parse (same posture as relation extraction): out-of-range case or
    candidate indices and repeated cases are dropped with a warning — the case
    falls back to the deterministic outcome (create a new entity).
    """
    chosen: dict[int, str] = {}
    for choice in verdict.choices:
        if not 0 <= choice.case_index < len(cases):
            logger.warning("Gray-band arbitration: dropping out-of-range case index %d", choice.case_index)
            continue
        if choice.case_index in chosen:
            logger.warning("Gray-band arbitration: dropping duplicate answer for case %d", choice.case_index)
            continue
        if choice.candidate_index == -1:
            continue
        candidates = cases[choice.case_index].candidates
        if not 0 <= choice.candidate_index < len(candidates):
            logger.warning(
                "Gray-band arbitration: dropping out-of-range candidate index %d for case %d",
                choice.candidate_index,
                choice.case_index,
            )
            continue
        chosen[choice.case_index] = candidates[choice.candidate_index].entity_id
    return chosen


async def arbitrate_gray_band(llm_config: Any, cases: list[GrayBandCase]) -> dict[int, str]:
    """One LLM call for all deferred cases. Returns {case_index: entity_id}."""
    if not cases:
        return {}
    raw, _usage = await llm_config.call(
        messages=[
            {"role": "system", "content": _ARBITRATION_PROMPT},
            {"role": "user", "content": render_cases(cases)},
        ],
        response_format=GrayBandVerdict,
        scope="entity_gray_band_arbitrate",
        temperature=0.0,
        skip_validation=True,
        return_usage=True,
    )
    verdict = GrayBandVerdict.model_validate_json(raw) if isinstance(raw, str) else GrayBandVerdict()
    return apply_verdict(verdict, cases)
