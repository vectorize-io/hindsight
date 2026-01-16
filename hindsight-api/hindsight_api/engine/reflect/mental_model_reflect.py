"""
Diff-Based Mental Model Reflect Agent.

This module implements a multi-phase agentic loop for generating and updating
mental model observations with evidence-grounded quotes and computed trends.

Phases:
0. UPDATE EXISTING: Search for new evidence for existing observations
1. SEED: Generate NEW candidate observations (skipping already-tracked patterns)
2. EVIDENCE HUNT: For each new candidate, search for supporting/contradicting evidence
3. VALIDATE: Validate new candidates and extract quotes
4. COMPARE: Merge updated existing + new validated observations
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from pydantic import BaseModel, Field, field_validator

from .observations import (
    CandidateObservation,
    CandidateWithEvidence,
    Observation,
    ObservationEvidence,
    verify_evidence_quotes,
)
from .prompts import (
    COMPARE_PHASE_SYSTEM_PROMPT,
    SEED_PHASE_SYSTEM_PROMPT,
    UPDATE_EXISTING_SYSTEM_PROMPT,
    VALIDATE_PHASE_SYSTEM_PROMPT,
    build_compare_phase_prompt,
    build_seed_phase_prompt,
    build_update_existing_prompt,
    build_validate_phase_prompt,
)

if TYPE_CHECKING:
    from ..llm_wrapper import LLMProvider

logger = logging.getLogger(__name__)


# =============================================================================
# Typed Models for Refresh State Tracking
# =============================================================================


class DispositionTraits(BaseModel):
    """Disposition traits for a memory bank."""

    skepticism: int = Field(default=3, ge=1, le=5)
    literalism: int = Field(default=3, ge=1, le=5)
    empathy: int = Field(default=3, ge=1, le=5)


class BankProfile(BaseModel):
    """Bank profile with mission and disposition."""

    bank_id: str = Field(default="")
    name: str = Field(default="")
    mission: str | None = Field(default=None)
    disposition: DispositionTraits = Field(default_factory=DispositionTraits)

    @field_validator("disposition", mode="before")
    @classmethod
    def parse_disposition(cls, v: DispositionTraits | dict | None) -> DispositionTraits:
        """Parse disposition from various formats."""
        if v is None:
            return DispositionTraits()
        if isinstance(v, DispositionTraits):
            return v
        if isinstance(v, dict):
            return DispositionTraits.model_validate(v)
        # Handle Pydantic v1 style models
        if hasattr(v, "model_dump"):
            return DispositionTraits.model_validate(v.model_dump())
        return DispositionTraits()


class DirectiveObservation(BaseModel):
    """A single observation in a directive mental model."""

    title: str = Field(default="")
    content: str = Field(default="")
    text: str = Field(default="")  # Legacy field

    @property
    def effective_content(self) -> str:
        """Get content, falling back to text for legacy data."""
        return self.content or self.text


class DirectiveMentalModel(BaseModel):
    """A directive mental model with its observations."""

    id: str = Field(default="")
    name: str = Field(default="")
    observations: list[DirectiveObservation] = Field(default_factory=list)

    @field_validator("observations", mode="before")
    @classmethod
    def parse_observations(cls, v: list | None) -> list[DirectiveObservation]:
        """Parse observations from various formats."""
        if not v:
            return []
        result = []
        for obs in v:
            if isinstance(obs, DirectiveObservation):
                result.append(obs)
            elif isinstance(obs, dict):
                result.append(DirectiveObservation.model_validate(obs))
        return result


class RefreshState(BaseModel):
    """State snapshot at the time of last refresh.

    Used to determine if a refresh is needed by comparing current state
    against this stored snapshot.
    """

    last_refresh_at: str = Field(description="ISO timestamp of last refresh")
    memories_count: int = Field(default=0, description="Total memory count at refresh time")
    mission_hash: str = Field(default="", description="Hash of bank mission text")
    disposition_hash: str = Field(default="", description="Hash of bank disposition values")
    directives_hash: str = Field(default="", description="Hash of all directive observations")


class RefreshCheckResult(BaseModel):
    """Result of checking if a mental model needs refresh."""

    needs_refresh: bool = Field(description="Whether a refresh is needed")
    reasons: list[str] = Field(default_factory=list, description="Reasons why refresh is needed")
    current_state: RefreshState | None = Field(default=None, description="Current state for comparison")


def _hash_string(s: str) -> str:
    """Create a short hash of a string."""
    if not s:
        return ""
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _hash_mission(mission: str | None) -> str:
    """Hash the bank mission text."""
    return _hash_string(mission or "")


def _hash_disposition(disposition: DispositionTraits) -> str:
    """Hash the bank disposition values."""
    # Create deterministic string from disposition values
    return _hash_string(
        f"skepticism:{disposition.skepticism}|literalism:{disposition.literalism}|empathy:{disposition.empathy}"
    )


def _hash_directives(directives: list[DirectiveMentalModel]) -> str:
    """Hash all directive observations."""
    if not directives:
        return ""

    # Create deterministic string from all directive observations
    parts = []
    for directive in sorted(directives, key=lambda d: d.id or d.name):
        directive_id = directive.id or directive.name
        for obs in directive.observations:
            parts.append(f"{directive_id}:{obs.title}:{obs.effective_content}")

    return _hash_string("|".join(parts))


def compute_refresh_state(
    memories_count: int,
    bank_profile: BankProfile,
    directives: list[DirectiveMentalModel],
) -> RefreshState:
    """Compute the current refresh state from inputs.

    Args:
        memories_count: Total number of memories in the bank
        bank_profile: Bank profile with mission and disposition
        directives: List of directive mental models
    """
    return RefreshState(
        last_refresh_at=datetime.now(timezone.utc).isoformat(),
        memories_count=memories_count,
        mission_hash=_hash_mission(bank_profile.mission),
        disposition_hash=_hash_disposition(bank_profile.disposition),
        directives_hash=_hash_directives(directives),
    )


def check_needs_refresh(
    stored_state: dict | RefreshState | None,
    current_memories_count: int,
    bank_profile: BankProfile,
    directives: list[DirectiveMentalModel],
) -> RefreshCheckResult:
    """Check if a mental model needs refresh by comparing states.

    Args:
        stored_state: Previously stored refresh state (or None if never refreshed)
        current_memories_count: Current total memory count
        bank_profile: Current bank profile
        directives: Current directive mental models

    Returns:
        RefreshCheckResult with needs_refresh flag and reasons
    """
    # Compute current state
    current_state = compute_refresh_state(current_memories_count, bank_profile, directives)

    # Never refreshed = definitely needs refresh
    if stored_state is None:
        return RefreshCheckResult(
            needs_refresh=True,
            reasons=["never_refreshed"],
            current_state=current_state,
        )

    # Parse stored state if dict
    if isinstance(stored_state, dict):
        try:
            stored = RefreshState.model_validate(stored_state)
        except Exception:
            return RefreshCheckResult(
                needs_refresh=True,
                reasons=["invalid_stored_state"],
                current_state=current_state,
            )
    else:
        stored = stored_state

    # Compare states
    reasons: list[str] = []

    if current_memories_count > stored.memories_count:
        reasons.append("new_memories")

    if current_state.mission_hash != stored.mission_hash:
        reasons.append("mission_changed")

    if current_state.disposition_hash != stored.disposition_hash:
        reasons.append("disposition_changed")

    if current_state.directives_hash != stored.directives_hash:
        reasons.append("directives_changed")

    return RefreshCheckResult(
        needs_refresh=len(reasons) > 0,
        reasons=reasons,
        current_state=current_state,
    )


class PhaseTokenUsage(BaseModel):
    """Token usage from a phase's LLM calls."""

    input_tokens: int = Field(default=0)
    output_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)

    def __add__(self, other: "PhaseTokenUsage") -> "PhaseTokenUsage":
        """Allow aggregating token usage."""
        return PhaseTokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class SeedPhaseResult(BaseModel):
    """Result from the seed phase."""

    candidates: list[CandidateObservation] = Field(default_factory=list)
    token_usage: PhaseTokenUsage = Field(default_factory=PhaseTokenUsage)


class UpdateExistingResult(BaseModel):
    """Result from the update existing phase."""

    updated_observations: list[dict] = Field(default_factory=list)
    contradicted_titles: list[str] = Field(default_factory=list)
    token_usage: PhaseTokenUsage = Field(default_factory=PhaseTokenUsage)


class ValidatePhaseResult(BaseModel):
    """Result from the validate phase."""

    verified_observations: list[dict] = Field(default_factory=list)
    token_usage: PhaseTokenUsage = Field(default_factory=PhaseTokenUsage)


class ComparePhaseResult(BaseModel):
    """Result from the compare phase."""

    observations: list[Observation] = Field(default_factory=list)
    changes: dict = Field(default_factory=dict)
    token_usage: PhaseTokenUsage = Field(default_factory=PhaseTokenUsage)


class MentalModelReflectResult(BaseModel):
    """Result from the mental model reflect process."""

    observations: list[Observation] = Field(default_factory=list, description="Final validated observations")
    version: int = Field(default=1, description="New version number for this mental model")
    changes: dict = Field(default_factory=dict, description="Summary of changes made")
    phases_completed: list[str] = Field(default_factory=list, description="Which phases were completed")
    duration_ms: int = Field(default=0, description="Total duration in milliseconds")
    memories_analyzed: int = Field(default=0, description="Number of memories analyzed")
    candidates_generated: int = Field(default=0, description="Number of candidate observations generated")
    candidates_validated: int = Field(default=0, description="Number of candidates that passed validation")
    # Token usage tracking
    input_tokens: int = Field(default=0, description="Total input tokens used across all LLM calls")
    output_tokens: int = Field(default=0, description="Total output tokens used across all LLM calls")
    total_tokens: int = Field(default=0, description="Total tokens used (input + output)")


class SeedPhaseOutput(BaseModel):
    """Output from the seed phase."""

    candidates: list[CandidateObservation] = Field(default_factory=list)


class ValidatePhaseOutput(BaseModel):
    """Output from the validate phase."""

    observations: list[dict] = Field(default_factory=list)
    discarded: list[dict] = Field(default_factory=list)
    merged: list[dict] = Field(default_factory=list)


class ComparePhaseOutput(BaseModel):
    """Output from the compare phase."""

    observations: list[dict] = Field(default_factory=list)
    changes: dict = Field(default_factory=dict)


class NewEvidenceItem(BaseModel):
    """A new evidence item from the update existing phase."""

    memory_id: str = Field(description="ID of the memory this quote is from")
    quote: str = Field(description="Exact quote from the memory")
    relevance: str = Field(default="", description="Why this quote supports the observation")
    timestamp: str = Field(default="", description="When the memory was created (ISO format)")


class UpdatedObservation(BaseModel):
    """Output for a single updated observation."""

    title: str = Field(default="")
    content: str = Field(default="")
    existing_evidence_count: int = Field(default=0)
    new_evidence: list[NewEvidenceItem] = Field(default_factory=list)
    has_contradiction: bool = Field(default=False)
    contradiction_note: str | None = Field(default=None)


class UpdateExistingPhaseOutput(BaseModel):
    """Output from the update existing phase."""

    updated_observations: list[UpdatedObservation] = Field(default_factory=list)


class ObservationWithEvidence(BaseModel):
    """An existing observation with newly found evidence for the update phase."""

    observation: dict = Field(description="The original observation dict")
    supporting_memories: list[dict] = Field(default_factory=list, description="Newly found supporting memories")
    contradicting_memories: list[dict] = Field(default_factory=list, description="Newly found contradicting memories")


async def run_mental_model_reflect(
    llm_config: "LLMProvider",
    bank_id: str,
    mental_model_id: str,
    mental_model_name: str,
    existing_observations: list[dict],
    current_version: int,
    get_diverse_memories_fn: Callable[[], Awaitable[list[dict]]],
    recall_fn: Callable[[str, int], Awaitable[dict[str, Any]]],
    topic: str | None = None,
    max_candidates: int = 15,
) -> MentalModelReflectResult:
    """
    Execute the diff-based mental model reflect loop.

    This is a 5-phase process that efficiently updates existing observations
    and discovers new patterns without redundant work:

    Phase 0: UPDATE EXISTING - Search for new evidence for existing observations
    Phase 1: SEED - Generate NEW candidate observations (skipping already-tracked patterns)
    Phase 2: EVIDENCE HUNT - Search for evidence for new candidates
    Phase 3: VALIDATE - Validate new candidates and extract quotes
    Phase 4: COMPARE - Merge updated existing + new validated observations

    Args:
        llm_config: LLM provider for agent calls
        bank_id: Bank identifier
        mental_model_id: ID of the mental model being updated
        mental_model_name: Name of the mental model (for context)
        existing_observations: Current observations in the mental model
        current_version: Current version number of the mental model
        get_diverse_memories_fn: Async function to get diverse memory sample
        recall_fn: Async function for semantic search (query, max_tokens) -> memories
        topic: Optional topic focus for the mental model
        max_candidates: Maximum number of candidate observations to generate

    Returns:
        MentalModelReflectResult with final observations and metadata
    """
    reflect_id = f"mm-{bank_id[:8]}-{int(time.time() * 1000) % 100000}"
    start_time = time.time()
    phases_completed: list[str] = []
    updated_existing: list[dict] = []
    contradicted_observations: list[str] = []
    total_usage = PhaseTokenUsage()  # Aggregate token usage across all phases

    logger.info(
        f"[MM-REFLECT {reflect_id}] Starting diff-based reflect for mental model '{mental_model_name}' "
        f"({len(existing_observations)} existing observations)"
    )

    # ==========================================================================
    # PHASE 0: UPDATE EXISTING (if there are existing observations)
    # ==========================================================================
    if existing_observations:
        logger.info(f"[MM-REFLECT {reflect_id}] Phase 0: UPDATE EXISTING - Searching for new evidence")
        phase0_start = time.time()

        update_result = await _run_update_existing_phase(
            llm_config=llm_config,
            existing_observations=existing_observations,
            recall_fn=recall_fn,
            reflect_id=reflect_id,
        )
        updated_existing = update_result.updated_observations
        contradicted_observations = update_result.contradicted_titles
        total_usage = total_usage + update_result.token_usage

        phase0_duration = int((time.time() - phase0_start) * 1000)
        logger.info(
            f"[MM-REFLECT {reflect_id}] Phase 0 complete: "
            f"{len(updated_existing)} observations updated, "
            f"{len(contradicted_observations)} flagged for removal ({phase0_duration}ms)"
        )
        phases_completed.append("update_existing")

    # ==========================================================================
    # PHASE 1: SEED (with existing observations context)
    # ==========================================================================
    logger.info(f"[MM-REFLECT {reflect_id}] Phase 1: SEED - Getting diverse memories")
    phase1_start = time.time()

    # Get diverse memory sample
    seed_memories = await get_diverse_memories_fn()
    if not seed_memories:
        logger.warning(f"[MM-REFLECT {reflect_id}] No memories found for seeding")
        # Return updated existing observations if we have them
        if updated_existing:
            return MentalModelReflectResult(
                observations=[_dict_to_observation(obs) for obs in updated_existing],
                version=current_version + 1,
                changes={
                    "note": "Updated existing observations, no new patterns found",
                    "updated": len(updated_existing),
                },
                phases_completed=phases_completed + ["seed_empty"],
                duration_ms=int((time.time() - start_time) * 1000),
            )
        return MentalModelReflectResult(
            observations=[_dict_to_observation(obs) for obs in existing_observations],
            version=current_version + 1,  # Always increment version when refresh runs
            changes={"note": "No memories available for analysis"},
            phases_completed=["seed_empty"],
            duration_ms=int((time.time() - start_time) * 1000),
        )

    # Generate NEW candidate observations (pass existing to avoid rediscovering them)
    seed_result = await _run_seed_phase(
        llm_config=llm_config,
        memories=seed_memories,
        topic=topic or mental_model_name,
        max_candidates=max_candidates,
        reflect_id=reflect_id,
        existing_observations=existing_observations,  # Pass existing to skip them
    )
    candidates = seed_result.candidates
    total_usage = total_usage + seed_result.token_usage

    phase1_duration = int((time.time() - phase1_start) * 1000)
    logger.info(
        f"[MM-REFLECT {reflect_id}] Phase 1 complete: {len(candidates)} NEW candidates from {len(seed_memories)} memories ({phase1_duration}ms)"
    )
    phases_completed.append("seed")

    # If no new candidates but we updated existing, return those
    if not candidates and updated_existing:
        logger.info(f"[MM-REFLECT {reflect_id}] No new patterns found, returning updated existing observations")
        return MentalModelReflectResult(
            observations=[_dict_to_observation(obs) for obs in updated_existing],
            version=current_version + 1,
            changes={
                "note": "Updated existing observations with new evidence, no new patterns discovered",
                "updated": len(updated_existing),
                "contradicted": contradicted_observations,
            },
            phases_completed=phases_completed,
            duration_ms=int((time.time() - start_time) * 1000),
            memories_analyzed=len(seed_memories),
            input_tokens=total_usage.input_tokens,
            output_tokens=total_usage.output_tokens,
            total_tokens=total_usage.total_tokens,
        )

    # If no candidates and no existing, return empty
    if not candidates:
        logger.warning(f"[MM-REFLECT {reflect_id}] No candidates generated in seed phase")
        return MentalModelReflectResult(
            observations=[_dict_to_observation(obs) for obs in existing_observations],
            version=current_version + 1,  # Always increment version when refresh runs
            changes={"note": "No candidate observations could be generated"},
            phases_completed=phases_completed,
            duration_ms=int((time.time() - start_time) * 1000),
            memories_analyzed=len(seed_memories),
            input_tokens=total_usage.input_tokens,
            output_tokens=total_usage.output_tokens,
            total_tokens=total_usage.total_tokens,
        )

    # ==========================================================================
    # PHASE 2: EVIDENCE HUNT (for new candidates only)
    # ==========================================================================
    logger.info(f"[MM-REFLECT {reflect_id}] Phase 2: EVIDENCE HUNT - Searching for evidence")
    phase2_start = time.time()

    candidates_with_evidence = await _run_evidence_hunt_phase(
        candidates=candidates,
        recall_fn=recall_fn,
        reflect_id=reflect_id,
    )

    phase2_duration = int((time.time() - phase2_start) * 1000)
    logger.info(
        f"[MM-REFLECT {reflect_id}] Phase 2 complete: Evidence gathered for {len(candidates_with_evidence)} candidates ({phase2_duration}ms)"
    )
    phases_completed.append("evidence_hunt")

    # ==========================================================================
    # PHASE 3: VALIDATE & REFINE (for new candidates only)
    # ==========================================================================
    logger.info(f"[MM-REFLECT {reflect_id}] Phase 3: VALIDATE - Validating candidates")
    phase3_start = time.time()

    validate_result = await _run_validate_phase(
        llm_config=llm_config,
        candidates_with_evidence=candidates_with_evidence,
        reflect_id=reflect_id,
    )
    validated_observations = validate_result.verified_observations
    total_usage = total_usage + validate_result.token_usage

    phase3_duration = int((time.time() - phase3_start) * 1000)
    logger.info(
        f"[MM-REFLECT {reflect_id}] Phase 3 complete: {len(validated_observations)} observations validated ({phase3_duration}ms)"
    )
    phases_completed.append("validate")

    # ==========================================================================
    # PHASE 4: COMPARE & MERGE
    # ==========================================================================
    logger.info(f"[MM-REFLECT {reflect_id}] Phase 4: COMPARE - Merging updated existing + new observations")
    phase4_start = time.time()

    # Use updated existing (with new evidence) instead of original existing
    observations_for_compare = updated_existing if updated_existing else existing_observations

    compare_result = await _run_compare_phase(
        llm_config=llm_config,
        existing_observations=observations_for_compare,
        new_observations=validated_observations,
        reflect_id=reflect_id,
    )
    final_observations = compare_result.observations
    changes = compare_result.changes
    total_usage = total_usage + compare_result.token_usage

    # Add contradiction info to changes
    if contradicted_observations:
        changes["contradicted"] = contradicted_observations

    phase4_duration = int((time.time() - phase4_start) * 1000)
    logger.info(
        f"[MM-REFLECT {reflect_id}] Phase 4 complete: {len(final_observations)} final observations ({phase4_duration}ms)"
    )
    phases_completed.append("compare")

    # ==========================================================================
    # FINALIZE
    # ==========================================================================
    total_duration = int((time.time() - start_time) * 1000)
    new_version = current_version + 1

    logger.info(
        f"[MM-REFLECT {reflect_id}] Complete: "
        f"v{current_version}→v{new_version}, "
        f"{len(final_observations)} observations, "
        f"{total_duration}ms total"
    )

    return MentalModelReflectResult(
        observations=final_observations,
        version=new_version,
        changes=changes,
        phases_completed=phases_completed,
        duration_ms=total_duration,
        memories_analyzed=len(seed_memories),
        candidates_generated=len(candidates),
        candidates_validated=len(validated_observations),
        input_tokens=total_usage.input_tokens,
        output_tokens=total_usage.output_tokens,
        total_tokens=total_usage.total_tokens,
    )


async def _run_seed_phase(
    llm_config: "LLMProvider",
    memories: list[dict],
    topic: str,
    max_candidates: int,
    reflect_id: str,
    existing_observations: list[dict] | None = None,
) -> SeedPhaseResult:
    """Phase 1: Generate NEW candidate observations from diverse memories.

    If existing_observations are provided, the LLM will be instructed to skip
    patterns that are already tracked, focusing only on genuinely new discoveries.
    """
    prompt = build_seed_phase_prompt(memories, topic, existing_observations)

    try:
        response, token_usage = await llm_config.call(
            messages=[
                {"role": "system", "content": SEED_PHASE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format=SeedPhaseOutput,
            scope="mm_reflect_seed",
            return_usage=True,
        )

        usage = PhaseTokenUsage(
            input_tokens=token_usage.input_tokens if token_usage else 0,
            output_tokens=token_usage.output_tokens if token_usage else 0,
            total_tokens=token_usage.total_tokens if token_usage else 0,
        )

        # Parse response
        if hasattr(response, "candidates"):
            candidates = response.candidates[:max_candidates]
        elif isinstance(response, dict) and "candidates" in response:
            candidates = [
                CandidateObservation(
                    content=c.get("content", ""),
                    seed_memory_ids=c.get("seed_memory_ids", []),
                )
                for c in response["candidates"][:max_candidates]
            ]
        else:
            # Try to parse as JSON
            try:
                data = json.loads(str(response))
                candidates = [
                    CandidateObservation(
                        content=c.get("content", ""),
                        seed_memory_ids=c.get("seed_memory_ids", []),
                    )
                    for c in data.get("candidates", [])[:max_candidates]
                ]
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"[MM-REFLECT {reflect_id}] Failed to parse seed phase response")
                candidates = []

        return SeedPhaseResult(candidates=candidates, token_usage=usage)

    except Exception as e:
        logger.error(f"[MM-REFLECT {reflect_id}] Seed phase failed: {e}")
        return SeedPhaseResult()


def _parse_update_existing_response(response: Any, reflect_id: str) -> UpdateExistingPhaseOutput:
    """Parse LLM response into UpdateExistingPhaseOutput.

    Handles multiple response formats: Pydantic model, dict, or JSON string.
    """
    # Already a Pydantic model
    if isinstance(response, UpdateExistingPhaseOutput):
        return response

    # Dict response - validate and convert
    if isinstance(response, dict):
        try:
            return UpdateExistingPhaseOutput.model_validate(response)
        except Exception as e:
            logger.warning(f"[MM-REFLECT {reflect_id}] Failed to validate dict response: {e}")
            return UpdateExistingPhaseOutput()

    # String response - try to parse as JSON
    try:
        data = json.loads(str(response))
        return UpdateExistingPhaseOutput.model_validate(data)
    except (json.JSONDecodeError, TypeError, Exception) as e:
        logger.warning(f"[MM-REFLECT {reflect_id}] Failed to parse update existing response: {e}")
        return UpdateExistingPhaseOutput()


async def _run_update_existing_phase(
    llm_config: "LLMProvider",
    existing_observations: list[dict],
    recall_fn: Callable[[str, int], Awaitable[dict[str, Any]]],
    reflect_id: str,
) -> UpdateExistingResult:
    """Phase 0: Search for new evidence for existing observations.

    For each existing observation:
    1. Search for new supporting evidence
    2. Search for contradicting evidence
    3. Extract new quotes and add to the observation
    4. Flag observations with strong contradictions
    """
    if not existing_observations:
        return UpdateExistingResult()

    # Search for evidence for all existing observations in parallel
    async def search_evidence_for_observation(obs: dict) -> ObservationWithEvidence:
        content = obs.get("content", "")
        existing_evidence = obs.get("evidence", [])
        existing_memory_ids = {e.get("memory_id") for e in existing_evidence if isinstance(e, dict)}

        # Search for supporting and contradicting evidence
        supporting_query = f"evidence supporting: {content}"
        contradicting_query = f"evidence against: {content}"

        supporting_result, contradicting_result = await asyncio.gather(
            recall_fn(supporting_query, 2048),
            recall_fn(contradicting_query, 2048),
            return_exceptions=True,
        )

        supporting_memories: list[dict] = []
        contradicting_memories: list[dict] = []

        if isinstance(supporting_result, dict) and "memories" in supporting_result:
            # Filter out memories we already have evidence from
            supporting_memories = [
                m
                for m in supporting_result["memories"]
                if isinstance(m, dict) and m.get("id") not in existing_memory_ids
            ]
        if isinstance(contradicting_result, dict) and "memories" in contradicting_result:
            contradicting_memories = [m for m in contradicting_result["memories"] if isinstance(m, dict)]

        return ObservationWithEvidence(
            observation=obs,
            supporting_memories=supporting_memories,
            contradicting_memories=contradicting_memories,
        )

    # Run all searches in parallel
    tasks = [search_evidence_for_observation(obs) for obs in existing_observations]
    search_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions and collect valid results
    valid_results: list[ObservationWithEvidence] = [r for r in search_results if isinstance(r, ObservationWithEvidence)]

    # Log any errors
    errors = [r for r in search_results if isinstance(r, Exception)]
    if errors:
        logger.warning(f"[MM-REFLECT {reflect_id}] {len(errors)} evidence search errors: {errors[:3]}")

    # If no new evidence found for any observation, return originals unchanged
    has_new_evidence = any(result.supporting_memories or result.contradicting_memories for result in valid_results)

    if not has_new_evidence:
        logger.info(f"[MM-REFLECT {reflect_id}] No new evidence found for existing observations")
        return UpdateExistingResult(updated_observations=existing_observations)

    # Build prompt data for LLM
    prompt_data = [
        {
            "observation": result.observation,
            "supporting_memories": result.supporting_memories,
            "contradicting_memories": result.contradicting_memories,
        }
        for result in valid_results
    ]

    # Call LLM to extract quotes from new evidence
    try:
        prompt = build_update_existing_prompt(prompt_data)

        response, token_usage = await llm_config.call(
            messages=[
                {"role": "system", "content": UPDATE_EXISTING_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format=UpdateExistingPhaseOutput,
            scope="mm_reflect_update_existing",
            return_usage=True,
        )

        usage = PhaseTokenUsage(
            input_tokens=token_usage.input_tokens if token_usage else 0,
            output_tokens=token_usage.output_tokens if token_usage else 0,
            total_tokens=token_usage.total_tokens if token_usage else 0,
        )

        # Parse response into typed model
        parsed_response = _parse_update_existing_response(response, reflect_id)

        if not parsed_response.updated_observations:
            logger.info(f"[MM-REFLECT {reflect_id}] No updated observations in LLM response")
            return UpdateExistingResult(updated_observations=existing_observations, token_usage=usage)

        # Build memory content map for quote verification
        memory_content_map: dict[str, str] = {}
        for result in valid_results:
            for mem in result.supporting_memories + result.contradicting_memories:
                mem_id = mem.get("id", "")
                mem_content = mem.get("content", mem.get("text", ""))
                if mem_id and mem_content:
                    memory_content_map[mem_id] = mem_content

        # Process updated observations
        updated_observations: list[dict] = []
        contradicted_titles: list[str] = []
        total_new_evidence = 0

        for i, updated in enumerate(parsed_response.updated_observations):
            if i >= len(existing_observations):
                break

            original_obs = existing_observations[i]

            # Check for contradiction
            if updated.has_contradiction:
                title = original_obs.get("title", f"Observation {i + 1}")
                contradicted_titles.append(title)
                logger.info(f"[MM-REFLECT {reflect_id}] Observation '{title}' flagged for contradiction")

            # Verify and add new evidence
            existing_evidence = original_obs.get("evidence", [])
            verified_new_evidence: list[dict] = []

            for ev in updated.new_evidence:
                memory_content = memory_content_map.get(ev.memory_id, "")

                # Verify quote exists in memory
                if (
                    ev.quote
                    and memory_content
                    and (ev.quote in memory_content or _fuzzy_quote_match(ev.quote, memory_content))
                ):
                    verified_new_evidence.append(
                        {
                            "memory_id": ev.memory_id,
                            "quote": ev.quote,
                            "relevance": ev.relevance,
                            "timestamp": ev.timestamp,
                        }
                    )

            total_new_evidence += len(verified_new_evidence)

            # Merge existing and new evidence
            merged_evidence = existing_evidence + verified_new_evidence

            # Create updated observation
            updated_obs = {
                **original_obs,
                "evidence": merged_evidence,
            }
            updated_observations.append(updated_obs)

        # For any observations not in the response, keep them unchanged
        for i in range(len(parsed_response.updated_observations), len(existing_observations)):
            updated_observations.append(existing_observations[i])

        logger.info(
            f"[MM-REFLECT {reflect_id}] Updated {len(updated_observations)} observations, "
            f"added {total_new_evidence} new evidence items"
        )

        return UpdateExistingResult(
            updated_observations=updated_observations,
            contradicted_titles=contradicted_titles,
            token_usage=usage,
        )

    except Exception as e:
        logger.error(f"[MM-REFLECT {reflect_id}] Update existing phase failed: {e}", exc_info=True)
        return UpdateExistingResult(updated_observations=existing_observations)


async def _run_evidence_hunt_phase(
    candidates: list[CandidateObservation],
    recall_fn: Callable[[str, int], Awaitable[dict[str, Any]]],
    reflect_id: str,
) -> list[CandidateWithEvidence]:
    """Phase 2: For each candidate, search for supporting and contradicting evidence."""
    results: list[CandidateWithEvidence] = []

    # Run evidence searches in parallel for all candidates
    async def search_evidence_for_candidate(candidate: CandidateObservation) -> CandidateWithEvidence:
        # Search for supporting evidence
        supporting_query = f"evidence supporting: {candidate.content}"
        contradicting_query = f"evidence against: {candidate.content}"

        supporting_result, contradicting_result = await asyncio.gather(
            recall_fn(supporting_query, 2048),
            recall_fn(contradicting_query, 2048),
            return_exceptions=True,
        )

        supporting_memories = []
        contradicting_memories = []

        if isinstance(supporting_result, dict) and "memories" in supporting_result:
            supporting_memories = supporting_result["memories"]
        if isinstance(contradicting_result, dict) and "memories" in contradicting_result:
            contradicting_memories = contradicting_result["memories"]

        return CandidateWithEvidence(
            candidate=candidate,
            supporting_memories=supporting_memories,
            contradicting_memories=contradicting_memories,
        )

    # Run all searches in parallel
    tasks = [search_evidence_for_candidate(c) for c in candidates]
    gather_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions
    valid_results: list[CandidateWithEvidence] = [r for r in gather_results if isinstance(r, CandidateWithEvidence)]

    logger.info(f"[MM-REFLECT {reflect_id}] Evidence hunt: {len(valid_results)}/{len(candidates)} candidates processed")
    return valid_results


async def _run_validate_phase(
    llm_config: "LLMProvider",
    candidates_with_evidence: list[CandidateWithEvidence],
    reflect_id: str,
) -> ValidatePhaseResult:
    """Phase 3: Validate candidates and extract exact quotes."""
    # Convert to dict format for prompt
    candidates_data = [
        {
            "candidate": {
                "content": c.candidate.content,
            },
            "supporting_memories": c.supporting_memories,
            "contradicting_memories": c.contradicting_memories,
        }
        for c in candidates_with_evidence
    ]

    prompt = build_validate_phase_prompt(candidates_data)

    try:
        response, token_usage = await llm_config.call(
            messages=[
                {"role": "system", "content": VALIDATE_PHASE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format=ValidatePhaseOutput,
            scope="mm_reflect_validate",
            return_usage=True,
        )

        usage = PhaseTokenUsage(
            input_tokens=token_usage.input_tokens if token_usage else 0,
            output_tokens=token_usage.output_tokens if token_usage else 0,
            total_tokens=token_usage.total_tokens if token_usage else 0,
        )

        # Parse response
        if hasattr(response, "observations"):
            observations = response.observations
        elif isinstance(response, dict) and "observations" in response:
            observations = response["observations"]
        else:
            # Try to parse as JSON
            try:
                data = json.loads(str(response))
                observations = data.get("observations", [])
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"[MM-REFLECT {reflect_id}] Failed to parse validate phase response")
                observations = []

        # Build memory content map for quote verification
        memory_content_map: dict[str, str] = {}
        for cwe in candidates_with_evidence:
            for mem in cwe.supporting_memories + cwe.contradicting_memories:
                mem_id = mem.get("id", "")
                content = mem.get("content", mem.get("text", ""))
                if mem_id and content:
                    memory_content_map[mem_id] = content

        # Verify quotes in observations
        verified_observations = []
        for obs in observations:
            evidence = obs.get("evidence", [])
            verified_evidence = []

            for ev in evidence:
                mem_id = ev.get("memory_id", "")
                quote = ev.get("quote", "")
                memory_content = memory_content_map.get(mem_id, "")

                # Check if quote exists in memory (allow partial match for flexibility)
                if quote and memory_content and (quote in memory_content or _fuzzy_quote_match(quote, memory_content)):
                    verified_evidence.append(ev)
                else:
                    logger.debug(f"[MM-REFLECT {reflect_id}] Quote verification failed for memory {mem_id}")

            if verified_evidence:
                obs["evidence"] = verified_evidence
                verified_observations.append(obs)
            else:
                logger.debug(
                    f"[MM-REFLECT {reflect_id}] Observation discarded - no verified evidence: {obs.get('content', '')[:50]}"
                )

        return ValidatePhaseResult(verified_observations=verified_observations, token_usage=usage)

    except Exception as e:
        logger.error(f"[MM-REFLECT {reflect_id}] Validate phase failed: {e}")
        return ValidatePhaseResult()


def _fuzzy_quote_match(quote: str, content: str, threshold: float = 0.8) -> bool:
    """Check if a quote roughly matches content (handles minor LLM variations)."""
    # Normalize both strings
    quote_words = set(quote.lower().split())
    content_words = set(content.lower().split())

    if not quote_words:
        return False

    # Check word overlap
    overlap = len(quote_words & content_words)
    similarity = overlap / len(quote_words)

    return similarity >= threshold


async def _run_compare_phase(
    llm_config: "LLMProvider",
    existing_observations: list[dict],
    new_observations: list[dict],
    reflect_id: str,
) -> ComparePhaseResult:
    """Phase 4: Merge new observations with existing mental model."""
    # If no existing observations, just convert new ones
    if not existing_observations:
        final_obs = [_dict_to_observation(obs) for obs in new_observations]
        changes = {
            "added": [obs.get("content", "") for obs in new_observations],
            "kept": [],
            "updated": [],
            "removed": [],
            "merged": [],
        }
        return ComparePhaseResult(observations=final_obs, changes=changes)

    # If no new observations, keep existing
    if not new_observations:
        final_obs = [_dict_to_observation(obs) for obs in existing_observations]
        changes = {
            "added": [],
            "kept": [obs.get("content", obs.get("text", "")) for obs in existing_observations],
            "updated": [],
            "removed": [],
            "merged": [],
        }
        return ComparePhaseResult(observations=final_obs, changes=changes)

    prompt = build_compare_phase_prompt(existing_observations, new_observations)

    try:
        response, token_usage = await llm_config.call(
            messages=[
                {"role": "system", "content": COMPARE_PHASE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format=ComparePhaseOutput,
            scope="mm_reflect_compare",
            return_usage=True,
        )

        usage = PhaseTokenUsage(
            input_tokens=token_usage.input_tokens if token_usage else 0,
            output_tokens=token_usage.output_tokens if token_usage else 0,
            total_tokens=token_usage.total_tokens if token_usage else 0,
        )

        # Parse response
        if hasattr(response, "observations"):
            observations_data = response.observations
            changes = response.changes if hasattr(response, "changes") else {}
        elif isinstance(response, dict):
            observations_data = response.get("observations", [])
            changes = response.get("changes", {})
        else:
            # Try to parse as JSON
            try:
                data = json.loads(str(response))
                observations_data = data.get("observations", [])
                changes = data.get("changes", {})
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"[MM-REFLECT {reflect_id}] Failed to parse compare phase response")
                # Fallback: just use new observations
                observations_data = new_observations
                changes = {"note": "Compare phase parsing failed, using new observations"}

        # Convert to Observation objects
        final_observations = [_dict_to_observation(obs) for obs in observations_data]
        return ComparePhaseResult(observations=final_observations, changes=changes, token_usage=usage)

    except Exception as e:
        logger.error(f"[MM-REFLECT {reflect_id}] Compare phase failed: {e}")
        # Fallback: merge by keeping all
        all_obs = existing_observations + new_observations
        final_obs = [_dict_to_observation(obs) for obs in all_obs]
        changes = {"note": f"Compare phase failed: {e}, keeping all observations"}
        return ComparePhaseResult(observations=final_obs, changes=changes)


def _dict_to_observation(data: dict) -> Observation:
    """Convert a dict to an Observation model."""
    # Handle both new format (title, content, evidence) and legacy format (title, text, memory_ids)
    title = data.get("title", "")
    content = data.get("content", "")

    if not content:
        # Legacy format: use text as content
        text = data.get("text", "")
        content = text

    if not title:
        # Generate title from content (first ~50 chars)
        title = content[:50].strip() + ("..." if len(content) > 50 else "")

    # Parse evidence
    evidence: list[ObservationEvidence] = []
    evidence_data = data.get("evidence", [])

    if evidence_data:
        for ev in evidence_data:
            try:
                timestamp = ev.get("timestamp")
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                elif timestamp is None:
                    timestamp = datetime.now(timezone.utc)

                evidence.append(
                    ObservationEvidence(
                        memory_id=ev.get("memory_id", ""),
                        quote=ev.get("quote", ""),
                        relevance=ev.get("relevance", ""),
                        timestamp=timestamp,
                    )
                )
            except Exception:
                pass
    else:
        # Legacy format: memory_ids without quotes
        memory_ids = data.get("memory_ids", []) or data.get("fact_ids", [])
        for mid in memory_ids:
            evidence.append(
                ObservationEvidence(
                    memory_id=mid,
                    quote="[migrated - quote not available]",
                    relevance="[migrated]",
                    timestamp=datetime.now(timezone.utc),
                )
            )

    # Parse created_at
    created_at = data.get("created_at")
    if isinstance(created_at, str):
        try:
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except ValueError:
            created_at = datetime.now(timezone.utc)
    elif created_at is None:
        created_at = datetime.now(timezone.utc)

    return Observation(
        title=title,
        content=content,
        evidence=evidence,
        created_at=created_at,
    )
