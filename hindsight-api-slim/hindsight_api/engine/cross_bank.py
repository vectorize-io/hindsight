"""
Cross-bank orchestration for Hindsight memory system.

This module provides the CrossBankOrchestrator which coordinates queries
across multiple memory banks, handling bank selection, budget allocation,
and result fusion.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hindsight_api.api.cross_bank_models import (
        BankInfo,
        CrossBankFact,
        CrossBankRecallResult,
        CrossBankReflectResult,
    )
    from hindsight_api.config_resolver import ConfigResolver
    from hindsight_api.engine.interface import MemoryEngineInterface
    from hindsight_api.engine.llm_wrapper import LLMProvider
    from hindsight_api.engine.memory_engine import Budget
    from hindsight_api.extensions.operation_validator import (
        CrossBankRecallContext,
        CrossBankReflectContext,
    )
    from hindsight_api.extensions.operation_validator import (
        CrossBankRecallResult as CrossBankRecallResultContext,
    )
    from hindsight_api.extensions.operation_validator import (
        CrossBankReflectResult as CrossBankReflectResultContext,
    )
    from hindsight_api.models import RequestContext

logger = logging.getLogger(__name__)


# =============================================================================
# Budget Allocation
# =============================================================================


class BudgetStrategy(str, Enum):
    """Strategies for allocating budget across banks."""

    EQUAL = "equal"  # Split budget equally across all banks
    PROPORTIONAL = "proportional"  # Allocate based on bank memory counts
    QUERY_RELEVANT = "query_relevant"  # Allocate based on estimated query relevance


@dataclass
class BudgetAllocation:
    """Result of budget allocation across banks."""

    per_bank: dict[str, int]  # bank_id -> token budget
    total_used: int
    strategy: BudgetStrategy
    synthesis_reserve: int = 0  # Tokens reserved for final synthesis (multi-step)


class BudgetAllocator:
    """
    Allocates query budget across multiple banks.

    Supports different allocation strategies to optimize
    cross-bank query performance.
    """

    SYNTHESIS_RESERVE_RATIO = 0.30  # Reserve 30% for synthesis in multi-step

    # Budget token values (matching MemoryEngine)
    BUDGET_VALUES = {
        "low": 100,
        "mid": 300,
        "high": 1000,
    }
    MIN_BUDGET_PER_BANK = 100  # Minimum budget per bank (LOW budget value)

    @staticmethod
    def equal_split(
        budget_value: int,
        bank_ids: list[str],
        ensure_minimum: bool = True,
    ) -> BudgetAllocation:
        """
        Split budget equally across all banks.

        Args:
            budget_value: Total budget tokens to allocate.
            bank_ids: List of bank IDs to allocate budget for.
            ensure_minimum: If True, ensure no bank gets less than MIN_BUDGET_PER_BANK.

        Returns:
            BudgetAllocation with equal per-bank amounts.

        Raises:
            ValueError: If budget is too low to provide minimum to each bank.
        """
        if not bank_ids:
            return BudgetAllocation(per_bank={}, total_used=0, strategy=BudgetStrategy.EQUAL)

        per_bank_raw = int(budget_value / len(bank_ids))

        # Ensure minimum budget per bank — auto-raise to minimum if too low
        if ensure_minimum and per_bank_raw < BudgetAllocator.MIN_BUDGET_PER_BANK:
            per_bank_raw = BudgetAllocator.MIN_BUDGET_PER_BANK

        allocation = {bank_id: per_bank_raw for bank_id in bank_ids}
        total_used = per_bank_raw * len(bank_ids)

        return BudgetAllocation(
            per_bank=allocation,
            total_used=total_used,
            strategy=BudgetStrategy.EQUAL,
        )

    @staticmethod
    def proportional(
        budget_value: int,
        bank_sizes: dict[str, int],
    ) -> BudgetAllocation:
        """
        Allocate budget proportional to bank memory counts.

        Banks with more memories get larger budget allocations.

        Args:
            budget_value: Total budget tokens to allocate.
            bank_sizes: Dict mapping bank_id to memory count.

        Returns:
            BudgetAllocation weighted by bank sizes.
        """
        if not bank_sizes:
            return BudgetAllocation(per_bank={}, total_used=0, strategy=BudgetStrategy.PROPORTIONAL)

        total_memories = sum(bank_sizes.values())
        if total_memories == 0:
            return BudgetAllocator.equal_split(budget_value, list(bank_sizes.keys()))

        allocation = {}
        total_used = 0
        for bank_id, size in bank_sizes.items():
            bank_budget = int(budget_value * (size / total_memories))
            allocation[bank_id] = bank_budget
            total_used += bank_budget

        return BudgetAllocation(
            per_bank=allocation,
            total_used=total_used,
            strategy=BudgetStrategy.PROPORTIONAL,
        )

    @staticmethod
    def query_relevant(
        budget_value: int,
        bank_relevance: dict[str, float],
    ) -> BudgetAllocation:
        """
        Allocate budget based on estimated query relevance per bank.

        Uses a lightweight pre-query (e.g., BM25) to estimate
        which banks are most relevant to the query.

        Args:
            budget_value: Total budget tokens to allocate.
            bank_relevance: Dict mapping bank_id to relevance score (0-1).

        Returns:
            BudgetAllocation weighted by relevance scores.
        """
        if not bank_relevance:
            return BudgetAllocation(per_bank={}, total_used=0, strategy=BudgetStrategy.QUERY_RELEVANT)

        total_relevance = sum(bank_relevance.values())
        if total_relevance == 0:
            return BudgetAllocator.equal_split(budget_value, list(bank_relevance.keys()))

        allocation = {}
        total_used = 0
        for bank_id, relevance in bank_relevance.items():
            bank_budget = int(budget_value * (relevance / total_relevance))
            allocation[bank_id] = bank_budget
            total_used += bank_budget

        return BudgetAllocation(
            per_bank=allocation,
            total_used=total_used,
            strategy=BudgetStrategy.QUERY_RELEVANT,
        )

    @staticmethod
    def allocate_for_multistep(
        total_budget: int,
        bank_ids: list[str],
        n_steps: int,
    ) -> tuple[BudgetAllocation, int]:
        """
        Allocate budget for multi-step reasoning.

        Reserves a portion for final synthesis and splits
        the remainder across steps and banks.

        Args:
            total_budget: Total budget tokens.
            bank_ids: List of bank IDs.
            n_steps: Number of reasoning steps.

        Returns:
            Tuple of (per_step_allocation, synthesis_budget).
        """
        synthesis_budget = int(total_budget * BudgetAllocator.SYNTHESIS_RESERVE_RATIO)
        remaining = total_budget - synthesis_budget

        per_step_per_bank = int(remaining / (n_steps * len(bank_ids))) if bank_ids and n_steps > 0 else 0
        allocation = {bank_id: per_step_per_bank for bank_id in bank_ids}

        return BudgetAllocation(
            per_bank=allocation,
            total_used=per_step_per_bank * len(bank_ids),
            strategy=BudgetStrategy.EQUAL,
            synthesis_reserve=synthesis_budget,
        ), synthesis_budget


# =============================================================================
# Bank Selection
# =============================================================================


class BankSelector:
    """
    Resolves which banks participate in a cross-bank query.

    Supports selection by explicit IDs, tags, or automatic
    discovery of accessible banks.
    """

    def __init__(self, engine: "MemoryEngineInterface"):
        """
        Initialize the bank selector.

        Args:
            engine: Memory engine instance for bank queries.
        """
        self._engine = engine

    async def resolve(
        self,
        bank_ids: list[str] | None = None,
        bank_tags: list[str] | None = None,
        request_context: "RequestContext | None" = None,
    ) -> list["BankInfo"]:
        """
        Resolve the list of banks to query.

        Selection priority:
        1. If bank_ids provided: validate access and return those banks
        2. If bank_tags provided: filter banks by tags
        3. Otherwise: return all accessible banks

        Args:
            bank_ids: Explicit list of bank IDs to query.
            bank_tags: Filter banks by these tags.
            request_context: Request context for authorization.

        Returns:
            List of BankInfo for participating banks.
        """
        # Create default context if not provided
        from hindsight_api.models import RequestContext

        ctx = request_context or RequestContext()

        if bank_ids:
            return await self._validate_and_load(bank_ids, ctx)
        if bank_tags:
            return await self._filter_by_tags(bank_tags, ctx)
        return await self._all_accessible(ctx)

    async def _validate_and_load(
        self,
        bank_ids: list[str],
        request_context: "RequestContext",
    ) -> list["BankInfo"]:
        """Validate access to specified banks and load their info."""
        from hindsight_api.api.cross_bank_models import BankInfo

        # Check if access is restricted by allowed_bank_ids
        allowed = request_context.allowed_bank_ids
        if allowed is not None:
            # Filter to only allowed banks
            invalid = set(bank_ids) - set(allowed)
            if invalid:
                logger.warning(f"Access denied to banks: {invalid}")
            bank_ids = [bid for bid in bank_ids if bid in allowed]

        result = []
        for bank_id in bank_ids:
            try:
                profile = await self._engine.get_bank_profile(bank_id, request_context=request_context)
                result.append(
                    BankInfo(
                        bank_id=bank_id,
                        name=profile.get("name"),
                        disposition=profile.get("disposition", {"skepticism": 3, "literalism": 3, "empathy": 3}),
                        background=profile.get("mission"),
                        tags=[],  # Tags not stored in bank profile currently
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to load bank {bank_id}: {e}")

        return result

    async def _filter_by_tags(
        self,
        tags: list[str],
        request_context: "RequestContext",
    ) -> list["BankInfo"]:
        """Find banks matching any of the specified tags."""
        from hindsight_api.api.cross_bank_models import BankInfo

        # Get all banks and filter by tags
        all_banks = await self._engine.list_banks(request_context=request_context)

        result = []
        for bank_data in all_banks:
            bank_tags = bank_data.get("tags", [])
            # Check if any of the requested tags match
            if any(tag in bank_tags for tag in tags):
                result.append(
                    BankInfo(
                        bank_id=bank_data["bank_id"],
                        name=bank_data.get("name"),
                        disposition=bank_data.get("disposition", {"skepticism": 3, "literalism": 3, "empathy": 3}),
                        background=bank_data.get("mission"),
                        tags=bank_tags,
                    )
                )

        return result

    async def _all_accessible(
        self,
        request_context: "RequestContext",
    ) -> list["BankInfo"]:
        """Get all banks accessible to the request context."""
        from hindsight_api.api.cross_bank_models import BankInfo

        all_banks = await self._engine.list_banks(request_context=request_context)

        # Check if access is restricted
        allowed = request_context.allowed_bank_ids

        result = []
        for bank_data in all_banks:
            bank_id = bank_data["bank_id"]

            # Skip if not in allowed list (when restricted)
            if allowed is not None and bank_id not in allowed:
                continue

            result.append(
                BankInfo(
                    bank_id=bank_id,
                    name=bank_data.get("name"),
                    disposition=bank_data.get("disposition", {"skepticism": 3, "literalism": 3, "empathy": 3}),
                    background=bank_data.get("mission"),
                    tags=bank_data.get("tags", []),
                )
            )

        return result


# =============================================================================
# Disposition Reconciliation
# =============================================================================


@dataclass
class ReconciledDisposition:
    """Result of reconciling dispositions across banks."""

    skepticism: int
    literalism: int
    empathy: int
    reconciliation_method: str
    bank_weights: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, int]:
        """Convert to disposition dict format."""
        return {
            "skepticism": self.skepticism,
            "literalism": self.literalism,
            "empathy": self.empathy,
        }


def reconcile_dispositions(
    bank_results: dict[str, list["CrossBankFact"]],
    bank_dispositions: dict[str, dict[str, int]],
) -> ReconciledDisposition:
    """
    Reconcile conflicting dispositions using relevance-weighted averaging.

    Weight each bank's disposition by how many relevant facts it contributed.

    Args:
        bank_results: Dict mapping bank_id to list of facts returned.
        bank_dispositions: Dict mapping bank_id to disposition traits.

    Returns:
        ReconciledDisposition with weighted average traits.
    """
    if not bank_results or not bank_dispositions:
        return ReconciledDisposition(
            skepticism=3,
            literalism=3,
            empathy=3,
            reconciliation_method="default",
        )

    total_facts = sum(len(facts) for facts in bank_results.values())
    if total_facts == 0:
        return ReconciledDisposition(
            skepticism=3,
            literalism=3,
            empathy=3,
            reconciliation_method="no_facts",
        )

    weighted_skepticism = 0.0
    weighted_literalism = 0.0
    weighted_empathy = 0.0
    bank_weights = {}

    for bank_id, facts in bank_results.items():
        if bank_id not in bank_dispositions:
            continue

        weight = len(facts) / total_facts
        bank_weights[bank_id] = weight

        disposition = bank_dispositions[bank_id]
        weighted_skepticism += disposition.get("skepticism", 3) * weight
        weighted_literalism += disposition.get("literalism", 3) * weight
        weighted_empathy += disposition.get("empathy", 3) * weight

    # Clamp values to valid 1-5 range
    skepticism_clamped = max(1, min(5, round(weighted_skepticism)))
    literalism_clamped = max(1, min(5, round(weighted_literalism)))
    empathy_clamped = max(1, min(5, round(weighted_empathy)))

    return ReconciledDisposition(
        skepticism=skepticism_clamped,
        literalism=literalism_clamped,
        empathy=empathy_clamped,
        reconciliation_method="relevance_weighted",
        bank_weights=bank_weights,
    )


# =============================================================================
# Cross-Bank Orchestrator
# =============================================================================


class CrossBankOrchestrator:
    """
    Orchestrates queries across multiple Hindsight banks.

    Coordinates parallel bank queries, result fusion, and
    multi-step reasoning for complex queries.
    """

    def __init__(
        self,
        engine: "MemoryEngineInterface",
        config_resolver: "ConfigResolver | None" = None,
    ):
        """
        Initialize the cross-bank orchestrator.

        Args:
            engine: Memory engine for executing per-bank queries.
            config_resolver: Optional config resolver for hierarchical config.
        """
        self._engine = engine
        self._config_resolver = config_resolver
        self._bank_selector = BankSelector(engine)

    async def cross_bank_recall(
        self,
        query: str,
        bank_ids: list[str] | None = None,
        bank_tags: list[str] | None = None,
        max_results: int = 20,
        budget: "Budget | None" = None,
        request_context: "RequestContext | None" = None,
    ) -> "CrossBankRecallResult":
        """
        Recall facts from multiple banks with fused ranking.

        Args:
            query: The search query.
            bank_ids: Specific banks to query (None = all accessible).
            bank_tags: Filter banks by tags.
            max_results: Maximum total results to return.
            budget: Budget level for the query.
            request_context: Request context for authorization.

        Returns:
            CrossBankRecallResult with fused and ranked facts.
        """
        from hindsight_api.api.cross_bank_models import CrossBankRecallResult
        from hindsight_api.extensions.operation_validator import (
            CrossBankRecallContext,
        )
        from hindsight_api.extensions.operation_validator import (
            CrossBankRecallResult as CrossBankRecallResultContext,
        )
        from hindsight_api.models import RequestContext

        # Create default context if not provided
        ctx = request_context or RequestContext()

        # Run pre-operation validation hook
        if hasattr(self._engine, '_operation_validator') and self._engine._operation_validator:
            validation_ctx = CrossBankRecallContext(
                bank_ids=bank_ids,
                query=query,
                request_context=ctx,
                budget=budget,
                bank_tags=bank_tags,
                max_results=max_results,
            )
            result = await self._engine._operation_validator.validate_cross_bank_recall(validation_ctx)
            if not result.allowed:
                from hindsight_api.extensions import OperationValidationError

                raise OperationValidationError(
                    result.reason or "Cross-bank recall not allowed",
                    result.status_code,
                )

        # Resolve banks to query
        banks = await self._bank_selector.resolve(
            bank_ids=bank_ids,
            bank_tags=bank_tags,
            request_context=ctx,
        )

        if not banks:
            return CrossBankRecallResult(
                results=[],
                bank_stats={},
                total_results=0,
                fusion_metadata={"strategy": "none", "reason": "no_banks"},
            )

        # Determine budget value
        budget_value = self._get_budget_value(budget)

        # Allocate budget across banks
        bank_id_list = [b.bank_id for b in banks]
        try:
            allocation = BudgetAllocator.equal_split(budget_value, bank_id_list)
        except ValueError as e:
            # Budget too low - use minimum per bank
            logger.warning(f"Budget allocation failed: {e}. Using minimum per bank.")
            allocation = BudgetAllocation(
                per_bank={bid: BudgetAllocator.MIN_BUDGET_PER_BANK for bid in bank_id_list},
                total_used=len(bank_id_list) * BudgetAllocator.MIN_BUDGET_PER_BANK,
                strategy=BudgetStrategy.EQUAL,
            )

        # Execute parallel recall
        bank_results = await self._parallel_recall(query, banks, allocation, ctx)

        # Fuse results with RRF
        fused = await self._fuse_results(bank_results, max_results)

        # Compute bank stats
        bank_stats = {bank_id: len(facts) for bank_id, facts in bank_results.items()}

        # Build result
        result = CrossBankRecallResult(
            results=fused,
            bank_stats=bank_stats,
            total_results=len(fused),
            fusion_metadata={
                "strategy": "reciprocal_rank_fusion",
                "k": 60,
                "banks_queried": len(banks),
                "banks_succeeded": len([b for b in bank_results.values() if b]),
            },
        )

        # Run post-operation completion hook
        if hasattr(self._engine, '_operation_validator') and self._engine._operation_validator:
            completion_result = CrossBankRecallResultContext(
                bank_ids=bank_id_list,
                query=query,
                request_context=ctx,
                budget=budget,
                bank_tags=bank_tags,
                max_results=max_results,
                results_per_bank=bank_stats,
                total_results=len(fused),
                fusion_metadata=result.fusion_metadata,
                success=True,
                error=None,
            )
            await self._engine._operation_validator.on_cross_bank_recall_complete(completion_result)

        return result

    async def cross_bank_reflect(
        self,
        query: str,
        bank_ids: list[str] | None = None,
        bank_tags: list[str] | None = None,
        budget: "Budget | None" = None,
        context: str | None = None,
        include_mental_models: bool = True,
        include_reasoning_chain: bool = False,
        response_schema: dict | None = None,
        request_context: "RequestContext | None" = None,
    ) -> "CrossBankReflectResult":
        """
        Reflect across multiple banks with disposition-aware synthesis.

        This method:
        1. Gathers evidence (facts + mental models) from multiple banks
        2. Fuses facts using Reciprocal Rank Fusion
        3. Reconciles bank dispositions weighted by fact count
        4. Makes a single LLM synthesis call with disposition-aware prompt
        5. Optionally extracts structured output if response_schema provided

        Args:
            query: The question to reflect on.
            bank_ids: Specific banks to query (None = all accessible).
            bank_tags: Filter banks by tags.
            budget: Budget level for the query.
            context: Additional context for the reflection.
            include_mental_models: Whether to consult mental models.
            include_reasoning_chain: Whether to decompose complex queries.
            response_schema: Optional JSON Schema for structured output.
            request_context: Request context for authorization.

        Returns:
            CrossBankReflectResult with synthesized response.
        """
        from hindsight_api.api.cross_bank_models import (
            CrossBankFact,
            CrossBankReflectResult,
            MentalModelReference,
        )
        from hindsight_api.engine.memory_engine import Budget
        from hindsight_api.extensions.operation_validator import (
            CrossBankReflectContext,
        )
        from hindsight_api.extensions.operation_validator import (
            CrossBankReflectResult as CrossBankReflectResultContext,
        )
        from hindsight_api.models import RequestContext

        # Create default context if not provided
        ctx = request_context or RequestContext()

        # Run pre-operation validation hook
        if hasattr(self._engine, '_operation_validator') and self._engine._operation_validator:
            validation_ctx = CrossBankReflectContext(
                bank_ids=bank_ids,
                query=query,
                request_context=ctx,
                budget=budget,
                bank_tags=bank_tags,
                context=context,
                include_mental_models=include_mental_models,
                include_reasoning_chain=include_reasoning_chain,
                response_schema=response_schema,
            )
            result = await self._engine._operation_validator.validate_cross_bank_reflect(validation_ctx)
            if not result.allowed:
                from hindsight_api.extensions import OperationValidationError

                raise OperationValidationError(
                    result.reason or "Cross-bank reflect not allowed",
                    result.status_code,
                )

        # Resolve banks to query
        banks = await self._bank_selector.resolve(
            bank_ids=bank_ids,
            bank_tags=bank_tags,
            request_context=ctx,
        )

        if not banks:
            return CrossBankReflectResult(
                text="No accessible banks found for cross-bank reflection.",
                based_on=[],
                mental_models_used=[],
                bank_dispositions={},
                structured_output=None,
                reasoning_chain=None,
                new_opinions=[],
            )

        bank_id_list = [b.bank_id for b in banks]

        # Calculate budget allocation
        budget_value = self._get_budget_value(budget)
        budget_allocation = BudgetAllocator.equal_split(budget_value, bank_id_list)

        # Gather evidence from all banks
        evidence = await self._gather_evidence(
            query=query,
            banks=banks,
            budget_allocation=budget_allocation,
            include_mental_models=include_mental_models,
            request_context=ctx,
        )

        # Extract fused facts
        fused_facts = evidence["fused_facts"]

        # Extract bank dispositions
        bank_dispositions = {bank.bank_id: bank.disposition for bank in banks}

        # Reconcile dispositions weighted by fact count
        reconciled = reconcile_dispositions(evidence["bank_results"], bank_dispositions)

        # Build synthesis prompt with disposition context
        synthesis_prompt = self._build_synthesis_prompt(
            query=query,
            fused_facts=fused_facts,
            mental_models=evidence.get("mental_models", []),
            reconciled_disposition=reconciled,
            context=context,
        )

        # Make single LLM synthesis call
        llm_provider = self._get_llm_provider()
        if llm_provider is None:
            return CrossBankReflectResult(
                text="LLM provider not configured. Set HINDSIGHT_API_LLM_API_KEY environment variable.",
                based_on=fused_facts,
                mental_models_used=evidence.get("mental_model_refs", []),
                bank_dispositions=bank_dispositions,
                structured_output=None,
                reasoning_chain=None,
                new_opinions=[],
            )

        # Get resolved config for the first bank (or use default)
        resolved_config = None
        if self._config_resolver and banks:
            resolved_config = await self._config_resolver.resolve_full_config(banks[0].bank_id, ctx)

        # Apply config to LLM provider if available
        if resolved_config:
            llm_provider = llm_provider.with_config(resolved_config)

        # Make the synthesis call
        try:
            response, _ = await llm_provider.call(
                messages=[
                    {"role": "system", "content": self._get_cross_bank_system_prompt(reconciled)},
                    {"role": "user", "content": synthesis_prompt},
                ],
                max_completion_tokens=4096,
                scope="cross_bank_reflect",
                return_usage=True,
            )

            synthesized_text = response.strip() if isinstance(response, str) else str(response)

        except Exception as e:
            logger.error(f"Cross-bank reflect LLM call failed: {e}")

            # Run post-operation completion hook (failure case)
            if hasattr(self._engine, '_operation_validator') and self._engine._operation_validator:
                completion_result = CrossBankReflectResultContext(
                    bank_ids=bank_id_list,
                    query=query,
                    request_context=ctx,
                    budget=budget,
                    bank_tags=bank_tags,
                    context=context,
                    include_mental_models=include_mental_models,
                    include_reasoning_chain=include_reasoning_chain,
                    response_schema=response_schema,
                    facts_per_bank={},
                    mental_models_per_bank={},
                    reasoning_steps=0,
                    output_tokens=0,
                    structured_output=None,
                    success=False,
                    error=str(e),
                )
                await self._engine._operation_validator.on_cross_bank_reflect_complete(completion_result)

            return CrossBankReflectResult(
                text=f"Failed to synthesize response: {e}",
                based_on=fused_facts,
                mental_models_used=evidence.get("mental_model_refs", []),
                bank_dispositions=bank_dispositions,
                structured_output=None,
                reasoning_chain=None,
                new_opinions=[],
            )

        # Generate structured output if schema provided
        structured_output = None
        if response_schema and synthesized_text:
            structured_output = await self._generate_structured_output(
                answer=synthesized_text,
                response_schema=response_schema,
                llm_provider=llm_provider,
            )

        # Compute facts per bank for the completion hook
        facts_per_bank = {}
        for bank_id, facts in evidence.get("bank_results", {}).items():
            facts_per_bank[bank_id] = len(facts)

        mental_models_per_bank = {}
        if evidence.get("mental_models"):
            for mm in evidence["mental_models"]:
                bank_id = mm.get("bank_id")
                if bank_id:
                    mental_models_per_bank[bank_id] = mental_models_per_bank.get(bank_id, 0) + 1

        result = CrossBankReflectResult(
            text=synthesized_text,
            based_on=fused_facts,
            mental_models_used=evidence.get("mental_model_refs", []),
            bank_dispositions=bank_dispositions,
            structured_output=structured_output,
            reasoning_chain=None,  # Multi-step reasoning not implemented in this sprint
            new_opinions=[],
        )

        # Run post-operation completion hook (success case)
        if hasattr(self._engine, '_operation_validator') and self._engine._operation_validator:
            completion_result = CrossBankReflectResultContext(
                bank_ids=bank_id_list,
                query=query,
                request_context=ctx,
                budget=budget,
                bank_tags=bank_tags,
                context=context,
                include_mental_models=include_mental_models,
                include_reasoning_chain=include_reasoning_chain,
                response_schema=response_schema,
                facts_per_bank=facts_per_bank,
                mental_models_per_bank=mental_models_per_bank,
                reasoning_steps=0,
                output_tokens=0,
                structured_output=structured_output,
                success=True,
                error=None,
            )
            await self._engine._operation_validator.on_cross_bank_reflect_complete(completion_result)

        return result

    async def _parallel_recall(
        self,
        query: str,
        banks: list["BankInfo"],
        budget_allocation: BudgetAllocation,
        request_context: "RequestContext",
    ) -> dict[str, list["CrossBankFact"]]:
        """Execute recall in parallel across specified banks."""
        from hindsight_api.api.cross_bank_models import CrossBankFact

        async def recall_single_bank(bank: "BankInfo") -> tuple[str, list["CrossBankFact"]]:
            """Recall from a single bank, handling failures gracefully."""
            try:
                bank_budget = budget_allocation.per_bank.get(bank.bank_id, BudgetAllocator.MIN_BUDGET_PER_BANK)
                # Convert token budget to Budget enum-like value
                budget = self._tokens_to_budget(bank_budget)

                result = await self._engine.recall_async(
                    bank_id=bank.bank_id,
                    query=query,
                    budget=budget,
                    request_context=request_context,
                )

                # Convert MemoryFact to CrossBankFact
                facts = []
                for fact in result.results:
                    facts.append(
                        CrossBankFact(
                            id=fact.id,
                            text=fact.text,
                            fact_type=fact.fact_type,
                            bank_id=bank.bank_id,
                            bank_name=bank.name,
                            context=fact.context,
                            confidence=None,  # MemoryFact doesn't have confidence
                            occurred_start=self._parse_datetime(fact.occurred_start),
                            entities=fact.entities or [],
                            tags=fact.tags or [],
                        )
                    )

                return (bank.bank_id, facts)

            except Exception as e:
                logger.error(f"Recall failed for bank {bank.bank_id}: {e}")
                return (bank.bank_id, [])

        # Execute all recalls in parallel with asyncio.gather
        tasks = [recall_single_bank(bank) for bank in banks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results, handling any exceptions
        bank_results = {}
        for i, result in enumerate(results):
            bank_id = banks[i].bank_id
            if isinstance(result, Exception):
                logger.error(f"Recall task failed for bank {bank_id}: {result}")
                bank_results[bank_id] = []
            elif isinstance(result, tuple):
                bank_id_result, facts = result
                bank_results[bank_id_result] = facts
            else:
                logger.warning(f"Unexpected result type for bank {bank_id}: {type(result)}")
                bank_results[bank_id] = []

        return bank_results

    async def _fuse_results(
        self,
        bank_results: dict[str, list["CrossBankFact"]],
        max_results: int,
    ) -> list["CrossBankFact"]:
        """Fuse and rank results from multiple banks using RRF."""
        # RRF constant
        K = 60

        # Build a map of unique facts by entity ID for deduplication
        # Facts are deduplicated by their ID (which should be unique across banks)
        all_facts: dict[str, "CrossBankFact"] = {}
        fact_ranks: dict[str, list[int]] = {}  # fact_id -> list of ranks from each bank

        for _, facts in bank_results.items():
            for rank, fact in enumerate(facts, start=1):
                fact_id = fact.id

                # Store the fact (may be overwritten but that's okay)
                if fact_id not in all_facts:
                    all_facts[fact_id] = fact
                    fact_ranks[fact_id] = []

                # Record the rank from this bank
                fact_ranks[fact_id].append(rank)

        # Compute RRF scores
        rrf_scores: list[tuple[str, float]] = []
        for fact_id, ranks in fact_ranks.items():
            # RRF score = sum over ranks: 1 / (k + rank)
            score = sum(1.0 / (K + rank) for rank in ranks)
            rrf_scores.append((fact_id, score))

        # Sort by RRF score descending
        rrf_scores.sort(key=lambda x: x[1], reverse=True)

        # Take top max_results
        top_fact_ids = [fact_id for fact_id, _ in rrf_scores[:max_results]]

        # Return the fused facts
        return [all_facts[fact_id] for fact_id in top_fact_ids]

    def _get_budget_value(self, budget: "Budget | None") -> int:
        """Convert Budget enum to token value."""
        from hindsight_api.engine.memory_engine import Budget

        if budget is None:
            return BudgetAllocator.BUDGET_VALUES["mid"]

        # Handle Budget enum
        if isinstance(budget, Budget):
            return BudgetAllocator.BUDGET_VALUES.get(budget.value, BudgetAllocator.BUDGET_VALUES["mid"])

        # Handle string budget values
        if isinstance(budget, str):
            budget_str = budget.lower()
            return BudgetAllocator.BUDGET_VALUES.get(budget_str, BudgetAllocator.BUDGET_VALUES["mid"])

        # Budget is already an int
        return int(budget) if isinstance(budget, int) else BudgetAllocator.BUDGET_VALUES["mid"]

    def _tokens_to_budget(self, tokens: int) -> "Budget":
        """Convert token count to Budget enum."""
        from hindsight_api.engine.memory_engine import Budget

        if tokens <= BudgetAllocator.BUDGET_VALUES["low"]:
            return Budget.LOW
        elif tokens <= BudgetAllocator.BUDGET_VALUES["mid"]:
            return Budget.MID
        return Budget.HIGH

    def _parse_datetime(self, dt_str: str | None) -> datetime | None:
        """Parse ISO datetime string to datetime object."""
        if dt_str is None:
            return None

        try:
            # Handle ISO format with or without timezone
            if dt_str.endswith("Z"):
                dt_str = dt_str[:-1] + "+00:00"
            return datetime.fromisoformat(dt_str)
        except (ValueError, TypeError):
            return None

    async def _gather_evidence(
        self,
        query: str,
        banks: list["BankInfo"],
        budget_allocation: BudgetAllocation,
        include_mental_models: bool,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """
        Gather evidence (facts + mental models) from multiple banks.

        This method:
        1. Performs parallel recall across all banks
        2. Optionally searches mental models in each bank
        3. Fuses all facts using Reciprocal Rank Fusion

        Args:
            query: The search query.
            banks: List of BankInfo for banks to query.
            budget_allocation: Budget allocation per bank.
            include_mental_models: Whether to include mental models.
            request_context: Request context for authorization.

        Returns:
            Dict containing:
                - bank_results: Dict mapping bank_id to list of CrossBankFact
                - fused_facts: List of fused and ranked CrossBankFact
                - mental_models: List of mental model dicts (if include_mental_models)
                - mental_model_refs: List of MentalModelReference objects
        """
        from hindsight_api.api.cross_bank_models import CrossBankFact, MentalModelReference

        # Step 1: Parallel recall across all banks
        bank_results = await self._parallel_recall(
            query=query,
            banks=banks,
            budget_allocation=budget_allocation,
            request_context=request_context,
        )

        # Step 2: Fuse results using RRF
        max_facts = 50  # Maximum facts to include in synthesis
        fused_facts = await self._fuse_results(bank_results, max_facts)

        # Step 3: Optionally gather mental models
        mental_models = []
        mental_model_refs = []

        if include_mental_models and banks:
            mental_model_results = await self._search_mental_models_parallel(
                query=query,
                banks=banks,
                max_results_per_bank=3,
                _request_context=request_context,
            )

            for bank_id, models in mental_model_results.items():
                for model in models:
                    mental_models.append(model)
                    mental_model_refs.append(
                        MentalModelReference(
                            id=model["id"],
                            name=model["name"],
                            bank_id=bank_id,
                            excerpt=model["content"][:200] + "..." if len(model["content"]) > 200 else model["content"],
                        )
                    )

        return {
            "bank_results": bank_results,
            "fused_facts": fused_facts,
            "mental_models": mental_models,
            "mental_model_refs": mental_model_refs,
        }

    async def _search_mental_models_parallel(
        self,
        query: str,
        banks: list["BankInfo"],
        max_results_per_bank: int,
        _request_context: "RequestContext",
    ) -> dict[str, list[dict[str, Any]]]:
        """Search mental models in parallel across banks."""
        from hindsight_api.engine.retain import embedding_utils

        # Generate embedding for the query
        try:
            embeddings = await embedding_utils.generate_embeddings_batch(
                self._engine.embeddings,
                [query],
            )
            query_embedding = embeddings[0]
        except Exception as e:
            logger.error(f"Failed to generate embedding for mental model search: {e}")
            return {}

        async def search_single_bank(bank: "BankInfo") -> tuple[str, list[dict[str, Any]]]:
            """Search mental models in a single bank."""
            try:
                pool = await self._engine._get_pool()
                async with pool.acquire() as conn:
                    from hindsight_api.engine.reflect.tools import tool_search_mental_models

                    result = await tool_search_mental_models(
                        conn,
                        bank.bank_id,
                        query,
                        query_embedding,
                        max_results=max_results_per_bank,
                    )

                    return (bank.bank_id, result.get("mental_models", []))
            except Exception as e:
                logger.error(f"Mental model search failed for bank {bank.bank_id}: {e}")
                return (bank.bank_id, [])

        # Execute all searches in parallel
        tasks = [search_single_bank(bank) for bank in banks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        mental_model_results = {}
        for i, result in enumerate(results):
            bank_id = banks[i].bank_id
            if isinstance(result, Exception):
                logger.error(f"Mental model search task failed for bank {bank_id}: {result}")
                mental_model_results[bank_id] = []
            elif isinstance(result, tuple):
                bank_id_result, models = result
                mental_model_results[bank_id_result] = models
            else:
                logger.warning(f"Unexpected result type for bank {bank_id}: {type(result)}")
                mental_model_results[bank_id] = []

        return mental_model_results

    def _get_llm_provider(self) -> "LLMProvider | None":
        """Get the LLM provider from the engine."""
        return getattr(self._engine, "_reflect_llm_config", None)

    def _get_cross_bank_system_prompt(self, reconciled_disposition: ReconciledDisposition) -> str:
        """Build system prompt with disposition traits for cross-bank synthesis."""
        skepticism = reconciled_disposition.skepticism
        literalism = reconciled_disposition.literalism
        empathy = reconciled_disposition.empathy

        # Build disposition-aware guidance
        guidance_parts = []

        # Skepticism guidance (1-5 scale)
        if skepticism >= 4:
            guidance_parts.append(
                "Be critical and skeptical. Question assumptions, look for contradictions, "
                "and explicitly note when evidence is weak or conflicting."
            )
        elif skepticism >= 3:
            guidance_parts.append(
                "Maintain healthy skepticism. Verify claims against available evidence "
                "and note any uncertainties."
            )
        else:
            guidance_parts.append(
                "Be accepting and open. Present information positively and focus on "
                "constructive synthesis rather than criticism."
            )

        # Literalism guidance (1-5 scale)
        if literalism >= 4:
            guidance_parts.append(
                "Be literal and precise. Quote facts directly, avoid metaphors or figurative "
                "language, and stick to what is explicitly stated."
            )
        elif literalism >= 3:
            guidance_parts.append(
                "Balance precision with clarity. Present facts accurately while making "
                "reasonable interpretations."
            )
        else:
            guidance_parts.append(
                "Be interpretive and contextual. Draw connections and inferences, "
                "and explain the broader meaning behind the facts."
            )

        # Empathy guidance (1-5 scale)
        if empathy >= 4:
            guidance_parts.append(
                "Be empathetic and understanding. Consider emotional contexts, "
                "acknowledge feelings, and present information with warmth."
            )
        elif empathy >= 3:
            guidance_parts.append(
                "Balance objectivity with understanding. Acknowledge different "
                "perspectives while presenting factual information."
            )
        else:
            guidance_parts.append(
                "Be objective and factual. Focus on the facts without emotional "
                "framing, presenting information in a straightforward manner."
            )

        guidance = "\n\n".join(guidance_parts)

        return f"""CRITICAL: You MUST ONLY use information from the provided memories and mental models. NEVER make up names, people, events, or entities.

You are a thoughtful assistant synthesizing answers from memories retrieved across multiple memory banks.

Your approach:
- Reason over the retrieved memories to answer the question
- Make reasonable inferences when the exact answer isn't explicitly stated
- Connect related memories to form a complete picture
- Be helpful - if you have related information, use it to give the best possible answer
- ONLY use information from the provided context - no external knowledge or guessing
- Attribute information to the appropriate source bank when relevant

DISPOSITION GUIDANCE (reconciled across banks):
{guidance}

Only say "I don't have information" if the retrieved data is truly unrelated to the question.

FORMATTING: Use proper markdown formatting in your answer:
- Headers (##, ###) for sections
- Lists (bullet or numbered) for enumerations
- Bold/italic for emphasis
- Tables with proper syntax (ensure blank line before and after)
- Code blocks where appropriate
- CRITICAL: Always add blank lines before and after block elements (tables, code blocks, lists)
- Proper spacing between sections

CRITICAL: Output ONLY the final synthesized answer. Do NOT include:
- Meta-commentary about what you're doing ("I'll search...", "Let me analyze...")
- Explanations of your reasoning process
- Descriptions of your approach
Just provide the direct answer with proper markdown formatting.

CRITICAL: This is a NON-CONVERSATIONAL system. NEVER ask follow-up questions, offer to search again, suggest alternatives, or end with anything like "Would you like me to..." or "Let me know if...". The user cannot reply. Your answer must be complete and self-contained."""

    def _build_synthesis_prompt(
        self,
        query: str,
        fused_facts: list["CrossBankFact"],
        mental_models: list[dict[str, Any]],
        reconciled_disposition: ReconciledDisposition,
        context: str | None,
    ) -> str:
        """Build the synthesis prompt with facts and mental models."""
        sections = []

        # Add context if provided
        if context:
            sections.append(f"## Context\n{context}")

        # Add mental models if available
        if mental_models:
            sections.append("## Mental Models")
            for model in mental_models[:5]:  # Limit to top 5
                sections.append(f"### {model['name']} (from {model.get('bank_id', 'unknown')})")
                sections.append(model["content"])
            sections.append("")

        # Add fused facts
        if fused_facts:
            sections.append("## Retrieved Memories")
            for i, fact in enumerate(fused_facts[:30], 1):  # Limit to 30 facts
                bank_info = f" (from {fact.bank_name or fact.bank_id})" if fact.bank_name or fact.bank_id else ""
                sections.append(f"{i}. {fact.text}{bank_info}")
            sections.append("")

        # Add the query
        sections.append("## Question")
        sections.append(query)

        # Add disposition note
        sections.append("")
        sections.append(
            f"Note: Synthesize this response with the following disposition traits: "
            f"Skepticism={reconciled_disposition.skepticism}/5, "
            f"Literalism={reconciled_disposition.literalism}/5, "
            f"Empathy={reconciled_disposition.empathy}/5"
        )

        return "\n".join(sections)

    async def _generate_structured_output(
        self,
        answer: str,
        response_schema: dict,
        llm_provider: "LLMProvider",
    ) -> dict[str, Any] | None:
        """Generate structured output from an answer using the provided JSON schema."""
        import json

        try:
            # Create prompt for structured extraction
            schema_str = json.dumps(response_schema, indent=2)
            extraction_prompt = f"""Given the following answer, extract structured data according to the JSON schema provided.

ANSWER:
{answer}

JSON SCHEMA:
{schema_str}

Output ONLY valid JSON matching the schema. No markdown, no explanation, just the JSON object."""

            response = await llm_provider.call(
                messages=[
                    {"role": "system", "content": "You extract structured data from text. Output only valid JSON."},
                    {"role": "user", "content": extraction_prompt},
                ],
                scope="cross_bank_structured_output",
                skip_validation=True,
            )

            # Parse the JSON response
            result = json.loads(response.strip())
            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse structured output JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to generate structured output: {e}")
            return None
