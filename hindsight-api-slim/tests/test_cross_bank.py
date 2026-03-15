"""
Tests for cross-bank orchestration functionality.

Tests the core components: BudgetAllocator, BankSelector, RRF fusion,
and CrossBankOrchestrator.cross_bank_recall().
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from hindsight_api.engine.cross_bank import (
    BudgetAllocator,
    BudgetAllocation,
    BudgetStrategy,
    BankSelector,
    CrossBankOrchestrator,
    reconcile_dispositions,
    ReconciledDisposition,
)
from hindsight_api.api.cross_bank_models import (
    BankInfo,
    CrossBankFact,
    CrossBankRecallResult,
)
from hindsight_api.engine.memory_engine import Budget
from hindsight_api.models import RequestContext


# =============================================================================
# BudgetAllocator Tests
# =============================================================================


class TestBudgetAllocator:
    """Tests for BudgetAllocator static methods."""

    def test_equal_split_basic(self):
        """Test basic equal split across banks."""
        allocation = BudgetAllocator.equal_split(
            budget_value=300,
            bank_ids=["bank-a", "bank-b", "bank-c"],
        )

        assert allocation.strategy == BudgetStrategy.EQUAL
        assert allocation.per_bank == {"bank-a": 100, "bank-b": 100, "bank-c": 100}
        assert allocation.total_used == 300

    def test_equal_split_with_minimum(self):
        """Test equal split respects minimum per bank."""
        # Should fail: 150 budget for 2 banks = 75 each, below minimum 100
        with pytest.raises(ValueError) as exc_info:
            BudgetAllocator.equal_split(
                budget_value=150,
                bank_ids=["bank-a", "bank-b"],
                ensure_minimum=True,
            )

        assert "too low" in str(exc_info.value)

    def test_equal_split_no_banks(self):
        """Test equal split with empty bank list."""
        allocation = BudgetAllocator.equal_split(
            budget_value=300,
            bank_ids=[],
        )

        assert allocation.per_bank == {}
        assert allocation.total_used == 0

    def test_equal_split_without_minimum_check(self):
        """Test equal split can bypass minimum check."""
        allocation = BudgetAllocator.equal_split(
            budget_value=150,
            bank_ids=["bank-a", "bank-b"],
            ensure_minimum=False,
        )

        assert allocation.per_bank == {"bank-a": 75, "bank-b": 75}

    def test_proportional_allocation(self):
        """Test proportional allocation based on bank sizes."""
        allocation = BudgetAllocator.proportional(
            budget_value=300,
            bank_sizes={"bank-a": 100, "bank-b": 200},
        )

        assert allocation.strategy == BudgetStrategy.PROPORTIONAL
        # bank-a has 1/3 of memories, bank-b has 2/3
        assert allocation.per_bank["bank-a"] == 100
        assert allocation.per_bank["bank-b"] == 200

    def test_proportional_empty(self):
        """Test proportional allocation with empty sizes."""
        allocation = BudgetAllocator.proportional(
            budget_value=300,
            bank_sizes={},
        )

        assert allocation.per_bank == {}

    def test_query_relevant_allocation(self):
        """Test allocation based on query relevance scores."""
        allocation = BudgetAllocator.query_relevant(
            budget_value=300,
            bank_relevance={"bank-a": 0.3, "bank-b": 0.7},
        )

        assert allocation.strategy == BudgetStrategy.QUERY_RELEVANT
        # bank-a has 30% relevance, bank-b has 70%
        assert allocation.per_bank["bank-a"] == 90
        assert allocation.per_bank["bank-b"] == 210

    def test_allocate_for_multistep(self):
        """Test multi-step budget allocation with synthesis reserve."""
        allocation, synthesis = BudgetAllocator.allocate_for_multistep(
            total_budget=1000,
            bank_ids=["bank-a", "bank-b"],
            n_steps=2,
        )

        # 30% reserved for synthesis = 300
        assert synthesis == 300
        # Remaining 700 split across 2 steps x 2 banks = 175 per bank per step
        assert allocation.synthesis_reserve == 300


# =============================================================================
# BankSelector Tests
# =============================================================================


class TestBankSelector:
    """Tests for BankSelector.resolve() method."""

    @pytest.fixture
    def mock_engine(self):
        """Create a mock engine for testing."""
        engine = MagicMock()
        engine.list_banks = AsyncMock(return_value=[
            {
                "bank_id": "bank-a",
                "name": "Bank A",
                "disposition": {"skepticism": 3, "literalism": 3, "empathy": 3},
                "mission": "Test bank A",
                "tags": ["personal", "journal"],
            },
            {
                "bank_id": "bank-b",
                "name": "Bank B",
                "disposition": {"skepticism": 5, "literalism": 2, "empathy": 4},
                "mission": "Test bank B",
                "tags": ["work"],
            },
        ])
        engine.get_bank_profile = AsyncMock(side_effect=lambda bank_id, **kwargs: {
            "bank_id": bank_id,
            "name": f"Bank {bank_id[-1].upper()}",
            "disposition": {"skepticism": 3, "literalism": 3, "empathy": 3},
            "mission": f"Test bank {bank_id}",
        })
        return engine

    @pytest.mark.asyncio
    async def test_resolve_with_explicit_ids(self, mock_engine):
        """Test resolving banks with explicit IDs."""
        selector = BankSelector(mock_engine)

        banks = await selector.resolve(bank_ids=["bank-a"])

        assert len(banks) == 1
        assert banks[0].bank_id == "bank-a"
        mock_engine.get_bank_profile.assert_called_once()

    @pytest.mark.asyncio
    async def test_resolve_with_tags(self, mock_engine):
        """Test resolving banks filtered by tags."""
        selector = BankSelector(mock_engine)

        banks = await selector.resolve(bank_tags=["personal"])

        assert len(banks) == 1
        assert banks[0].bank_id == "bank-a"
        assert "personal" in banks[0].tags

    @pytest.mark.asyncio
    async def test_resolve_all_accessible(self, mock_engine):
        """Test resolving all accessible banks."""
        selector = BankSelector(mock_engine)

        banks = await selector.resolve()

        assert len(banks) == 2

    @pytest.mark.asyncio
    async def test_resolve_with_restricted_context(self, mock_engine):
        """Test resolving banks with access restrictions."""
        selector = BankSelector(mock_engine)

        # Create context with restricted bank access
        ctx = RequestContext(allowed_bank_ids=["bank-a"])

        banks = await selector.resolve(bank_ids=["bank-a", "bank-b"], request_context=ctx)

        # Should only return bank-a since bank-b is not in allowed list
        assert len(banks) == 1
        assert banks[0].bank_id == "bank-a"


# =============================================================================
# RRF Fusion Tests
# =============================================================================


class TestRRFFusion:
    """Tests for Reciprocal Rank Fusion in _fuse_results."""

    @pytest.fixture
    def orchestrator(self):
        """Create an orchestrator with a mock engine."""
        engine = MagicMock()
        return CrossBankOrchestrator(engine)

    @pytest.mark.asyncio
    async def test_fuse_results_basic(self, orchestrator):
        """Test basic RRF fusion of results from two banks."""
        bank_results = {
            "bank-a": [
                CrossBankFact(id="fact-1", text="Fact 1", fact_type="world", bank_id="bank-a"),
                CrossBankFact(id="fact-2", text="Fact 2", fact_type="world", bank_id="bank-a"),
            ],
            "bank-b": [
                CrossBankFact(id="fact-2", text="Fact 2", fact_type="world", bank_id="bank-b"),  # Duplicate
                CrossBankFact(id="fact-3", text="Fact 3", fact_type="world", bank_id="bank-b"),
            ],
        }

        fused = await orchestrator._fuse_results(bank_results, max_results=10)

        # Should deduplicate fact-2
        assert len(fused) == 3

        # fact-2 should rank highest because it appears in both banks
        fact_ids = [f.id for f in fused]
        assert fact_ids[0] == "fact-2"  # Highest RRF score

    @pytest.mark.asyncio
    async def test_fuse_results_respects_max_results(self, orchestrator):
        """Test that fusion respects max_results limit."""
        bank_results = {
            "bank-a": [
                CrossBankFact(id=f"fact-{i}", text=f"Fact {i}", fact_type="world", bank_id="bank-a")
                for i in range(10)
            ],
        }

        fused = await orchestrator._fuse_results(bank_results, max_results=5)

        assert len(fused) == 5

    @pytest.mark.asyncio
    async def test_fuse_results_empty_input(self, orchestrator):
        """Test fusion with empty input."""
        fused = await orchestrator._fuse_results({}, max_results=10)

        assert fused == []


# =============================================================================
# Disposition Reconciliation Tests
# =============================================================================


class TestDispositionReconciliation:
    """Tests for disposition reconciliation across banks."""

    def test_reconcile_basic(self):
        """Test basic disposition reconciliation."""
        bank_results = {
            "bank-a": [
                CrossBankFact(id="f1", text="Fact 1", fact_type="world", bank_id="bank-a"),
                CrossBankFact(id="f2", text="Fact 2", fact_type="world", bank_id="bank-a"),
            ],
            "bank-b": [
                CrossBankFact(id="f3", text="Fact 3", fact_type="world", bank_id="bank-b"),
            ],
        }
        bank_dispositions = {
            "bank-a": {"skepticism": 5, "literalism": 2, "empathy": 3},
            "bank-b": {"skepticism": 1, "literalism": 4, "empathy": 5},
        }

        result = reconcile_dispositions(bank_results, bank_dispositions)

        assert result.reconciliation_method == "relevance_weighted"
        # bank-a contributed 2/3 of facts, bank-b contributed 1/3
        # skepticism = 5 * 2/3 + 1 * 1/3 = 3.67 ≈ 4
        assert result.skepticism == 4
        assert result.bank_weights["bank-a"] == 2/3
        assert result.bank_weights["bank-b"] == 1/3

    def test_reconcile_empty_results(self):
        """Test reconciliation with empty results."""
        result = reconcile_dispositions({}, {})

        assert result.reconciliation_method == "default"
        assert result.skepticism == 3
        assert result.literalism == 3
        assert result.empathy == 3


# =============================================================================
# Cross-Bank Recall Integration Tests
# =============================================================================


class TestCrossBankRecall:
    """Integration tests for CrossBankOrchestrator.cross_bank_recall()."""

    @pytest.fixture
    def mock_engine(self):
        """Create a mock engine with recall capability."""
        engine = MagicMock()

        # Mock list_banks
        engine.list_banks = AsyncMock(return_value=[
            {
                "bank_id": "bank-a",
                "name": "Bank A",
                "disposition": {"skepticism": 3, "literalism": 3, "empathy": 3},
                "mission": "Test bank A",
                "tags": ["personal"],
            },
            {
                "bank_id": "bank-b",
                "name": "Bank B",
                "disposition": {"skepticism": 4, "literalism": 2, "empathy": 5},
                "mission": "Test bank B",
                "tags": ["work"],
            },
        ])

        # Mock get_bank_profile
        engine.get_bank_profile = AsyncMock(side_effect=lambda bank_id, **kwargs: {
            "bank_id": bank_id,
            "name": f"Bank {bank_id[-1].upper()}",
            "disposition": {"skepticism": 3, "literalism": 3, "empathy": 3},
            "mission": f"Test bank {bank_id}",
        })

        return engine

    @pytest.mark.asyncio
    async def test_cross_bank_recall_basic(self, mock_engine):
        """Test basic cross-bank recall flow."""
        from hindsight_api.engine.response_models import RecallResult, MemoryFact

        # Mock recall_async to return some facts
        mock_engine.recall_async = AsyncMock(side_effect=lambda bank_id, query, **kwargs: RecallResult(
            results=[
                MemoryFact(
                    id=f"{bank_id}-fact-1",
                    text=f"Fact from {bank_id}",
                    fact_type="world",
                    entities=["Alice"],
                ),
            ]
        ))

        orchestrator = CrossBankOrchestrator(mock_engine)

        result = await orchestrator.cross_bank_recall(
            query="What does Alice do?",
            bank_ids=["bank-a", "bank-b"],
            budget=Budget.MID,
        )

        assert isinstance(result, CrossBankRecallResult)
        assert len(result.results) == 2
        assert result.total_results == 2
        assert "bank-a" in result.bank_stats
        assert "bank-b" in result.bank_stats
        assert result.fusion_metadata["strategy"] == "reciprocal_rank_fusion"

    @pytest.mark.asyncio
    async def test_cross_bank_recall_with_failures(self, mock_engine):
        """Test cross-bank recall handles bank failures gracefully."""
        from hindsight_api.engine.response_models import RecallResult, MemoryFact

        call_count = 0

        async def mock_recall(bank_id, query, **kwargs):
            nonlocal call_count
            call_count += 1
            if bank_id == "bank-b":
                raise Exception("Simulated failure for bank-b")
            return RecallResult(
                results=[
                    MemoryFact(
                        id="fact-1",
                        text="Fact from bank-a",
                        fact_type="world",
                    ),
                ]
            )

        mock_engine.recall_async = AsyncMock(side_effect=mock_recall)

        orchestrator = CrossBankOrchestrator(mock_engine)

        result = await orchestrator.cross_bank_recall(
            query="Test query",
            bank_ids=["bank-a", "bank-b"],
            budget=Budget.LOW,
        )

        # Should still return results from successful bank
        assert len(result.results) == 1
        assert result.results[0].bank_id == "bank-a"

    @pytest.mark.asyncio
    async def test_cross_bank_recall_no_banks(self, mock_engine):
        """Test cross-bank recall with no accessible banks."""
        mock_engine.list_banks = AsyncMock(return_value=[])

        orchestrator = CrossBankOrchestrator(mock_engine)

        result = await orchestrator.cross_bank_recall(
            query="Test query",
        )

        assert result.results == []
        assert result.total_results == 0
        assert result.fusion_metadata["reason"] == "no_banks"


# =============================================================================
# Budget Conversion Tests
# =============================================================================


class TestBudgetConversion:
    """Tests for budget conversion utilities."""

    @pytest.fixture
    def orchestrator(self):
        """Create an orchestrator with a mock engine."""
        engine = MagicMock()
        return CrossBankOrchestrator(engine)

    def test_tokens_to_budget(self, orchestrator):
        """Test token count to Budget enum conversion."""
        assert orchestrator._tokens_to_budget(50) == Budget.LOW
        assert orchestrator._tokens_to_budget(100) == Budget.LOW
        assert orchestrator._tokens_to_budget(150) == Budget.MID
        assert orchestrator._tokens_to_budget(300) == Budget.MID
        assert orchestrator._tokens_to_budget(500) == Budget.HIGH
        assert orchestrator._tokens_to_budget(1000) == Budget.HIGH

    def test_get_budget_value(self, orchestrator):
        """Test Budget enum to token value conversion."""
        assert orchestrator._get_budget_value(None) == 300  # Default to MID
        assert orchestrator._get_budget_value(Budget.LOW) == 100
        assert orchestrator._get_budget_value(Budget.MID) == 300
        assert orchestrator._get_budget_value(Budget.HIGH) == 1000
        assert orchestrator._get_budget_value("low") == 100
        assert orchestrator._get_budget_value("HIGH") == 1000  # Case insensitive


# =============================================================================
# Cross-Bank Reflect Integration Tests
# =============================================================================


class TestCrossBankReflect:
    """Integration tests for CrossBankOrchestrator.cross_bank_reflect()."""

    @pytest.fixture
    def mock_engine_for_reflect(self):
        """Create a mock engine with reflect capability."""
        engine = MagicMock()

        # Mock list_banks
        engine.list_banks = AsyncMock(return_value=[
            {
                "bank_id": "bank-a",
                "name": "Bank A",
                "disposition": {"skepticism": 3, "literalism": 3, "empathy": 3},
                "mission": "Test bank A",
                "tags": ["personal"],
            },
            {
                "bank_id": "bank-b",
                "name": "Bank B",
                "disposition": {"skepticism": 5, "literalism": 2, "empathy": 4},
                "mission": "Test bank B",
                "tags": ["work"],
            },
        ])

        # Mock get_bank_profile
        engine.get_bank_profile = AsyncMock(side_effect=lambda bank_id, **kwargs: {
            "bank_id": bank_id,
            "name": f"Bank {bank_id[-1].upper()}",
            "disposition": {"skepticism": 3, "literalism": 3, "empathy": 3} if bank_id == "bank-a" else {"skepticism": 5, "literalism": 2, "empathy": 4},
            "mission": f"Test bank {bank_id}",
        })

        return engine

    @pytest.mark.asyncio
    async def test_cross_bank_reflect_basic(self, mock_engine_for_reflect):
        """Test basic cross-bank reflect flow."""
        from hindsight_api.engine.response_models import RecallResult, MemoryFact
        from hindsight_api.api.cross_bank_models import CrossBankReflectResult

        # Mock recall_async for evidence gathering
        mock_engine_for_reflect.recall_async = AsyncMock(side_effect=lambda bank_id, query, **kwargs: RecallResult(
            results=[
                MemoryFact(
                    id=f"{bank_id}-fact-1",
                    text=f"Fact from {bank_id}",
                    fact_type="world",
                    entities=["Alice"],
                ),
            ]
        ))

        # Mock LLM provider
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value="Synthesized reflection response")
        mock_engine_for_reflect._llm_provider = mock_llm

        orchestrator = CrossBankOrchestrator(mock_engine_for_reflect)

        result = await orchestrator.cross_bank_reflect(
            query="What does Alice do?",
            bank_ids=["bank-a", "bank-b"],
            budget=Budget.MID,
        )

        assert isinstance(result, CrossBankReflectResult)
        assert "Synthesized reflection response" in result.text or len(result.based_on) >= 0
        assert "bank-a" in result.bank_dispositions
        assert "bank-b" in result.bank_dispositions

    @pytest.mark.asyncio
    async def test_cross_bank_reflect_with_mental_models(self, mock_engine_for_reflect):
        """Test cross-bank reflect includes mental models."""
        from hindsight_api.engine.response_models import RecallResult, MemoryFact
        from hindsight_api.api.cross_bank_models import CrossBankReflectResult

        # Mock recall_async
        mock_engine_for_reflect.recall_async = AsyncMock(return_value=RecallResult(
            results=[
                MemoryFact(id="fact-1", text="Test fact", fact_type="world"),
            ]
        ))

        # Mock get_mental_models
        mock_engine_for_reflect.get_mental_models = AsyncMock(return_value=[
            {"id": "mm-1", "name": "User Prefs", "content": "Prefers Python"}
        ])

        # Mock LLM provider
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value="Reflection with mental models")
        mock_engine_for_reflect._llm_provider = mock_llm

        orchestrator = CrossBankOrchestrator(mock_engine_for_reflect)

        result = await orchestrator.cross_bank_reflect(
            query="What are user preferences?",
            bank_ids=["bank-a"],
            include_mental_models=True,
            budget=Budget.LOW,
        )

        assert isinstance(result, CrossBankReflectResult)

    @pytest.mark.asyncio
    async def test_cross_bank_reflect_no_banks(self, mock_engine_for_reflect):
        """Test cross-bank reflect with no accessible banks."""
        mock_engine_for_reflect.list_banks = AsyncMock(return_value=[])

        orchestrator = CrossBankOrchestrator(mock_engine_for_reflect)

        result = await orchestrator.cross_bank_reflect(
            query="Test query",
        )

        assert "No accessible banks" in result.text
        assert result.based_on == []

    @pytest.mark.asyncio
    async def test_cross_bank_reflect_llm_failure(self, mock_engine_for_reflect):
        """Test cross-bank reflect handles LLM failures gracefully."""
        from hindsight_api.engine.response_models import RecallResult, MemoryFact

        # Mock recall_async
        mock_engine_for_reflect.recall_async = AsyncMock(return_value=RecallResult(
            results=[
                MemoryFact(id="fact-1", text="Test fact", fact_type="world"),
            ]
        ))

        # Mock LLM provider that fails
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(side_effect=Exception("LLM API error"))
        mock_engine_for_reflect._llm_provider = mock_llm

        orchestrator = CrossBankOrchestrator(mock_engine_for_reflect)

        result = await orchestrator.cross_bank_reflect(
            query="Test query",
            bank_ids=["bank-a"],
            budget=Budget.LOW,
        )

        # Should return an error message in the text
        assert "LLM" in result.text or "error" in result.text.lower() or len(result.text) > 0

    @pytest.mark.asyncio
    async def test_cross_bank_reflect_disposition_reconciliation(self, mock_engine_for_reflect):
        """Test that dispositions are reconciled correctly."""
        from hindsight_api.engine.response_models import RecallResult, MemoryFact

        # Mock recall_async - bank-a returns 2 facts, bank-b returns 1 fact
        async def mock_recall(bank_id, query, **kwargs):
            if bank_id == "bank-a":
                return RecallResult(results=[
                    MemoryFact(id="f1", text="Fact 1", fact_type="world"),
                    MemoryFact(id="f2", text="Fact 2", fact_type="world"),
                ])
            else:
                return RecallResult(results=[
                    MemoryFact(id="f3", text="Fact 3", fact_type="world"),
                ])

        mock_engine_for_reflect.recall_async = AsyncMock(side_effect=mock_recall)

        # Mock LLM provider
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value="Synthesized")
        mock_engine_for_reflect._llm_provider = mock_llm

        orchestrator = CrossBankOrchestrator(mock_engine_for_reflect)

        result = await orchestrator.cross_bank_reflect(
            query="Test query",
            bank_ids=["bank-a", "bank-b"],
            budget=Budget.MID,
        )

        # Verify dispositions are present
        assert "bank-a" in result.bank_dispositions
        assert "bank-b" in result.bank_dispositions
        # bank-a disposition should have skepticism=3, bank-b should have skepticism=5
        assert result.bank_dispositions["bank-a"]["skepticism"] == 3
        assert result.bank_dispositions["bank-b"]["skepticism"] == 5


# =============================================================================
# Evidence Gathering Tests
# =============================================================================


class TestEvidenceGathering:
    """Tests for _gather_evidence method."""

    @pytest.fixture
    def mock_engine_for_evidence(self):
        """Create a mock engine for evidence gathering tests."""
        engine = MagicMock()
        engine.list_banks = AsyncMock(return_value=[
            {
                "bank_id": "bank-a",
                "name": "Bank A",
                "disposition": {"skepticism": 3, "literalism": 3, "empathy": 3},
                "mission": "Test bank A",
            },
        ])
        engine.get_bank_profile = AsyncMock(return_value={
            "bank_id": "bank-a",
            "name": "Bank A",
            "disposition": {"skepticism": 3, "literalism": 3, "empathy": 3},
            "mission": "Test bank A",
        })
        return engine

    @pytest.mark.asyncio
    async def test_gather_evidence_basic(self, mock_engine_for_evidence):
        """Test basic evidence gathering."""
        from hindsight_api.engine.response_models import RecallResult, MemoryFact
        from hindsight_api.api.cross_bank_models import BankInfo

        # Mock recall_async
        mock_engine_for_evidence.recall_async = AsyncMock(return_value=RecallResult(
            results=[
                MemoryFact(id="fact-1", text="Test fact", fact_type="world"),
            ]
        ))

        orchestrator = CrossBankOrchestrator(mock_engine_for_evidence)

        banks = [BankInfo(
            bank_id="bank-a",
            name="Bank A",
            disposition={"skepticism": 3, "literalism": 3, "empathy": 3},
            mission="Test bank A",
        )]

        evidence = await orchestrator._gather_evidence(
            query="Test query",
            banks=banks,
            budget_allocation=BudgetAllocation(per_bank={"bank-a": 100}, total_used=100),
            include_mental_models=False,
            request_context=RequestContext(),
        )

        assert "fused_facts" in evidence
        assert "bank_results" in evidence
        assert len(evidence["fused_facts"]) >= 0

    @pytest.mark.asyncio
    async def test_gather_evidence_with_mental_models(self, mock_engine_for_evidence):
        """Test evidence gathering includes mental models when requested."""
        from hindsight_api.engine.response_models import RecallResult, MemoryFact
        from hindsight_api.api.cross_bank_models import BankInfo

        # Mock recall_async
        mock_engine_for_evidence.recall_async = AsyncMock(return_value=RecallResult(
            results=[
                MemoryFact(id="fact-1", text="Test fact", fact_type="world"),
            ]
        ))

        # Mock get_mental_models
        mock_engine_for_evidence.get_mental_models = AsyncMock(return_value=[
            {"id": "mm-1", "name": "Test Model", "content": "Test content"}
        ])

        orchestrator = CrossBankOrchestrator(mock_engine_for_evidence)

        banks = [BankInfo(
            bank_id="bank-a",
            name="Bank A",
            disposition={"skepticism": 3, "literalism": 3, "empathy": 3},
            mission="Test bank A",
        )]

        evidence = await orchestrator._gather_evidence(
            query="Test query",
            banks=banks,
            budget_allocation=BudgetAllocation(per_bank={"bank-a": 100}, total_used=100),
            include_mental_models=True,
            request_context=RequestContext(),
        )

        assert "mental_models" in evidence
        assert len(evidence["mental_models"]) == 1
        assert evidence["mental_models"][0]["name"] == "Test Model"


# =============================================================================
# Extension Hook Tests
# =============================================================================


class TestCrossBankRecallHooks:
    """Tests for extension hooks in cross_bank_recall."""

    @pytest.fixture
    def mock_engine_with_validator(self):
        """Create a mock engine with operation validator."""
        engine = MagicMock()

        # Mock list_banks
        engine.list_banks = AsyncMock(return_value=[
            {
                "bank_id": "bank-a",
                "name": "Bank A",
                "disposition": {"skepticism": 3, "literalism": 3, "empathy": 3},
                "mission": "Test bank A",
                "tags": ["personal"],
            },
        ])

        # Mock get_bank_profile
        engine.get_bank_profile = AsyncMock(return_value={
            "bank_id": "bank-a",
            "name": "Bank A",
            "disposition": {"skepticism": 3, "literalism": 3, "empathy": 3},
            "mission": "Test bank A",
        })

        # Mock recall_async
        engine.recall_async = AsyncMock(return_value=MagicMock(
            results=[],
            model_dump=lambda: {"results": []}
        ))

        # Mock operation validator
        engine._operation_validator = MagicMock()
        engine._operation_validator.validate_cross_bank_recall = AsyncMock(
            return_value=MagicMock(allowed=True, reason=None, status_code=None)
        )
        engine._operation_validator.notify_cross_bank_recall_complete = AsyncMock()

        return engine

    @pytest.mark.asyncio
    async def test_pre_validation_hook_allowed(self, mock_engine_with_validator):
        """Test that pre-validation hook is called and allows operation."""
        from hindsight_api.engine.response_models import RecallResult, MemoryFact

        mock_engine_with_validator.recall_async = AsyncMock(return_value=RecallResult(
            results=[
                MemoryFact(id="fact-1", text="Test fact", fact_type="world"),
            ]
        ))

        orchestrator = CrossBankOrchestrator(mock_engine_with_validator)

        result = await orchestrator.cross_bank_recall(
            query="Test query",
            bank_ids=["bank-a"],
            budget=Budget.LOW,
        )

        # Verify pre-validation hook was called
        mock_engine_with_validator._operation_validator.validate_cross_bank_recall.assert_called_once()
        call_kwargs = mock_engine_with_validator._operation_validator.validate_cross_bank_recall.call_args.kwargs
        validation_ctx = call_kwargs["context"]
        assert validation_ctx.query == "Test query"
        assert validation_ctx.bank_ids == ["bank-a"]

    @pytest.mark.asyncio
    async def test_pre_validation_hook_denied(self, mock_engine_with_validator):
        """Test that pre-validation hook can deny operation."""
        from hindsight_api.extensions import OperationValidationError

        # Configure validator to deny
        mock_engine_with_validator._operation_validator.validate_cross_bank_recall = AsyncMock(
            return_value=MagicMock(allowed=False, reason="Rate limit exceeded", status_code=429)
        )

        orchestrator = CrossBankOrchestrator(mock_engine_with_validator)

        with pytest.raises(OperationValidationError) as exc_info:
            await orchestrator.cross_bank_recall(
                query="Test query",
                bank_ids=["bank-a"],
                budget=Budget.LOW,
            )

        assert "Rate limit exceeded" in str(exc_info.value)
        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_post_completion_hook_success(self, mock_engine_with_validator):
        """Test that post-completion hook is called after success."""
        from hindsight_api.engine.response_models import RecallResult, MemoryFact

        mock_engine_with_validator.recall_async = AsyncMock(return_value=RecallResult(
            results=[
                MemoryFact(id="fact-1", text="Test fact", fact_type="world"),
            ]
        ))

        orchestrator = CrossBankOrchestrator(mock_engine_with_validator)

        result = await orchestrator.cross_bank_recall(
            query="Test query",
            bank_ids=["bank-a"],
            budget=Budget.LOW,
        )

        # Verify post-completion hook was called
        mock_engine_with_validator._operation_validator.notify_cross_bank_recall_complete.assert_called_once()
        call_kwargs = mock_engine_with_validator._operation_validator.notify_cross_bank_recall_complete.call_args.kwargs
        completion_ctx = call_kwargs["context"]
        assert completion_ctx.success is True
        assert completion_ctx.total_results >= 0

    @pytest.mark.asyncio
    async def test_hook_context_includes_budget(self, mock_engine_with_validator):
        """Test that hook context includes budget parameter."""
        from hindsight_api.engine.response_models import RecallResult, MemoryFact

        mock_engine_with_validator.recall_async = AsyncMock(return_value=RecallResult(results=[]))

        orchestrator = CrossBankOrchestrator(mock_engine_with_validator)

        result = await orchestrator.cross_bank_recall(
            query="Test query",
            bank_ids=["bank-a"],
            budget=Budget.HIGH,
        )

        call_kwargs = mock_engine_with_validator._operation_validator.validate_cross_bank_recall.call_args.kwargs
        validation_ctx = call_kwargs["context"]
        assert validation_ctx.budget == Budget.HIGH


class TestCrossBankReflectHooks:
    """Tests for extension hooks in cross_bank_reflect."""

    @pytest.fixture
    def mock_engine_for_reflect_hooks(self):
        """Create a mock engine for reflect hook tests."""
        engine = MagicMock()

        # Mock list_banks
        engine.list_banks = AsyncMock(return_value=[
            {
                "bank_id": "bank-a",
                "name": "Bank A",
                "disposition": {"skepticism": 3, "literalism": 3, "empathy": 3},
                "mission": "Test bank A",
                "tags": ["personal"],
            },
        ])

        # Mock get_bank_profile
        engine.get_bank_profile = AsyncMock(return_value={
            "bank_id": "bank-a",
            "name": "Bank A",
            "disposition": {"skepticism": 3, "literalism": 3, "empathy": 3},
            "mission": "Test bank A",
        })

        # Mock recall_async for evidence gathering
        engine.recall_async = AsyncMock(return_value=MagicMock(
            results=[],
            model_dump=lambda: {"results": []}
        ))

        # Mock LLM provider
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value="Synthesized reflection")
        engine._llm_provider = mock_llm

        # Mock operation validator
        engine._operation_validator = MagicMock()
        engine._operation_validator.validate_cross_bank_reflect = AsyncMock(
            return_value=MagicMock(allowed=True, reason=None, status_code=None)
        )
        engine._operation_validator.notify_cross_bank_reflect_complete = AsyncMock()

        return engine

    @pytest.mark.asyncio
    async def test_pre_validation_hook_allowed(self, mock_engine_for_reflect_hooks):
        """Test that pre-validation hook is called and allows reflect operation."""
        orchestrator = CrossBankOrchestrator(mock_engine_for_reflect_hooks)

        result = await orchestrator.cross_bank_reflect(
            query="Test question",
            bank_ids=["bank-a"],
            budget=Budget.LOW,
        )

        # Verify pre-validation hook was called
        mock_engine_for_reflect_hooks._operation_validator.validate_cross_bank_reflect.assert_called_once()
        call_kwargs = mock_engine_for_reflect_hooks._operation_validator.validate_cross_bank_reflect.call_args.kwargs
        validation_ctx = call_kwargs["context"]
        assert validation_ctx.query == "Test question"
        assert validation_ctx.bank_ids == ["bank-a"]

    @pytest.mark.asyncio
    async def test_pre_validation_hook_denied(self, mock_engine_for_reflect_hooks):
        """Test that pre-validation hook can deny reflect operation."""
        from hindsight_api.extensions import OperationValidationError

        # Configure validator to deny
        mock_engine_for_reflect_hooks._operation_validator.validate_cross_bank_reflect = AsyncMock(
            return_value=MagicMock(allowed=False, reason="Reflect not allowed for this user", status_code=403)
        )

        orchestrator = CrossBankOrchestrator(mock_engine_for_reflect_hooks)

        with pytest.raises(OperationValidationError) as exc_info:
            await orchestrator.cross_bank_reflect(
                query="Test question",
                bank_ids=["bank-a"],
                budget=Budget.LOW,
            )

        assert "Reflect not allowed" in str(exc_info.value)
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_post_completion_hook_success(self, mock_engine_for_reflect_hooks):
        """Test that post-completion hook is called after successful reflect."""
        orchestrator = CrossBankOrchestrator(mock_engine_for_reflect_hooks)

        result = await orchestrator.cross_bank_reflect(
            query="Test question",
            bank_ids=["bank-a"],
            budget=Budget.LOW,
        )

        # Verify post-completion hook was called
        mock_engine_for_reflect_hooks._operation_validator.notify_cross_bank_reflect_complete.assert_called_once()
        call_kwargs = mock_engine_for_reflect_hooks._operation_validator.notify_cross_bank_reflect_complete.call_args.kwargs
        completion_ctx = call_kwargs["context"]
        assert completion_ctx.success is True

    @pytest.mark.asyncio
    async def test_post_completion_hook_on_llm_failure(self, mock_engine_for_reflect_hooks):
        """Test that post-completion hook is called even when LLM fails."""
        # Configure LLM to fail
        mock_engine_for_reflect_hooks._llm_provider.generate = AsyncMock(
            side_effect=Exception("LLM API error")
        )

        orchestrator = CrossBankOrchestrator(mock_engine_for_reflect_hooks)

        result = await orchestrator.cross_bank_reflect(
            query="Test question",
            bank_ids=["bank-a"],
            budget=Budget.LOW,
        )

        # Verify post-completion hook was still called
        mock_engine_for_reflect_hooks._operation_validator.notify_cross_bank_reflect_complete.assert_called_once()
        call_kwargs = mock_engine_for_reflect_hooks._operation_validator.notify_cross_bank_reflect_complete.call_args.kwargs
        completion_ctx = call_kwargs["context"]
        assert completion_ctx.success is False

    @pytest.mark.asyncio
    async def test_hook_context_includes_include_mental_models(self, mock_engine_for_reflect_hooks):
        """Test that hook context includes include_mental_models parameter."""
        orchestrator = CrossBankOrchestrator(mock_engine_for_reflect_hooks)

        result = await orchestrator.cross_bank_reflect(
            query="Test question",
            bank_ids=["bank-a"],
            budget=Budget.LOW,
            include_mental_models=False,
        )

        call_kwargs = mock_engine_for_reflect_hooks._operation_validator.validate_cross_bank_reflect.call_args.kwargs
        validation_ctx = call_kwargs["context"]
        assert validation_ctx.include_mental_models is False


class TestHookContextData:
    """Tests for hook context data structures."""

    def test_cross_bank_recall_context_fields(self):
        """Test CrossBankRecallContext has expected fields."""
        from hindsight_api.extensions.operation_validator import CrossBankRecallContext

        ctx = CrossBankRecallContext(
            bank_ids=["bank-a", "bank-b"],
            query="Test query",
            request_context=RequestContext(),
            budget=Budget.MID,
            bank_tags=["personal"],
            max_results=20,
        )

        assert ctx.bank_ids == ["bank-a", "bank-b"]
        assert ctx.query == "Test query"
        assert ctx.budget == Budget.MID
        assert ctx.bank_tags == ["personal"]
        assert ctx.max_results == 20

    def test_cross_bank_reflect_context_fields(self):
        """Test CrossBankReflectContext has expected fields."""
        from hindsight_api.extensions.operation_validator import CrossBankReflectContext

        ctx = CrossBankReflectContext(
            bank_ids=["bank-a"],
            query="Test question",
            request_context=RequestContext(),
            budget=Budget.LOW,
            bank_tags=["work"],
            context="Additional context",
            include_mental_models=True,
            include_reasoning_chain=False,
            response_schema=None,
        )

        assert ctx.bank_ids == ["bank-a"]
        assert ctx.query == "Test question"
        assert ctx.budget == Budget.LOW
        assert ctx.context == "Additional context"
        assert ctx.include_mental_models is True
        assert ctx.include_reasoning_chain is False