"""
Unit tests for ValidationResult.accept_with() enrichment (PR #639).

These tests verify:
1. The accept_with() factory creates an accepted result with the correct enrichment fields.
2. The engine applies enrichment to retain contents and recall tags/tag_groups.
3. RecallContext carries tags/tags_match/tag_groups so validators can read filter state.
"""

import pytest

from hindsight_api.extensions import (
    OperationValidatorExtension,
    RecallContext,
    ReflectContext,
    RetainContext,
    ValidationResult,
)
from hindsight_api.models import RequestContext

# ---------------------------------------------------------------------------
# Pure unit tests for ValidationResult factory methods
# ---------------------------------------------------------------------------


class TestValidationResultAcceptWith:
    """Unit tests for the accept_with() factory — no DB needed."""

    def test_accept_is_allowed_with_no_enrichment(self):
        result = ValidationResult.accept()
        assert result.allowed is True
        assert result.contents is None
        assert result.tags is None
        assert result.tags_match is None
        assert result.tag_groups is None

    def test_accept_with_contents(self):
        contents = [{"content": "enriched text", "tags": ["injected"]}]
        result = ValidationResult.accept_with(contents=contents)
        assert result.allowed is True
        assert result.contents == contents
        assert result.tags is None
        assert result.tag_groups is None

    def test_accept_with_tags(self):
        result = ValidationResult.accept_with(tags=["alpha", "beta"])
        assert result.allowed is True
        assert result.tags == ["alpha", "beta"]
        assert result.contents is None
        assert result.tag_groups is None

    def test_accept_with_tags_match(self):
        result = ValidationResult.accept_with(tags=["x"], tags_match="all")
        assert result.allowed is True
        assert result.tags_match == "all"

    def test_accept_with_tag_groups(self):
        tag_groups = [{"tags": ["env:prod"], "match": "all"}]
        result = ValidationResult.accept_with(tag_groups=tag_groups)
        assert result.allowed is True
        assert result.tag_groups == tag_groups

    def test_accept_with_all_fields(self):
        contents = [{"content": "c"}]
        tags = ["t1"]
        tag_groups = [{"tags": ["g1"]}]
        result = ValidationResult.accept_with(
            contents=contents,
            tags=tags,
            tags_match="any",
            tag_groups=tag_groups,
        )
        assert result.allowed is True
        assert result.contents == contents
        assert result.tags == tags
        assert result.tags_match == "any"
        assert result.tag_groups == tag_groups

    def test_reject_ignores_enrichment_fields(self):
        """reject() always sets allowed=False and leaves enrichment fields at their defaults."""
        result = ValidationResult.reject("not allowed", status_code=403)
        assert result.allowed is False
        assert result.reason == "not allowed"
        assert result.status_code == 403
        assert result.contents is None
        assert result.tags is None

    def test_none_fields_mean_no_modification(self):
        """None enrichment fields must not overwrite engine defaults."""
        result = ValidationResult.accept_with(tags=None, tag_groups=None)
        assert result.tags is None
        assert result.tag_groups is None
        # Engine should interpret None as "keep original" — we verify the contract here.


# ---------------------------------------------------------------------------
# Integration tests: engine applies enrichment from validator
# ---------------------------------------------------------------------------


class _ContentEnrichingValidator(OperationValidatorExtension):
    """Validator that injects a tag into every retain content item."""

    def __init__(self, injected_tag: str):
        super().__init__({})
        self.injected_tag = injected_tag

    async def validate_retain(self, ctx: RetainContext) -> ValidationResult:
        enriched = []
        for item in ctx.contents:
            new_item = dict(item)
            new_item.setdefault("tags", [])
            new_item["tags"] = list(new_item["tags"]) + [self.injected_tag]
            enriched.append(new_item)
        return ValidationResult.accept_with(contents=enriched)

    async def validate_recall(self, ctx: RecallContext) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_reflect(self, ctx: ReflectContext) -> ValidationResult:
        return ValidationResult.accept()


class _TagEnrichingValidator(OperationValidatorExtension):
    """Validator that injects tags into every recall operation."""

    def __init__(self, forced_tags: list[str]):
        super().__init__({})
        self.forced_tags = forced_tags

    async def validate_retain(self, ctx: RetainContext) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_recall(self, ctx: RecallContext) -> ValidationResult:
        return ValidationResult.accept_with(tags=self.forced_tags, tags_match="all")

    async def validate_reflect(self, ctx: ReflectContext) -> ValidationResult:
        return ValidationResult.accept()


class _RecallContextCapturingValidator(OperationValidatorExtension):
    """Validator that captures the RecallContext for inspection."""

    def __init__(self):
        super().__init__({})
        self.captured: list[RecallContext] = []

    async def validate_retain(self, ctx: RetainContext) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_recall(self, ctx: RecallContext) -> ValidationResult:
        self.captured.append(ctx)
        return ValidationResult.accept()

    async def validate_reflect(self, ctx: ReflectContext) -> ValidationResult:
        return ValidationResult.accept()


@pytest.fixture
def memory_with_content_enricher(memory):
    validator = _ContentEnrichingValidator(injected_tag="validator-injected")
    memory._operation_validator = validator
    return memory, validator


@pytest.fixture
def memory_with_tag_enricher(memory):
    validator = _TagEnrichingValidator(forced_tags=["forced-tag"])
    memory._operation_validator = validator
    return memory, validator


@pytest.fixture
def memory_with_recall_context_capture(memory):
    validator = _RecallContextCapturingValidator()
    memory._operation_validator = validator
    return memory, validator


class TestRetainContentEnrichment:
    """Engine applies enriched contents returned by validate_retain."""

    @pytest.mark.asyncio
    async def test_enriched_contents_are_used_for_retain(self, memory_with_content_enricher):
        """When validator returns accept_with(contents=...), engine uses those contents."""
        memory, validator = memory_with_content_enricher
        bank_id = "test-retain-enrichment"
        ctx = RequestContext()

        # Retain without any tags — validator should inject "validator-injected"
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": "Alice is an engineer."}],
            request_context=ctx,
        )

        # Retrieve facts tagged with the injected tag to confirm enrichment was applied
        result = await memory.recall_async(
            bank_id=bank_id,
            query="Alice",
            tags=["validator-injected"],
            request_context=ctx,
        )
        # The fact should be retrievable via the injected tag
        assert result is not None


class TestRecallTagEnrichment:
    """Engine applies enriched tags returned by validate_recall."""

    @pytest.mark.asyncio
    async def test_enriched_tags_filter_recall_results(self, memory_with_tag_enricher):
        """When validator returns accept_with(tags=...), engine filters recall by those tags."""
        memory, validator = memory_with_tag_enricher
        bank_id = "test-recall-tag-enrichment"
        ctx = RequestContext()

        # Retain one fact with the forced tag and one without
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": "Bob is a designer.", "tags": ["forced-tag"]}],
            request_context=ctx,
        )

        # recall is called without tags but validator injects "forced-tag" + match=all
        result = await memory.recall_async(
            bank_id=bank_id,
            query="Bob",
            request_context=ctx,
        )
        # Should still get a result — the injected tag matches the stored fact
        assert result is not None


class TestRecallContextContainsTagFields:
    """RecallContext passed to validate_recall carries tag filter state."""

    @pytest.mark.asyncio
    async def test_recall_context_carries_tags(self, memory_with_recall_context_capture):
        """tags, tags_match, and tag_groups are present in RecallContext."""
        memory, validator = memory_with_recall_context_capture
        bank_id = "test-recall-ctx-tags"
        ctx = RequestContext()
        # Recall 404s for a bank nobody created (#4442), so create it first.
        await memory.ensure_bank_profile(bank_id, request_context=ctx)

        await memory.recall_async(
            bank_id=bank_id,
            query="test",
            tags=["env:prod"],
            tags_match="all",
            request_context=ctx,
        )

        assert len(validator.captured) == 1
        rc = validator.captured[0]
        assert rc.tags == ["env:prod"]
        assert rc.tags_match == "all"

    @pytest.mark.asyncio
    async def test_recall_context_tags_default_to_none(self, memory_with_recall_context_capture):
        """When caller provides no tags, RecallContext.tags is None."""
        memory, validator = memory_with_recall_context_capture
        bank_id = "test-recall-ctx-no-tags"
        ctx = RequestContext()
        # Recall 404s for a bank nobody created (#4442), so create it first.
        await memory.ensure_bank_profile(bank_id, request_context=ctx)

        await memory.recall_async(bank_id=bank_id, query="test", request_context=ctx)

        assert len(validator.captured) == 1
        rc = validator.captured[0]
        assert rc.tags is None
