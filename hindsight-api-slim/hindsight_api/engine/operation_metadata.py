"""
Typed metadata models for async operations.

These dataclasses define the structure of result_metadata for different operation types.
The metadata is exposed in the API for debugging purposes and may change without notice.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

MAX_EXTRACTION_ERROR_SAMPLES = 5


@dataclass
class BatchRetainParentMetadata:
    """Metadata for parent batch_retain operations (when split into sub-batches)."""

    items_count: int
    total_tokens: int
    num_sub_batches: int
    is_parent: bool = True
    # Set only when the whole batch targets a single document, so the operations
    # list surfaces which document an in-flight retain is (re)writing. The
    # documents UI cross-checks this to badge rows as "updating". Multi-document
    # batches leave it None and are matched per single-document child instead.
    document_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization, omitting document_id when unset."""
        data = asdict(self)
        if data.get("document_id") is None:
            data.pop("document_id", None)
        return data


@dataclass
class BatchRetainChildMetadata:
    """Metadata for child batch_retain operations (individual sub-batches)."""

    items_count: int
    parent_operation_id: str
    sub_batch_index: int
    total_sub_batches: int
    # Set only when this child processes a single document (see the parent's note).
    document_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization, omitting document_id when unset."""
        data = asdict(self)
        if data.get("document_id") is None:
            data.pop("document_id", None)
        return data


@dataclass
class RetainMetadata:
    """Metadata for regular retain operations (non-batched, deprecated async path)."""

    items_count: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return asdict(self)


@dataclass
class RetainExtractionErrors:
    """Non-fatal fact extraction failures observed inside one retain operation."""

    count: int = 0
    sample: list[str] = field(default_factory=list)

    def add(self, message: str) -> None:
        """Record one extraction error while keeping the stored sample bounded."""
        self.count += 1
        if len(self.sample) < MAX_EXTRACTION_ERROR_SAMPLES:
            self.sample.append(message[:500])

    def merge_metadata(self, metadata: Mapping[str, Any]) -> None:
        """Merge errors already present on an operation result_metadata object."""
        self.count += int(metadata.get("extraction_errors_count") or 0)

        sample = metadata.get("extraction_errors_sample") or []
        if isinstance(sample, str):
            sample = [sample]
        if isinstance(sample, list):
            for entry in sample:
                if isinstance(entry, str) and len(self.sample) < MAX_EXTRACTION_ERROR_SAMPLES:
                    self.sample.append(entry[:500])

    def to_dict(self) -> dict[str, Any]:
        """Convert to the public result_metadata field shape."""
        data: dict[str, Any] = {"extraction_errors_count": self.count}
        if self.sample:
            data["extraction_errors_sample"] = self.sample
        return data


@dataclass
class RetainOutcomeMetadata:
    """Machine-readable outcome metadata for a completed retain operation."""

    unit_ids_count: int
    extraction_errors_count: int = 0
    extraction_errors_sample: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization, omitting empty optional samples."""
        data: dict[str, Any] = {
            "unit_ids_count": self.unit_ids_count,
            "extraction_errors_count": self.extraction_errors_count,
        }
        if self.extraction_errors_sample:
            data["extraction_errors_sample"] = self.extraction_errors_sample[:MAX_EXTRACTION_ERROR_SAMPLES]
        return data


@dataclass
class RetainOutcomeAggregate:
    """Aggregate retain outcome metadata from child retain operations."""

    unit_ids_count: int = 0
    extraction_errors: RetainExtractionErrors = field(default_factory=RetainExtractionErrors)

    def add_metadata(self, metadata: Mapping[str, Any]) -> None:
        """Fold one child operation's result_metadata into the aggregate."""
        self.unit_ids_count += int(metadata.get("unit_ids_count") or 0)
        self.extraction_errors.merge_metadata(metadata)

    def to_outcome_metadata(self) -> RetainOutcomeMetadata:
        """Return the aggregate in the public result_metadata field shape."""
        return RetainOutcomeMetadata(
            unit_ids_count=self.unit_ids_count,
            extraction_errors_count=self.extraction_errors.count,
            extraction_errors_sample=self.extraction_errors.sample,
        )


@dataclass
class ConsolidationMetadata:
    """Metadata for consolidation operations."""

    # Currently empty, but structure for future fields
    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return asdict(self)


@dataclass
class RefreshMentalModelMetadata:
    """Metadata for mental model refresh operations."""

    mental_model_id: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return asdict(self)


@dataclass
class RefreshMentalModelOutcomeMetadata:
    """Machine-readable outcome metadata for a completed refresh_mental_model operation.

    Refresh parity with RetainOutcomeMetadata (#2605): lets a monitoring layer
    distinguish "refreshed with real content" from "refreshed empty" by reading
    result_metadata alone, without a follow-up content fetch.
    """

    content_len: int
    populated_content: bool
    based_on_counts: dict[str, int] = field(default_factory=dict)
    # Delta operations the model emitted, as applied vs rejected. A refresh whose
    # ops are routinely rejected still completes successfully with a plausible
    # document, so the count is the only signal that some of this run's new facts
    # never reached it. Both are 0 for a full-mode refresh, which emits no ops.
    delta_ops_applied: int = 0
    delta_ops_skipped: int = 0
    # Which tier actually served this refresh: "tier0" (delta window empty, document
    # preserved, no LLM call), "tier1" (one structured-delta call), or "tier2" (the
    # agentic reflect loop). Without this the tier is only readable from
    # ``mental_models.reflect_response.fast_path``, which holds the LATEST refresh per
    # model and is overwritten by the next one -- so a two-tier system's tier
    # distribution is unrecoverable the moment a model refreshes again. Operating and
    # measuring a tiered system requires knowing, per operation, which tier ran.
    serving_tier: str | None = None
    # Set only on tier2, naming why the fast path handed back ("no_delta_baseline",
    # "needs_full_context", "delta_ops_failed", "delta_ops_invalid",
    # "delta_ops_all_skipped"). A tier2 rate is a number; this is the reason behind it,
    # and the two failure directions the plan flags -- never falling back (dead escape
    # hatch) vs always falling back (fast path not earning its place) -- are only
    # distinguishable with it.
    fast_path_fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return asdict(self)
