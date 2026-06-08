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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return asdict(self)


@dataclass
class BatchRetainChildMetadata:
    """Metadata for child batch_retain operations (individual sub-batches)."""

    items_count: int
    parent_operation_id: str
    sub_batch_index: int
    total_sub_batches: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return asdict(self)


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
