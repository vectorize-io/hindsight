"""Execution-scoped identity for a claimed async operation."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class OperationClaim:
    """The exact database lease held by one execution attempt."""

    worker_id: str
    claimed_at: datetime

    @classmethod
    def from_database(cls, worker_id: object, claimed_at: object) -> "OperationClaim":
        """Build a claim from PostgreSQL or Oracle result values."""

        if not isinstance(worker_id, str):
            raise TypeError("Claimed operation returned a non-string worker_id")
        if isinstance(claimed_at, str):
            try:
                claimed_at = datetime.fromisoformat(claimed_at)
            except ValueError as exc:
                raise ValueError("Claimed operation returned an invalid claimed_at timestamp") from exc
        if not isinstance(claimed_at, datetime):
            raise TypeError("Claimed operation returned a non-datetime claimed_at")
        if claimed_at.tzinfo is None:
            claimed_at = claimed_at.replace(tzinfo=UTC)
        return cls(worker_id=worker_id, claimed_at=claimed_at)


_current_operation_claim: ContextVar[OperationClaim | None] = ContextVar("current_operation_claim", default=None)


@contextmanager
def bind_operation_claim(claim: OperationClaim | None) -> Iterator[None]:
    """Bind a claim to the current task without altering its persisted payload."""

    token = _current_operation_claim.set(claim)
    try:
        yield
    finally:
        _current_operation_claim.reset(token)


def get_operation_claim() -> OperationClaim | None:
    """Return the claim owned by the current task execution, if any."""

    return _current_operation_claim.get()
