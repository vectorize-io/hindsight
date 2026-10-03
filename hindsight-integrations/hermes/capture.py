"""Detached automatic-retain events, with the source clock separate from audit time."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from agent.memory_sync_snapshot import CompletedTurnSnapshot


class SnapshotMessage(BaseModel):
    role: str
    content: Any = None
    timestamp: Any = None


@dataclass(frozen=True)
class CapturedMessage:
    role: str
    content: str
    timestamp: str


@dataclass(frozen=True)
class CapturedTurn:
    messages: tuple[CapturedMessage, ...]

    @property
    def content_size(self) -> int:
        return sum(len(message.content) for message in self.messages)


def normalize_timestamp(value: Any) -> str | None:
    """Normalize an epoch or aware ISO instant; never guess a timezone."""
    if isinstance(value, bool):
        return None
    try:
        if isinstance(value, (int, float)):
            if not math.isfinite(value):
                return None
            stamp = datetime.fromtimestamp(value, tz=timezone.utc)
        elif isinstance(value, str):
            stamp = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
            if stamp.tzinfo is None or stamp.utcoffset() is None:
                return None
        else:
            return None
        return stamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except (ValueError, OverflowError, OSError):
        return None


def capture_turn(
    user_content: str,
    assistant_content: str,
    *,
    snapshot: CompletedTurnSnapshot | None,
    fallback_timestamp: str,
    user_prefix: str,
    assistant_prefix: str,
) -> CapturedTurn:
    """Copy only the same user/final-assistant text captured before this feature.

    Tool messages and arguments are intentionally excluded. The host owns the
    bounded snapshot contract; missing/invalid dates use one enqueue-time clock.
    """
    user_time = assistant_time = fallback_timestamp
    if snapshot is not None:
        messages = [SnapshotMessage.model_validate(message) for message in snapshot.messages()]
        # Match the host's completed-turn contract rather than picking an
        # intermediate tool-calling assistant response as the final answer.
        if (
            messages
            and messages[0].role == "user"
            and messages[-1].role == "assistant"
            and messages[-1].content == assistant_content
        ):
            # MemoryManager may strip skill scaffolding from user_content after
            # building the snapshot. Its source timestamp still belongs to this
            # turn; do not require equality with the filtered prompt text.
            user_time = normalize_timestamp(messages[0].timestamp) or fallback_timestamp
            assistant_time = normalize_timestamp(messages[-1].timestamp) or fallback_timestamp
    return CapturedTurn(
        messages=(
            CapturedMessage("user", f"{user_prefix}: {user_content}", user_time),
            CapturedMessage("assistant", f"{assistant_prefix}: {assistant_content}", assistant_time),
        )
    )
