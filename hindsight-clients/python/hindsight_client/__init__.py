"""
Hindsight Client - Clean, pythonic wrapper for the Hindsight API.

This package provides a high-level interface for common Hindsight operations.
For advanced use cases, use the auto-generated API client directly.

Example:
    ```python
    from hindsight_client import Hindsight

    client = Hindsight(base_url="http://localhost:8888")

    # Store a memory
    result = client.retain(bank_id="alice", content="Alice loves AI")
    print(result.success)

    # Search memories
    response = client.recall(bank_id="alice", query="What does Alice like?")
    for r in response.results:
        print(r.text)

    # Generate contextual answer
    answer = client.reflect(bank_id="alice", query="What are my interests?")
    print(answer.text)
    ```
"""

from .hindsight_client import Hindsight

# Re-export response types for convenient access
from hindsight_client_api.models.retain_response import RetainResponse
from hindsight_client_api.models.recall_response import RecallResponse as _RecallResponse
from hindsight_client_api.models.recall_result import RecallResult as _RecallResult
from hindsight_client_api.models.reflect_response import ReflectResponse
from hindsight_client_api.models.reflect_fact import ReflectFact
from hindsight_client_api.models.list_memory_units_response import ListMemoryUnitsResponse
from hindsight_client_api.models.bank_profile_response import BankProfileResponse
from hindsight_client_api.models.disposition_traits import DispositionTraits


# Add cleaner __repr__ and __iter__ for REPL usability
def _recall_result_repr(self):
    text_preview = self.text[:80] + "..." if len(self.text) > 80 else self.text
    return f"RecallResult(id='{self.id[:8]}...', type='{self.type}', text='{text_preview}')"


def _recall_response_repr(self):
    count = len(self.results) if self.results else 0
    extras = []
    if self.trace:
        extras.append("trace=True")
    if self.entities:
        extras.append(f"entities={len(self.entities)}")
    if self.chunks:
        extras.append(f"chunks={len(self.chunks)}")
    extras_str = ", " + ", ".join(extras) if extras else ""
    return f"RecallResponse({count} results{extras_str})"


def _recall_response_iter(self):
    """Iterate directly over results for convenience."""
    return iter(self.results or [])


def _recall_response_len(self):
    """Return number of results."""
    return len(self.results) if self.results else 0


def _recall_response_getitem(self, index):
    """Access results by index."""
    return self.results[index]


_RecallResult.__repr__ = _recall_result_repr
_RecallResponse.__repr__ = _recall_response_repr
_RecallResponse.__iter__ = _recall_response_iter
_RecallResponse.__len__ = _recall_response_len
_RecallResponse.__getitem__ = _recall_response_getitem

# Re-export with patched repr
RecallResult = _RecallResult
RecallResponse = _RecallResponse

__all__ = [
    "Hindsight",
    # Response types
    "RetainResponse",
    "RecallResponse",
    "RecallResult",
    "ReflectResponse",
    "ReflectFact",
    "ListMemoryUnitsResponse",
    "BankProfileResponse",
    "DispositionTraits",
]
