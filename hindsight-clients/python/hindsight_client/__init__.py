"""
Hindsight Client - Clean, pythonic wrapper for the Hindsight API.

This package provides a high-level ``Hindsight`` class with simplified methods
for the most common operations (retain, recall, reflect, banks, mental models,
directives).

For operations not available as convenience methods — such as documents,
entities, async operations, webhooks, and monitoring — use the low-level API
clients exposed as properties on the ``Hindsight`` instance (e.g.
``client.documents``, ``client.entities``, ``client.operations``).
All low-level methods are async.

Quick start::

    from hindsight_client import Hindsight

    client = Hindsight(base_url="http://localhost:8888")

    # Store a memory
    client.retain(bank_id="alice", content="Alice loves AI")

    # Search memories
    response = client.recall(bank_id="alice", query="What does Alice like?")
    for r in response.results:
        print(r.text)

    # Generate contextual answer
    answer = client.reflect(bank_id="alice", query="What are my interests?")
    print(answer.text)

Low-level API access::

    import asyncio

    # List documents
    docs = asyncio.run(client.documents.list_documents("alice"))

    # Check operation status
    status = asyncio.run(client.operations.get_operation_status("alice", "op-id"))

    # List entities
    entities = asyncio.run(client.entities.list_entities("alice"))
"""

from hindsight_client_api.models.bank_profile_response import BankProfileResponse
from hindsight_client_api.models.disposition_traits import DispositionTraits
from hindsight_client_api.models.list_memory_units_response import ListMemoryUnitsResponse
from hindsight_client_api.models.recall_response import RecallResponse as _RecallResponse
from hindsight_client_api.models.recall_result import RecallResult as _RecallResult
from hindsight_client_api.models.reflect_fact import ReflectFact
from hindsight_client_api.models.reflect_response import ReflectResponse

# Re-export response types for convenient access
from hindsight_client_api.models.retain_response import RetainResponse

from .hindsight_client import Hindsight


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


def _recall_response_to_prompt_string(self) -> str:
    """Serialize the recall response to a string suitable for LLM prompts.

    Builds a prompt containing:
    - Facts: each result as a JSON object with ``text``, ``context``, and
      temporal fields (``occurred_start``, ``occurred_end``, ``mentioned_at``).
      If the result has a ``chunk_id`` matching a chunk in the response, the
      chunk text is included as ``source_chunk``.
    - Entities: entity summaries from observations, formatted as sections.

    This mirrors the format used internally by Hindsight's reflect operation.
    """
    import json

    chunks_map = self.chunks or {}
    sections: list[str] = []

    # Facts
    formatted_facts: list[dict] = []
    for result in self.results or []:
        fact_obj: dict = {"text": result.text}
        if result.context:
            fact_obj["context"] = result.context
        for field in ("occurred_start", "occurred_end", "mentioned_at"):
            value = getattr(result, field, None)
            if value:
                fact_obj[field] = value
        if result.chunk_id and result.chunk_id in chunks_map:
            fact_obj["source_chunk"] = chunks_map[result.chunk_id].text
        formatted_facts.append(fact_obj)
    sections.append("FACTS:\n" + json.dumps(formatted_facts, indent=2))

    # Entities
    if self.entities:
        entity_parts: list[str] = []
        for name, state in self.entities.items():
            if state.observations:
                obs_text = state.observations[0].text
                entity_parts.append(f"## {name}\n{obs_text}")
        if entity_parts:
            sections.append("ENTITIES:\n" + "\n\n".join(entity_parts))

    return "\n\n".join(sections)


_RecallResponse.to_prompt_string = _recall_response_to_prompt_string

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
