"""Document transfer: export/import documents between banks without re-running the LLM.

An export is a ZIP of already-extracted facts (text, entities by canonical name,
causal relations, chunks) — never embeddings or DB ids. An import replays the
deterministic half of the retain pipeline against the target bank: it re-embeds
locally with the target bank's embedding model, re-resolves entities, and
recreates temporal/semantic/causal links relative to the target bank's existing
memories. No LLM fact-extraction is involved.

Consolidated observations (``fact_type='observation'``) are intentionally
excluded from export — they are derived by consolidation and are regenerated in
the target bank.
"""

from .export import export_bank, export_documents
from .importer import BankImportResult, ImportResult, import_bank, import_documents
from .schema import (
    SCHEMA_VERSION,
    TransferCausalRelation,
    TransferChunk,
    TransferDocument,
    TransferFact,
    TransferManifest,
    TransferObservation,
    TransferObservationSource,
)

__all__ = [
    "SCHEMA_VERSION",
    "BankImportResult",
    "ImportResult",
    "TransferCausalRelation",
    "TransferChunk",
    "TransferDocument",
    "TransferFact",
    "TransferManifest",
    "TransferObservation",
    "TransferObservationSource",
    "export_bank",
    "export_documents",
    "import_bank",
    "import_documents",
]
