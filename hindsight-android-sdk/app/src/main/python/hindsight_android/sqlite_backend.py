"""SQLite-based memory backend for on-device Hindsight.

Replaces PostgreSQL + pgvector with SQLite + Python-side cosine similarity.
Designed for Android where running a PG server is not feasible.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class Fact:
    id: str
    bank_id: str
    text: str
    fact_type: str  # "world" or "experience"
    embedding: list[float] | None = None
    entities: list[str] = field(default_factory=list)
    context: str | None = None
    occurred_start: str | None = None
    occurred_end: str | None = None
    metadata: dict[str, str] | None = None
    tags: list[str] = field(default_factory=list)
    created_at: float = 0.0


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SQLiteMemoryBackend:
    """In-process SQLite memory store with vector similarity search."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def initialize(self) -> None:
        """Create tables if they don't exist."""
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS banks (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                mission TEXT DEFAULT '',
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory_units (
                id TEXT PRIMARY KEY,
                bank_id TEXT NOT NULL REFERENCES banks(id),
                text TEXT NOT NULL,
                fact_type TEXT NOT NULL DEFAULT 'world',
                embedding BLOB,
                entities TEXT DEFAULT '[]',
                context TEXT,
                occurred_start TEXT,
                occurred_end TEXT,
                metadata TEXT DEFAULT '{}',
                tags TEXT DEFAULT '[]',
                created_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_memory_units_bank
                ON memory_units(bank_id);
        """)
        self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def create_bank(self, bank_id: str, name: str, mission: str = "") -> None:
        assert self._conn is not None
        self._conn.execute(
            "INSERT OR IGNORE INTO banks (id, name, mission, created_at) VALUES (?, ?, ?, ?)",
            (bank_id, name, mission, time.time()),
        )
        self._conn.commit()

    def get_bank(self, bank_id: str) -> dict | None:
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT id, name, mission FROM banks WHERE id = ?", (bank_id,)
        ).fetchone()
        if row is None:
            return None
        return {"id": row[0], "name": row[1], "mission": row[2]}

    def store_facts(self, facts: list[Fact]) -> None:
        """Store extracted facts with embeddings."""
        assert self._conn is not None
        for fact in facts:
            embedding_blob = (
                _encode_embedding(fact.embedding) if fact.embedding else None
            )
            self._conn.execute(
                """INSERT INTO memory_units
                   (id, bank_id, text, fact_type, embedding, entities, context,
                    occurred_start, occurred_end, metadata, tags, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fact.id,
                    fact.bank_id,
                    fact.text,
                    fact.fact_type,
                    embedding_blob,
                    json.dumps(fact.entities),
                    fact.context,
                    fact.occurred_start,
                    fact.occurred_end,
                    json.dumps(fact.metadata or {}),
                    json.dumps(fact.tags),
                    fact.created_at or time.time(),
                ),
            )
        self._conn.commit()

    def search(
        self,
        bank_id: str,
        query_embedding: list[float],
        limit: int = 20,
        fact_types: list[str] | None = None,
    ) -> list[Fact]:
        """Vector similarity search using Python-side cosine similarity."""
        assert self._conn is not None

        query = "SELECT * FROM memory_units WHERE bank_id = ?"
        params: list = [bank_id]

        if fact_types:
            placeholders = ",".join("?" for _ in fact_types)
            query += f" AND fact_type IN ({placeholders})"
            params.extend(fact_types)

        rows = self._conn.execute(query, params).fetchall()

        scored: list[tuple[float, Fact]] = []
        for row in rows:
            fact = _row_to_fact(row)
            if fact.embedding is None:
                continue
            score = cosine_similarity(query_embedding, fact.embedding)
            scored.append((score, fact))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [fact for _, fact in scored[:limit]]

    def count_facts(self, bank_id: str) -> int:
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT COUNT(*) FROM memory_units WHERE bank_id = ?", (bank_id,)
        ).fetchone()
        return row[0] if row else 0


def _encode_embedding(embedding: list[float]) -> bytes:
    """Encode embedding as compact binary (4 bytes per float)."""
    import struct

    return struct.pack(f"{len(embedding)}f", *embedding)


def _decode_embedding(data: bytes) -> list[float]:
    """Decode embedding from binary."""
    import struct

    n = len(data) // 4
    return list(struct.unpack(f"{n}f", data))


def _row_to_fact(row: tuple) -> Fact:
    """Convert a database row to a Fact."""
    return Fact(
        id=row[0],
        bank_id=row[1],
        text=row[2],
        fact_type=row[3],
        embedding=_decode_embedding(row[4]) if row[4] else None,
        entities=json.loads(row[5]) if row[5] else [],
        context=row[6],
        occurred_start=row[7],
        occurred_end=row[8],
        metadata=json.loads(row[9]) if row[9] else {},
        tags=json.loads(row[10]) if row[10] else [],
        created_at=row[11],
    )
