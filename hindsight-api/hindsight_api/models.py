"""
SQLAlchemy models for the memory system.
"""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID as PyUUID


@dataclass
class RequestContext:
    """
    Context for request authentication and authorization.

    This dataclass carries authentication data from HTTP requests to the
    memory engine operations. It can be extended to include additional
    context like headers, tokens, user info, etc.
    """

    api_key: str | None = None
    api_key_id: str | None = None  # UUID of the API key used for authentication
    tenant_id: str | None = None  # Tenant identifier (set by extension after auth)
    internal: bool = False  # True for background/internal operations (not user-visible)


from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    Text,
    func,
)
from sqlalchemy import (
    text as sql_text,
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from .config import EMBEDDING_DIMENSION


class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all models."""

    pass


class Document(Base):
    """Source documents for memory units."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    bank_id: Mapped[str] = mapped_column(Text, primary_key=True)
    original_text: Mapped[str | None] = mapped_column(Text)
    content_hash: Mapped[str | None] = mapped_column(Text)
    doc_metadata: Mapped[dict] = mapped_column("metadata", JSONB, server_default=sql_text("'{}'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())

    # Relationships
    memory_units = relationship("MemoryUnit", back_populates="document", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_documents_bank_id", "bank_id"),
        Index("idx_documents_content_hash", "content_hash"),
    )


class MemoryUnit(Base):
    """Individual sentence-level memories."""

    __tablename__ = "memory_units"

    id: Mapped[PyUUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sql_text("gen_random_uuid()")
    )
    bank_id: Mapped[str] = mapped_column(Text, nullable=False)
    document_id: Mapped[str | None] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding = mapped_column(Vector(EMBEDDING_DIMENSION))  # pgvector type
    context: Mapped[str | None] = mapped_column(Text)
    event_date: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )  # Kept for backward compatibility
    occurred_start: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True)
    )  # When fact occurred (range start)
    occurred_end: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))  # When fact occurred (range end)
    mentioned_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))  # When fact was mentioned
    fact_type: Mapped[str] = mapped_column(Text, nullable=False, server_default="world")
    confidence_score: Mapped[float | None] = mapped_column(Float)
    access_count: Mapped[int] = mapped_column(Integer, server_default="0")
    unit_metadata: Mapped[dict] = mapped_column(
        "metadata", JSONB, server_default=sql_text("'{}'::jsonb")
    )  # User-defined metadata (str->str)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())

    # Relationships
    document = relationship("Document", back_populates="memory_units")
    unit_entities = relationship("UnitEntity", back_populates="memory_unit", cascade="all, delete-orphan")
    outgoing_links = relationship(
        "MemoryLink", foreign_keys="MemoryLink.from_unit_id", back_populates="from_unit", cascade="all, delete-orphan"
    )
    incoming_links = relationship(
        "MemoryLink", foreign_keys="MemoryLink.to_unit_id", back_populates="to_unit", cascade="all, delete-orphan"
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["document_id", "bank_id"],
            ["documents.id", "documents.bank_id"],
            name="memory_units_document_fkey",
            ondelete="CASCADE",
        ),
        CheckConstraint("fact_type IN ('world', 'experience', 'opinion', 'observation')"),
        CheckConstraint("confidence_score IS NULL OR (confidence_score >= 0.0 AND confidence_score <= 1.0)"),
        CheckConstraint(
            "(fact_type = 'opinion' AND confidence_score IS NOT NULL) OR "
            "(fact_type = 'observation') OR "
            "(fact_type NOT IN ('opinion', 'observation') AND confidence_score IS NULL)",
            name="confidence_score_fact_type_check",
        ),
        Index("idx_memory_units_bank_id", "bank_id"),
        Index("idx_memory_units_document_id", "document_id"),
        Index("idx_memory_units_event_date", "event_date", postgresql_ops={"event_date": "DESC"}),
        Index("idx_memory_units_bank_date", "bank_id", "event_date", postgresql_ops={"event_date": "DESC"}),
        Index("idx_memory_units_access_count", "access_count", postgresql_ops={"access_count": "DESC"}),
        Index("idx_memory_units_fact_type", "fact_type"),
        Index("idx_memory_units_bank_fact_type", "bank_id", "fact_type"),
        Index(
            "idx_memory_units_bank_type_date",
            "bank_id",
            "fact_type",
            "event_date",
            postgresql_ops={"event_date": "DESC"},
        ),
        Index(
            "idx_memory_units_opinion_confidence",
            "bank_id",
            "confidence_score",
            postgresql_where=sql_text("fact_type = 'opinion'"),
            postgresql_ops={"confidence_score": "DESC"},
        ),
        Index(
            "idx_memory_units_opinion_date",
            "bank_id",
            "event_date",
            postgresql_where=sql_text("fact_type = 'opinion'"),
            postgresql_ops={"event_date": "DESC"},
        ),
        Index(
            "idx_memory_units_observation_date",
            "bank_id",
            "event_date",
            postgresql_where=sql_text("fact_type = 'observation'"),
            postgresql_ops={"event_date": "DESC"},
        ),
        Index(
            "idx_memory_units_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class Entity(Base):
    """Resolved entities (people, organizations, locations, etc.)."""

    __tablename__ = "entities"

    id: Mapped[PyUUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=sql_text("gen_random_uuid()")
    )
    canonical_name: Mapped[str] = mapped_column(Text, nullable=False)
    bank_id: Mapped[str] = mapped_column(Text, nullable=False)
    entity_metadata: Mapped[dict] = mapped_column("metadata", JSONB, server_default=sql_text("'{}'::jsonb"))
    first_seen: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())
    last_seen: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())
    mention_count: Mapped[int] = mapped_column(Integer, server_default="1")

    # Relationships
    unit_entities = relationship("UnitEntity", back_populates="entity", cascade="all, delete-orphan")
    memory_links = relationship("MemoryLink", back_populates="entity", cascade="all, delete-orphan")
    cooccurrences_1 = relationship(
        "EntityCooccurrence",
        foreign_keys="EntityCooccurrence.entity_id_1",
        back_populates="entity_1",
        cascade="all, delete-orphan",
    )
    cooccurrences_2 = relationship(
        "EntityCooccurrence",
        foreign_keys="EntityCooccurrence.entity_id_2",
        back_populates="entity_2",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("idx_entities_bank_id", "bank_id"),
        Index("idx_entities_canonical_name", "canonical_name"),
        Index("idx_entities_bank_name", "bank_id", "canonical_name"),
    )


class UnitEntity(Base):
    """Association between memory units and entities."""

    __tablename__ = "unit_entities"

    unit_id: Mapped[PyUUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memory_units.id", ondelete="CASCADE"), primary_key=True
    )
    entity_id: Mapped[PyUUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("entities.id", ondelete="CASCADE"), primary_key=True
    )

    # Relationships
    memory_unit = relationship("MemoryUnit", back_populates="unit_entities")
    entity = relationship("Entity", back_populates="unit_entities")

    __table_args__ = (
        Index("idx_unit_entities_unit", "unit_id"),
        Index("idx_unit_entities_entity", "entity_id"),
    )


class EntityCooccurrence(Base):
    """Materialized cache of entity co-occurrences."""

    __tablename__ = "entity_cooccurrences"

    entity_id_1: Mapped[PyUUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("entities.id", ondelete="CASCADE"), primary_key=True
    )
    entity_id_2: Mapped[PyUUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("entities.id", ondelete="CASCADE"), primary_key=True
    )
    cooccurrence_count: Mapped[int] = mapped_column(Integer, server_default="1")
    last_cooccurred: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())

    # Relationships
    entity_1 = relationship("Entity", foreign_keys=[entity_id_1], back_populates="cooccurrences_1")
    entity_2 = relationship("Entity", foreign_keys=[entity_id_2], back_populates="cooccurrences_2")

    __table_args__ = (
        CheckConstraint("entity_id_1 < entity_id_2", name="entity_cooccurrence_order_check"),
        Index("idx_entity_cooccurrences_entity1", "entity_id_1"),
        Index("idx_entity_cooccurrences_entity2", "entity_id_2"),
        Index("idx_entity_cooccurrences_count", "cooccurrence_count", postgresql_ops={"cooccurrence_count": "DESC"}),
    )


class MemoryLink(Base):
    """Links between memory units (temporal, semantic, entity)."""

    __tablename__ = "memory_links"

    from_unit_id: Mapped[PyUUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memory_units.id", ondelete="CASCADE"), primary_key=True
    )
    to_unit_id: Mapped[PyUUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memory_units.id", ondelete="CASCADE"), primary_key=True
    )
    link_type: Mapped[str] = mapped_column(Text, primary_key=True)
    entity_id: Mapped[PyUUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("entities.id", ondelete="CASCADE"), primary_key=True
    )
    weight: Mapped[float] = mapped_column(Float, nullable=False, server_default="1.0")
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())

    # Relationships
    from_unit = relationship("MemoryUnit", foreign_keys=[from_unit_id], back_populates="outgoing_links")
    to_unit = relationship("MemoryUnit", foreign_keys=[to_unit_id], back_populates="incoming_links")
    entity = relationship("Entity", back_populates="memory_links")

    __table_args__ = (
        CheckConstraint(
            "link_type IN ('temporal', 'semantic', 'entity', 'causes', 'caused_by', 'enables', 'prevents')",
            name="memory_links_link_type_check",
        ),
        CheckConstraint("weight >= 0.0 AND weight <= 1.0", name="memory_links_weight_check"),
        Index("idx_memory_links_from", "from_unit_id"),
        Index("idx_memory_links_to", "to_unit_id"),
        Index("idx_memory_links_type", "link_type"),
        Index("idx_memory_links_entity", "entity_id", postgresql_where=sql_text("entity_id IS NOT NULL")),
        Index(
            "idx_memory_links_from_weight",
            "from_unit_id",
            "weight",
            postgresql_where=sql_text("weight >= 0.1"),
            postgresql_ops={"weight": "DESC"},
        ),
    )


class Bank(Base):
    """Memory bank profiles with disposition traits and background."""

    __tablename__ = "banks"

    bank_id: Mapped[str] = mapped_column(Text, primary_key=True)
    disposition: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=sql_text('\'{"skepticism": 3, "literalism": 3, "empathy": 3}\'::jsonb')
    )
    background: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())

    __table_args__ = (Index("idx_banks_bank_id", "bank_id"),)
