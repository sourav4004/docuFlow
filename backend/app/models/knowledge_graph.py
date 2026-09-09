"""Knowledge graph entity and relationship models."""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Float, Boolean
from sqlalchemy.orm import relationship
from ..core.database import Base


class Entity(Base):
    """Knowledge graph entity."""
    __tablename__ = "entities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False)
    entity_type = Column(String(100), nullable=False)  # person, company, project, concept, etc.
    aliases = Column(Text, nullable=True)  # JSON array of alternative names
    # Phase 16 — canonicalization (no destructive auto-merge)
    normalized_name = Column(String(255), nullable=True)
    source_count = Column(Integer, nullable=False, default=1)
    metadata_json = Column(Text, nullable=True)  # Additional metadata
    confidence = Column(Float, default=1.0)  # entity 2.0: extraction confidence
    first_seen_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_seen_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    workspace = relationship("Workspace")
    source_entities = relationship("EntityRelationship", foreign_keys="EntityRelationship.source_id", back_populates="source")
    target_entities = relationship("EntityRelationship", foreign_keys="EntityRelationship.target_id", back_populates="target")

    def __repr__(self):
        return f"<Entity {self.name} ({self.entity_type})>"


class EntityRelationship(Base):
    """Relationship between entities."""
    __tablename__ = "entity_relationships"

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    source_id = Column(Integer, ForeignKey("entities.id", ondelete="CASCADE"), nullable=False)
    target_id = Column(Integer, ForeignKey("entities.id", ondelete="CASCADE"), nullable=False)
    relationship_type = Column(String(100), nullable=False)  # works_at, acquired, belongs_to, etc.
    confidence = Column(Float, default=1.0)
    source_document_id = Column(Integer, ForeignKey("documents.id", ondelete="SET NULL"), nullable=True)
    source_chunk_id = Column(Integer, nullable=True)
    # Phase 16 — validity window + observation time
    valid_from = Column(DateTime(timezone=True), nullable=True)
    valid_until = Column(DateTime(timezone=True), nullable=True)
    observed_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    is_current = Column(Boolean, nullable=False, default=True)
    metadata_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    workspace = relationship("Workspace")
    source = relationship("Entity", foreign_keys=[source_id], back_populates="source_entities")
    target = relationship("Entity", foreign_keys=[target_id], back_populates="target_entities")
    source_document = relationship("Document")

    def __repr__(self):
        return f"<EntityRelationship {self.source_id}->{self.target_id} ({self.relationship_type})>"
