from sqlalchemy import Column, String, DateTime, Integer, ForeignKey, Table, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..core.database import Base


# Association table for many-to-many: collections <-> documents
collection_documents = Table(
    "collection_documents",
    Base.metadata,
    Column("collection_id", Integer, ForeignKey("collections.id", ondelete="CASCADE"), primary_key=True),
    Column("document_id", Integer, ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True),
)


class Collection(Base):
    """A workspace/collection grouping documents for scoped RAG queries.

    One user → many collections.
    One collection → many documents (many-to-many).
    """

    __tablename__ = "collections"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Optional workspace scope (added Phase 14). Legacy collections have NULL
    # workspace_id and remain user-scoped. Workspace-scoped collections let
    # enterprise features (health, copilots, reports) operate per workspace.
    workspace_id = Column(
        Integer,
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    name = Column(String(255), nullable=False)
    description = Column(String(500), nullable=True, default="")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    user = relationship("User", backref="collections")
    workspace = relationship("Workspace", backref="collections")
    documents = relationship(
        "Document",
        secondary=collection_documents,
        backref="collections",
        lazy="select",
    )

    __table_args__ = (
        Index("ix_collections_user_name", "user_id", "name"),
    )

    def __repr__(self):
        return f"<Collection(id={self.id}, user_id={self.user_id}, name={self.name!r})>"
