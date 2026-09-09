from sqlalchemy import Column, String, DateTime, Integer, ForeignKey, Text, CheckConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..core.database import Base


# Valid message roles
VALID_ROLES = ("user", "assistant")


class Message(Base):
    """A single message within a conversation.

    Messages are ordered by id (deterministic, stable).
    Role is constrained to 'user' or 'assistant'.
    """

    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(
        Integer,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    sources = relationship(
        "MessageSource",
        back_populates="message",
        cascade="all, delete-orphan",
        order_by="(MessageSource.chunk_index, MessageSource.id)",
    )

    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant')", name="ck_messages_role"),
    )

    def __repr__(self):
        preview = self.content[:40].replace("\n", "\\n") if self.content else ""
        return (
            f"<Message(id={self.id}, conv={self.conversation_id}, "
            f"role={self.role!r}, content={preview!r})>"
        )
