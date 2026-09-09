"""Conversation context service.

Loads recent conversation history for use in conversation-aware RAG.

Design decisions:
- Bounded history: configurable max messages (default 10)
- Chronological ordering: messages returned in creation order
- Current message excluded: the just-posted user message is not included
- Ownership not enforced here: the calling layer (API) enforces ownership
- No LLM calls: this is a pure data-loading service
- Deterministic: same conversation + same limit = same output
"""

import logging
from typing import List, Optional
from dataclasses import dataclass

from sqlalchemy.orm import Session

from ..models.message import Message

logger = logging.getLogger(__name__)

# Default maximum number of history messages to include
DEFAULT_MAX_HISTORY_MESSAGES = 10


@dataclass
class HistoryMessage:
    """A single message from conversation history."""
    role: str
    content: str


def load_conversation_history(
    db: Session,
    conversation_id: int,
    exclude_message_id: Optional[int] = None,
    max_messages: int = DEFAULT_MAX_HISTORY_MESSAGES,
    max_chars: Optional[int] = None,
) -> List[HistoryMessage]:
    """Load recent conversation history messages.

    Returns the most recent `max_messages` messages in chronological order,
    optionally excluding a specific message. Respects a character budget
    by preferring recent messages that fit within the limit.

    Args:
        db: SQLAlchemy session.
        conversation_id: The conversation to load history from.
        exclude_message_id: If provided, exclude this message from results.
        max_messages: Maximum number of history messages to return.
        max_chars: Optional character budget. If set, messages are included
            from newest to oldest until the budget is exhausted.

    Returns:
        List of HistoryMessage in chronological order (oldest first).
    """
    query = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
    )

    if exclude_message_id is not None:
        query = query.filter(Message.id != exclude_message_id)

    # Get the most recent messages, then reverse to chronological order
    recent_messages = (
        query
        .order_by(Message.id.desc())
        .limit(max_messages)
        .all()
    )

    # Reverse to chronological order (oldest first)
    recent_messages.reverse()

    history = [
        HistoryMessage(role=msg.role, content=msg.content)
        for msg in recent_messages
    ]

    # Apply character budget: prefer recent messages
    if max_chars is not None and max_chars > 0:
        history = _apply_char_budget(history, max_chars)

    return history


def _apply_char_budget(
    history: List[HistoryMessage],
    max_chars: int,
) -> List[HistoryMessage]:
    """Apply a character budget to history, keeping most recent messages.

    Iterates from newest to oldest, adding messages until the budget
    is exhausted. Returns the result in chronological order.

    Args:
        history: List of HistoryMessage in chronological order.
        max_chars: Maximum total character count.

    Returns:
        Trimmed list in chronological order.
    """
    if not history:
        return []

    # Iterate from newest to oldest, collecting messages that fit
    selected: List[HistoryMessage] = []
    total = 0
    for msg in reversed(history):
        msg_len = len(msg.content)
        if total + msg_len > max_chars and selected:
            break
        selected.append(msg)
        total += msg_len

    # Reverse to chronological order
    selected.reverse()
    return selected


def format_history_for_prompt(history: List[HistoryMessage]) -> str:
    """Format conversation history for inclusion in the RAG prompt.

    Produces a clear, readable format that the LLM can use to understand
    conversation context.

    Args:
        history: List of HistoryMessage in chronological order.

    Returns:
        Formatted history string. Empty if no history.
    """
    if not history:
        return ""

    lines = []
    for msg in history:
        label = "User" if msg.role == "user" else "Assistant"
        lines.append(f"{label}: {msg.content}")

    return "\n".join(lines)
