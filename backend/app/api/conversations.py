"""Conversation API endpoints."""

import json
import logging
import re
from datetime import datetime, timezone
from urllib.parse import quote
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import PlainTextResponse, StreamingResponse
from sqlalchemy.orm import Session, subqueryload
from sqlalchemy.sql import func as sqlfunc

from ..core.database import get_db
from ..core.auth import get_current_user
from ..core.config import settings
from ..models.user import User
from ..models.conversation import Conversation
from ..models.message import Message, VALID_ROLES
from ..schemas.conversation import (
    ConversationCreate,
    ConversationUpdate,
    ConversationResponse,
    ConversationDetailResponse,
    ConversationListResponse,
    MessageResponse,
    MessageCreate,
    SendMessageResponse,
    SourceInfo,
    PaginatedMessageResponse,
)
from ..schemas.auth import MessageResponse as SuccessMessageResponse
from ..services.rag_service import answer_question_with_history, answer_question_with_history_streaming, RAGError, _build_sources, _compute_confidence
from ..services.llm.service import LLMService
from ..services.retrieval_service import retrieve_context
from ..services.conversation_context import load_conversation_history, format_history_for_prompt
from ..services.source_service import persist_sources
from ..services.rag_prompt import get_system_prompt, build_conversation_aware_user_prompt
from ..models.document import Document

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["conversations"])


# ---------------------------------------------------------------------------
# Helper: get conversation with ownership check
# ---------------------------------------------------------------------------

def _get_owned_conversation(
    db: Session,
    user_id: int,
    conversation_id: int,
) -> Conversation:
    """Retrieve a conversation that belongs to the user.

    Returns 404 if not found (consistent with existing security model).
    """
    conv = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == user_id,
    ).first()

    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    return conv


# ---------------------------------------------------------------------------
# POST /conversations — Create
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new conversation",
)
def create_conversation(
    request: ConversationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new conversation for the authenticated user.

    The authenticated user becomes the owner. user_id is never accepted from client.
    """
    conversation = Conversation(
        user_id=current_user.id,
        title=request.title.strip(),
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    return conversation


# ---------------------------------------------------------------------------
# GET /conversations — List
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=ConversationListResponse,
    status_code=status.HTTP_200_OK,
    summary="List authenticated user's conversations",
)
def list_conversations(
    limit: int = Query(20, ge=1, le=100, description="Number of items to return"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
    search: str = Query(None, max_length=200, description="Search conversation titles (case-insensitive, partial match)"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return paginated conversations belonging to the authenticated user.

    Ordered by updated_at DESC, then id DESC for deterministic ordering.
    Optional search parameter filters by conversation title (case-insensitive partial match).
    """
    query = db.query(Conversation).filter(Conversation.user_id == current_user.id)

    # Apply title search if provided
    if search and search.strip():
        query = query.filter(Conversation.title.ilike(f"%{search.strip()}%"))

    total = query.count()
    items = (
        query
        .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    has_next = offset + limit < total
    has_previous = offset > 0

    return ConversationListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        has_next=has_next,
        has_previous=has_previous,
    )


# ---------------------------------------------------------------------------
# GET /conversations/{conversation_id} — Get
# ---------------------------------------------------------------------------

@router.get(
    "/{conversation_id}",
    response_model=ConversationDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Get a conversation with messages",
)
def get_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return a conversation with its messages.

    Ownership is enforced. Returns 404 if not found or unauthorized.
    Messages have their sources eagerly loaded to prevent N+1 queries.
    """
    conv = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id,
        )
        .options(
            subqueryload(Conversation.messages).subqueryload(Message.sources)
        )
        .first()
    )
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    return conv


# ---------------------------------------------------------------------------
# PATCH /conversations/{conversation_id} — Update
# ---------------------------------------------------------------------------

@router.patch(
    "/{conversation_id}",
    response_model=ConversationResponse,
    status_code=status.HTTP_200_OK,
    summary="Update a conversation",
)
def update_conversation(
    conversation_id: int,
    request: ConversationUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update a conversation's title.

    Ownership is enforced. Only title can be changed.
    """
    conv = _get_owned_conversation(db, current_user.id, conversation_id)
    conv.title = request.title.strip()
    db.commit()
    db.refresh(conv)
    return conv


# ---------------------------------------------------------------------------
# DELETE /conversations/{conversation_id} — Delete
# ---------------------------------------------------------------------------

@router.delete(
    "/{conversation_id}",
    response_model=SuccessMessageResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete a conversation",
)
def delete_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a conversation and all its messages (cascade).

    Ownership is enforced. Returns 404 if not found or unauthorized.
    """
    conv = _get_owned_conversation(db, current_user.id, conversation_id)
    db.delete(conv)
    db.commit()
    return SuccessMessageResponse(message="Conversation deleted successfully")


# ---------------------------------------------------------------------------
# GET /conversations/{conversation_id}/export — Export as Markdown
# ---------------------------------------------------------------------------

@router.get(
    "/{conversation_id}/export",
    response_class=PlainTextResponse,
    status_code=status.HTTP_200_OK,
    summary="Export conversation as Markdown",
)
def export_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Export a conversation and its messages as a downloadable Markdown file.

    Ownership is enforced. Returns a plain text response with
    Content-Disposition header for browser download.
    """
    conv = _get_owned_conversation(db, current_user.id, conversation_id)

    # Load all messages for this conversation with sources eagerly loaded
    # to avoid N+1 queries when accessing msg.sources in the loop
    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.id)
        .options(subqueryload(Message.sources))
        .all()
    )

    # Build Markdown content
    lines = []
    lines.append(f"# {conv.title}")
    lines.append("")
    lines.append(f"*Exported on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*")
    lines.append("")
    lines.append("---")
    lines.append("")

    for msg in messages:
        label = "**You**" if msg.role == "user" else "**DocuFlow**"
        timestamp = msg.created_at.strftime("%H:%M") if msg.created_at else ""
        lines.append(f"### {label} — {timestamp}")
        lines.append("")
        lines.append(msg.content)
        lines.append("")

        # Include source references if present
        if msg.sources:
            lines.append("*Sources:*")
            for src in msg.sources:
                doc_label = f"Document #{src.document_id}" if src.document_id else "Unknown"
                page_info = f", pages {src.page_start}-{src.page_end}" if src.page_start else ""
                chunk_info = f", chunk {src.chunk_index}" if src.chunk_index is not None else ""
                score_info = f", score {src.similarity_score:.2f}" if src.similarity_score is not None else ""
                lines.append(f"- {doc_label}{page_info}{chunk_info}{score_info}")
            lines.append("")

        lines.append("---")
        lines.append("")

    content = "\n".join(lines)

    # Sanitize title for filename
    # Strip CR/LF, path separators, and other dangerous characters
    sanitized = re.sub(r'[\\/:*?"<>|\r\n\x00]', '', conv.title)
    sanitized = sanitized.strip()[:50] or "conversation"
    # Replace whitespace runs with underscores
    safe_title = re.sub(r'\s+', '_', sanitized)
    # Remove any remaining non-ASCII chars for the basic filename
    ascii_title = re.sub(r'[^a-zA-Z0-9_-]', '', safe_title) or "conversation"
    filename = f"{ascii_title}_{conv.id}.md"

    # RFC 5987 encoded filename for Unicode support
    encoded_title = quote(safe_title[:50], safe='_-~')
    encoded_filename = f"{encoded_title}_{conv.id}.md"

    return PlainTextResponse(
        content=content,
        media_type="text/markdown",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{filename}"; '
                f"filename*=UTF-8''{encoded_filename}"
            ),
        },
    )


# ---------------------------------------------------------------------------
# GET /conversations/{conversation_id}/messages — List messages (paginated)
# ---------------------------------------------------------------------------

DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200


@router.get(
    "/{conversation_id}/messages",
    response_model=PaginatedMessageResponse,
    status_code=status.HTTP_200_OK,
    summary="List messages in a conversation (paginated)",
)
def list_messages(
    conversation_id: int,
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE, description="Messages per page"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return paginated messages for a conversation owned by the authenticated user.

    Messages are ordered by id (deterministic chronological order).
    Ownership is enforced on the conversation.

    Pagination is performed at the database level using COUNT + LIMIT/OFFSET.
    """
    _get_owned_conversation(db, current_user.id, conversation_id)

    # Total count query
    total = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .count()
    )

    # Paginated query with sources eagerly loaded to prevent N+1
    offset = (page - 1) * page_size
    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.id)
        .options(subqueryload(Message.sources))
        .offset(offset)
        .limit(page_size)
        .all()
    )

    has_next = offset + page_size < total
    has_previous = page > 1

    return PaginatedMessageResponse(
        messages=messages,
        page=page,
        page_size=page_size,
        total=total,
        has_next=has_next,
        has_previous=has_previous,
    )


# ---------------------------------------------------------------------------
# GET /conversations/{conversation_id}/messages/search — Search messages
# ---------------------------------------------------------------------------

@router.get(
    "/{conversation_id}/messages/search",
    response_model=PaginatedMessageResponse,
    status_code=status.HTTP_200_OK,
    summary="Search messages within a conversation",
)
def search_messages(
    conversation_id: int,
    q: str = Query(..., min_length=1, max_length=200, description="Search query"),
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE, description="Results per page"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Search messages within a conversation by content (case-insensitive partial match).

    Ownership is enforced. Returns 404 if conversation not found.
    """
    _get_owned_conversation(db, current_user.id, conversation_id)

    search_term = q.strip()
    if not search_term:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search query cannot be empty or whitespace-only",
        )

    base_query = (
        db.query(Message)
        .filter(
            Message.conversation_id == conversation_id,
            Message.content.ilike(f"%{search_term}%"),
        )
    )

    total = base_query.count()
    offset = (page - 1) * page_size
    messages = (
        base_query
        .order_by(Message.id)
        .options(subqueryload(Message.sources))
        .offset(offset)
        .limit(page_size)
        .all()
    )

    has_next = offset + page_size < total
    has_previous = page > 1

    return PaginatedMessageResponse(
        messages=messages,
        page=page,
        page_size=page_size,
        total=total,
        has_next=has_next,
        has_previous=has_previous,
    )


# ---------------------------------------------------------------------------
# POST /conversations/{conversation_id}/messages — Send message + RAG
# ---------------------------------------------------------------------------

@router.post(
    "/{conversation_id}/messages",
    response_model=SendMessageResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Send a message and get a RAG-powered response",
)
def send_message(
    conversation_id: int,
    request: MessageCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Send a user message and generate an assistant response via RAG.

    Flow:
    1. Verify conversation ownership
    2. Validate document ownership if document_id supplied
    3. Create user message
    4. Load conversation history (excluding current message)
    5. Run conversation-aware RAG pipeline
    6. Create assistant message
    7. Return both messages with sources

    If RAG fails, the user message is persisted but no fake assistant
    message is created. The error is returned to the client.
    """
    # 1. Verify conversation ownership
    conv = _get_owned_conversation(db, current_user.id, conversation_id)

    # 2. Validate document ownership if document_id supplied
    if request.document_id is not None:
        doc = db.query(Document).filter(
            Document.id == request.document_id,
            Document.user_id == current_user.id,
        ).first()
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found",
            )

    # 3. Create user message
    user_msg = Message(
        conversation_id=conversation_id,
        role="user",
        content=request.content.strip(),
    )
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)

    # 4. Load conversation history (excluding the just-created user message)
    history_messages = load_conversation_history(
        db=db,
        conversation_id=conversation_id,
        exclude_message_id=user_msg.id,
        max_messages=settings.max_history_messages,
        max_chars=settings.max_history_chars,
    )
    conversation_history = format_history_for_prompt(history_messages)

    # 5. Run conversation-aware RAG pipeline with optional controls
    try:
        rag_kwargs = {
            "db": db,
            "user_id": current_user.id,
            "question": request.content.strip(),
            "conversation_history": conversation_history,
            "document_id": request.document_id,
        }
        if request.collection_id is not None:
            rag_kwargs["collection_id"] = request.collection_id
        if request.top_k is not None:
            rag_kwargs["top_k"] = request.top_k
        if request.min_similarity is not None:
            rag_kwargs["min_similarity"] = request.min_similarity

        rag_result = answer_question_with_history(**rag_kwargs)
    except RAGError as exc:
        logger.error("RAG failed for conversation %d: %s", conversation_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate a response. Please try again.",
        )
    except Exception as exc:
        logger.exception("Unhandled error in message endpoint")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your message.",
        )

    # 5. Create assistant message
    assistant_msg = Message(
        conversation_id=conversation_id,
        role="assistant",
        content=rag_result.answer,
    )
    db.add(assistant_msg)
    db.flush()  # Get assistant_msg.id before creating sources

    # 6. Persist RAG sources as MessageSource records
    source_records = persist_sources(
        db=db,
        message_id=assistant_msg.id,
        sources=rag_result.sources,
        owner_user_id=current_user.id,
    )

    # 7. Update conversation timestamp
    conv.updated_at = sqlfunc.now()

    db.commit()
    db.refresh(assistant_msg)
    db.refresh(conv)

    # 8. Build response
    return SendMessageResponse(
        user_message=MessageResponse.model_validate(user_msg),
        assistant_message=MessageResponse.model_validate(assistant_msg),
        sources=[
            SourceInfo(
                document_id=s.document_id,
                filename=s.filename,
                chunk_id=s.chunk_id,
                chunk_index=s.chunk_index,
                page_start=s.page_start,
                page_end=s.page_end,
                similarity_score=s.similarity_score,
            )
            for s in rag_result.sources
        ],
        grounded=rag_result.grounded,
    )


# ---------------------------------------------------------------------------
# POST /conversations/{conversation_id}/messages/stream — Streaming RAG
# ---------------------------------------------------------------------------

@router.post(
    "/{conversation_id}/messages/stream",
    status_code=status.HTTP_200_OK,
    summary="Send a message and stream the RAG response",
)
def send_message_stream(
    conversation_id: int,
    request: MessageCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Stream a RAG response using Server-Sent Events (SSE).

    Flow:
    1. Verify conversation ownership
    2. Validate document ownership if document_id supplied
    3. Create user message
    4. Load conversation history
    5. Stream LLM tokens via SSE
    6. Accumulate full response server-side
    7. Persist assistant message after completion
    8. Persist MessageSource records
    9. Send completion event with message_id

    SSE Event Types:
    - token: {"text": "..."}
    - sources: {"sources": [...], "confidence": {...}}
    - complete: {"message_id": ..., "grounded": ...}
    - error: {"message": "..."}
    """
    # 1. Verify conversation ownership
    conv = _get_owned_conversation(db, current_user.id, conversation_id)

    # 2. Validate document ownership if document_id supplied
    if request.document_id is not None:
        doc = db.query(Document).filter(
            Document.id == request.document_id,
            Document.user_id == current_user.id,
        ).first()
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found",
            )

    # 3. Create user message
    user_msg = Message(
        conversation_id=conversation_id,
        role="user",
        content=request.content.strip(),
    )
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)

    # 4. Load conversation history
    history_messages = load_conversation_history(
        db=db,
        conversation_id=conversation_id,
        exclude_message_id=user_msg.id,
        max_messages=settings.max_history_messages,
        max_chars=settings.max_history_chars,
    )
    conversation_history = format_history_for_prompt(history_messages)

    def event_stream():
        """Generate SSE events for the streaming RAG response."""
        accumulated_text = ""
        assistant_msg_id = None

        try:
            # We need to do retrieval to get sources, but streaming the LLM
            # For now, we'll do retrieval synchronously, then stream LLM
            try:
                retrieval = retrieve_context(
                    db=db,
                    user_id=current_user.id,
                    query=request.content.strip(),
                    document_id=request.document_id,
                )
            except Exception as e:
                logger.error("Retrieval failed during streaming: %s", e)
                yield f"event: error\ndata: {json.dumps({'message': 'Retrieval failed'})}\n\n"
                return

            # Build RAG prompt with retrieved context
            system_prompt = get_system_prompt()
            user_prompt = build_conversation_aware_user_prompt(
                question=request.content.strip(),
                context=retrieval.context,
                conversation_history=conversation_history,
            )

            # Stream LLM generation
            llm_service = LLMService()
            try:
                for chunk in llm_service.stream_generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                ):
                    if chunk:
                        accumulated_text += chunk
                        yield f"event: token\ndata: {json.dumps({'text': chunk})}\n\n"
            except Exception as e:
                logger.error("LLM streaming failed: %s", e)
                yield f"event: error\ndata: {json.dumps({'message': 'Failed to generate response'})}\n\n"
                return

            # Build sources
            sources = _build_sources(retrieval)
            confidence = _compute_confidence(retrieval, sources)

            # Send sources event
            sources_data = [
                {
                    "document_id": s.document_id,
                    "filename": s.filename,
                    "chunk_id": s.chunk_id,
                    "chunk_index": s.chunk_index,
                    "page_start": s.page_start,
                    "page_end": s.page_end,
                    "similarity_score": s.similarity_score,
                }
                for s in sources
            ]
            yield f"event: sources\ndata: {json.dumps({'sources': sources_data, 'confidence': {'level': confidence.level, 'grounding_score': confidence.grounding_score}})}\n\n"

            # Persist assistant message
            answer_text = accumulated_text.strip()
            if not answer_text:
                answer_text = "I don't have enough information in the provided documents to answer this question."

            assistant_msg = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=answer_text,
            )
            db.add(assistant_msg)
            db.flush()

            # Persist sources
            persist_sources(
                db=db,
                message_id=assistant_msg.id,
                sources=sources,
                owner_user_id=current_user.id,
            )

            # Update conversation timestamp
            conv.updated_at = sqlfunc.now()
            db.commit()

            assistant_msg_id = assistant_msg.id

        except Exception as exc:
            logger.exception("Error in streaming RAG endpoint")
            try:
                db.rollback()
            except Exception:
                pass
            yield f"event: error\ndata: {json.dumps({'message': 'An error occurred while processing your message.'})}\n\n"
            return

        # Send completion event
        yield f"event: complete\ndata: {json.dumps({'message_id': assistant_msg_id, 'grounded': bool(sources_data)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Type": "text/event-stream",
        },
    )
