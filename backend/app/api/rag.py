"""RAG API endpoint."""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.auth import get_current_user
from ..models.user import User
from ..schemas.document import RAGRequest, RAGResponse, RAGSourceResponse
from ..services.rag_service import answer_question, RAGError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post(
    "/ask",
    response_model=RAGResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question about your documents",
)
def ask_question(
    request: RAGRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Answer a question using RAG over the authenticated user's documents.

    Retrieves relevant chunks from the user's documents, constructs context,
    and generates a grounded answer using the configured LLM provider.

    If no relevant context is found, returns a controlled response without
    calling the LLM (preventing hallucination).
    """
    try:
        rag_response = answer_question(
            db=db,
            user_id=current_user.id,
            question=request.question,
            document_id=request.document_id,
            top_k=request.top_k,
        )
    except RAGError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Unhandled error in RAG endpoint")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your question.",
        )

    return RAGResponse(
        answer=rag_response.answer,
        sources=[
            RAGSourceResponse(
                document_id=s.document_id,
                filename=s.filename,
                chunk_id=s.chunk_id,
                chunk_index=s.chunk_index,
                page_start=s.page_start,
                page_end=s.page_end,
                similarity_score=s.similarity_score,
            )
            for s in rag_response.sources
        ],
        grounded=rag_response.grounded,
        retrieval_count=rag_response.retrieval_count,
        model=rag_response.model,
        provider=rag_response.provider,
    )
