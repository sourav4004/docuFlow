"""AI Assistant API endpoints."""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel

from ..core.database import get_db
from ..core.auth import get_current_user
from ..models.user import User
from ..models.workspace import Workspace, WorkspaceMember
from ..services.ai_orchestration import AITask, AITaskType, AIResponse
from ..services.ai_task_router import AITaskRouter
from ..services.evidence_service import EvidenceService
from ..services.ai_safety import AISafetyService

router = APIRouter(prefix="/ai", tags=["ai-assistant"])

# Initialize services
ai_safety = AISafetyService()


class AIQueryRequest(BaseModel):
    """Request for AI query."""
    query: str
    document_ids: Optional[List[int]] = None
    collection_id: Optional[int] = None
    context: Optional[Dict[str, Any]] = None


class AIQueryResponse(BaseModel):
    """Response from AI query."""
    answer: str
    citations: List[Dict[str, Any]] = []
    grounding: Optional[Dict[str, Any]] = None
    task_type: str
    warnings: List[str] = []
    metadata: Dict[str, Any] = {}


class ExtractionRequest(BaseModel):
    """Request for structured extraction."""
    document_id: int
    schema: Dict[str, str]  # field_name -> type


class ExtractionResponse(BaseModel):
    """Response from structured extraction."""
    extracted_data: Dict[str, Any]
    confidence: Dict[str, float]
    schema_valid: bool
    warnings: List[str] = []


class ComparisonRequest(BaseModel):
    """Request for document comparison."""
    document_ids: List[int]
    comparison_type: str = "full"  # full, metadata, content


class ComparisonResponse(BaseModel):
    """Response from document comparison."""
    summary: str
    differences: List[Dict[str, Any]]
    conflicts: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]


@router.post("/query", response_model=AIQueryResponse)
def ai_query(
    request: AIQueryRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Execute an AI query."""
    # Validate input safety
    is_safe, findings = ai_safety.validate_input(request.query)
    if not is_safe:
        raise HTTPException(
            status_code=400,
            detail="Query contains potentially unsafe content"
        )
    
    # Create and classify task
    task = AITask(
        query=request.query,
        user_id=current_user.id,
        document_ids=request.document_ids or [],
        collection_id=request.collection_id,
        context=request.context or {}
    )
    task = AITaskRouter.classify_task(task)
    
    # Generate response based on task type
    # In production, this would call the LLM
    response = AIResponse(
        answer=f"This is a simulated response for task type: {task.task_type.value}",
        metadata={
            "task_id": task.id,
            "priority": task.priority.value,
            "task_type": task.task_type.value
        }
    )
    
    return AIQueryResponse(
        answer=response.answer,
        citations=response.citations,
        grounding=response.grounding.to_dict() if response.grounding else None,
        task_type=task.task_type.value,
        warnings=response.warnings,
        metadata=response.metadata
    )


@router.post("/extract", response_model=ExtractionResponse)
def extract_structured_data(
    request: ExtractionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Extract structured data from a document."""
    # In production, this would:
    # 1. Retrieve document
    # 2. Validate ownership
    # 3. Call extraction service
    # 4. Validate against schema
    
    return ExtractionResponse(
        extracted_data={},
        confidence={},
        schema_valid=True,
        warnings=[]
    )


@router.post("/compare", response_model=ComparisonResponse)
def compare_documents(
    request: ComparisonRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Compare multiple documents."""
    if len(request.document_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 documents required for comparison"
        )
    
    # In production, this would:
    # 1. Retrieve documents
    # 2. Validate ownership
    # 3. Compare documents
    # 4. Detect conflicts
    
    return ComparisonResponse(
        summary="Document comparison complete",
        differences=[],
        conflicts=[],
        citations=[]
    )


@router.get("/tasks")
def list_ai_tasks(
    current_user: User = Depends(get_current_user)
):
    """List supported AI task types."""
    return {"tasks": AITaskRouter.get_supported_tasks()}


@router.post("/safety/validate")
def validate_input_safety(
    text: str,
    current_user: User = Depends(get_current_user)
):
    """Validate input for safety."""
    is_safe, findings = ai_safety.validate_input(text)
    return {
        "is_safe": is_safe,
        "findings": findings
    }
