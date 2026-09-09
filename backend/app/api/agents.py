"""Agent execution API endpoints."""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel

from ..core.database import get_db
from ..core.auth import get_current_user
from ..models.user import User
from ..services.ai_orchestration import AgentPlan, AgentStatus, AITask, AITaskType
from ..services.ai_task_router import AITaskRouter
from ..services.agent_engine import AgentEngine, AgentExecution
from ..services.tool_framework import create_default_tool_registry

router = APIRouter(prefix="/agents", tags=["agents"])

# Initialize agent engine
tool_registry = create_default_tool_registry()
agent_engine = AgentEngine(tool_registry)


class AgentCreateRequest(BaseModel):
    """Request to create an agent."""
    goal: str
    document_ids: Optional[List[int]] = None
    collection_id: Optional[int] = None
    max_steps: int = 10
    timeout_seconds: int = 300


class AgentResponse(BaseModel):
    """Agent execution response."""
    id: str
    status: str
    goal: str
    current_step: int
    steps_completed: int
    max_steps: int
    duration_ms: float
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    approval_required: bool = False


class ApprovalRequest(BaseModel):
    """Approval request."""
    approval_id: str


@router.post("", response_model=AgentResponse)
def create_agent(
    request: AgentCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create and start an agent execution."""
    # Create task
    task = AITask(
        query=request.goal,
        user_id=current_user.id,
        document_ids=request.document_ids or [],
        collection_id=request.collection_id
    )
    task = AITaskRouter.classify_task(task)
    
    # Create plan
    plan = AgentPlan(
        goal=request.goal,
        steps=[
            {"type": "retrieval", "description": "Search for relevant documents"},
            {"type": "analysis", "description": "Analyze document content"},
            {"type": "synthesis", "description": "Synthesize findings"}
        ],
        max_steps=request.max_steps
    )
    
    # Create execution
    execution = agent_engine.create_execution(
        task=task,
        plan=plan,
        max_steps=request.max_steps,
        timeout_seconds=request.timeout_seconds
    )
    
    # Start execution
    execution.start()
    
    return AgentResponse(
        id=execution.id,
        status=execution.status.value,
        goal=request.goal,
        current_step=execution.current_step,
        steps_completed=len(execution.steps_completed),
        max_steps=execution.max_steps,
        duration_ms=execution.get_duration_ms(),
        result=execution.result,
        error=execution.error,
        approval_required=execution.approval_required
    )


@router.get("/{agent_id}", response_model=AgentResponse)
def get_agent(
    agent_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get agent execution status."""
    execution = agent_engine.get_execution(agent_id)
    if not execution:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return AgentResponse(
        id=execution.id,
        status=execution.status.value,
        goal=execution.task.query,
        current_step=execution.current_step,
        steps_completed=len(execution.steps_completed),
        max_steps=execution.max_steps,
        duration_ms=execution.get_duration_ms(),
        result=execution.result,
        error=execution.error,
        approval_required=execution.approval_required
    )


@router.post("/{agent_id}/cancel")
def cancel_agent(
    agent_id: str,
    current_user: User = Depends(get_current_user)
):
    """Cancel an agent execution."""
    success = agent_engine.cancel_execution(agent_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot cancel agent")
    
    return {"message": "Agent cancelled"}


@router.post("/{agent_id}/approve")
def approve_agent_step(
    agent_id: str,
    request: ApprovalRequest,
    current_user: User = Depends(get_current_user)
):
    """Approve a pending agent step."""
    execution = agent_engine.get_execution(agent_id)
    if not execution:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    success = execution.approve_step(request.approval_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot approve step")
    
    return {"message": "Step approved"}


@router.post("/{agent_id}/reject")
def reject_agent_step(
    agent_id: str,
    request: ApprovalRequest,
    current_user: User = Depends(get_current_user)
):
    """Reject a pending agent step."""
    execution = agent_engine.get_execution(agent_id)
    if not execution:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    success = execution.reject_step(request.approval_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot reject step")
    
    return {"message": "Step rejected"}


@router.get("")
def list_agents(
    status: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """List agent executions."""
    agent_status = None
    if status:
        try:
            agent_status = AgentStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid status")
    
    executions = agent_engine.list_executions(
        user_id=current_user.id,
        status=agent_status
    )
    
    return {
        "items": [
            {
                "id": e.id,
                "status": e.status.value,
                "goal": e.task.query,
                "duration_ms": e.get_duration_ms()
            }
            for e in executions
        ]
    }
