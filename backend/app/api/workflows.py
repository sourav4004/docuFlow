"""Workflow API endpoints."""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel

from ..core.database import get_db
from ..core.auth import get_current_user
from ..models.user import User
from ..services.workflow_engine import (
    WorkflowEngine, WorkflowNodeType, WorkflowNode,
    WorkflowRunStatus, WORKFLOW_TEMPLATES
)

router = APIRouter(prefix="/workflows", tags=["workflows"])

# Initialize workflow engine
workflow_engine = WorkflowEngine()


class WorkflowCreateRequest(BaseModel):
    """Request to create a workflow."""
    name: str
    description: str
    nodes: Optional[List[Dict[str, Any]]] = None


class WorkflowNodeRequest(BaseModel):
    """Workflow node request."""
    type: str
    name: str
    config: Optional[Dict[str, Any]] = None
    inputs: Optional[List[str]] = None


class WorkflowResponse(BaseModel):
    """Workflow response."""
    id: str
    name: str
    description: str
    workspace_id: int
    created_by: int
    nodes: List[Dict[str, Any]]
    is_active: bool
    created_at: str
    updated_at: str


class WorkflowRunRequest(BaseModel):
    """Request to run a workflow."""
    input_data: Optional[Dict[str, Any]] = None


class WorkflowRunResponse(BaseModel):
    """Workflow run response."""
    id: str
    workflow_id: str
    status: str
    triggered_by: int
    trigger_type: str
    output_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str


@router.post("", response_model=WorkflowResponse)
def create_workflow(
    request: WorkflowCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new workflow."""
    # Parse nodes
    nodes = []
    if request.nodes:
        for node_data in request.nodes:
            try:
                node_type = WorkflowNodeType(node_data.get("type", "search"))
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid node type: {node_data.get('type')}")
            
            node = WorkflowNode(
                node_id=node_data.get("id", f"node_{len(nodes)}"),
                node_type=node_type,
                name=node_data.get("name", f"Step {len(nodes) + 1}"),
                config=node_data.get("config", {})
            )
            nodes.append(node)
    
    # Create workflow
    definition = workflow_engine.create_definition(
        name=request.name,
        description=request.description,
        workspace_id=1,  # Would get from context
        created_by=current_user.id,
        nodes=nodes
    )
    
    return WorkflowResponse(**definition.to_dict())


@router.get("/templates")
def list_workflow_templates():
    """List available workflow templates."""
    return {"templates": WORKFLOW_TEMPLATES}


@router.post("/templates/{template_name}/create")
def create_from_template(
    template_name: str,
    current_user: User = Depends(get_current_user)
):
    """Create a workflow from a template."""
    template = WORKFLOW_TEMPLATES.get(template_name)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    
    # Create workflow from template
    nodes = []
    for i, node_data in enumerate(template["nodes"]):
        node = WorkflowNode(
            node_id=f"node_{i}",
            node_type=WorkflowNodeType(node_data["type"]),
            name=node_data["name"]
        )
        nodes.append(node)
    
    definition = workflow_engine.create_definition(
        name=template["name"],
        description=template["description"],
        workspace_id=1,  # Would get from context
        created_by=current_user.id,
        nodes=nodes
    )
    
    return WorkflowResponse(**definition.to_dict())


@router.get("/{workflow_id}", response_model=WorkflowResponse)
def get_workflow(
    workflow_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get a workflow definition."""
    definition = workflow_engine.get_definition(workflow_id)
    if not definition:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    return WorkflowResponse(**definition.to_dict())


@router.put("/{workflow_id}", response_model=WorkflowResponse)
def update_workflow(
    workflow_id: str,
    request: WorkflowCreateRequest,
    current_user: User = Depends(get_current_user)
):
    """Update a workflow definition."""
    success = workflow_engine.update_definition(
        workflow_id=workflow_id,
        name=request.name,
        description=request.description
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    definition = workflow_engine.get_definition(workflow_id)
    return WorkflowResponse(**definition.to_dict())


@router.delete("/{workflow_id}")
def delete_workflow(
    workflow_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete a workflow definition."""
    success = workflow_engine.delete_definition(workflow_id)
    if not success:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    return {"message": "Workflow deleted"}


@router.post("/{workflow_id}/runs", response_model=WorkflowRunResponse)
def start_workflow_run(
    workflow_id: str,
    request: WorkflowRunRequest,
    current_user: User = Depends(get_current_user)
):
    """Start a workflow run."""
    run = workflow_engine.start_run(
        workflow_id=workflow_id,
        triggered_by=current_user.id,
        input_data=request.input_data
    )
    
    if not run:
        raise HTTPException(status_code=404, detail="Workflow not found or inactive")
    
    return WorkflowRunResponse(**run.to_dict())


@router.get("/{workflow_id}/runs")
def list_workflow_runs(
    workflow_id: str,
    status: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """List workflow runs."""
    run_status = None
    if status:
        try:
            run_status = WorkflowRunStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid status")
    
    runs = workflow_engine.list_runs(
        workflow_id=workflow_id,
        status=run_status
    )
    
    return {
        "items": [run.to_dict() for run in runs]
    }


@router.get("/runs/{run_id}", response_model=WorkflowRunResponse)
def get_workflow_run(
    run_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get a workflow run."""
    run = workflow_engine.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    return WorkflowRunResponse(**run.to_dict())


@router.post("/runs/{run_id}/cancel")
def cancel_workflow_run(
    run_id: str,
    current_user: User = Depends(get_current_user)
):
    """Cancel a workflow run."""
    success = workflow_engine.cancel_run(run_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot cancel run")
    
    return {"message": "Run cancelled"}
