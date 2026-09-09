"""Workflow Engine - User-approved AI workflows."""

from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import uuid


class WorkflowNodeType(str, Enum):
    """Workflow node types."""
    SEARCH = "search"
    RETRIEVE = "retrieve"
    EXTRACT = "extract"
    SUMMARIZE = "summarize"
    COMPARE = "compare"
    CLASSIFY = "classify"
    VALIDATE = "validate"
    NOTIFY = "notify"
    APPROVAL = "approval"


class WorkflowRunStatus(str, Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowNode:
    """A single node in a workflow."""
    
    def __init__(
        self,
        node_id: str,
        node_type: WorkflowNodeType,
        name: str,
        config: Optional[Dict[str, Any]] = None
    ):
        self.id = node_id
        self.type = node_type
        self.name = name
        self.config = config or {}
        self.inputs: List[str] = []  # Node IDs this depends on
        self.outputs: List[str] = []  # Node IDs that depend on this
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "config": self.config,
            "inputs": self.inputs,
            "outputs": self.outputs
        }


class WorkflowDefinition:
    """A workflow definition."""
    
    def __init__(
        self,
        workflow_id: str,
        name: str,
        description: str,
        workspace_id: int,
        created_by: int,
        nodes: Optional[List[WorkflowNode]] = None
    ):
        self.id = workflow_id
        self.name = name
        self.description = description
        self.workspace_id = workspace_id
        self.created_by = created_by
        self.nodes: List[WorkflowNode] = nodes or []
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        self.is_active = True
    
    def add_node(self, node: WorkflowNode):
        """Add a node to the workflow."""
        self.nodes.append(node)
        self.updated_at = datetime.now(timezone.utc)
    
    def remove_node(self, node_id: str):
        """Remove a node from the workflow."""
        self.nodes = [n for n in self.nodes if n.id != node_id]
        self.updated_at = datetime.now(timezone.utc)
    
    def get_execution_order(self) -> List[WorkflowNode]:
        """Get nodes in execution order (topological sort)."""
        # Simple topological sort
        visited = set()
        order = []
        
        def dfs(node_id: str):
            if node_id in visited:
                return
            visited.add(node_id)
            
            for node in self.nodes:
                if node.id == node_id:
                    for input_id in node.inputs:
                        dfs(input_id)
                    order.append(node)
                    break
        
        # Start from nodes with no inputs
        for node in self.nodes:
            if not node.inputs:
                dfs(node.id)
        
        return order
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "workspace_id": self.workspace_id,
            "created_by": self.created_by,
            "nodes": [n.to_dict() for n in self.nodes],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "is_active": self.is_active
        }


class WorkflowRun:
    """A workflow execution instance."""
    
    def __init__(
        self,
        run_id: str,
        workflow_id: str,
        triggered_by: int,
        trigger_type: str = "manual",
        input_data: Optional[Dict[str, Any]] = None
    ):
        self.id = run_id
        self.workflow_id = workflow_id
        self.triggered_by = triggered_by
        self.trigger_type = trigger_type
        self.input_data = input_data or {}
        self.status = WorkflowRunStatus.PENDING
        self.current_node: Optional[str] = None
        self.node_results: Dict[str, Any] = {}
        self.output_data: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.created_at = datetime.now(timezone.utc)
    
    def start(self) -> bool:
        """Start workflow execution."""
        if self.status != WorkflowRunStatus.PENDING:
            return False
        self.status = WorkflowRunStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)
        return True
    
    def complete(self, output_data: Dict[str, Any]) -> bool:
        """Complete workflow execution."""
        if self.status != WorkflowRunStatus.RUNNING:
            return False
        self.status = WorkflowRunStatus.COMPLETED
        self.output_data = output_data
        self.completed_at = datetime.now(timezone.utc)
        return True
    
    def fail(self, error: str) -> bool:
        """Fail workflow execution."""
        self.status = WorkflowRunStatus.FAILED
        self.error = error
        self.completed_at = datetime.now(timezone.utc)
        return True
    
    def cancel(self) -> bool:
        """Cancel workflow execution."""
        if self.status in (WorkflowRunStatus.COMPLETED, WorkflowRunStatus.FAILED):
            return False
        self.status = WorkflowRunStatus.CANCELLED
        self.completed_at = datetime.now(timezone.utc)
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "status": self.status.value,
            "triggered_by": self.triggered_by,
            "trigger_type": self.trigger_type,
            "current_node": self.current_node,
            "node_results": self.node_results,
            "output_data": self.output_data,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "created_at": self.created_at.isoformat()
        }


class WorkflowEngine:
    """Engine for managing and executing workflows."""
    
    def __init__(self):
        self._definitions: Dict[str, WorkflowDefinition] = {}
        self._runs: Dict[str, WorkflowRun] = {}
    
    def create_definition(
        self,
        name: str,
        description: str,
        workspace_id: int,
        created_by: int,
        nodes: Optional[List[WorkflowNode]] = None
    ) -> WorkflowDefinition:
        """Create a new workflow definition."""
        workflow_id = str(uuid.uuid4())
        definition = WorkflowDefinition(
            workflow_id=workflow_id,
            name=name,
            description=description,
            workspace_id=workspace_id,
            created_by=created_by,
            nodes=nodes
        )
        self._definitions[workflow_id] = definition
        return definition
    
    def get_definition(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        """Get a workflow definition."""
        return self._definitions.get(workflow_id)
    
    def update_definition(
        self,
        workflow_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        nodes: Optional[List[WorkflowNode]] = None
    ) -> bool:
        """Update a workflow definition."""
        definition = self._definitions.get(workflow_id)
        if not definition:
            return False
        
        if name is not None:
            definition.name = name
        if description is not None:
            definition.description = description
        if nodes is not None:
            definition.nodes = nodes
        
        definition.updated_at = datetime.now(timezone.utc)
        return True
    
    def delete_definition(self, workflow_id: str) -> bool:
        """Delete a workflow definition."""
        if workflow_id in self._definitions:
            del self._definitions[workflow_id]
            return True
        return False
    
    def start_run(
        self,
        workflow_id: str,
        triggered_by: int,
        trigger_type: str = "manual",
        input_data: Optional[Dict[str, Any]] = None
    ) -> Optional[WorkflowRun]:
        """Start a workflow run."""
        definition = self._definitions.get(workflow_id)
        if not definition or not definition.is_active:
            return None
        
        run_id = str(uuid.uuid4())
        run = WorkflowRun(
            run_id=run_id,
            workflow_id=workflow_id,
            triggered_by=triggered_by,
            trigger_type=trigger_type,
            input_data=input_data
        )
        self._runs[run_id] = run
        run.start()
        return run
    
    def get_run(self, run_id: str) -> Optional[WorkflowRun]:
        """Get a workflow run."""
        return self._runs.get(run_id)
    
    def list_runs(
        self,
        workflow_id: Optional[str] = None,
        workspace_id: Optional[int] = None,
        status: Optional[WorkflowRunStatus] = None
    ) -> List[WorkflowRun]:
        """List workflow runs with filters."""
        runs = list(self._runs.values())
        
        if workflow_id:
            runs = [r for r in runs if r.workflow_id == workflow_id]
        if status:
            runs = [r for r in runs if r.status == status]
        
        return runs
    
    def cancel_run(self, run_id: str) -> bool:
        """Cancel a workflow run."""
        run = self._runs.get(run_id)
        if run:
            return run.cancel()
        return False


# Predefined workflow templates
WORKFLOW_TEMPLATES = {
    "contract_review": {
        "name": "Contract Review",
        "description": "Analyze contracts for key terms, risks, and obligations",
        "nodes": [
            {"type": "search", "name": "Find contracts"},
            {"type": "retrieve", "name": "Retrieve contract content"},
            {"type": "extract", "name": "Extract key terms"},
            {"type": "validate", "name": "Validate extraction"},
            {"type": "summarize", "name": "Generate summary"},
        ]
    },
    "document_summarization": {
        "name": "Document Summarization",
        "description": "Generate summaries for documents",
        "nodes": [
            {"type": "retrieve", "name": "Retrieve document"},
            {"type": "summarize", "name": "Generate summary"},
        ]
    },
    "policy_change_detection": {
        "name": "Policy Change Detection",
        "description": "Detect changes between policy versions",
        "nodes": [
            {"type": "search", "name": "Find policy versions"},
            {"type": "retrieve", "name": "Retrieve versions"},
            {"type": "compare", "name": "Compare versions"},
            {"type": "validate", "name": "Validate changes"},
        ]
    },
    "invoice_extraction": {
        "name": "Invoice Data Extraction",
        "description": "Extract structured data from invoices",
        "nodes": [
            {"type": "retrieve", "name": "Retrieve invoice"},
            {"type": "extract", "name": "Extract invoice data"},
            {"type": "validate", "name": "Validate extraction"},
        ]
    },
    "compliance_check": {
        "name": "Compliance Check",
        "description": "Check documents against compliance requirements",
        "nodes": [
            {"type": "search", "name": "Find requirements"},
            {"type": "retrieve", "name": "Retrieve documents"},
            {"type": "extract", "name": "Extract requirements"},
            {"type": "validate", "name": "Check compliance"},
            {"type": "notify", "name": "Notify results"},
        ]
    }
}
