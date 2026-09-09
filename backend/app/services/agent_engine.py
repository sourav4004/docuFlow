"""Agent Execution Engine - Controlled agent lifecycle management."""

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import time

from .ai_orchestration import (
    AgentStatus, AgentPlan, AITask, AITrace, ExecutionStep,
    ToolResult, ApprovalRequest, ToolRiskLevel
)
from .tool_framework import ToolRegistry, ToolExecutor


class AgentExecution:
    """Represents an agent execution instance."""
    
    def __init__(
        self,
        agent_id: str,
        task: AITask,
        plan: AgentPlan,
        tool_registry: ToolRegistry,
        max_steps: int = 10,
        timeout_seconds: int = 300
    ):
        self.id = agent_id
        self.task = task
        self.plan = plan
        self.tool_executor = ToolExecutor(tool_registry)
        self.max_steps = max_steps
        self.timeout_seconds = timeout_seconds
        
        self.status = AgentStatus.CREATED
        self.current_step = 0
        self.steps_completed: List[ExecutionStep] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.approval_required: bool = False
        self.pending_approval: Optional[ApprovalRequest] = None
    
    def start(self) -> bool:
        """Start agent execution."""
        if self.status != AgentStatus.CREATED:
            return False
        
        self.status = AgentStatus.PLANNING
        self.start_time = time.time()
        return True
    
    def plan_complete(self) -> bool:
        """Mark planning as complete and start execution."""
        if self.status != AgentStatus.PLANNING:
            return False
        
        self.status = AgentStatus.EXECUTING
        return True
    
    def execute_step(self, step: ExecutionStep) -> bool:
        """Execute a single step."""
        if self.status != AgentStatus.EXECUTING:
            return False
        
        # Check step limit
        if len(self.steps_completed) >= self.max_steps:
            self.status = AgentStatus.FAILED
            self.error = f"Maximum steps ({self.max_steps}) exceeded"
            return False
        
        # Check timeout
        if self.start_time and (time.time() - self.start_time) > self.timeout_seconds:
            self.status = AgentStatus.FAILED
            self.error = f"Timeout ({self.timeout_seconds}s) exceeded"
            return False
        
        step.status = "running"
        step.started_at = datetime.now(timezone.utc)
        
        # Execute based on step type
        try:
            if step.step_type == "tool_call":
                result = self._execute_tool_step(step)
                if not result.success:
                    if "Approval required" in (result.error or ""):
                        self.status = AgentStatus.WAITING_FOR_APPROVAL
                        self.approval_required = True
                        return False
                    step.status = "failed"
                    step.error = result.error
                else:
                    step.output_data = result.output
                    step.status = "completed"
            elif step.step_type == "retrieval":
                # Simulate retrieval
                step.output_data = {"chunks": [], "count": 0}
                step.status = "completed"
            elif step.step_type == "reasoning":
                # Simulate reasoning
                step.output_data = {"analysis": "Analysis complete"}
                step.status = "completed"
            elif step.step_type == "validation":
                # Simulate validation
                step.output_data = {"valid": True}
                step.status = "completed"
            else:
                step.output_data = {}
                step.status = "completed"
            
            step.completed_at = datetime.now(timezone.utc)
            self.steps_completed.append(step)
            self.current_step += 1
            
            return True
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.completed_at = datetime.now(timezone.utc)
            return False
    
    def _execute_tool_step(self, step: ExecutionStep) -> ToolResult:
        """Execute a tool call step."""
        tool_name = step.input_data.get("tool_name", "")
        parameters = step.input_data.get("parameters", {})
        
        return self.tool_executor.execute_tool(
            tool_name=tool_name,
            parameters=parameters,
            user_id=self.task.user_id or 0,
            workspace_id=self.task.workspace_id or 0,
            user_role="MEMBER"
        )
    
    def approve_step(self, approval_id: str) -> bool:
        """Approve a pending step."""
        if self.status != AgentStatus.WAITING_FOR_APPROVAL:
            return False
        
        success = self.tool_executor.approve_execution(approval_id)
        if success:
            self.status = AgentStatus.EXECUTING
            self.approval_required = False
        return success
    
    def reject_step(self, approval_id: str) -> bool:
        """Reject a pending step."""
        success = self.tool_executor.reject_execution(approval_id)
        if success:
            self.status = AgentStatus.CANCELLED
        return success
    
    def complete(self, result: Dict[str, Any]) -> bool:
        """Complete agent execution."""
        if self.status not in (AgentStatus.EXECUTING, AgentStatus.VALIDATING):
            return False
        
        self.status = AgentStatus.COMPLETED
        self.result = result
        self.end_time = time.time()
        return True
    
    def fail(self, error: str) -> bool:
        """Mark agent execution as failed."""
        self.status = AgentStatus.FAILED
        self.error = error
        self.end_time = time.time()
        return True
    
    def cancel(self) -> bool:
        """Cancel agent execution."""
        if self.status in (AgentStatus.COMPLETED, AgentStatus.FAILED):
            return False
        
        self.status = AgentStatus.CANCELLED
        self.end_time = time.time()
        return True
    
    def get_duration_ms(self) -> float:
        """Get execution duration in milliseconds."""
        if not self.start_time:
            return 0.0
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/response."""
        return {
            "id": self.id,
            "task_id": self.task.id,
            "status": self.status.value,
            "current_step": self.current_step,
            "steps_completed": len(self.steps_completed),
            "max_steps": self.max_steps,
            "duration_ms": self.get_duration_ms(),
            "result": self.result,
            "error": self.error,
            "approval_required": self.approval_required
        }


class AgentEngine:
    """Engine for managing agent executions."""
    
    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry
        self._executions: Dict[str, AgentExecution] = {}
    
    def create_execution(
        self,
        task: AITask,
        plan: AgentPlan,
        max_steps: int = 10,
        timeout_seconds: int = 300
    ) -> AgentExecution:
        """Create a new agent execution."""
        execution = AgentExecution(
            agent_id=plan.id,
            task=task,
            plan=plan,
            tool_registry=self.tool_registry,
            max_steps=max_steps,
            timeout_seconds=timeout_seconds
        )
        self._executions[execution.id] = execution
        return execution
    
    def get_execution(self, execution_id: str) -> Optional[AgentExecution]:
        """Get an execution by ID."""
        return self._executions.get(execution_id)
    
    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an execution."""
        execution = self._executions.get(execution_id)
        if execution:
            return execution.cancel()
        return False
    
    def list_executions(
        self,
        workspace_id: Optional[int] = None,
        user_id: Optional[int] = None,
        status: Optional[AgentStatus] = None
    ) -> List[AgentExecution]:
        """List executions with optional filters."""
        executions = list(self._executions.values())
        
        if workspace_id:
            executions = [e for e in executions if e.task.workspace_id == workspace_id]
        if user_id:
            executions = [e for e in executions if e.task.user_id == user_id]
        if status:
            executions = [e for e in executions if e.status == status]
        
        return executions
