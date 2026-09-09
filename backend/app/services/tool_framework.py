"""Tool Framework - Safe tool execution for AI agents."""

from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timezone
import time

from .ai_orchestration import (
    ToolDefinition, ToolResult, ToolRiskLevel, ApprovalRequest
)


class ToolRegistry:
    """Registry for AI tools with permission controls."""
    
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._handlers: Dict[str, Callable] = {}
    
    def register_tool(
        self,
        definition: ToolDefinition,
        handler: Callable[[Dict[str, Any]], Dict[str, Any]]
    ):
        """Register a tool with its handler."""
        self._tools[definition.name] = definition
        self._handlers[definition.name] = handler
    
    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get tool definition by name."""
        return self._tools.get(name)
    
    def get_handler(self, name: str) -> Optional[Callable]:
        """Get tool handler by name."""
        return self._handlers.get(name)
    
    def list_tools(self) -> List[ToolDefinition]:
        """List all registered tools."""
        return list(self._tools.values())
    
    def list_tools_by_risk(self, risk_level: ToolRiskLevel) -> List[ToolDefinition]:
        """List tools by risk level."""
        return [t for t in self._tools.values() if t.risk_level == risk_level]


class ToolExecutor:
    """Executes tools with safety checks and permission controls."""
    
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self._pending_approvals: Dict[str, ApprovalRequest] = {}
    
    def can_execute_without_approval(
        self,
        tool_name: str,
        user_role: str = "MEMBER"
    ) -> bool:
        """Check if tool can execute without user approval."""
        tool = self.registry.get_tool(tool_name)
        if not tool:
            return False
        
        # READ_ONLY tools can always execute
        if tool.risk_level == ToolRiskLevel.READ_ONLY:
            return True
        
        # REVERSIBLE_WRITE requires MEMBER+ role
        if tool.risk_level == ToolRiskLevel.REVERSIBLE_WRITE:
            return user_role in ("OWNER", "ADMIN", "MEMBER")
        
        # IRREVERSIBLE_WRITE always requires approval
        return False
    
    def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        user_id: int,
        workspace_id: int,
        user_role: str = "MEMBER"
    ) -> ToolResult:
        """Execute a tool with safety checks."""
        tool = self.registry.get_tool(tool_name)
        handler = self.registry.get_handler(tool_name)
        
        if not tool or not handler:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Tool '{tool_name}' not found"
            )
        
        # Check authorization
        if tool.requires_auth and not user_id:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error="Authentication required"
            )
        
        if tool.requires_workspace and not workspace_id:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error="Workspace context required"
            )
        
        # Check if approval is needed
        if not self.can_execute_without_approval(tool_name, user_role):
            approval = self._create_approval_request(
                tool_name=tool_name,
                parameters=parameters,
                user_id=user_id,
                workspace_id=workspace_id,
                risk_level=tool.risk_level
            )
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Approval required: {approval.id}",
                output={"approval_id": approval.id}
            )
        
        # Execute tool
        start_time = time.time()
        try:
            # Add context to parameters
            parameters["_user_id"] = user_id
            parameters["_workspace_id"] = workspace_id
            
            output = handler(parameters)
            execution_time = (time.time() - start_time) * 1000
            
            return ToolResult(
                tool_name=tool_name,
                success=True,
                output=output,
                execution_time_ms=execution_time
            )
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=str(e),
                execution_time_ms=execution_time
            )
    
    def _create_approval_request(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        user_id: int,
        workspace_id: int,
        risk_level: ToolRiskLevel
    ) -> ApprovalRequest:
        """Create an approval request for a tool execution."""
        approval = ApprovalRequest(
            action=f"Execute tool: {tool_name}",
            reason=f"Tool requires user approval (risk level: {risk_level.value})",
            affected_resources=[{
                "type": "tool_execution",
                "tool_name": tool_name,
                "workspace_id": workspace_id
            }],
            proposed_parameters=parameters,
            risk_level=risk_level,
            status="pending",
            expires_at=datetime.now(timezone.utc).replace(
                hour=datetime.now(timezone.utc).hour + 1
            )
        )
        self._pending_approvals[approval.id] = approval
        return approval
    
    def approve_execution(self, approval_id: str) -> bool:
        """Approve a pending tool execution."""
        approval = self._pending_approvals.get(approval_id)
        if not approval or approval.status != "pending":
            return False
        
        # Check expiration
        if approval.expires_at and datetime.now(timezone.utc) > approval.expires_at:
            approval.status = "expired"
            return False
        
        approval.status = "approved"
        approval.resolved_at = datetime.now(timezone.utc)
        return True
    
    def reject_execution(self, approval_id: str) -> bool:
        """Reject a pending tool execution."""
        approval = self._pending_approvals.get(approval_id)
        if not approval or approval.status != "pending":
            return False
        
        approval.status = "rejected"
        approval.resolved_at = datetime.now(timezone.utc)
        return True


# Create default tool registry with common tools
def create_default_tool_registry() -> ToolRegistry:
    """Create a registry with default tools."""
    registry = ToolRegistry()
    
    # Document Search Tool
    registry.register_tool(
        definition=ToolDefinition(
            name="document_search",
            description="Search for documents by keyword or semantic query",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 10}
                },
                "required": ["query"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "results": {"type": "array"},
                    "count": {"type": "integer"}
                }
            },
            risk_level=ToolRiskLevel.READ_ONLY,
            requires_auth=True,
            requires_workspace=True
        ),
        handler=lambda params: {"results": [], "count": 0}
    )
    
    # Collection Search Tool
    registry.register_tool(
        definition=ToolDefinition(
            name="collection_search",
            description="Search within a specific collection",
            input_schema={
                "type": "object",
                "properties": {
                    "collection_id": {"type": "integer"},
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 10}
                },
                "required": ["collection_id", "query"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "results": {"type": "array"},
                    "count": {"type": "integer"}
                }
            },
            risk_level=ToolRiskLevel.READ_ONLY,
            requires_auth=True,
            requires_workspace=True
        ),
        handler=lambda params: {"results": [], "count": 0}
    )
    
    # Document Retrieval Tool
    registry.register_tool(
        definition=ToolDefinition(
            name="document_retrieve",
            description="Retrieve a specific document by ID",
            input_schema={
                "type": "object",
                "properties": {
                    "document_id": {"type": "integer"}
                },
                "required": ["document_id"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "document": {"type": "object"}
                }
            },
            risk_level=ToolRiskLevel.READ_ONLY,
            requires_auth=True,
            requires_workspace=True
        ),
        handler=lambda params: {"document": None}
    )
    
    # Entity Lookup Tool
    registry.register_tool(
        definition=ToolDefinition(
            name="entity_lookup",
            description="Look up entities and their relationships",
            input_schema={
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string"},
                    "entity_type": {"type": "string"}
                },
                "required": ["entity_name"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "entities": {"type": "array"},
                    "relationships": {"type": "array"}
                }
            },
            risk_level=ToolRiskLevel.READ_ONLY,
            requires_auth=True,
            requires_workspace=True
        ),
        handler=lambda params: {"entities": [], "relationships": []}
    )
    
    # Summarization Tool
    registry.register_tool(
        definition=ToolDefinition(
            name="summarize",
            description="Generate a summary of provided content",
            input_schema={
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "max_length": {"type": "integer", "default": 200}
                },
                "required": ["content"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "summary": {"type": "string"}
                }
            },
            risk_level=ToolRiskLevel.READ_ONLY,
            requires_auth=True,
            requires_workspace=True
        ),
        handler=lambda params: {"summary": "Summary not available"}
    )
    
    # Notification Tool (requires approval)
    registry.register_tool(
        definition=ToolDefinition(
            name="send_notification",
            description="Send a notification to workspace members",
            input_schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "message": {"type": "string"},
                    "user_ids": {"type": "array", "items": {"type": "integer"}}
                },
                "required": ["title", "message"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "sent": {"type": "boolean"},
                    "notification_ids": {"type": "array"}
                }
            },
            risk_level=ToolRiskLevel.REVERSIBLE_WRITE,
            requires_auth=True,
            requires_workspace=True
        ),
        handler=lambda params: {"sent": True, "notification_ids": []}
    )
    
    return registry
