"""AI Governance - Enterprise AI controls and settings."""

from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone


class AIGovernanceLevel(str, Enum):
    """AI governance security levels."""
    RESTRICTED = "restricted"
    STANDARD = "standard"
    PERMISSIVE = "permissive"


class AIProviderStatus(str, Enum):
    """AI provider availability status."""
    ENABLED = "enabled"
    DISABLED = "disabled"
    RATE_LIMITED = "rate_limited"
    UNAVAILABLE = "unavailable"


class WorkspaceAISettings:
    """Workspace-level AI governance settings."""
    
    def __init__(self, workspace_id: int):
        self.workspace_id = workspace_id
        self.ai_enabled = True
        self.agent_enabled = True
        self.governance_level = AIGovernanceLevel.STANDARD
        
        # Allowed providers/models
        self.allowed_providers: List[str] = ["fake", "openai_compatible"]
        self.allowed_models: List[str] = ["fake-llm", "gpt-4", "gpt-3.5-turbo"]
        
        # Budget limits
        self.daily_ai_budget: float = 100.0  # USD
        self.monthly_ai_budget: float = 1000.0  # USD
        self.daily_request_limit: int = 1000
        self.monthly_request_limit: int = 30000
        
        # Tool permissions
        self.enabled_tool_categories: List[str] = ["read_only", "reversible_write"]
        self.disabled_tools: List[str] = []
        
        # Agent settings
        self.max_agent_steps: int = 10
        self.agent_timeout_seconds: int = 300
        self.require_approval_for_writes: bool = True
        
        # Content safety
        self.enable_prompt_injection_defense: bool = True
        self.enable_content_sanitization: bool = True
        self.max_input_length: int = 10000
        
        # Audit
        self.log_ai_executions: bool = True
        self.log_tool_invocations: bool = True
        
        # Retention
        self.ai_trace_retention_days: int = 30
        self.workflow_retention_days: int = 90
    
    def can_use_provider(self, provider: str) -> bool:
        """Check if a provider is allowed."""
        return provider in self.allowed_providers
    
    def can_use_model(self, model: str) -> bool:
        """Check if a model is allowed."""
        return model in self.allowed_models
    
    def can_use_tool(self, tool_name: str) -> bool:
        """Check if a tool is allowed."""
        return tool_name not in self.disabled_tools
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "ai_enabled": self.ai_enabled,
            "agent_enabled": self.agent_enabled,
            "governance_level": self.governance_level.value,
            "allowed_providers": self.allowed_providers,
            "allowed_models": self.allowed_models,
            "daily_ai_budget": self.daily_ai_budget,
            "monthly_ai_budget": self.monthly_ai_budget,
            "daily_request_limit": self.daily_request_limit,
            "monthly_request_limit": self.monthly_request_limit,
            "enabled_tool_categories": self.enabled_tool_categories,
            "disabled_tools": self.disabled_tools,
            "max_agent_steps": self.max_agent_steps,
            "agent_timeout_seconds": self.agent_timeout_seconds,
            "require_approval_for_writes": self.require_approval_for_writes,
            "enable_prompt_injection_defense": self.enable_prompt_injection_defense,
            "enable_content_sanitization": self.enable_content_sanitization,
            "max_input_length": self.max_input_length,
            "log_ai_executions": self.log_ai_executions,
            "log_tool_invocations": self.log_tool_invocations,
            "ai_trace_retention_days": self.ai_trace_retention_days,
            "workflow_retention_days": self.workflow_retention_days
        }


class AIBudgetTracker:
    """Tracks AI usage against budget limits."""
    
    def __init__(self):
        self._usage: Dict[int, Dict[str, Any]] = {}  # workspace_id -> usage data
    
    def get_usage(self, workspace_id: int) -> Dict[str, Any]:
        """Get current usage for a workspace."""
        if workspace_id not in self._usage:
            self._usage[workspace_id] = {
                "daily_requests": 0,
                "monthly_requests": 0,
                "daily_cost": 0.0,
                "monthly_cost": 0.0,
                "last_reset_daily": datetime.now(timezone.utc).date(),
                "last_reset_monthly": datetime.now(timezone.utc).month
            }
        
        usage = self._usage[workspace_id]
        now = datetime.now(timezone.utc).date()
        
        # Reset daily counter if new day
        if usage["last_reset_daily"] != now:
            usage["daily_requests"] = 0
            usage["daily_cost"] = 0.0
            usage["last_reset_daily"] = now
        
        # Reset monthly counter if new month
        if usage["last_reset_monthly"] != now.month:
            usage["monthly_requests"] = 0
            usage["monthly_cost"] = 0.0
            usage["last_reset_monthly"] = now.month
        
        return usage
    
    def can_make_request(
        self,
        workspace_id: int,
        settings: WorkspaceAISettings,
        estimated_cost: float = 0.0
    ) -> tuple[bool, str]:
        """Check if a request can be made within budget."""
        usage = self.get_usage(workspace_id)
        
        # Check daily request limit
        if usage["daily_requests"] >= settings.daily_request_limit:
            return False, "Daily request limit exceeded"
        
        # Check monthly request limit
        if usage["monthly_requests"] >= settings.monthly_request_limit:
            return False, "Monthly request limit exceeded"
        
        # Check daily cost limit
        if usage["daily_cost"] + estimated_cost > settings.daily_ai_budget:
            return False, "Daily budget exceeded"
        
        # Check monthly cost limit
        if usage["monthly_cost"] + estimated_cost > settings.monthly_ai_budget:
            return False, "Monthly budget exceeded"
        
        return True, ""
    
    def record_request(
        self,
        workspace_id: int,
        cost: float = 0.0
    ):
        """Record an AI request."""
        usage = self.get_usage(workspace_id)
        usage["daily_requests"] += 1
        usage["monthly_requests"] += 1
        usage["daily_cost"] += cost
        usage["monthly_cost"] += cost


class AIGovernanceService:
    """Service for managing AI governance."""
    
    def __init__(self):
        self._settings: Dict[int, WorkspaceAISettings] = {}
        self.budget_tracker = AIBudgetTracker()
    
    def get_settings(self, workspace_id: int) -> WorkspaceAISettings:
        """Get AI settings for a workspace."""
        if workspace_id not in self._settings:
            self._settings[workspace_id] = WorkspaceAISettings(workspace_id)
        return self._settings[workspace_id]
    
    def update_settings(
        self,
        workspace_id: int,
        **kwargs
    ) -> bool:
        """Update AI settings for a workspace."""
        settings = self.get_settings(workspace_id)
        
        for key, value in kwargs.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
        
        return True
    
    def validate_ai_request(
        self,
        workspace_id: int,
        provider: str,
        model: str,
        estimated_cost: float = 0.0
    ) -> tuple[bool, str]:
        """Validate an AI request against governance settings."""
        settings = self.get_settings(workspace_id)
        
        # Check if AI is enabled
        if not settings.ai_enabled:
            return False, "AI is disabled for this workspace"
        
        # Check provider
        if not settings.can_use_provider(provider):
            return False, f"Provider '{provider}' is not allowed"
        
        # Check model
        if not settings.can_use_model(model):
            return False, f"Model '{model}' is not allowed"
        
        # Check budget
        can_proceed, reason = self.budget_tracker.can_make_request(
            workspace_id, settings, estimated_cost
        )
        if not can_proceed:
            return False, reason
        
        return True, ""
    
    def record_ai_usage(
        self,
        workspace_id: int,
        cost: float = 0.0
    ):
        """Record AI usage."""
        self.budget_tracker.record_request(workspace_id, cost)
