"""AI Orchestration Architecture - Core abstractions for AI-native knowledge platform."""

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import uuid


class AITaskType(str, Enum):
    """AI task types for routing."""
    DOCUMENT_QA = "document_qa"
    MULTI_DOCUMENT_ANALYSIS = "multi_document_analysis"
    SUMMARIZATION = "summarization"
    EXTRACTION = "extraction"
    COMPARISON = "comparison"
    CLASSIFICATION = "classification"
    SEARCH = "search"
    RECOMMENDATION = "recommendation"
    DRAFTING = "drafting"
    WORKFLOW_EXECUTION = "workflow_execution"
    STRUCTURED_ANALYSIS = "structured_analysis"
    CONFLICT_DETECTION = "conflict_detection"
    ENTITY_EXTRACTION = "entity_extraction"


class AITaskPriority(str, Enum):
    """AI task priority levels."""
    FAST = "fast"          # Simple classification/search
    BALANCED = "balanced"  # Summarization
    POWERFUL = "powerful"  # Complex reasoning
    CHEAP = "cheap"        # Background processing


class AgentStatus(str, Enum):
    """Agent execution lifecycle states."""
    CREATED = "created"
    PLANNING = "planning"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    EXECUTING = "executing"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ToolRiskLevel(str, Enum):
    """Tool risk classification."""
    READ_ONLY = "read_only"
    REVERSIBLE_WRITE = "reversible_write"
    IRREVERSIBLE_WRITE = "irreversible_write"


class GroundingStatus(str, Enum):
    """Evidence grounding status."""
    GROUNDED = "grounded"
    PARTIALLY_GROUNDED = "partially_grounded"
    UNSUPPORTED = "unsupported"


@dataclass
class AITask:
    """Represents an AI task to be executed."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: AITaskType = AITaskType.DOCUMENT_QA
    priority: AITaskPriority = AITaskPriority.BALANCED
    query: str = ""
    workspace_id: Optional[int] = None
    user_id: Optional[int] = None
    document_ids: List[int] = field(default_factory=list)
    collection_id: Optional[int] = None
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ExecutionStep:
    """A single step in an AI execution."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step_type: str = ""  # retrieval, tool_call, reasoning, validation
    description: str = ""
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending, running, completed, failed
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


@dataclass
class AITrace:
    """Safe execution trace for AI operations."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    workspace_id: Optional[int] = None
    user_id: Optional[int] = None
    task_type: Optional[AITaskType] = None
    status: str = "running"
    steps: List[ExecutionStep] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    documents_referenced: List[int] = field(default_factory=list)
    token_usage: Dict[str, int] = field(default_factory=dict)
    estimated_cost: float = 0.0
    model: str = ""
    validation_results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    duration_ms: Optional[float] = None


@dataclass
class EvidencePack:
    """Structured evidence container for RAG and agent workflows."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    source_documents: List[Dict[str, Any]] = field(default_factory=list)
    page_information: List[Dict[str, Any]] = field(default_factory=list)
    relevance_scores: List[float] = field(default_factory=list)
    retrieval_method: str = ""  # vector, keyword, hybrid
    conflicting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    evidence_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class GroundingResult:
    """Result of claim-to-evidence validation."""
    status: GroundingStatus = GroundingStatus.UNSUPPORTED
    grounded_claims: List[str] = field(default_factory=list)
    partially_grounded_claims: List[str] = field(default_factory=list)
    unsupported_claims: List[str] = field(default_factory=list)
    grounding_score: float = 0.0
    evidence_coverage: float = 0.0


@dataclass
class ConflictResult:
    """Result of document conflict detection."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_a_id: int = 0
    document_b_id: int = 0
    claim_a: str = ""
    claim_b: str = ""
    conflict_type: str = ""  # factual, temporal, policy, numerical
    severity: str = "medium"  # low, medium, high
    confidence: float = 0.0
    evidence: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AIResponse:
    """Standardized AI response envelope."""
    answer: str = ""
    citations: List[Dict[str, Any]] = field(default_factory=list)
    confidence: Dict[str, Any] = field(default_factory=dict)
    grounding: Optional[GroundingResult] = None
    conflicts: List[ConflictResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_trace_id: Optional[str] = None


@dataclass
class ToolDefinition:
    """Definition of an available AI tool."""
    name: str = ""
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    risk_level: ToolRiskLevel = ToolRiskLevel.READ_ONLY
    requires_auth: bool = True
    requires_workspace: bool = True


@dataclass
class ToolResult:
    """Result of a tool execution."""
    tool_name: str = ""
    success: bool = True
    output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time_ms: float = 0.0


@dataclass
class ApprovalRequest:
    """Request for human-in-the-loop approval."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    action: str = ""
    reason: str = ""
    affected_resources: List[Dict[str, Any]] = field(default_factory=list)
    proposed_parameters: Dict[str, Any] = field(default_factory=dict)
    risk_level: ToolRiskLevel = ToolRiskLevel.READ_ONLY
    status: str = "pending"  # pending, approved, rejected, expired
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None


@dataclass
class AgentPlan:
    """Structured agent execution plan."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    goal: str = ""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    estimated_steps: int = 0
    max_steps: int = 10
    timeout_seconds: int = 300
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
