"""Model Capability Registry - Centralized model configuration and routing."""

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone


class ModelCapability(str, Enum):
    """Model capabilities."""
    TEXT_GENERATION = "text_generation"
    STREAMING = "streaming"
    STRUCTURED_OUTPUT = "structured_output"
    TOOL_CALLING = "tool_calling"
    VISION = "vision"
    EMBEDDING = "embedding"
    MULTIMODAL = "multimodal"


class TaskComplexity(str, Enum):
    """Task complexity levels for model routing."""
    SIMPLE = "simple"      # classification, search, filter
    STANDARD = "standard"  # summarization, extraction
    COMPLEX = "complex"    # multi-document reasoning, comparison
    MULTIMODAL = "multimodal"  # image, table, chart analysis
    AGENT = "agent"        # tool calling, planning


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    provider: str
    model: str
    display_name: str = ""
    capabilities: List[ModelCapability] = field(default_factory=list)
    context_window: int = 4096
    max_output_tokens: int = 4096
    supports_structured_output: bool = False
    supports_tool_calling: bool = False
    supports_vision: bool = False
    supports_streaming: bool = True
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    is_enabled: bool = True
    priority: int = 0  # Higher = preferred
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def has_capability(self, capability: ModelCapability) -> bool:
        """Check if model has a specific capability."""
        return capability in self.capabilities

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "display_name": self.display_name or self.model,
            "capabilities": [c.value for c in self.capabilities],
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "supports_structured_output": self.supports_structured_output,
            "supports_tool_calling": self.supports_tool_calling,
            "supports_vision": self.supports_vision,
            "supports_streaming": self.supports_streaming,
            "cost_per_1k_input": self.cost_per_1k_input,
            "cost_per_1k_output": self.cost_per_1k_output,
            "is_enabled": self.is_enabled,
            "priority": self.priority
        }


class ModelRegistry:
    """Centralized model registry for capability-based routing."""
    
    def __init__(self):
        self._models: Dict[str, ModelConfig] = {}
        self._register_defaults()
    
    def _register_defaults(self):
        """Register default models."""
        # Fake model for testing
        self.register_model(ModelConfig(
            provider="fake",
            model="fake-llm",
            display_name="Fake LLM (Testing)",
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.STRUCTURED_OUTPUT,
            ],
            context_window=4096,
            max_output_tokens=2048,
            supports_structured_output=True,
            is_enabled=True,
            priority=0
        ))
        
        # OpenAI-compatible models
        self.register_model(ModelConfig(
            provider="openai_compatible",
            model="gpt-4",
            display_name="GPT-4",
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.STREAMING,
                ModelCapability.STRUCTURED_OUTPUT,
                ModelCapability.TOOL_CALLING,
                ModelCapability.VISION,
            ],
            context_window=128000,
            max_output_tokens=4096,
            supports_structured_output=True,
            supports_tool_calling=True,
            supports_vision=True,
            cost_per_1k_input=0.03,
            cost_per_1k_output=0.06,
            priority=100
        ))
        
        self.register_model(ModelConfig(
            provider="openai_compatible",
            model="gpt-4-turbo",
            display_name="GPT-4 Turbo",
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.STREAMING,
                ModelCapability.STRUCTURED_OUTPUT,
                ModelCapability.TOOL_CALLING,
                ModelCapability.VISION,
            ],
            context_window=128000,
            max_output_tokens=4096,
            supports_structured_output=True,
            supports_tool_calling=True,
            supports_vision=True,
            cost_per_1k_input=0.01,
            cost_per_1k_output=0.03,
            priority=90
        ))
        
        self.register_model(ModelConfig(
            provider="openai_compatible",
            model="gpt-3.5-turbo",
            display_name="GPT-3.5 Turbo",
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.STREAMING,
                ModelCapability.STRUCTURED_OUTPUT,
                ModelCapability.TOOL_CALLING,
            ],
            context_window=16385,
            max_output_tokens=4096,
            supports_structured_output=True,
            supports_tool_calling=True,
            cost_per_1k_input=0.0005,
            cost_per_1k_output=0.0015,
            priority=50
        ))
    
    def register_model(self, config: ModelConfig):
        """Register a model configuration."""
        key = f"{config.provider}:{config.model}"
        self._models[key] = config
    
    def get_model(self, provider: str, model: str) -> Optional[ModelConfig]:
        """Get a model configuration."""
        key = f"{provider}:{model}"
        return self._models.get(key)
    
    def list_models(self, enabled_only: bool = True) -> List[ModelConfig]:
        """List all registered models."""
        models = list(self._models.values())
        if enabled_only:
            models = [m for m in models if m.is_enabled]
        return models
    
    def select_model_for_task(
        self,
        task_complexity: TaskComplexity,
        required_capabilities: Optional[List[ModelCapability]] = None,
        preferred_provider: Optional[str] = None
    ) -> Optional[ModelConfig]:
        """Select the best model for a given task."""
        candidates = [m for m in self._models.values() if m.is_enabled]
        
        # Filter by required capabilities
        if required_capabilities:
            candidates = [
                m for m in candidates
                if all(m.has_capability(cap) for cap in required_capabilities)
            ]
        
        # Filter by preferred provider if specified
        if preferred_provider:
            provider_candidates = [m for m in candidates if m.provider == preferred_provider]
            if provider_candidates:
                candidates = provider_candidates
        
        if not candidates:
            return None
        
        # Sort by priority (higher is better)
        candidates.sort(key=lambda m: m.priority, reverse=True)
        
        # For complex tasks, prefer higher-capability models
        if task_complexity in (TaskComplexity.COMPLEX, TaskComplexity.AGENT):
            # Prefer models with tool calling and structured output
            complex_models = [
                m for m in candidates
                if m.supports_tool_calling and m.supports_structured_output
            ]
            if complex_models:
                return complex_models[0]
        
        # For multimodal tasks, require vision
        if task_complexity == TaskComplexity.MULTIMODAL:
            vision_models = [m for m in candidates if m.supports_vision]
            if vision_models:
                return vision_models[0]
        
        # Return highest priority candidate
        return candidates[0] if candidates else None
    
    def estimate_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Estimate cost for a request."""
        config = self.get_model(provider, model)
        if not config:
            return 0.0
        
        input_cost = (input_tokens / 1000) * config.cost_per_1k_input
        output_cost = (output_tokens / 1000) * config.cost_per_1k_output
        return input_cost + output_cost


# Global registry instance
model_registry = ModelRegistry()
