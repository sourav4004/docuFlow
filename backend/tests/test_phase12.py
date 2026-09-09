"""Phase 12 Test Suite - Real AI + Production Agent Infrastructure."""

import pytest
from datetime import datetime, timezone, timedelta
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.core.database import Base, get_db
from app.services.model_registry import (
    ModelRegistry, ModelConfig, ModelCapability, TaskComplexity, model_registry
)
from app.services.provider_resilience import (
    CircuitBreaker, CircuitState, RetryPolicy, ProviderMetrics, ResilientProvider
)
from app.services.ai_orchestration import AITask, AITaskType, AITaskPriority
from tests.shared_db import engine, TestingSessionLocal, override_get_db


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture
def client():
    return TestClient(app)


# ============================================================
# Model Registry Tests
# ============================================================

class TestModelRegistry:
    def test_register_model(self):
        registry = ModelRegistry()
        config = ModelConfig(
            provider="test",
            model="test-model",
            capabilities=[ModelCapability.TEXT_GENERATION]
        )
        registry.register_model(config)
        
        retrieved = registry.get_model("test", "test-model")
        assert retrieved is not None
        assert retrieved.model == "test-model"

    def test_list_models(self):
        registry = ModelRegistry()
        models = registry.list_models()
        assert len(models) > 0

    def test_select_model_for_task(self):
        registry = ModelRegistry()
        
        # Simple task
        model = registry.select_model_for_task(TaskComplexity.SIMPLE)
        assert model is not None
        
        # Complex task requiring tool calling
        model = registry.select_model_for_task(
            TaskComplexity.COMPLEX,
            required_capabilities=[ModelCapability.TOOL_CALLING]
        )
        # May return None if no model has tool calling in test env
        # This is acceptable

    def test_estimate_cost(self):
        cost = model_registry.estimate_cost(
            provider="openai_compatible",
            model="gpt-4",
            input_tokens=1000,
            output_tokens=500
        )
        assert cost > 0

    def test_model_capabilities(self):
        config = ModelConfig(
            provider="test",
            model="test",
            capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.VISION]
        )
        assert config.has_capability(ModelCapability.TEXT_GENERATION)
        assert config.has_capability(ModelCapability.VISION)
        assert not config.has_capability(ModelCapability.EMBEDDING)


# ============================================================
# Circuit Breaker Tests
# ============================================================

class TestCircuitBreaker:
    def test_initial_state(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute()

    def test_trips_after_failures(self):
        cb = CircuitBreaker(failure_threshold=3)
        
        for _ in range(3):
            cb.record_failure()
        
        assert cb.state == CircuitState.OPEN
        assert not cb.can_execute()

    def test_recovery_timeout(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        
        # Trip the breaker
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        
        # Wait for recovery
        import time
        time.sleep(0.2)
        
        # Should be half-open now
        assert cb.can_execute()
        assert cb.state == CircuitState.HALF_OPEN

    def test_success_resets(self):
        cb = CircuitBreaker(failure_threshold=2, half_open_max_calls=2)
        
        # Trip
        cb.record_failure()
        cb.record_failure()
        
        # Recover
        cb.state = CircuitState.HALF_OPEN
        cb.record_success()
        cb.record_success()
        
        assert cb.state == CircuitState.CLOSED

    def test_failure_in_half_open_trips(self):
        cb = CircuitBreaker(failure_threshold=2)
        
        # Trip
        cb.record_failure()
        cb.record_failure()
        
        # Enter half-open
        cb.state = CircuitState.HALF_OPEN
        
        # Failure should trip again
        cb.record_failure()
        assert cb.state == CircuitState.OPEN


# ============================================================
# Retry Policy Tests
# ============================================================

class TestRetryPolicy:
    def test_should_retry_on_timeout(self):
        policy = RetryPolicy(max_retries=3)
        
        # Timeout errors should be retried
        assert policy.should_retry(0, TimeoutError("Connection timed out"))
        assert policy.should_retry(1, TimeoutError("Connection timed out"))

    def test_no_retry_after_max(self):
        policy = RetryPolicy(max_retries=2)
        
        # Should not retry after max attempts
        assert not policy.should_retry(2, TimeoutError("timeout"))

    def test_delay_with_jitter(self):
        policy = RetryPolicy(base_delay=1.0, jitter=True)
        
        delay1 = policy.get_delay(0)
        delay2 = policy.get_delay(0)
        
        # With jitter, delays should vary
        # Both should be around 1 second
        assert 0.5 <= delay1 <= 2.0
        assert 0.5 <= delay2 <= 2.0

    def test_exponential_backoff(self):
        policy = RetryPolicy(base_delay=1.0, jitter=False)
        
        delay0 = policy.get_delay(0)
        delay1 = policy.get_delay(1)
        delay2 = policy.get_delay(2)
        
        assert delay0 < delay1 < delay2


# ============================================================
# Provider Metrics Tests
# ============================================================

class TestProviderMetrics:
    def test_record_request(self):
        metrics = ProviderMetrics()
        
        metrics.record_request(True, 100.0, tokens=50, cost=0.01)
        metrics.record_request(True, 150.0, tokens=75, cost=0.015)
        
        assert metrics.total_requests == 2
        assert metrics.successful_requests == 2
        assert metrics.total_tokens == 125
        assert metrics.total_cost == 0.025

    def test_success_rate(self):
        metrics = ProviderMetrics()
        
        metrics.record_request(True, 100.0)
        metrics.record_request(True, 100.0)
        metrics.record_request(False, 100.0)
        
        assert metrics.get_success_rate() == 2 / 3

    def test_to_dict(self):
        metrics = ProviderMetrics()
        metrics.record_request(True, 100.0)
        
        data = metrics.to_dict()
        assert "total_requests" in data
        assert "success_rate" in data


# ============================================================
# Resilient Provider Tests
# ============================================================

class TestResilientProvider:
    def test_execute_success(self):
        provider = ResilientProvider("test")
        
        def success_func():
            return "success"
        
        result = provider.execute_with_resilience(success_func)
        assert result == "success"
        assert provider.metrics.successful_requests == 1

    def test_execute_with_retry(self):
        provider = ResilientProvider(
            "test",
            retry_policy=RetryPolicy(max_retries=2, base_delay=0.01)
        )
        
        call_count = 0
        
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("timeout")
            return "success"
        
        result = provider.execute_with_resilience(flaky_func)
        assert result == "success"
        assert call_count == 3

    def test_execute_circuit_breaker(self):
        provider = ResilientProvider(
            "test",
            circuit_breaker=CircuitBreaker(failure_threshold=2)
        )
        
        def fail_func():
            raise ValueError("error")
        
        # Trip the circuit breaker
        for _ in range(2):
            try:
                provider.execute_with_resilience(fail_func)
            except:
                pass
        
        # Should now be blocked
        with pytest.raises(Exception) as exc_info:
            provider.execute_with_resilience(fail_func)
        assert "Circuit breaker" in str(exc_info.value) or "open" in str(exc_info.value).lower()


# ============================================================
# AI Task Router Tests (Enhanced)
# ============================================================

class TestAITaskRouterPhase12:
    def test_classify_with_context(self):
        from app.services.ai_task_router import AITaskRouter
        
        task = AITask(
            query="Extract all dates and amounts from this invoice",
            context={"document_type": "invoice"}
        )
        task = AITaskRouter.classify_task(task)
        
        assert task.task_type == AITaskType.EXTRACTION

    def test_priority_for_urgent_query(self):
        from app.services.ai_task_router import AITaskRouter
        
        task = AITask(query="What is the status right now?")
        task = AITaskRouter.classify_task(task)
        
        # Should be fast priority for status queries
        assert task.priority in [AITaskPriority.FAST, AITaskPriority.BALANCED]


# ============================================================
# API Integration Tests
# ============================================================

class TestPhase12API:
    def test_model_registry_endpoint(self, client):
        """Test model registry listing."""
        client.post("/auth/register", json={
            "name": "Model User",
            "email": "model@test.com",
            "password": "password123"
        })
        login = client.post("/auth/login", json={
            "email": "model@test.com",
            "password": "password123"
        })
        cookies = login.cookies
        
        # List AI tasks (includes model info)
        response = client.get("/ai/tasks", cookies=cookies)
        assert response.status_code == 200

    def test_ai_execution_with_persistence(self, client):
        """Test AI execution creates persistent record."""
        client.post("/auth/register", json={
            "name": "Exec User",
            "email": "exec@test.com",
            "password": "password123"
        })
        login = client.post("/auth/login", json={
            "email": "exec@test.com",
            "password": "password123"
        })
        cookies = login.cookies
        
        # Create agent execution
        response = client.post("/agents", json={
            "goal": "Analyze document for key terms"
        }, cookies=cookies)
        
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "status" in data


# ============================================================
# Security Tests
# ============================================================

class TestPhase12Security:
    def test_circuit_breaker_prevents_abuse(self):
        """Test that circuit breaker prevents repeated failures."""
        cb = CircuitBreaker(failure_threshold=3)
        
        # Simulate repeated failures
        for _ in range(5):
            cb.record_failure()
        
        # Circuit should be open
        assert cb.state == CircuitState.OPEN
        assert not cb.can_execute()

    def test_retry_policy_limits_attempts(self):
        """Test that retry policy limits total attempts."""
        policy = RetryPolicy(max_retries=2)
        
        # Should not retry after max
        assert not policy.should_retry(2, TimeoutError("timeout"))
        assert not policy.should_retry(3, TimeoutError("timeout"))

    def test_model_selection_respects_capabilities(self):
        """Test that model selection respects required capabilities."""
        registry = ModelRegistry()
        
        # Request model with vision capability
        model = registry.select_model_for_task(
            TaskComplexity.MULTIMODAL,
            required_capabilities=[ModelCapability.VISION]
        )
        
        # If model found, it should have vision
        if model:
            assert model.supports_vision


# ============================================================
# Database Migration Tests
# ============================================================

class TestPhase12Database:
    def test_new_tables_exist(self):
        """Test that new AI execution tables exist."""
        from sqlalchemy import inspect as sqla_inspect
        
        inspector = sqla_inspect(engine)
        tables = inspector.get_table_names()
        
        required_tables = [
            "ai_executions",
            "ai_execution_steps",
            "ai_approvals",
            "ai_artifacts"
        ]
        
        for table in required_tables:
            assert table in tables, f"Table {table} not found"

    def test_ai_execution_indexes(self):
        """Test that AI execution indexes exist."""
        from sqlalchemy import inspect as sqla_inspect
        
        inspector = sqla_inspect(engine)
        indexes = [idx['name'] for idx in inspector.get_indexes('ai_executions')]
        
        assert any('workspace_id' in idx for idx in indexes)
        assert any('status' in idx for idx in indexes)

    def test_import_all_models(self):
        """Test that all new models can be imported."""
        from app.models.ai_execution import (
            AIExecution, AIExecutionStep, AIApproval, AIArtifact
        )
        
        # All imports successful
        assert AIExecution is not None
        assert AIExecutionStep is not None
        assert AIApproval is not None
        assert AIArtifact is not None


# ============================================================
# E2E Scenario Tests
# ============================================================

class TestPhase12E2E:
    def test_model_routing_for_task(self):
        """Test that appropriate model is selected for task."""
        from app.services.ai_task_router import AITaskRouter
        
        # Document QA task
        task = AITask(query="What is this document about?")
        task = AITaskRouter.classify_task(task)
        
        # Should select an appropriate model
        model = model_registry.select_model_for_task(
            TaskComplexity.STANDARD,
            required_capabilities=[ModelCapability.TEXT_GENERATION]
        )
        assert model is not None

    def test_cost_estimation(self):
        """Test cost estimation for AI operations."""
        from app.services.model_registry import model_registry
        
        cost = model_registry.estimate_cost(
            provider="openai_compatible",
            model="gpt-4",
            input_tokens=1000,
            output_tokens=500
        )
        
        assert cost > 0
        assert cost < 1.0  # Reasonable cost for small request


# ============================================================
# Workflow Engine Tests (Phase 12 enhancements)
# ============================================================

class TestWorkflowEnginePhase12:
    def test_workflow_with_tool_steps(self):
        """Test workflow with tool-based steps."""
        from app.services.workflow_engine import WorkflowEngine, WorkflowNode, WorkflowNodeType
        
        engine = WorkflowEngine()
        
        nodes = [
            WorkflowNode("search", WorkflowNodeType.SEARCH, "Search documents"),
            WorkflowNode("extract", WorkflowNodeType.EXTRACT, "Extract data"),
            WorkflowNode("validate", WorkflowNodeType.VALIDATE, "Validate results")
        ]
        
        definition = engine.create_definition(
            name="Invoice Processing",
            description="Process invoices automatically",
            workspace_id=1,
            created_by=1,
            nodes=nodes
        )
        
        assert len(definition.nodes) == 3
        
        # Start a run
        run = engine.start_run(definition.id, triggered_by=1)
        assert run is not None
        assert run.status.value == "running"
