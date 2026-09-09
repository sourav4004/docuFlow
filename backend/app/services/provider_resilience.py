"""Provider Resilience - Circuit breaker, retry policy, and fallback."""

import time
import random
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


class RetryableError(Exception):
    """Error that should be retried."""


class NonRetryableError(Exception):
    """Error that should NOT be retried."""


@dataclass
class CircuitBreaker:
    """Circuit breaker for provider fault tolerance."""
    
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    half_open_max_calls: int = 3
    
    # Internal state
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[float] = None
    half_open_calls: int = 0
    success_count: int = 0
    
    def record_success(self):
        """Record a successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.half_open_max_calls:
                self._reset()
        elif self.state == CircuitState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self):
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self._trip()
        elif self.failure_count >= self.failure_threshold:
            self._trip()
    
    def _trip(self):
        """Trip the circuit breaker."""
        self.state = CircuitState.OPEN
        self.last_failure_time = time.time()
        logger.warning("Circuit breaker tripped (failures=%d)", self.failure_count)
    
    def _reset(self):
        """Reset the circuit breaker."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.half_open_calls = 0
        self.success_count = 0
        logger.info("Circuit breaker reset")
    
    def can_execute(self) -> bool:
        """Check if a call is allowed."""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            if self.last_failure_time:
                elapsed = time.time() - self.last_failure_time
                if elapsed >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    self.success_count = 0
                    logger.info("Circuit breaker entering half-open state")
                    return True
            return False
        
        if self.state == CircuitState.HALF_OPEN:
            return self.half_open_calls < self.half_open_max_calls
        
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "success_count": self.success_count
        }


@dataclass
class RetryPolicy:
    """Retry policy with exponential backoff."""
    
    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 30.0
    jitter: bool = True
    retryable_status_codes: List[int] = field(default_factory=lambda: [429, 500, 502, 503, 504])
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt."""
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        
        if self.jitter:
            delay = delay * (0.5 + random.random() * 0.5)
        
        return delay
    
    def should_retry(self, attempt: int, error: Exception) -> bool:
        """Determine if a request should be retried."""
        if attempt >= self.max_retries:
            return False
        
        # Don't retry non-retryable errors
        if isinstance(error, NonRetryableError):
            return False
        
        # Check error type
        error_str = str(error).lower()
        
        # Always retry on these conditions
        retryable_conditions = [
            "timeout" in error_str,
            "connection" in error_str,
            "rate limit" in error_str,
            "temporary" in error_str,
            "500" in error_str,
            "502" in error_str,
            "503" in error_str,
            "504" in error_str,
        ]
        
        return any(retryable_conditions)


@dataclass
class ProviderMetrics:
    """Track provider performance metrics."""
    
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_latency_ms: float = 0.0
    last_request_time: Optional[float] = None
    
    def record_request(self, success: bool, latency_ms: float, tokens: int = 0, cost: float = 0.0):
        """Record a request."""
        self.total_requests += 1
        self.last_request_time = time.time()
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        self.total_tokens += tokens
        self.total_cost += cost
        
        # Update average latency
        if self.total_requests == 1:
            self.avg_latency_ms = latency_ms
        else:
            self.avg_latency_ms = (self.avg_latency_ms * (self.total_requests - 1) + latency_ms) / self.total_requests
    
    def get_success_rate(self) -> float:
        """Get success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.get_success_rate(),
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "avg_latency_ms": self.avg_latency_ms
        }


class ResilientProvider:
    """Wrapper for providers with resilience patterns."""
    
    def __init__(
        self,
        name: str,
        circuit_breaker: Optional[CircuitBreaker] = None,
        retry_policy: Optional[RetryPolicy] = None
    ):
        self.name = name
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.retry_policy = retry_policy or RetryPolicy()
        self.metrics = ProviderMetrics()
    
    def execute_with_resilience(
        self,
        func: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        """Execute a function with circuit breaker and retry logic."""
        last_error = None
        
        for attempt in range(self.retry_policy.max_retries + 1):
            # Check circuit breaker
            if not self.circuit_breaker.can_execute():
                raise NonRetryableError(
                    f"Circuit breaker is open for provider {self.name}"
                )
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                latency_ms = (time.time() - start_time) * 1000
                
                # Record success
                self.circuit_breaker.record_success()
                self.metrics.record_request(True, latency_ms)
                
                return result
            except NonRetryableError:
                raise
            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                last_error = e
                
                # Record failure
                self.circuit_breaker.record_failure()
                self.metrics.record_request(False, latency_ms)
                
                # Check if we should retry
                if not self.retry_policy.should_retry(attempt, e):
                    raise
                
                # Wait before retry
                delay = self.retry_policy.get_delay(attempt)
                logger.warning(
                    "Provider %s call failed (attempt %d/%d), retrying in %.1fs: %s",
                    self.name, attempt + 1, self.retry_policy.max_retries, delay, str(e)
                )
                time.sleep(delay)
        
        # All retries exhausted
        raise last_error or NonRetryableError("All retries exhausted")


# Global circuit breakers for providers
_provider_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(provider_name: str) -> CircuitBreaker:
    """Get or create a circuit breaker for a provider."""
    if provider_name not in _provider_circuit_breakers:
        _provider_circuit_breakers[provider_name] = CircuitBreaker()
    return _provider_circuit_breakers[provider_name]


def get_provider_metrics() -> Dict[str, Dict[str, Any]]:
    """Get metrics for all providers."""
    return {
        name: cb.get_status()
        for name, cb in _provider_circuit_breakers.items()
    }
