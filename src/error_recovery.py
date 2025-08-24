"""
Comprehensive error recovery and resilience system for BriefX
Provides retry policies, circuit breakers, and fallback strategies
"""

import asyncio
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ErrorType(Enum):
    """Classification of error types for recovery decisions"""
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    TEMPORARY_SERVICE_ERROR = "temporary_service_error"
    AUTHENTICATION_ERROR = "authentication_error"
    QUOTA_EXCEEDED_ERROR = "quota_exceeded_error"
    INTERNAL_SERVER_ERROR = "internal_server_error"
    BAD_GATEWAY_ERROR = "bad_gateway_error"
    SERVICE_UNAVAILABLE_ERROR = "service_unavailable_error"


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class FallbackStrategy(Enum):
    """Fallback strategies when primary operation fails"""
    RETURN_CACHED = "return_cached"
    RETURN_DEFAULT = "return_default"
    RETURN_PARTIAL_RESULTS = "return_partial_results"
    SKIP_NON_ESSENTIAL = "skip_non_essential"
    DEGRADE_SERVICE = "degrade_service"
    FAIL_FAST = "fail_fast"
    USE_ALTERNATIVE_PROVIDER = "use_alternative_provider"


class RecoveryMethod(Enum):
    """Methods used for recovery"""
    RETRY = "retry"
    CIRCUIT_BREAKER_BYPASS = "circuit_breaker_bypass"
    FALLBACK_EXECUTION = "fallback_execution"
    PROVIDER_FAILOVER = "provider_failover"
    GRACEFUL_DEGRADATION = "graceful_degradation"


@dataclass
class RetryPolicy:
    """Configuration for retry behavior"""
    max_attempts: int = 3
    initial_delay: float = 0.5  # seconds
    max_delay: float = 30.0  # seconds
    backoff_multiplier: float = 2.0
    jitter: bool = True
    retry_on: List[ErrorType] = field(default_factory=lambda: [
        ErrorType.NETWORK_ERROR,
        ErrorType.TIMEOUT_ERROR,
        ErrorType.TEMPORARY_SERVICE_ERROR,
    ])


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 60.0  # seconds
    max_requests_half_open: int = 1


@dataclass
class CircuitBreaker:
    """Circuit breaker implementation"""
    config: CircuitBreakerConfig
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    half_open_requests: int = 0


@dataclass
class RecoveryStatistics:
    """Statistics for error recovery system"""
    total_errors: int = 0
    recovered_errors: int = 0
    failed_recoveries: int = 0
    retry_attempts: int = 0
    circuit_breaker_trips: int = 0
    fallback_activations: int = 0
    recovery_success_rate: float = 0.0
    average_recovery_time_ms: float = 0.0
    error_breakdown: Dict[str, int] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RecoveryResult:
    """Result of an operation with recovery"""
    result: Optional[Any] = None
    error: Optional[Exception] = None
    success: bool = False
    attempts: int = 0
    total_time: float = 0.0
    recovery_method: Optional[RecoveryMethod] = None
    fallback_used: bool = False


class ErrorRecoverySystem:
    """Main error recovery system"""
    
    def __init__(self):
        self.retry_policies: Dict[str, RetryPolicy] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.fallback_strategies: Dict[str, FallbackStrategy] = {}
        self.recovery_stats = RecoveryStatistics()
        self.cache: Dict[str, Any] = {}  # Simple cache for fallback
        self._setup_default_policies()
    
    def _setup_default_policies(self):
        """Initialize default retry policies and fallback strategies"""
        
        # Default retry policy for API calls
        self.retry_policies["api_call"] = RetryPolicy(
            max_attempts=3,
            initial_delay=0.5,
            max_delay=30.0,
            backoff_multiplier=2.0,
            jitter=True,
            retry_on=[
                ErrorType.NETWORK_ERROR,
                ErrorType.TIMEOUT_ERROR,
                ErrorType.TEMPORARY_SERVICE_ERROR,
                ErrorType.INTERNAL_SERVER_ERROR,
                ErrorType.BAD_GATEWAY_ERROR,
                ErrorType.SERVICE_UNAVAILABLE_ERROR,
            ]
        )
        
        # Retry policy for LLM providers
        self.retry_policies["llm_provider"] = RetryPolicy(
            max_attempts=5,
            initial_delay=1.0,
            max_delay=60.0,
            backoff_multiplier=1.5,
            jitter=True,
            retry_on=[
                ErrorType.RATE_LIMIT_ERROR,
                ErrorType.TEMPORARY_SERVICE_ERROR,
                ErrorType.TIMEOUT_ERROR,
                ErrorType.SERVICE_UNAVAILABLE_ERROR,
            ]
        )
        
        # Retry policy for embedding generation
        self.retry_policies["embedding_generation"] = RetryPolicy(
            max_attempts=3,
            initial_delay=2.0,
            max_delay=45.0,
            backoff_multiplier=2.0,
            jitter=True,
            retry_on=[
                ErrorType.TIMEOUT_ERROR,
                ErrorType.RATE_LIMIT_ERROR,
                ErrorType.TEMPORARY_SERVICE_ERROR,
            ]
        )
        
        # Setup fallback strategies
        self.fallback_strategies["llm_provider"] = FallbackStrategy.USE_ALTERNATIVE_PROVIDER
        self.fallback_strategies["embedding_generation"] = FallbackStrategy.USE_ALTERNATIVE_PROVIDER
        self.fallback_strategies["clustering"] = FallbackStrategy.RETURN_PARTIAL_RESULTS
        self.fallback_strategies["facet_extraction"] = FallbackStrategy.SKIP_NON_ESSENTIAL
    
    async def execute_with_recovery(
        self,
        operation_name: str,
        operation: Callable,
        *args,
        **kwargs
    ) -> RecoveryResult:
        """Execute an operation with error recovery"""
        
        start_time = time.time()
        attempts = 0
        last_error = None
        recovery_method = None
        
        # Check circuit breaker
        circuit_breaker = self.circuit_breakers.get(operation_name)
        if circuit_breaker:
            if circuit_breaker.state == CircuitBreakerState.OPEN:
                # Check if timeout has passed
                if circuit_breaker.last_failure_time:
                    elapsed = time.time() - circuit_breaker.last_failure_time
                    if elapsed >= circuit_breaker.config.timeout:
                        circuit_breaker.state = CircuitBreakerState.HALF_OPEN
                        circuit_breaker.half_open_requests = 0
                        logger.info(f"Circuit breaker for '{operation_name}' moved to HALF_OPEN")
                    else:
                        # Still open, try fallback
                        return await self._execute_fallback(
                            operation_name, start_time, args, kwargs
                        )
                else:
                    return await self._execute_fallback(
                        operation_name, start_time, args, kwargs
                    )
            
            elif circuit_breaker.state == CircuitBreakerState.HALF_OPEN:
                if circuit_breaker.half_open_requests >= circuit_breaker.config.max_requests_half_open:
                    # Too many half-open requests, wait
                    await asyncio.sleep(0.5)
                circuit_breaker.half_open_requests += 1
        
        # Get retry policy
        retry_policy = self.retry_policies.get(
            operation_name,
            self._get_default_retry_policy()
        )
        
        # Execute with retry logic
        while attempts < retry_policy.max_attempts:
            attempts += 1
            
            try:
                # Execute the operation
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = operation(*args, **kwargs)
                
                # Record success
                await self._record_success(operation_name, attempts, start_time)
                
                if attempts > 1:
                    recovery_method = RecoveryMethod.RETRY
                
                return RecoveryResult(
                    result=result,
                    success=True,
                    attempts=attempts,
                    total_time=time.time() - start_time,
                    recovery_method=recovery_method,
                    fallback_used=False
                )
                
            except Exception as error:
                last_error = error
                error_type = self._classify_error(error)
                
                # Record failure
                await self._record_failure(operation_name, error_type)
                
                # Check if we should retry this error type
                if error_type not in retry_policy.retry_on:
                    break
                
                # Don't retry on the last attempt
                if attempts >= retry_policy.max_attempts:
                    break
                
                # Calculate delay with exponential backoff and jitter
                delay = self._calculate_retry_delay(retry_policy, attempts)
                
                logger.info(
                    f"Retrying operation '{operation_name}' after error: {error} "
                    f"(attempt {attempts}/{retry_policy.max_attempts})"
                )
                
                await asyncio.sleep(delay)
        
        # All retries failed, try fallback
        fallback_strategy = self.fallback_strategies.get(operation_name)
        if fallback_strategy:
            logger.warning(
                f"All retries failed for operation '{operation_name}', "
                f"executing fallback strategy: {fallback_strategy}"
            )
            return await self._execute_fallback(
                operation_name, start_time, args, kwargs, last_error
            )
        
        # No fallback available, return error
        return RecoveryResult(
            error=last_error,
            success=False,
            attempts=attempts,
            total_time=time.time() - start_time,
            recovery_method=None,
            fallback_used=False
        )
    
    async def _execute_fallback(
        self,
        operation_name: str,
        start_time: float,
        args: tuple,
        kwargs: dict,
        error: Optional[Exception] = None
    ) -> RecoveryResult:
        """Execute fallback strategy"""
        
        strategy = self.fallback_strategies.get(operation_name, FallbackStrategy.FAIL_FAST)
        
        logger.info(f"Executing fallback strategy '{strategy}' for operation '{operation_name}'")
        await self._record_fallback_activation(operation_name)
        
        result = None
        fallback_error = None
        
        try:
            if strategy == FallbackStrategy.RETURN_CACHED:
                # Return cached result if available
                cache_key = f"{operation_name}:{str(args)}:{str(kwargs)}"
                result = self.cache.get(cache_key)
                if result is None:
                    fallback_error = Exception("No cached result available")
                    
            elif strategy == FallbackStrategy.RETURN_DEFAULT:
                # Return a default value based on operation
                result = self._get_default_value(operation_name)
                
            elif strategy == FallbackStrategy.RETURN_PARTIAL_RESULTS:
                # Return partial results (operation-specific)
                result = {"partial": True, "data": [], "error": str(error)}
                
            elif strategy == FallbackStrategy.SKIP_NON_ESSENTIAL:
                # Skip the operation and return None
                result = None
                
            elif strategy == FallbackStrategy.DEGRADE_SERVICE:
                # Return degraded service response
                result = {"degraded": True, "reason": str(error)}
                
            elif strategy == FallbackStrategy.USE_ALTERNATIVE_PROVIDER:
                # This would be handled by provider failover
                fallback_error = Exception("Alternative provider failover not implemented here")
                
            elif strategy == FallbackStrategy.FAIL_FAST:
                # Fail immediately
                fallback_error = error or Exception("Operation failed - fail fast strategy")
                
        except Exception as e:
            fallback_error = e
        
        return RecoveryResult(
            result=result,
            error=fallback_error,
            success=fallback_error is None,
            attempts=0,
            total_time=time.time() - start_time,
            recovery_method=RecoveryMethod.FALLBACK_EXECUTION if fallback_error is None else None,
            fallback_used=True
        )
    
    async def execute_with_provider_failover(
        self,
        providers: List[str],
        operation: Callable,
        *args,
        **kwargs
    ) -> RecoveryResult:
        """Execute operation with provider failover"""
        
        start_time = time.time()
        attempts = 0
        last_error = None
        
        for provider in providers:
            attempts += 1
            
            try:
                # Add provider to kwargs
                kwargs['provider'] = provider
                
                # Execute operation
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = operation(*args, **kwargs)
                
                if attempts > 1:
                    logger.info(f"Provider failover successful, used provider: {provider}")
                
                return RecoveryResult(
                    result=result,
                    success=True,
                    attempts=attempts,
                    total_time=time.time() - start_time,
                    recovery_method=RecoveryMethod.PROVIDER_FAILOVER if attempts > 1 else None,
                    fallback_used=False
                )
                
            except Exception as error:
                logger.warning(f"Provider {provider} failed: {error}")
                last_error = error
        
        # All providers failed
        logger.error(f"All providers failed after {attempts} attempts")
        
        return RecoveryResult(
            error=last_error,
            success=False,
            attempts=attempts,
            total_time=time.time() - start_time,
            recovery_method=None,
            fallback_used=False
        )
    
    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify error type for recovery decisions"""
        
        error_str = str(error).lower()
        
        if "timeout" in error_str or "timed out" in error_str:
            return ErrorType.TIMEOUT_ERROR
        elif "network" in error_str or "connection" in error_str:
            return ErrorType.NETWORK_ERROR
        elif "rate limit" in error_str or "too many requests" in error_str:
            return ErrorType.RATE_LIMIT_ERROR
        elif "authentication" in error_str or "unauthorized" in error_str:
            return ErrorType.AUTHENTICATION_ERROR
        elif "quota" in error_str or "limit exceeded" in error_str:
            return ErrorType.QUOTA_EXCEEDED_ERROR
        elif "500" in error_str or "internal server error" in error_str:
            return ErrorType.INTERNAL_SERVER_ERROR
        elif "502" in error_str or "bad gateway" in error_str:
            return ErrorType.BAD_GATEWAY_ERROR
        elif "503" in error_str or "service unavailable" in error_str:
            return ErrorType.SERVICE_UNAVAILABLE_ERROR
        else:
            return ErrorType.TEMPORARY_SERVICE_ERROR
    
    def _calculate_retry_delay(self, policy: RetryPolicy, attempt: int) -> float:
        """Calculate retry delay with exponential backoff and jitter"""
        
        base_delay = policy.initial_delay
        multiplier = policy.backoff_multiplier ** (attempt - 1)
        delay = base_delay * multiplier
        
        # Apply jitter if enabled
        if policy.jitter:
            jitter_factor = random.random() * 0.1  # ±10% jitter
            delay *= (1.0 + jitter_factor - 0.05)
        
        # Cap at max delay
        delay = min(delay, policy.max_delay)
        
        return delay
    
    async def _record_success(self, operation_name: str, attempts: int, start_time: float):
        """Record successful operation"""
        
        if attempts > 1:
            self.recovery_stats.recovered_errors += 1
            self.recovery_stats.retry_attempts += attempts - 1
        
        # Update circuit breaker
        circuit_breaker = self.circuit_breakers.get(operation_name)
        if circuit_breaker:
            circuit_breaker.success_count += 1
            
            if circuit_breaker.state == CircuitBreakerState.HALF_OPEN:
                if circuit_breaker.success_count >= circuit_breaker.config.success_threshold:
                    circuit_breaker.state = CircuitBreakerState.CLOSED
                    circuit_breaker.failure_count = 0
                    logger.info(f"Circuit breaker for '{operation_name}' moved to CLOSED")
            elif circuit_breaker.state == CircuitBreakerState.CLOSED:
                circuit_breaker.failure_count = 0
        
        self._update_recovery_statistics(time.time() - start_time)
    
    async def _record_failure(self, operation_name: str, error_type: ErrorType):
        """Record failed operation"""
        
        self.recovery_stats.total_errors += 1
        
        error_key = error_type.value
        if error_key not in self.recovery_stats.error_breakdown:
            self.recovery_stats.error_breakdown[error_key] = 0
        self.recovery_stats.error_breakdown[error_key] += 1
        
        # Update circuit breaker
        circuit_breaker = self.circuit_breakers.get(operation_name)
        if circuit_breaker:
            circuit_breaker.failure_count += 1
            circuit_breaker.last_failure_time = time.time()
            
            if circuit_breaker.failure_count >= circuit_breaker.config.failure_threshold:
                if circuit_breaker.state != CircuitBreakerState.OPEN:
                    circuit_breaker.state = CircuitBreakerState.OPEN
                    self.recovery_stats.circuit_breaker_trips += 1
                    logger.warning(f"Circuit breaker for '{operation_name}' tripped to OPEN")
        
        self.recovery_stats.last_updated = datetime.utcnow()
    
    async def _record_fallback_activation(self, operation_name: str):
        """Record fallback activation"""
        
        self.recovery_stats.fallback_activations += 1
        self.recovery_stats.last_updated = datetime.utcnow()
        logger.info(f"Fallback activated for operation: {operation_name}")
    
    def _update_recovery_statistics(self, duration: float):
        """Update recovery statistics"""
        
        # Update average recovery time
        total_operations = self.recovery_stats.recovered_errors + self.recovery_stats.failed_recoveries
        if total_operations > 0:
            current_avg = self.recovery_stats.average_recovery_time_ms
            new_time = duration * 1000  # Convert to milliseconds
            self.recovery_stats.average_recovery_time_ms = (
                (current_avg * (total_operations - 1) + new_time) / total_operations
            )
        
        # Update success rate
        if self.recovery_stats.total_errors > 0:
            self.recovery_stats.recovery_success_rate = (
                (self.recovery_stats.recovered_errors / self.recovery_stats.total_errors) * 100.0
            )
        
        self.recovery_stats.last_updated = datetime.utcnow()
    
    def _get_default_retry_policy(self) -> RetryPolicy:
        """Get default retry policy"""
        
        return RetryPolicy(
            max_attempts=2,
            initial_delay=1.0,
            max_delay=10.0,
            backoff_multiplier=1.5,
            jitter=True,
            retry_on=[
                ErrorType.NETWORK_ERROR,
                ErrorType.TIMEOUT_ERROR,
                ErrorType.TEMPORARY_SERVICE_ERROR,
            ]
        )
    
    def _get_default_value(self, operation_name: str) -> Any:
        """Get default value for operation"""
        
        defaults = {
            "embedding_generation": [],
            "facet_extraction": [],
            "clustering": {"clusters": [], "error": "Fallback to empty clusters"},
            "llm_provider": {"response": "", "error": "Fallback to empty response"}
        }
        
        return defaults.get(operation_name, None)
    
    def add_circuit_breaker(self, operation_name: str, config: CircuitBreakerConfig):
        """Add a circuit breaker for an operation"""
        
        self.circuit_breakers[operation_name] = CircuitBreaker(config=config)
    
    def reset_circuit_breaker(self, operation_name: str):
        """Reset a circuit breaker"""
        
        if operation_name in self.circuit_breakers:
            breaker = self.circuit_breakers[operation_name]
            breaker.state = CircuitBreakerState.CLOSED
            breaker.failure_count = 0
            breaker.success_count = 0
            breaker.last_failure_time = None
            logger.info(f"Circuit breaker for '{operation_name}' has been reset")
    
    def get_circuit_breaker_status(self, operation_name: str) -> Optional[Dict[str, Any]]:
        """Get circuit breaker status"""
        
        breaker = self.circuit_breakers.get(operation_name)
        if breaker:
            return {
                "state": breaker.state.value,
                "failure_count": breaker.failure_count,
                "success_count": breaker.success_count,
                "last_failure_time": breaker.last_failure_time
            }
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get recovery statistics"""
        
        return {
            "total_errors": self.recovery_stats.total_errors,
            "recovered_errors": self.recovery_stats.recovered_errors,
            "failed_recoveries": self.recovery_stats.failed_recoveries,
            "retry_attempts": self.recovery_stats.retry_attempts,
            "circuit_breaker_trips": self.recovery_stats.circuit_breaker_trips,
            "fallback_activations": self.recovery_stats.fallback_activations,
            "recovery_success_rate": self.recovery_stats.recovery_success_rate,
            "average_recovery_time_ms": self.recovery_stats.average_recovery_time_ms,
            "error_breakdown": self.recovery_stats.error_breakdown,
            "last_updated": self.recovery_stats.last_updated.isoformat()
        }
    
    def cache_result(self, operation_name: str, args: tuple, kwargs: dict, result: Any):
        """Cache a result for fallback use"""
        
        cache_key = f"{operation_name}:{str(args)}:{str(kwargs)}"
        self.cache[cache_key] = result


# Global error recovery system instance
error_recovery_system = ErrorRecoverySystem()


# Convenience decorator for wrapping functions with error recovery
def with_recovery(operation_name: str):
    """Decorator to wrap functions with error recovery"""
    
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            result = await error_recovery_system.execute_with_recovery(
                operation_name, func, *args, **kwargs
            )
            if result.success:
                return result.result
            else:
                raise result.error
        
        def sync_wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(
                error_recovery_system.execute_with_recovery(
                    operation_name, func, *args, **kwargs
                )
            )
            if result.success:
                return result.result
            else:
                raise result.error
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator