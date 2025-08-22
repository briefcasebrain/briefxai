use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::time::sleep;
use tracing::{error, info, instrument, warn};

/// Comprehensive error recovery and resilience system for BriefXAI
#[derive(Debug, Clone)]
pub struct ErrorRecoverySystem {
    retry_policies: HashMap<String, RetryPolicy>,
    circuit_breakers: Arc<Mutex<HashMap<String, CircuitBreaker>>>,
    fallback_strategies: HashMap<String, FallbackStrategy>,
    recovery_stats: Arc<Mutex<RecoveryStatistics>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub initial_delay: Duration,
    pub max_delay: Duration,
    pub backoff_multiplier: f64,
    pub jitter: bool,
    pub retry_on: Vec<ErrorType>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ErrorType {
    NetworkError,
    TimeoutError,
    RateLimitError,
    TemporaryServiceError,
    AuthenticationError,
    QuotaExceededError,
    InternalServerError,
    BadGatewayError,
    ServiceUnavailableError,
}

#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    state: CircuitBreakerState,
    failure_count: u32,
    success_count: u32,
    last_failure_time: Option<Instant>,
    config: CircuitBreakerConfig,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CircuitBreakerState {
    Closed,
    Open,
    HalfOpen,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: u32,
    pub success_threshold: u32,
    pub timeout: Duration,
    pub max_requests_half_open: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FallbackStrategy {
    ReturnCached,
    ReturnDefault,
    ReturnPartialResults,
    SkipNonEssential,
    DegradeService,
    FailFast,
    UseAlternativeProvider,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryStatistics {
    pub total_errors: u64,
    pub recovered_errors: u64,
    pub failed_recoveries: u64,
    pub retry_attempts: u64,
    pub circuit_breaker_trips: u64,
    pub fallback_activations: u64,
    pub recovery_success_rate: f64,
    pub average_recovery_time_ms: f64,
    pub error_breakdown: HashMap<String, u64>,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug)]
pub struct RecoveryResult<T> {
    pub result: Result<T>,
    pub attempts: u32,
    pub total_time: Duration,
    pub recovery_method: Option<RecoveryMethod>,
    pub fallback_used: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryMethod {
    Retry,
    CircuitBreakerBypass,
    FallbackExecution,
    ProviderFailover,
    GracefulDegradation,
}

impl ErrorRecoverySystem {
    pub fn new() -> Self {
        let mut system = Self {
            retry_policies: HashMap::new(),
            circuit_breakers: Arc::new(Mutex::new(HashMap::new())),
            fallback_strategies: HashMap::new(),
            recovery_stats: Arc::new(Mutex::new(RecoveryStatistics::default())),
        };

        // Initialize default policies
        system.setup_default_policies();
        system
    }

    fn setup_default_policies(&mut self) {
        // Default retry policy for API calls
        self.retry_policies.insert(
            "api_call".to_string(),
            RetryPolicy {
                max_attempts: 3,
                initial_delay: Duration::from_millis(500),
                max_delay: Duration::from_secs(30),
                backoff_multiplier: 2.0,
                jitter: true,
                retry_on: vec![
                    ErrorType::NetworkError,
                    ErrorType::TimeoutError,
                    ErrorType::TemporaryServiceError,
                    ErrorType::InternalServerError,
                    ErrorType::BadGatewayError,
                    ErrorType::ServiceUnavailableError,
                ],
            },
        );

        // Retry policy for LLM providers
        self.retry_policies.insert(
            "llm_provider".to_string(),
            RetryPolicy {
                max_attempts: 5,
                initial_delay: Duration::from_millis(1000),
                max_delay: Duration::from_secs(60),
                backoff_multiplier: 1.5,
                jitter: true,
                retry_on: vec![
                    ErrorType::RateLimitError,
                    ErrorType::TemporaryServiceError,
                    ErrorType::TimeoutError,
                    ErrorType::ServiceUnavailableError,
                ],
            },
        );

        // Retry policy for embedding generation
        self.retry_policies.insert(
            "embedding_generation".to_string(),
            RetryPolicy {
                max_attempts: 3,
                initial_delay: Duration::from_millis(2000),
                max_delay: Duration::from_secs(45),
                backoff_multiplier: 2.0,
                jitter: true,
                retry_on: vec![
                    ErrorType::TimeoutError,
                    ErrorType::RateLimitError,
                    ErrorType::TemporaryServiceError,
                ],
            },
        );

        // Setup fallback strategies
        self.fallback_strategies.insert(
            "llm_provider".to_string(),
            FallbackStrategy::UseAlternativeProvider,
        );
        self.fallback_strategies.insert(
            "embedding_generation".to_string(),
            FallbackStrategy::UseAlternativeProvider,
        );
        self.fallback_strategies.insert(
            "clustering".to_string(),
            FallbackStrategy::ReturnPartialResults,
        );
        self.fallback_strategies.insert(
            "facet_extraction".to_string(),
            FallbackStrategy::SkipNonEssential,
        );
    }

    #[instrument(skip(self, operation))]
    pub async fn execute_with_recovery<T, F, Fut>(
        &self,
        operation_name: &str,
        operation: F,
    ) -> RecoveryResult<T>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let start_time = Instant::now();
        let mut attempts = 0;
        let mut last_error = None;
        let mut recovery_method = None;

        // Check circuit breaker
        if let Some(circuit_breaker) = self.get_circuit_breaker(operation_name) {
            if circuit_breaker.state == CircuitBreakerState::Open {
                if let Some(fallback) = self.fallback_strategies.get(operation_name) {
                    return self
                        .execute_fallback(operation_name, fallback.clone(), start_time)
                        .await;
                } else {
                    return RecoveryResult {
                        result: Err(anyhow!("Circuit breaker is open and no fallback available")),
                        attempts: 0,
                        total_time: start_time.elapsed(),
                        recovery_method: None,
                        fallback_used: false,
                    };
                }
            }
        }

        // Get retry policy
        let retry_policy = self
            .retry_policies
            .get(operation_name)
            .cloned()
            .unwrap_or_else(|| self.get_default_retry_policy());

        // Execute with retry logic
        while attempts < retry_policy.max_attempts {
            attempts += 1;

            match operation().await {
                Ok(result) => {
                    // Record success
                    self.record_success(operation_name, attempts, start_time.elapsed())
                        .await;

                    if attempts > 1 {
                        recovery_method = Some(RecoveryMethod::Retry);
                    }

                    return RecoveryResult {
                        result: Ok(result),
                        attempts,
                        total_time: start_time.elapsed(),
                        recovery_method,
                        fallback_used: false,
                    };
                }
                Err(error) => {
                    last_error = Some(anyhow::Error::msg(error.to_string()));
                    let error_type = self.classify_error(&error);

                    // Record failure
                    self.record_failure(operation_name, &error_type).await;

                    // Check if we should retry this error type
                    if !retry_policy.retry_on.contains(&error_type) {
                        break;
                    }

                    // Don't retry on the last attempt
                    if attempts >= retry_policy.max_attempts {
                        break;
                    }

                    // Calculate delay with exponential backoff and jitter
                    let delay = self.calculate_retry_delay(&retry_policy, attempts);

                    info!(
                        "Retrying operation '{}' after error: {} (attempt {}/{})",
                        operation_name, error, attempts, retry_policy.max_attempts
                    );

                    sleep(delay).await;
                }
            }
        }

        // All retries failed, try fallback
        if let Some(fallback) = self.fallback_strategies.get(operation_name) {
            warn!(
                "All retries failed for operation '{}', executing fallback strategy: {:?}",
                operation_name, fallback
            );

            return self
                .execute_fallback(operation_name, fallback.clone(), start_time)
                .await;
        }

        // No fallback available, return the last error
        let final_error =
            last_error.unwrap_or_else(|| anyhow!("Operation failed without specific error"));

        RecoveryResult {
            result: Err(final_error),
            attempts,
            total_time: start_time.elapsed(),
            recovery_method: None,
            fallback_used: false,
        }
    }

    async fn execute_fallback<T>(
        &self,
        operation_name: &str,
        fallback: FallbackStrategy,
        start_time: Instant,
    ) -> RecoveryResult<T> {
        info!(
            "Executing fallback strategy '{:?}' for operation '{}'",
            fallback, operation_name
        );

        self.record_fallback_activation(operation_name).await;

        // Implementation would depend on the specific fallback strategy
        // For now, we'll return an error indicating fallback was attempted
        RecoveryResult {
            result: Err(anyhow!(
                "Fallback strategy {:?} executed but not implemented",
                fallback
            )),
            attempts: 0,
            total_time: start_time.elapsed(),
            recovery_method: Some(RecoveryMethod::FallbackExecution),
            fallback_used: true,
        }
    }

    fn classify_error(&self, error: &anyhow::Error) -> ErrorType {
        let error_string = error.to_string().to_lowercase();

        if error_string.contains("timeout") || error_string.contains("timed out") {
            ErrorType::TimeoutError
        } else if error_string.contains("network") || error_string.contains("connection") {
            ErrorType::NetworkError
        } else if error_string.contains("rate limit") || error_string.contains("too many requests")
        {
            ErrorType::RateLimitError
        } else if error_string.contains("authentication") || error_string.contains("unauthorized") {
            ErrorType::AuthenticationError
        } else if error_string.contains("quota") || error_string.contains("limit exceeded") {
            ErrorType::QuotaExceededError
        } else if error_string.contains("500") || error_string.contains("internal server error") {
            ErrorType::InternalServerError
        } else if error_string.contains("502") || error_string.contains("bad gateway") {
            ErrorType::BadGatewayError
        } else if error_string.contains("503") || error_string.contains("service unavailable") {
            ErrorType::ServiceUnavailableError
        } else {
            ErrorType::TemporaryServiceError
        }
    }

    fn calculate_retry_delay(&self, policy: &RetryPolicy, attempt: u32) -> Duration {
        let base_delay = policy.initial_delay.as_millis() as f64;
        let multiplier = policy.backoff_multiplier.powi((attempt - 1) as i32);
        let mut delay_ms = base_delay * multiplier;

        // Apply jitter if enabled
        if policy.jitter {
            let mut rng = rand::thread_rng();
            let jitter_factor = rng.gen::<f64>() * 0.1; // ±10% jitter
            delay_ms *= 1.0 + (jitter_factor - 0.05);
        }

        // Cap at max delay
        let max_delay_ms = policy.max_delay.as_millis() as f64;
        delay_ms = delay_ms.min(max_delay_ms);

        Duration::from_millis(delay_ms as u64)
    }

    fn get_circuit_breaker(&self, operation_name: &str) -> Option<CircuitBreaker> {
        let breakers = self.circuit_breakers.lock().unwrap();
        breakers.get(operation_name).cloned()
    }

    pub fn add_circuit_breaker(&self, operation_name: &str, config: CircuitBreakerConfig) {
        let mut breakers = self.circuit_breakers.lock().unwrap();
        breakers.insert(
            operation_name.to_string(),
            CircuitBreaker {
                state: CircuitBreakerState::Closed,
                failure_count: 0,
                success_count: 0,
                last_failure_time: None,
                config,
            },
        );
    }

    async fn record_success(&self, operation_name: &str, attempts: u32, duration: Duration) {
        let mut stats = self.recovery_stats.lock().unwrap();

        if attempts > 1 {
            stats.recovered_errors += 1;
            stats.retry_attempts += attempts as u64 - 1;
        }

        // Update circuit breaker
        if let Ok(mut breakers) = self.circuit_breakers.try_lock() {
            if let Some(breaker) = breakers.get_mut(operation_name) {
                breaker.success_count += 1;

                match breaker.state {
                    CircuitBreakerState::HalfOpen => {
                        if breaker.success_count >= breaker.config.success_threshold {
                            breaker.state = CircuitBreakerState::Closed;
                            breaker.failure_count = 0;
                            info!(
                                "Circuit breaker for '{}' moved to CLOSED state",
                                operation_name
                            );
                        }
                    }
                    CircuitBreakerState::Open => {
                        // Check if timeout has passed
                        if let Some(last_failure) = breaker.last_failure_time {
                            if last_failure.elapsed() >= breaker.config.timeout {
                                breaker.state = CircuitBreakerState::HalfOpen;
                                breaker.success_count = 1;
                                info!(
                                    "Circuit breaker for '{}' moved to HALF_OPEN state",
                                    operation_name
                                );
                            }
                        }
                    }
                    CircuitBreakerState::Closed => {
                        breaker.failure_count = 0; // Reset failure count on success
                    }
                }
            }
        }

        self.update_recovery_statistics(&mut stats, duration);
    }

    async fn record_failure(&self, operation_name: &str, error_type: &ErrorType) {
        let mut stats = self.recovery_stats.lock().unwrap();
        stats.total_errors += 1;

        let error_key = format!("{:?}", error_type);
        *stats.error_breakdown.entry(error_key).or_insert(0) += 1;

        // Update circuit breaker
        if let Ok(mut breakers) = self.circuit_breakers.try_lock() {
            if let Some(breaker) = breakers.get_mut(operation_name) {
                breaker.failure_count += 1;
                breaker.last_failure_time = Some(Instant::now());

                if breaker.failure_count >= breaker.config.failure_threshold {
                    breaker.state = CircuitBreakerState::Open;
                    stats.circuit_breaker_trips += 1;
                    warn!(
                        "Circuit breaker for '{}' tripped to OPEN state",
                        operation_name
                    );
                }
            }
        }

        stats.last_updated = Utc::now();
    }

    async fn record_fallback_activation(&self, operation_name: &str) {
        let mut stats = self.recovery_stats.lock().unwrap();
        stats.fallback_activations += 1;
        stats.last_updated = Utc::now();

        info!("Fallback activated for operation: {}", operation_name);
    }

    fn update_recovery_statistics(&self, stats: &mut RecoveryStatistics, duration: Duration) {
        // Update average recovery time
        let total_operations = stats.recovered_errors + stats.failed_recoveries;
        if total_operations > 0 {
            let current_avg = stats.average_recovery_time_ms;
            let new_time = duration.as_millis() as f64;
            stats.average_recovery_time_ms =
                (current_avg * (total_operations - 1) as f64 + new_time) / total_operations as f64;
        }

        // Update success rate
        if stats.total_errors > 0 {
            stats.recovery_success_rate =
                (stats.recovered_errors as f64 / stats.total_errors as f64) * 100.0;
        }

        stats.last_updated = Utc::now();
    }

    fn get_default_retry_policy(&self) -> RetryPolicy {
        RetryPolicy {
            max_attempts: 2,
            initial_delay: Duration::from_millis(1000),
            max_delay: Duration::from_secs(10),
            backoff_multiplier: 1.5,
            jitter: true,
            retry_on: vec![
                ErrorType::NetworkError,
                ErrorType::TimeoutError,
                ErrorType::TemporaryServiceError,
            ],
        }
    }

    pub fn get_statistics(&self) -> RecoveryStatistics {
        self.recovery_stats.lock().unwrap().clone()
    }

    pub fn reset_circuit_breaker(&self, operation_name: &str) -> Result<()> {
        let mut breakers = self.circuit_breakers.lock().unwrap();
        if let Some(breaker) = breakers.get_mut(operation_name) {
            breaker.state = CircuitBreakerState::Closed;
            breaker.failure_count = 0;
            breaker.success_count = 0;
            breaker.last_failure_time = None;
            info!("Circuit breaker for '{}' has been reset", operation_name);
            Ok(())
        } else {
            Err(anyhow!(
                "Circuit breaker not found for operation: {}",
                operation_name
            ))
        }
    }

    pub fn get_circuit_breaker_status(&self, operation_name: &str) -> Option<CircuitBreakerStatus> {
        let breakers = self.circuit_breakers.lock().unwrap();
        breakers
            .get(operation_name)
            .map(|breaker| CircuitBreakerStatus {
                state: breaker.state.clone(),
                failure_count: breaker.failure_count,
                success_count: breaker.success_count,
                last_failure_time: breaker.last_failure_time,
            })
    }

    /// Provider failover for API calls
    pub async fn execute_with_provider_failover<T, F, Fut>(
        &self,
        providers: Vec<String>,
        operation: F,
    ) -> RecoveryResult<T>
    where
        F: Fn(String) -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let start_time = Instant::now();
        let mut attempts = 0;
        let mut last_error = None;

        for provider in providers {
            attempts += 1;

            match operation(provider.clone()).await {
                Ok(result) => {
                    if attempts > 1 {
                        info!("Provider failover successful, used provider: {}", provider);
                    }

                    return RecoveryResult {
                        result: Ok(result),
                        attempts,
                        total_time: start_time.elapsed(),
                        recovery_method: if attempts > 1 {
                            Some(RecoveryMethod::ProviderFailover)
                        } else {
                            None
                        },
                        fallback_used: false,
                    };
                }
                Err(error) => {
                    warn!("Provider {} failed: {}", provider, error);
                    last_error = Some(error);
                }
            }
        }

        // All providers failed
        let final_error = last_error.unwrap_or_else(|| anyhow!("All providers failed"));
        error!("All providers failed after {} attempts", attempts);

        RecoveryResult {
            result: Err(final_error),
            attempts,
            total_time: start_time.elapsed(),
            recovery_method: None,
            fallback_used: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CircuitBreakerStatus {
    pub state: CircuitBreakerState,
    pub failure_count: u32,
    pub success_count: u32,
    pub last_failure_time: Option<Instant>,
}

impl Default for RecoveryStatistics {
    fn default() -> Self {
        Self {
            total_errors: 0,
            recovered_errors: 0,
            failed_recoveries: 0,
            retry_attempts: 0,
            circuit_breaker_trips: 0,
            fallback_activations: 0,
            recovery_success_rate: 0.0,
            average_recovery_time_ms: 0.0,
            error_breakdown: HashMap::new(),
            last_updated: Utc::now(),
        }
    }
}

/// Convenience macro for wrapping operations with error recovery
#[macro_export]
macro_rules! with_recovery {
    ($recovery_system:expr, $operation_name:expr, $operation:expr) => {
        $recovery_system
            .execute_with_recovery($operation_name, || async { $operation })
            .await
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_successful_operation_no_retry() {
        let recovery_system = ErrorRecoverySystem::new();

        let result = recovery_system
            .execute_with_recovery("test_op", || async { Ok::<i32, anyhow::Error>(42) })
            .await;

        assert!(result.result.is_ok());
        assert_eq!(result.result.unwrap(), 42);
        assert_eq!(result.attempts, 1);
        assert!(result.recovery_method.is_none());
    }

    #[tokio::test]
    async fn test_retry_on_failure() {
        let recovery_system = ErrorRecoverySystem::new();
        let attempt_count = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));

        let result = recovery_system
            .execute_with_recovery("api_call", || {
                let count_clone = attempt_count.clone();
                async move {
                    let count = count_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                    if count < 3 {
                        Err(anyhow!("Network error"))
                    } else {
                        Ok(42)
                    }
                }
            })
            .await;

        assert!(result.result.is_ok());
        assert_eq!(result.result.unwrap(), 42);
        assert_eq!(result.attempts, 3);
        assert!(matches!(
            result.recovery_method,
            Some(RecoveryMethod::Retry)
        ));
    }

    #[tokio::test]
    async fn test_circuit_breaker() {
        let recovery_system = ErrorRecoverySystem::new();

        // Add circuit breaker
        recovery_system.add_circuit_breaker(
            "test_cb",
            CircuitBreakerConfig {
                failure_threshold: 2,
                success_threshold: 2,
                timeout: Duration::from_secs(1),
                max_requests_half_open: 1,
            },
        );

        // First two failures should trip the circuit breaker
        for _ in 0..2 {
            let _ = recovery_system
                .execute_with_recovery("test_cb", || async {
                    Err::<i32, anyhow::Error>(anyhow!("Service error"))
                })
                .await;
        }

        // Circuit breaker should now be open
        let status = recovery_system
            .get_circuit_breaker_status("test_cb")
            .unwrap();
        assert!(matches!(status.state, CircuitBreakerState::Open));
    }

    #[tokio::test]
    async fn test_provider_failover() {
        let recovery_system = ErrorRecoverySystem::new();
        let providers = vec![
            "provider1".to_string(),
            "provider2".to_string(),
            "provider3".to_string(),
        ];

        let result = recovery_system
            .execute_with_provider_failover(providers, |provider| async move {
                if provider == "provider3" {
                    Ok(format!("Success with {}", provider))
                } else {
                    Err(anyhow!("Provider {} failed", provider))
                }
            })
            .await;

        assert!(result.result.is_ok());
        assert_eq!(result.result.unwrap(), "Success with provider3");
        assert_eq!(result.attempts, 3);
        assert!(matches!(
            result.recovery_method,
            Some(RecoveryMethod::ProviderFailover)
        ));
    }

    #[tokio::test]
    async fn test_error_classification() {
        let recovery_system = ErrorRecoverySystem::new();

        assert!(matches!(
            recovery_system.classify_error(&anyhow!("Connection timeout")),
            ErrorType::TimeoutError
        ));

        assert!(matches!(
            recovery_system.classify_error(&anyhow!("Rate limit exceeded")),
            ErrorType::RateLimitError
        ));

        assert!(matches!(
            recovery_system.classify_error(&anyhow!("Network connection failed")),
            ErrorType::NetworkError
        ));
    }

    #[tokio::test]
    async fn test_statistics_tracking() {
        let recovery_system = ErrorRecoverySystem::new();

        // Simulate some operations
        let _ = recovery_system
            .execute_with_recovery("test", || async {
                Err::<i32, anyhow::Error>(anyhow!("Network error"))
            })
            .await;

        let stats = recovery_system.get_statistics();
        assert_eq!(stats.total_errors, 2); // Includes retries
        assert!(stats.error_breakdown.contains_key("NetworkError"));
    }
}
