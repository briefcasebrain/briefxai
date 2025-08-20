use anyhow::{Result, Context, bail};
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};
use std::collections::{HashMap, VecDeque};
use tracing::{info, warn, error, debug};
use futures;

use crate::llm::LlmProviderTrait;
use crate::persistence_v2::{ProviderConfig, ProviderType};

// ============================================================================
// Health Check Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub is_healthy: bool,
    #[serde(skip)]
    pub last_check: Option<Instant>,
    pub consecutive_failures: u32,
    pub average_latency_ms: f64,
    pub success_rate: f64,
    pub error_message: Option<String>,
}

impl Default for HealthStatus {
    fn default() -> Self {
        Self {
            is_healthy: true,
            last_check: Some(Instant::now()),
            consecutive_failures: 0,
            average_latency_ms: 0.0,
            success_rate: 1.0,
            error_message: None,
        }
    }
}

// ============================================================================
// Load Balancing Strategies
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastLatency,
    Random,
    Priority,
    CostOptimized,
}

pub struct LoadBalancer {
    strategy: LoadBalancingStrategy,
    round_robin_index: Arc<RwLock<usize>>,
}

impl LoadBalancer {
    pub fn new(strategy: LoadBalancingStrategy) -> Self {
        Self {
            strategy,
            round_robin_index: Arc::new(RwLock::new(0)),
        }
    }
    
    pub async fn select_provider(
        &self,
        providers: &[ManagedProvider],
    ) -> Option<usize> {
        let healthy_providers: Vec<usize> = providers
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                if futures::executor::block_on(p.is_available()).unwrap_or(false) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        
        if healthy_providers.is_empty() {
            return None;
        }
        
        match self.strategy {
            LoadBalancingStrategy::RoundRobin => {
                let mut index = self.round_robin_index.write().await;
                let selected = healthy_providers[*index % healthy_providers.len()];
                *index = (*index + 1) % healthy_providers.len();
                Some(selected)
            }
            LoadBalancingStrategy::LeastLatency => {
                // For LeastLatency, we need to handle async, so we'll use a simple fallback
                // In a real implementation, you'd want to restructure this to be fully async
                healthy_providers.into_iter().next()
            }
            LoadBalancingStrategy::Random => {
                use rand::Rng;
                let mut rng = rand::thread_rng();
                let index = rng.gen_range(0..healthy_providers.len());
                Some(healthy_providers[index])
            }
            LoadBalancingStrategy::Priority => {
                healthy_providers
                    .into_iter()
                    .min_by_key(|&i| providers[i].config.priority)
            }
            LoadBalancingStrategy::CostOptimized => {
                // Select cheapest available provider
                healthy_providers
                    .into_iter()
                    .min_by_key(|&i| {
                        providers[i].config.cost_per_token
                            .as_ref()
                            .and_then(|c| c.get("input"))
                            .and_then(|v| v.as_f64())
                            .map(|cost| (cost * 10000.0) as u64)
                            .unwrap_or(u64::MAX)
                    })
            }
        }
    }
}

// ============================================================================
// Circuit Breaker
// ============================================================================

#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    failure_threshold: u32,
    recovery_timeout: Duration,
    half_open_requests: u32,
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            recovery_timeout: Duration::from_secs(60),
            half_open_requests: 3,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CircuitState {
    Closed,     // Normal operation
    Open,       // Failing, reject requests
    HalfOpen,   // Testing recovery
}

// ============================================================================
// Managed Provider
// ============================================================================

#[derive(Clone)]
pub struct ManagedProvider {
    pub config: ProviderConfig,
    provider: Arc<Box<dyn LlmProviderTrait>>,
    health_status: Arc<RwLock<HealthStatus>>,
    circuit_breaker: CircuitBreaker,
    circuit_state: Arc<RwLock<CircuitState>>,
    last_state_change: Arc<RwLock<Instant>>,
    rate_limiter: Option<Arc<Semaphore>>,
    latency_history: Arc<RwLock<VecDeque<f64>>>,
}

impl ManagedProvider {
    pub fn new(
        config: ProviderConfig,
        provider: Box<dyn LlmProviderTrait>,
    ) -> Self {
        let rate_limiter = config.rate_limit.as_ref()
            .and_then(|r| r.get("requests_per_minute"))
            .and_then(|v| v.as_u64())
            .map(|rpm| Arc::new(Semaphore::new((rpm / 60).max(1) as usize)));
        
        Self {
            config,
            provider: Arc::new(provider),
            health_status: Arc::new(RwLock::new(HealthStatus::default())),
            circuit_breaker: CircuitBreaker::default(),
            circuit_state: Arc::new(RwLock::new(CircuitState::Closed)),
            last_state_change: Arc::new(RwLock::new(Instant::now())),
            rate_limiter,
            latency_history: Arc::new(RwLock::new(VecDeque::with_capacity(100))),
        }
    }
    
    pub async fn is_available(&self) -> Result<bool> {
        let state = self.circuit_state.read().await;
        let health = self.health_status.read().await;
        
        match *state {
            CircuitState::Open => {
                // Check if we should transition to half-open
                let last_change = *self.last_state_change.read().await;
                if last_change.elapsed() > self.circuit_breaker.recovery_timeout {
                    drop(state);
                    *self.circuit_state.write().await = CircuitState::HalfOpen;
                    *self.last_state_change.write().await = Instant::now();
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
            CircuitState::HalfOpen | CircuitState::Closed => {
                Ok(self.config.is_active && health.is_healthy)
            }
        }
    }
    
    pub async fn execute<T>(
        &self,
        operation: impl FnOnce(Arc<Box<dyn LlmProviderTrait>>) -> T,
    ) -> Result<T::Output>
    where
        T: std::future::Future,
        T::Output: Sized,
    {
        // Check circuit breaker
        if !self.is_available().await? {
            bail!("Provider {} is not available", self.config.name);
        }
        
        // Apply rate limiting if configured
        let _permit = if let Some(ref limiter) = self.rate_limiter {
            Some(limiter.acquire().await?)
        } else {
            None
        };
        
        let start = Instant::now();
        let provider = self.provider.clone();
        
        let result = operation(provider).await;
        self.record_success(start.elapsed()).await;
        Ok(result)
    }
    
    async fn record_success(&self, latency: Duration) {
        let mut health = self.health_status.write().await;
        health.consecutive_failures = 0;
        health.error_message = None;
        
        // Update latency history
        let mut history = self.latency_history.write().await;
        let latency_ms = latency.as_millis() as f64;
        history.push_back(latency_ms);
        if history.len() > 100 {
            history.pop_front();
        }
        
        // Calculate average latency
        health.average_latency_ms = history.iter().sum::<f64>() / history.len() as f64;
        
        // Update success rate
        health.success_rate = (health.success_rate * 0.95) + 0.05;
        
        // Update circuit breaker
        let mut state = self.circuit_state.write().await;
        if *state == CircuitState::HalfOpen {
            *state = CircuitState::Closed;
            *self.last_state_change.write().await = Instant::now();
            info!("Provider {} circuit breaker closed", self.config.name);
        }
    }
    
    async fn record_failure(&self, error: String) {
        let mut health = self.health_status.write().await;
        health.consecutive_failures += 1;
        health.error_message = Some(error);
        health.success_rate = health.success_rate * 0.95;
        
        // Check if we should open the circuit
        if health.consecutive_failures >= self.circuit_breaker.failure_threshold {
            let mut state = self.circuit_state.write().await;
            if *state != CircuitState::Open {
                *state = CircuitState::Open;
                *self.last_state_change.write().await = Instant::now();
                health.is_healthy = false;
                error!("Provider {} circuit breaker opened after {} failures", 
                    self.config.name, health.consecutive_failures);
            }
        }
    }
    
    pub async fn health_check(&self) -> Result<HealthStatus> {
        debug!("Performing health check for provider {}", self.config.name);
        
        let start = Instant::now();
        let test_prompt = "Say 'ok' if you're working.";
        
        match self.provider.complete(test_prompt, &Default::default()).await {
            Ok(_) => {
                self.record_success(start.elapsed()).await;
                let health = self.health_status.read().await;
                Ok(health.clone())
            }
            Err(e) => {
                self.record_failure(e.to_string()).await;
                let health = self.health_status.read().await;
                Ok(health.clone())
            }
        }
    }
}

// ============================================================================
// Provider Manager
// ============================================================================

pub struct ProviderManager {
    providers: Vec<ManagedProvider>,
    fallback_chain: Vec<usize>,
    load_balancer: LoadBalancer,
    health_check_interval: Duration,
    persistence: Arc<crate::persistence_v2::EnhancedPersistenceLayer>,
}

impl ProviderManager {
    pub async fn new(
        persistence: Arc<crate::persistence_v2::EnhancedPersistenceLayer>,
        load_balancing_strategy: LoadBalancingStrategy,
    ) -> Result<Self> {
        // Load provider configurations from database
        let configs = persistence.provider_manager().get_active_providers().await?;
        
        let mut providers = Vec::new();
        let mut fallback_chain = Vec::new();
        
        for (i, config) in configs.into_iter().enumerate() {
            // Create provider instance based on type
            let provider = Self::create_provider(&config).await?;
            providers.push(ManagedProvider::new(config.clone(), provider));
            
            if config.is_fallback {
                fallback_chain.push(i);
            }
        }
        
        // Sort fallback chain by priority
        fallback_chain.sort_by_key(|&i| providers[i].config.priority);
        
        let manager = Self {
            providers,
            fallback_chain,
            load_balancer: LoadBalancer::new(load_balancing_strategy),
            health_check_interval: Duration::from_secs(60),
            persistence,
        };
        
        // Start health check task
        manager.start_health_checks();
        
        Ok(manager)
    }
    
    async fn create_provider(config: &ProviderConfig) -> Result<Box<dyn LlmProviderTrait>> {
        match config.provider_type {
            ProviderType::OpenAI => {
                let api_key = config.config.get("api_key")
                    .and_then(|v| v.as_str())
                    .context("OpenAI API key not found")?;
                let model = config.config.get("model")
                    .and_then(|v| v.as_str())
                    .unwrap_or("gpt-4o-mini");
                
                let provider = crate::llm::OpenAIProvider::new(
                    model.to_string(),
                    Some(api_key.to_string()),
                )?;
                Ok(Box::new(provider))
            }
            ProviderType::Ollama => {
                let model = config.config.get("model")
                    .and_then(|v| v.as_str())
                    .unwrap_or("llama2");
                
                let provider = crate::llm::OllamaProvider::new(
                    model.to_string(),
                ).await?;
                Ok(Box::new(provider))
            }
            _ => bail!("Provider type {:?} not yet implemented", config.provider_type),
        }
    }
    
    pub async fn execute_with_fallback(&self, prompt: &str) -> Result<String> {
        // Try primary providers first
        if let Some(index) = self.load_balancer.select_provider(&self.providers).await {
            let provider = &self.providers[index];
            
            match provider.execute(|p| async move {
                p.complete(prompt, &Default::default()).await
            }).await {
                Ok(result) => {
                    self.record_usage(&provider.config.id, true, None).await;
                    return result;
                }
                Err(e) => {
                    warn!("Provider {} failed: {}", provider.config.name, e);
                    self.record_usage(&provider.config.id, false, Some(e.to_string())).await;
                }
            }
        }
        
        // Try fallback chain
        for &index in &self.fallback_chain {
            let provider = &self.providers[index];
            
            if !provider.is_available().await.unwrap_or(false) {
                continue;
            }
            
            match provider.execute(|p| async move {
                p.complete(prompt, &Default::default()).await
            }).await {
                Ok(result) => {
                    info!("Fallback provider {} succeeded", provider.config.name);
                    self.record_usage(&provider.config.id, true, None).await;
                    return result;
                }
                Err(e) => {
                    warn!("Fallback provider {} failed: {}", provider.config.name, e);
                    self.record_usage(&provider.config.id, false, Some(e.to_string())).await;
                }
            }
        }
        
        bail!("All providers failed")
    }
    
    pub async fn get_cheapest_available(&self) -> Option<&ManagedProvider> {
        let mut cheapest: Option<&ManagedProvider> = None;
        let mut min_cost = f64::MAX;
        
        for provider in &self.providers {
            if !provider.is_available().await.unwrap_or(false) {
                continue;
            }
            
            if let Some(cost) = provider.config.cost_per_token
                .as_ref()
                .and_then(|c| c.get("input"))
                .and_then(|v| v.as_f64()) {
                
                if cost < min_cost {
                    min_cost = cost;
                    cheapest = Some(provider);
                }
            }
        }
        
        cheapest
    }
    
    pub async fn health_check_all(&self) -> HashMap<String, HealthStatus> {
        let mut results = HashMap::new();
        
        for provider in &self.providers {
            let status = provider.health_check().await.unwrap_or_else(|_| {
                let mut status = HealthStatus::default();
                status.is_healthy = false;
                status.error_message = Some("Health check failed".to_string());
                status
            });
            
            results.insert(provider.config.name.clone(), status);
        }
        
        results
    }
    
    fn start_health_checks(&self) {
        let providers = self.providers.clone();
        let interval = self.health_check_interval;
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(interval);
            
            loop {
                interval.tick().await;
                
                for provider in &providers {
                    if let Err(e) = provider.health_check().await {
                        error!("Health check failed for {}: {}", provider.config.name, e);
                    }
                }
            }
        });
    }
    
    async fn record_usage(&self, provider_id: &str, success: bool, error: Option<String>) {
        // Record to database for analytics
        if let Err(e) = self.persistence.provider_manager().record_usage(
            provider_id,
            None,
            0,  // Token count would be calculated properly
            None,
            success,
            error.as_deref(),
            0,  // Response time would be measured
        ).await {
            warn!("Failed to record provider usage: {}", e);
        }
    }
    
    pub fn get_provider_stats(&self) -> Vec<ProviderStats> {
        self.providers
            .iter()
            .map(|p| {
                let health = futures::executor::block_on(p.health_status.read());
                ProviderStats {
                    name: p.config.name.clone(),
                    provider_type: p.config.provider_type.clone(),
                    is_active: p.config.is_active,
                    is_healthy: health.is_healthy,
                    success_rate: health.success_rate,
                    average_latency_ms: health.average_latency_ms,
                    consecutive_failures: health.consecutive_failures,
                }
            })
            .collect()
    }
}

// ============================================================================
// Statistics
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderStats {
    pub name: String,
    pub provider_type: ProviderType,
    pub is_active: bool,
    pub is_healthy: bool,
    pub success_rate: f64,
    pub average_latency_ms: f64,
    pub consecutive_failures: u32,
}