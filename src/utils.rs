use anyhow::{Result, Context, bail};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{info, warn, debug, error};

use crate::types::ConversationData;
use crate::prompts::conversation_to_string;

pub fn dedup_data(data: Vec<ConversationData>) -> Result<Vec<ConversationData>> {
    info!("Deduplicating {} conversations", data.len());
    
    let mut seen = HashSet::new();
    let mut deduped = Vec::new();
    
    for conversation in data {
        let key = conversation_to_string(&conversation);
        if seen.insert(key) {
            deduped.push(conversation);
        }
    }
    
    info!("Deduplicated to {} unique conversations", deduped.len());
    Ok(deduped)
}

pub fn dedup_by_key<T, K, F>(items: Vec<T>, key_fn: F) -> Vec<T>
where
    K: Eq + Hash,
    F: Fn(&T) -> K,
{
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    
    for item in items {
        let key = key_fn(&item);
        if seen.insert(key) {
            result.push(item);
        }
    }
    
    result
}

pub fn flatten<T>(nested: Vec<Vec<T>>) -> Vec<T> {
    nested.into_iter().flatten().collect()
}

pub fn unflatten<T: Clone>(flat: Vec<T>, sizes: Vec<usize>) -> Vec<Vec<T>> {
    let mut result = Vec::new();
    let mut offset = 0;
    
    for size in sizes {
        let chunk = flat[offset..offset + size].to_vec();
        result.push(chunk);
        offset += size;
    }
    
    result
}


pub fn sample_items<T: Clone>(items: &[T], n: usize, seed: Option<u64>) -> Vec<T> {
    use rand::{seq::SliceRandom, SeedableRng};
    
    if items.len() <= n {
        return items.to_vec();
    }
    
    let mut rng = if let Some(seed) = seed {
        rand::rngs::StdRng::seed_from_u64(seed)
    } else {
        rand::rngs::StdRng::from_entropy()
    };
    
    let mut indices: Vec<usize> = (0..items.len()).collect();
    indices.shuffle(&mut rng);
    
    indices.into_iter()
        .take(n)
        .map(|i| items[i].clone())
        .collect()
}

pub fn most_common<T: Eq + Hash + Clone>(items: Vec<T>) -> Option<T> {
    if items.is_empty() {
        return None;
    }
    
    let mut counts = HashMap::new();
    for item in items {
        *counts.entry(item).or_insert(0) += 1;
    }
    
    counts.into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(item, _)| item)
}

pub fn chunk_data<T: Clone>(data: Vec<T>, max_size: usize) -> Vec<Vec<T>> {
    let mut chunks = Vec::new();
    let mut current_chunk = Vec::new();
    let mut current_size = 0;
    
    for item in data {
        let item_size = std::mem::size_of_val(&item);
        
        if current_size + item_size > max_size && !current_chunk.is_empty() {
            chunks.push(current_chunk);
            current_chunk = Vec::new();
            current_size = 0;
        }
        
        current_chunk.push(item);
        current_size += item_size;
    }
    
    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }
    
    chunks
}

pub fn truncate_conversation(
    conversation: &ConversationData,
    max_tokens: usize,
) -> ConversationData {
    // Simple token estimation: ~4 characters per token
    let estimate_tokens = |s: &str| s.len() / 4;
    
    let mut truncated = Vec::new();
    let mut total_tokens = 0;
    
    for message in conversation.messages.iter().rev() {
        let message_tokens = estimate_tokens(&message.content);
        
        if total_tokens + message_tokens > max_tokens {
            break;
        }
        
        truncated.push(message.clone());
        total_tokens += message_tokens;
    }
    
    truncated.reverse();
    
    // Ensure we end with an assistant message if possible
    if !truncated.is_empty() && truncated.last().unwrap().role == "user" {
        truncated.pop();
    }
    
    ConversationData {
        messages: truncated,
        metadata: conversation.metadata.clone(),
    }
}

// Retry configuration
#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_attempts: u32,
    pub initial_delay: Duration,
    pub max_delay: Duration,
    pub exponential_base: f64,
    pub jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(1000),
            max_delay: Duration::from_secs(30),
            exponential_base: 2.0,
            jitter: true,
        }
    }
}

// Retry mechanism with exponential backoff
pub async fn retry_with_backoff<F, Fut, T>(
    mut operation: F,
    config: RetryConfig,
    operation_name: &str,
) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    let mut attempt = 0;
    let mut delay = config.initial_delay;
    
    loop {
        attempt += 1;
        debug!("Attempting {} (attempt {}/{})", operation_name, attempt, config.max_attempts);
        
        match operation().await {
            Ok(result) => {
                if attempt > 1 {
                    info!("{} succeeded after {} attempts", operation_name, attempt);
                }
                return Ok(result);
            }
            Err(e) if attempt >= config.max_attempts => {
                error!("{} failed after {} attempts: {}", operation_name, attempt, e);
                return Err(e).context(format!("{} failed after {} attempts", operation_name, config.max_attempts));
            }
            Err(e) => {
                warn!("{} attempt {} failed: {}, retrying in {:?}", 
                      operation_name, attempt, e, delay);
                
                sleep(delay).await;
                
                // Calculate next delay with exponential backoff
                delay = Duration::from_secs_f64(
                    (delay.as_secs_f64() * config.exponential_base).min(config.max_delay.as_secs_f64())
                );
                
                // Add jitter if configured
                if config.jitter {
                    use rand::Rng;
                    let jitter = rand::thread_rng().gen_range(0.8..1.2);
                    delay = Duration::from_secs_f64(delay.as_secs_f64() * jitter);
                }
            }
        }
    }
}

// Error recovery strategies
pub enum RecoveryStrategy {
    Retry(RetryConfig),
    Fallback(Box<dyn Fn() -> Result<()> + Send + Sync>),
    Skip,
    Fail,
}

// Error handler with recovery
pub async fn handle_error_with_recovery<T, F, Fut>(
    operation: F,
    strategy: RecoveryStrategy,
    context: &str,
) -> Result<Option<T>>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    match operation().await {
        Ok(result) => Ok(Some(result)),
        Err(e) => {
            error!("Error in {}: {}", context, e);
            
            match strategy {
                RecoveryStrategy::Retry(config) => {
                    match retry_with_backoff(operation, config, context).await {
                        Ok(result) => Ok(Some(result)),
                        Err(e) => {
                            error!("Retry failed for {}: {}", context, e);
                            Ok(None)
                        }
                    }
                }
                RecoveryStrategy::Fallback(fallback_fn) => {
                    warn!("Using fallback for {}", context);
                    fallback_fn()?;
                    Ok(None)
                }
                RecoveryStrategy::Skip => {
                    warn!("Skipping {} due to error", context);
                    Ok(None)
                }
                RecoveryStrategy::Fail => {
                    bail!("Critical error in {}: {}", context, e)
                }
            }
        }
    }
}

// Batch processing with error handling
pub async fn process_batch_with_errors<T, R, F, Fut>(
    items: Vec<T>,
    processor: F,
    batch_size: usize,
    continue_on_error: bool,
) -> Result<Vec<R>>
where
    T: Clone + Send + Sync,
    R: Send,
    F: Fn(T) -> Fut + Clone,
    Fut: std::future::Future<Output = Result<R>> + Send,
{
    use futures_util::future::join_all;
    
    let mut all_results = Vec::new();
    let mut errors = Vec::new();
    
    for chunk in items.chunks(batch_size) {
        let futures: Vec<_> = chunk
            .iter()
            .cloned()
            .map(|item| {
                let processor = processor.clone();
                async move { processor(item).await }
            })
            .collect();
        
        let results = join_all(futures).await;
        
        for (i, result) in results.into_iter().enumerate() {
            match result {
                Ok(r) => all_results.push(r),
                Err(e) => {
                    let error_msg = format!("Error processing item {}: {}", i, e);
                    error!("{}", error_msg);
                    errors.push(error_msg);
                    
                    if !continue_on_error {
                        bail!("Batch processing failed: {}", errors.join("; "));
                    }
                }
            }
        }
    }
    
    if !errors.is_empty() && !continue_on_error {
        bail!("Batch processing had {} errors: {}", errors.len(), errors.join("; "));
    }
    
    if !errors.is_empty() {
        warn!("Batch processing completed with {} errors", errors.len());
    }
    
    Ok(all_results)
}

// Circuit breaker for preventing cascading failures
pub struct CircuitBreaker {
    failure_count: std::sync::Arc<std::sync::atomic::AtomicU32>,
    threshold: u32,
    reset_timeout: Duration,
    last_failure_time: std::sync::Arc<tokio::sync::RwLock<Option<std::time::Instant>>>,
}

impl CircuitBreaker {
    pub fn new(threshold: u32, reset_timeout: Duration) -> Self {
        Self {
            failure_count: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
            threshold,
            reset_timeout,
            last_failure_time: std::sync::Arc::new(tokio::sync::RwLock::new(None)),
        }
    }
    
    pub async fn call<F, Fut, T>(&self, operation: F) -> Result<T>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        // Check if circuit is open
        if self.is_open().await {
            bail!("Circuit breaker is open - too many failures");
        }
        
        match operation().await {
            Ok(result) => {
                self.on_success().await;
                Ok(result)
            }
            Err(e) => {
                self.on_failure().await;
                Err(e)
            }
        }
    }
    
    async fn is_open(&self) -> bool {
        let count = self.failure_count.load(std::sync::atomic::Ordering::Relaxed);
        if count >= self.threshold {
            // Check if we should reset
            let last_failure = self.last_failure_time.read().await;
            if let Some(time) = *last_failure {
                if time.elapsed() > self.reset_timeout {
                    // Reset the circuit
                    self.failure_count.store(0, std::sync::atomic::Ordering::Relaxed);
                    return false;
                }
            }
            true
        } else {
            false
        }
    }
    
    async fn on_success(&self) {
        self.failure_count.store(0, std::sync::atomic::Ordering::Relaxed);
    }
    
    async fn on_failure(&self) {
        self.failure_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut last_failure = self.last_failure_time.write().await;
        *last_failure = Some(std::time::Instant::now());
    }
}

// Timeout wrapper
pub async fn with_timeout<F, Fut, T>(
    operation: F,
    timeout: Duration,
    operation_name: &str,
) -> Result<T>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    match tokio::time::timeout(timeout, operation()).await {
        Ok(Ok(result)) => Ok(result),
        Ok(Err(e)) => Err(e),
        Err(_) => bail!("{} timed out after {:?}", operation_name, timeout),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dedup_by_key() {
        let items = vec![1, 2, 2, 3, 3, 3, 4];
        let deduped = dedup_by_key(items, |x| *x);
        assert_eq!(deduped, vec![1, 2, 3, 4]);
    }
    
    #[test]
    fn test_flatten_unflatten() {
        let nested = vec![vec![1, 2], vec![3, 4, 5], vec![6]];
        let sizes: Vec<_> = nested.iter().map(|v| v.len()).collect();
        
        let flat = flatten(nested.clone());
        assert_eq!(flat, vec![1, 2, 3, 4, 5, 6]);
        
        let unflat = unflatten(flat, sizes);
        assert_eq!(unflat, nested);
    }
    
    #[test]
    fn test_most_common() {
        let items = vec!["a", "b", "b", "c", "b", "c"];
        assert_eq!(most_common(items), Some("b"));
    }
    
    #[tokio::test]
    async fn test_retry_with_backoff() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let attempt = AtomicUsize::new(0);
        let operation = || async {
            let current = attempt.fetch_add(1, Ordering::SeqCst);
            if current < 2 {
                Err(anyhow::anyhow!("Simulated failure"))
            } else {
                Ok("Success")
            }
        };
        
        let config = RetryConfig {
            max_attempts: 3,
            initial_delay: Duration::from_millis(10),
            max_delay: Duration::from_millis(100),
            exponential_base: 2.0,
            jitter: false,
        };
        
        let result = retry_with_backoff(operation, config, "test_operation").await;
        assert!(result.is_ok());
    }
}