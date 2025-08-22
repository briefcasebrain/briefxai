use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{debug, error, info, instrument, warn};

/// Comprehensive monitoring and observability system for BriefXAI
#[derive(Debug, Clone)]
pub struct MonitoringSystem {
    metrics: Arc<Mutex<SystemMetrics>>,
    start_time: Instant,
    performance_tracker: Arc<Mutex<PerformanceTracker>>,
    #[allow(dead_code)]
    resource_monitor: Arc<Mutex<ResourceMonitor>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub average_response_time_ms: f64,
    pub active_connections: u32,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub api_calls: HashMap<String, ApiMetrics>,
    pub component_metrics: HashMap<String, ComponentMetrics>,
    pub uptime_seconds: u64,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiMetrics {
    pub total_calls: u64,
    pub successful_calls: u64,
    pub failed_calls: u64,
    pub average_latency_ms: f64,
    pub last_call: Option<DateTime<Utc>>,
    pub rate_limited_calls: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentMetrics {
    pub executions: u64,
    pub total_time_ms: u64,
    pub average_time_ms: f64,
    pub errors: u64,
    pub last_execution: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone)]
pub struct PerformanceTracker {
    #[allow(dead_code)]
    operation_times: HashMap<String, Vec<Duration>>,
    memory_snapshots: Vec<MemorySnapshot>,
    throughput_measurements: Vec<ThroughputMeasurement>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    pub timestamp: DateTime<Utc>,
    pub heap_used_mb: f64,
    pub heap_total_mb: f64,
    pub rss_mb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputMeasurement {
    pub timestamp: DateTime<Utc>,
    pub component: String,
    pub items_processed: u64,
    pub duration_ms: u64,
    pub items_per_second: f64,
}

#[derive(Debug, Clone)]
pub struct ResourceMonitor {
    #[allow(dead_code)]
    cpu_usage_history: Vec<f64>,
    #[allow(dead_code)]
    memory_usage_history: Vec<f64>,
    #[allow(dead_code)]
    disk_io_history: Vec<DiskIOStats>,
    #[allow(dead_code)]
    network_io_history: Vec<NetworkIOStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskIOStats {
    pub timestamp: DateTime<Utc>,
    pub read_bytes: u64,
    pub write_bytes: u64,
    pub read_ops: u64,
    pub write_ops: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkIOStats {
    pub timestamp: DateTime<Utc>,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub packets_sent: u64,
    pub packets_received: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    pub status: HealthStatus,
    pub timestamp: DateTime<Utc>,
    pub checks: HashMap<String, ComponentHealth>,
    pub overall_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub status: HealthStatus,
    pub response_time_ms: Option<f64>,
    pub error_rate: f64,
    pub last_error: Option<String>,
    pub uptime_percent: f64,
}

impl MonitoringSystem {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(Mutex::new(SystemMetrics::default())),
            start_time: Instant::now(),
            performance_tracker: Arc::new(Mutex::new(PerformanceTracker::default())),
            resource_monitor: Arc::new(Mutex::new(ResourceMonitor::default())),
        }
    }

    #[instrument(skip(self))]
    pub fn record_request(&self, success: bool, duration: Duration) {
        let mut metrics = self.metrics.lock().unwrap();
        metrics.total_requests += 1;

        if success {
            metrics.successful_requests += 1;
        } else {
            metrics.failed_requests += 1;
        }

        // Update average response time
        let total_time = metrics.average_response_time_ms * (metrics.total_requests - 1) as f64
            + duration.as_millis() as f64;
        metrics.average_response_time_ms = total_time / metrics.total_requests as f64;

        metrics.last_updated = Utc::now();

        debug!(
            "Request recorded: success={}, duration={:?}, total={}",
            success, duration, metrics.total_requests
        );
    }

    #[instrument(skip(self))]
    pub fn record_api_call(&self, provider: &str, success: bool, duration: Duration) {
        let mut metrics = self.metrics.lock().unwrap();
        let api_metrics = metrics
            .api_calls
            .entry(provider.to_string())
            .or_insert_with(ApiMetrics::default);

        api_metrics.total_calls += 1;
        api_metrics.last_call = Some(Utc::now());

        if success {
            api_metrics.successful_calls += 1;
        } else {
            api_metrics.failed_calls += 1;
        }

        // Update average latency
        let total_time = api_metrics.average_latency_ms * (api_metrics.total_calls - 1) as f64
            + duration.as_millis() as f64;
        api_metrics.average_latency_ms = total_time / api_metrics.total_calls as f64;

        info!(
            "API call recorded: provider={}, success={}, duration={:?}",
            provider, success, duration
        );
    }

    #[instrument(skip(self))]
    pub fn record_component_execution(&self, component: &str, duration: Duration, success: bool) {
        let mut metrics = self.metrics.lock().unwrap();
        let comp_metrics = metrics
            .component_metrics
            .entry(component.to_string())
            .or_insert_with(ComponentMetrics::default);

        comp_metrics.executions += 1;
        comp_metrics.total_time_ms += duration.as_millis() as u64;
        comp_metrics.average_time_ms =
            comp_metrics.total_time_ms as f64 / comp_metrics.executions as f64;
        comp_metrics.last_execution = Some(Utc::now());

        if !success {
            comp_metrics.errors += 1;
        }

        debug!(
            "Component execution recorded: component={}, duration={:?}, success={}",
            component, duration, success
        );
    }

    #[instrument(skip(self))]
    pub fn record_cache_hit(&self, hit: bool) {
        let mut metrics = self.metrics.lock().unwrap();
        if hit {
            metrics.cache_hits += 1;
        } else {
            metrics.cache_misses += 1;
        }
    }

    #[instrument(skip(self))]
    pub fn record_throughput(&self, component: &str, items_processed: u64, duration: Duration) {
        let mut tracker = self.performance_tracker.lock().unwrap();
        let items_per_second = items_processed as f64 / duration.as_secs_f64();

        tracker.throughput_measurements.push(ThroughputMeasurement {
            timestamp: Utc::now(),
            component: component.to_string(),
            items_processed,
            duration_ms: duration.as_millis() as u64,
            items_per_second,
        });

        // Keep only last 1000 measurements
        if tracker.throughput_measurements.len() > 1000 {
            tracker.throughput_measurements.remove(0);
        }

        info!(
            "Throughput recorded: component={}, items={}, rate={:.2}/sec",
            component, items_processed, items_per_second
        );
    }

    #[instrument(skip(self))]
    pub fn record_memory_usage(&self) -> Result<()> {
        // In a real implementation, you'd use system APIs to get actual memory usage
        // For now, we'll simulate this
        let mut tracker = self.performance_tracker.lock().unwrap();

        let snapshot = MemorySnapshot {
            timestamp: Utc::now(),
            heap_used_mb: 0.0,  // Would get from system
            heap_total_mb: 0.0, // Would get from system
            rss_mb: 0.0,        // Would get from system
        };

        tracker.memory_snapshots.push(snapshot);

        // Keep only last 1000 snapshots
        if tracker.memory_snapshots.len() > 1000 {
            tracker.memory_snapshots.remove(0);
        }

        Ok(())
    }

    #[instrument(skip(self))]
    pub fn perform_health_check(&self) -> HealthCheckResult {
        let metrics = self.metrics.lock().unwrap();
        let mut checks = HashMap::new();
        let mut total_score = 0.0;
        let mut check_count = 0;

        // Check API health
        for (provider, api_metrics) in &metrics.api_calls {
            let error_rate = if api_metrics.total_calls > 0 {
                api_metrics.failed_calls as f64 / api_metrics.total_calls as f64
            } else {
                0.0
            };

            let status = match error_rate {
                rate if rate < 0.01 => HealthStatus::Healthy,
                rate if rate < 0.05 => HealthStatus::Degraded,
                rate if rate < 0.20 => HealthStatus::Unhealthy,
                _ => HealthStatus::Critical,
            };

            let score = match status {
                HealthStatus::Healthy => 100.0,
                HealthStatus::Degraded => 75.0,
                HealthStatus::Unhealthy => 50.0,
                HealthStatus::Critical => 0.0,
            };

            checks.insert(
                format!("api_{}", provider),
                ComponentHealth {
                    status,
                    response_time_ms: Some(api_metrics.average_latency_ms),
                    error_rate,
                    last_error: None,
                    uptime_percent: 100.0 - (error_rate * 100.0),
                },
            );

            total_score += score;
            check_count += 1;
        }

        // Check component health
        for (component, comp_metrics) in &metrics.component_metrics {
            let error_rate = if comp_metrics.executions > 0 {
                comp_metrics.errors as f64 / comp_metrics.executions as f64
            } else {
                0.0
            };

            let status = match error_rate {
                rate if rate < 0.01 => HealthStatus::Healthy,
                rate if rate < 0.05 => HealthStatus::Degraded,
                rate if rate < 0.20 => HealthStatus::Unhealthy,
                _ => HealthStatus::Critical,
            };

            let score = match status {
                HealthStatus::Healthy => 100.0,
                HealthStatus::Degraded => 75.0,
                HealthStatus::Unhealthy => 50.0,
                HealthStatus::Critical => 0.0,
            };

            checks.insert(
                component.clone(),
                ComponentHealth {
                    status,
                    response_time_ms: Some(comp_metrics.average_time_ms),
                    error_rate,
                    last_error: None,
                    uptime_percent: 100.0 - (error_rate * 100.0),
                },
            );

            total_score += score;
            check_count += 1;
        }

        let overall_score = if check_count > 0 {
            total_score / check_count as f64
        } else {
            100.0
        };

        let overall_status = match overall_score {
            score if score >= 90.0 => HealthStatus::Healthy,
            score if score >= 70.0 => HealthStatus::Degraded,
            score if score >= 50.0 => HealthStatus::Unhealthy,
            _ => HealthStatus::Critical,
        };

        HealthCheckResult {
            status: overall_status,
            timestamp: Utc::now(),
            checks,
            overall_score,
        }
    }

    pub fn get_metrics(&self) -> SystemMetrics {
        let mut metrics = self.metrics.lock().unwrap();
        metrics.uptime_seconds = self.start_time.elapsed().as_secs();
        metrics.clone()
    }

    pub fn get_performance_report(&self) -> PerformanceReport {
        let tracker = self.performance_tracker.lock().unwrap();
        let metrics = self.metrics.lock().unwrap();

        PerformanceReport {
            uptime_seconds: self.start_time.elapsed().as_secs(),
            total_requests: metrics.total_requests,
            average_response_time_ms: metrics.average_response_time_ms,
            throughput_summary: self.calculate_throughput_summary(&tracker),
            memory_summary: self.calculate_memory_summary(&tracker),
            top_slow_components: self.get_slowest_components(&metrics),
            cache_efficiency: self.calculate_cache_efficiency(&metrics),
        }
    }

    fn calculate_throughput_summary(&self, tracker: &PerformanceTracker) -> HashMap<String, f64> {
        let mut summary = HashMap::new();

        for measurement in &tracker.throughput_measurements {
            let avg = summary.entry(measurement.component.clone()).or_insert(0.0);
            *avg = (*avg + measurement.items_per_second) / 2.0;
        }

        summary
    }

    fn calculate_memory_summary(&self, tracker: &PerformanceTracker) -> MemorySummary {
        if tracker.memory_snapshots.is_empty() {
            return MemorySummary::default();
        }

        let total_snapshots = tracker.memory_snapshots.len() as f64;
        let avg_heap = tracker
            .memory_snapshots
            .iter()
            .map(|s| s.heap_used_mb)
            .sum::<f64>()
            / total_snapshots;
        let avg_rss = tracker
            .memory_snapshots
            .iter()
            .map(|s| s.rss_mb)
            .sum::<f64>()
            / total_snapshots;

        let max_heap = tracker
            .memory_snapshots
            .iter()
            .map(|s| s.heap_used_mb)
            .fold(0.0f64, f64::max);
        let max_rss = tracker
            .memory_snapshots
            .iter()
            .map(|s| s.rss_mb)
            .fold(0.0f64, f64::max);

        MemorySummary {
            average_heap_mb: avg_heap,
            average_rss_mb: avg_rss,
            peak_heap_mb: max_heap,
            peak_rss_mb: max_rss,
        }
    }

    fn get_slowest_components(&self, metrics: &SystemMetrics) -> Vec<(String, f64)> {
        let mut components: Vec<_> = metrics
            .component_metrics
            .iter()
            .map(|(name, metrics)| (name.clone(), metrics.average_time_ms))
            .collect();
        components.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        components.into_iter().take(10).collect()
    }

    fn calculate_cache_efficiency(&self, metrics: &SystemMetrics) -> f64 {
        let total_cache_ops = metrics.cache_hits + metrics.cache_misses;
        if total_cache_ops > 0 {
            metrics.cache_hits as f64 / total_cache_ops as f64 * 100.0
        } else {
            0.0
        }
    }

    pub fn export_metrics(&self, format: MetricsFormat) -> Result<String> {
        let metrics = self.get_metrics();
        let health = self.perform_health_check();
        let performance = self.get_performance_report();

        match format {
            MetricsFormat::Json => {
                let export = MetricsExport {
                    metrics,
                    health,
                    performance,
                    timestamp: Utc::now(),
                };
                Ok(serde_json::to_string_pretty(&export)?)
            }
            MetricsFormat::Prometheus => Ok(self.format_prometheus_metrics(&metrics)),
        }
    }

    fn format_prometheus_metrics(&self, metrics: &SystemMetrics) -> String {
        let mut output = String::new();

        output.push_str("# HELP briefxai_requests_total Total number of requests\n");
        output.push_str("# TYPE briefxai_requests_total counter\n");
        output.push_str(&format!(
            "briefxai_requests_total {}\n",
            metrics.total_requests
        ));

        output.push_str(
            "# HELP briefxai_requests_successful_total Total number of successful requests\n",
        );
        output.push_str("# TYPE briefxai_requests_successful_total counter\n");
        output.push_str(&format!(
            "briefxai_requests_successful_total {}\n",
            metrics.successful_requests
        ));

        output.push_str("# HELP briefxai_response_time_ms Average response time in milliseconds\n");
        output.push_str("# TYPE briefxai_response_time_ms gauge\n");
        output.push_str(&format!(
            "briefxai_response_time_ms {}\n",
            metrics.average_response_time_ms
        ));

        output.push_str("# HELP briefxai_cache_hits_total Total number of cache hits\n");
        output.push_str("# TYPE briefxai_cache_hits_total counter\n");
        output.push_str(&format!(
            "briefxai_cache_hits_total {}\n",
            metrics.cache_hits
        ));

        for (provider, api_metrics) in &metrics.api_calls {
            output.push_str(&format!(
                "# HELP briefxai_api_calls_total Total API calls for {}\n",
                provider
            ));
            output.push_str(&format!("# TYPE briefxai_api_calls_total counter\n"));
            output.push_str(&format!(
                "briefxai_api_calls_total{{provider=\"{}\"}} {}\n",
                provider, api_metrics.total_calls
            ));
        }

        output
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub uptime_seconds: u64,
    pub total_requests: u64,
    pub average_response_time_ms: f64,
    pub throughput_summary: HashMap<String, f64>,
    pub memory_summary: MemorySummary,
    pub top_slow_components: Vec<(String, f64)>,
    pub cache_efficiency: f64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct MemorySummary {
    pub average_heap_mb: f64,
    pub average_rss_mb: f64,
    pub peak_heap_mb: f64,
    pub peak_rss_mb: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MetricsExport {
    pub metrics: SystemMetrics,
    pub health: HealthCheckResult,
    pub performance: PerformanceReport,
    pub timestamp: DateTime<Utc>,
}

pub enum MetricsFormat {
    Json,
    Prometheus,
}

// Default implementations
impl Default for SystemMetrics {
    fn default() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            average_response_time_ms: 0.0,
            active_connections: 0,
            cache_hits: 0,
            cache_misses: 0,
            api_calls: HashMap::new(),
            component_metrics: HashMap::new(),
            uptime_seconds: 0,
            last_updated: Utc::now(),
        }
    }
}

impl Default for ApiMetrics {
    fn default() -> Self {
        Self {
            total_calls: 0,
            successful_calls: 0,
            failed_calls: 0,
            average_latency_ms: 0.0,
            last_call: None,
            rate_limited_calls: 0,
        }
    }
}

impl Default for ComponentMetrics {
    fn default() -> Self {
        Self {
            executions: 0,
            total_time_ms: 0,
            average_time_ms: 0.0,
            errors: 0,
            last_execution: None,
        }
    }
}

impl Default for PerformanceTracker {
    fn default() -> Self {
        Self {
            operation_times: HashMap::new(),
            memory_snapshots: Vec::new(),
            throughput_measurements: Vec::new(),
        }
    }
}

impl Default for ResourceMonitor {
    fn default() -> Self {
        Self {
            cpu_usage_history: Vec::new(),
            memory_usage_history: Vec::new(),
            disk_io_history: Vec::new(),
            network_io_history: Vec::new(),
        }
    }
}

/// Monitoring middleware for tracking operation performance
pub struct MonitoringWrapper<T> {
    inner: T,
    monitoring: Arc<MonitoringSystem>,
    component_name: String,
}

impl<T> MonitoringWrapper<T> {
    pub fn new(inner: T, monitoring: Arc<MonitoringSystem>, component_name: String) -> Self {
        Self {
            inner,
            monitoring,
            component_name,
        }
    }

    pub async fn execute<F, R>(&self, operation: F) -> Result<R>
    where
        F: FnOnce(&T) -> Result<R>,
    {
        let start = Instant::now();
        let result = operation(&self.inner);
        let duration = start.elapsed();

        let success = result.is_ok();
        self.monitoring
            .record_component_execution(&self.component_name, duration, success);

        if let Err(ref e) = result {
            error!("Component {} failed: {}", self.component_name, e);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitoring_system_creation() {
        let monitoring = MonitoringSystem::new();
        let metrics = monitoring.get_metrics();

        assert_eq!(metrics.total_requests, 0);
        assert_eq!(metrics.successful_requests, 0);
        assert_eq!(metrics.failed_requests, 0);
    }

    #[test]
    fn test_request_recording() {
        let monitoring = MonitoringSystem::new();

        monitoring.record_request(true, Duration::from_millis(100));
        monitoring.record_request(false, Duration::from_millis(200));

        let metrics = monitoring.get_metrics();
        assert_eq!(metrics.total_requests, 2);
        assert_eq!(metrics.successful_requests, 1);
        assert_eq!(metrics.failed_requests, 1);
        assert_eq!(metrics.average_response_time_ms, 150.0);
    }

    #[test]
    fn test_api_call_recording() {
        let monitoring = MonitoringSystem::new();

        monitoring.record_api_call("openai", true, Duration::from_millis(500));
        monitoring.record_api_call("openai", false, Duration::from_millis(1000));

        let metrics = monitoring.get_metrics();
        let openai_metrics = &metrics.api_calls["openai"];

        assert_eq!(openai_metrics.total_calls, 2);
        assert_eq!(openai_metrics.successful_calls, 1);
        assert_eq!(openai_metrics.failed_calls, 1);
        assert_eq!(openai_metrics.average_latency_ms, 750.0);
    }

    #[test]
    fn test_component_execution_recording() {
        let monitoring = MonitoringSystem::new();

        monitoring.record_component_execution("clustering", Duration::from_millis(300), true);
        monitoring.record_component_execution("clustering", Duration::from_millis(700), false);

        let metrics = monitoring.get_metrics();
        let clustering_metrics = &metrics.component_metrics["clustering"];

        assert_eq!(clustering_metrics.executions, 2);
        assert_eq!(clustering_metrics.errors, 1);
        assert_eq!(clustering_metrics.average_time_ms, 500.0);
    }

    #[test]
    fn test_cache_recording() {
        let monitoring = MonitoringSystem::new();

        monitoring.record_cache_hit(true);
        monitoring.record_cache_hit(true);
        monitoring.record_cache_hit(false);

        let metrics = monitoring.get_metrics();
        assert_eq!(metrics.cache_hits, 2);
        assert_eq!(metrics.cache_misses, 1);
    }

    #[test]
    fn test_health_check() {
        let monitoring = MonitoringSystem::new();

        // Record some API calls with different success rates
        for _ in 0..10 {
            monitoring.record_api_call("good_api", true, Duration::from_millis(100));
        }

        for _ in 0..5 {
            monitoring.record_api_call("bad_api", false, Duration::from_millis(200));
        }
        monitoring.record_api_call("bad_api", true, Duration::from_millis(150));

        let health = monitoring.perform_health_check();

        assert!(matches!(
            health.status,
            HealthStatus::Degraded | HealthStatus::Unhealthy
        ));
        assert!(health.checks.contains_key("api_good_api"));
        assert!(health.checks.contains_key("api_bad_api"));
    }

    #[test]
    fn test_prometheus_export() {
        let monitoring = MonitoringSystem::new();

        monitoring.record_request(true, Duration::from_millis(100));
        monitoring.record_api_call("test", true, Duration::from_millis(50));

        let prometheus_output = monitoring
            .export_metrics(MetricsFormat::Prometheus)
            .unwrap();

        assert!(prometheus_output.contains("briefxai_requests_total 1"));
        assert!(prometheus_output.contains("briefxai_requests_successful_total 1"));
        assert!(prometheus_output.contains("briefxai_api_calls_total{provider=\"test\"} 1"));
    }

    #[test]
    fn test_json_export() {
        let monitoring = MonitoringSystem::new();

        monitoring.record_request(true, Duration::from_millis(100));

        let json_output = monitoring.export_metrics(MetricsFormat::Json).unwrap();
        let parsed: MetricsExport = serde_json::from_str(&json_output).unwrap();

        assert_eq!(parsed.metrics.total_requests, 1);
        assert_eq!(parsed.metrics.successful_requests, 1);
    }
}
