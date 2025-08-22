use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tracing::{
    field::{Field, Visit},
    Event, Subscriber,
};
use tracing_subscriber::{
    filter::{EnvFilter, LevelFilter},
    fmt::{self, format::FmtSpan},
    layer::{Layer, SubscriberExt},
    util::SubscriberInitExt,
    Registry,
};

/// Comprehensive logging system for BriefXAI with structured logging,
/// audit trails, security logging, and performance metrics
#[derive(Debug, Clone)]
pub struct LoggingSystem {
    config: LoggingConfig,
    audit_logger: Arc<Mutex<AuditLogger>>,
    security_logger: Arc<Mutex<SecurityLogger>>,
    performance_logger: Arc<Mutex<PerformanceLogger>>,
    metrics_collector: Arc<Mutex<MetricsCollector>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: LogLevel,
    pub format: LogFormat,
    pub output: LogOutput,
    pub enable_audit: bool,
    pub enable_security: bool,
    pub enable_performance: bool,
    pub enable_metrics: bool,
    pub structured_logging: bool,
    pub correlation_id_header: String,
    pub sensitive_fields: Vec<String>,
    pub log_retention_days: u32,
    pub max_log_file_size_mb: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogFormat {
    Json,
    Pretty,
    Compact,
    Full,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogOutput {
    Console,
    File(String),
    Both(String),
    Syslog,
    Remote(String),
}

/// Structured log entry for consistent logging across the application
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub timestamp: DateTime<Utc>,
    pub level: LogLevel,
    pub module: String,
    pub message: String,
    pub correlation_id: Option<String>,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub request_id: Option<String>,
    pub fields: HashMap<String, serde_json::Value>,
    pub error: Option<ErrorDetails>,
    pub performance: Option<PerformanceMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDetails {
    pub error_type: String,
    pub error_message: String,
    pub stack_trace: Option<String>,
    pub error_code: Option<String>,
    pub context: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub operation: String,
    pub duration_ms: u64,
    pub memory_used_mb: Option<f64>,
    pub cpu_usage_percent: Option<f64>,
    pub throughput: Option<f64>,
    pub custom_metrics: HashMap<String, f64>,
}

/// Audit logging for tracking sensitive operations and data access
#[derive(Debug)]
pub struct AuditLogger {
    entries: Vec<AuditEntry>,
    #[allow(dead_code)]
    config: AuditConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub timestamp: DateTime<Utc>,
    pub event_type: AuditEventType,
    pub user_id: Option<String>,
    pub resource: String,
    pub action: String,
    pub result: AuditResult,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
    pub correlation_id: Option<String>,
    pub details: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditEventType {
    DataAccess,
    UserAuthentication,
    UserAuthorization,
    ConfigurationChange,
    DataModification,
    ApiKeyUsage,
    PrivacyOperation,
    ModelExecution,
    FileAccess,
    SystemEvent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditResult {
    Success,
    Failure,
    Partial,
    Denied,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    pub enable_data_access_logging: bool,
    pub enable_auth_logging: bool,
    pub enable_config_change_logging: bool,
    pub log_failed_attempts: bool,
    pub include_sensitive_data: bool,
    pub retention_days: u32,
}

/// Security logging for tracking security-related events and threats
#[derive(Debug)]
pub struct SecurityLogger {
    events: Vec<SecurityEvent>,
    #[allow(dead_code)]
    threat_detector: ThreatDetector,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityEvent {
    pub timestamp: DateTime<Utc>,
    pub event_type: SecurityEventType,
    pub severity: SecuritySeverity,
    pub source_ip: Option<String>,
    pub user_id: Option<String>,
    pub description: String,
    pub indicators: Vec<ThreatIndicator>,
    pub mitigations: Vec<String>,
    pub correlation_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityEventType {
    SuspiciousActivity,
    AuthenticationFailure,
    AuthorizationFailure,
    RateLimitExceeded,
    MaliciousRequest,
    DataExfiltrationAttempt,
    InjectionAttempt,
    AnomalousPattern,
    PrivilegeEscalation,
    UnauthorizedAccess,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SecuritySeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatIndicator {
    pub indicator_type: String,
    pub value: String,
    pub confidence: f64,
    pub description: String,
}

#[derive(Debug)]
pub struct ThreatDetector {
    #[allow(dead_code)]
    patterns: Vec<ThreatPattern>,
    #[allow(dead_code)]
    rate_limits: HashMap<String, RateLimit>,
}

#[derive(Debug, Clone)]
pub struct ThreatPattern {
    pub name: String,
    pub pattern: String,
    pub severity: SecuritySeverity,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct RateLimit {
    pub requests: u32,
    pub window_minutes: u32,
    pub current_count: u32,
    pub window_start: DateTime<Utc>,
}

/// Performance logging for tracking operation performance and optimization
#[derive(Debug)]
pub struct PerformanceLogger {
    entries: Vec<PerformanceEntry>,
    #[allow(dead_code)]
    thresholds: PerformanceThresholds,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceEntry {
    pub timestamp: DateTime<Utc>,
    pub operation: String,
    pub duration_ms: u64,
    pub memory_delta_mb: Option<f64>,
    pub cpu_usage: Option<f64>,
    pub input_size: Option<u64>,
    pub output_size: Option<u64>,
    pub cache_hit: Option<bool>,
    pub error_occurred: bool,
    pub custom_metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    pub slow_operation_ms: u64,
    pub high_memory_usage_mb: f64,
    pub high_cpu_usage_percent: f64,
    pub alert_on_threshold_breach: bool,
}

/// Metrics collection for aggregating performance and usage statistics
#[derive(Debug)]
pub struct MetricsCollector {
    counters: HashMap<String, u64>,
    gauges: HashMap<String, f64>,
    histograms: HashMap<String, Vec<f64>>,
    timers: HashMap<String, Vec<u64>>,
}

impl LoggingSystem {
    pub fn new(config: LoggingConfig) -> Result<Self> {
        let system = Self {
            config: config.clone(),
            audit_logger: Arc::new(Mutex::new(AuditLogger::new(AuditConfig::default()))),
            security_logger: Arc::new(Mutex::new(SecurityLogger::new())),
            performance_logger: Arc::new(Mutex::new(PerformanceLogger::new())),
            metrics_collector: Arc::new(Mutex::new(MetricsCollector::new())),
        };

        system.initialize_tracing()?;
        Ok(system)
    }

    fn initialize_tracing(&self) -> Result<()> {
        let filter = match self.config.level {
            LogLevel::Trace => LevelFilter::TRACE,
            LogLevel::Debug => LevelFilter::DEBUG,
            LogLevel::Info => LevelFilter::INFO,
            LogLevel::Warn => LevelFilter::WARN,
            LogLevel::Error => LevelFilter::ERROR,
        };

        let env_filter = EnvFilter::from_default_env()
            .add_directive(filter.into())
            .add_directive("briefxai=debug".parse()?)
            .add_directive("hyper=info".parse()?)
            .add_directive("reqwest=info".parse()?);

        let fmt_layer = match self.config.format {
            LogFormat::Json => fmt::layer().with_target(true).boxed(),
            LogFormat::Pretty => fmt::layer()
                .pretty()
                .with_target(true)
                .with_thread_ids(true)
                .with_file(true)
                .with_line_number(true)
                .boxed(),
            LogFormat::Compact => fmt::layer().compact().with_target(false).boxed(),
            LogFormat::Full => fmt::layer()
                .with_target(true)
                .with_thread_ids(true)
                .with_file(true)
                .with_line_number(true)
                .with_span_events(FmtSpan::CLOSE)
                .boxed(),
        };

        let _ = Registry::default()
            .with(env_filter)
            .with(fmt_layer)
            // Removed CustomLayer to prevent recursion
            .try_init();

        Ok(())
    }

    pub fn log_structured(&self, entry: LogEntry) {
        // Filter sensitive fields
        let filtered_entry = self.filter_sensitive_fields(entry);

        match filtered_entry.level {
            LogLevel::Error => tracing::error!(
                message = %filtered_entry.message,
                module = %filtered_entry.module,
                correlation_id = ?filtered_entry.correlation_id,
                "Structured log entry"
            ),
            LogLevel::Warn => tracing::warn!(
                message = %filtered_entry.message,
                module = %filtered_entry.module,
                correlation_id = ?filtered_entry.correlation_id,
                "Structured log entry"
            ),
            LogLevel::Info => tracing::info!(
                message = %filtered_entry.message,
                module = %filtered_entry.module,
                correlation_id = ?filtered_entry.correlation_id,
                "Structured log entry"
            ),
            LogLevel::Debug => tracing::debug!(
                message = %filtered_entry.message,
                module = %filtered_entry.module,
                correlation_id = ?filtered_entry.correlation_id,
                "Structured log entry"
            ),
            LogLevel::Trace => tracing::trace!(
                message = %filtered_entry.message,
                module = %filtered_entry.module,
                correlation_id = ?filtered_entry.correlation_id,
                "Structured log entry"
            ),
        }
    }

    pub fn log_audit(&self, entry: AuditEntry) {
        if !self.config.enable_audit {
            return;
        }

        tracing::info!(
            event_type = ?entry.event_type,
            user_id = ?entry.user_id,
            resource = %entry.resource,
            action = %entry.action,
            result = ?entry.result,
            correlation_id = ?entry.correlation_id,
            "Audit event"
        );

        if let Ok(mut logger) = self.audit_logger.lock() {
            logger.log_entry(entry);
        }
    }

    pub fn log_security(&self, event: SecurityEvent) {
        if !self.config.enable_security {
            return;
        }

        match event.severity {
            SecuritySeverity::Critical => tracing::error!(
                event_type = ?event.event_type,
                severity = ?event.severity,
                source_ip = ?event.source_ip,
                description = %event.description,
                "Critical security event"
            ),
            SecuritySeverity::High => tracing::warn!(
                event_type = ?event.event_type,
                severity = ?event.severity,
                source_ip = ?event.source_ip,
                description = %event.description,
                "High severity security event"
            ),
            _ => tracing::info!(
                event_type = ?event.event_type,
                severity = ?event.severity,
                source_ip = ?event.source_ip,
                description = %event.description,
                "Security event"
            ),
        }

        if let Ok(mut logger) = self.security_logger.lock() {
            logger.log_event(event);
        }
    }

    pub fn log_performance(&self, entry: PerformanceEntry) {
        if !self.config.enable_performance {
            return;
        }

        let is_slow = entry.duration_ms > 1000; // 1 second threshold

        if is_slow {
            tracing::warn!(
                operation = %entry.operation,
                duration_ms = entry.duration_ms,
                memory_delta_mb = ?entry.memory_delta_mb,
                "Slow operation detected"
            );
        } else {
            tracing::debug!(
                operation = %entry.operation,
                duration_ms = entry.duration_ms,
                memory_delta_mb = ?entry.memory_delta_mb,
                "Performance metrics"
            );
        }

        if let Ok(mut logger) = self.performance_logger.lock() {
            logger.log_entry(entry);
        }
    }

    pub fn increment_counter(&self, name: &str, value: u64) {
        if let Ok(mut collector) = self.metrics_collector.lock() {
            *collector.counters.entry(name.to_string()).or_insert(0) += value;
        }
    }

    pub fn set_gauge(&self, name: &str, value: f64) {
        if let Ok(mut collector) = self.metrics_collector.lock() {
            collector.gauges.insert(name.to_string(), value);
        }
    }

    pub fn record_histogram(&self, name: &str, value: f64) {
        if let Ok(mut collector) = self.metrics_collector.lock() {
            collector
                .histograms
                .entry(name.to_string())
                .or_insert_with(Vec::new)
                .push(value);
        }
    }

    pub fn record_timer(&self, name: &str, duration_ms: u64) {
        if let Ok(mut collector) = self.metrics_collector.lock() {
            collector
                .timers
                .entry(name.to_string())
                .or_insert_with(Vec::new)
                .push(duration_ms);
        }
    }

    fn filter_sensitive_fields(&self, mut entry: LogEntry) -> LogEntry {
        for sensitive_field in &self.config.sensitive_fields {
            if entry.fields.contains_key(sensitive_field) {
                entry.fields.insert(
                    sensitive_field.clone(),
                    serde_json::Value::String("[REDACTED]".to_string()),
                );
            }
        }
        entry
    }

    pub fn get_audit_logs(&self, limit: Option<usize>) -> Vec<AuditEntry> {
        if let Ok(logger) = self.audit_logger.lock() {
            logger.get_entries(limit)
        } else {
            Vec::new()
        }
    }

    pub fn get_security_events(
        &self,
        severity_filter: Option<SecuritySeverity>,
    ) -> Vec<SecurityEvent> {
        if let Ok(logger) = self.security_logger.lock() {
            logger.get_events(severity_filter)
        } else {
            Vec::new()
        }
    }

    pub fn get_performance_stats(&self) -> PerformanceStatistics {
        if let Ok(logger) = self.performance_logger.lock() {
            logger.get_statistics()
        } else {
            PerformanceStatistics::default()
        }
    }

    pub fn export_logs(&self, format: LogExportFormat) -> Result<String> {
        match format {
            LogExportFormat::Json => {
                let export = LogExport {
                    audit_logs: self.get_audit_logs(Some(1000)),
                    security_events: self.get_security_events(None),
                    performance_stats: self.get_performance_stats(),
                    timestamp: Utc::now(),
                };
                Ok(serde_json::to_string_pretty(&export)?)
            }
            LogExportFormat::Csv => {
                // Implement CSV export logic
                Ok("CSV export not yet implemented".to_string())
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStatistics {
    pub total_operations: u64,
    pub average_duration_ms: f64,
    pub slowest_operations: Vec<(String, u64)>,
    pub memory_usage_stats: MemoryStats,
    pub error_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub average_usage_mb: f64,
    pub peak_usage_mb: f64,
    pub current_usage_mb: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LogExport {
    pub audit_logs: Vec<AuditEntry>,
    pub security_events: Vec<SecurityEvent>,
    pub performance_stats: PerformanceStatistics,
    pub timestamp: DateTime<Utc>,
}

pub enum LogExportFormat {
    Json,
    Csv,
}

/// Custom tracing layer for advanced logging features
#[allow(dead_code)]
struct CustomLayer {
    logging_system: LoggingSystem,
    recursion_guard: std::cell::RefCell<bool>,
}

#[allow(dead_code)]
impl CustomLayer {
    fn new(logging_system: LoggingSystem) -> Self {
        Self {
            logging_system,
            recursion_guard: std::cell::RefCell::new(false),
        }
    }
}

impl<S> Layer<S> for CustomLayer
where
    S: Subscriber,
{
    fn on_event(&self, event: &Event<'_>, _ctx: tracing_subscriber::layer::Context<'_, S>) {
        // Check if we're already processing an event to prevent recursion
        if *self.recursion_guard.borrow() {
            return;
        }

        // Set the guard
        *self.recursion_guard.borrow_mut() = true;

        // Extract event information and create structured log entry
        let mut visitor = EventVisitor::new();
        event.record(&mut visitor);

        if let Some(_entry) = visitor.into_log_entry() {
            // Don't call log_structured as it would create a new tracing event
            // Instead, just store the entry or process it without creating new events
            // For now, we'll skip the structured logging to avoid recursion
        }

        // Clear the guard
        *self.recursion_guard.borrow_mut() = false;
    }
}

#[allow(dead_code)]
struct EventVisitor {
    fields: HashMap<String, serde_json::Value>,
    message: Option<String>,
}

#[allow(dead_code)]
impl EventVisitor {
    fn new() -> Self {
        Self {
            fields: HashMap::new(),
            message: None,
        }
    }

    fn into_log_entry(self) -> Option<LogEntry> {
        Some(LogEntry {
            timestamp: Utc::now(),
            level: LogLevel::Info,         // Would extract from event metadata
            module: "unknown".to_string(), // Would extract from event metadata
            message: self.message.unwrap_or_else(|| "No message".to_string()),
            correlation_id: None,
            user_id: None,
            session_id: None,
            request_id: None,
            fields: self.fields,
            error: None,
            performance: None,
        })
    }
}

impl Visit for EventVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.fields.insert(
            field.name().to_string(),
            serde_json::Value::String(format!("{:?}", value)),
        );
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message = Some(value.to_string());
        } else {
            self.fields.insert(
                field.name().to_string(),
                serde_json::Value::String(value.to_string()),
            );
        }
    }
}

// Implementation of various loggers
impl AuditLogger {
    fn new(config: AuditConfig) -> Self {
        Self {
            entries: Vec::new(),
            config,
        }
    }

    fn log_entry(&mut self, entry: AuditEntry) {
        self.entries.push(entry);

        // Keep only recent entries based on retention policy
        if self.entries.len() > 10000 {
            self.entries.remove(0);
        }
    }

    fn get_entries(&self, limit: Option<usize>) -> Vec<AuditEntry> {
        match limit {
            Some(n) => self.entries.iter().rev().take(n).cloned().collect(),
            None => self.entries.clone(),
        }
    }
}

impl SecurityLogger {
    fn new() -> Self {
        Self {
            events: Vec::new(),
            threat_detector: ThreatDetector::new(),
        }
    }

    fn log_event(&mut self, event: SecurityEvent) {
        self.events.push(event);

        // Keep only recent events
        if self.events.len() > 5000 {
            self.events.remove(0);
        }
    }

    fn get_events(&self, severity_filter: Option<SecuritySeverity>) -> Vec<SecurityEvent> {
        match severity_filter {
            Some(filter_severity) => self
                .events
                .iter()
                .filter(|e| e.severity == filter_severity)
                .cloned()
                .collect(),
            None => self.events.clone(),
        }
    }
}

impl ThreatDetector {
    fn new() -> Self {
        Self {
            patterns: Vec::new(),
            rate_limits: HashMap::new(),
        }
    }
}

impl PerformanceLogger {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            thresholds: PerformanceThresholds {
                slow_operation_ms: 1000,
                high_memory_usage_mb: 500.0,
                high_cpu_usage_percent: 80.0,
                alert_on_threshold_breach: true,
            },
        }
    }

    fn log_entry(&mut self, entry: PerformanceEntry) {
        self.entries.push(entry);

        // Keep only recent entries
        if self.entries.len() > 10000 {
            self.entries.remove(0);
        }
    }

    fn get_statistics(&self) -> PerformanceStatistics {
        if self.entries.is_empty() {
            return PerformanceStatistics::default();
        }

        let total_operations = self.entries.len() as u64;
        let average_duration = self.entries.iter().map(|e| e.duration_ms).sum::<u64>() as f64
            / total_operations as f64;

        let error_count = self.entries.iter().filter(|e| e.error_occurred).count();
        let error_rate = error_count as f64 / total_operations as f64 * 100.0;

        let mut operations_by_duration: Vec<_> = self
            .entries
            .iter()
            .map(|e| (e.operation.clone(), e.duration_ms))
            .collect();
        operations_by_duration.sort_by(|a, b| b.1.cmp(&a.1));

        PerformanceStatistics {
            total_operations,
            average_duration_ms: average_duration,
            slowest_operations: operations_by_duration.into_iter().take(10).collect(),
            memory_usage_stats: MemoryStats {
                average_usage_mb: 0.0, // Would calculate from memory_delta_mb
                peak_usage_mb: 0.0,
                current_usage_mb: 0.0,
            },
            error_rate,
        }
    }
}

impl MetricsCollector {
    fn new() -> Self {
        Self {
            counters: HashMap::new(),
            gauges: HashMap::new(),
            histograms: HashMap::new(),
            timers: HashMap::new(),
        }
    }
}

// Default implementations
impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            format: LogFormat::Json,
            output: LogOutput::Console,
            enable_audit: true,
            enable_security: true,
            enable_performance: true,
            enable_metrics: true,
            structured_logging: true,
            correlation_id_header: "x-correlation-id".to_string(),
            sensitive_fields: vec![
                "password".to_string(),
                "api_key".to_string(),
                "token".to_string(),
                "secret".to_string(),
            ],
            log_retention_days: 30,
            max_log_file_size_mb: 100,
        }
    }
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enable_data_access_logging: true,
            enable_auth_logging: true,
            enable_config_change_logging: true,
            log_failed_attempts: true,
            include_sensitive_data: false,
            retention_days: 90,
        }
    }
}

impl Default for PerformanceStatistics {
    fn default() -> Self {
        Self {
            total_operations: 0,
            average_duration_ms: 0.0,
            slowest_operations: Vec::new(),
            memory_usage_stats: MemoryStats {
                average_usage_mb: 0.0,
                peak_usage_mb: 0.0,
                current_usage_mb: 0.0,
            },
            error_rate: 0.0,
        }
    }
}

/// Convenience macros for structured logging
#[macro_export]
macro_rules! log_audit {
    ($logging_system:expr, $event_type:expr, $user_id:expr, $resource:expr, $action:expr, $result:expr) => {
        $logging_system.log_audit(AuditEntry {
            timestamp: chrono::Utc::now(),
            event_type: $event_type,
            user_id: $user_id,
            resource: $resource.to_string(),
            action: $action.to_string(),
            result: $result,
            ip_address: None,
            user_agent: None,
            correlation_id: None,
            details: std::collections::HashMap::new(),
        });
    };
}

#[macro_export]
macro_rules! log_security {
    ($logging_system:expr, $event_type:expr, $severity:expr, $description:expr) => {
        $logging_system.log_security(SecurityEvent {
            timestamp: chrono::Utc::now(),
            event_type: $event_type,
            severity: $severity,
            source_ip: None,
            user_id: None,
            description: $description.to_string(),
            indicators: Vec::new(),
            mitigations: Vec::new(),
            correlation_id: None,
        });
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logging_system_creation() {
        let config = LoggingConfig::default();
        let logging_system = LoggingSystem::new(config);
        assert!(logging_system.is_ok());
    }

    #[test]
    fn test_structured_logging() {
        let config = LoggingConfig::default();
        let logging_system = LoggingSystem::new(config).unwrap();

        let entry = LogEntry {
            timestamp: Utc::now(),
            level: LogLevel::Info,
            module: "test".to_string(),
            message: "Test message".to_string(),
            correlation_id: Some("test-123".to_string()),
            user_id: None,
            session_id: None,
            request_id: None,
            fields: HashMap::new(),
            error: None,
            performance: None,
        };

        logging_system.log_structured(entry);
    }

    #[test]
    fn test_audit_logging() {
        let config = LoggingConfig::default();
        let logging_system = LoggingSystem::new(config).unwrap();

        let audit_entry = AuditEntry {
            timestamp: Utc::now(),
            event_type: AuditEventType::DataAccess,
            user_id: Some("user123".to_string()),
            resource: "conversations".to_string(),
            action: "read".to_string(),
            result: AuditResult::Success,
            ip_address: Some("192.168.1.1".to_string()),
            user_agent: None,
            correlation_id: None,
            details: HashMap::new(),
        };

        logging_system.log_audit(audit_entry);

        let logs = logging_system.get_audit_logs(Some(10));
        assert_eq!(logs.len(), 1);
    }

    #[test]
    fn test_performance_logging() {
        let config = LoggingConfig::default();
        let logging_system = LoggingSystem::new(config).unwrap();

        let perf_entry = PerformanceEntry {
            timestamp: Utc::now(),
            operation: "clustering".to_string(),
            duration_ms: 1500,
            memory_delta_mb: Some(50.0),
            cpu_usage: Some(75.0),
            input_size: Some(1000),
            output_size: Some(10),
            cache_hit: Some(false),
            error_occurred: false,
            custom_metrics: HashMap::new(),
        };

        logging_system.log_performance(perf_entry);

        let stats = logging_system.get_performance_stats();
        assert_eq!(stats.total_operations, 1);
        assert_eq!(stats.average_duration_ms, 1500.0);
    }

    #[test]
    fn test_metrics_collection() {
        let config = LoggingConfig::default();
        let logging_system = LoggingSystem::new(config).unwrap();

        logging_system.increment_counter("requests", 5);
        logging_system.set_gauge("cpu_usage", 65.5);
        logging_system.record_histogram("response_time", 250.0);
        logging_system.record_timer("operation_duration", 1000);

        // Verify metrics are collected (would need accessor methods in real implementation)
    }

    #[test]
    fn test_no_stack_overflow_with_heavy_logging() {
        // This test ensures that logging within logging doesn't cause stack overflow
        let config = LoggingConfig::default();
        let logging_system = LoggingSystem::new(config).unwrap();

        // Try to log multiple times in a way that could cause recursion
        // Also test with different log levels to ensure all paths work
        for i in 0..20 {
            for level in [
                LogLevel::Trace,
                LogLevel::Debug,
                LogLevel::Info,
                LogLevel::Warn,
                LogLevel::Error,
            ] {
                let entry = LogEntry {
                    timestamp: Utc::now(),
                    level,
                    module: format!("test_{}", i),
                    message: format!("Test message {} at level {:?}", i, level),
                    correlation_id: Some(format!("corr_{}", i)),
                    user_id: None,
                    session_id: None,
                    request_id: None,
                    fields: HashMap::new(),
                    error: None,
                    performance: None,
                };

                logging_system.log_structured(entry);
            }
        }

        // If we get here without stack overflow, the test passes
        assert!(true);
    }

    #[test]
    fn test_sensitive_field_filtering() {
        let config = LoggingConfig::default();
        let logging_system = LoggingSystem::new(config).unwrap();

        let mut fields = HashMap::new();
        fields.insert(
            "user_name".to_string(),
            serde_json::Value::String("john".to_string()),
        );
        fields.insert(
            "password".to_string(),
            serde_json::Value::String("secret123".to_string()),
        );

        let entry = LogEntry {
            timestamp: Utc::now(),
            level: LogLevel::Info,
            module: "auth".to_string(),
            message: "User login".to_string(),
            correlation_id: None,
            user_id: None,
            session_id: None,
            request_id: None,
            fields,
            error: None,
            performance: None,
        };

        let filtered = logging_system.filter_sensitive_fields(entry);
        assert_eq!(
            filtered.fields["password"],
            serde_json::Value::String("[REDACTED]".to_string())
        );
        assert_eq!(
            filtered.fields["user_name"],
            serde_json::Value::String("john".to_string())
        );
    }
}
