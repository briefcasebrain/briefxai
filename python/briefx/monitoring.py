"""
Comprehensive monitoring and observability system for BriefX
Provides metrics, health checks, and performance tracking
"""

import json
import logging
import psutil
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class ApiMetrics:
    """Metrics for API calls"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    average_latency_ms: float = 0.0
    last_call: Optional[datetime] = None
    rate_limited_calls: int = 0


@dataclass
class ComponentMetrics:
    """Metrics for system components"""
    executions: int = 0
    total_time_ms: int = 0
    average_time_ms: float = 0.0
    errors: int = 0
    last_execution: Optional[datetime] = None


@dataclass
class SystemMetrics:
    """Overall system metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time_ms: float = 0.0
    active_connections: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    api_calls: Dict[str, ApiMetrics] = field(default_factory=dict)
    component_metrics: Dict[str, ComponentMetrics] = field(default_factory=dict)
    uptime_seconds: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MemorySnapshot:
    """Memory usage snapshot"""
    timestamp: datetime
    heap_used_mb: float
    heap_total_mb: float
    rss_mb: float
    vms_mb: float
    percent_used: float


@dataclass
class ThroughputMeasurement:
    """Throughput measurement for a component"""
    timestamp: datetime
    component: str
    items_processed: int
    duration_ms: int
    items_per_second: float


@dataclass
class ComponentHealth:
    """Health status of a component"""
    status: HealthStatus
    response_time_ms: Optional[float] = None
    error_rate: float = 0.0
    last_error: Optional[str] = None
    uptime_percent: float = 100.0


@dataclass
class HealthCheckResult:
    """Result of health check"""
    status: HealthStatus
    timestamp: datetime
    checks: Dict[str, ComponentHealth]
    overall_score: float


@dataclass
class PerformanceReport:
    """Performance report summary"""
    uptime_seconds: float
    total_requests: int
    average_response_time_ms: float
    throughput_summary: Dict[str, float]
    memory_summary: Dict[str, float]
    top_slow_components: List[Tuple[str, float]]
    cache_efficiency: float


class MonitoringSystem:
    """Main monitoring and observability system"""
    
    def __init__(self):
        self.metrics = SystemMetrics()
        self.start_time = time.time()
        self.memory_snapshots = deque(maxlen=1000)
        self.throughput_measurements = deque(maxlen=1000)
        self.operation_times = defaultdict(lambda: deque(maxlen=100))
        self.alerts = []
        self.custom_metrics = {}
        
        # Performance thresholds
        self.thresholds = {
            "response_time_ms": 5000,
            "error_rate": 0.05,
            "memory_percent": 80,
            "cpu_percent": 80
        }
    
    def record_request(self, success: bool, duration: float):
        """Record a request"""
        
        self.metrics.total_requests += 1
        
        if success:
            self.metrics.successful_requests += 1
        else:
            self.metrics.failed_requests += 1
        
        # Update average response time
        duration_ms = duration * 1000
        total_time = self.metrics.average_response_time_ms * (self.metrics.total_requests - 1)
        total_time += duration_ms
        self.metrics.average_response_time_ms = total_time / self.metrics.total_requests
        
        self.metrics.last_updated = datetime.utcnow()
        
        # Check for alerts
        if duration_ms > self.thresholds["response_time_ms"]:
            self._add_alert(f"Slow request: {duration_ms:.2f}ms")
        
        logger.debug(
            f"Request recorded: success={success}, duration={duration:.3f}s, "
            f"total={self.metrics.total_requests}"
        )
    
    def record_api_call(self, provider: str, success: bool, duration: float):
        """Record an API call"""
        
        if provider not in self.metrics.api_calls:
            self.metrics.api_calls[provider] = ApiMetrics()
        
        api_metrics = self.metrics.api_calls[provider]
        api_metrics.total_calls += 1
        api_metrics.last_call = datetime.utcnow()
        
        if success:
            api_metrics.successful_calls += 1
        else:
            api_metrics.failed_calls += 1
        
        # Update average latency
        duration_ms = duration * 1000
        total_time = api_metrics.average_latency_ms * (api_metrics.total_calls - 1)
        total_time += duration_ms
        api_metrics.average_latency_ms = total_time / api_metrics.total_calls
        
        logger.info(
            f"API call recorded: provider={provider}, success={success}, "
            f"duration={duration:.3f}s"
        )
    
    def record_component_execution(self, component: str, duration: float, success: bool):
        """Record component execution"""
        
        if component not in self.metrics.component_metrics:
            self.metrics.component_metrics[component] = ComponentMetrics()
        
        comp_metrics = self.metrics.component_metrics[component]
        comp_metrics.executions += 1
        duration_ms = int(duration * 1000)
        comp_metrics.total_time_ms += duration_ms
        comp_metrics.average_time_ms = comp_metrics.total_time_ms / comp_metrics.executions
        comp_metrics.last_execution = datetime.utcnow()
        
        if not success:
            comp_metrics.errors += 1
        
        # Track operation times for performance analysis
        self.operation_times[component].append(duration)
        
        logger.debug(
            f"Component execution recorded: component={component}, "
            f"duration={duration:.3f}s, success={success}"
        )
    
    def record_cache_hit(self, hit: bool):
        """Record cache hit/miss"""
        
        if hit:
            self.metrics.cache_hits += 1
        else:
            self.metrics.cache_misses += 1
    
    def record_throughput(self, component: str, items_processed: int, duration: float):
        """Record throughput measurement"""
        
        items_per_second = items_processed / duration if duration > 0 else 0
        
        measurement = ThroughputMeasurement(
            timestamp=datetime.utcnow(),
            component=component,
            items_processed=items_processed,
            duration_ms=int(duration * 1000),
            items_per_second=items_per_second
        )
        
        self.throughput_measurements.append(measurement)
        
        logger.info(
            f"Throughput recorded: component={component}, items={items_processed}, "
            f"rate={items_per_second:.2f}/sec"
        )
    
    def record_memory_usage(self):
        """Record current memory usage"""
        
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            snapshot = MemorySnapshot(
                timestamp=datetime.utcnow(),
                heap_used_mb=memory_info.rss / 1024 / 1024,
                heap_total_mb=memory_info.vms / 1024 / 1024,
                rss_mb=memory_info.rss / 1024 / 1024,
                vms_mb=memory_info.vms / 1024 / 1024,
                percent_used=memory_percent
            )
            
            self.memory_snapshots.append(snapshot)
            
            # Check for memory alert
            if memory_percent > self.thresholds["memory_percent"]:
                self._add_alert(f"High memory usage: {memory_percent:.1f}%")
            
        except Exception as e:
            logger.error(f"Failed to record memory usage: {e}")
    
    def record_custom_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a custom metric"""
        
        if name not in self.custom_metrics:
            self.custom_metrics[name] = []
        
        self.custom_metrics[name].append({
            "timestamp": datetime.utcnow(),
            "value": value,
            "tags": tags or {}
        })
        
        # Keep only last 1000 values per metric
        if len(self.custom_metrics[name]) > 1000:
            self.custom_metrics[name].pop(0)
    
    def perform_health_check(self) -> HealthCheckResult:
        """Perform comprehensive health check"""
        
        checks = {}
        total_score = 0.0
        check_count = 0
        
        # Check API health
        for provider, api_metrics in self.metrics.api_calls.items():
            error_rate = 0.0
            if api_metrics.total_calls > 0:
                error_rate = api_metrics.failed_calls / api_metrics.total_calls
            
            status = self._get_health_status_from_error_rate(error_rate)
            score = self._get_score_from_status(status)
            
            checks[f"api_{provider}"] = ComponentHealth(
                status=status,
                response_time_ms=api_metrics.average_latency_ms,
                error_rate=error_rate,
                last_error=None,
                uptime_percent=100.0 - (error_rate * 100.0)
            )
            
            total_score += score
            check_count += 1
        
        # Check component health
        for component, comp_metrics in self.metrics.component_metrics.items():
            error_rate = 0.0
            if comp_metrics.executions > 0:
                error_rate = comp_metrics.errors / comp_metrics.executions
            
            status = self._get_health_status_from_error_rate(error_rate)
            score = self._get_score_from_status(status)
            
            checks[component] = ComponentHealth(
                status=status,
                response_time_ms=comp_metrics.average_time_ms,
                error_rate=error_rate,
                last_error=None,
                uptime_percent=100.0 - (error_rate * 100.0)
            )
            
            total_score += score
            check_count += 1
        
        # Check system resources
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            
            # CPU health
            cpu_status = HealthStatus.HEALTHY
            if cpu_percent > 90:
                cpu_status = HealthStatus.CRITICAL
            elif cpu_percent > 80:
                cpu_status = HealthStatus.UNHEALTHY
            elif cpu_percent > 70:
                cpu_status = HealthStatus.DEGRADED
            
            checks["system_cpu"] = ComponentHealth(
                status=cpu_status,
                response_time_ms=None,
                error_rate=0.0,
                last_error=None,
                uptime_percent=100.0
            )
            
            total_score += self._get_score_from_status(cpu_status)
            check_count += 1
            
            # Memory health
            memory_status = HealthStatus.HEALTHY
            if memory_percent > 90:
                memory_status = HealthStatus.CRITICAL
            elif memory_percent > 80:
                memory_status = HealthStatus.UNHEALTHY
            elif memory_percent > 70:
                memory_status = HealthStatus.DEGRADED
            
            checks["system_memory"] = ComponentHealth(
                status=memory_status,
                response_time_ms=None,
                error_rate=0.0,
                last_error=None,
                uptime_percent=100.0
            )
            
            total_score += self._get_score_from_status(memory_status)
            check_count += 1
            
        except Exception as e:
            logger.error(f"Failed to check system resources: {e}")
        
        # Calculate overall score and status
        overall_score = total_score / check_count if check_count > 0 else 100.0
        
        if overall_score >= 90:
            overall_status = HealthStatus.HEALTHY
        elif overall_score >= 70:
            overall_status = HealthStatus.DEGRADED
        elif overall_score >= 50:
            overall_status = HealthStatus.UNHEALTHY
        else:
            overall_status = HealthStatus.CRITICAL
        
        return HealthCheckResult(
            status=overall_status,
            timestamp=datetime.utcnow(),
            checks=checks,
            overall_score=overall_score
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        
        self.metrics.uptime_seconds = time.time() - self.start_time
        
        return {
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "average_response_time_ms": self.metrics.average_response_time_ms,
            "active_connections": self.metrics.active_connections,
            "cache_hits": self.metrics.cache_hits,
            "cache_misses": self.metrics.cache_misses,
            "cache_efficiency": self._calculate_cache_efficiency(),
            "api_calls": {
                provider: {
                    "total_calls": metrics.total_calls,
                    "successful_calls": metrics.successful_calls,
                    "failed_calls": metrics.failed_calls,
                    "average_latency_ms": metrics.average_latency_ms,
                    "last_call": metrics.last_call.isoformat() if metrics.last_call else None
                }
                for provider, metrics in self.metrics.api_calls.items()
            },
            "component_metrics": {
                component: {
                    "executions": metrics.executions,
                    "average_time_ms": metrics.average_time_ms,
                    "errors": metrics.errors,
                    "error_rate": metrics.errors / metrics.executions if metrics.executions > 0 else 0,
                    "last_execution": metrics.last_execution.isoformat() if metrics.last_execution else None
                }
                for component, metrics in self.metrics.component_metrics.items()
            },
            "uptime_seconds": self.metrics.uptime_seconds,
            "last_updated": self.metrics.last_updated.isoformat()
        }
    
    def get_performance_report(self) -> PerformanceReport:
        """Generate performance report"""
        
        return PerformanceReport(
            uptime_seconds=time.time() - self.start_time,
            total_requests=self.metrics.total_requests,
            average_response_time_ms=self.metrics.average_response_time_ms,
            throughput_summary=self._calculate_throughput_summary(),
            memory_summary=self._calculate_memory_summary(),
            top_slow_components=self._get_slowest_components(),
            cache_efficiency=self._calculate_cache_efficiency()
        )
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format"""
        
        metrics = self.get_metrics()
        
        if format == "json":
            return json.dumps(metrics, indent=2, default=str)
        elif format == "prometheus":
            return self._format_prometheus_metrics(metrics)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _get_health_status_from_error_rate(self, error_rate: float) -> HealthStatus:
        """Get health status from error rate"""
        
        if error_rate < 0.01:
            return HealthStatus.HEALTHY
        elif error_rate < 0.05:
            return HealthStatus.DEGRADED
        elif error_rate < 0.20:
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.CRITICAL
    
    def _get_score_from_status(self, status: HealthStatus) -> float:
        """Get numerical score from health status"""
        
        scores = {
            HealthStatus.HEALTHY: 100.0,
            HealthStatus.DEGRADED: 75.0,
            HealthStatus.UNHEALTHY: 50.0,
            HealthStatus.CRITICAL: 0.0
        }
        return scores.get(status, 0.0)
    
    def _calculate_cache_efficiency(self) -> float:
        """Calculate cache efficiency percentage"""
        
        total_cache_ops = self.metrics.cache_hits + self.metrics.cache_misses
        if total_cache_ops == 0:
            return 0.0
        return (self.metrics.cache_hits / total_cache_ops) * 100.0
    
    def _calculate_throughput_summary(self) -> Dict[str, float]:
        """Calculate throughput summary by component"""
        
        summary = {}
        component_measurements = defaultdict(list)
        
        for measurement in self.throughput_measurements:
            component_measurements[measurement.component].append(measurement.items_per_second)
        
        for component, rates in component_measurements.items():
            if rates:
                summary[component] = sum(rates) / len(rates)
        
        return summary
    
    def _calculate_memory_summary(self) -> Dict[str, float]:
        """Calculate memory usage summary"""
        
        if not self.memory_snapshots:
            return {}
        
        recent_snapshots = list(self.memory_snapshots)[-100:]
        
        return {
            "current_mb": recent_snapshots[-1].rss_mb if recent_snapshots else 0,
            "average_mb": sum(s.rss_mb for s in recent_snapshots) / len(recent_snapshots),
            "peak_mb": max(s.rss_mb for s in recent_snapshots),
            "current_percent": recent_snapshots[-1].percent_used if recent_snapshots else 0
        }
    
    def _get_slowest_components(self, limit: int = 5) -> List[Tuple[str, float]]:
        """Get slowest components by average execution time"""
        
        components = []
        
        for component, metrics in self.metrics.component_metrics.items():
            if metrics.executions > 0:
                components.append((component, metrics.average_time_ms))
        
        components.sort(key=lambda x: x[1], reverse=True)
        return components[:limit]
    
    def _format_prometheus_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format metrics in Prometheus format"""
        
        lines = []
        
        # Request metrics
        lines.append(f"briefx_total_requests {metrics['total_requests']}")
        lines.append(f"briefx_successful_requests {metrics['successful_requests']}")
        lines.append(f"briefx_failed_requests {metrics['failed_requests']}")
        lines.append(f"briefx_average_response_time_ms {metrics['average_response_time_ms']}")
        
        # Cache metrics
        lines.append(f"briefx_cache_hits {metrics['cache_hits']}")
        lines.append(f"briefx_cache_misses {metrics['cache_misses']}")
        lines.append(f"briefx_cache_efficiency {metrics['cache_efficiency']}")
        
        # API metrics
        for provider, api_metrics in metrics['api_calls'].items():
            lines.append(f'briefx_api_total_calls{{provider="{provider}"}} {api_metrics["total_calls"]}')
            lines.append(f'briefx_api_successful_calls{{provider="{provider}"}} {api_metrics["successful_calls"]}')
            lines.append(f'briefx_api_failed_calls{{provider="{provider}"}} {api_metrics["failed_calls"]}')
            lines.append(f'briefx_api_average_latency_ms{{provider="{provider}"}} {api_metrics["average_latency_ms"]}')
        
        # Component metrics
        for component, comp_metrics in metrics['component_metrics'].items():
            lines.append(f'briefx_component_executions{{component="{component}"}} {comp_metrics["executions"]}')
            lines.append(f'briefx_component_average_time_ms{{component="{component}"}} {comp_metrics["average_time_ms"]}')
            lines.append(f'briefx_component_errors{{component="{component}"}} {comp_metrics["errors"]}')
        
        # System metrics
        lines.append(f"briefx_uptime_seconds {metrics['uptime_seconds']}")
        
        return "\n".join(lines)
    
    def _add_alert(self, message: str):
        """Add an alert"""
        
        alert = {
            "timestamp": datetime.utcnow(),
            "message": message
        }
        self.alerts.append(alert)
        
        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts.pop(0)
        
        logger.warning(f"Alert: {message}")
    
    def get_alerts(self, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get alerts since specified time"""
        
        if since:
            return [
                {"timestamp": a["timestamp"].isoformat(), "message": a["message"]}
                for a in self.alerts
                if a["timestamp"] >= since
            ]
        return [
            {"timestamp": a["timestamp"].isoformat(), "message": a["message"]}
            for a in self.alerts
        ]


# Global monitoring system instance
monitoring_system = MonitoringSystem()


# Convenience decorator for monitoring function execution
def monitor_execution(component: str):
    """Decorator to monitor function execution"""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            result = None
            
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                success = False
                raise e
            finally:
                duration = time.time() - start_time
                monitoring_system.record_component_execution(component, duration, success)
            
            return result
        
        return wrapper
    
    return decorator