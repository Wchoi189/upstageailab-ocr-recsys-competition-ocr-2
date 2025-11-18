"""Telemetry and metrics API router for monitoring worker performance.

Provides endpoints for:
- Worker lifecycle events
- Performance metrics
- Cache hit rates
- Fallback routing statistics
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()

# In-memory metrics store (replace with Redis/TimescaleDB in production)
_metrics_store: dict[str, list[dict[str, Any]]] = {
    "worker_events": [],
    "performance_metrics": [],
    "cache_metrics": [],
    "fallback_metrics": [],
}

# Configuration
MAX_METRICS_PER_TYPE = 10000  # Keep last 10k events per type
METRICS_RETENTION_HOURS = 24  # Keep metrics for 24 hours


class WorkerEvent(BaseModel):
    """Worker lifecycle event."""

    event_type: Literal["task_queued", "task_started", "task_completed", "task_failed", "task_cancelled"]
    worker_id: str = Field(description="Unique identifier for the worker")
    task_id: str = Field(description="Unique identifier for the task")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional event data")


class PerformanceMetric(BaseModel):
    """Performance measurement for a task."""

    task_type: Literal["auto_contrast", "gaussian_blur", "resize", "rembg", "inference", "evaluation"]
    duration_ms: float = Field(description="Task duration in milliseconds")
    image_size: int | None = Field(None, description="Image size in bytes")
    success: bool = Field(default=True)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CacheMetric(BaseModel):
    """Cache hit/miss tracking."""

    operation: str = Field(description="Operation type (e.g., 'rembg', 'inference')")
    hit: bool = Field(description="Whether cache was hit")
    cache_key: str | None = Field(None, description="Cache key (hashed)")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class FallbackMetric(BaseModel):
    """Fallback routing decision tracking."""

    operation: str = Field(description="Operation that triggered fallback decision")
    routed_to: Literal["client", "backend"] = Field(description="Where the operation was routed")
    reason: str = Field(description="Reason for routing decision")
    image_size: int | None = Field(None, description="Image size in bytes")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MetricsSummary(BaseModel):
    """Aggregated metrics summary."""

    total_tasks: int
    tasks_by_status: dict[str, int]
    avg_duration_ms: dict[str, float]  # By task type
    cache_hit_rate: float
    fallback_rate: dict[str, float]  # By operation
    worker_queue_depth: int
    time_range_hours: int


def _cleanup_old_metrics() -> None:
    """Remove metrics older than retention period."""
    cutoff_time = datetime.utcnow() - timedelta(hours=METRICS_RETENTION_HOURS)

    for metric_type, events in _metrics_store.items():
        _metrics_store[metric_type] = [
            event for event in events if event.get("timestamp", datetime.utcnow()) > cutoff_time
        ]


def _trim_metrics() -> None:
    """Trim metrics to max size per type."""
    for metric_type, events in _metrics_store.items():
        if len(events) > MAX_METRICS_PER_TYPE:
            # Keep most recent events
            _metrics_store[metric_type] = events[-MAX_METRICS_PER_TYPE :]


@router.post("/events/worker", status_code=201)
def log_worker_event(event: WorkerEvent) -> dict[str, str]:
    """Log a worker lifecycle event."""
    _cleanup_old_metrics()
    _metrics_store["worker_events"].append(event.model_dump())
    _trim_metrics()
    return {"status": "logged"}


@router.post("/events/performance", status_code=201)
def log_performance_metric(metric: PerformanceMetric) -> dict[str, str]:
    """Log a performance metric."""
    _cleanup_old_metrics()
    _metrics_store["performance_metrics"].append(metric.model_dump())
    _trim_metrics()
    return {"status": "logged"}


@router.post("/events/cache", status_code=201)
def log_cache_metric(metric: CacheMetric) -> dict[str, str]:
    """Log a cache hit/miss event."""
    _cleanup_old_metrics()
    _metrics_store["cache_metrics"].append(metric.model_dump())
    _trim_metrics()
    return {"status": "logged"}


@router.post("/events/fallback", status_code=201)
def log_fallback_metric(metric: FallbackMetric) -> dict[str, str]:
    """Log a fallback routing decision."""
    _cleanup_old_metrics()
    _metrics_store["fallback_metrics"].append(metric.model_dump())
    _trim_metrics()
    return {"status": "logged"}


@router.get("/summary", response_model=MetricsSummary)
def get_metrics_summary(hours: int = 1) -> MetricsSummary:
    """Get aggregated metrics summary for the last N hours."""
    _cleanup_old_metrics()

    cutoff_time = datetime.utcnow() - timedelta(hours=hours)

    # Filter events by time range
    recent_worker_events = [
        e for e in _metrics_store["worker_events"] if e.get("timestamp", datetime.utcnow()) > cutoff_time
    ]
    recent_perf_metrics = [
        m for m in _metrics_store["performance_metrics"] if m.get("timestamp", datetime.utcnow()) > cutoff_time
    ]
    recent_cache_metrics = [
        m for m in _metrics_store["cache_metrics"] if m.get("timestamp", datetime.utcnow()) > cutoff_time
    ]
    recent_fallback_metrics = [
        m for m in _metrics_store["fallback_metrics"] if m.get("timestamp", datetime.utcnow()) > cutoff_time
    ]

    # Calculate total tasks
    total_tasks = len([e for e in recent_worker_events if e.get("event_type") == "task_queued"])

    # Tasks by status
    tasks_by_status: dict[str, int] = {}
    for event in recent_worker_events:
        event_type = event.get("event_type", "unknown")
        tasks_by_status[event_type] = tasks_by_status.get(event_type, 0) + 1

    # Average duration by task type
    avg_duration_ms: dict[str, float] = {}
    task_durations: dict[str, list[float]] = {}
    for metric in recent_perf_metrics:
        task_type = metric.get("task_type", "unknown")
        duration = metric.get("duration_ms", 0.0)
        if task_type not in task_durations:
            task_durations[task_type] = []
        task_durations[task_type].append(duration)

    for task_type, durations in task_durations.items():
        avg_duration_ms[task_type] = sum(durations) / len(durations) if durations else 0.0

    # Cache hit rate
    cache_hits = len([m for m in recent_cache_metrics if m.get("hit", False)])
    cache_total = len(recent_cache_metrics)
    cache_hit_rate = cache_hits / cache_total if cache_total > 0 else 0.0

    # Fallback rate by operation
    fallback_rate: dict[str, float] = {}
    fallback_by_op: dict[str, dict[str, int]] = {}
    for metric in recent_fallback_metrics:
        operation = metric.get("operation", "unknown")
        routed_to = metric.get("routed_to", "unknown")
        if operation not in fallback_by_op:
            fallback_by_op[operation] = {"client": 0, "backend": 0}
        fallback_by_op[operation][routed_to] = fallback_by_op[operation].get(routed_to, 0) + 1

    for operation, counts in fallback_by_op.items():
        total = counts.get("client", 0) + counts.get("backend", 0)
        backend_count = counts.get("backend", 0)
        fallback_rate[operation] = backend_count / total if total > 0 else 0.0

    # Current worker queue depth (count queued but not started)
    queued_tasks = {e.get("task_id") for e in recent_worker_events if e.get("event_type") == "task_queued"}
    started_tasks = {e.get("task_id") for e in recent_worker_events if e.get("event_type") == "task_started"}
    worker_queue_depth = len(queued_tasks - started_tasks)

    return MetricsSummary(
        total_tasks=total_tasks,
        tasks_by_status=tasks_by_status,
        avg_duration_ms=avg_duration_ms,
        cache_hit_rate=cache_hit_rate,
        fallback_rate=fallback_rate,
        worker_queue_depth=worker_queue_depth,
        time_range_hours=hours,
    )


@router.get("/events/recent")
def get_recent_events(
    event_type: Literal["worker", "performance", "cache", "fallback"] = "worker", limit: int = 100
) -> list[dict[str, Any]]:
    """Get recent events of a specific type."""
    _cleanup_old_metrics()

    metric_key_map = {
        "worker": "worker_events",
        "performance": "performance_metrics",
        "cache": "cache_metrics",
        "fallback": "fallback_metrics",
    }

    metric_key = metric_key_map.get(event_type, "worker_events")
    events = _metrics_store[metric_key]

    # Return most recent events
    return events[-limit:]


@router.delete("/events", status_code=204)
def clear_metrics() -> None:
    """Clear all metrics (for testing/debugging)."""
    for key in _metrics_store:
        _metrics_store[key] = []
