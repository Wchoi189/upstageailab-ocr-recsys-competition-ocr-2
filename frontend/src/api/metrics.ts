/**
 * Telemetry and Metrics API Client
 *
 * Provides methods for logging worker events, performance metrics,
 * cache hits/misses, and fallback routing decisions.
 */

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export type WorkerEventType =
  | "task_queued"
  | "task_started"
  | "task_completed"
  | "task_failed"
  | "task_cancelled";

export type TaskType =
  | "auto_contrast"
  | "gaussian_blur"
  | "resize"
  | "rembg"
  | "inference"
  | "evaluation";

export type RoutingTarget = "client" | "backend";

export interface WorkerEvent {
  event_type: WorkerEventType;
  worker_id: string;
  task_id: string;
  timestamp?: string;
  metadata?: Record<string, unknown>;
}

export interface PerformanceMetric {
  task_type: TaskType;
  duration_ms: number;
  image_size?: number | null;
  success?: boolean;
  timestamp?: string;
  metadata?: Record<string, unknown>;
}

export interface CacheMetric {
  operation: string;
  hit: boolean;
  cache_key?: string | null;
  timestamp?: string;
}

export interface FallbackMetric {
  operation: string;
  routed_to: RoutingTarget;
  reason: string;
  image_size?: number | null;
  timestamp?: string;
}

export interface MetricsSummary {
  total_tasks: number;
  tasks_by_status: Record<string, number>;
  avg_duration_ms: Record<string, number>;
  cache_hit_rate: number;
  fallback_rate: Record<string, number>;
  worker_queue_depth: number;
  time_range_hours: number;
}

/**
 * Log a worker lifecycle event
 */
export async function logWorkerEvent(event: WorkerEvent): Promise<void> {
  const response = await fetch(`${API_BASE}/api/metrics/events/worker`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(event),
  });

  if (!response.ok) {
    console.warn("Failed to log worker event:", await response.text());
  }
}

/**
 * Log a performance metric
 */
export async function logPerformanceMetric(
  metric: PerformanceMetric,
): Promise<void> {
  const response = await fetch(`${API_BASE}/api/metrics/events/performance`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(metric),
  });

  if (!response.ok) {
    console.warn("Failed to log performance metric:", await response.text());
  }
}

/**
 * Log a cache hit/miss event
 */
export async function logCacheMetric(metric: CacheMetric): Promise<void> {
  const response = await fetch(`${API_BASE}/api/metrics/events/cache`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(metric),
  });

  if (!response.ok) {
    console.warn("Failed to log cache metric:", await response.text());
  }
}

/**
 * Log a fallback routing decision
 */
export async function logFallbackMetric(metric: FallbackMetric): Promise<void> {
  const response = await fetch(`${API_BASE}/api/metrics/events/fallback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(metric),
  });

  if (!response.ok) {
    console.warn("Failed to log fallback metric:", await response.text());
  }
}

/**
 * Get aggregated metrics summary
 */
export async function getMetricsSummary(
  hours: number = 1,
): Promise<MetricsSummary> {
  const response = await fetch(
    `${API_BASE}/api/metrics/summary?hours=${hours}`,
  );

  if (!response.ok) {
    throw new Error(`Failed to fetch metrics summary: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get recent events of a specific type
 */
export async function getRecentEvents(
  eventType: "worker" | "performance" | "cache" | "fallback" = "worker",
  limit: number = 100,
): Promise<Record<string, unknown>[]> {
  const response = await fetch(
    `${API_BASE}/api/metrics/events/recent?event_type=${eventType}&limit=${limit}`,
  );

  if (!response.ok) {
    throw new Error(`Failed to fetch recent events: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Clear all metrics (for testing/debugging)
 */
export async function clearMetrics(): Promise<void> {
  const response = await fetch(`${API_BASE}/api/metrics/events`, {
    method: "DELETE",
  });

  if (!response.ok) {
    throw new Error(`Failed to clear metrics: ${response.statusText}`);
  }
}
