/**
 * Worker Telemetry Integration
 *
 * Provides telemetry logging for worker lifecycle events and performance metrics.
 * Integrates with the metrics API to track worker performance in production.
 */

import {
  logWorkerEvent,
  logPerformanceMetric,
  type WorkerEventType,
  type TaskType,
} from "../api/metrics";

/**
 * Generate unique task ID
 */
export function generateTaskId(): string {
  return `task_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Generate unique worker ID
 */
export function generateWorkerId(): string {
  return `worker_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Log worker lifecycle event (fire-and-forget)
 */
export function logWorkerLifecycleEvent(
  eventType: WorkerEventType,
  workerId: string,
  taskId: string,
  metadata?: Record<string, unknown>,
): void {
  // Fire-and-forget - don't block on telemetry
  logWorkerEvent({
    event_type: eventType,
    worker_id: workerId,
    task_id: taskId,
    metadata,
  }).catch((err) => {
    console.warn("Failed to log worker event:", err);
  });
}

/**
 * Log performance metric (fire-and-forget)
 */
export function logTaskPerformance(
  taskType: TaskType,
  durationMs: number,
  imageSize: number | null = null,
  success: boolean = true,
  metadata?: Record<string, unknown>,
): void {
  // Fire-and-forget - don't block on telemetry
  logPerformanceMetric({
    task_type: taskType,
    duration_ms: durationMs,
    image_size: imageSize,
    success,
    metadata,
  }).catch((err) => {
    console.warn("Failed to log performance metric:", err);
  });
}

/**
 * Map operation names to TaskType for telemetry
 */
export function mapOperationToTaskType(operation: string): TaskType {
  const mapping: Record<string, TaskType> = {
    auto_contrast: "auto_contrast",
    gaussian_blur: "gaussian_blur",
    resize: "resize",
    rembg: "rembg",
    inference: "inference",
    evaluation: "evaluation",
  };

  return mapping[operation] || "auto_contrast"; // Default fallback
}

/**
 * Performance tracker for measuring task duration
 */
export class PerformanceTracker {
  private startTime: number;
  private taskType: TaskType;
  private imageSize: number | null;

  constructor(taskType: TaskType, imageSize: number | null = null) {
    this.startTime = performance.now();
    this.taskType = taskType;
    this.imageSize = imageSize;
  }

  /**
   * End tracking and log performance metric
   */
  end(success: boolean = true, metadata?: Record<string, unknown>): void {
    const durationMs = performance.now() - this.startTime;
    logTaskPerformance(this.taskType, durationMs, this.imageSize, success, metadata);
  }

  /**
   * Get current duration without logging
   */
  getCurrentDuration(): number {
    return performance.now() - this.startTime;
  }
}

/**
 * Expose queue depth to window for E2E testing
 */
export function exposeQueueDepth(depth: number): void {
  if (typeof window !== "undefined") {
    (window as any).__workerPoolQueueDepth__ = depth;
  }
}
