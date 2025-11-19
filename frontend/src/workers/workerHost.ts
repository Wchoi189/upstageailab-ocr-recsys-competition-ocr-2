import type { WorkerTask, WorkerResult } from "../../workers/types";
import {
  generateTaskId,
  generateWorkerId,
  logWorkerLifecycleEvent,
  exposeQueueDepth,
} from "./workerTelemetry";

/**
 * Priority level for task scheduling
 */
export type TaskPriority = "high" | "normal" | "low";

/**
 * Queued task with priority and cancellation support
 */
interface QueuedTask {
  task: WorkerTask;
  priority: TaskPriority;
  resolve: (result: WorkerResult) => void;
  reject: (error: Error) => void;
  cancelled: boolean;
  taskId: string; // Unique ID for telemetry
  workerId: string; // Worker ID for telemetry
}

/**
 * Cancellation token for aborting queued tasks
 */
export class CancellationToken {
  private _cancelled = false;
  private _callbacks: (() => void)[] = [];

  get cancelled(): boolean {
    return this._cancelled;
  }

  cancel(): void {
    if (this._cancelled) return;
    this._cancelled = true;
    this._callbacks.forEach((cb) => cb());
    this._callbacks = [];
  }

  onCancel(callback: () => void): void {
    if (this._cancelled) {
      callback();
    } else {
      this._callbacks.push(callback);
    }
  }
}

/**
 * Worker pool manager with dynamic sizing and priority scheduling
 */
export class WorkerPool {
  private workers: Worker[] = [];
  private availableWorkers: Worker[] = [];
  private taskQueue: QueuedTask[] = [];
  private poolSize: number;
  private maxPoolSize: number;
  private workerPath: string;

  constructor(workerPath: string, initialSize = 2, maxSize = 4) {
    this.workerPath = workerPath;
    this.poolSize = initialSize;
    this.maxPoolSize = maxSize;
    this._initializePool();
  }

  /**
   * Initialize worker pool
   */
  private _initializePool(): void {
    for (let i = 0; i < this.poolSize; i++) {
      this._createWorker();
    }
  }

  /**
   * Create a new worker instance
   */
  private _createWorker(): Worker {
    const worker = new Worker(this.workerPath, { type: "module" });
    this.workers.push(worker);
    this.availableWorkers.push(worker);
    return worker;
  }

  /**
   * Get available worker or create new one if under max pool size
   */
  private _getWorker(): Worker | null {
    if (this.availableWorkers.length > 0) {
      return this.availableWorkers.pop()!;
    }

    // Dynamically grow pool if under max size
    if (this.workers.length < this.maxPoolSize) {
      return this._createWorker();
    }

    return null;
  }

  /**
   * Return worker to available pool
   */
  private _releaseWorker(worker: Worker): void {
    if (!this.availableWorkers.includes(worker)) {
      this.availableWorkers.push(worker);
    }
    this._processQueue();
  }

  /**
   * Process queued tasks in priority order
   */
  private _processQueue(): void {
    if (this.taskQueue.length === 0) return;

    const worker = this._getWorker();
    if (!worker) return;

    // Sort by priority (high > normal > low)
    this.taskQueue.sort((a, b) => {
      const priorityOrder = { high: 0, normal: 1, low: 2 };
      return priorityOrder[a.priority] - priorityOrder[b.priority];
    });

    // Get highest priority task that hasn't been cancelled
    let queuedTask: QueuedTask | undefined;
    while (this.taskQueue.length > 0) {
      const candidate = this.taskQueue.shift()!;
      if (!candidate.cancelled) {
        queuedTask = candidate;
        break;
      }
    }

    if (!queuedTask) {
      this._releaseWorker(worker);
      return;
    }

    this._executeTask(worker, queuedTask);
  }

  /**
   * Execute task on worker
   */
  private _executeTask(worker: Worker, queuedTask: QueuedTask): void {
    const { task, resolve, reject, taskId, workerId } = queuedTask;

    // Log task started
    logWorkerLifecycleEvent("task_started", workerId, taskId, {
      operation: task.type,
      priority: queuedTask.priority,
    });

    const onMessage = (event: MessageEvent<WorkerResult>): void => {
      cleanup();
      // Log task completed
      logWorkerLifecycleEvent("task_completed", workerId, taskId, {
        operation: task.type,
      });
      this._releaseWorker(worker);
      resolve(event.data);
    };

    const onError = (error: ErrorEvent): void => {
      cleanup();
      // Log task failed
      logWorkerLifecycleEvent("task_failed", workerId, taskId, {
        operation: task.type,
        error: error.message,
      });
      this._releaseWorker(worker);
      reject(new Error(`Worker error: ${error.message}`));
    };

    const cleanup = (): void => {
      worker.removeEventListener("message", onMessage);
      worker.removeEventListener("error", onError);
    };

    worker.addEventListener("message", onMessage);
    worker.addEventListener("error", onError);
    worker.postMessage(task, [task.imageBuffer]);
  }

  /**
   * Submit task to worker pool with priority
   */
  submitTask(
    task: WorkerTask,
    priority: TaskPriority = "normal",
    cancellationToken?: CancellationToken,
  ): Promise<WorkerResult> {
    return new Promise<WorkerResult>((resolve, reject) => {
      // Generate unique IDs for telemetry
      const taskId = generateTaskId();
      const workerId = generateWorkerId();

      const queuedTask: QueuedTask = {
        task,
        priority,
        resolve,
        reject,
        cancelled: false,
        taskId,
        workerId,
      };

      // Log task queued
      logWorkerLifecycleEvent("task_queued", workerId, taskId, {
        operation: task.type,
        priority,
      });

      // Wire cancellation token
      if (cancellationToken) {
        cancellationToken.onCancel(() => {
          queuedTask.cancelled = true;
          // Log task cancelled
          logWorkerLifecycleEvent("task_cancelled", workerId, taskId, {
            operation: task.type,
          });
          reject(new Error("Task cancelled"));
        });
      }

      this.taskQueue.push(queuedTask);
      // Expose queue depth for E2E testing
      exposeQueueDepth(this.taskQueue.length);
      this._processQueue();
    });
  }

  /**
   * Get current queue depth
   */
  getQueueDepth(): number {
    return this.taskQueue.length;
  }

  /**
   * Cleanup: terminate all workers
   */
  terminate(): void {
    this.workers.forEach((worker) => worker.terminate());
    this.workers = [];
    this.availableWorkers = [];
    this.taskQueue = [];
  }
}

/**
 * Singleton worker pool instance
 */
let globalWorkerPool: WorkerPool | null = null;

/**
 * Get or create global worker pool instance
 */
export function getWorkerPool(): WorkerPool {
  if (!globalWorkerPool) {
    globalWorkerPool = new WorkerPool(
      new URL("../../workers/pipelineWorker.ts", import.meta.url).href,
      2, // initial size
      4, // max size
    );
  }
  return globalWorkerPool;
}

/**
 * Reset worker pool (useful for testing)
 */
export function resetWorkerPool(): void {
  if (globalWorkerPool) {
    globalWorkerPool.terminate();
    globalWorkerPool = null;
  }
}
