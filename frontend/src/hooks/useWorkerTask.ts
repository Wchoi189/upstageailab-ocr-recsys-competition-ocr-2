import { useState, useEffect, useRef, useCallback } from "react";
import type { WorkerTask, WorkerResult } from "../../workers/types";
import {
  getWorkerPool,
  CancellationToken,
  type TaskPriority,
} from "../workers/workerHost";

/**
 * Hook state for worker task execution
 */
export interface WorkerTaskState<TPayload = Record<string, unknown>> {
  result: WorkerResult<TPayload> | null;
  loading: boolean;
  error: string | null;
}

/**
 * Options for worker task execution
 */
export interface UseWorkerTaskOptions {
  priority?: TaskPriority;
  debounceMs?: number;
  enabled?: boolean;
}

/**
 * React hook for executing tasks on web worker pool
 *
 * Handles task submission, cancellation, and debouncing for slider spam
 *
 * @param task - Worker task to execute
 * @param options - Execution options (priority, debouncing, enabled flag)
 * @returns Worker task state with result, loading, and error
 */
export function useWorkerTask<TPayload = Record<string, unknown>>(
  task: WorkerTask | null,
  options: UseWorkerTaskOptions = {},
): WorkerTaskState<TPayload> {
  const {
    priority = "normal",
    debounceMs = 0,
    enabled = true,
  } = options;

  const [state, setState] = useState<WorkerTaskState<TPayload>>({
    result: null,
    loading: false,
    error: null,
  });

  const debounceTimerRef = useRef<number | null>(null);
  const cancellationTokenRef = useRef<CancellationToken | null>(null);
  const currentTaskIdRef = useRef<string | null>(null);

  const executeTask = useCallback(
    async (taskToExecute: WorkerTask) => {
      // Cancel previous task if exists
      if (cancellationTokenRef.current) {
        cancellationTokenRef.current.cancel();
      }

      const token = new CancellationToken();
      cancellationTokenRef.current = token;
      currentTaskIdRef.current = taskToExecute.taskId;

      setState((prev) => ({ ...prev, loading: true, error: null }));

      try {
        const pool = getWorkerPool();
        const result = await pool.submitTask(taskToExecute, priority, token);

        // Only update state if this is still the current task
        if (currentTaskIdRef.current === taskToExecute.taskId) {
          setState({
            result: result as WorkerResult<TPayload>,
            loading: false,
            error: result.status === "error" ? result.error || null : null,
          });
        }
      } catch (error) {
        // Only update state if this is still the current task and not cancelled
        if (
          currentTaskIdRef.current === taskToExecute.taskId &&
          !token.cancelled
        ) {
          setState({
            result: null,
            loading: false,
            error:
              error instanceof Error ? error.message : "Unknown error",
          });
        }
      }
    },
    [priority],
  );

  useEffect(() => {
    if (!task || !enabled) {
      setState({ result: null, loading: false, error: null });
      return;
    }

    // Clear existing debounce timer
    if (debounceTimerRef.current !== null) {
      window.clearTimeout(debounceTimerRef.current);
    }

    // Debounce task execution (useful for slider spam)
    if (debounceMs > 0) {
      debounceTimerRef.current = window.setTimeout(() => {
        executeTask(task);
      }, debounceMs);
    } else {
      executeTask(task);
    }

    // Cleanup on unmount or task change
    return () => {
      if (debounceTimerRef.current !== null) {
        window.clearTimeout(debounceTimerRef.current);
      }
      if (cancellationTokenRef.current) {
        cancellationTokenRef.current.cancel();
      }
    };
  }, [task, enabled, debounceMs, executeTask]);

  return state;
}
