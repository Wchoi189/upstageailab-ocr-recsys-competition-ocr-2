/// <reference lib="webworker" />

import type { TransformContext, TransformHandler, WorkerResult, WorkerTask } from "./types";
import { runAutoContrast, runGaussianBlur, runRembgLite, runResize } from "./transforms";

declare const self: DedicatedWorkerGlobalScope;

const ctx: TransformContext = (() => {
  const canvas = new OffscreenCanvas(1, 1);
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) {
    throw new Error("Unable to initialize worker canvas context");
  }
  return { canvas, ctx: context };
})();

const registry: Record<string, TransformHandler> = {
  autocontrast: runAutoContrast,
  "gaussian-blur": runGaussianBlur,
  resize: runResize,
  "rembg-lite": runRembgLite,
};

async function handleTask(task: WorkerTask): Promise<WorkerResult> {
  const start = performance.now();
  const handler = registry[task.type];
  if (!handler) {
    return {
      taskId: task.taskId,
      status: "error",
      durationMs: 0,
      error: `Unknown task type: ${task.type}`,
    };
  }

  try {
    const bitmap = await handler(task, ctx);
    const durationMs = performance.now() - start;
    return {
      taskId: task.taskId,
      status: "success",
      payload: { bitmap },
      durationMs,
      routedBackend: "client",
    };
  } catch (error) {
    const durationMs = performance.now() - start;
    const errorMessage = error instanceof Error
      ? error.message || error.toString() || "Unknown error"
      : String(error) || "Unknown error";

    // Log error details in worker console
    console.error(`[Worker] Task ${task.taskId} failed:`, errorMessage);
    if (error instanceof Error && error.stack) {
      console.error(`[Worker] Stack trace:`, error.stack);
    }

    return {
      taskId: task.taskId,
      status: "error",
      error: errorMessage,
      durationMs,
    };
  }
}

// Global error handler for uncaught errors in worker
self.onerror = (error: ErrorEvent): void => {
  console.error("[Worker] Uncaught error:", {
    message: error.message,
    filename: error.filename,
    lineno: error.lineno,
    colno: error.colno,
    error: error.error,
  });

  // Try to send error back to main thread
  try {
    self.postMessage({
      taskId: "unknown",
      status: "error",
      error: `Uncaught worker error: ${error.message || "Unknown error"} (${error.filename}:${error.lineno})`,
      durationMs: 0,
    });
  } catch (e) {
    // If postMessage fails, we can't do much
    console.error("[Worker] Failed to send error to main thread:", e);
  }
};

self.onmessage = async (event: MessageEvent<WorkerTask>) => {
  try {
    const result = await handleTask(event.data);
    self.postMessage(result, result.payload?.bitmap ? [result.payload.bitmap] : []);
  } catch (error) {
    // Fallback error handler (shouldn't be needed due to try-catch in handleTask)
    console.error("[Worker] Unhandled error in message handler:", error);
    const errorMessage = error instanceof Error
      ? error.message || error.toString() || "Unknown error"
      : String(error) || "Unknown error";

    self.postMessage({
      taskId: event.data?.taskId || "unknown",
      status: "error",
      error: `Unhandled error: ${errorMessage}`,
      durationMs: 0,
    });
  }
};


