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
    return {
      taskId: task.taskId,
      status: "error",
      error: error instanceof Error ? error.message : String(error),
      durationMs,
    };
  }
}

self.onmessage = async (event: MessageEvent<WorkerTask>) => {
  const result = await handleTask(event.data);
  self.postMessage(result, result.payload?.bitmap ? [result.payload.bitmap] : []);
};


