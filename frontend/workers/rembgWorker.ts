/// <reference lib="webworker" />

import type { WorkerResult, WorkerTask } from "./types";
import { runRembgLite } from "./transforms";

declare const self: DedicatedWorkerGlobalScope;

const canvas = new OffscreenCanvas(1, 1);
const ctx = canvas.getContext("2d", { willReadFrequently: true });
if (!ctx) {
  throw new Error("Failed to initialize rembg worker context");
}

self.onmessage = async (event: MessageEvent<WorkerTask>) => {
  const start = performance.now();
  try {
    const bitmap = await runRembgLite(event.data, { canvas, ctx });
    const response: WorkerResult = {
      taskId: event.data.taskId,
      status: "success",
      payload: { bitmap },
      durationMs: performance.now() - start,
      routedBackend: "client",
      notes: ["rembg-lite placeholder"],
    };
    self.postMessage(response, [bitmap]);
  } catch (error) {
    const response: WorkerResult = {
      taskId: event.data.taskId,
      status: "error",
      durationMs: performance.now() - start,
      error: error instanceof Error ? error.message : String(error),
      routedBackend: "client",
    };
    self.postMessage(response);
  }
};


