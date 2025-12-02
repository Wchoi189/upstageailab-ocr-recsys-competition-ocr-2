---
title: "Worker Utilization Blueprint"
status: "draft"
---

# Worker Utilization Blueprint

## Goals

- Exploit the two-day unlimited web-worker allowance for spike builds.
- Guarantee <100 ms slider feedback for lightweight transforms and <400 ms for client-side rembg.
- Provide automatic routing to backend services when devices are saturated (>8 MP images, throttled CPUs).

## Architecture

1. **Task Types**
   - `preview::transform` – synchronous canvas updates (contrast, blur, crop, rembg-lite).
   - `preview::layout` – layout/text recognition (future TODO).
   - `job::batch` – long-running preprocessing/inference jobs forwarded to FastAPI + background queue.
2. **Pool Manager**
   - Default pool size = `min(available_cores - 1, 6)`.
   - Dynamic scaling: workers spin up lazily; after 60 s idle they terminate.
   - Queues implemented via `PriorityQueue` (user interactions priority=1, background tasks priority=5).
3. **RPC Interface**
   - Comlink wrappers with message envelope `{taskId, type, payload, traceId}`.
   - Cancellation tokens: if a slider emits a newer task with same `controlId`, previous task is cancelled.
4. **rembg Routing**
   - First attempt: ONNX.js model (~3 MB) running in dedicated worker with WASM SIMD.
   - If image > 2048px on longer side or worker latency > 400 ms, fall back to `/api/inference/preview` with `routed_backend="server-rembg"`.

## Telemetry

- Worker lifecycle events dispatched via `postMessage` to main thread and mirrored to `/api/metrics` (future).
- Metrics captured:
  - Queue depth per task type.
  - Average duration per transform.
  - Client vs backend route counts.
  - rembg cache hit rate `(imageHash, paramsHash)`.
- Surfaced in UI via `WorkerStatusList` component.

## Failure Modes

- **Worker crash**: auto respawn + toast message referencing affected control.
- **SharedArrayBuffer restrictions**: fallback to single-threaded path (flag surfaces in telemetry).
- **rembg memory pressure**: release ONNX session when queue idle for >30 s.

## Implementation Steps

1. Define TypeScript contract `WorkerTask`, `WorkerResult`, `WorkerError`.
2. Build `packages/workers/pipelineWorker.ts` handling transform registry + cancellation.
3. Implement `workerHost.ts` to manage pools (create, recycle, monitor).
4. Connect UI controls to host via hooks (`useWorkerTask`).
5. Integrate backend fallback by calling `/api/inference/preview` when heuristics trigger.
6. Emit telemetry events and bind to HUD.

## Stop Conditions

- Pause development if browser metrics show consistent >400 ms latency even after throttling – escalate to GPU-backed rendering.
- Suspend rembg client path if ONNX bundle destabilizes worker memory.
