---
type: architecture
component: worker_pipeline
status: current
version: "1.0"
last_updated: "2025-12-15"
---

# Worker Utilization Blueprint

**Purpose**: Web worker architecture for <100ms slider feedback, <400ms client rembg; automatic backend routing on device saturation.

---

## Goals

| Goal | Target | Implementation |
|------|--------|----------------|
| **Slider feedback** | <100ms | Lightweight transforms (contrast, blur, crop) |
| **Client rembg** | <400ms | ONNX.js model (~3MB) with WASM SIMD |
| **Backend routing** | Automatic | >8MP images or throttled CPUs |

---

## Task Types

| Task Type | Execution | Priority | Notes |
|-----------|-----------|----------|-------|
| `preview::transform` | Synchronous canvas updates | High | Contrast, blur, crop, rembg-lite |
| `preview::layout` | Layout/text recognition | Medium | Future TODO |
| `job::batch` | Long-running preprocessing/inference | Low | Forwarded to FastAPI + background queue |

---

## Pool Manager

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Pool size** | `min(available_cores - 1, 6)` | Default |
| **Dynamic scaling** | Lazy spin-up | Workers terminate after 60s idle |
| **Queue** | `PriorityQueue` | User interactions priority=1, background=5 |

---

## RPC Interface

**Message Envelope**:
```typescript
{
  taskId: string,
  type: TaskType,
  payload: any,
  traceId: string
}
```

**Features**:
- Comlink wrappers for RPC
- Cancellation tokens: newer task with same `controlId` cancels previous

---

## rembg Routing Strategy

| Condition | Route | Implementation |
|-----------|-------|----------------|
| **Image ≤ 2048px** | Client | ONNX.js model in dedicated worker |
| **Latency < 400ms** | Client | Continue client-side processing |
| **Image > 2048px** | Backend | `/api/inference/preview` with `routed_backend="server-rembg"` |
| **Latency > 400ms** | Backend | Fallback to server |

---

## Telemetry

**Metrics Captured**:
- Queue depth per task type
- Average duration per transform
- Client vs backend route counts
- rembg cache hit rate `(imageHash, paramsHash)`

**Destination**: `postMessage` to main thread → `/api/metrics` (future)

**UI Surface**: `WorkerStatusList` component

---

## Failure Modes

| Failure | Recovery | UI Feedback |
|---------|----------|-------------|
| **Worker crash** | Auto respawn | Toast message referencing affected control |
| **SharedArrayBuffer restrictions** | Fallback to single-threaded | Flag surfaces in telemetry |
| **rembg memory pressure** | Release ONNX session after 30s idle | None (automatic) |

---

## Implementation Steps

| Step | Component | Purpose |
|------|-----------|---------|
| 1 | `WorkerTask`, `WorkerResult`, `WorkerError` | TypeScript contracts |
| 2 | `packages/workers/pipelineWorker.ts` | Transform registry + cancellation |
| 3 | `workerHost.ts` | Pool management (create, recycle, monitor) |
| 4 | `useWorkerTask` hook | Connect UI controls to host |
| 5 | Backend fallback integration | Call `/api/inference/preview` on heuristics |
| 6 | Telemetry | Emit events, bind to HUD |

---

## Dependencies

| Component | Dependencies |
|-----------|-------------|
| **ONNX.js** | WASM SIMD support |
| **Comlink** | RPC wrappers |
| **Backend API** | `/api/inference/preview`, `/api/metrics` |

---

## Constraints

- Pool size limited to `min(cores - 1, 6)`
- rembg model: ~3MB download
- Worker idle timeout: 60s
- Backend fallback threshold: 400ms latency or >2048px image

---

## Stop Conditions

| Condition | Action |
|-----------|--------|
| Consistent >400ms latency | Escalate to GPU-backed rendering |
| ONNX bundle destabilizes worker memory | Suspend rembg client path |

---

## Backward Compatibility

**Status**: New system (v1.0)

**Breaking Changes**: N/A (new implementation)

---

## References

- [Design System](design-system.md)
- [Testing Observability](testing-observability.md)
- [Backend Pipeline Contract](../backend/api/backend-pipeline-contract.md)
