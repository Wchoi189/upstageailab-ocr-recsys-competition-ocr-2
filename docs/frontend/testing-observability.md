---
type: testing
component: playground
status: current
version: "1.0"
last_updated: "2025-12-15"
---

# Playground Testing & Observability

**Purpose**: Playwright E2E suite, performance benchmarks, telemetry for SPA playground; ensures <100ms latency and worker stability.

---

## Playwright E2E Suite

| Scenario | File | Test Coverage | Notes |
|----------|------|---------------|-------|
| **Command Builder** | `command-builder.spec.ts` | Fill training form, assert `/api/commands/build` response, diff CLI output | Mocked FastAPI responses |
| **Preprocessing** | `preprocessing.spec.ts` | Upload image, spam sliders, verify worker HUD <5 queued tasks, canvas updates <100ms | Uses `scripts/datasets/sample_images.py` manifest |
| **Inference** | `inference.spec.ts` | Select checkpoint, toggle rembg, validate `/api/pipelines/fallback` call | Mocks pipeline endpoint latency for routing |
| **Comparison** | `comparison.spec.ts` | Upload dual CSVs, verify metrics table + gallery renders | Blocks until evaluation endpoint `status=accepted` |

**Execution**:
```bash
pnpm playwright test --config tests/ui/playwright.config.ts
```

---

## Performance Benchmarks

| Benchmark | Script | Thresholds (CI Enforced) |
|-----------|--------|--------------------------|
| **Transform latency** | `tests/perf/pipeline_bench.py` | `autocontrast.mean_ms < 20` |
| **Client rembg** | `tests/perf/pipeline_bench.py` | `rembg_client.p95_ms < 400`, `mean_ms > 0` (sanity) |
| **Backend fallback** | `tests/perf/pipeline_bench.py` | `rembg_backend.p95_ms < 800` |

**Inputs**: Manifests from `scripts/datasets/sample_images.py`

---

## Telemetry

### Worker Events

**Payload** (via `window.postMessage` → `/api/metrics`):
```json
{
  "time": "2025-11-17T04:00:00Z",
  "queueDepth": 2,
  "taskType": "autocontrast",
  "latencyMs": 42.1,
  "routedBackend": "client"
}
```

### Backend Logs

| Event | Data |
|-------|------|
| **Cache hits/misses** | `cache_key` |
| **Pipeline routing** | `routed_backend` (client/server) |
| **API latency** | Request duration |

### Dashboards (Grafana/Chronograf)

| Metric | Visualization |
|--------|--------------|
| **Worker latency percentiles** | Line chart (p50, p95, p99) |
| **Backend fallback frequency** | Bar chart |
| **Command builder API latency** | Histogram |

---

## Rollout Guards

| Guard | Implementation | Purpose |
|-------|----------------|---------|
| **Feature flag** | `PLAYGROUND_BETA_URL` | Toggle CTA banners |
| **Health checks** | `/api/commands/schemas`, `/api/pipelines/preview`, `/api/inference/checkpoints` | Hide CTA if any fail |

---

## Dependencies

| Component | Dependencies |
|-----------|-------------|
| **Playwright** | Node.js 18+, test fixtures |
| **Performance bench** | Python 3.10+, sample dataset manifests |
| **Telemetry** | `/api/metrics` endpoint (future) |

---

## Constraints

- E2E tests require mocked FastAPI responses
- Performance benchmarks enforce CI thresholds
- Telemetry requires `/api/metrics` endpoint (future implementation)

---

## Backward Compatibility

**Status**: New system (v1.0)

**Breaking Changes**: N/A (new testing infrastructure)

---

## References

- [Worker Blueprint](worker-blueprint.md)
- [High Performance Playground](high-performance-playground.md)
- [Migration Roadmap](migration-roadmap.md)
