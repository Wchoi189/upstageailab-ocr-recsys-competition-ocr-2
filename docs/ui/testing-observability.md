---
title: "Playground Testing & Observability"
status: "draft"
---

# Testing Plan

## Playwright E2E Suite

| Scenario | Description | Notes |
| --- | --- | --- |
| `command-builder.spec.ts` | Fill training form, assert `/api/commands/build` response, diff CLI output. | Runs against SPA dev server with mock FastAPI responses. |
| `preprocessing.spec.ts` | Upload sample image, spam sliders, ensure worker HUD stays <5 queued tasks and preview canvas updates within 100 ms budget. | Uses manifest from `scripts/datasets/sample_images.py`. |
| `inference.spec.ts` | Select checkpoint, toggle rembg, validate fallback call to `/api/pipelines/fallback`. | Mocks pipeline endpoint latency to verify routing. |
| `comparison.spec.ts` | Upload dual CSVs, check metrics table + gallery renders. | Blocks until evaluation endpoint returns `status=accepted`. |

Automation hook: `pnpm playwright test --config tests/ui/playwright.config.ts` (config to be added with SPA scaffold).

## Performance Benchmarks

- Script: `tests/perf/pipeline_bench.py`
- Inputs: manifests from `scripts/datasets/sample_images.py`.
- Thresholds (CI enforced):
  - `autocontrast.mean_ms < 20`
  - `rembg_client.p95_ms < 400`
  - `rembg_client.mean_ms > 0` (sanity)

## Telemetry

- Frontend emits worker events via `window.postMessage` and persists to `/api/metrics` (future) with payload:
  ```json
  {
    "time": "2025-11-17T04:00:00Z",
    "queueDepth": 2,
    "taskType": "autocontrast",
    "latencyMs": 42.1,
    "routedBackend": "client"
  }
  ```
- FastAPI logs include cache hits/misses via `cache_key`.
- Dashboards (Grafana/Chronograf) to plot:
  - Worker latency percentiles
  - Backend fallback frequency
  - Command builder API latency

## Rollout Guards

- Feature flag `PLAYGROUND_BETA_URL` toggles CTA banners.
- Health checks:
  - `/api/commands/schemas`
  - `/api/pipelines/preview`
  - `/api/inference/checkpoints`
- If any check fails, CTA banner hides automatically (planned).
