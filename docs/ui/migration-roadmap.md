---
title: "Playground Migration Roadmap"
status: "draft"
---

# Playground Migration Roadmap

## Phase 0 – Dual Runway

- Deliverables: FastAPI service stubs, parity matrix, ADR, worker blueprint (✅).
- Actions:
  - Surface “Try the new playground” banner in Command Builder + Unified App (links to SPA dev server once available).
  - Gate SPA access behind feature flag `PLAYGROUND_BETA_URL`.
- Exit criteria: SPA skeleton renders, command builder API reachable.

## Phase 1 – Command Builder Parity

- Scope:
  - Port training/test/predict forms into SPA command console.
  - Mirror validation + recommendation logic via `/api/commands/build`.
  - Integrate command diff + execution drawer.
- KPIs:
  - Generated CLI parity ≥ 99% vs. Streamlit baseline (diff script).
  - <150 ms latency for schema updates after slider changes.
- Stop conditions:
  - Command mismatches detected or CLI execution fails -> pause rollout.

## Phase 2 – Preprocessing & Inference Studios

- Scope:
  - Wire worker pools + rembg hybrid routing for preprocessing page.
  - Port Inference single/batch mode with checkpoint picker + results canvas.
  - Introduce Upstage-style layout panel placeholders (layout recognition TODO).
- KPIs:
  - Preview latency: <100 ms (contrast) / <400 ms (client rembg) / <800 ms (backend fallback).
  - Worker queue depth < 5 during slider spam test (Playwright).
- Stop conditions:
  - Worker instability or rembg memory issues -> escalate to backend-only mode.

## Phase 3 – Comparison & Evaluation

- Scope:
  - Replace Streamlit Evaluation Viewer with SPA comparison suite leveraging `/api/evaluation/*`.
  - Add document gallery inspired by Upstage Document OCR console.
  - Introduce layout/text recognition modules (TODO) leveraging future models.
- KPIs:
  - Ability to compare ≥ 3 configurations with streaming metrics.
  - Gallery load < 2 s for 20 thumbnails.
- Stop conditions:
  - KPI regressions or missing dataset access -> hold Streamlit sunset.

## Phase 4 – Streamlit Sunset

- Criteria:
  - 4 weeks of steady SPA usage with no blocking bugs.
  - Telemetry coverage for worker + backend pipelines.
  - Docs updated (README, run_spa instructions).
- Actions:
  - Replace Streamlit commands with links to SPA.
  - Archive Streamlit apps under `docs/archive`.

## Testing & Observability (Cross-Phase)

- Playwright suite (see `docs/ui/testing-observability.md`) runs on every CI build.
- Worker telemetry shipped to future `/api/metrics`.
- Benchmarks use `tests/perf/pipeline_bench.py` with manifest from `scripts/datasets/sample_images.py`.
