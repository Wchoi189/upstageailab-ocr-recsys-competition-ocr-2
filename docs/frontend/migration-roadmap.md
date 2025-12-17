---
type: roadmap
component: playground_migration
status: current
version: "1.0"
last_updated: "2025-12-15"
---

# Playground Migration Roadmap

**Purpose**: Phased Streamlit→SPA migration; 4-phase rollout with KPIs, stop conditions, and sunset criteria.

---

## Phase 0 – Dual Runway

| Deliverable | Status | Exit Criteria |
|------------|--------|---------------|
| FastAPI service stubs | ✅ | `/api/commands/schemas`, `/api/pipelines/preview` reachable |
| Parity matrix | ✅ | [parity.md](parity.md) published |
| ADR | ✅ | Architecture decision record |
| Worker blueprint | ✅ | [worker-blueprint.md](worker-blueprint.md) |
| Feature flag | Pending | `PLAYGROUND_BETA_URL` banner in Command Builder |

**Actions**:
- Surface "Try the new playground" banner in Streamlit Command Builder
- Gate SPA access behind `PLAYGROUND_BETA_URL` feature flag

---

## Phase 1 – Command Builder Parity

**Scope**:
- Port training/test/predict forms to SPA command console
- Mirror validation + recommendation logic via `/api/commands/build`
- Integrate command diff + execution drawer

**KPIs**:

| KPI | Target | Measurement |
|-----|--------|-------------|
| **Generated CLI parity** | ≥99% vs Streamlit | Diff script comparison |
| **Schema update latency** | <150ms | Slider changes → API response |

**Stop Conditions**:
- Command mismatches detected → pause rollout
- CLI execution fails → investigate and fix

---

## Phase 2 – Preprocessing & Inference Studios

**Scope**:
- Wire worker pools + rembg hybrid routing for preprocessing
- Port Inference single/batch mode with checkpoint picker + canvas
- Introduce Upstage-style layout panel placeholders (layout recognition TODO)

**KPIs**:

| KPI | Target | Measurement |
|-----|--------|-------------|
| **Preview latency (contrast)** | <100ms | Playwright test |
| **Client rembg** | <400ms | Playwright test |
| **Backend fallback** | <800ms | Playwright test |
| **Worker queue depth** | <5 tasks | Slider spam test |

**Stop Conditions**:
- Worker instability → escalate to backend-only mode
- rembg memory issues → suspend rembg client path

---

## Phase 3 – Comparison & Evaluation

**Scope**:
- Replace Streamlit Evaluation Viewer with SPA comparison suite via `/api/evaluation/*`
- Add document gallery (Upstage-inspired)
- Introduce layout/text recognition modules (future models)

**KPIs**:

| KPI | Target | Measurement |
|-----|--------|-------------|
| **Configuration comparison** | ≥3 configs | Streaming metrics support |
| **Gallery load time** | <2s | 20 thumbnails |

**Stop Conditions**:
- KPI regressions → hold Streamlit sunset
- Missing dataset access → delay migration

---

## Phase 4 – Streamlit Sunset

**Criteria**:
- 4 weeks of steady SPA usage with no blocking bugs
- Telemetry coverage for worker + backend pipelines
- Docs updated (README, `run_spa.py` instructions)

**Actions**:

| Action | Implementation |
|--------|----------------|
| Replace Streamlit commands | Link to SPA routes |
| Archive Streamlit apps | Move to `docs/archive` |
| Update docs | README, quickstart guides |

---

## Cross-Phase: Testing & Observability

| Activity | Implementation | Frequency |
|----------|----------------|-----------|
| **Playwright suite** | [testing-observability.md](testing-observability.md) | Every CI build |
| **Worker telemetry** | Ship to `/api/metrics` (future) | Continuous |
| **Benchmarks** | `tests/perf/pipeline_bench.py` | Every CI build |

---

## Dependencies

| Phase | Dependencies |
|-------|-------------|
| **Phase 1** | Schema loader API, command synthesis service |
| **Phase 2** | Worker pipeline, rembg routing, checkpoint catalog API |
| **Phase 3** | Evaluation services, gallery components, layout recognition (future) |
| **Phase 4** | Full feature parity, telemetry, 4-week stability |

---

## Constraints

- Feature flag (`PLAYGROUND_BETA_URL`) controls access
- Stop conditions must be enforced before advancing phases
- Streamlit apps remain available until Phase 4 complete

---

## Backward Compatibility

**Status**: Transition period (Phases 0-3)

**Legacy Support**: Streamlit apps available during migration

**Sunset**: Phase 4 (after 4 weeks stable SPA usage)

---

## References

- [Parity Matrix](parity.md)
- [Testing Observability](testing-observability.md)
- [Worker Blueprint](worker-blueprint.md)
- [High Performance Playground](high-performance-playground.md)
- [Design System](design-system.md)
