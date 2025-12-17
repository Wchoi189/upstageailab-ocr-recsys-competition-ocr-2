---
type: reference
component: feature_parity
status: current
version: "1.0"
last_updated: "2025-12-15"
---

# UI Feature Parity Matrix

**Purpose**: Track Streamlit→SPA migration; canonical reference for feature support across legacy and target surfaces.

---

## Legend

| Status | Meaning |
|--------|---------|
| ✅ | Implemented in target surface |
| ⚠️ | Partially implemented / blocked |
| ❌ | Missing entirely |

---

## Command Builder Parity

| Feature | Legacy (Streamlit) | Target (SPA) | Status | Dependencies |
|---------|-------------------|--------------|--------|--------------|
| **Training command builder** | `ui/apps/command_builder/components/training.py` | SPA Command Console | ❌ | Schema loader service, command API; `ui.utils.command.CommandBuilder` |
| **Validation + suggestions** | `components/suggestions.py` | SPA sidebar recommendations | ❌ | `UseCaseRecommendationService` backend endpoint |
| **Execution panel** | `components/execution.py` | SPA command drawer + log console | ❌ | API wrapper over CLI runner |
| **Test command builder** | `components/test.py` | SPA command console (test tab) | ❌ | Shared schema infrastructure |
| **Predict command builder** | `components/predict.py` | SPA inference command builder | ❌ | Reuse inference preview modules |

---

## Evaluation Viewer Parity

| Feature | Legacy (Streamlit) | Target (SPA) | Status | Dependencies |
|---------|-------------------|--------------|--------|--------------|
| **Single run analysis** | `ui/evaluation/single_run.py` | Unified App Comparison page | ⚠️ | File-system-backed loaders, chart components |
| **Model comparison** | `ui/evaluation/comparison.py` | SPA Comparison Studio | ❌ | Multi-file upload, before/after canvas |
| **Image gallery** | `ui/evaluation/gallery.py` | SPA Gallery subview | ❌ | Responsive masonry grid, worker lazy loading |
| **Ground-truth overlays** | `ui/visualization/display_visual_comparison` | SPA overlay layer | ❌ | Layout recognition + polygon renderer |

---

## Inference App Parity

| Feature | Legacy (Streamlit) | Target (SPA) | Status | Dependencies |
|---------|-------------------|--------------|--------|--------------|
| **Checkpoint catalog** | `ui/apps/inference/services/checkpoint.py` | Unified App Inference page / SPA picker | ⚠️ | API to expose catalog + search |
| **Single image inference** | `ui/apps/inference/components/results.py` | Unified App + SPA preview canvas | ⚠️ | New design, workerized overlay layers |
| **Batch inference** | `ui/apps/inference/app.py` (batch) | SPA batch job wizard | ❌ | Asynchronous job via backend queue |
| **Hyperparameter adjustments** | `components/sidebar.py` | SPA control tray | ⚠️ | Schema-driven controls (shared with command builder) |

---

## Unified App Gaps

| Page | Current Capabilities | Missing for Parity | Blockers |
|------|---------------------|-------------------|----------|
| **Preprocessing** | Image upload, parameter panel, pipeline viewer, presets | rembg, hybrid worker/backend, before/after diff, caching | Worker pipeline + backend API, dataset sampling |
| **Inference** | Single/batch modes, checkpoint selector, results viewer | Preprocessing integration, layout overlays, job queue, streaming | Service refactor + worker contracts |
| **Comparison** | Parameter sweep UI, result tabs, export controls | Real metrics, dataset evaluation, document preview, text/layout recognition | Evaluation services, visualization components |

---

## Shared Logic Extraction

| Legacy Module | Target API | Purpose |
|--------------|------------|---------|
| `ui.utils.command` | FastAPI endpoints | Command synthesis, schema metadata |
| `ui.utils.config_parser` | FastAPI endpoints | Config parsing |
| `ui/apps/command_builder/services/*` | FastAPI endpoints | Command builder services |
| `ui/apps/inference/services/*` | FastAPI endpoints | Inference + checkpoint discovery |
| `ui/visualization/*` | Service outputs (GeoJSON/polygons) | Reusable canvas data |

---

## Dependencies

| Component | Dependencies |
|-----------|-------------|
| **Command Builder** | Schema loader, command API, validation service |
| **Evaluation** | Metrics service, chart components, file loaders |
| **Inference** | Checkpoint catalog API, worker pipeline |

---

## Constraints

- Streamlit session state must be replaced with stateless API calls
- Workers require SharedArrayBuffer (secure context)
- Backend API contracts must stabilize before full migration

---

## Backward Compatibility

**Status**: Transition period

**Legacy Support**: Streamlit apps remain available during SPA development

**Sunset Criteria**: 4 weeks of stable SPA usage + feature parity + telemetry coverage

---

## References

- [Migration Roadmap](migration-roadmap.md)
- [Design System](design-system.md)
- [Worker Blueprint](worker-blueprint.md)
- [High Performance Playground](high-performance-playground.md)
