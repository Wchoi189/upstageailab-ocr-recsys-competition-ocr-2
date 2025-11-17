---
title: "UI Feature Parity Matrix"
status: "draft"
---

# UI Feature Parity Matrix

This document captures the current state of the Streamlit-based OCR tooling (Command Builder, Evaluation Viewer, Inference) and the partially implemented Unified OCR App (Preprocessing, Inference, Comparison). It is the canonical reference for tracking which capabilities must be supported as we migrate toward the high-performance playground architecture inspired by Albumentations and the Upstage Playground console.

## Legend

- **Status**
  - `✅` already implemented in the target surface
  - `⚠️` partially implemented / blocked
  - `❌` missing entirely
- **Dependencies** call out key Python modules or datasets that must be abstracted into backend services.

## Command Builder vs. Future Surfaces

| Legacy Feature | Streamlit Location | Unified/SPA Target | Status | Notes & Dependencies |
| --- | --- | --- | --- | --- |
| Metadata-aware training command builder (schema-driven forms) | `ui/apps/command_builder/components/training.py` + `schemas/command_builder_train.yaml` | SPA “Command Console” module (Track B) | ❌ | Requires schema loader service + command rendering API; depends on `ui.utils.command.CommandBuilder` and `ui.utils.ui_generator`. |
| Validation + suggestions (Use case recommendations, overrides, validation errors) | `components/training.py`, `components/suggestions.py` | SPA sidebar recommendation panel | ❌ | Must surface `UseCaseRecommendationService` via backend endpoint so frontend stays stateless. |
| Execution panel (run command, copy, download) | `components/execution.py` | SPA command drawer + log console | ❌ | Needs API wrapper over CLI runner; align with Albumentations-style console. |
| Test command builder | `components/test.py` | SPA command console (test tab) | ❌ | Shares schema infra; blocked until service layer exposes schema metadata. |
| Predict command builder | `components/predict.py` | SPA inference command builder | ❌ | Same dependencies as above; future UI should reuse inference preview modules. |

## Evaluation Viewer vs. Unified App “Comparison”

| Legacy Feature | Streamlit Location | Unified/SPA Target | Status | Notes & Dependencies |
| --- | --- | --- | --- | --- |
| Single run analysis (load submission, metrics, visualization) | `ui/evaluation/single_run.py` | Unified App `3_📊_Comparison.py` → `render_metrics_display` | ⚠️ | Base scaffolding exists but lacks file-system-backed loaders + charts parity. Needs dataset services + chart components. |
| Model comparison (dual CSV upload, stats, diff, visual gallery) | `ui/evaluation/comparison.py` | Future SPA “Comparison Studio” route | ❌ | Need multi-file upload component + before/after canvas similar to Upstage Document OCR preview. |
| Image gallery for qualitative review | `ui/evaluation/gallery.py` | SPA Gallery subview | ❌ | Requires responsive masonry grid + worker-powered lazy loading. |
| Ground-truth overlays | `display_visual_comparison` in `ui/visualization` | SPA overlay layer | ❌ | Dependent on layout recognition TODO plus worker-friendly polygon renderer. |

## Inference App vs. Unified App “Inference”

| Legacy Feature | Streamlit Location | Unified/SPA Target | Status | Notes & Dependencies |
| --- | --- | --- | --- | --- |
| Checkpoint catalog + metadata | `ui/apps/inference/services/checkpoint.py` | Unified App `2_🤖_Inference.py` (selector) → SPA checkpoint picker | ⚠️ | Basic selector exists but still Streamlit dependent; needs API to expose catalog + search. |
| Single image inference with visualization | `ui/apps/inference/components/results.py` etc. | Unified App inference page + SPA preview canvas | ⚠️ | Flow exists but lacks new design + workerized overlay layers. |
| Batch inference (directory scanning, output writing) | `ui/apps/inference/app.py` (batch branch) | SPA batch job wizard | ❌ | Should become asynchronous job submitted via backend queue. |
| Hyperparameter adjustments | `components/sidebar.py` | SPA control tray | ⚠️ | Need schema-driven controls shared with command builder. |

## Unified App Gaps

| Unified Page | Current Capabilities | Missing for Parity | Blockers |
| --- | --- | --- | --- |
| `1_🎨_Preprocessing` | Image upload, parameter panel, pipeline viewer, presets | Background removal (`rembg`), hybrid worker/backend execution, real-time before/after diff, caching across sessions | Need worker pipeline + backend API; also need dataset sampling hooks. |
| `2_🤖_Inference` | Single/batch modes, checkpoint selector, results viewer | Integration with preprocessing results, layout recognition overlays, job queue for batch, streaming results | Depends on service refactor + worker contracts. |
| `3_📊_Comparison` | Parameter sweep UI, result tabs, export controls placeholder | Real metrics ingestion, dataset-driven evaluation, Upstage-style document preview, text/layout recognition TODO | Requires evaluation services and new visualization components. |

## Shared Logic Extraction Targets

- `ui.utils.command`, `ui.utils.config_parser`, `ui.apps.command_builder.services.*`: expose as FastAPI endpoints for command synthesis and schema metadata.
- `ui/apps/inference/services/*`: wrap inference + checkpoint discovery in API surfaces.
- `ui/visualization/*`: convert to reusable service outputs (GeoJSON/polygons) for new frontend canvases.

## Next Steps

1. Finalize service contracts (Track A item 2) so frontends can operate without Streamlit session state.
2. Feed this matrix into the ADR and design system work (Track B), ensuring every missing capability has a UI placeholder or redesign note.

