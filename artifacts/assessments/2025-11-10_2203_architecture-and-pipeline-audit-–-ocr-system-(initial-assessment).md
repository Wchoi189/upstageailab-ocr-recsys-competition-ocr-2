---
title: "Architecture and Pipeline Audit – OCR System (Initial Assessment)"
author: "ai-agent"
date: "2025-11-10"
type: "assessment"
category: "architecture"
status: "in_progress"
version: "0.2"
tags: ["assessment", "architecture", "pipeline", "audit", "technical-debt", "cuda", "loss", "imports"]
timestamp: "2025-11-10 22:03 KST"
---

# Architecture and Pipeline Audit – OCR System (Initial Assessment)

## Progress Tracker
- **Status:** In Progress
- **Current Step:** Investigating CUDA illegal-instruction and NaN sources
- **Last Completed Task:** Profile import time hotspots and unused deps
- **Next Task:** Diagnose CUDA illegal instruction vectors
- **Notes:** WandB import remains heavy even when disabled; consider gating.

### Task Checklist
- [x] Repo discovery and entrypoints identified
- [x] Locate DiceLoss and DBLoss usage
- [x] Map full training data flow with config overlays
- [x] Trace inference/UI flows and shared components
- [x] Catalog deprecated/duplicate modules
- [x] Measure import time hotspots and unused deps
- [ ] Diagnose CUDA illegal instruction vectors
- [ ] Propose pruning and migration plan

## Scope
- End-to-end data flow (config → dataloaders → model → loss/metrics → logging/checkpointing → UI/inference)
- Identify deprecated/legacy modules, wrappers, duplicates, redundancies
- Analyze current breakages: CUDA illegal instructions, Dice Loss/NaN, import times, unused dependencies
- Produce remediation plan and pruning candidates

## Executive Summary (Initial)
- Training entrypoint remains `runners/train.py` with Hydra preset layering across `configs/` groups.
- Loss stack anchored by `ocr/models/loss/db_loss.py` and `ocr/models/loss/dice_loss.py`; Dice safeguards exist but remain expensive and late in the pipeline.
- CUDA handling injected via env flags in `runners/train.py`; additional watchdog scripts exist yet core runtime still allows invalid tensors to reach kernels.
- UI inference stack duplicates logic between `ui/utils/inference` and `ui/apps/unified_ocr_app/services`, increasing maintenance drag.
- Dependency footprint (docs, stubs, UI extras) widens import graph and slows cold start for training and scripts.

## Training Pipeline Map (Config → Loss)

### Config Composition Layer
- `configs/train.yaml` composes `_self_`, `base`, data, transforms, dataloaders, and preset model/lightning modules; `base.yaml` wires model/dataset registries and default groups (`trainer`, `callbacks`, `logger`).
- `configs/preset/lightning_modules/base.yaml` pins Lightning entrypoints: `${lightning_path}.OCRPLModule` and `${lightning_path}.OCRDataPLModule`.
- Model defaults resolve through `configs/model/default.yaml`, which instantiates `ocr.models.architecture.OCRModel` and cascades into encoder/decoder/head configs from architecture presets.
- **Inefficiencies:** automatic GPU scaling in `runners/train.py` mutates `config.trainer.devices` on the fly; with small datasets this eagerly jumps to multi-GPU DDP and multiplies import/launch overhead without ensuring dataset safety.

### Dataset Assembly & Validation (`ocr/datasets`)
- `ocr.datasets.get_datasets_by_cfg` instantiates four `ValidatedOCRDataset` instances (train/val/test/predict) via Hydra and applies optional sample limits from `data.*` controls.
- `ValidatedOCRDataset.__getitem__` performs a 10-step pipeline (cache lookup → EXIF normalization → polygon validation/filtering → transform → shapely-based repair → map loading → Pydantic validation) and can return `None` when all polygons are dropped.
- **Inefficiencies:** heavy Pydantic validation plus Shapely fixes run for every sample, doubling CPU time; returning `None` shrinks effective batch size and creates dynamic workloads; cache manager logging occurs per epoch even when caching disabled.

### Collation & Map Generation (`ocr/datasets/db_collate_fn.py`)
- `DBCollateFN.__call__` filters `None` samples, re-validates polygons, and either loads cached maps or regenerates them per image; inference mode toggles skip map generation but training path still performs expensive CPU work.
- **Inefficiencies:** polygon filtering duplicates dataset checks; fallback zero-maps keep invalid samples in batch, masking upstream issues and producing gradients on empty targets; per-call logging synchronizes stdout and can stall workers.

### Lightning Modules & Training Loop
- `ocr.lightning_modules.get_pl_modules_by_cfg` builds the model/data modules; `OCRPLModule.training_step` re-validates tensors for NaN/Inf and logs losses, while `OCRModel.forward` performs additional CPU detaches to inspect encoder outputs.
- `configure_optimizers` delegates to `OCRModel.get_optimizers` (which instantiates optimizer/scheduler from config) and stores scheduler for manual stepping in `on_train_epoch_end`.
- **Inefficiencies:** repeated `.detach().cpu()` calls in `OCRModel.forward` force host synchronization, dramatically slowing training and possibly provoking CUDA illegal instruction retries; layered validation in `training_step` and gradient inspection duplicates work already done in loss modules.

### Loss & Metric Path
- Forward pass returns prediction dict which routes to `DBLoss` → `DiceLoss` + BCE/L1 components; metric config builds `CLEvalMetric` and `CLEvalEvaluator` for validation/test loops.
- **Inefficiencies:** `DiceLoss` clamps predictions only after detection, so upstream tensors can still saturate; loss raises `ValueError` on NaN/Inf, but detection happens post-forward, missing earlier gradient anomalies.

### Logging, Callbacks, Post-run Management
- `runners/train.py` conditionally instantiates `WandbLogger`/`TensorBoardLogger`, resolves Hydra callbacks (`configs/callbacks/*`), adds `LearningRateMonitor`, and enforces directory creation under `config.paths`.
- **Inefficiencies:** callback list includes multiple WandB-related hooks even when logger disabled; `LearningRateMonitor` always added despite potential cost in high-frequency logging; `finalize_run` scans all callback metrics converting tensors to CPU scalars, extending teardown.

## Import-Time Profiling (Python 3.10, cold start)
- Ran targeted import timing via `importlib` from a clean interpreter (`python - <<'PY' ...`) to surface slow modules. Results (single-process sequential import):
  - `lightning.pytorch`: **21.18 s**
  - `wandb`: **19.57 s** (even with `WANDB_MODE=disabled`, import still initializes network shims and pydantic models)
  - `torch`: **10.89 s**
  - `ui.apps.inference.app`: **3.83 s** (pulls in `streamlit`, config loaders, and inference services)
  - `streamlit`: **3.45 s** (logging warns about cache backend when no runtime)
  - `albumentations`: **3.31 s**, `doctr`: **3.07 s**
  - Core OCR modules (`ocr.models.architecture`, `ocr.lightning_modules.ocr_pl`) add ~1.7–1.8 s each.
- Combined cost for training entrypoint (torch + lightning + hydra + wandb) exceeds **52 s** before any user code runs. UI cold start accumulates `streamlit` + inference layers on top, explaining sluggish launch.
- `wandb` import dominates even when training config sets `logger.wandb=false`; callbacks still reference WandB classes, triggering import. Opportunity: keep wandb fully optional by gating `_get_wandb()` usage and removing default callback when disabled.
- `lightning.pytorch` loads extensive plugins; enabling `PL_DISABLE_IMPORT_ERROR=1` or trimming unused accelerators can shave seconds. Consider deferring Lightning import until after config parsing in CLI utilities.
- UI-specific modules (`streamlit`, `doctr`) should not load during CLI tooling; ensure scripts avoid importing `ui.apps.*` at module scope.
- Albumentations/Doctr are only needed in data transforms and optional preprocessing—hydrate them lazily inside transform builders to cut general startup time.

## CUDA & NaN Diagnostics
- **Step function still unstable:** `ocr/models/head/db_head.py` retains the original `1 / (1 + exp(-k(x-y)))` implementation with `k=50`, which the bug report `BUG-20251110-002` identified as the NaN trigger. The numerically stable `torch.sigmoid(k * (x - y))` fix has not landed, so overflow remains possible.
  ```158:198:ocr/models/head/db_head.py
  def _step_function(self, x, y):
      return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))
  ```
  *Action:* Replace with `torch.sigmoid` (and clamp inputs) to prevent exponential blow-ups, then add an integration test that asserts loss stays finite for random tensors.
- **Dataset quality gap:** `BUG-20251110-001` documents ~26% of training polygons exceeding image bounds. Current pipeline filters them late: dataset returns `None`, `DBCollateFN` recreates zeroed maps, and training proceeds with empty targets—driving unstable gradients that later propagate to Dice/BCE losses. Upstream fix must clean annotations or clamp coordinates before batching.
- **BCE guardrails present but symptomatic:** `ocr/models/loss/bce_loss.py` now issues device/shape checks and moves masks to CPU; however `.cpu()` still fails if CUDA memory is already corrupted, meaning earlier layers (dataset/step function) must be cleaned or the safeguard simply raises later.
- **Lightning module runtime checks:** `ocr/lightning_modules/ocr_pl.py` adds extensive NaN/Inf validation for inputs, losses, and gradients. While this catches corruption, it also calls `.detach().cpu()` on encoder outputs each step, inducing host-device sync and masking the root issue by zeroing gradients.
- **cuDNN workspace exhaustion:** `BUG-20251110-003` shows RTX 3060 runs exhaust workspace memory with batch=4, leading to `FIND was unable to find an engine` and subsequent illegal access. Recommend defaulting 12 GB presets to batch=2 with gradient accumulation, or enabling AMP (`trainer.precision=16-mixed`) in preset configs.
- **Diagnostics tooling exists but expensive:** `scripts/diagnose_cuda_issue.py` bundles dataset and model checks, yet importing it pulls heavy dependencies; integrate it into CI with a minimal preset to regularly validate transformer/dataloader behaviour.
- **Pending validation:** bug reports mark fixes as implemented but unverified. Need targeted reproducible tests (e.g., run 200 steps after step-function patch, ensure no NaNs) before closing issues.

## Legacy & Redundancy Catalogue

### Inference Stack Duplication & Drift
- `ui/apps/inference/services/inference_runner` persists the legacy request-flow, writing every upload to a tempfile and reloading checkpoints via `run_inference_on_image`; the unified OCR app re-wraps the same logic (`services/inference_service`) and performs another tempfile write before dispatch. Outcome: two Streamlit stacks share no singleton engine, so every call rebuilds CUDA state and doubles disk I/O (`_perform_inference`, `_run_inference_internal`).
- `ui/utils/inference/run_inference_on_image` instantiates a fresh `InferenceEngine` per invocation, forcing full model + config reloads even during batch requests. The engine itself lacks memoization hooks, so higher layers cannot reuse state.

### WandB Callback Proliferation
- Lightning callback registry includes `WandbImageLoggingCallback` and `WandbCompletionCallback` alongside Hydra-configured callbacks; both assume WandB linkage even when `config.logger.wandb` is false, and each performs filesystem writes (`.SUCCESS`/`.FAILURE`) plus extra dataset traversal. These callbacks overlap with Weights & Biases logging in `ocr/lightning_modules/loggers/wandb_loggers.py`, creating redundant W&B touch points.
- `configs/callbacks/default.yaml` still enables these callbacks by default, so opting out of WandB in config only disables the logger, not the callback side effects.

### Script & Tooling Backups
- `.backup/scripts-backup-20251109-222902` mirrors large portions of `scripts/`, including CUDA utilities, artifact validators, and maintenance scripts. Many filenames match active scripts (e.g., `debug_cuda.sh`, `validate_metadata.py`), increasing confusion and encouraging stale CLI usage.
- Legacy shell helpers (`scripts/debug_cuda.sh`) coexist with the newer Python diagnostic (`scripts/diagnose_cuda_issue.py`), leading to inconsistent debugging paths and duplicated environment variable handling.

### Dataset & Collate Redundancies
- Polygon sanitization occurs in `ValidatedOCRDataset`, `DBCollateFN`, and again in `WandbImageLoggingCallback`; each reimplements degenerate/out-of-bounds filtering, but inconsistently (e.g., collate uses zero-map fallbacks, callbacks drop polygons silently). A single shared utility should own these rules.
- Cache and preprocessing flags are checked in both dataset constructor and collate function, generating repeated warnings and runtime branching that contradicts the “single source of truth” comment headers in `ocr/datasets/base.py`.

### Configuration Sprawl
- Callback configs under `configs/callbacks/` enumerate eight separate YAML files, but most pipelines only require checkpointing + LR monitor. Extras (`performance_profiler`, `metadata`, `model_summary`, `rich_progress_bar`) add import-time cost and instantiate even when flagged “disabled” in practice, because the Hydra defaults include the entire group.
- Preset files in `configs/preset/models/*` duplicate architecture definitions already present in `configs/model/architectures/*`, forcing dual maintenance when updating encoder/decoder parameters.

## Pruning & Remediation Plan (Draft)
- **Stabilize Core Training Loop**
  - Patch `DBHead._step_function` to use `torch.sigmoid` with input clamping; add regression test that runs 200 synthetic steps and asserts finite loss. Remove redundant `.detach().cpu()` checks once upstream NaNs are addressed to regain GPU throughput.
  - Consolidate polygon validation into a single utility (e.g., `ocr.utils.polygon_validators`) consumed by dataset, collate, and logging. Fail fast on out-of-bounds annotations and provide script to auto-clamp or drop invalid samples (invoke `scripts/data/clean_dataset.py --remove-bad` in CI).
  - Default CUDA-safe presets for 12 GB GPUs: set `batch_size=2`, enable `trainer.accumulate_grad_batches=2`, and expose toggles for AMP. Document in `configs/dataloaders/*` and hardware presets.
- **Shrink Import Surface**
  - Move WandB callbacks behind `config.logger.wandb` guard; adjust `configs/callbacks/default.yaml` to include only essentials. Lazy-import `streamlit`, `wandb`, and docTR modules at runtime entrypoints instead of module scope.
  - Collapse inference services (`ui/apps/inference` + `ui/apps/unified_ocr_app`) into a shared engine that caches loaded checkpoints per process; eliminate tempfile duplication by streaming numpy arrays directly.
  - Audit `pyproject.toml` dependencies (stub packages, mkdocs stack) and relocate optional tooling into extras or dev groups to reduce training environment footprint.
- **Retire Redundant Assets**
  - Archive or delete `.backup/scripts-backup-*/`; migrate any still-needed utilities into `scripts/` with documentation. Enforce lint to block future backups from committing.
  - Merge duplicated preset configs: keep `configs/model/architectures/*` as source of truth, replace preset overrides with references.
  - Replace shell `debug_cuda.sh` with Python diagnosis entrypoint (wrapper script can remain but call the Python tool).
- **Operational Guardrails**
  - Integrate `scripts/diagnose_cuda_issue.py --check-dataloader` into nightly CI with reduced batch to detect regressions early.
  - Add Hydra validation for wandb off mode to ensure callbacks aren’t instantiated when disabled.
  - Capture import-time telemetry in CI (run the same timing script) to monitor regressions as dependencies change.
- **Next Steps**
  - Socialize plan with maintainers; prioritize fixes that unblock training (step function, dataset QA, wandb gating).
  - Schedule implementation sprints: (1) data & loss stabilization, (2) inference consolidation, (3) dependency pruning.
  - Update documentation (`docs/maintainers/architecture/`) once new structure is in place.

## Inference/UI Pipeline Map (Streamlit → Engine)

### Streamlit Inference App (`ui/apps/inference`)
- `ui/inference_ui.py` is a legacy shim that dispatches to `ui.apps.inference.app.run()`; configuration resolves through `configs/ui/inference.yaml`.
- `app.run()` assembles `InferenceState`, loads checkpoint catalog via `_load_catalog()` (cached), and routes requests to `InferenceService.run()` or `.run_batch_prediction()`.
- `InferenceService._perform_inference()` (single/batch) writes uploaded files to `tempfile`, optionally preprocesses with `DocumentPreprocessor`, then invokes `ui.utils.inference.run_inference_on_image()`.
- **Inefficiencies:** every call to `run_inference_on_image()` constructs a fresh `InferenceEngine`, reloads model weights from disk, and rebuilds transforms/postprocessing—~2–4s per request on GPU; temp file churn adds I/O pressure and leaves GC-sensitive files on failure; preprocessing fallback swallows exceptions and silently disables docTR.

### Unified OCR App (`ui/apps/unified_ocr_app`)
- `services/comparison_service.py` orchestrates preprocessing → inference → visualization loops for multiple configs, relying on `_get_inference_service()` to lazily import the Streamlit `InferenceService` defined above.
- `services/inference_service.InferenceService.run_inference()` is `@st.cache_data`; it wraps `_run_inference_internal()` which again drops images to disk and calls `run_inference_on_image()` with raw hyperparameters.
- `state` objects cache processed buckets, but both apps maintain independent cache registries, doubling memory footprint; unified app rebuilds inference service per Streamlit session despite caching decorator missing `ttl`/`persist` controls.
- **Inefficiencies:** duplicated state management (`InferenceState` vs unified app state) causes drift; caching annotated as `@st.cache_data` yet underlying engine rebuild nullifies benefit; repeated temp file serialization for every frame bottlenecks SSD and fails on read-only environments.

### Shared Utility Layer (`ui/utils/inference`)
- `InferenceEngine` encapsulates checkpoint loading, preprocessing, postprocessing, but lacks persistent singleton usage; `run_inference_on_image()` instantiates `InferenceEngine()` per call, loads checkpoint (`load_checkpoint`) and model (`instantiate_model`) freshly, then destroys it—huge overhead.
- Configuration helpers (`config_loader`, `model_loader`, `postprocess`) duplicate logic already present in training model factory (`ocr.models.architecture`); no shared registry ensures parity, leading to mismatched defaults (e.g., postprocess thresholds diverge from Lightning head config).
- **Inefficiencies:** `InferenceEngine.predict_image()` performs PIL→NumPy→OpenCV conversions for every call; orientation remapping occurs post-decoding, duplicating `orientation` utilities present in dataset transforms; fallback postprocess uses CPU-bound polygon decoding even when GPU results available.

### Failure Modes & Legacy Debt
- CUDA context churn: repeated checkpoint loads trigger `torch.cuda.memory_allocated()` spikes and increase risk of illegal instruction when multiple Streamlit sessions overlap.
- docTR preprocessing toggles (`validated_request.use_preprocessing`) hinge on optional dependency; when missing, UI shows toggle but silently skips processing, causing inconsistent outputs across environments.
- Duplicated `SubmissionWriter` implementations (UI vs training) diverge on output schema, complicating downstream evaluation.
- Legacy wrappers (`ui/utils/inference/utils.generate_mock_predictions`) mask engine failures with fake data, hiding regressions.

## Initial Findings

### Entrypoints and Config
- Training: `runners/train.py` (`@hydra.main`) with presets under `configs/`.
- Inference UIs: `run_ui.py`, `ui/inference_ui.py`, unified app under `ui/apps/unified_ocr_app/`.
- Config topology includes `configs/model/*`, `configs/preset/models/*`, `configs/trainer/*`, `configs/callbacks/*`, and schemas in `configs/schemas/*`.

### Model and Loss
- `ocr/models/architecture.py`, encoders/decoders/heads under `ocr/models/*`.
- `ocr/models/loss/db_loss.py` composes BCE + Dice + L1; `ocr/models/loss/dice_loss.py` contains additional validation to prevent NaN/Inf and out-of-range predictions.

### CUDA and Device Handling
- `runners/train.py` sets `CUDA_LAUNCH_BLOCKING` and `TORCH_USE_CUDA_DSA` when `runtime.debug_cuda` or `DEBUG_CUDA=1`.
- Scripts present for CUDA diagnosis and tests under `scripts/` (e.g., `diagnose_cuda_issue.py`, `test_basic_cuda.py`).

### Duplications/Legacy (Candidates)
- Two inference stacks: `ui/utils/inference/*` vs `ui/apps/unified_ocr_app/services/*` with overlapping responsibilities.
- Multiple callback configs under `configs/callbacks/` with potential redundancy.
- Legacy docker CPU profile and backup scripts under `.backup/` suggest drift.

### Data Pipeline Observations
- `ValidatedOCRDataset` still enforces per-sample Pydantic + Shapely repairs; when polygons collapse it returns `None`, causing batch size variability downstream.
- `DBCollateFN` repeats polygon filtering and generates zeroed maps for invalid samples, letting bad data continue while diluting gradient signal.
- `OCRModel.forward` performs CPU detaches to check tensors each step, introducing sync overhead and hiding earlier NaN sources from Lightning anomaly detection.

## Risks and Inefficiencies (Preliminary)
- Large dependency set (Streamlit/UI/docs extras) inflates cold-start import times for training scripts.
- Automatic multi-GPU scaling in `runners/train.py` can trigger DDP on fragile configs, amplifying CUDA launch points without stabilizing data contracts.
- `ValidatedOCRDataset` + `DBCollateFN` duplicate polygon validation and rely on Shapely repairs; when polygons fail, batches shrink or carry zeroed maps, leading to unstable gradients and NaN-prone loss spikes.
- `OCRModel.forward` and `OCRPLModule.training_step` perform repeated `.detach().cpu()` validation passes, inducing host/device syncs that slow training and can mask root-cause tensors until after CUDA illegal instruction errors emerge.
- Dice loss guardrails act late in the pipeline; if upstream tensors saturate, the loss still throws, forcing reruns instead of preventing the bad data at source.

## Next Steps
1. Trace inference/UI flows (`ui/utils/inference`, `ui/apps/unified_ocr_app`) and document duplication vs shared components.
2. Catalog deprecated or redundant modules (callbacks, dataset helpers, docker backups) and flag migration blockers.
3. Profile import timings and dependency usage during training startup; identify candidates for lazy/optional import or removal.
4. Deep-dive CUDA illegal instruction and NaN reproduction path (dataset → collate → loss) and propose guardrails that fail fast.
5. Outline pruning and refactor plan covering training, inference, and tooling layers.

