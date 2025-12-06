## Organizing the `outputs/` directory

### 1. Define a stable top-level layout

Adopt a small, fixed set of subfolders under `outputs/` so every artifact type has a clear home:

- **`outputs/experiments/`**: end-to-end experiment runs (training, preprocessing, ablations).
- **`outputs/checkpoints/`**: long-term model checkpoints you intend to keep independent of runs.
- **`outputs/logs/`**: global logs not tied to a single run (e.g., system-wide or cron jobs).
- **`outputs/artifacts/`**: shared derived assets (plots, datasets, exported features) that are reused across runs.

### 2. Standardize experiment run structure

Inside `outputs/experiments/`, use a consistent hierarchy based on task, model, and run id:

- **Suggested pattern**: `outputs/experiments/<task>/<model>/<run_id>/`
- **`<task>`**: e.g., `ocr`, `recsys`, `preproc`, `debug`.
- **`<model>`**: e.g., `bert-base`, `vit-large`, `xgb-v1`.
- **`<run_id>`**: timestamp plus short tag, e.g., `2025-12-04_14-32-10_seed42`.
- **Within each run directory**, mirror a simple, repeatable structure:
- **`config.yaml`** (or `config.json`): exact configuration used.
- **`logs/`**: run-specific logs (`train.log`, `eval.log`, `preproc.log`).
- **`checkpoints/`**: run’s checkpoints (`epoch_*.pt`, `best.pt`).
- **`metrics/`**: JSON/CSV metric dumps (`val_metrics.json`, `test_metrics.csv`).
- **`plots/`**: loss curves, confusion matrices, PR curves.
- **`data/`** (optional): run-specific derived data (e.g., sampled subsets, temporary artifacts).

### 3. Separate reusable artifacts from run-local outputs

To avoid hunting through runs for something you reuse later, promote stable artifacts into `outputs/artifacts/`:

- **`outputs/artifacts/datasets/`**: processed datasets or feature stores with clear names, e.g., `ocr_train_v2/`, `features_v1.parquet`.
- **`outputs/artifacts/plots/`**: final figures for reports/papers, named by task and date.
- **`outputs/artifacts/exports/`**: ONNX/TorchScript models, submission files, etc.
- Keep a short **`README.md`** inside `outputs/artifacts/` describing naming conventions and which project components consume them.

### 4. Use clear naming conventions and metadata

Make individual files self-explanatory even when viewed out of context:

- **File naming**:
- Include **split**, **metric**, and **stage** where sensible, e.g., `val_auc.json`, `train_loss_curve.png`, `test_predictions.parquet`.
- For checkpoints, encode **epoch/step and metric**, e.g., `epoch03_valAcc0.87.pt`.
- **Metadata file per run**:
- Add a small `run_info.json` or `README.md` in each run directory summarizing: command used, git commit (if available), seeds, key metrics, and notes.

### 5. Distinguish short-lived vs long-lived outputs

Avoid clutter by deciding which outputs are ephemeral and can be cleaned up:

- Within each run directory, treat:
- **Ephemeral**: large intermediate tensors, temporary caches, batch-wise debug dumps → put under `tmp/` or `cache/` inside the run and make them safe to delete.
- **Persistent**: final checkpoints, metrics, and plots.
- Optionally add a simple cleanup script or Make target that removes all `tmp/` and old checkpoints except `best.pt` and last epoch.

### 6. Make preprocessing experiments first-class

Treat preprocessing experiments similarly to training runs for traceability:

- Use `outputs/experiments/preproc/<pipeline_name>/<run_id>/`.
- Inside, include:
- `config.yaml` (schema versions, augmentations, filters).
- `logs/` and `metrics/` (e.g., coverage stats, missing value counts).
- `data/` pointing to or containing the resulting processed dataset.
- When a preprocessing run becomes “blessed” for training, promote its resulting dataset into `outputs/artifacts/datasets/` with a versioned name, and reference it from training configs.

### 7. Update experiment-tracker references

- Mirror the new `outputs/` layout inside `experiment-tracker/experiments/*/[assessments|incident_reports]/` so every Markdown file points to the canonical run folder.
- Add a lightweight script or checklist to regenerate links after moving experiments so documentation stays in sync.
- Include relative links when possible so the tracker stays portable across machines.

### 8. Document and enforce the layout lightly

- Add a brief section to your main project `README` or a dedicated `docs/outputs_structure.md` explaining:
- The `outputs/` layout.
- How run ids are constructed.
- Which folders are safe to delete.
- If you have experiment runner scripts, encode this layout in them so paths are auto-created following the convention, minimizing manual work.
