## `outputs/` directory layout

This project uses a structured layout for all generated artifacts. The goal is to keep
training runs, evaluation runs, and long-lived artifacts easy to find and safe to clean.

### Top-level

- `outputs/experiments/` – run-local outputs from training, evaluation, and preprocessing.
- `outputs/artifacts/` – promoted, reusable assets (datasets, models, submissions, final plots).
- `outputs/logs/` – global free-form logs (e.g., markdown notes of manual tests).
- `outputs/tmp/` – scratch space and ephemeral junk.

### Experiments hierarchy

All Hydra-driven runs write under:

- `outputs/experiments/<kind>/<task>/<name>/<run_id>/`

Where:

- `<kind>` – logical category of run (e.g., `train`, `eval`, `preproc`).
- `<task>` – domain/task label (e.g., `ocr`).
- `<name>` – experiment name from `exp_name` (e.g., `ocr_training_b`).
- `<run_id>` – timestamp + seed from the Hydra config (`${now:%Y%m%d_%H%M%S}_${seed}`).

Inside a typical **training** run (e.g., `train/ocr/ocr_training_b/...`) you will find:

- `.hydra/` – resolved Hydra configs for this run.
- `logs/` – `train.log` and auxiliary JSON/JSONL logs.
- `checkpoints/` – all `best-*.ckpt`, `last-*.ckpt` and metadata.
- `submissions/` – run-specific submission files (if any).
- `wandb/` – per-run W&B directory (if enabled).

Legacy runs (e.g., the original `outputs/ocr_training_b/`) have been **copied** into
`outputs/experiments/...` under a `legacy_run/` subdirectory so older references continue
to work while the new structure is adopted.

### Evaluation runs

Common evaluation families are organized under:

- `outputs/experiments/eval/improved_edge/...`
- `outputs/experiments/eval/perspective/...`
- `outputs/experiments/eval/rembg/...`
- `outputs/experiments/eval/worst_performers/...`

Each leaf directory typically contains JSON metrics, lists of failing examples, and similar
artifacts copied from the legacy flat `outputs/*` folders.

### Artifacts

Reusable artifacts that are not tied to a single run belong under:

- `outputs/artifacts/datasets/` – processed datasets or feature stores (not yet populated).
- `outputs/artifacts/models/` – shared checkpoints or exported models.
- `outputs/artifacts/submissions/` – final or canonical submission files.

These directories are intended to hold *versioned* assets that other configs/scripts
refer to via stable paths, rather than by pointing into individual run folders.

### Deletion / cleanup rules

Safe to delete (when not needed for debugging):

- `outputs/tmp/` and any `tmp/` subdirectory under a specific run.
- Old evaluation runs under `outputs/experiments/eval/*` that are already documented
  in `experiment_manager/`.
- Intermediate or superseded checkpoints inside a run’s `checkpoints/` directory,
  as long as you keep the `best` and/or final `last` checkpoint you care about.

Use caution before deleting:

- Anything under `outputs/experiments/train/...` for runs you may want to resume from.
- Items under `outputs/artifacts/`, since these are shared, versioned assets.

