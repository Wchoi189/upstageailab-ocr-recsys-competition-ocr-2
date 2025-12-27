# Walkthrough - ETK Self-Healing & Documentation Updates

I have successfully implemented the "Self-Healing State Machine" recommendations, optimized the project documentation, hardened the installation script, fully restored the experiment registry, added lifecycle management commands, and configured the web worker environment.

## Changes

### 1. **`etk reconcile` Command**
- **New Command**: `etk reconcile [experiment_id] [--all]`
- **Functionality**: Scans `.metadata/` for all artifacts, rebuilds the [artifacts](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/experiment_manager/src/etk/core.py#229-231) list in [manifest.json](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/experiment_manager/experiments/20251217_024343_image_enhancements_implementation/manifest.json), and updates `last_reconciled`.
- **Global Audit**: Added `--all` flag to reconcile every experiment in the registry at once.
- **Fix**: Updated `reconcile --all` to scan the filesystem directly instead of querying the database, resolving a "chicken-and-egg" issue.

### 2. **`etk prune` Command**
- **New Command**: `etk prune [--dry-run]`
- **Functionality**: detects experiments present in the database but missing from the filesystem (e.g. manually archived/deleted) and removes them from the database.
- **Safety**: Includes `--dry-run` to preview changes before deletion.

### 3. **Schema Evolution**
- **Manifest Schema**: Updated [manifest.schema.json](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/experiment_manager/.schemas/manifest.schema.json) to include:
  - `subtype`: Allows distinguishing special report types (e.g., `handoff`).
  - `last_reconciled`: Tracks when the manifest was last synced with the filesystem.
- **Compliance Checker**: Updated validation logic to relax strict field requirements for reports with `subtype: handoff`.

### 4. **Workflow & Documentation**
- **Workflows**: Updated [tier4-workflows/experiment-workflow.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/.ai-instructions/tier4-workflows/experiment-workflow.yaml) to use `uv run python` for robust execution and added [reconcile](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/experiment_manager/src/etk/reconciler.py#20-51) steps.
- **Agents Guide**: Updated [AGENTS.md](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AGENTS.md) with a "Quick Link" to the Experiment Manager (ETK) CLI and resources.
- **Installation Script**: Modified [install-etk.sh](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/experiment_manager/install-etk.sh) to generate a **smart shim** (wrapper script) instead of a symlink.

### 5. **Performance Optimization**
- **Result**: Reduced command latency from **~5.0s** to **~1.36s** (3.5x speedup) by prioritizing direct venv execution in the shim.

### 6. **Web Worker Configuration**
- **Config**: Updated [docker/Dockerfile](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/docker/Dockerfile) to include an alias for [etk](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/experiment_manager/install-etk.sh#57-107).
- **Implementation**: Added `echo 'alias etk="uv run python experiment_manager/etk.py"' >> ~/.bashrc` to the Docker setup.
- **Benefit**: Ensures any container (worker) built from this image has [etk](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/experiment_manager/install-etk.sh#57-107) available by default without manual installation steps.

## Validation

### Verification Steps
1. **Schema Compliance**: Updated [manifest.schema.json](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/experiment_manager/.schemas/manifest.schema.json) to use correct lowercase types.
2. **Reconciliation**:
   - Ran `etk reconcile --all` -> Successfully synced all 7 experiments.
3. **Registry**:
   - Ran `etk sync --all` -> Synced 59 artifacts.
   - Ran `etk list` -> Displayed all 7 experiments.
4. **Performance**:
   - Benchmarked `etk --version`: 1.39s.
5. **Pruning**:
   - Verified `etk prune` correctly removes stale entries.
6. **Docker Config**:
   - Verified [docker/Dockerfile](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/docker/Dockerfile) contains the [etk](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/experiment_manager/install-etk.sh#57-107) alias.

## Usage
To fix filesystem drift across the entire project:
```bash
uv run python experiment_manager/etk.py reconcile --all
```

To cleanup stale entries after manual archiving:
```bash
etk prune --dry-run
etk prune
```

To install the global [etk](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/experiment_manager/install-etk.sh#57-107) command (optimized):
```bash
bash experiment_manager/install-etk.sh
```
