# Codebase Partitioning & Optimization Plan

## 🚨 Goal: Reduce "Context Bloat" for AI Performance
To improve AI agent navigation and reasoning, we will aggressively partition the codebase, moving non-essential assets (logs, data, archives) outside the project root (`/workspaces/upstageailab-ocr-recsys-competition-ocr-2`).

## 📋 Phase 1: The "Detox" (Immediate Action - High Impact)
**Goal:** Stop the automatic generation of noise files in the root.

### 1.1 Logs & Artifacts
**Action:** Move `wandb`, `lightning_logs`, and `outputs` directories to `/workspaces/project-artifacts/`.

*   **Create Directory:**
    ```bash
    mkdir -p /workspaces/project-artifacts/{wandb,lightning_logs,outputs,hydra_outputs}
    ```
*   **Update `.gitignore`:**
    ```gitignore
    # Ignore these in root
    wandb/
    lightning_logs/
    outputs/
    .uptraint_cache/
    ```
*   **Environment Configuration (`setup-env.sh` or `.bashrc`):**
    ```bash
    export WANDB_DIR="/workspaces/project-artifacts/wandb"
    # For PyTorch Lightning (if using generic PL defaults)
    # Note: PL usually follows the logger's dir, so WANDB_DIR helps.
    ```

### 1.2 Python Cache
**Action:** Disable `__pycache__` generation globally for this workspace.

*   **Environment Configuration:**
    ```bash
    export PYTHONDONTWRITEBYTECODE=1
    ```
*   **Cleanup Command:**
    ```bash
    find . -type d -name "__pycache__" -exec rm -rf {} +
    ```

---

## 📦 Phase 2: The "Vault" (Heavy Assets)
**Goal:** Move gigabytes of static data/archives out of the "reasoning path".

### 2.1 Archives
**Action:** Move `archive/`, `_archive/`, and `.archive/` folders.

*   **Command:**
    ```bash
    mkdir -p /workspaces/project-artifacts/archive
    mv archive/* /workspaces/project-artifacts/archive/
    mv _archive/* /workspaces/project-artifacts/archive/
    rm -rf archive _archive
    ```
*   **Rationale:** AI agents often "hallucinate" by reading archived code. Removing it is critical for correctness.

### 2.2 Data
**Action:** Move `data/` folder.

*   **Command:**
    ```bash
    mv data /workspaces/project-artifacts/data
    ln -s /workspaces/project-artifacts/data data  # OPTIONAL symlink for local code compatibility
    ```
*   **AI Instruction:** If you keep a symlink, create a `.agentignore` (if your tool supports it) or explicitly instruct agents to IGNORE `data/` in your prompt rules.

---

## 🧠 Phase 3: The "Triage" (Apps & Docs)
**Goal:** Separate "Application Logic" from "Core Library".

### 3.1 Frontend & UI (`apps/`)
**Situation:** `apps/` likely contains independent applications (e.g., Streamlit, FastAPI).
**Recommendation:** Move them to a sibling directory if they are loosely coupled.
*   **New Structure:**
    ```text
    /workspaces/
    ├── core-ocr-library/      (The current repo, trimmed down)
    └── frontend-apps/         (Functionally separate repo/folder)
    ```
*   **Wait:** Do this *after* Phases 1 & 2.

### 3.2 Documentation (`docs/`)
**Action:** Keep *Source* docs, remove *Generated* docs.
*   **Keep:** `docs/src`, `specs/`, `CONTRIBUTING.md`.
*   **Move:** `docs/site`, `docs/_build`.

---

## ⚙️ Phase 4: Configuration Updates (The "Rewiring")

### 4.1 Hydra Configuration
You need to tell Hydra where to write its logs/outputs.
**Edit:** `configs/hydra/default.yaml` (or equivalent).

```yaml
# Example Hydra Config Update
hydra:
  run:
    dir: /workspaces/project-artifacts/hydra_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
  sweep:
    dir: /workspaces/project-artifacts/hydra_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

### 4.2 PyTorch Lightning Logger
**Edit:** Your Trainer configuration (e.g., `ocr/engines/trainer.py` or `.yaml`).

```yaml
# YAML example
logger:
  _target_: lightning.pytorch.loggers.WandbLogger
  save_dir: "/workspaces/project-artifacts/wandb"
```

## ✅ Checklist for Execution
1. [ ] Run **Phase 1** cleanup commands (Logs/Cache).
2. [ ] executing **Phase 2** moves (Archive/Data).
3. [ ] Edit `setup-env.sh` to set `WANDB_DIR` and `PYTHONDONTWRITEBYTECODE`.
4. [ ] Update Hydra `configs/` to point to new output paths.
5. [ ] Update `pyproject.toml` or `mypy.ini` exclude lists to stop checking the artifact paths.
