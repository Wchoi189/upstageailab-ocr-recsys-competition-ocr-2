# Agent Instructions

This document defines the development standards and operational procedures for future AI agents working on this project.

## Dependency Management

### UV Package Manager
- **Strict Requirement**: You MUST use `uv` for all Python dependency management and execution.
- **Path**: The correct executable is located at `/opt/uv/bin/uv`.
- **Usage**:
  - Install dependencies: `/opt/uv/bin/uv pip install <package>`
  - Run scripts: `/opt/uv/bin/uv run <script>`
  - Do NOT use `pip` directly.
  - Do NOT install `uv` yourself; use the existing one at `/opt/`.

### Virtual Environment
- Verify `.venv` exists at the project root.
- If missing, create it using: `/opt/uv/bin/uv venv .venv --python 3.11`

## Python Version
- The project uses Python 3.11 (`.python-version`).
- Ensure `PYENV_VERSION` is NOT set to conflicting values like `doc-pyenv`.

## Running the Application
### Backend
```bash
export PYTHONPATH=$(pwd)
export OCR_CHECKPOINT_PATH=outputs/ocr_training_b/checkpoints/best.ckpt
/opt/uv/bin/uv run uvicorn apps.ocr_inference_console.adapters.api:app --reload --port 8000
```

### Frontend
```bash
cd apps/ocr-inference-console
npm run dev
```
