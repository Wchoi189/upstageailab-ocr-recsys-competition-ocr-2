# workspaces Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-02-12

## Active Technologies
- Python 3.11 + PyTorch Lightning, Hydra/OmegaConf, WandB, TorchMetrics, PIL/OpenCV (001-wandb-config-logging)
- WandB run artifacts/tables + local outputs under `outputs/` (001-wandb-config-logging)
- Python 3.11 (repo standard via `uv run`) + PyTorch/Lightning OCR stack, Hydra/OmegaConf configs, Weights & Biases audit logging, ETK (`experiment_manager`) (003-ocr-data-quality-remediation)
- Filesystem artifacts (`docs/reports`, `specs/*`, `dev_tools/experiment_manager/experiments/*`), JSON/CSV manifests, W&B media/table artifacts (003-ocr-data-quality-remediation)

- Python 3.11 + `mcp`, `asyncio`, AgentQMS middleware, `project_compass`, `agent_debug_toolkit`, `starlette`/`uvicorn` (SSE transport) (001-mcp-tooling-refactor)

## Project Structure

```text
src/
tests/
```

## Commands

cd src [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] pytest [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] ruff check .

## Code Style

Python 3.11: Follow standard conventions

## Recent Changes
- 003-ocr-data-quality-remediation: Added Python 3.11 (repo standard via `uv run`) + PyTorch/Lightning OCR stack, Hydra/OmegaConf configs, Weights & Biases audit logging, ETK (`experiment_manager`)
- 001-wandb-config-logging: Added Python 3.11 + PyTorch Lightning, Hydra/OmegaConf, WandB, TorchMetrics, PIL/OpenCV
- 001-wandb-config-logging: Added [if applicable, e.g., PostgreSQL, CoreData, files or N/A]


<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
