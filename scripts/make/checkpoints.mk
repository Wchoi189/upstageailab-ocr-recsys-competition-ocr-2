# Checkpoint Management Targets

.PHONY: checkpoint-metadata
checkpoint-metadata: ## Generate metadata files for all checkpoints
	uv run python scripts/checkpoints/generate_metadata.py

.PHONY: checkpoint-index-rebuild
checkpoint-index-rebuild: ## Rebuild checkpoint index from file system
	uv run python -c "from pathlib import Path; from ocr.utils.checkpoints.index import CheckpointIndex; import time; outputs_dir = Path('outputs'); index = CheckpointIndex(outputs_dir, include_legacy=True); index.rebuild()"
