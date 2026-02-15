# OCR Project Development Makefile
# ============================================================================

.DEFAULT_GOAL := help

# Include modular makefiles
include scripts/make/*.mk

# ============================================================================
# CORE TARGETS
# ============================================================================

.PHONY: install
install: ## Install production dependencies
	uv sync --no-dev

.PHONY: dev-install
dev-install: ## Install development dependencies
	uv sync --group dev
	@echo "Installing workspace dev tools..."
	uv pip install -e dev_tools/agent_debug_toolkit \
	                -e dev_tools/experiment_manager \
	                -e dev_tools/project_compass \
	                -e dev_tools/airflow_batch_processor

.PHONY: setup-dev
setup-dev: dev-install pre-commit-install ## Full development environment setup
	@echo "✅ Development environment setup complete!"

.PHONY: help
help: ## Display this help screen
	@echo "OCR Project Development Commands"
	@echo "================================"
	@echo ""
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo ""

.PHONY: fix-hydra
fix-hydra: ## Fix Hydra CLI overrides by auto-adding + prefix (pass CMD="your command")
	@if [ -z "$(CMD)" ]; then \
		echo "Usage: make fix-hydra CMD='uv run python scripts/runners/train.py mode=train ...'"; \
		exit 1; \
	fi
	@uv run python scripts/utils/fix_hydra_overrides.py "$(CMD)" --verbose
