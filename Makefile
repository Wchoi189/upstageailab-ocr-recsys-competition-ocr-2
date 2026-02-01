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
	uv sync

.PHONY: dev-install
dev-install: ## Install development dependencies
	uv sync --extra dev

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
