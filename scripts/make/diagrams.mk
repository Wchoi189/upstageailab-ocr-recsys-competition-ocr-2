# Diagram Generation Targets

DIAGRAMS_SCRIPT = scripts/documentation/generate_diagrams.py

.PHONY: diagrams-check
diagrams-check: ## Check which diagrams need updates
	uv run python $(DIAGRAMS_SCRIPT) --check-changes

.PHONY: diagrams-update
diagrams-update: ## Update diagrams that have changed
	uv run python $(DIAGRAMS_SCRIPT) --update

.PHONY: diagrams-force-update
diagrams-force-update: ## Force update all diagrams
	uv run python $(DIAGRAMS_SCRIPT) --update --force

.PHONY: diagrams-validate
diagrams-validate: ## Validate diagram syntax
	uv run python $(DIAGRAMS_SCRIPT) --validate
