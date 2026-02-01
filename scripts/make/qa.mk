# QA and Code Quality Targets

.PHONY: lint
lint: ## Run linting checks
	uv run ruff check .

.PHONY: lint-fix
lint-fix: ## Run linting checks and auto-fix issues
	uv run ruff check --fix .

.PHONY: lint-check-json
lint-check-json: ## Output ruff results as JSON for AI processing
	uv run ruff check . --output-format=json

.PHONY: format
format: ## Format code with ruff
	uv run ruff format .

.PHONY: quality-check
quality-check: ## Run comprehensive code quality checks (lint, type-check, format-check)
	uv run ruff check .
	uv run mypy ocr/
	uv run ruff format --check .

.PHONY: test
test: ## Run tests
	uv run pytest tests/ -v

.PHONY: test-cov
test-cov: ## Run tests with coverage report
	uv run pytest tests/ -v --cov=ocr --cov-report=html

.PHONY: pre-commit
pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

.PHONY: pre-commit-install
pre-commit-install: ## Install pre-commit hooks
	pre-commit install
	@echo "✅ Pre-commit hooks installed"

.PHONY: clean
clean: ## Clean up cache files and build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ .coverage htmlcov/
