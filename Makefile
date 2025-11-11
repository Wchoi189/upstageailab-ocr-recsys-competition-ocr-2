# Makefile for OCR Project Development
# Last Updated: 2025-10-21
# Version: 0.1.1
# Changes: Reorganized structure, eliminated duplication, improved help system

PORT ?= 8501

# UI Apps
UI_APPS := command_builder evaluation_viewer inference preprocessing_viewer resource_monitor unified_app

.PHONY: help install dev-install test test-cov lint lint-fix format quality-check quality-fix clean docs-build docs-serve docs-deploy diagrams-check diagrams-update diagrams-force-update diagrams-validate diagrams-update-specific serve-% stop-% status-% logs-% clear-logs-% list-ui-processes stop-all-ui pre-commit setup-dev ci context-log-start context-log-summarize quick-fix-log

# ============================================================================
# HELP
# ============================================================================

help:
	@echo "OCR Project Development Commands"
	@echo "================================"
	@echo ""
	@echo "📦 INSTALLATION"
	@echo "  install             - Install production dependencies"
	@echo "  dev-install         - Install development dependencies"
	@echo "  setup-dev           - Full development environment setup"
	@echo ""
	@echo "🧪 TESTING"
	@echo "  test                - Run tests"
	@echo "  test-cov            - Run tests with coverage report"
	@echo ""
	@echo "💅 CODE QUALITY"
	@echo "  lint                - Run linting checks"
	@echo "  lint-fix            - Run linting checks and auto-fix issues"
	@echo "  format              - Format code with black and isort"
	@echo "  quality-check       - Run comprehensive code quality checks"
	@echo "  quality-fix         - Auto-fix code quality issues"
	@echo "  pre-commit          - Install and run pre-commit hooks"
	@echo ""
	@echo "📚 DOCUMENTATION"
	@echo "  docs-build          - Build MkDocs documentation"
	@echo "  docs-serve          - Serve MkDocs documentation locally"
	@echo "  docs-deploy         - Deploy MkDocs documentation to GitHub Pages"
	@echo ""
	@echo "📊 DIAGRAMS"
	@echo "  diagrams-check      - Check which diagrams need updates"
	@echo "  diagrams-update     - Update diagrams that have changed"
	@echo "  diagrams-force-update - Force update all diagrams"
	@echo "  diagrams-validate   - Validate diagram syntax"
	@echo "  diagrams-update-specific - Update specific diagrams (use DIAGRAMS=...)"
	@echo ""
	@echo "🖥️  UI APPLICATIONS"
	@echo "  serve-<app>         - Start UI app (apps: $(UI_APPS))"
	@echo "  stop-<app>          - Stop UI app"
	@echo "  status-<app>        - Check UI app status"
	@echo "  logs-<app>          - View UI app logs"
	@echo "  clear-logs-<app>    - Clear UI app logs"
	@echo "  list-ui-processes   - List all running UI processes"
	@echo "  stop-all-ui         - Stop all UI processes"
	@echo ""
	@echo "🔧 DEVELOPMENT WORKFLOW"
	@echo "  clean               - Clean up cache files and build artifacts"
	@echo "  ci                  - Run CI checks (quality + tests)"
	@echo "  context-log-start   - Create context log (use LABEL=...)"
	@echo "  context-log-summarize - Summarize context log (use LOG=...)"
	@echo "  quick-fix-log       - Log quick fix (use TYPE= TITLE= ISSUE= FIX= FILES=)"

# ============================================================================
# INSTALLATION
# ============================================================================

install:
	uv sync

dev-install:
	uv sync --extra dev

setup-dev: dev-install pre-commit
	@echo "Development environment setup complete!"
	@echo "Run 'make quality-check' to verify everything is working"

# ============================================================================
# TESTING
# ============================================================================

test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ -v --cov=ocr --cov-report=html

# ============================================================================
# CODE QUALITY
# ============================================================================

lint:
	uv run ruff check .

lint-fix:
	uv run ruff check --fix .

format:
	uv run ruff format .

quality-check:
	uv run ruff check .
	uv run mypy ocr/
	uv run ruff format --check .

quality-fix:
	./scripts/code-quality.sh

pre-commit:
	pre-commit install
	pre-commit run --all-files

# ============================================================================
# CLEANUP
# ============================================================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	rm -rf build/ dist/ .coverage htmlcov/

# ============================================================================
# DOCUMENTATION
# ============================================================================

docs-build:
	uv run mkdocs build

docs-serve:
	uv run mkdocs serve --dev-addr=127.0.0.1:8000

docs-deploy:
	uv run mkdocs gh-deploy

# ============================================================================
# DIAGRAMS
# ============================================================================

diagrams-check:
	@echo "🔍 Checking for diagram updates..."
	python scripts/documentation/generate_diagrams.py --check-changes

diagrams-update:
	@echo "🔄 Updating diagrams that have changed..."
	python scripts/documentation/generate_diagrams.py --update

diagrams-force-update:
	@echo "🔄 Force updating all diagrams..."
	python scripts/documentation/generate_diagrams.py --update --force

diagrams-validate:
	@echo "✅ Validating diagram syntax..."
	python scripts/documentation/generate_diagrams.py --validate

diagrams-update-specific:
	@echo "🔄 Updating specific diagrams: $(DIAGRAMS)"
	python scripts/documentation/generate_diagrams.py --update $(DIAGRAMS)

# ============================================================================
# UI APPLICATIONS (Parameterized)
# ============================================================================

# Start UI applications
serve-command_builder serve-ui:
	uv run python scripts/utilities/process_manager.py start command_builder --port=$(PORT)

serve-evaluation_viewer serve-evaluation-ui:
	uv run python scripts/utilities/process_manager.py start evaluation_viewer --port=$(PORT)

serve-inference serve-inference-ui:
	uv run python scripts/utilities/process_manager.py start inference --port=$(PORT)

serve-preprocessing_viewer serve-preprocessing-viewer:
	uv run python scripts/utilities/process_manager.py start preprocessing_viewer --port=$(PORT)

serve-resource_monitor serve-resource-monitor:
	uv run python scripts/utilities/process_manager.py start resource_monitor --port=$(PORT)

serve-unified_app serve-unified-app:
	uv run python scripts/utilities/process_manager.py start unified_app --port=$(PORT)

# Stop UI applications
stop-command_builder stop-ui:
	uv run python scripts/utilities/process_manager.py stop command_builder --port=$(PORT)

stop-evaluation_viewer stop-evaluation-ui:
	uv run python scripts/utilities/process_manager.py stop evaluation_viewer --port=$(PORT)

stop-inference stop-inference-ui:
	uv run python scripts/utilities/process_manager.py stop inference --port=$(PORT)

stop-preprocessing_viewer stop-preprocessing-viewer:
	uv run python scripts/utilities/process_manager.py stop preprocessing_viewer --port=$(PORT)

stop-resource_monitor stop-resource-monitor:
	uv run python scripts/utilities/process_manager.py stop resource_monitor --port=$(PORT)

stop-unified_app stop-unified-app:
	uv run python scripts/utilities/process_manager.py stop unified_app --port=$(PORT)

# Check UI application status
status-command_builder status-ui:
	uv run python scripts/utilities/process_manager.py status command_builder --port=$(PORT)

status-evaluation_viewer status-evaluation-ui:
	uv run python scripts/utilities/process_manager.py status evaluation_viewer --port=$(PORT)

status-inference status-inference-ui:
	uv run python scripts/utilities/process_manager.py status inference --port=$(PORT)

status-preprocessing_viewer status-preprocessing-viewer:
	uv run python scripts/utilities/process_manager.py status preprocessing_viewer --port=$(PORT)

status-resource_monitor status-resource-monitor:
	uv run python scripts/utilities/process_manager.py status resource_monitor --port=$(PORT)

status-unified_app status-unified-app:
	uv run python scripts/utilities/process_manager.py status unified_app --port=$(PORT)

# View UI application logs
logs-command_builder logs-ui:
	uv run python scripts/utilities/process_manager.py logs command_builder --port=$(PORT)

logs-evaluation_viewer logs-evaluation-ui:
	uv run python scripts/utilities/process_manager.py logs evaluation_viewer --port=$(PORT)

logs-inference logs-inference-ui:
	uv run python scripts/utilities/process_manager.py logs inference --port=$(PORT)

logs-preprocessing_viewer logs-preprocessing-viewer:
	uv run python scripts/utilities/process_manager.py logs preprocessing_viewer --port=$(PORT)

logs-resource_monitor logs-resource-monitor:
	uv run python scripts/utilities/process_manager.py logs resource_monitor --port=$(PORT)

logs-unified_app logs-unified-app:
	uv run python scripts/utilities/process_manager.py logs unified_app --port=$(PORT)

# Clear UI application logs
clear-logs-command_builder clear-logs-ui:
	uv run python scripts/utilities/process_manager.py clear-logs command_builder --port=$(PORT)

clear-logs-evaluation_viewer clear-logs-evaluation-ui:
	uv run python scripts/utilities/process_manager.py clear-logs evaluation_viewer --port=$(PORT)

clear-logs-inference clear-logs-inference-ui:
	uv run python scripts/utilities/process_manager.py clear-logs inference --port=$(PORT)

clear-logs-preprocessing_viewer clear-logs-preprocessing-viewer:
	uv run python scripts/utilities/process_manager.py clear-logs preprocessing_viewer --port=$(PORT)

clear-logs-resource_monitor clear-logs-resource-monitor:
	uv run python scripts/utilities/process_manager.py clear-logs resource_monitor --port=$(PORT)

clear-logs-unified_app clear-logs-unified-app:
	uv run python scripts/utilities/process_manager.py clear-logs unified_app --port=$(PORT)

# UI process management
list-ui-processes:
	uv run python scripts/utilities/process_manager.py list

stop-all-ui:
	uv run python scripts/utilities/process_manager.py stop-all

# ============================================================================
# DEVELOPMENT WORKFLOW
# ============================================================================

context-log-start:
	uv run python scripts/agent_tools/utilities/context_log.py start $(if $(LABEL),--label "$(LABEL)")

context-log-summarize:
	@if [ -z "$(LOG)" ]; then \
		echo "Usage: make context-log-summarize LOG=logs/agent_runs/<file>.jsonl"; \
		exit 1; \
	fi
	uv run python scripts/agent_tools/utilities/context_log.py summarize --log-file $(LOG)

quick-fix-log:
	@if [ -z "$(TYPE)" ] || [ -z "$(TITLE)" ] || [ -z "$(ISSUE)" ] || [ -z "$(FIX)" ] || [ -z "$(FILES)" ]; then \
		echo "Usage: make quick-fix-log TYPE=<type> TITLE=\"<title>\" ISSUE=\"<issue>\" FIX=\"<fix>\" FILES=\"<files>\" [IMPACT=<impact>] [TEST=<test>]"; \
		echo "Types: bug, compat, config, dep, doc, perf, sec, ui"; \
		echo "Example: make quick-fix-log TYPE=bug TITLE=\"Pydantic compatibility\" ISSUE=\"replace() error\" FIX=\"Use model_copy\" FILES=\"ui/state.py\""; \
		exit 1; \
	fi
	uv run python scripts/agent_tools/utilities/quick_fix_log.py $(TYPE) "$(TITLE)" --issue "$(ISSUE)" --fix "$(FIX)" --files "$(FILES)" $(if $(IMPACT),--impact $(IMPACT)) $(if $(TEST),--test $(TEST))

# ============================================================================
# CI SIMULATION
# ============================================================================

ci: quality-check test
	@echo "CI checks passed! ✅"
