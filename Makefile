# Makefile for OCR Project Development
# Last Updated: 2025-10-21
# Version: 0.1.1
# Changes: Reorganized structure, eliminated duplication, improved help system

PORT ?= 8501
FRONTEND_DIR := frontend
FRONTEND_HOST ?= 0.0.0.0
FRONTEND_PORT ?= 5173
BACKEND_HOST ?= 127.0.0.1
BACKEND_PORT ?= 8000
BACKEND_APP ?= services.playground_api.app:app

# UI Apps
UI_APPS := command_builder evaluation_viewer inference preprocessing_viewer resource_monitor unified_app

.PHONY: help install dev-install test test-cov lint lint-fix format quality-check quality-fix clean docs-build docs-serve docs-deploy diagrams-check diagrams-update diagrams-force-update diagrams-validate diagrams-update-specific serve-% stop-% status-% logs-% clear-logs-% list-ui-processes stop-all-ui pre-commit setup-dev ci frontend-ci console-ci context-log-start context-log-summarize quick-fix-log start stop cb eval infer prep monitor ua stop-cb stop-eval stop-infer stop-prep stop-monitor stop-ua frontend-dev frontend-stop fe sfe console-dev console-build console-lint backend-dev backend-stop stack-dev stack-stop fs stop-fs

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
	@echo "  Shortcuts           - cb eval infer prep monitor ua (start Unified App)"
	@echo "  Stop Shortcuts      - stop-cb stop-eval stop-infer stop-prep stop-monitor stop-ua"
	@echo "  start / stop        - Start/Stop Unified App (aliases for ua)"
	@echo ""
	@echo "🌐 FRONTEND"
	@echo "  fe                 - Start Vite dev server (alias for frontend-dev)"
	@echo "  frontend-dev       - Vite dev server on $(FRONTEND_HOST):$(FRONTEND_PORT)"
	@echo "  sfe                - Stop Vite dev server listening on $(FRONTEND_PORT)"
	@echo ""
	@echo "🧭 NEXT.JS CONSOLE"
	@echo "  console-dev        - Start Next.js App Router dev server (apps/playground-console)"
	@echo "  console-build      - Build the Next.js console for production deploys"
	@echo "  console-lint       - Run console linting (see docs/maintainers/coding_standards.md)"
	@echo ""
	@echo "🧩 SPA STACK"
	@echo "  backend-dev        - Start FastAPI playground backend (reload)"
	@echo "  backend-stop       - Stop FastAPI backend on $(BACKEND_PORT)"
	@echo "  fs                 - Run backend + frontend together (alias for stack-dev)"
	@echo "  stack-dev          - Combined dev stack (kills backend when frontend exits)"
	@echo "  stop-fs            - Stop combined stack (alias for stack-stop)"
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
	python scripts/generate_diagrams.py --check-changes

diagrams-update:
	@echo "🔄 Updating diagrams that have changed..."
	python scripts/generate_diagrams.py --update

diagrams-force-update:
	@echo "🔄 Force updating all diagrams..."
	python scripts/generate_diagrams.py --update --force

diagrams-validate:
	@echo "✅ Validating diagram syntax..."
	python scripts/generate_diagrams.py --validate

diagrams-update-specific:
	@echo "🔄 Updating specific diagrams: $(DIAGRAMS)"
	python scripts/generate_diagrams.py --update $(DIAGRAMS)

# ============================================================================
# UI APPLICATIONS (Parameterized)
# ============================================================================

# Friendly aliases (one-word shortcuts)
cb: serve-command_builder
eval: serve-evaluation_viewer
infer: serve-inference
prep: serve-preprocessing_viewer
monitor: serve-resource_monitor
ua: serve-unified_app

stop-cb: stop-command_builder
stop-eval: stop-evaluation_viewer
stop-infer: stop-inference
stop-prep: stop-preprocessing_viewer
stop-monitor: stop-resource_monitor
stop-ua: stop-unified_app

start: ua
stop: stop-unified_app

fe: frontend-dev

frontend-dev:
	cd $(FRONTEND_DIR) && npm run dev -- --host $(FRONTEND_HOST) --port $(FRONTEND_PORT)

console-dev:
	npm run dev:console

console-build:
	npm run build:console

console-lint:
	npm run lint:console

sfe: frontend-stop

frontend-stop:
	@PORT=$(FRONTEND_PORT); \
	PIDS=""; \
	if command -v lsof >/dev/null 2>&1; then \
		for PID in $$(lsof -t -i:$$PORT 2>/dev/null || true); do \
			CMD=$$(ps -p $$PID -o args= 2>/dev/null || true); \
			if echo "$$CMD" | grep -qi "vite"; then \
				PIDS="$$PIDS $$PID"; \
			fi; \
		done; \
	elif command -v pgrep >/dev/null 2>&1; then \
		PIDS=$$(pgrep -f "vite.*--port $$PORT" || true); \
	fi; \
	if [ -n "$$PIDS" ]; then \
		echo "Stopping project frontend dev server on port $$PORT (PID(s) $$PIDS)"; \
		kill $$PIDS; \
	else \
		echo "No project frontend dev server detected on port $$PORT"; \
	fi

backend-dev:
	uv run uvicorn $(BACKEND_APP) --host $(BACKEND_HOST) --port $(BACKEND_PORT) --reload

backend-stop:
	@PORT=$(BACKEND_PORT); \
	PIDS=""; \
	if command -v lsof >/dev/null 2>&1; then \
		for PID in $$(lsof -t -i:$$PORT 2>/dev/null || true); do \
			CMD=$$(ps -p $$PID -o args= 2>/dev/null || true); \
			if echo "$$CMD" | grep -Eq "uvicorn|run_spa\.py|run_ui\.py"; then \
				PIDS="$$PIDS $$PID"; \
			fi; \
		done; \
	elif command -v pgrep >/dev/null 2>&1; then \
		PIDS=$$(pgrep -f "uvicorn.*$(BACKEND_APP)" || true); \
	fi; \
	if [ -n "$$PIDS" ]; then \
		echo "Stopping project backend server on port $$PORT (PID(s) $$PIDS)"; \
		kill $$PIDS; \
	else \
		echo "No project backend server detected on port $$PORT"; \
	fi

fs: stack-dev

stack-dev:
	@bash -c 'set -euo pipefail; \
		echo "Starting FastAPI backend on $(BACKEND_HOST):$(BACKEND_PORT)"; \
		uv run uvicorn $(BACKEND_APP) --host $(BACKEND_HOST) --port $(BACKEND_PORT) --reload & \
		BACK_PID=$$!; \
		trap "echo \"Stopping backend (PID $$BACK_PID)\"; kill $$BACK_PID 2>/dev/null || true" EXIT INT TERM; \
		cd $(FRONTEND_DIR); \
		npm run dev -- --host $(FRONTEND_HOST) --port $(FRONTEND_PORT); \
	'

sfs: stack-stop

stack-stop: backend-stop
	$(MAKE) sfe

# Start UI applications
serve-command_builder:
	uv run python scripts/process_manager.py start command_builder --port=$(PORT)

serve-evaluation_viewer:
	uv run python scripts/process_manager.py start evaluation_viewer --port=$(PORT)

serve-inference:
	uv run python scripts/process_manager.py start inference --port=$(PORT)

serve-preprocessing_viewer:
	uv run python scripts/process_manager.py start preprocessing_viewer --port=$(PORT)

serve-resource_monitor:
	uv run python scripts/process_manager.py start resource_monitor --port=$(PORT)

serve-unified_app:
	uv run python scripts/process_manager.py start unified_app --port=$(PORT)

# Stop UI applications
stop-command_builder:
	uv run python scripts/process_manager.py stop command_builder --port=$(PORT)

stop-evaluation_viewer:
	uv run python scripts/process_manager.py stop evaluation_viewer --port=$(PORT)

stop-inference:
	uv run python scripts/process_manager.py stop inference --port=$(PORT)

stop-preprocessing_viewer:
	uv run python scripts/process_manager.py stop preprocessing_viewer --port=$(PORT)

stop-resource_monitor:
	uv run python scripts/process_manager.py stop resource_monitor --port=$(PORT)

stop-unified_app:
	uv run python scripts/process_manager.py stop unified_app --port=$(PORT)

# Check UI application status
status-command_builder:
	uv run python scripts/process_manager.py status command_builder --port=$(PORT)

status-evaluation_viewer:
	uv run python scripts/process_manager.py status evaluation_viewer --port=$(PORT)

status-inference:
	uv run python scripts/process_manager.py status inference --port=$(PORT)

status-preprocessing_viewer:
	uv run python scripts/process_manager.py status preprocessing_viewer --port=$(PORT)

status-resource_monitor:
	uv run python scripts/process_manager.py status resource_monitor --port=$(PORT)

status-unified_app:
	uv run python scripts/process_manager.py status unified_app --port=$(PORT)

# View UI application logs
logs-command_builder:
	uv run python scripts/process_manager.py logs command_builder --port=$(PORT)

logs-evaluation_viewer:
	uv run python scripts/process_manager.py logs evaluation_viewer --port=$(PORT)

logs-inference:
	uv run python scripts/process_manager.py logs inference --port=$(PORT)

logs-preprocessing_viewer:
	uv run python scripts/process_manager.py logs preprocessing_viewer --port=$(PORT)

logs-resource_monitor:
	uv run python scripts/process_manager.py logs resource_monitor --port=$(PORT)

logs-unified_app:
	uv run python scripts/process_manager.py logs unified_app --port=$(PORT)

# Clear UI application logs
clear-logs-command_builder:
	uv run python scripts/process_manager.py clear-logs command_builder --port=$(PORT)

clear-logs-evaluation_viewer:
	uv run python scripts/process_manager.py clear-logs evaluation_viewer --port=$(PORT)

clear-logs-inference:
	uv run python scripts/process_manager.py clear-logs inference --port=$(PORT)

clear-logs-preprocessing_viewer:
	uv run python scripts/process_manager.py clear-logs preprocessing_viewer --port=$(PORT)

clear-logs-resource_monitor:
	uv run python scripts/process_manager.py clear-logs resource_monitor --port=$(PORT)

clear-logs-unified_app:
	uv run python scripts/process_manager.py clear-logs unified_app --port=$(PORT)

# UI process management
list-ui-processes:
	uv run python scripts/process_manager.py list

stop-all-ui:
	uv run python scripts/process_manager.py stop-all

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

ci: quality-check frontend-ci console-ci test
	@echo "CI checks passed! ✅"

frontend-ci:
	npm run lint:spa
	npm run build:spa

console-ci:
	npm run lint:console
	npm run build:console
