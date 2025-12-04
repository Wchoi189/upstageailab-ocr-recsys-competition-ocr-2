# ============================================================================
# CONFIGURATION SYSTEM MAINTENANCE
# ============================================================================

.PHONY: config-validate
config-validate:  ## Validate configuration system for errors
	python scripts/validate_config.py

.PHONY: config-archive
config-archive:   ## Archive legacy configs to .deprecated/
	mkdir -p configs/.deprecated/schemas configs/.deprecated/benchmark configs/.deprecated/tools
	mv configs/schemas/*.yaml configs/.deprecated/schemas/ 2>/dev/null || true
	mv configs/benchmark/*.yaml configs/.deprecated/benchmark/ 2>/dev/null || true
	mv configs/tools/* configs/.deprecated/tools/ 2>/dev/null || true
	@echo "✅ Legacy configs archived to configs/.deprecated/"

.PHONY: config-show-structure
config-show-structure:  ## Show current config structure
	find configs -name "*.yaml" -type f | wc -l | xargs echo "Total config files:"
	find configs -mindepth 1 -maxdepth 1 -type d | wc -l | xargs echo "Config groups:"
	grep -r "@package" configs --include="*.yaml" | cut -d: -f2 | sort | uniq -c | sort -rn
# Makefile for OCR Project Development
# Last Updated: 2025-10-21
# Version: 0.1.1
# Changes: Reorganized structure, eliminated duplication, improved help system

PORT ?= 8501
FRONTEND_DIR := apps/frontend
FRONTEND_HOST ?= 0.0.0.0
FRONTEND_PORT ?= 5173
BACKEND_HOST ?= 127.0.0.1
BACKEND_PORT ?= 8000
BACKEND_APP ?= apps.backend.services.playground_api.app:app

# UI Apps
UI_APPS := command_builder evaluation_viewer inference preprocessing_viewer resource_monitor unified_app

.PHONY: help install dev-install test test-cov lint lint-fix format quality-check quality-fix clean docs-build docs-serve docs-deploy diagrams-check diagrams-update diagrams-force-update diagrams-validate diagrams-update-specific serve-% stop-% status-% logs-% clear-logs-% list-ui-processes stop-all-ui pre-commit setup-dev ci frontend-ci console-ci context-log-start context-log-summarize quick-fix-log start stop cb eval infer prep monitor ua stop-cb stop-eval stop-infer stop-prep stop-monitor stop-ua frontend-dev frontend-stop fe sfe console-dev console-build console-lint backend-dev backend-stop backend-force-kill stack-dev stack-stop fs stop-fs checkpoint-metadata checkpoint-metadata-dry-run checkpoint-index-rebuild checkpoint-index-rebuild-all checkpoint-index-verify qms-plan qms-bug qms-validate qms-compliance qms-boundary qms-context qms-context-dev qms-context-docs qms-context-debug qms-context-plan

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
	@echo "  backend-force-kill - Force kill ANY process using port $(BACKEND_PORT) (last resort)"
	@echo "  fs                 - Run backend + frontend together (alias for stack-dev)"
	@echo "  stack-dev          - Combined dev stack (kills backend when frontend exits)"
	@echo "  stop-fs            - Stop combined stack (alias for stack-stop)"
	@echo ""
	@echo "📦 CHECKPOINT MANAGEMENT"
	@echo "  checkpoint-metadata  - Generate metadata files for all checkpoints (speeds up loading)"
	@echo "  checkpoint-metadata-dry-run - Preview metadata generation without creating files"
	@echo "  checkpoint-index-rebuild - Rebuild checkpoint index from file system"
	@echo "  checkpoint-index-rebuild-all - Rebuild all indices including legacy runs"
	@echo "  checkpoint-index-verify - Verify checkpoint index integrity"
	@echo ""
	@echo "🔧 DEVELOPMENT WORKFLOW"
	@echo "  clean               - Clean up cache files and build artifacts"
	@echo "  ci                  - Run CI checks (quality + tests)"
	@echo "  context-log-start   - Create context log (use LABEL=...)"
	@echo "  context-log-summarize - Summarize context log (use LOG=...)"
	@echo "  quick-fix-log       - Log quick fix (use TYPE= TITLE= ISSUE= FIX= FILES=)"
	@echo ""
	@echo "🧩 AgentQMS (Quality Management)"
	@echo "  qms-plan            - Create implementation plan artifact"
	@echo "  qms-bug             - Create bug report artifact"
	@echo "  qms-validate        - Validate all QMS artifacts"
	@echo "  qms-compliance      - Run full QMS compliance checks"
	@echo "  qms-boundary        - Validate AgentQMS/project boundaries"
	@echo "  qms-context         - Generate task-specific context bundle (use TASK=...)"
	@echo "  qms-context-dev     - Load development context bundle"
	@echo "  qms-context-docs    - Load documentation context bundle"
	@echo "  qms-context-debug   - Load debugging context bundle"
	@echo "  qms-context-plan    - Load planning context bundle"

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
	PARENT_PIDS=""; \
	\
	# Find processes listening on the port \
	if command -v lsof >/dev/null 2>&1; then \
		for PID in $$(lsof -t -i:$$PORT 2>/dev/null || true); do \
			CMD=$$(ps -p $$PID -o args= 2>/dev/null || true); \
			if echo "$$CMD" | grep -qi "vite"; then \
				PIDS="$$PIDS $$PID"; \
				# Find parent processes (uv run make, make, npm, etc.) \
				PPID=$$(ps -p $$PID -o ppid= 2>/dev/null | tr -d ' ' || true); \
				if [ -n "$$PPID" ] && [ "$$PPID" != "1" ]; then \
					PCMD=$$(ps -p $$PPID -o args= 2>/dev/null || true); \
					if echo "$$PCMD" | grep -Eq "uv run.*make.*fe|make.*frontend-dev|npm.*dev.*--port $$PORT"; then \
						PARENT_PIDS="$$PARENT_PIDS $$PPID"; \
					fi; \
				fi; \
			fi; \
		done; \
	fi; \
	\
	# Also find by process name pattern (fallback) \
	if command -v pgrep >/dev/null 2>&1; then \
		for PID in $$(pgrep -f "vite.*--port $$PORT|vite.*--host.*--port $$PORT" 2>/dev/null || true); do \
			if ! echo "$$PIDS" | grep -q "$$PID"; then \
				PIDS="$$PIDS $$PID"; \
			fi; \
		done; \
		# Find uv run make fe wrappers \
		for PID in $$(pgrep -f "uv run.*make.*fe|uv run.*make.*frontend-dev" 2>/dev/null || true); do \
			if ! echo "$$PARENT_PIDS" | grep -q "$$PID"; then \
				PARENT_PIDS="$$PARENT_PIDS $$PID"; \
			fi; \
		done; \
	fi; \
	\
	# Kill all found processes (children first, then parents) \
	if [ -n "$$PIDS" ] || [ -n "$$PARENT_PIDS" ]; then \
		echo "Stopping project frontend dev server on port $$PORT"; \
		if [ -n "$$PIDS" ]; then \
			echo "  Killing server processes (PID(s) $$PIDS)"; \
			kill $$PIDS 2>/dev/null || true; \
			sleep 0.5; \
		fi; \
		if [ -n "$$PARENT_PIDS" ]; then \
			echo "  Killing parent wrapper processes (PID(s) $$PARENT_PIDS)"; \
			kill $$PARENT_PIDS 2>/dev/null || true; \
		fi; \
		# Force kill if still running after 1 second \
		sleep 0.5; \
		if [ -n "$$PIDS" ]; then \
			for PID in $$PIDS; do \
				if ps -p $$PID >/dev/null 2>&1; then \
					echo "  Force killing PID $$PID"; \
					kill -9 $$PID 2>/dev/null || true; \
				fi; \
			done; \
		fi; \
	else \
		echo "No project frontend dev server detected on port $$PORT"; \
	fi

backend-dev:
	uv run uvicorn $(BACKEND_APP) --host $(BACKEND_HOST) --port $(BACKEND_PORT) --reload

backend-stop:
	@PORT=$(BACKEND_PORT); \
	PIDS=""; \
	PARENT_PIDS=""; \
	APP_PATTERN="$(BACKEND_APP)"; \
	CURRENT_PID=$$$$; \
	EXCLUDE_PIDS="$$CURRENT_PID"; \
	\
	# Build exclusion list: current process and all ancestors (up to 5 levels) \
	PPID=$$CURRENT_PID; \
	for i in 1 2 3 4 5; do \
		PPID=$$(ps -p $$PPID -o ppid= 2>/dev/null | tr -d ' ' || echo ""); \
		if [ -z "$$PPID" ] || [ "$$PPID" = "1" ]; then \
			break; \
		fi; \
		EXCLUDE_PIDS="$$EXCLUDE_PIDS $$PPID"; \
	done; \
	\
	# Find processes listening on the port \
	if command -v lsof >/dev/null 2>&1; then \
		for PID in $$(lsof -t -i:$$PORT 2>/dev/null || true); do \
			# Skip if PID is in exclusion list \
			if echo "$$EXCLUDE_PIDS" | grep -qw "$$PID"; then \
				continue; \
			fi; \
			CMD=$$(ps -p $$PID -o args= 2>/dev/null || true); \
			if echo "$$CMD" | grep -Eq "uvicorn.*$$APP_PATTERN|run_spa\.py|run_ui\.py|mkdocs"; then \
				PIDS="$$PIDS $$PID"; \
				# Find parent processes (uv run make, make, etc.) \
				PPID=$$(ps -p $$PID -o ppid= 2>/dev/null | tr -d ' ' || true); \
				if [ -n "$$PPID" ] && [ "$$PPID" != "1" ] && ! echo "$$EXCLUDE_PIDS" | grep -qw "$$PPID"; then \
					PCMD=$$(ps -p $$PPID -o args= 2>/dev/null || true); \
					if echo "$$PCMD" | grep -Eq "uv run.*make.*backend|make.*backend-dev|make.*docs-serve|uv run.*uvicorn|uv run.*mkdocs"; then \
						PARENT_PIDS="$$PARENT_PIDS $$PPID"; \
					fi; \
				fi; \
			fi; \
		done; \
	fi; \
	\
	# Also find by process name pattern (fallback if lsof fails or misses processes) \
	if command -v pgrep >/dev/null 2>&1; then \
		for PID in $$(pgrep -f "uvicorn.*$$APP_PATTERN" 2>/dev/null || true); do \
			if ! echo "$$EXCLUDE_PIDS" | grep -qw "$$PID" && ! echo "$$PIDS" | grep -qw "$$PID"; then \
				PIDS="$$PIDS $$PID"; \
			fi; \
		done; \
		# Find mkdocs processes (also uses port 8000) \
		for PID in $$(pgrep -f "mkdocs.*serve.*8000" 2>/dev/null || true); do \
			if ! echo "$$EXCLUDE_PIDS" | grep -qw "$$PID" && ! echo "$$PIDS" | grep -qw "$$PID"; then \
				PIDS="$$PIDS $$PID"; \
			fi; \
		done; \
		# Find uv run make backend-dev wrappers \
		for PID in $$(pgrep -f "uv run.*make.*backend-dev" 2>/dev/null || true); do \
			if ! echo "$$EXCLUDE_PIDS" | grep -qw "$$PID" && ! echo "$$PARENT_PIDS" | grep -qw "$$PID"; then \
				PARENT_PIDS="$$PARENT_PIDS $$PID"; \
			fi; \
		done; \
	fi; \
	\
	# Kill all found processes (children first, then parents) \
	if [ -n "$$PIDS" ] || [ -n "$$PARENT_PIDS" ]; then \
		echo "Stopping project backend server on port $$PORT"; \
		if [ -n "$$PIDS" ]; then \
			echo "  Killing server processes (PID(s) $$PIDS)"; \
			for PID in $$PIDS; do \
				if ps -p $$PID >/dev/null 2>&1; then \
					kill $$PID 2>/dev/null && echo "    Sent TERM to PID $$PID" || echo "    Failed to kill PID $$PID"; \
				fi; \
			done; \
			sleep 0.5; \
		fi; \
		if [ -n "$$PARENT_PIDS" ]; then \
			echo "  Killing parent wrapper processes (PID(s) $$PARENT_PIDS)"; \
			for PID in $$PARENT_PIDS; do \
				if ps -p $$PID >/dev/null 2>&1; then \
					kill $$PID 2>/dev/null && echo "    Sent TERM to parent PID $$PID" || echo "    Failed to kill parent PID $$PID"; \
				fi; \
			done; \
		fi; \
		# Force kill if still running after 1 second \
		sleep 0.5; \
		STILL_RUNNING=""; \
		if [ -n "$$PIDS" ]; then \
			for PID in $$PIDS; do \
				if ps -p $$PID >/dev/null 2>&1; then \
					echo "  Force killing PID $$PID"; \
					kill -9 $$PID 2>/dev/null || true; \
					STILL_RUNNING="$$STILL_RUNNING $$PID"; \
				fi; \
			done; \
		fi; \
		if [ -n "$$PARENT_PIDS" ]; then \
			for PID in $$PARENT_PIDS; do \
				if ps -p $$PID >/dev/null 2>&1; then \
					echo "  Force killing parent PID $$PID"; \
					kill -9 $$PID 2>/dev/null || true; \
					STILL_RUNNING="$$STILL_RUNNING $$PID"; \
				fi; \
			done; \
		fi; \
		if [ -n "$$STILL_RUNNING" ]; then \
			echo "  Warning: Some processes may still be running: $$STILL_RUNNING"; \
		else \
			echo "  All processes stopped successfully"; \
		fi; \
	else \
		echo "No project backend server detected on port $$PORT"; \
	fi

fs: stack-dev

stack-dev:
	@bash -c 'set -euo pipefail; \
		echo "Starting FastAPI backend on $(BACKEND_HOST):$(BACKEND_PORT)"; \
		uv run uvicorn $(BACKEND_APP) --host $(BACKEND_HOST) --port $(BACKEND_PORT) --reload & \
		BACK_PID=$$!; \
		trap "echo \"Stopping backend (PID $$BACK_PID)\"; kill $$BACK_PID 2>/dev/null || true; $(MAKE) backend-stop 2>/dev/null || true" EXIT INT TERM; \
		echo "Backend started (PID $$BACK_PID), waiting for port $(BACKEND_PORT) to be ready..."; \
		MAX_WAIT=30; \
		WAITED=0; \
		while [ $$WAITED -lt $$MAX_WAIT ]; do \
			if command -v lsof >/dev/null 2>&1; then \
				if lsof -i:$(BACKEND_PORT) >/dev/null 2>&1; then \
					echo "Backend is ready on port $(BACKEND_PORT)"; \
					break; \
				fi; \
			elif command -v nc >/dev/null 2>&1; then \
				if nc -z $(BACKEND_HOST) $(BACKEND_PORT) 2>/dev/null; then \
					echo "Backend is ready on port $(BACKEND_PORT)"; \
					break; \
				fi; \
			else \
				# Fallback: just wait a bit longer \
				sleep 3; \
				break; \
			fi; \
			sleep 0.5; \
			WAITED=$$((WAITED + 1)); \
		done; \
		if [ $$WAITED -ge $$MAX_WAIT ]; then \
			echo "Warning: Backend may not be ready yet, but starting frontend anyway"; \
		fi; \
		cd $(FRONTEND_DIR); \
		echo "Starting frontend dev server on $(FRONTEND_HOST):$(FRONTEND_PORT)"; \
		npm run dev -- --host $(FRONTEND_HOST) --port $(FRONTEND_PORT); \
	'

sfs: stack-stop

stack-stop: backend-stop
	$(MAKE) sfe

cd: cleanup-dev
# Force cleanup of any orphaned backend/frontend processes (use when stop targets fail)
cleanup-dev:
	@trap '' TERM INT; \
	echo "Force cleaning up all dev processes..."; \
	$(MAKE) backend-stop 2>&1 || true; \
	$(MAKE) sfe 2>&1 || true; \
	if command -v pkill >/dev/null 2>&1; then \
		CURRENT_PID=$$$$; \
		for PID in $$(pgrep -f "uv run.*make.*backend-dev|uv run.*make.*fe|uv run.*make.*frontend-dev" 2>/dev/null || true); do \
			if [ "$$PID" != "$$CURRENT_PID" ]; then \
				pkill -P $$PID 2>/dev/null || true; \
				kill $$PID 2>/dev/null || true; \
			fi; \
		done; \
	fi; \
	echo "Cleanup complete."

# Force kill ANY process using port 8000 (use as last resort)
backend-force-kill:
	@PORT=$(BACKEND_PORT); \
	echo "Force killing ANY process using port $$PORT..."; \
	if command -v lsof >/dev/null 2>&1; then \
		for PID in $$(lsof -t -i:$$PORT 2>/dev/null || true); do \
			echo "  Killing PID $$PID"; \
			kill -9 $$PID 2>/dev/null || true; \
		done; \
		sleep 1; \
		# Check if port is still in use \
		if lsof -t -i:$$PORT >/dev/null 2>&1; then \
			echo "  Warning: Port $$PORT may still be in use (could be TIME_WAIT state)"; \
			echo "  Waiting 5 seconds for TIME_WAIT sockets to clear..."; \
			sleep 5; \
		else \
			echo "  Port $$PORT is now free"; \
		fi; \
	else \
		echo "  lsof not available, trying alternative methods..."; \
		if command -v fuser >/dev/null 2>&1; then \
			fuser -k $$PORT/tcp 2>/dev/null || true; \
		fi; \
	fi

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
# CHECKPOINT MANAGEMENT
# ============================================================================

checkpoint-metadata:
	@echo "Generating metadata files for all checkpoints..."
	@echo "This will speed up checkpoint catalog loading significantly."
	uv run python scripts/checkpoints/generate_metadata.py

checkpoint-metadata-dry-run:
	@echo "Previewing metadata generation (dry run)..."
	uv run python scripts/checkpoints/generate_metadata.py --dry-run

checkpoint-index-rebuild:
	@echo "Rebuilding checkpoint index from file system..."
	@echo "This scans outputs/ directory and creates a fast lookup index."
	uv run python -c "from pathlib import Path; from ocr.utils.checkpoints.index import CheckpointIndex; import time; outputs_dir = Path('outputs'); index = CheckpointIndex(outputs_dir, include_legacy=True); print(f'Rebuilding index from {outputs_dir}...'); start = time.time(); index.rebuild(); duration = time.time() - start; print(f'✅ Index rebuilt in {duration:.2f}s'); print(f'Total checkpoints indexed: {len(index.get_checkpoint_paths())}')"

checkpoint-index-rebuild-all:
	@echo "Rebuilding ALL checkpoint indices (including legacy runs)..."
	uv run python -c "from pathlib import Path; from ocr.utils.checkpoints.index import CheckpointIndex; import time; outputs_dir = Path('outputs'); index = CheckpointIndex(outputs_dir, include_legacy=True); print(f'Full rebuild with legacy runs...'); start = time.time(); index.rebuild(); duration = time.time() - start; runs = index.get_runs(); print(f'✅ Index rebuilt in {duration:.2f}s'); print(f'Total checkpoints: {len(index.get_checkpoint_paths())}'); print(f'Total runs: {len(runs)}'); [print(f'  - {run_name}: {len(run_data.get(\"checkpoints\", []))} checkpoints') for run_name, run_data in runs.items()]"

checkpoint-index-verify:
	@echo "Verifying checkpoint index integrity..."
	uv run python -c "from pathlib import Path; from ocr.utils.checkpoints.index import CheckpointIndex; outputs_dir = Path('outputs'); index = CheckpointIndex(outputs_dir, include_legacy=True); index._load_index(); checkpoints = index.get_checkpoint_paths(); print(f'✅ Index verification passed'); print(f'Index file: {index.index_file}'); print(f'Index size: {index.index_file.stat().st_size} bytes'); print(f'Checkpoints indexed: {len(checkpoints)}')"

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

# ============================================================================
# AgentQMS – Quality Management Shortcuts
# ============================================================================

qms-plan:
	@cd AgentQMS/interface && make create-plan NAME=$(if $(NAME),$(NAME),my-plan) TITLE="$(if $(TITLE),$(TITLE),Implementation Plan)"

qms-bug:
	@cd AgentQMS/interface && make create-bug-report NAME=$(if $(NAME),$(NAME),my-bug) TITLE="$(if $(TITLE),$(TITLE),Bug Report)"

qms-validate:
	@cd AgentQMS/interface && make validate

qms-compliance:
	@cd AgentQMS/interface && make compliance

qms-boundary:
	@cd AgentQMS/interface && make boundary

qms-context:
	@cd AgentQMS/interface && make context $(if $(TASK),TASK="$(TASK)",)

qms-context-dev:
	@cd AgentQMS/interface && make context-development

qms-context-docs:
	@cd AgentQMS/interface && make context-docs

qms-context-debug:
	@cd AgentQMS/interface && make context-debug

qms-context-plan:
	@cd AgentQMS/interface && make context-plan
