# ============================================================================
# CONFIGURATION SYSTEM MAINTENANCE
# ============================================================================

.PHONY: config-validate
config-validate:  ## Validate configuration system for errors
	@unset PYENV_VERSION && uv run python scripts/validate_config.py

.PHONY: config-archive
config-archive:   ## Archive legacy configs to .deprecated/
	mkdir -p configs/.deprecated/schemas configs/.deprecated/benchmark
	mv configs/schemas/*.yaml configs/.deprecated/schemas/ 2>/dev/null || true
	mv configs/benchmark/*.yaml configs/.deprecated/benchmark/ 2>/dev/null || true
	@echo "✅ Legacy configs archived to configs/.deprecated/"

.PHONY: config-show-structure
config-show-structure:  ## Show current config structure
	find configs -name "*.yaml" -type f | wc -l | xargs echo "Total config files:"
	find configs -mindepth 1 -maxdepth 1 -type d | wc -l | xargs echo "Config groups:"
	grep -r "@package" configs --include="*.yaml" | cut -d: -f2 | sort | uniq -c | sort -rn

.PHONY: profile-imports
profile-imports:  ## Profile import times to identify startup bottlenecks
	@echo "⏱️  Profiling import times (this will take ~90s)..."
	uv run python scripts/performance/profile_imports.py

.PHONY: analyze-imports
analyze-imports:  ## Analyze import structure to identify heavy dependencies
	uv run python scripts/performance/analyze_imports.py

.PHONY: benchmark-startup
benchmark-startup:  ## Benchmark startup times for training scripts
	@echo "Benchmarking train.py (monolithic imports)..."
	@/usr/bin/time -f "Time: %E (user: %U, sys: %S)" uv run python -c "import runners.train" 2>&1 | grep "Time:"
	@echo ""
	@echo "Benchmarking train_fast.py (lazy imports)..."
	@/usr/bin/time -f "Time: %E (user: %U, sys: %S)" uv run python -c "import runners.train_fast" 2>&1 | grep "Time:"

.PHONY: test-config-validation
test-config-validation:  ## Test config validation speed with fast entry point
	@echo "Testing config validation with train_fast.py..."
	@/usr/bin/time -f "⏱️  Total time: %E" uv run python runners/train_fast.py validate_only=true

.PHONY: test-fast-train
test-fast-train:  ## Run quick training test with optimized entry point
	uv run python runners/train_fast.py \
		trainer.max_epochs=1 \
		trainer.limit_train_batches=0.25 \
		trainer.limit_val_batches=0.25 \
		exp_name=fast_train_test
# Makefile for OCR Project Development
# Last Updated: 2025-10-21
# Version: 0.1.1
# Changes: Reorganized structure, eliminated duplication, improved help system

PORT ?= 8501
FRONTEND_HOST ?= 0.0.0.0
FRONTEND_PORT ?= 5173

.PHONY: help install dev-install test test-cov lint lint-fix format quality-check quality-fix clean docs-build docs-serve docs-deploy diagrams-check diagrams-update diagrams-force-update diagrams-validate diagrams-update-specific serve-% stop-% status-% logs-% clear-logs-% list-ui-processes stop-all-ui pre-commit setup-dev ci frontend-ci console-ci start stop cb eval infer prep monitor ua stop-cb stop-eval stop-infer stop-prep stop-monitor stop-ua console-dev console-build console-lint checkpoint-metadata checkpoint-metadata-dry-run checkpoint-index-rebuild checkpoint-index-rebuild-all checkpoint-index-verify qms-plan qms-bug qms-validate qms-compliance qms-boundary qms-context qms-context-dev qms-context-docs qms-context-debug qms-context-plan serve-ocr-console ocr-console-backend ocr-console-stack playground-console-dev playground-console-backend playground-console-stack kill-ports

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
	@echo "  lint-check-json     - Output ruff results as JSON for AI processing"
	@echo "  lint-fix-ai         - Run AI-powered linting fixes using Grok (requires XAI_API_KEY)"
	@echo "  lint-fix-ai-dry-run - Preview AI-powered fixes without applying"
	@echo "  format              - Format code with black and isort"
	@echo "  quality-check       - Run comprehensive code quality checks"
	@echo "  quality-fix         - Auto-fix code quality issues"
	@echo "  pre-commit          - Install and run pre-commit hooks"
	@echo "  pre-commit-install  - Install pre-commit hooks only"
	@echo "  pre-commit-run      - Run pre-commit on all files"
	@echo "  pre-commit-update   - Update pre-commit hook versions"
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
	@echo "🧭 NEXT.JS CONSOLE"
	@echo "  See 'playground-console-*' targets below."
	@echo ""
	@echo "🔍 OCR INFERENCE CONSOLE"
	@echo "  serve-ocr-console       - Start frontend only (port 5173)"
	@echo "  ocr-console-backend     - Start backend only (port 8002, requires checkpoint)"
	@echo "  ocr-console-stack       - Start backend + frontend together"
	@echo ""
	@echo "🧩 APP SERVERS"
	@echo "  playground-console-dev      - Start Playground Console frontend (Next.js, port 3000)"
	@echo "  playground-console-backend  - Start Playground Console backend (port 8001)"
	@echo "  playground-console-stack    - Start backend + frontend together"
	@echo ""
	@echo "⚙️  PROCESS MANAGEMENT"
	@echo "  kill-ports         - Force kill processes on ports 3000, 5173, 8000, 8001, 8002 (use when servers hang)"
	@echo ""
	@echo "ℹ️  DOMAIN-DRIVEN ARCHITECTURE"
	@echo "  Each app manages its own backend and is started independently."
	@echo "  Legacy unified backend/frontend have been archived."
	@echo "  See DEPRECATION_MANIFEST.md for migration details."
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

lint-check-json:  ## Output ruff results as JSON for AI processing
	uv run ruff check . --output-format=json

lint-fix-ai:  ## Run AI-powered linting fixes using Grok
	@echo "🤖 Running AI-powered linting fixes..."
	@uv run ruff check . --output-format=json > /tmp/lint_errors.json || true
	@ERROR_COUNT=$$(cat /tmp/lint_errors.json | uv run python -c "import sys, json; print(len(json.load(sys.stdin)))"); \
	if [ "$$ERROR_COUNT" -eq "0" ]; then \
		echo "✅ No linting errors found!"; \
	else \
		echo "Found $$ERROR_COUNT errors. Running Grok fixer..."; \
		uv run python AgentQMS/tools/utilities/grok_linter.py --input /tmp/lint_errors.json --limit 5 --verbose; \
	fi

lint-fix-ai-dry-run:  ## Preview AI-powered linting fixes without applying
	@echo "🤖 Running AI-powered linting fixes (DRY RUN)..."
	@uv run ruff check . --output-format=json > /tmp/lint_errors.json || true
	@uv run python AgentQMS/tools/utilities/grok_linter.py --input /tmp/lint_errors.json --dry-run --verbose

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

pre-commit-install:  ## Install pre-commit hooks
	pre-commit install
	@echo "✅ Pre-commit hooks installed"
	@echo "Run 'git commit' to trigger automatic checks"

pre-commit-run:  ## Run pre-commit on all files
	pre-commit run --all-files

pre-commit-update:  ## Update pre-commit hook versions
	pre-commit autoupdate

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
	uv run python scripts/generate_diagrams.py --check-changes

diagrams-update:
	@echo "🔄 Updating diagrams that have changed..."
	uv run python scripts/generate_diagrams.py --update

diagrams-force-update:
	@echo "🔄 Force updating all diagrams..."
	uv run python scripts/generate_diagrams.py --update --force

diagrams-validate:
	@echo "✅ Validating diagram syntax..."
	uv run python scripts/generate_diagrams.py --validate

diagrams-update-specific:
	@echo "🔄 Updating specific diagrams: $(DIAGRAMS)"
	uv run python scripts/generate_diagrams.py --update $(DIAGRAMS)

# ============================================================================
# DOMAIN-DRIVEN APP SERVERS
# ============================================================================

# These are the canonical targets for running the applications.
# Legacy console-* and ocr-console-dev targets have been removed to avoid confusion.

# OCR Inference Console with auto-detected checkpoint
serve-ocr-console:
	@bash -c 'set -euo pipefail; \
		export OCR_CHECKPOINT_PATH=$$(find outputs/experiments/train/ocr -name "*.ckpt" 2>/dev/null | head -n 1); \
		if [ -z "$$OCR_CHECKPOINT_PATH" ]; then \
			echo "⚠️  No OCR checkpoint found in outputs/experiments/train/ocr/"; \
			echo "   Set OCR_CHECKPOINT_PATH manually if needed."; \
			echo "   Proceeding with OCR Console frontend only..."; \
		else \
			echo "✅ Auto-detected checkpoint: $$OCR_CHECKPOINT_PATH"; \
		fi; \
		echo "Starting OCR Inference Console on http://localhost:5173"; \
		echo "Using VITE_API_URL=$${VITE_API_URL:-http://127.0.0.1:8002/api}"; \
		cd apps/ocr_inference_console; \
		VITE_API_URL=$${VITE_API_URL:-http://127.0.0.1:8002/api} npm run dev -- --host 0.0.0.0 --port 5173; \
	'

# OCR Console backend server (requires checkpoint)
ocr-console-backend:
	@bash -c 'set -euo pipefail; \
		export OCR_CHECKPOINT_PATH=$$(find outputs/experiments/train/ocr -name "*.ckpt" 2>/dev/null | head -n 1); \
		if [ -z "$$OCR_CHECKPOINT_PATH" ]; then \
			echo "❌ Error: No OCR checkpoint found in outputs/experiments/train/ocr/"; \
			echo "   Please train a model or set OCR_CHECKPOINT_PATH manually."; \
			exit 1; \
		fi; \
		echo "✅ Using checkpoint: $$OCR_CHECKPOINT_PATH"; \
		echo "🚀 Starting OCR Console backend on http://0.0.0.0:8002"; \
		echo "   API Docs: http://0.0.0.0:8002/docs"; \
		uv run uvicorn apps.ocr_inference_console.backend.main:app --host 0.0.0.0 --port 8002 --reload --reload-dir apps/ocr_inference_console/backend; \
	'

# OCR Console full stack (backend + frontend)
ocr-console-stack:
	@bash -c 'set -euo pipefail; \
		export OCR_CHECKPOINT_PATH=$$(find outputs/experiments/train/ocr -name "*.ckpt" 2>/dev/null | head -n 1); \
		if [ -z "$$OCR_CHECKPOINT_PATH" ]; then \
			echo "⚠️  No OCR checkpoint found. Backend will be unavailable."; \
			echo "   Start frontend only with: make serve-ocr-console"; \
		else \
			echo "✅ Starting backend with checkpoint: $$OCR_CHECKPOINT_PATH"; \
			cd apps/ocr_inference_console/backend && \
			uv run uvicorn main:app --host 127.0.0.1 --port 8002 --reload &\
			BACK_PID=$$!; \
			trap "echo Killing backend; kill $$BACK_PID 2>/dev/null || true" EXIT INT TERM; \
			sleep 2; \
		fi; \
		echo "🌐 Starting frontend on http://localhost:5173"; \
		echo "Using VITE_API_URL=$${VITE_API_URL:-http://127.0.0.1:8002/api}"; \
		cd apps/ocr-inference-console && \
		VITE_API_URL=$${VITE_API_URL:-http://127.0.0.1:8002/api} npm run dev -- --host 0.0.0.0 --port 5173; \
	'

# Playground Console (Next.js App Router)
playground-console-dev:
	npm run dev:console

# Playground Console backend server (port 8001)
playground-console-backend:
	@bash -c 'set -euo pipefail; \
		export OCR_CHECKPOINT_PATH=$$(find outputs/experiments/train/ocr -name "*.ckpt" 2>/dev/null | head -n 1); \
		if [ -z "$$OCR_CHECKPOINT_PATH" ]; then \
			echo "⚠️  Warning: No OCR checkpoint found in outputs/experiments/train/ocr/"; \
			echo "   Backend will start but inference may fail without a checkpoint."; \
		else \
			echo "✅ Found checkpoint: $$OCR_CHECKPOINT_PATH"; \
		fi; \
		echo "🚀 Starting Playground Console backend on http://127.0.0.1:8001"; \
		echo "   API Docs: http://127.0.0.1:8001/docs"; \
		uv run uvicorn apps.playground-console.backend.main:app --host 127.0.0.1 --port 8001 --reload --reload-dir apps/playground-console/backend; \
	'

# Playground Console full stack (backend + frontend)
playground-console-stack:
	@bash -c 'set -euo pipefail; \
		export OCR_CHECKPOINT_PATH=$$(find outputs/experiments/train/ocr -name "*.ckpt" 2>/dev/null | head -n 1); \
		if [ -z "$$OCR_CHECKPOINT_PATH" ]; then \
			echo "⚠️  No OCR checkpoint found. Backend will be unavailable."; \
		else \
			echo "✅ Starting backend with checkpoint: $$OCR_CHECKPOINT_PATH"; \
			cd apps/playground-console/backend && \
			uv run uvicorn main:app --host 127.0.0.1 --port 8001 --reload & \
			BACK_PID=$$!; \
			trap "echo Killing backend; kill $$BACK_PID 2>/dev/null || true" EXIT INT TERM; \
			sleep 2; \
		fi; \
		echo "🌐 Starting frontend on http://localhost:3000"; \
		npm run dev:console; \
	'

# Process Management Utilities
kill-ports:
	@echo "Killing processes on ports 3000, 5173, 8000, 8001, and 8002..."; \
	if command -v lsof >/dev/null 2>&1; then \
		for PORT in 3000 5173 8000 8001 8002; do \
			PIDS=$$(lsof -t -i:$$PORT 2>/dev/null || true); \
			if [ -n "$$PIDS" ]; then \
				echo "  Killing processes on port $$PORT: $$PIDS"; \
				kill -9 $$PIDS 2>/dev/null || true; \
			else \
				echo "  No process on port $$PORT"; \
			fi; \
		done; \
		echo "✅ Done"; \
	else \
		echo "  lsof not available, trying fuser..."; \
		if command -v fuser >/dev/null 2>&1; then \
			for PORT in 3000 5173 8000 8001 8002; do \
				echo "  Killing processes on port $$PORT..."; \
				fuser -k $$PORT/tcp 2>/dev/null || true; \
			done; \
		else \
			echo "  No suitable tool found (lsof/fuser)"; \
			exit 1; \
		fi; \
	fi

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

refresh-mcp:  ## Refresh MCP Tool definitions from sub-packages
	PYTHONPATH=.::agent-debug-toolkit/src uv run python scripts/mcp/refresh_tools.py

# ============================================================================
# CI SIMULATION
# ============================================================================

ci: quality-check frontend-ci console-ci test
	@echo "CI checks passed! ✅"

frontend-ci:
	npm run lint:spa
	npm run build:spa

console-ci:
	cd apps/playground-console && npm run lint
	cd apps/playground-console && npm run build

# ============================================================================
# AgentQMS – Quality Management Shortcuts
# ============================================================================

qms-plan:
	@make -C AgentQMS/bin create-plan NAME=$(if $(NAME),$(NAME),my-plan) TITLE="$(if $(TITLE),$(TITLE),Implementation Plan)"

qms-bug:
	@make -C AgentQMS/bin create-bug-report NAME=$(if $(NAME),$(NAME),my-bug) TITLE="$(if $(TITLE),$(TITLE),Bug Report)"

debug-session:
	@uv run python AgentQMS/tools/utilities/init_debug_session.py \
		--id $(BUG_ID) --title "$(TITLE)" $(if $(SEVERITY),--severity $(SEVERITY),)

qms-validate:
	@make -C AgentQMS/bin validate

qms-compliance:
	@make -C AgentQMS/bin compliance

qms-boundary:
	@make -C AgentQMS/bin boundary

qms-context:
	@make -C AgentQMS/bin context $(if $(TASK),TASK="$(TASK)",)

qms-context-dev:
	@make -C AgentQMS/bin context-development

qms-context-docs:
	@make -C AgentQMS/bin context-docs

qms-context-debug:
	@make -C AgentQMS/bin context-debug

qms-context-plan:
	@make -C AgentQMS/bin context-plan

qms-validate-staged:
	@uv run python AgentQMS/tools/compliance/validate_artifacts.py --staged

qms-validate-new:
	@uv run python AgentQMS/tools/compliance/validate_artifacts.py --staged

qms-context-suggest:
	@uv run python AgentQMS/tools/utilities/suggest_context.py $(if $(TASK),--task "$(TASK)",)

qms-plan-progress:
	@uv run python AgentQMS/tools/utilities/plan_progress.py $(if $(PLAN),--plan "$(PLAN)",) $(if $(TASK),--task "$(TASK)",)

qms-migrate-legacy:
	@uv run python AgentQMS/tools/utilities/legacy_migrator.py $(if $(LIMIT),--limit $(LIMIT),--limit 20) $(if $(DRY_RUN),--dry-run,) $(if $(AUTOFIX),--autofix,)

qms-tracking-repair:
	@uv run python AgentQMS/tools/utilities/tracking_repair.py $(if $(DRY_RUN),--dry-run,)

# Wildcard pattern for extensibility: make agentqms-<target> delegates to AgentQMS/bin
agentqms-%:
	@make -C AgentQMS/bin $*
