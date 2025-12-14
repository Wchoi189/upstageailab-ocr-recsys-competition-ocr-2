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

.PHONY: profile-imports
profile-imports:  ## Profile import times to identify startup bottlenecks
	@echo "⏱️  Profiling import times (this will take ~90s)..."
	python scripts/profile_imports.py

.PHONY: analyze-imports
analyze-imports:  ## Analyze import structure to identify heavy dependencies
	python scripts/analyze_imports.py

.PHONY: benchmark-startup
benchmark-startup:  ## Benchmark startup times for training scripts
	@echo "Benchmarking train.py (monolithic imports)..."
	@/usr/bin/time -f "Time: %E (user: %U, sys: %S)" python -c "import runners.train" 2>&1 | grep "Time:"
	@echo ""
	@echo "Benchmarking train_fast.py (lazy imports)..."
	@/usr/bin/time -f "Time: %E (user: %U, sys: %S)" python -c "import runners.train_fast" 2>&1 | grep "Time:"

.PHONY: test-config-validation
test-config-validation:  ## Test config validation speed with fast entry point
	@echo "Testing config validation with train_fast.py..."
	@/usr/bin/time -f "⏱️  Total time: %E" python runners/train_fast.py validate_only=true

.PHONY: test-fast-train
test-fast-train:  ## Run quick training test with optimized entry point
	python runners/train_fast.py \
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

.PHONY: help install dev-install test test-cov lint lint-fix format quality-check quality-fix clean docs-build docs-serve docs-deploy diagrams-check diagrams-update diagrams-force-update diagrams-validate diagrams-update-specific serve-% stop-% status-% logs-% clear-logs-% list-ui-processes stop-all-ui pre-commit setup-dev ci frontend-ci console-ci context-log-start context-log-summarize quick-fix-log start stop cb eval infer prep monitor ua stop-cb stop-eval stop-infer stop-prep stop-monitor stop-ua console-dev console-build console-lint checkpoint-metadata checkpoint-metadata-dry-run checkpoint-index-rebuild checkpoint-index-rebuild-all checkpoint-index-verify qms-plan qms-bug qms-validate qms-compliance qms-boundary qms-context qms-context-dev qms-context-docs qms-context-debug qms-context-plan serve-ocr-console playground-console-dev kill-ports

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
	@echo "🧭 NEXT.JS CONSOLE"
	@echo "  console-dev        - Start Next.js App Router dev server (apps/playground-console)"
	@echo "  console-build      - Build the Next.js console for production deploys"
	@echo "  console-lint       - Run console linting (see docs/maintainers/coding_standards.md)"
	@echo ""
	@echo "🔍 OCR INFERENCE CONSOLE"
	@echo "  serve-ocr-console  - Start OCR Console frontend with auto-detected checkpoint"
	@echo "  ocr-console-dev    - Start just the OCR Console frontend (Vite dev server)"
	@echo ""
	@echo "🧩 APP SERVERS"
	@echo "  playground-console-dev - Start Playground Console (alias for console-dev)"
	@echo ""
	@echo "⚙️  PROCESS MANAGEMENT"
	@echo "  kill-ports         - Force kill processes on ports 3000, 5173, 8000 (use when servers hang)"
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

# Friendly aliases (Legacy removed)
# cb, eval, infer, prep, monitor, ua removed
# frontend-dev, fe, sfe, frontend-stop - removed (deprecated legacy Vite app)
# See apps/playground-console or apps/ocr-inference-console for current frontend apps

console-dev:
	npm run dev:console

console-build:
	npm run build:console

console-lint:
	npm run lint:console

ocr-console-dev:
	cd apps/ocr-inference-console && npm run dev -- --host $(FRONTEND_HOST) --port $(FRONTEND_PORT)

# backend-dev, backend-ocr - removed (deprecated unified backend)
# Use domain-specific backends instead:
#   - apps/playground-console/backend for playground-specific features
#   - apps/ocr-inference-console/backend for OCR inference
#   - apps/shared/backend_shared for shared InferenceEngine

# ============================================================================
# DOMAIN-DRIVEN APP SERVERS
# ============================================================================

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
		cd apps/ocr-inference-console; \
		npm run dev -- --host 0.0.0.0 --port 5173; \
	'

# Playground Console (Next.js App Router)
playground-console-dev:
	npm run dev:console

# Process Management Utilities
kill-ports:
	@echo "Killing processes on ports 3000, 5173, and 8000..."; \
	if command -v lsof >/dev/null 2>&1; then \
		for PORT in 3000 5173 8000; do \
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
			for PORT in 3000 5173 8000; do \
				echo "  Killing processes on port $$PORT..."; \
				fuser -k $$PORT/tcp 2>/dev/null || true; \
			done; \
		else \
			echo "  No suitable tool found (lsof/fuser)"; \
			exit 1; \
		fi; \
	fi

# ============================================================================
# UI APPLICATIONS (Legacy - Archived)
# ============================================================================

# Note: Streamlit apps have been archived to docs/archive/legacy_ui_code/
# Use 'frontend-dev' or 'console-dev' instead.

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

qms-validate-staged:
	@cd AgentQMS/interface && python ../agent_tools/compliance/validate_artifacts.py --staged

qms-validate-new:
	@cd AgentQMS/interface && python ../agent_tools/compliance/validate_artifacts.py --staged

qms-context-suggest:
	@cd AgentQMS/interface && python ../agent_tools/utilities/suggest_context.py $(if $(TASK),--task "$(TASK)",)

qms-plan-progress:
	@cd AgentQMS/interface && python ../agent_tools/utilities/plan_progress.py $(if $(PLAN),--plan "$(PLAN)",) $(if $(TASK),--task "$(TASK)",)

qms-migrate-legacy:
	@cd AgentQMS/interface && python ../agent_tools/utilities/legacy_migrator.py $(if $(LIMIT),--limit $(LIMIT),--limit 20) $(if $(DRY_RUN),--dry-run,) $(if $(AUTOFIX),--autofix,)

qms-tracking-repair:
	@cd AgentQMS/interface && python ../agent_tools/utilities/tracking_repair.py $(if $(DRY_RUN),--dry-run,)

# Wildcard pattern for extensibility: make agentqms-<target> delegates to AgentQMS/interface
agentqms-%:
	@cd AgentQMS/interface && make $*
