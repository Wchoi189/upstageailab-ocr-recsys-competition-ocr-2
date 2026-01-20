# Agent-Exclusive Makefile
# This Makefile is ONLY for AI agents - humans should not use these commands
# For human development, use the main project Makefile

.PHONY: help discover tools status clean validate compliance docs feedback quality audit audit-framework audit-framework-verbose audit-framework-json audit-framework-plugins audit-framework-templates artifacts-status artifacts-status-dashboard artifacts-status-compact artifacts-status-aging artifacts-status-json artifacts-status-threshold

export PYTHONPATH := $(abspath ../..):$(PYTHONPATH)

# Timestamp (KST) shown in help for visibility
LAST_UPDATED := $(shell TZ=Asia/Seoul date +%Y-%m-%dT%H:%M:%S%z)

# Default target - show help
help: ## Show agent help message
	@echo "🤖 Agent Tools Makefile (AGENT-ONLY)"
	@echo "====================================="
	@echo ""
	@echo "⚠️  WARNING: This Makefile is for AI agents only!"
	@echo "   Humans should use the main project Makefile."
	@echo ""
	@echo "Last Updated (KST): $(LAST_UPDATED)"
	@echo ""
	@echo "📋 Available Agent Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "💡 Quick Start for Agents:"
	@echo "  make discover    - List all available tools"
	@echo "  make status      - Check system status"
	@echo "  make validate    - Validate all artifacts"
	@echo "  make compliance   - Check compliance status"
	@echo ""

# Discovery and Help
discover: ## Discover all available agent tools
	uv run python ../tools/core/discover.py

tools: discover ## Alias for discover

status: ## Check overall system status
	@echo "🔍 Checking Agent Tools Status..."
	@echo ""
	@echo "📁 Tool Organization:"
	@ls -la ../tools/ | grep "^d" | grep -v "__pycache__" | awk '{print "  " $$9}'
	@echo ""
	@echo "🛠️ Agent Wrapper Scripts:"
	@ls -la ../*.sh 2>/dev/null | awk '{print "  " $$9}' || echo "  No wrapper scripts found"
	@echo ""
	@echo "📊 Recent Activity:"
	@find ../../artifacts -name "*.md" -mtime -1 2>/dev/null | head -5 | awk '{print "  " $$0}' || echo "  No recent artifacts"

# Artifact Management
create-plan: ## Create implementation plan (usage: make create-plan NAME=my-plan TITLE="My Plan")
	qms artifact create --type implementation_plan --name $(NAME) --title "$(TITLE)"

create-assessment: ## Create assessment (usage: make create-assessment NAME=my-assessment TITLE="My Assessment")
	qms artifact create --type assessment --name $(NAME) --title "$(TITLE)"

create-audit: ## Create audit (usage: make create-audit NAME=my-audit TITLE="My Audit")
	qms artifact create --type audit --name $(NAME) --title "$(TITLE)"

create-design: ## Create design document (canonical: design_document) (usage: make create-design NAME=my-design TITLE="My Design")
	qms artifact create --type design_document --name $(NAME) --title "$(TITLE)"

create-research: ## Create research document (mapped to assessment) (usage: make create-research NAME=my-research TITLE="My Research")
	@echo "ℹ️  'research' is not a canonical artifact type; mapping to 'assessment'."
	qms artifact create --type assessment --name $(NAME) --title "$(TITLE)"

create-template: ## Deprecated – use docs/_templates for reusable templates
	@echo "⚠️  'template' artifact type is deprecated. Place reusable templates under docs/_templates/."
	@echo "   For new documentation, prefer canonical types (assessment, design_document, implementation_plan, etc.)."

create-bug-report: ## Create bug report (usage: make create-bug-report NAME=my-bug TITLE="My Bug Report")
	qms artifact create --type bug_report --name $(NAME) --title "$(TITLE)"

# Smart Auto-Population for Artifacts
smart-suggest-metadata: ## Suggest metadata for artifact (usage: make smart-suggest-metadata TYPE=implementation_plan)
	@if [ -z "$(TYPE)" ]; then \
		echo "ERROR: TYPE parameter required. Usage: make smart-suggest-metadata TYPE=implementation_plan"; \
		exit 1; \
	fi
	uv run python ../tools/utilities/smart_populate.py suggest-metadata --type "$(TYPE)" --json

smart-suggest-tags: ## Suggest tags for artifact type (usage: make smart-suggest-tags TYPE=assessment)
	@if [ -z "$(TYPE)" ]; then \
		echo "ERROR: TYPE parameter required."; \
		exit 1; \
	fi
	uv run python ../tools/utilities/smart_populate.py suggest-tags --type "$(TYPE)"

smart-suggest-files: ## Suggest related files (usage: make smart-suggest-files TYPE=bug_report LIMIT=5)
	@if [ -z "$(TYPE)" ]; then \
		echo "ERROR: TYPE parameter required."; \
		exit 1; \
	fi
	uv run python ../tools/utilities/smart_populate.py suggest-files --type "$(TYPE)" $(if $(LIMIT),--limit $(LIMIT),)

smart-frontmatter: ## Generate artifact frontmatter (usage: make smart-frontmatter TYPE=design)
	@if [ -z "$(TYPE)" ]; then \
		echo "ERROR: TYPE parameter required."; \
		exit 1; \
	fi
	uv run python ../tools/utilities/smart_populate.py frontmatter --type "$(TYPE)"

# Legacy Artifact Migration
artifacts-find: ## Find legacy artifacts (usage: make artifacts-find LIMIT=10)
	uv run python ../tools/utilities/legacy_migrator.py find $(if $(LIMIT),--limit $(LIMIT),)

artifacts-migrate-dry: ## Show artifact migrations without applying (usage: make artifacts-migrate-dry FILE=path/to/artifact.md)
	@if [ -n "$(FILE)" ]; then \
		python ../tools/utilities/legacy_migrator.py migrate --file "$(FILE)" --dry-run; \
	else \
		echo "ERROR: FILE parameter required. Usage: make artifacts-migrate-dry FILE=path/to/artifact.md"; \
		exit 1; \
	fi

artifacts-migrate: ## Migrate artifacts to current naming convention (usage: make artifacts-migrate FILE=path --autofix)
	@if [ -n "$(FILE)" ]; then \
		python ../tools/utilities/legacy_migrator.py migrate --file "$(FILE)" --autofix; \
	elif [ "$(ALL)" = "1" ]; then \
		python ../tools/utilities/legacy_migrator.py migrate --all --autofix; \
	else \
		echo "ERROR: Use FILE=path or ALL=1. Example: make artifacts-migrate FILE=docs/artifacts/old.md --autofix"; \
		exit 1; \
	fi

artifacts-detect-moves: ## Detect manually moved/renamed artifacts
	uv run python ../tools/utilities/legacy_migrator.py detect-moves

artifacts-repair-moves: ## Repair manually moved artifacts (remove duplicates)
	uv run python ../tools/utilities/legacy_migrator.py detect-moves --repair

artifacts-repair-moves-preview: ## Preview manual move repairs
	uv run python ../tools/utilities/legacy_migrator.py detect-moves --repair --dry-run

# Validation and Compliance

.PHONY: plugins-snapshot
plugins-snapshot: ## Refresh .agentqms/state/plugins.yaml (auto)
	@python -m AgentQMS.tools.core.plugins --write-snapshot --json > /dev/null 2>&1 || echo "⚠️  Plugin snapshot skipped (optional)"

validate: plugins-snapshot ## Validate all artifacts
	qms validate --all

.PHONY: reindex
reindex: ## Reindex artifacts
	uv run python ../tools/documentation/reindex_artifacts.py

.PHONY: check-links
check-links: ## Check Markdown links referring to artifacts
	uv run python ../tools/documentation/validate_links.py

.PHONY: fix
fix: ## Safe autofix pipeline
	uv run python ../tools/utilities/autofix_artifacts.py $(ARGS)

.PHONY: commit-fixes
commit-fixes: ## Commit artifact fixes with standard message (usage: make commit-fixes MSG="description")
	@if git diff --quiet && git diff --staged --quiet; then \
		echo "⚠️  No changes to commit"; \
	else \
		git add docs/artifacts/ AgentQMS/; \
		if [ -z "$(MSG)" ]; then \
			git commit -m "AgentQMS: artifact fixes"; \
		else \
			git commit -m "AgentQMS: $(MSG)"; \
		fi; \
		echo "✅ Changes committed"; \
	fi

validate-file: ## Validate specific file (usage: make validate-file FILE=path/to/file.md)
	qms validate --file $(FILE)

validate-naming: ## Check naming conventions only
	qms validate --check-naming

boundary: ## Validate framework/project boundaries
	uv run python ../tools/compliance/validate_boundaries.py

audit-framework: ## (DEPRECATED) Use 'make validate' or 'make compliance' instead
	@echo "⚠️  'audit-framework' is deprecated. Use 'make validate' or 'make compliance'."
	@exit 0

audit-framework-verbose: ## (DEPRECATED) Use 'make validate' or 'make compliance' instead
	@echo "⚠️  'audit-framework-verbose' is deprecated. Use 'make validate' or 'make compliance'."
	@exit 0

audit-framework-json: ## (DEPRECATED) Use 'make validate' or 'make compliance' instead
	@echo "⚠️  'audit-framework-json' is deprecated. Use 'make validate' or 'make compliance'."
	@exit 0

audit-framework-plugins: ## (DEPRECATED) Use 'make validate' instead
	@echo "⚠️  'audit-framework-plugins' is deprecated. Use 'make validate'."
	@exit 0

audit-framework-templates: ## (DEPRECATED) Use 'make validate' instead
	@echo "⚠️  'audit-framework-templates' is deprecated. Use 'make validate'."
	@exit 0

# Artifact Audit (Safe Mode)
audit-fix-batch: ## (DEPRECATED) Use 'make fix' pipeline instead
	@echo "⚠️  'audit-fix-batch' is deprecated. Use 'make fix ARGS=\"--limit N --dry-run\"'."
	@exit 0

audit-fix-batch-apply: ## (DEPRECATED) Use 'make fix' pipeline instead
	@echo "⚠️  'audit-fix-batch-apply' is deprecated. Use 'make fix'."
	@exit 0

audit-fix-all: ## (DEPRECATED) Use 'make fix' pipeline instead
	@echo "⚠️  'audit-fix-all' is deprecated. Use 'make fix'."
	@exit 0

audit-fix-all-force: ## (DEPRECATED) Use 'make fix' pipeline instead
	@echo "⚠️  'audit-fix-all-force' is deprecated. Use 'make fix'."
	@exit 0

audit-report: ## (DEPRECATED) Use 'make validate' with reports
	@echo "⚠️  'audit-report' is deprecated. Use 'make validate' or 'make compliance'."
	@exit 0

audit-report-all: ## (DEPRECATED) Use 'make validate' with reports
	@echo "⚠️  'audit-report-all' is deprecated. Use 'make validate' or 'make compliance'."
	@exit 0

# Archive Management
archive-artifacts-dry: ## Preview artifacts marked for archival (status='archived')
	@echo "🗄️  Previewing artifacts marked for archival..."
	uv run python ../tools/archive/archive_artifacts.py --dry-run
	@echo ""
	@echo "💡 To move files to archive, run: make archive-artifacts"

archive-artifacts: ## Move archived artifacts to archive/artifacts/
	uv run python ../tools/archive/archive_artifacts.py

audit-vlm-migrate: ## Migrate VLM reports from BUG prefix to vlm_report_ prefix
	@echo "🔍 Previewing VLM reports migration..."
	uv run python ../tools/utilities/migrate_vlm_reports.py --dry-run
	@echo ""
	@echo "💡 To apply migration, run: make audit-vlm-migrate-apply"

audit-vlm-migrate-apply: ## Apply VLM reports migration
	uv run python ../tools/utilities/migrate_vlm_reports.py

# Deprecated Code Registry
deprecated-list: ## List all deprecated symbols
	uv run python ../tools/utilities/deprecated_registry.py list

deprecated-register: ## Register a deprecated symbol (usage: make deprecated-register SYMBOL=Name FILE=path REPLACEMENT=path)
	@if [ -z "$(SYMBOL)" ] || [ -z "$(FILE)" ] || [ -z "$(REPLACEMENT)" ]; then \
		echo "ERROR: SYMBOL, FILE, and REPLACEMENT parameters required."; \
		exit 1; \
	fi
	uv run python ../tools/utilities/deprecated_registry.py register "$(SYMBOL)" "$(FILE)" "$(REPLACEMENT)" \
		$(if $(REMOVAL_PLAN),--removal-plan "$(REMOVAL_PLAN)",) \
		$(if $(REMOVAL_DATE),--removal-date "$(REMOVAL_DATE)",) \
		$(if $(DESCRIPTION),--description "$(DESCRIPTION)",)

deprecated-validate: ## Validate artifacts for deprecated references (usage: make deprecated-validate [FILE=path|DIRECTORY=path|ALL=1])
	@if [ -n "$(FILE)" ]; then \
		python ../tools/utilities/deprecated_registry.py validate --file "$(FILE)"; \
	elif [ -n "$(DIRECTORY)" ]; then \
		python ../tools/utilities/deprecated_registry.py validate --directory "$(DIRECTORY)"; \
	elif [ "$(ALL)" = "1" ]; then \
		python ../tools/utilities/deprecated_registry.py validate --all; \
	else \
		python ../tools/utilities/deprecated_registry.py validate --all; \
	fi

# Tracking Database Repair
track-init: ## Initialize tracking database
	@uv run python -c "import sys; sys.path.insert(0, '../..'); from AgentQMS.tools.utilities.tracking.db import init_db; init_db(); print('✅ Tracking database initialized')"

track-repair: ## Repair tracking database artifact paths (usage: make track-repair [DRY_RUN=1])
	@if [ "$(DRY_RUN)" = "1" ]; then \
		python ../tools/utilities/tracking_repair.py --dry-run; \
	else \
		python ../tools/utilities/tracking_repair.py; \
	fi

track-repair-preview: ## Preview tracking database repairs without applying
	uv run python ../tools/utilities/tracking_repair.py --dry-run

track-status: ## Show tracking database status
	@uv run python -c "import sys; sys.path.insert(0, '../..'); from AgentQMS.tools.utilities.tracking.query import get_status; print(get_status('all'))"

compliance: plugins-snapshot ## Check compliance status
	qms monitor --check

compliance-fix: ## Fix compliance issues automatically (Legacy)
	@echo "⚠️  compliance-fix is deprecated. Use 'make fix' for standard fixes or 'make compliance-fix-ai' for AI fixes."

compliance-fix-ai: ## Fix compliance issues using Grok AI (usage: make compliance-fix-ai LIMIT=5)
	uv run python ../tools/utilities/grok_fixer.py $(if $(LIMIT),--limit $(LIMIT),) $(if $(DRY_RUN),--dry-run,)


compliance-dashboard: ## Open enhanced compliance dashboard (DEPRECATED)
	@echo "⚠️  compliance-dashboard is deprecated. Use 'make compliance' for checks." && exit 0

# Documentation Management
docs-generate: ## Generate documentation index
	uv run python ../tools/documentation/auto_generate_index.py --validate

docs-regenerate: ## Regenerate all documentation
	uv run python ../tools/documentation/regenerate_docs.py

docs-update-indexes: ## Update artifact indexes
	uv run python ../tools/documentation/update_artifact_indexes.py --all

docs-validate-links: ## Validate documentation links
	uv run python ../tools/documentation/validate_links.py AgentQMS/knowledge

docs-validate-manifest: ## Validate documentation manifest
	uv run python ../tools/documentation/validate_manifest.py

docs-validate-metadata: ## Validate documentation metadata
	uv run python ../tools/documentation/validate_metadata.py

# Context Bundles
context: ## Generate context bundle for task (usage: make context TASK="task description")
	@if [ -z "$(TASK)" ]; then \
		echo "ERROR: TASK parameter required. Usage: make context TASK=\"task description\""; \
		exit 1; \
	fi
	uv run python ../tools/utilities/get_context.py --task "$(TASK)"

context-development: ## Get development context bundle
	uv run python ../tools/utilities/get_context.py --type development

context-docs: ## Get documentation context bundle
	uv run python ../tools/utilities/get_context.py --type documentation

context-debug: ## Get debugging context bundle
	uv run python ../tools/utilities/get_context.py --type debugging

context-plan: ## Get planning context bundle
	uv run python ../tools/utilities/get_context.py --type planning

context-list: ## List all available context bundles
	uv run python ../tools/utilities/get_context.py --list-bundles

context-suggest: ## Suggest context bundles for task (usage: make context-suggest TASK="task description")
	@if [ -z "$(TASK)" ]; then \
		echo "ERROR: TASK parameter required. Usage: make context-suggest TASK=\"task description\""; \
		exit 1; \
	fi
	uv run python ../tools/utilities/suggest_context.py "$(TASK)"

# Context Control & Inspection
context-status: ## Show context bundling system status
	uv run python ../tools/utilities/context_control.py --status

context-enable: ## Re-enable context bundling after maintenance
	uv run python ../tools/utilities/context_control.py --enable

context-disable: ## Disable bundling for maintenance (usage: make context-disable REASON="..." HOURS=2)
	@if [ -z "$(REASON)" ]; then \
		echo "ERROR: REASON parameter required. Usage: make context-disable REASON=\"...\" [HOURS=2]"; \
		exit 1; \
	fi
	uv run python ../tools/utilities/context_control.py --disable --maintenance $(if $(HOURS),--duration $(HOURS),) --reason "$(REASON)"

context-feedback: ## Submit AI feedback (usage: make context-feedback BUNDLE=name SCORE=80 TOKENS=2048 TEXT="useful")
	@if [ -z "$(BUNDLE)" ] || [ -z "$(SCORE)" ] || [ -z "$(TOKENS)" ] || [ -z "$(TEXT)" ]; then \
		echo "ERROR: BUNDLE, SCORE, TOKENS, and TEXT required."; \
		exit 1; \
	fi
	uv run python ../tools/utilities/context_control.py --feedback "$(BUNDLE)" "$(SCORE)" "$(TOKENS)" "$(TEXT)"

context-memory: ## Analyze memory footprint of context bundles
	uv run python ../tools/utilities/context_inspector.py --memory

context-stale: ## Detect stale context (usage: make context-stale DAYS=7)
	uv run python ../tools/utilities/context_inspector.py --stale $(if $(DAYS),--threshold $(DAYS),)

context-inspect: ## Inspect a specific bundle (usage: make context-inspect BUNDLE=name)
	@if [ -z "$(BUNDLE)" ]; then \
		echo "ERROR: BUNDLE parameter required."; \
		exit 1; \
	fi
	uv run python ../tools/utilities/context_inspector.py --inspect "$(BUNDLE)"

plan-progress: ## Show or update plan progress (usage: make plan-progress show FILE=path/to/plan.md)
	@if [ -z "$(FILE)" ]; then \
		echo "ERROR: FILE parameter required. Usage: make plan-progress show FILE=path/to/plan.md"; \
		exit 1; \
	fi
	uv run python ../tools/utilities/plan_progress.py $(ACTION) --file "$(FILE)"

plan-progress-show: ## Show plan progress (usage: make plan-progress-show FILE=path/to/plan.md)
	@if [ -z "$(FILE)" ]; then \
		echo "ERROR: FILE parameter required."; \
		exit 1; \
	fi
	uv run python ../tools/utilities/plan_progress.py show --file "$(FILE)"

plan-progress-update: ## Update plan status (usage: make plan-progress-update FILE=plan.md STATUS="New Status")
	@if [ -z "$(FILE)" ] || [ -z "$(STATUS)" ]; then \
		echo "ERROR: FILE and STATUS parameters required."; \
		exit 1; \
	fi
	uv run python ../tools/utilities/plan_progress.py update-status --file "$(FILE)" --status "$(STATUS)"

plan-progress-complete: ## Mark task as complete (usage: make plan-progress-complete FILE=plan.md TASK="Task X.X")
	@if [ -z "$(FILE)" ] || [ -z "$(TASK)" ]; then \
		echo "ERROR: FILE and TASK parameters required."; \
		exit 1; \
	fi
	uv run python ../tools/utilities/plan_progress.py mark-complete --file "$(FILE)" --task "$(TASK)"

# Audit Framework Tools
audit-init: ## (DEPRECATED) Legacy audit framework - use assessments/design docs
	@echo "⚠️  'audit-init' is deprecated. Prefer assessments or design_documents."
	@exit 0

audit-validate: ## (DEPRECATED) Legacy audit framework
	@echo "⚠️  'audit-validate' is deprecated. Use 'make validate'."
	@exit 0

audit-checklist-generate: ## (DEPRECATED) Legacy audit framework
	@echo "⚠️  'audit-checklist-generate' is deprecated. Track tasks in implementation plans."
	@exit 0

audit-checklist-report: ## (DEPRECATED) Legacy audit framework
	@echo "⚠️  'audit-checklist-report' is deprecated. Use Project Compass or plans."
	@exit 0

docs-validate-templates: ## Validate documentation templates
	uv run python ../tools/documentation/validate_templates.py

docs-validate-ui-schema: ## Validate UI schema documentation
	uv run python ../tools/documentation/validate_ui_schema.py

docs-validate-ai: ## Validate AI documentation (DEPRECATED)
	@echo "⚠️  docs-validate-ai is deprecated. Use other docs-validate-* commands." && exit 0

# Feedback and Quality
feedback-report: ## Generate feedback report
	qms feedback list

feedback-issue: ## Report documentation issue (usage: make feedback-issue ISSUE="description" FILE="path")
	@echo "⚠️  feedback-issue: Use 'qms feedback report' instead"
	@echo "   Example: qms feedback report --issue-type documentation --description \"$(ISSUE)\" --file \"$(FILE)\""

feedback-suggest: ## Suggest improvement (usage: make feedback-suggest AREA="area" CURRENT="current" CHANGE="suggested" RATIONALE="reason")
	@echo "⚠️  feedback-suggest: Use 'qms feedback suggest' instead"
	@echo "   Example: qms feedback suggest --area \"$(AREA)\" --current \"$(CURRENT)\" --change \"$(CHANGE)\" --rationale \"$(RATIONALE)\""

quality-check: ## Check documentation quality
	qms quality --check

quality-check-output: ## Check documentation quality and save to file
	@echo "⚠️  quality-check-output: Use 'qms quality --check --json' instead for structured output"
	qms quality --check

ast-analyze: ## Run AST analysis on codebase (usage: make ast-analyze [TARGET=path/to/code])
	@if [ -n "$(TARGET)" ]; then \
		python ./cli_tools/ast_analysis.py analyze "$(TARGET)"; \
	else \
		python ./cli_tools/ast_analysis.py analyze; \
	fi

ast-generate-tests: ## Generate test scaffolds using AST analysis (usage: make ast-generate-tests TARGET=path/to/module.py)
	@if [ -z "$(TARGET)" ]; then \
		echo "ERROR: TARGET parameter required. Usage: make ast-generate-tests TARGET=path/to/module.py"; \
		exit 1; \
	fi
	uv run python ./cli_tools/ast_analysis.py generate-tests "$(TARGET)"

ast-extract-docs: ## Extract documentation from code using AST (usage: make ast-extract-docs TARGET=path/to/module.py)
	@if [ -z "$(TARGET)" ]; then \
		echo "ERROR: TARGET parameter required. Usage: make ast-extract-docs TARGET=path/to/module.py"; \
		exit 1; \
	fi
	uv run python ./cli_tools/ast_analysis.py extract-docs "$(TARGET)"

ast-check-quality: ## Check code quality using AST analysis (usage: make ast-check-quality [TARGET=path/to/code])
	@if [ -n "$(TARGET)" ]; then \
		python ./cli_tools/ast_analysis.py check-quality "$(TARGET)"; \
	else \
		python ./cli_tools/ast_analysis.py check-quality; \
	fi

context-bundle: ## Get specific context bundle (usage: make context-bundle BUNDLE=bundle-name)
	uv run python ../tools/utilities/get_context.py --bundle $(BUNDLE)

# Workflow Commands
workflow-create: ## Complete workflow: create artifact, validate, update indexes
	@echo "🚀 Starting complete artifact workflow..."
	@echo "📝 Creating artifact..."
	@echo "⚠️  Please specify TYPE, NAME, and TITLE:"
	@echo "   make workflow-create TYPE=implementation_plan NAME=my-plan TITLE=\"My Plan\""
	@if [ -z "$(TYPE)" ] || [ -z "$(NAME)" ] || [ -z "$(TITLE)" ]; then \
		echo "❌ Missing required parameters"; \
		exit 1; \
	fi
	qms artifact create --type $(TYPE) --name $(NAME) --title "$(TITLE)"
	@echo "✅ Artifact created"
	@echo "🔍 Validating artifact..."
	qms validate --all
	@echo "📊 Updating indexes..."
	uv run python ../tools/documentation/update_artifact_indexes.py --all
	@echo "🎉 Workflow completed successfully!"

workflow-validate: ## Complete validation workflow: validate, fix, monitor
	@echo "🔍 Starting validation workflow..."
	@echo "📋 Validating all artifacts..."
	qms validate --all
	@echo "📊 Checking compliance status..."
	qms monitor --check
	@echo "💡 Note: Automated compliance fixes are available via maintenance scripts"
	@echo "🎉 Validation workflow completed!"

workflow-docs: ## Complete documentation workflow: generate, validate, update
	@echo "📚 Starting documentation workflow..."
	@echo "🔄 Generating documentation index..."
	uv run python ../tools/documentation/auto_generate_index.py --validate
	@echo "🔗 Validating documentation links..."
	uv run python ../tools/documentation/validate_links.py AgentQMS/knowledge
	@echo "📊 Updating artifact indexes..."
	uv run python ../tools/documentation/update_artifact_indexes.py --all
	@echo "🎉 Documentation workflow completed!"

# Tracking (CLI-first)
plan-new: ## Create plan (usage: make plan-new TITLE="..." OWNER="me" [KEY=slug])
	uv run python ../tools/utilities/tracking/cli.py plan new --title "$(TITLE)" $(if $(OWNER),--owner "$(OWNER)",) $(if $(KEY),--key "$(KEY)",)

exp-new: ## Create experiment (usage: make exp-new TITLE="..." OBJECTIVE="..." OWNER="me" [KEY=slug])
	uv run python ../tools/utilities/tracking/cli.py exp new --title "$(TITLE)" $(if $(OBJECTIVE),--objective "$(OBJECTIVE)",) $(if $(OWNER),--owner "$(OWNER)",) $(if $(KEY),--key "$(KEY)",)

exp-export: ## Export experiment runs to CSV (usage: make exp-export OUT=data/ops/experiment_runs.csv)
	@if [ -z "$(OUT)" ]; then \
		echo "ERROR: OUT parameter required. Example: make exp-export OUT=data/ops/experiment_runs.csv"; \
		exit 1; \
	fi
	uv run python ../tools/utilities/tracking/cli.py exp export-runs --out "$(OUT)"

# Framework Export & Adoption
agentqms-init: ## Initialize AgentQMS in a new project (usage: make agentqms-init TARGET=/path/to/project)
	@if [ -z "$(TARGET)" ]; then \
		echo "ERROR: TARGET parameter required. Usage: make agentqms-init TARGET=/path/to/project"; \
		exit 1; \
	fi
	./workflows/init_framework.sh "$(TARGET)"

agentqms-init-here: ## Initialize AgentQMS in current directory
	./workflows/init_framework.sh .

# Maintenance Commands
clean: ## Clean up temporary files and caches
	@echo "🧹 Cleaning up temporary files..."
	find .. -name "*.pyc" -delete
	find .. -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find .. -name "*.log" -delete
	@echo "✅ Cleanup completed"

clean-feedback: ## Clean up old feedback files
	@echo "🧹 Cleaning up old feedback files..."
	find logs/feedback -name "*.json" -mtime +30 -delete 2>/dev/null || true
	@echo "✅ Feedback cleanup completed"

# Development Commands
dev-setup: ## Setup development environment
	@echo "🛠️ Setting up development environment..."
	cd ../.. && uv sync
	@echo "✅ Development environment ready"

dev-test: ## Test all agent tools
	@echo "🧪 Testing agent tools..."
	uv run python ../tools/core/discover.py
	qms artifact --help
	qms validate --help
	qms monitor --help
	qms feedback --help
	qms quality --help
	uv run python ../tools/documentation/auto_generate_index.py --help
	@echo "✅ All tools tested successfully"

# Information Commands
info-tools: ## Show detailed tool information
	@echo "🛠️ Agent Tools Information"
	@echo "========================="
	@echo ""
	@echo "📁 Core Tools:"
	@ls ../tools/core/*.py 2>/dev/null | sed 's/^/  /'
	@echo ""
	@echo "📁 Compliance Tools:"
	@ls ../tools/compliance/*.py 2>/dev/null | sed 's/^/  /'
	@echo ""
	@echo "📁 Documentation Tools:"
	@ls ../tools/documentation/*.py 2>/dev/null | sed 's/^/  /'
	@echo ""
	@echo "📁 Utility Tools:"
	@ls ../tools/utilities/*.py 2>/dev/null | sed 's/^/  /'

info-stats: ## Show project statistics
	@echo "📊 Project Statistics"
	@echo "====================="
	@echo ""
	@echo "📄 Total Artifacts:"
	@find ../../artifacts -name "*.md" 2>/dev/null | wc -l | sed 's/^/  /'
	@echo ""
	@echo "📁 Artifact Types:"
	@find ../../artifacts -name "*.md" 2>/dev/null | sed 's/.*\///' | sed 's/^[0-9-]*_[0-9-]*_//' | sed 's/_.*$$//' | sort | uniq -c | sort -nr | sed 's/^/  /' || echo "  No artifacts found"
	@echo ""
	@echo "📅 Recent Activity (last 7 days):"
	@find ../../artifacts -name "*.md" -mtime -7 2>/dev/null | wc -l | sed 's/^/  /'

# Quick Commands (most commonly used)
quick-help: help ## Quick help (alias for help)
quick-discover: discover ## Quick discover (alias for discover)
quick-validate: validate ## Quick validate (alias for validate)
quick-compliance: compliance ## Quick compliance check (alias for compliance)
quick-status: status ## Quick status check (alias for status)

# Catalog and Index Generation
tool-catalog: ## Generate tool catalog and scripts index
	uv run python ../tools/utilities/generate_tool_catalog.py



# Export Framework
export: ## Export framework to project-agnostic package (usage: make export OUTPUT=export_package/)
	@if [ -z "$(OUTPUT)" ]; then \
		echo "ERROR: OUTPUT parameter required. Usage: make export OUTPUT=export_package/"; \
		exit 1; \
	fi
	uv run python ../tools/utilities/export_framework.py --output "$(OUTPUT)" --project-root ../..

export-dry-run: ## Dry run export (see what would be exported) (usage: make export-dry-run OUTPUT=export_package/)
	@if [ -z "$(OUTPUT)" ]; then \
		echo "ERROR: OUTPUT parameter required. Usage: make export-dry-run OUTPUT=export_package/"; \
		exit 1; \
	fi
	uv run python ../tools/utilities/export_framework.py --output "$(OUTPUT)" --project-root ../.. --dry-run

export-validate: ## Export framework with validation (usage: make export-validate OUTPUT=export_package/)
	@if [ -z "$(OUTPUT)" ]; then \
		echo "ERROR: OUTPUT parameter required. Usage: make export-validate OUTPUT=export_package/"; \
		exit 1; \
	fi
	uv run python ../tools/utilities/export_framework.py --output "$(OUTPUT)" --project-root ../.. --validate

# Versioning and Deprecation
version-bump: ## Bump project version (usage: make version-bump VERSION=0.8.0 NOTES="..." DATE=YYYY-MM-DDTHH:MM:SSZ)
	@if [ -z "$(VERSION)" ]; then echo "ERROR: VERSION required"; exit 1; fi
	@DATE_VAL=$(if [ -n "$(DATE)" ]; then echo "$(DATE)"; else date -u +%Y-%m-%dT%H:%M:%SZ; fi); \
	echo "version: $(VERSION)" > ../project_version.yaml; \
	echo "release_date: $$DATE_VAL" >> ../project_version.yaml; \
	echo "notes: $(NOTES)" >> ../project_version.yaml; \
	echo "breaking_changes: []" >> ../project_version.yaml; \
	echo "Wrote ../project_version.yaml"

docs-deprecate: ## Deprecate docs based on project_version.yaml (add --dry-run with DRY_RUN=1)
	uv run python ../tools/documentation/deprecate_docs.py $(if $(DRY_RUN),--dry-run,)
	uv run python ../tools/documentation/update_artifact_indexes.py --all
	uv run python ../tools/compliance/validate_artifacts.py --all

changelog-draft: ## Generate changelog draft from tracking DB and git log
	uv run python ../tools/documentation/generate_changelog_draft.py

changelog-preview: ## Preview changelog draft without writing file
	uv run python ../tools/documentation/generate_changelog_draft.py --preview

# Artifact Status and Lifecycle Management
artifacts-status: ## Show artifact status dashboard with aging and lifecycle information
	uv run python ../tools/utilities/artifacts_status.py

artifacts-status-dashboard: ## Show full dashboard view of artifact health
	uv run python ../tools/utilities/artifacts_status.py --dashboard

artifacts-status-compact: ## Show compact table view of artifact status
	uv run python ../tools/utilities/artifacts_status.py --compact

artifacts-status-aging: ## Show aging information for all artifacts
	uv run python ../tools/utilities/artifacts_status.py --aging-only

artifacts-status-json: ## Output artifact status as JSON
	uv run python ../tools/utilities/artifacts_status.py --json

artifacts-status-threshold: ## Show artifacts older than threshold days (usage: make artifacts-status-threshold DAYS=90)
	@if [ -z "$(DAYS)" ]; then \
		echo "ERROR: DAYS parameter required. Usage: make artifacts-status-threshold DAYS=90"; \
		exit 1; \
	fi
	uv run python ../tools/utilities/artifacts_status.py --threshold $(DAYS)

# All-in-one Commands
all-check: validate compliance quality-check ## Run all checks: validate, compliance, quality
all-docs: docs-generate docs-validate-links docs-update-indexes ## Run all documentation tasks
all-feedback: feedback-report quality-check ## Run all feedback and quality checks

# Emergency Commands
emergency-fix: compliance-fix docs-update-indexes ## Emergency: fix all issues and update indexes
emergency-reset: clean clean-feedback ## Emergency: reset and clean everything

# IDE Configuration
ide-config: ## Generate/Update configuration files for all IDEs (Antigravity, Cursor, etc.)
	uv run python ../tools/utilities/generate_ide_configs.py

# Show version and system info
version: ## Show system version information
	@echo "🤖 Agent Tools Makefile (AGENT-ONLY)"
	@echo "Version: 2.0"
	@echo "Date: $(shell date)"
	@echo "Python: $(shell python --version 2>/dev/null || echo 'Not available')"
	@echo "Make: $(shell make --version | head -1)"
	@echo "OS: $(shell uname -s 2>/dev/null || echo 'Unknown')"
	@echo ""
	@echo "Implementation Layer: AgentQMS/tools/ (canonical)"
	@echo "⚠️  This Makefile is for AI agents only!"
	@echo "   Humans should use the main project Makefile."
