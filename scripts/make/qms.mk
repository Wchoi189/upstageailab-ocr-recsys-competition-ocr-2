# AgentQMS Shortcuts

.PHONY: qms-plan
qms-plan: ## Create implementation plan artifact
	uv run python -m AgentQMS.cli create-plan --name $(if $(NAME),$(NAME),my-plan)

.PHONY: qms-validate
qms-validate: ## Validate all QMS artifacts (use ARGS="--all" or ARGS="--artifacts-root <path>")
	uv run python AgentQMS/tools/compliance/validate_artifacts.py $(ARGS)

.PHONY: qms-compliance
qms-compliance: ## Run full QMS compliance checks (artifact and organization)
	uv run python -m AgentQMS.cli artifact check-compliance

.PHONY: qms-boundary
qms-boundary: ## Verify framework boundaries
	uv run python AgentQMS/tools/compliance/validate_boundaries.py

.PHONY: qms-context
qms-context: ## Generate task-specific context bundle
	uv run python -m AgentQMS.cli context $(if $(TASK),--task "$(TASK)",)

.PHONY: qms-context-suggest
qms-context-suggest: ## Suggest context for a task
	uv run python -m AgentQMS.cli suggest-context $(if $(TASK),--task "$(TASK)",)

.PHONY: qms-plan-progress
qms-plan-progress: ## View or update plan progress
	uv run python AgentQMS/tools/utils/plan_progress.py $(if $(PLAN),--plan "$(PLAN)",)

.PHONY: qms-discover
qms-discover: ## List all AgentQMS tools
	uv run python AgentQMS/tools/core/plugins/discovery.py

.PHONY: qms-status
qms-status: ## Check framework status
	uv run python -m AgentQMS.cli monitor --report

.PHONY: qms-registry
qms-registry: ## Generate registry from specs directory
	uv run python scripts/utils/generate_registry.py

.PHONY: qms-bundle-tokens
qms-bundle-tokens: ## Measure token usage for all context bundles
	@echo "📊 Measuring context bundle token usage..."
	@uv run python archive/migration-scripts/2026-02-03-spec-kit-migration/measure_bundle_tokens.py

# ============================================================================
# Middleware Observability (Phase C)
# ============================================================================

.PHONY: qms-middleware-health
qms-middleware-health: ## Check middleware health status
	@echo "🩺 Checking middleware health..."
	@uv run python AgentQMS/tools/middleware/dashboard.py --health-only

.PHONY: qms-middleware-dashboard
qms-middleware-dashboard: ## View middleware statistics dashboard
	@echo "📊 Middleware Dashboard:"
	@uv run python AgentQMS/tools/middleware/dashboard.py

.PHONY: qms-middleware-stats
qms-middleware-stats: ## Export middleware statistics as JSON
	@uv run python AgentQMS/tools/middleware/dashboard.py --json

.PHONY: qms-middleware-logs
qms-middleware-logs: ## Show recent middleware logs
	@echo "📋 Recent middleware logs:"
	@find outputs/logs/middleware -name "*.log" -type f 2>/dev/null | while read log; do \
		echo ""; echo "=== $$log ==="; tail -n 5 "$$log"; \
	done || echo "No logs found. Run middleware operations first."
