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
