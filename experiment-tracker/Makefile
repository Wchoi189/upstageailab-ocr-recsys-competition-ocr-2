# Experiment Tracker Makefile Wrapper
# Simplifies interaction with etk.py for common workflows

.PHONY: help exp-init exp-status exp-list exp-sync exp-analytics

# Default experiment ID detection (uses .current if it exists)
CURRENT_EXP = $(shell cat experiments/.current 2>/dev/null || echo "")

help:
	@echo "Experiment Tracker Kit (ETK) - Makefile Commands"
	@echo ""
	@echo "Usage:"
	@echo "  make exp-init name=\"name\"          - Initialize new experiment"
	@echo "  make exp-status [id=\"id\"]          - Show status of current or specific experiment"
	@echo "  make exp-list                      - List all experiments"
	@echo "  make exp-sync [all=true]           - Sync artifacts to database"
	@echo "  make exp-analytics                 - Show analytics dashboard"
	@echo "  make exp-validate [id=\"id\"]        - Validate experiment compliance"
	@echo ""

exp-init:
	@if [ -z "$(name)" ]; then echo "Error: name is required. Example: make exp-init name=\"baseline_test\""; exit 1; fi
	python3 etk.py init "$(name)"

exp-status:
	@if [ -z "$(id)" ]; then \
		python3 etk.py status; \
	else \
		python3 etk.py status "$(id)"; \
	fi

exp-list:
	python3 etk.py list

exp-sync:
	@if [ "$(all)" = "true" ]; then \
		python3 etk.py sync --all; \
	else \
		python3 etk.py sync; \
	fi

exp-analytics:
	python3 etk.py analytics

exp-validate:
	@if [ -z "$(id)" ]; then \
		python3 etk.py validate; \
	else \
		python3 etk.py validate "$(id)"; \
	fi
