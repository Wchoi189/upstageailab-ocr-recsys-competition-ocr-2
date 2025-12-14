.PHONY: help install dev build test lint format clean restart-servers stop-servers logs-backend logs-frontend

# Color output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Directories
FRONTEND_DIR := frontend
BACKEND_DIR := backend
SHARED_DIR := ../..

# Python Configuration
PYTHON := /home/vscode/.pyenv/versions/3.11.14/bin/python
UV := uv

help:
	@echo "$(BLUE)=== AgentQMS Dashboard Makefile ===$(NC)"
	@echo ""
	@echo "$(GREEN)Setup & Installation:$(NC)"
	@echo "  make install              - Install all dependencies (frontend + backend)"
	@echo "  make install-frontend     - Install frontend dependencies only"
	@echo "  make install-backend      - Install backend dependencies only"
	@echo ""
	@echo "$(GREEN)Development:$(NC)"
	@echo "  make dev                  - Start both frontend (3000) and backend (8000) servers"
	@echo "  make dev-frontend         - Start frontend dev server (port 3000)"
	@echo "  make dev-backend          - Start backend dev server (port 8000)"
	@echo "  make build                - Build production frontend bundle"
	@echo ""
	@echo "$(GREEN)Testing & Quality:$(NC)"
	@echo "  make test                 - Run all tests (frontend + backend)"
	@echo "  make test-frontend        - Run frontend tests"
	@echo "  make test-backend         - Run backend tests"
	@echo "  make lint                 - Lint both frontend and backend code"
	@echo "  make lint-frontend        - Lint frontend code only"
	@echo "  make lint-backend         - Lint backend code only"
	@echo "  make format               - Format code (frontend + backend)"
	@echo "  make format-frontend      - Format frontend code only"
	@echo "  make format-backend       - Format backend code only"
	@echo ""
	@echo "$(GREEN)Server Management:$(NC)"
	@echo "  make restart-servers      - Restart both frontend and backend servers"
	@echo "  make restart-frontend     - Restart frontend server only"
	@echo "  make restart-backend      - Restart backend server only"
	@echo "  make stop-servers         - Stop both frontend and backend servers"
	@echo "  make stop-frontend        - Stop frontend server only"
	@echo "  make stop-backend         - Stop backend server only"
	@echo "  make logs-frontend        - Show frontend server logs"
	@echo "  make logs-backend         - Show backend server logs"
	@echo "  make status               - Show server status and port info"
	@echo ""
	@echo "$(GREEN)Database & Utilities:$(NC)"
	@echo "  make db-check             - Check tracking database status"
	@echo "  make db-reset             - Reset tracking database (⚠️  destructive)"
	@echo "  make clean                - Clean all generated files and caches"
	@echo ""

# ═════════════════════════════════════════════════════════════════════════════
# Setup & Installation
# ═════════════════════════════════════════════════════════════════════════════

install: install-frontend install-backend
	@echo "$(GREEN)✓ All dependencies installed$(NC)"

install-frontend:
	@echo "$(BLUE)Installing frontend dependencies...$(NC)"
	cd $(FRONTEND_DIR) && npm install
	@echo "$(GREEN)✓ Frontend dependencies installed$(NC)"

install-backend:
	@echo "$(BLUE)Installing backend dependencies with uv...$(NC)"
	cd $(BACKEND_DIR) && $(UV) pip install -r requirements.txt
	@echo "$(GREEN)✓ Backend dependencies installed$(NC)"

# ═════════════════════════════════════════════════════════════════════════════
# Development Servers
# ═════════════════════════════════════════════════════════════════════════════

dev: dev-backend dev-frontend
	@echo "$(GREEN)✓ Both servers running$(NC)"
	@echo "  Frontend: http://localhost:3000"
	@echo "  Backend:  http://localhost:8000"

dev-frontend:
	@echo "$(BLUE)Starting frontend dev server on port 3000...$(NC)"
	cd $(FRONTEND_DIR) && npm run dev

dev-backend:
	@echo "$(BLUE)Starting backend dev server on port 8000...$(NC)"
	cd $(BACKEND_DIR) && $(PYTHON) server.py

# ═════════════════════════════════════════════════════════════════════════════
# Build
# ═════════════════════════════════════════════════════════════════════════════

build:
	@echo "$(BLUE)Building production frontend bundle...$(NC)"
	cd $(FRONTEND_DIR) && npm run build
	@echo "$(GREEN)✓ Build complete. Output in frontend/dist$(NC)"

# ═════════════════════════════════════════════════════════════════════════════
# Testing
# ═════════════════════════════════════════════════════════════════════════════

test: test-frontend test-backend
	@echo "$(GREEN)✓ All tests completed$(NC)"

test-frontend:
	@echo "$(BLUE)Running frontend tests...$(NC)"
	cd $(FRONTEND_DIR) && npm test

test-backend:
	@echo "$(BLUE)Running backend tests...$(NC)"
	cd $(BACKEND_DIR) && $(PYTHON) -m pytest tests/ -v

# ═════════════════════════════════════════════════════════════════════════════
# Linting
# ═════════════════════════════════════════════════════════════════════════════

lint: lint-frontend lint-backend
	@echo "$(GREEN)✓ Linting complete$(NC)"

lint-frontend:
	@echo "$(BLUE)Linting frontend code...$(NC)"
	cd $(FRONTEND_DIR) && npm run lint

lint-backend:
	@echo "$(BLUE)Linting backend code...$(NC)"
	cd $(BACKEND_DIR) && $(PYTHON) -m pylint src/ || true

# ═════════════════════════════════════════════════════════════════════════════
# Formatting
# ═════════════════════════════════════════════════════════════════════════════

format: format-frontend format-backend
	@echo "$(GREEN)✓ Code formatting complete$(NC)"

format-frontend:
	@echo "$(BLUE)Formatting frontend code...$(NC)"
	cd $(FRONTEND_DIR) && npm run format

format-backend:
	@echo "$(BLUE)Formatting backend code...$(NC)"
	cd $(BACKEND_DIR) && $(PYTHON) -m black src/ || true

# ═════════════════════════════════════════════════════════════════════════════
# Server Management
# ═════════════════════════════════════════════════════════════════════════════

restart-servers: stop-servers
	@echo "$(BLUE)Starting servers...$(NC)"
	@sleep 1
	@echo "$(BLUE)Starting backend on port 8000...$(NC)"
	cd $(BACKEND_DIR) && $(PYTHON) server.py > /tmp/backend.log 2>&1 &
	@sleep 2
	@echo "$(BLUE)Starting frontend on port 3000...$(NC)"
	cd $(FRONTEND_DIR) && npm run dev > /tmp/frontend.log 2>&1 &
	@sleep 2
	@echo "$(GREEN)✓ Servers restarted$(NC)"
	@echo "  Frontend: http://localhost:3000"
	@echo "  Backend:  http://localhost:8000"
	@echo ""
	@echo "View logs with: $(BLUE)make logs-backend$(NC) or $(BLUE)make logs-frontend$(NC)"

restart-frontend: stop-frontend
	@echo "$(BLUE)Starting frontend on port 3000...$(NC)"
	cd $(FRONTEND_DIR) && npm run dev > /tmp/frontend.log 2>&1 &
	@sleep 2
	@echo "$(GREEN)✓ Frontend restarted$(NC)"
	@echo "  Frontend: http://localhost:3000"

restart-backend: stop-backend
	@echo "$(BLUE)Starting backend on port 8000...$(NC)"
	cd $(BACKEND_DIR) && $(PYTHON) server.py > /tmp/backend.log 2>&1 &
	@sleep 1
	@echo "$(GREEN)✓ Backend restarted$(NC)"
	@echo "  Backend: http://localhost:8000"

stop-servers:
	@echo "$(YELLOW)Stopping all servers...$(NC)"
	@pkill -f "npm run dev" && echo "$(GREEN)✓ Frontend stopped$(NC)" || echo "$(YELLOW)Frontend not running$(NC)"
	@pkill -f "python server.py" && echo "$(GREEN)✓ Backend stopped$(NC)" || echo "$(YELLOW)Backend not running$(NC)"
	@echo ""

stop-frontend:
	@echo "$(YELLOW)Stopping frontend...$(NC)"
	@pkill -f "npm run dev" && echo "$(GREEN)✓ Frontend stopped$(NC)" || echo "$(YELLOW)Frontend not running$(NC)"

stop-backend:
	@echo "$(YELLOW)Stopping backend...$(NC)"
	@pkill -f "python server.py" && echo "$(GREEN)✓ Backend stopped$(NC)" || echo "$(YELLOW)Backend not running$(NC)"

logs-frontend:
	@echo "$(BLUE)Frontend logs (last 50 lines):$(NC)"
	@cd $(FRONTEND_DIR) && npm run dev 2>&1 | tail -50

logs-backend:
	@echo "$(BLUE)Backend logs (last 50 lines):$(NC)"
	@cd $(BACKEND_DIR) && $(PYTHON) server.py 2>&1 | tail -50

# ═════════════════════════════════════════════════════════════════════════════
# Database & Utilities
# ═════════════════════════════════════════════════════════════════════════════

db-check:
	@echo "$(BLUE)Checking tracking database status...$(NC)"
	@if [ -f "$(SHARED_DIR)/data/ops/tracking.db" ]; then \
		echo "$(GREEN)✓ Database found: $(SHARED_DIR)/data/ops/tracking.db$(NC)"; \
		ls -lh "$(SHARED_DIR)/data/ops/tracking.db"; \
	else \
		echo "$(RED)✗ Database not found$(NC)"; \
	fi

db-reset:
	@echo "$(RED)⚠️  WARNING: This will delete the tracking database!$(NC)"
	@read -p "Are you sure? (yes/no): " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		rm -f "$(SHARED_DIR)/data/ops/tracking.db"; \
		echo "$(YELLOW)Database reset. Restart backend to recreate.$(NC)"; \
	else \
		echo "$(GREEN)Cancelled$(NC)"; \
	fi

# ═════════════════════════════════════════════════════════════════════════════
# Cleanup
# ═════════════════════════════════════════════════════════════════════════════

clean:
	@echo "$(BLUE)Cleaning generated files...$(NC)"
	cd $(FRONTEND_DIR) && rm -rf node_modules dist .vite
	cd $(BACKEND_DIR) && rm -rf __pycache__ .pytest_cache dist build *.egg-info
	cd $(BACKEND_DIR) && find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

# ═════════════════════════════════════════════════════════════════════════════
# Quick Start (Common Workflow)
# ═════════════════════════════════════════════════════════════════════════════

quick-start: install lint
	@echo ""
	@echo "$(GREEN)✓ Quick start setup complete!$(NC)"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Run: $(BLUE)make dev$(NC)"
	@echo "  2. Open: $(BLUE)http://localhost:3000$(NC)"
	@echo "  3. Backend API: $(BLUE)http://localhost:8000/api/v1/health$(NC)"

# ═════════════════════════════════════════════════════════════════════════════
# Development Workflow (with watch)
# ═════════════════════════════════════════════════════════════════════════════

dev-watch: install
	@echo "$(BLUE)Starting development with file watching...$(NC)"
	@echo "Frontend changes will auto-reload"
	@echo "Backend: Press Ctrl+C to stop"
	cd $(FRONTEND_DIR) && npm run dev &
	cd $(BACKEND_DIR) && $(PYTHON) server.py

# ═════════════════════════════════════════════════════════════════════════════
# Production Build & Run
# ═════════════════════════════════════════════════════════════════════════════

prod-build: build
	@echo "$(GREEN)✓ Production build ready$(NC)"
	@echo "Run with: $(BLUE)make prod-serve$(NC)"

prod-serve:
	@echo "$(BLUE)Starting production frontend server...$(NC)"
	cd $(FRONTEND_DIR) && npm run preview &
	@echo "$(BLUE)Starting backend server...$(NC)"
	cd $(BACKEND_DIR) && $(PYTHON) server.py
	@echo "$(GREEN)✓ Production servers running$(NC)"

# ═════════════════════════════════════════════════════════════════════════════
# Status & Info
# ═════════════════════════════════════════════════════════════════════════════

status:
	@echo "$(BLUE)=== AgentQMS Dashboard Status ===$(NC)"
	@echo ""
	@echo "Processes:"
	@pgrep -f "npm run dev" > /dev/null && echo "$(GREEN)✓ Frontend running$(NC)" || echo "$(RED)✗ Frontend not running$(NC)"
	@pgrep -f "python server.py" > /dev/null && echo "$(GREEN)✓ Backend running$(NC)" || echo "$(RED)✗ Backend not running$(NC)"
	@echo ""
	@echo "Ports:"
	@netstat -tuln 2>/dev/null | grep -E ":3000|:8000" || echo "  Checking..."
	@echo ""
	@make db-check

version:
	@echo "$(BLUE)=== Dependency Versions ===$(NC)"
	@echo ""
	@echo "Python: $(PYTHON)"
	@$(PYTHON) --version
	@echo ""
	@echo "Frontend:"
	@cd $(FRONTEND_DIR) && npm list react typescript vite 2>/dev/null || echo "  Run: make install-frontend"
	@echo ""
	@echo "Backend:"
	@cd $(BACKEND_DIR) && $(PYTHON) -c "import fastapi; print(f'  FastAPI: {fastapi.__version__}')" 2>/dev/null || echo "  Run: make install-backend"
