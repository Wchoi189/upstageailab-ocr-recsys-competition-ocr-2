# Windows Agent Bridge Makefile
#
# This Makefile provides convenient commands for managing the Windows Agent Bridge.
# Note: This is designed to work with Windows environments that have Make available
# (such as through Git Bash, WSL, or other Unix-like shells on Windows).

.PHONY: install run test clean help stop-bg

# Default target
help:
	@echo "Windows Agent Bridge Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make install          Install dependencies"
	@echo "  make run              Run the bridge server"
	@echo "  make run-bg           Run the bridge server in the background"
	@echo "  make test             Run a test (see README for client usage)"
	@echo "  make clean            Clean up virtual environment"
	@echo "  make stop-bg          Stop the background bridge server"
	@echo "  make help             Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make install"
	@echo "  make run"
	@echo "  make test"
	@echo ""
	@echo "Note: For client usage, set environment variable:"
	@echo "  set RABBITMQ_HOST=localhost && uv run python bridge_client.py \"docker ps\""

install:
	.venv\Scripts\activate && pip install -r requirements.txt

run:
	.venv\Scripts\activate && uv run python bridge_server.py

run-bg:
	@echo "Starting bridge server in background..."
	cmd /c ".venv\Scripts\activate && start /b uv run python bridge_server.py"
	@echo "Bridge server started in background."

test:
	@echo "Testing bridge connection..."
	@echo "To test the bridge, run from another terminal:"
	@echo "  set RABBITMQ_HOST=localhost && .venv\Scripts\activate && uv run python bridge_client.py \"docker ps\""
	@echo "Or use the test_env.bat script that was created."

clean:
	@if exist .venv rmdir /s /q .venv
	@echo "Virtual environment cleaned."

# For stopping the background server
stop-bg:
	@powershell -Command "Get-Process python -ErrorAction SilentlyContinue | Where-Object {$_.MainWindowTitle -like '*bridge_server*'} | Stop-Process -Force"
	@echo "Background bridge server stopped (if running)."

# Create a test batch file for easy client testing
create-test-script:
	@echo @echo off > test_env.bat
	@echo call .venv\Scripts\activate.bat >> test_env.bat
	@echo set RABBITMQ_HOST=localhost >> test_env.bat
	@echo uv run python bridge_client.py "docker ps" >> test_env.bat
	@echo echo Test completed. >> test_env.bat
	@echo pause >> test_env.bat
	@echo Created test_env.bat for easy client testing