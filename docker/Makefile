# Makefile for Docker Compose Management
# This Makefile abstracts common Docker commands for the OCR development environment

.PHONY: help build up down restart logs ssh status clean

# Default target
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

build: ## Build the Docker image
	docker compose --profile dev -f docker-compose.yml build ocr-dev

up: ## Start the container in detached mode
	docker compose --profile dev -f docker-compose.yml up ocr-dev -d

down: ## Stop and remove the container
	docker compose --profile dev -f docker-compose.yml down ocr-dev

restart: ## Restart the container
	docker compose --profile dev -f docker-compose.yml restart ocr-dev

logs: ## Show container logs
	docker compose --profile dev -f docker-compose.yml logs ocr-dev -f

ssh: ## Connect to the container via SSH
	ssh -p 2223 vscode@localhost

status: ## Show container status
	docker compose --profile dev -f docker-compose.yml ps ocr-dev

clean: ## Remove containers, networks, and volumes
	docker compose --profile dev -f docker-compose.yml down ocr-dev -v --remove-orphans

shell: ## Open a shell in the running container
	docker compose --profile dev -f docker-compose.yml exec ocr-dev bash

rebuild: ## Rebuild and restart the container
	docker compose --profile dev -f docker-compose.yml down ocr-dev
	docker compose --profile dev -f docker-compose.yml build ocr-dev --no-cache
	docker compose --profile dev -f docker-compose.yml up ocr-dev -d

build-docker-dev: ## Build the docker-dev Docker image
	docker compose --profile dev -f docker-compose.yml build docker-dev

up-docker-dev: ## Start the docker-dev container in detached mode
	docker compose --profile dev -f docker-compose.yml up docker-dev -d

down-docker-dev: ## Stop and remove the docker-dev container
	docker compose --profile dev -f docker-compose.yml down docker-dev

restart-docker-dev: ## Restart the docker-dev container
	docker compose --profile dev -f docker-compose.yml restart docker-dev

logs-docker-dev: ## Show docker-dev container logs
	docker compose --profile dev -f docker-compose.yml logs docker-dev -f

ssh-docker-dev: ## Connect to the docker-dev container via SSH
	ssh -p 2225 vscode@localhost

status-docker-dev: ## Show docker-dev container status
	docker compose --profile dev -f docker-compose.yml ps docker-dev

clean-docker-dev: ## Remove docker-dev containers, networks, and volumes
	docker compose --profile dev -f docker-compose.yml down docker-dev -v --remove-orphans

shell-docker-dev: ## Open a shell in the running docker-dev container
	docker compose --profile dev -f docker-compose.yml exec docker-dev bash

rebuild-docker-dev: ## Rebuild and restart the docker-dev container
	docker compose --profile dev -f docker-compose.yml down docker-dev
	docker compose --profile dev -f docker-compose.yml build docker-dev --no-cache
	docker compose --profile dev -f docker-compose.yml up docker-dev -d