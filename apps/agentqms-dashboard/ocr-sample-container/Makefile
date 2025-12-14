# OCR Sample Container Makefile

.PHONY: help build run stop clean logs validate test shell push

help:
	@echo "OCR Sample Container - Available Commands"
	@echo ""
	@echo "Build & Run:"
	@echo "  make build              Build Docker image"
	@echo "  make run                Run container with docker-compose"
	@echo "  make shell              Open shell in running container"
	@echo ""
	@echo "Manage:"
	@echo "  make stop               Stop running containers"
	@echo "  make logs               View container logs"
	@echo "  make clean              Remove containers and images"
	@echo ""
	@echo "Test & Validate:"
	@echo "  make validate           Validate dataset integrity"
	@echo "  make test               Run data loader tests"
	@echo ""
	@echo "Deploy:"
	@echo "  make push               Push image to registry"
	@echo ""

build:
	@echo "Building OCR sample container..."
	docker build -t ocr-sample:latest .
	@echo "✅ Build complete"

run:
	@echo "Starting OCR sample container..."
	docker-compose up -d
	@docker-compose ps
	@echo "✅ Container running. Use 'make logs' to view logs"

stop:
	@echo "Stopping containers..."
	docker-compose down
	@echo "✅ Stopped"

logs:
	docker-compose logs -f ocr-sample

shell:
	@echo "Opening shell in container..."
	docker-compose exec ocr-sample bash

validate:
	@echo "Validating dataset..."
	docker-compose exec ocr-sample python scripts/load_ocr_data.py

test:
	@echo "Running data loader tests..."
	docker-compose run --rm ocr-sample python -m pytest scripts/ -v 2>/dev/null || \
		docker-compose exec ocr-sample python scripts/load_ocr_data.py

clean:
	@echo "Cleaning up..."
	docker-compose down --rmi all
	docker system prune -f
	@echo "✅ Cleanup complete"

push:
	@echo "Pushing image to registry..."
	docker tag ocr-sample:latest $(REGISTRY)/ocr-sample:latest
	docker push $(REGISTRY)/ocr-sample:latest
	@echo "✅ Push complete"

status:
	@echo "Container Status:"
	docker-compose ps
	@echo ""
	@echo "Image Info:"
	docker images | grep ocr-sample || echo "No ocr-sample image found"
	@echo ""
	@echo "Network:"
	docker network ls | grep ocr-network || echo "Network not found"

inspect:
	@echo "Inspecting container..."
	docker inspect ocr-sample-data | python -m json.tool

health:
	@echo "Checking container health..."
	docker-compose ps
	@docker-compose exec ocr-sample python -c "import json; import sys; \
		json.load(open('/app/data/annotations.json')); print('✅ Dataset healthy')" || \
		echo "❌ Dataset validation failed"

# Development targets
dev-logs:
	docker-compose logs -f --tail=50

dev-build:
	docker-compose build --no-cache

dev-restart:
	docker-compose restart ocr-sample

dev-stats:
	docker stats ocr-sample-data

# Export targets
export-zip:
	@echo "Creating ZIP archive..."
	cd .. && zip -r -q ocr-sample-container.zip ocr-sample-container/
	@ls -lh ../ocr-sample-container.zip

export-tar:
	@echo "Creating TAR.GZ archive..."
	cd .. && tar -czf ocr-sample-container.tar.gz ocr-sample-container/
	@ls -lh ../ocr-sample-container.tar.gz

export-all: export-zip export-tar
	@echo "✅ Export complete"
