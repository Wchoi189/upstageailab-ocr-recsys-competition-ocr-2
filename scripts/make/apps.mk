# App Server Targets

.PHONY: serve-ocr-console
serve-ocr-console: ## Start OCR Inference Console frontend
	VITE_API_URL=$${VITE_API_URL:-http://127.0.0.1:8002/api} npm --prefix apps/ocr_inference_console run dev -- --host 0.0.0.0 --port 5173

.PHONY: ocr-console-backend
ocr-console-backend: ## Start OCR Console backend
	@export OCR_CHECKPOINT_PATH=$$(find outputs/experiments/train/ocr -name "*.ckpt" 2>/dev/null | head -n 1); \
	if [ -z "$$OCR_CHECKPOINT_PATH" ]; then echo "❌ No checkpoint found"; exit 1; fi; \
	uv run uvicorn apps.ocr_inference_console.backend.main:app --host 0.0.0.0 --port 8002 --reload

.PHONY: playground-console-dev
playground-console-dev: ## Start Playground Console frontend (Next.js)
	npm run dev:console

.PHONY: kill-ports
kill-ports: ## Force kill processes on common app ports
	@for port in 3000 5173 8000 8001 8002; do \
		pid=$$(lsof -t -i:$$port 2>/dev/null || true); \
		if [ -n "$$pid" ]; then kill -9 $$pid 2>/dev/null; echo "Killed port $$port"; fi; \
	done
