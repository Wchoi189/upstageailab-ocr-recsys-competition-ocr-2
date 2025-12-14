# Multi-stage build for OCR Sample Data Container
FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY config/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy data and configs
COPY data/ ./data/
COPY config/ ./config/
COPY docs/ ./docs/
COPY scripts/ ./scripts/

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import json; json.load(open('data/annotations.json'))" || exit 1

# Default command
CMD ["python", "-c", "import json; print('OCR Sample Container Ready'); print(json.load(open('data/annotations.json')))"]
