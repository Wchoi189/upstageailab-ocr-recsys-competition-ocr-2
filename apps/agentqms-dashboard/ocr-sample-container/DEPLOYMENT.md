# Docker Build and Deployment Guide

Complete guide for building, testing, and deploying the OCR Sample Container.

## Table of Contents
1. [Local Development](#local-development)
2. [Building Images](#building-images)
3. [Testing](#testing)
4. [Registry Deployment](#registry-deployment)
5. [Production Deployment](#production-deployment)
6. [Troubleshooting](#troubleshooting)

---

## Local Development

### Development Workflow

```bash
# 1. Build with no cache (ensures latest)
docker build -t ocr-sample:dev .

# 2. Run with hot-reload volume mount
docker run -it \
  -v $(pwd)/scripts:/app/scripts \
  -v $(pwd)/data:/app/data:ro \
  ocr-sample:dev bash

# 3. Make changes locally
# 4. Test in container
# 5. Rebuild when dependencies change
```

### Using Docker Compose for Development

```bash
# Build and run
docker-compose up -d

# View live logs
docker-compose logs -f

# Run a command
docker-compose exec ocr-sample python scripts/load_ocr_data.py

# Modify and test
docker-compose exec ocr-sample bash

# Restart when changes made
docker-compose restart ocr-sample
```

---

## Building Images

### Basic Build
```bash
docker build -t ocr-sample:latest .
```

### Build with Custom Tag
```bash
docker build -t ocr-sample:v1.0 .
docker build -t ocr-sample:$(date +%Y%m%d) .
```

### Build with BuildKit (Faster)
```bash
DOCKER_BUILDKIT=1 docker build -t ocr-sample:latest .
```

### Build for Multiple Platforms
```bash
docker buildx build --platform linux/amd64,linux/arm64 -t ocr-sample:latest .
```

### Multi-stage Build (if needed)
```bash
# Build stage 1: builder
# Build stage 2: runtime
# Result: smaller final image
docker build -f Dockerfile.multi -t ocr-sample:slim .
```

### Check Image Size
```bash
docker images ocr-sample

# Output:
# REPOSITORY   TAG      IMAGE ID      SIZE
# ocr-sample   latest   abc123def456  245MB
```

---

## Testing

### Container Health Check
```bash
# Run health check
docker-compose exec ocr-sample python -c \
  "import json; json.load(open('/app/data/annotations.json')); print('✅ Healthy')"
```

### Data Validation
```bash
docker-compose exec ocr-sample python scripts/load_ocr_data.py

# Expected output:
# ==================================================
# OCR SAMPLE DATASET SUMMARY
# ==================================================
# Images:               3
# Annotations:          5
# Categories:           1
# Avg Regions/Image:    1.67
# ==================================================
```

### Image Inspection
```bash
# Load and verify images
docker-compose exec ocr-sample python << 'EOF'
from PIL import Image
import json

with open('/app/data/annotations.json') as f:
    coco = json.load(f)

for img_info in coco['images']:
    path = f"/app/data/{img_info['file_name']}"
    try:
        img = Image.open(path)
        print(f"✅ {img_info['file_name']}: {img.size} {img.format}")
    except Exception as e:
        print(f"❌ {img_info['file_name']}: {e}")
EOF
```

### Full Integration Test
```bash
# Run complete validation suite
docker-compose run --rm ocr-sample bash << 'EOF'
set -e
echo "1. Loading data..."
python scripts/load_ocr_data.py

echo ""
echo "2. Checking image integrity..."
python -c "
from PIL import Image
import json

with open('/app/data/annotations.json') as f:
    coco = json.load(f)

for img in coco['images']:
    Image.open(f'/app/data/{img[\"file_name\"]}'
    print(f'✅ {img[\"file_name\"]}')
"

echo ""
echo "3. Validating annotations..."
python -c "
import json
with open('/app/data/annotations.json') as f:
    data = json.load(f)
    print(f'✅ {len(data[\"annotations\"])} annotations loaded')
"

echo ""
echo "✅ All tests passed!"
EOF
```

---

## Registry Deployment

### Login to Registry
```bash
# Docker Hub
docker login

# Or specify registry
docker login docker.io
docker login ghcr.io
docker login registry.example.com
```

### Tag for Registry
```bash
# Docker Hub
docker tag ocr-sample:latest myusername/ocr-sample:latest
docker tag ocr-sample:latest myusername/ocr-sample:v1.0

# GitHub Container Registry
docker tag ocr-sample:latest ghcr.io/myusername/ocr-sample:latest

# Private Registry
docker tag ocr-sample:latest registry.example.com/ocr-sample:latest
```

### Push Image
```bash
# Docker Hub
docker push myusername/ocr-sample:latest

# GitHub Container Registry
docker push ghcr.io/myusername/ocr-sample:latest

# Private Registry
docker push registry.example.com/ocr-sample:latest
```

### Pull from Registry
```bash
docker pull myusername/ocr-sample:latest
docker run -it myusername/ocr-sample:latest
```

### View Pushed Images
```bash
# List local images
docker images ocr-sample

# Check registry (e.g., Docker Hub)
# Go to hub.docker.com/r/myusername/ocr-sample
```

---

## Production Deployment

### Deploy on Cloud Run (Google Cloud)
```bash
# 1. Tag for Google Cloud
docker tag ocr-sample:latest gcr.io/my-project/ocr-sample:latest

# 2. Push to GCR
docker push gcr.io/my-project/ocr-sample:latest

# 3. Deploy
gcloud run deploy ocr-sample \
  --image gcr.io/my-project/ocr-sample:latest \
  --platform managed \
  --region us-central1 \
  --memory 512Mi \
  --cpu 1 \
  --allow-unauthenticated
```

### Deploy on Kubernetes
```bash
# 1. Create deployment manifest
cat > deployment.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ocr-sample
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ocr-sample
  template:
    metadata:
      labels:
        app: ocr-sample
    spec:
      containers:
      - name: ocr-sample
        image: ghcr.io/myusername/ocr-sample:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          exec:
            command:
            - python
            - -c
            - import json; json.load(open('/app/data/annotations.json'))
          initialDelaySeconds: 10
          periodSeconds: 30
EOF

# 2. Apply deployment
kubectl apply -f deployment.yaml

# 3. Verify
kubectl get deployments
kubectl get pods
```

### Docker Compose Production
```bash
# Use production compose file
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Scale services
docker-compose up -d --scale ocr-sample=3
```

---

## Image Optimization

### Reduce Image Size

1. **Use slim base image**
   ```dockerfile
   FROM python:3.11-slim
   ```

2. **Remove unnecessary files**
   ```dockerfile
   RUN apt-get clean && rm -rf /var/lib/apt/lists/*
   ```

3. **Multi-stage build** (if needed)
   ```dockerfile
   FROM python:3.11 as builder
   WORKDIR /build
   COPY . .
   RUN pip install --user --no-cache-dir -r requirements.txt

   FROM python:3.11-slim
   COPY --from=builder /root/.local /root/.local
   ENV PATH=/root/.local/bin:$PATH
   ```

4. **Check size before/after**
   ```bash
   docker images ocr-sample
   ```

---

## Troubleshooting

### Build Fails
```bash
# 1. Check Docker daemon
docker ps

# 2. Build with verbose output
docker build --progress=plain -t ocr-sample .

# 3. Clean up and retry
docker system prune -a
docker build -t ocr-sample .
```

### Container Won't Start
```bash
# 1. Check logs
docker-compose logs

# 2. Run with debugging
docker run -it --entrypoint bash ocr-sample:latest

# 3. Verify base image
docker pull python:3.11-slim
```

### Push/Pull Fails
```bash
# 1. Check authentication
docker login

# 2. Verify image name
docker images | grep ocr-sample

# 3. Check repository exists
# For Docker Hub: https://hub.docker.com/r/myusername/ocr-sample
```

### Performance Issues
```bash
# 1. Check resource usage
docker stats ocr-sample-data

# 2. Check logs for errors
docker-compose logs | grep -i error

# 3. Increase resources in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 1G
      cpus: '2'
```

---

## Best Practices

✅ **DO:**
- Use specific version tags (`v1.0` not `latest`)
- Include health checks
- Keep images under 500MB
- Use non-root user
- Scan images for vulnerabilities
- Use `.dockerignore`

❌ **DON'T:**
- Use `latest` tag in production
- Run as root
- Store secrets in images
- Include unnecessary files
- Skip health checks

---

## Security

### Scan for Vulnerabilities
```bash
# Using Trivy
trivy image ocr-sample:latest

# Using Docker Scout
docker scout cves ocr-sample:latest
```

### Sign Images
```bash
# Enable Docker Content Trust
export DOCKER_CONTENT_TRUST=1
docker push myusername/ocr-sample:latest
```

### Run as Non-root
```dockerfile
RUN useradd -m ocr
USER ocr
```

---

**Created**: December 11, 2025
**Version**: 1.0
**License**: MIT
