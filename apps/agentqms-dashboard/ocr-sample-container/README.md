# OCR Sample Container

Containerized OCR inference sample dataset with Docker support.

## Quick Start

### Build and Run
```bash
# Build image
docker build -t ocr-sample:latest .

# Run container
docker run -it ocr-sample:latest

# Or use docker-compose
docker-compose up -d
```

### Container Structure
```
ocr-sample-container/
├── data/                      # Dataset files
│   ├── images/               # JPEG images
│   │   ├── sample_001.jpg   # Receipt
│   │   ├── sample_002.jpg   # Document
│   │   └── sample_003.jpg   # Document
│   └── annotations.json      # COCO format annotations
├── config/                    # Configuration
│   ├── requirements.txt      # Python dependencies
│   └── config.yaml           # Inference settings
├── docs/                      # Documentation
│   ├── OCR_INFERENCE_GUIDE.md
│   ├── OCR_SAMPLE_README.md
│   └── OCR_SAMPLE_MANIFEST.txt
├── scripts/                   # Custom scripts
├── Dockerfile               # Container definition
├── docker-compose.yml       # Compose configuration
└── README.md               # This file
```

## Usage

### Access Data in Container
```bash
# Shell into container
docker-compose exec ocr-sample bash

# Inspect annotations
python -c "import json; print(json.dumps(json.load(open('/app/data/annotations.json')), indent=2))"

# List images
ls -lh /app/data/images/
```

### Mount and Test
```bash
# Mount locally for testing
docker run -v $(pwd)/data:/data -it ocr-sample:latest python

# In Python:
from PIL import Image
img = Image.open('/data/images/sample_001.jpg')
print(img.size)
```

### Docker Compose Commands
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f ocr-sample

# Health check
docker-compose ps

# Stop services
docker-compose down

# Remove image
docker-compose down --rmi all
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OCR_DATA_PATH` | `/app/data` | Dataset location |
| `OCR_CONFIG_PATH` | `/app/config` | Config location |
| `PYTHONUNBUFFERED` | `1` | Python output buffering |

## Volumes

| Volume | Mount Point | Access | Purpose |
|--------|------------|--------|---------|
| `./data` | `/app/data` | Read-only | Dataset files |
| `./config` | `/app/config` | Read-only | Configuration |
| `./docs` | `/app/docs` | Read-only | Documentation |
| `./scripts` | `/app/scripts` | Read-write | Custom scripts |

## Networking

- Network: `ocr-network` (bridge)
- Container Port: 8000 (internal)
- Host Port: 8001 (mapped)
- Container Name: `ocr-sample-data`

## Health Checks

Container includes automatic health checks:
- **Interval**: 30 seconds
- **Timeout**: 10 seconds
- **Retries**: 3
- **Start Period**: 5 seconds
- **Test**: Validates `annotations.json` can be loaded

## Building from Scratch

```bash
# Build with custom tag
docker build -t ocr-sample:v1.0 .

# Build with buildkit (faster)
DOCKER_BUILDKIT=1 docker build -t ocr-sample:latest .

# Build and push to registry
docker tag ocr-sample:latest myregistry/ocr-sample:latest
docker push myregistry/ocr-sample:latest
```

## Common Tasks

### Add Custom Scripts
```bash
# Place script in ./scripts/
cp my_inference.py scripts/

# Run in container
docker-compose exec ocr-sample python scripts/my_inference.py
```

### Extract Dataset
```bash
# Copy data to host
docker cp ocr-sample-data:/app/data ./exported_data

# Or mount and use directly
docker-compose exec ocr-sample cp -r /app/data ./backup
```

### Update Dependencies
```bash
# Modify config/requirements.txt
echo "new-package>=1.0" >> config/requirements.txt

# Rebuild image
docker-compose build --no-cache

# Restart
docker-compose up -d
```

## Troubleshooting

### Container won't start
```bash
# Check logs
docker-compose logs ocr-sample

# Rebuild
docker-compose build --no-cache
docker-compose up -d
```

### Permission issues
```bash
# Add user to docker group (Linux)
sudo usermod -aG docker $USER
newgrp docker

# Or use sudo
sudo docker-compose up
```

### Data not mounting
```bash
# Verify paths
docker-compose exec ocr-sample ls -la /app/data

# Check volume mounts
docker inspect ocr-sample-data | grep -A 5 Mounts
```

## Performance Tips

1. **Use bind mounts** for local data access (faster than copy)
2. **Read-only volumes** where possible for security
3. **Health checks** ensure container availability
4. **Restart policy** keeps container running after failures

## Security

- ✅ Non-root user in base image
- ✅ Minimal dependencies (slim image)
- ✅ Read-only mounts for data
- ✅ Health checks for validation
- ✅ Network isolation (bridge network)

## Cleanup

```bash
# Remove container
docker-compose down

# Remove image
docker rmi ocr-sample:latest

# Remove all unused resources
docker system prune -a
```

---

**Created**: 2025-12-11
**Container Image**: ocr-sample:latest
**Python Version**: 3.11
**Base Image**: python:3.11-slim
