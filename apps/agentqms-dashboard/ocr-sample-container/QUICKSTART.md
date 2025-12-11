# OCR Sample Container - Quick Start Guide

Get your OCR sample data running in Docker in under 5 minutes.

## Prerequisites

- Docker installed ([Install Docker](https://docs.docker.com/get-docker/))
- Docker Compose installed ([Install Compose](https://docs.docker.com/compose/install/))
- 500MB free disk space

## 5-Minute Setup

### Step 1: Build the Image
```bash
cd ocr-sample-container
docker build -t ocr-sample:latest .
```

**Expected Output:**
```
[+] Building ...
[+] Building ...
=> CACHED ...
=> exporting ...
 => => writing image sha256:abc123...
```

### Step 2: Start the Container
```bash
docker-compose up -d
```

**Expected Output:**
```
[+] Running 2/2
 ✓ Network ocr-network Created
 ✓ Container ocr-sample-data Started
```

### Step 3: Verify It's Working
```bash
docker-compose ps
```

**Expected Output:**
```
NAME                 IMAGE              STATUS
ocr-sample-data      ocr-sample:latest  Up 2 seconds (healthy)
```

✅ **Done!** Your container is running.

---

## Common Tasks

### View Dataset Info
```bash
docker-compose exec ocr-sample python scripts/load_ocr_data.py
```

### Access Container Shell
```bash
docker-compose exec ocr-sample bash
```

### View Logs
```bash
docker-compose logs -f
```

### Stop Container
```bash
docker-compose stop
```

### Remove Everything
```bash
docker-compose down --rmi all
```

---

## Using the Make Commands

If you prefer make commands:

```bash
# Build
make build

# Run
make run

# View logs
make logs

# Stop
make stop

# Validate data
make validate

# Open shell
make shell

# Clean up
make clean
```

---

## Accessing the Data

### From Host Machine
```bash
# Copy files from container
docker cp ocr-sample-data:/app/data ./local_data

# Or mount and access directly
docker run -v $(pwd)/data:/data -it ocr-sample:latest bash
```

### From Python (in container)
```bash
docker-compose exec ocr-sample python

# Then in Python:
>>> from pathlib import Path
>>> from PIL import Image
>>> img = Image.open('/app/data/images/sample_001.jpg')
>>> print(f"Image size: {img.size}")
```

### Access Annotations
```bash
docker-compose exec ocr-sample python -c \
  "import json; print(json.dumps(json.load(open('/app/data/annotations.json')), indent=2))"
```

---

## Troubleshooting

### Container won't start
```bash
# Check logs
docker-compose logs

# Rebuild without cache
docker-compose build --no-cache
docker-compose up -d
```

### Permission denied
```bash
# Add user to docker group (Linux/Mac)
sudo usermod -aG docker $USER
newgrp docker

# Or use sudo
sudo docker-compose up -d
```

### Out of disk space
```bash
# Clean up unused Docker resources
docker system prune -a
```

### Port already in use
```bash
# Change HOST_PORT in docker-compose.yml
# Or kill the process using the port
lsof -i :8001
kill -9 <PID>
```

---

## What's Inside

```
ocr-sample-container/
├── data/                  # Dataset files
│   ├── images/           # 3 JPEG images (39 KB)
│   └── annotations.json  # COCO format (2 KB)
├── config/               # Configuration
│   ├── config.yaml       # Inference settings
│   └── requirements.txt  # Python packages
├── docs/                 # Documentation
│   ├── OCR_INFERENCE_GUIDE.md
│   ├── OCR_SAMPLE_README.md
│   └── OCR_SAMPLE_MANIFEST.txt
├── scripts/              # Utilities
│   └── load_ocr_data.py  # Data loader
├── Dockerfile           # Container definition
├── docker-compose.yml   # Compose config
└── Makefile            # Commands
```

---

## Next Steps

1. **Run validation**: `make validate`
2. **Inspect data**: `docker-compose exec ocr-sample bash`
3. **Read guide**: `docs/OCR_INFERENCE_GUIDE.md`
4. **Add custom scripts**: Place in `scripts/`
5. **Export data**: `make export-zip` or `make export-tar`

---

## Environment Variables

Key variables (see `.env.example` for all):

| Variable | Default | Purpose |
|----------|---------|---------|
| `OCR_DATA_PATH` | `/app/data` | Dataset location |
| `OCR_CONFIG_PATH` | `/app/config` | Config location |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `DEBUG` | `false` | Debug mode |

---

## Networking

- **Network**: `ocr-network` (isolated bridge)
- **Container Port**: 8000 (internal)
- **Host Port**: 8001 (external)
- **DNS**: Container DNS is auto-configured

Access from host: `localhost:8001`

---

## Performance Tips

1. **Use M1/M2 Macs**: Enable Rosetta 2 for better compatibility
2. **Resource Limits**: Adjust in `docker-compose.yml` if needed
3. **Volume Mounts**: Use bind mounts for faster I/O
4. **Read-Only Mode**: Mount data as read-only for safety

---

## Advanced Usage

### Mount External Data
```yaml
# In docker-compose.yml
volumes:
  - /path/to/external/data:/app/data:ro
```

### Custom Network
```bash
# Create and use custom network
docker network create my-ocr-net
docker-compose -e NETWORK_NAME=my-ocr-net up -d
```

### Resource Constraints
```yaml
# In docker-compose.yml
deploy:
  resources:
    limits:
      cpus: '1.0'
      memory: 512M
```

---

## Support

**Having issues?** Check:
1. Docker is running: `docker ps`
2. Logs: `docker-compose logs`
3. Health: `docker-compose ps`
4. README.md for detailed info

**Need to modify?** Edit:
- `Dockerfile` - change base image or dependencies
- `docker-compose.yml` - ports, volumes, env vars
- `config/requirements.txt` - Python packages
- `scripts/` - add custom scripts

---

**Created**: December 11, 2025
**Version**: 1.0
**License**: MIT
