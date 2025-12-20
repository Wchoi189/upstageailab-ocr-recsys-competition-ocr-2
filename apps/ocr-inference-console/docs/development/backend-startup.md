
# Backend Startup Guide for OCR Inference Console

## Quick Start

### Option 1: Start Both Frontend and Backend Together (Recommended)
```bash
make serve-ocr-console
```
This command:
- Auto-detects the latest checkpoint
- Starts the backend on port 8000
- Waits for backend to be ready (up to 30 seconds)
- Starts the frontend on port 5173

### Option 2: Start Backend Separately (For Debugging)

**For OCR Inference Console:**
```bash
make backend-ocr
```
This command:
- Auto-detects the latest checkpoint
- Starts the backend with OCR endpoints
- Uses the same backend as playground-console (includes both OCR and playground APIs)

**Alternative (if you need to specify checkpoint manually):**
```bash
export OCR_CHECKPOINT_PATH="outputs/experiments/train/ocr/.../epoch-14.ckpt"
make backend-dev
```

### Option 3: Start Frontend Separately
```bash
make ocr-console-dev
# or
cd apps/ocr-inference-console && npm run dev
```

## Backend Architecture

Both `backend-dev` and `backend-ocr` use the **same backend application**:
- **App**: `apps.backend.services.playground_api.app:app`
- **Includes**: Both OCR endpoints (`/ocr/*`) and Playground endpoints (`/api/*`)
- **Port**: 8000 (default)

The playground API backend includes the OCR bridge router, so it serves both consoles.

## Troubleshooting

### Backend Takes Too Long to Start

The backend may take 10-30 seconds to start because:
1. **Model Loading**: The OCR model loads on first inference request (lazy loading)
2. **Checkpoint Detection**: Auto-detection scans for checkpoints
3. **Path Setup**: Project paths are initialized on startup

**Solutions:**
- Wait for "Application startup complete" message
- Check backend logs for errors
- Verify checkpoint path is correct: `echo $OCR_CHECKPOINT_PATH`

### Backend Becomes Unresponsive

If the terminal becomes unresponsive:
1. **Stop the backend**: Press `Ctrl+C` once and wait 5-10 seconds
2. **Force stop**: `make backend-stop`
3. **Force kill**: `make backend-force-kill` (last resort)
4. **Check for hanging processes**: `make status-backend` or `lsof -i:8000`

### Port Already in Use

```bash
# Check what's using port 8000
lsof -i:8000

# Stop the backend
make backend-stop

# Or force kill
make backend-force-kill
```

### Checkpoint Not Found

```bash
# Find available checkpoints
find outputs/experiments/train/ocr -name "*.ckpt" | head -n 5

# Set manually
export OCR_CHECKPOINT_PATH="outputs/experiments/train/ocr/.../epoch-14.ckpt"
make backend-ocr
```

## Backend Endpoints

Once the backend is running, you can access:

- **Health Check**: `http://localhost:8000/ocr/health`
- **API Docs**: `http://localhost:8000/docs`
- **OCR Predict**: `POST http://localhost:8000/ocr/predict`
- **List Checkpoints**: `GET http://localhost:8000/ocr/checkpoints`

## Development Workflow

1. **Start backend in one terminal:**
   ```bash
   make backend-ocr
   ```

2. **Start frontend in another terminal:**
   ```bash
   make ocr-console-dev
   ```

3. **Or use the combined command:**
   ```bash
   make serve-ocr-console
   ```

## Differences from Playground Console

- **OCR Inference Console** uses `/ocr/*` endpoints
- **Playground Console** uses `/api/*` endpoints
- Both are served by the same backend (`apps.backend.services.playground_api.app:app`)
- The backend includes both routers, so either console can use it
