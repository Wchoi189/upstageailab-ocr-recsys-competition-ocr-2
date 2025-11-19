# Testing New Features

This document describes how to test the newly implemented features.

## ✅ Features Implemented

1. **Task 1.2: ONNX.js rembg Integration** - Client-side background removal
2. **Task 3.2: Missing API Endpoints** - Pipeline job status and gallery image management

---

## 🧪 Testing Checklist

### 1. TypeScript Type Checking ✅

```bash
cd frontend
npm run type-check
```

**Status**: ✅ PASSED - No type errors

---

### 2. ONNX Model File ✅

```bash
# Verify model file exists
ls -lh frontend/public/models/u2net.onnx
```

**Status**: ✅ VERIFIED - Model file exists (168MB)

---

### 3. Backend API Endpoints

#### Start the API Server

```bash
# Terminal 1: Start the API server
uv run uvicorn services.playground_api.app:app --reload
```

#### Test Pipeline Status Endpoint

```bash
# Terminal 2: Run test script
uv run python tests/scripts/test_new_features.py
```

Or manually test with curl:

```bash
# 1. Create a pipeline preview job
curl -X POST http://127.0.0.1:8000/api/pipelines/preview \
  -H "Content-Type: application/json" \
  -d '{
    "pipeline_id": "test-pipeline",
    "image_path": "data/datasets/images/val/sample.jpg",
    "params": {"autocontrast": true}
  }'

# This returns a job_id. Use it in the next command:
# 2. Check job status
curl http://127.0.0.1:8000/api/pipelines/status/{job_id}
```

#### Test Gallery Images Endpoint

```bash
# List gallery images
curl http://127.0.0.1:8000/api/evaluation/gallery-images?limit=10

# Get gallery root
curl http://127.0.0.1:8000/api/evaluation/gallery-root
```

#### Interactive API Docs

```bash
# Open in browser
open http://127.0.0.1:8000/docs
```

Navigate to:
- `/api/pipelines/status/{job_id}` - Test GET endpoint
- `/api/evaluation/gallery-images` - Test GET endpoint with query params

---

### 4. Frontend ONNX.js Integration

#### Start Frontend Dev Server

```bash
# Terminal 2: Start frontend
cd frontend
npm run dev
```

#### Test Background Removal

1. Open browser: `http://localhost:5173/preprocessing`
2. Upload an image
3. Enable "Background Removal" toggle
4. Verify:
   - ✅ Image is processed client-side using ONNX.js
   - ✅ Processing time is displayed
   - ✅ Result shows background removed with white background
   - ✅ Console shows no errors (check DevTools)

#### Check Browser Console

Open DevTools (F12) and check:
- ✅ No errors loading ONNX model from `/models/u2net.onnx`
- ✅ No WASM loading errors
- ✅ Worker successfully processes rembg task
- ✅ Fallback message if ONNX inference fails (should fallback to autocontrast)

---

### 5. E2E Tests

```bash
# Run preprocessing E2E tests (includes rembg test)
cd frontend
npm run test:e2e preprocessing.spec.ts
```

**Expected Tests**:
- ✅ Background removal toggle works
- ✅ Processing completes within timeout
- ✅ Canvas updates with processed image

---

### 6. Integration Tests

```bash
# Test backend endpoints (requires API server running)
uv run python tests/scripts/test_new_features.py
```

**Tests Included**:
- ✅ Pipeline preview creates job
- ✅ Pipeline status endpoint returns job info
- ✅ Gallery images endpoint lists images
- ✅ Gallery root endpoint returns path

---

## 🐛 Known Issues & Limitations

### ESLint Warnings (Pre-existing)

Some ESLint warnings exist but are **not related to new features**:
- Unused variables in some components
- React hooks warnings in `PreprocessingCanvas.tsx`

These can be fixed separately.

### ONNX.js Limitations

1. **Model Size**: 168MB model needs to be downloaded on first use
2. **Performance**: Large images (>2048px) should fallback to server-side
3. **WASM Loading**: First inference may be slower due to WASM initialization

### Testing Notes

- Backend API server must be running for endpoint tests
- Frontend dev server must be running for E2E tests
- Browser must support Web Workers and WASM for ONNX.js

---

## 📊 Test Results Summary

| Feature | Type Check | File Exists | Implementation | E2E Test | Status |
|---------|-----------|-------------|----------------|----------|--------|
| ONNX.js rembg | ✅ | ✅ | ✅ | ⏳ | Ready |
| Pipeline Status | ✅ | ✅ | ✅ | ⏳ | Ready |
| Gallery Images | ✅ | ✅ | ✅ | ⏳ | Ready |

**Legend**:
- ✅ = Verified/Complete
- ⏳ = Requires running server/app

---

## 🚀 Quick Test Commands

```bash
# All-in-one test script (requires API server)
uv run python tests/scripts/test_new_features.py

# Type checking
cd frontend && npm run type-check

# Linting (has some pre-existing warnings)
cd frontend && npm run lint

# E2E tests (requires both servers running)
cd frontend && npm run test:e2e
```

---

## 📝 Next Steps

1. **Manual Testing**: Start both servers and test in browser
2. **E2E Tests**: Run full E2E test suite
3. **Performance Testing**: Test with various image sizes
4. **Error Handling**: Test fallback mechanisms

---

**Created**: 2025-11-20
**Last Updated**: 2025-11-20

