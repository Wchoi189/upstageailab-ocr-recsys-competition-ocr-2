---
type: api_contract
component: backend_pipeline
status: current
version: "1.0"
last_updated: "2025-12-15"
---

# Backend Pipeline Contract

**Purpose**: SPA ↔ FastAPI contract for pipeline preview, fallback, and command building; routes `/api/pipelines/*`, `/api/commands/*`.

---

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/pipelines/preview` | POST | Request pipeline preview (client-side workers or backend fallback) |
| `/api/pipelines/fallback` | POST | Request backend fallback execution |
| `/api/commands/build` | POST | Build CLI command from form values (already implemented) |

---

## `/api/pipelines/preview`

**Request**:
```json
{
  "pipeline_id": "preprocessing",
  "checkpoint_path": "outputs/exp/checkpoints/epoch=12.ckpt",
  "image_base64": "<base64>",
  "image_path": "data/datasets/images/val/sample.jpg",
  "params": {
    "autocontrast": true,
    "rembg": "client",
    "blur": {"kernelSize": 5}
  }
}
```

**Response**:
```json
{
  "status": "accepted",
  "job_id": "preprocessing-a1b2c3",
  "routed_backend": "client-workers",
  "cache_key": "preprocessing:91ff34aa2211",
  "notes": ["Client should use cache_key to reuse previously computed previews."]
}
```

**Client Obligations**:
- Use `cache_key` for worker-level memoization
- On queue saturation or device downgrade, set `params.background_removal = "server"` for fallback

---

## `/api/pipelines/fallback`

**Request**:
```json
{
  "pipeline_id": "rembg",
  "image_path": "data/datasets/images/val/sample.jpg",
  "params": {"strength": 0.85}
}
```

**Response**:
```json
{
  "status": "accepted",
  "routed_backend": "server-rembg",
  "result_path": null,
  "notes": [
    "Backend execution not wired in yet.",
    "When implemented, store outputs under outputs/playground/{pipeline_id}/"
  ]
}
```

**Future**: Processed artifacts stored at `outputs/playground/{pipeline_id}/{job_id}.png`; WebSocket updates

---

## `/api/commands/build`

**Status**: Already implemented in `command_builder.py`

**Usage**: Frontend calls on form changes (debounced); streams command string to console drawer

---

## Error Handling

| Error Type | HTTP Status | Payload |
|------------|-------------|---------|
| **Validation Error** | 4xx | `{"detail": "<descriptive message>"}` |
| **Worker Correlation** | N/A | Worker hosts must propagate `taskId` for response correlation |

---

## Caching & Telemetry

| Feature | Implementation |
|---------|----------------|
| **Cache Key** | `(image, params)` combos reusable across client/backend paths |
| **Telemetry** | Workers emit queue depth, durations; optionally POST to `/api/metrics` (future) |

---

## Dependencies

| Component | Dependencies |
|-----------|-------------|
| **Backend API** | FastAPI, InferenceEngine, client workers |
| **Client Workers** | SPA, cache_key memoization, taskId propagation |

---

## Constraints

- **Cache Key Format**: `{pipeline_id}:{hash(image,params)}`
- **Backend Fallback**: Not fully wired; future implementation stores outputs in `outputs/playground/{pipeline_id}/`
- **WebSocket**: Future implementation for streaming updates

---

## Backward Compatibility

**Status**: Current (v1.0)

**Breaking Changes**: None (new API)

**Compatibility Matrix**:

| Interface | v1.0 | Notes |
|-----------|------|-------|
| `/api/pipelines/preview` | ✅ | Current |
| `/api/pipelines/fallback` | 🟡 | Future implementation |
| `/api/commands/build` | ✅ | Already implemented |

---

## References

- [System Architecture](../../architecture/system-architecture.md)
- [API Decoupling](../../architecture/api-decoupling.md)
- [Backend Frontend Recommendations](../../architecture/backend-frontend-architecture-recommendations.md)
