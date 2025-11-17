---
title: "Pipeline Preview Contract"
status: "draft"
---

# Pipeline Preview & Fallback Contract

## Overview

The SPA communicates with FastAPI endpoints in `services/playground_api` to coordinate client-side workers and backend fallbacks. This contract governs `/api/pipelines/*` and `/api/commands/*` interactions.

## `/api/pipelines/preview`

- **Method**: `POST`
- **Body**:
  ```json
  {
    "pipeline_id": "preprocessing",
    "checkpoint_path": "outputs/exp/checkpoints/epoch=12.ckpt",
    "image_base64": "<base64>",
    "image_path": "data/datasets/images/val/sample.jpg",
    "params": {
      "autocontrast": true,
      "rembg": "client",
      "blur": { "kernelSize": 5 }
    }
  }
  ```
- **Response**:
  ```json
  {
    "status": "accepted",
    "job_id": "preprocessing-a1b2c3",
    "routed_backend": "client-workers",
    "cache_key": "preprocessing:91ff34aa2211",
    "notes": [
      "Client should use cache_key to reuse previously computed previews."
    ]
  }
  ```
- **Client Obligations**:
  - Use `cache_key` as the lookup for worker-level memoization.
  - If UI detects queue saturation or device downgrade, send same payload but set `params.background_removal = "server"` to request fallback immediately.

## `/api/pipelines/fallback`

- **Method**: `POST`
- **Body**:
  ```json
  {
    "pipeline_id": "rembg",
    "image_path": "data/datasets/images/val/sample.jpg",
    "params": { "strength": 0.85 }
  }
  ```
- **Response**:
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
- Future implementation will store processed artifacts under `outputs/playground/{pipeline_id}/{job_id}.png` and stream WebSocket updates.

## `/api/commands/build`

- Already implemented in `command_builder.py`.
- Frontend should call this endpoint whenever a form changes (debounced), then stream the resulting command string to the console drawer.

## Error Handling

- All endpoints return `HTTP 4xx` for validation errors with descriptive `detail`.
- Worker hosts must propagate `taskId` so the SPA can correlate responses.

## Caching & Telemetry

- `cache_key` ensures `(image, params)` combos can be reused across both client and backend paths.
- Workers must emit telemetry (queue depth, durations) and optionally POST them to `/api/metrics` (future endpoint) for observability.


