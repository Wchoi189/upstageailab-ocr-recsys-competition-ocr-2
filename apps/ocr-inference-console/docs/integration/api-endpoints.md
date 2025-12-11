# API Endpoints

The Inference Console backend exposes a REST API built with FastAPI.

## Base URL
`http://localhost:8000`

## Endpoints

### Health Check
**GET** `/health`
- **Description**: Returns the status of the API and the loaded model.
- **Response**:
  ```json
  {
    "status": "ok",
    "model_loaded": true,
    "device": "cuda:0"
  }
  ```

### Inference
**POST** `/predict`
- **Description**: Run OCR on an uploaded image.
- **Content-Type**: `multipart/form-data`
- **Parameters**:
    - `file`: The image file (jpg, png).
- **Response**:
  ```json
  {
    "filename": "image.jpg",
    "predictions": [
      {
        "points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
        "confidence": 0.95
      }
    ]
  }
  ```

### Models
**GET** `/models`
- **Description**: List available checkpoints.
- **Response**:
  ```json
  [
    {"id": "best.ckpt", "path": "outputs/.../best.ckpt"}
  ]
  ```

