import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

API_PREFIX = "/api"
DEFAULT_CHECKPOINT_ROOT = Path("outputs/experiments/train/ocr")

app = FastAPI(title="OCR Inference Console Backend", version="0.1.0")

# Allow local dev CORS; tighten for prod as needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InferenceRequest(BaseModel):
    checkpoint_path: Optional[str] = Field(default=None, description="Path to the model checkpoint")
    image_base64: str = Field(..., description="Image content encoded as base64 data URL")
    confidence_threshold: float = 0.1
    nms_threshold: float = 0.4


class Region(BaseModel):
    polygon: List[List[float]]
    confidence: float
    text: Optional[str] = None


class InferenceMeta(BaseModel):
    coordinate_system: str = "pixel"
    processed_size: List[int] = [0, 0]
    original_size: List[int] = [0, 0]
    padding: dict = {"top": 0, "bottom": 0, "left": 0, "right": 0}
    scale: float = 1.0


class InferenceResponse(BaseModel):
    status: str = "success"
    regions: List[Region] = []
    meta: InferenceMeta = InferenceMeta()
    preview_image_base64: Optional[str] = None


class Checkpoint(BaseModel):
    checkpoint_path: str
    display_name: str
    size_mb: float
    modified_at: str


def _discover_checkpoints(limit: int = 100) -> List[Checkpoint]:
    if not DEFAULT_CHECKPOINT_ROOT.exists():
        logger.warning("Checkpoint root missing: %s", DEFAULT_CHECKPOINT_ROOT)
        return []

    ckpts = sorted(
        DEFAULT_CHECKPOINT_ROOT.rglob("*.ckpt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    results: List[Checkpoint] = []
    for p in ckpts[:limit]:
        stat = p.stat()
        display_name = str(p.relative_to(DEFAULT_CHECKPOINT_ROOT))
        results.append(
            Checkpoint(
                checkpoint_path=str(p),
                display_name=display_name,
                size_mb=round(stat.st_size / (1024 * 1024), 2),
                modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            )
        )
    return results


@app.get(f"{API_PREFIX}/health")
async def health():
    return {"status": "ok", "checkpoint_root": str(DEFAULT_CHECKPOINT_ROOT)}


@app.get(f"{API_PREFIX}/inference/checkpoints", response_model=List[Checkpoint])
async def list_checkpoints(limit: int = 100):
    checkpoints = _discover_checkpoints(limit=limit)
    return checkpoints


@app.post(f"{API_PREFIX}/inference/preview", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    # Placeholder inference response; wire real model here.
    if request.checkpoint_path:
        ckpt_path = Path(request.checkpoint_path)
        if not ckpt_path.exists():
            raise HTTPException(status_code=400, detail="Checkpoint path not found")
    return InferenceResponse(
        status="success",
        regions=[],
        meta=InferenceMeta(),
        preview_image_base64=None,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("BACKEND_HOST", "127.0.0.1"),
        port=int(os.getenv("BACKEND_PORT", "8002")),
        reload=True,
    )
