from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import io
from PIL import Image
import os
import glob

# Import the bridge
try:
    from .ocr_bridge import OCRBridge
except ImportError:
    # Fallback for when running as script
    from ocr_bridge import OCRBridge


app = FastAPI(title="OCR Inference API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global bridge instance
bridge = None
# Path to best checkpoint - hardcoded for now or env var
CHECKPOINT_PATH = os.getenv("OCR_CHECKPOINT_PATH", "../../outputs/ocr_training_b/checkpoints/best.ckpt")

@app.on_event("startup")
async def startup_event():
    global bridge
    if os.path.exists(CHECKPOINT_PATH):
        try:
            bridge = OCRBridge(CHECKPOINT_PATH)
            print(f"Model loaded from {CHECKPOINT_PATH}")
        except Exception as e:
            print(f"Failed to load model: {e}")
    else:
        print(f"Checkpoint not found at {CHECKPOINT_PATH}")

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": bridge is not None,
        "device": bridge.device if bridge else "N/A"
    }

@app.get("/models")
def list_models():
    # Helper to find checkpoints
    # This is a simple glob, adjust path as needed
    base_dir = "outputs/experiments/train/ocr"
    checkpoints = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".ckpt"):
                checkpoints.append({"id": file, "path": os.path.join(root, file)})
    return checkpoints[0:10] # limit 10

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not bridge:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        boxes, scores = bridge.predict(image)

        # boxes is list of numpy arrays (N, 2)
        # Convert to list of lists
        predictions = []
        for box, score in zip(boxes, scores):
            predictions.append({
                "points": box.tolist(), # [[x,y], [x,y]...]
                "confidence": float(score)
            })

        return {
            "filename": file.filename,
            "predictions": predictions
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
