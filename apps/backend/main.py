from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from apps.backend.services.ocr_bridge import router as ocr_router

app = FastAPI(title="OCR Inference Backend")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for development console
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(ocr_router)

@app.get("/")
def root():
    return {"message": "OCR Inference Backend is running", "docs": "/docs"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
