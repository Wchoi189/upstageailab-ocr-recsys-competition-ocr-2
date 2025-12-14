"""Playground Console Backend - FastAPI Server

Port 8001 | Domain: ML Playground & Experimentation
Serves Playground console frontend with inference, commands, and evaluation endpoints.
Uses shared InferenceEngine from apps.shared.backend_shared.

See: docs/guides/setting-up-app-backends.md
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.shared.backend_shared.inference import InferenceEngine

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

API_PREFIX = "/api/v1"

# Global inference engine (lazy loaded)
_inference_engine: InferenceEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for app startup/shutdown."""
    global _inference_engine

    logger.info("🚀 Starting Playground Console Backend (Port 8001)")

    # Initialize engine (model loads on first inference request)
    _inference_engine = InferenceEngine()
    logger.info("✅ InferenceEngine initialized (lazy loading enabled)")

    yield

    logger.info("🛑 Shutting down Playground Console Backend")


app = FastAPI(
    title="Playground Console Backend",
    description="RESTful API for ML playground, experimentation, and inference",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for local dev
env = os.getenv("ENV", "development")
if env == "production":
    allowed_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
else:
    allowed_origins = [
        "http://localhost:3000",  # Next.js dev server
        "http://127.0.0.1:3000",
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(f"{API_PREFIX}/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "playground-console-backend",
        "port": int(os.getenv("BACKEND_PORT", "8001")),
        "engine_loaded": _inference_engine is not None,
    }


# Import and include routers (to be implemented)
# from .routers import inference, commands, evaluation, checkpoints
# app.include_router(inference.router, prefix=f"{API_PREFIX}/inference", tags=["inference"])
# app.include_router(commands.router, prefix=f"{API_PREFIX}/commands", tags=["commands"])
# app.include_router(evaluation.router, prefix=f"{API_PREFIX}/evaluation", tags=["evaluation"])
# app.include_router(checkpoints.router, prefix=f"{API_PREFIX}/checkpoints", tags=["checkpoints"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("BACKEND_HOST", "127.0.0.1"),
        port=int(os.getenv("BACKEND_PORT", "8001")),
        reload=True,
    )
