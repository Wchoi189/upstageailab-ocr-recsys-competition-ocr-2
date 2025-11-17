"""FastAPI application that powers the high-performance playground services."""

from __future__ import annotations

from fastapi import FastAPI

from .routers import command_builder, evaluation, inference, pipeline


def create_app() -> FastAPI:
    """Factory for the playground API FastAPI application."""
    app = FastAPI(
        title="OCR Playground API",
        version="0.1.0",
        description=(
            "Backend services for the Albumentations-style playground. "
            "Provides command builder, inference, and evaluation endpoints."
        ),
    )

    app.include_router(command_builder.router, prefix="/api/commands", tags=["command-builder"])
    app.include_router(inference.router, prefix="/api/inference", tags=["inference"])
    app.include_router(pipeline.router, prefix="/api/pipelines", tags=["pipelines"])
    app.include_router(evaluation.router, prefix="/api/evaluation", tags=["evaluation"])

    return app


app = create_app()


