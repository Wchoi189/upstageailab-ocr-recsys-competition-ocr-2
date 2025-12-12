"""FastAPI application that powers the high-performance playground services."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ocr.utils.path_utils import setup_project_paths

from .routers import command_builder, evaluation, inference, metrics, pipeline
from apps.backend.services import ocr_bridge

LOGGER = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Factory for the playground API FastAPI application."""
    app = FastAPI(
        title="OCR Playground API",
        version="0.1.0",
        description=(
            "Backend services for the Albumentations-style playground. " "Provides command builder, inference, and evaluation endpoints."
        ),
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(command_builder.router, prefix="/api/commands", tags=["command-builder"])
    app.include_router(inference.router, prefix="/api/inference", tags=["inference"])
    app.include_router(pipeline.router, prefix="/api/pipelines", tags=["pipelines"])
    app.include_router(evaluation.router, prefix="/api/evaluation", tags=["evaluation"])
    app.include_router(metrics.router, prefix="/api/metrics", tags=["metrics"])
    app.include_router(ocr_bridge.router)

    @app.on_event("startup")
    def setup_paths_on_startup() -> None:
        """Initialize paths from environment variables on FastAPI startup.

        This ensures that path configuration is consistent across all modules
        and supports environment variable overrides for deployment scenarios.
        """
        resolver = setup_project_paths()  # Reads OCR_* env vars if set

        # Log path configuration for debugging
        LOGGER.info("=== Path Configuration ===")
        LOGGER.info(f"Project root: {resolver.config.project_root}")
        LOGGER.info(f"Config directory: {resolver.config.config_dir}")
        LOGGER.info(f"Output directory: {resolver.config.output_dir}")
        LOGGER.info(f"Data directory: {resolver.config.data_dir}")

        # Check if environment variables were used
        import os

        env_vars_used = [
            name
            for name in os.environ.keys()
            if name.startswith("OCR_") and name != "OCR_PROJECT_ROOT"  # OCR_PROJECT_ROOT is always checked
        ]
        if env_vars_used:
            LOGGER.info(f"Using environment variables: {', '.join(sorted(env_vars_used))}")
        else:
            LOGGER.info("Using auto-detected paths (no environment variables set)")

    return app


app = create_app()
