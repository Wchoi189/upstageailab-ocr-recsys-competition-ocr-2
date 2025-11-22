"""FastAPI application that exposes stateless services for the new playground UI."""

from .app import create_app

__all__ = ["create_app"]
