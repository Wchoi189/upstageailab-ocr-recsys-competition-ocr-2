"""Error response models for structured API error handling."""

from __future__ import annotations

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """Structured error response for API endpoints.

    Provides machine-readable error codes and human-readable messages
    for consistent error handling across the API.
    """

    error_code: str
    message: str
    details: dict = {}
    request_id: str | None = None
