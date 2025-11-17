"""Path utilities shared by the playground API modules."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

__all__ = ["PROJECT_ROOT"]

