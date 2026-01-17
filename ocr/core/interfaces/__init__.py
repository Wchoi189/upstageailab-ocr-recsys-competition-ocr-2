"""Core Interface Layer - Domain-agnostic data contracts."""
from .schemas import Box, DetectionResult, RecognitionResult, PageResult

__all__ = ["Box", "DetectionResult", "RecognitionResult", "PageResult"]
