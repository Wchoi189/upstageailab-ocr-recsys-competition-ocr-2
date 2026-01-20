"""
Legacy import path compatibility shim.

This module provides backward compatibility for code importing from the old path:
    from ocr.core.evaluation.evaluator import CLEvalEvaluator

The actual implementation has been moved to:
    ocr.domains.detection.evaluation

This shim will be deprecated in a future release.
"""

from ocr.domains.detection.evaluation import CLEvalEvaluator

__all__ = ["CLEvalEvaluator"]
