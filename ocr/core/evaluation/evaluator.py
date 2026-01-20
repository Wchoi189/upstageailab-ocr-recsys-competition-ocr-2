"""\nDEPRECATED: 
Legacy import path compatibility shim.

This module provides backward compatibility for code importing from the old path:
    from ocr.core.evaluation.evaluator import CLEvalEvaluator

The actual implementation has been moved to:
⚠️  WARNING: This compatibility shim will be REMOVED in v0.4.0
⚠️  Update your imports to use the new path

"""

import warnings

warnings.warn(
    "Importing from 'ocr.core.evaluation.evaluator' is deprecated. "
    "Use 'ocr.domains.detection.evaluation' instead. "
    "This compatibility shim will be removed in v0.4.0.",
    DeprecationWarning,
    stacklevel=2
)


from ocr.domains.detection.evaluation import CLEvalEvaluator

__all__ = ["CLEvalEvaluator"]
