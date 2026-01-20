"""
DEPRECATED: Legacy import path compatibility shim.

This module provides backward compatibility for code importing from the old path:
    from ocr.data.datasets.db_collate_fn import DBCollateFN

The actual implementation has been moved to:
    ocr.domains.detection.data.collate_db.DBCollateFN

⚠️  WARNING: This compatibility shim will be REMOVED in v0.4.0
⚠️  Update your imports to use the new path

New import (use this):
    from ocr.domains.detection.data.collate_db import DBCollateFN
"""

import warnings

warnings.warn(
    "Importing from 'ocr.data.datasets.db_collate_fn' is deprecated. "
    "Use 'ocr.domains.detection.data.collate_db' instead. "
    "This compatibility shim will be removed in v0.4.0.",
    DeprecationWarning,
    stacklevel=2
)

from ocr.domains.detection.data.collate_db import DBCollateFN

__all__ = ["DBCollateFN"]
