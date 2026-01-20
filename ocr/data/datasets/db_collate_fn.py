"""
Legacy import path compatibility shim.

This module provides backward compatibility for code importing from the old path:
    from ocr.data.datasets.db_collate_fn import DBCollateFN

The actual implementation has been moved to:
    ocr.domains.detection.data.collate_db.DBCollateFN

This shim will be deprecated in a future release.
"""

from ocr.domains.detection.data.collate_db import DBCollateFN

__all__ = ["DBCollateFN"]
