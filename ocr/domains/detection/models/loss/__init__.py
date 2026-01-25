"""Detection loss functions."""

from ocr.domains.detection.models.loss.db_loss import DBLoss
from ocr.domains.detection.models.loss.craft_loss import CraftLoss

__all__ = ["DBLoss", "CraftLoss"]
