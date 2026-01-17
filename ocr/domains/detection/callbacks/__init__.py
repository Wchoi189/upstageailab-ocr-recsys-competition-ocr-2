# ocr/domains/detection/callbacks/__init__.py
"""Detection domain callbacks for training and logging."""

from ocr.domains.detection.callbacks.wandb import log_validation_images

__all__ = ["log_validation_images"]
