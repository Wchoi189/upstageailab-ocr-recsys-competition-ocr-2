# ocr/domains/recognition/callbacks/__init__.py
"""Recognition domain callbacks for training and logging."""

from ocr.domains.recognition.callbacks.wandb_logging import log_recognition_images

__all__ = ["log_recognition_images"]
