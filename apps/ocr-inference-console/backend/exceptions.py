"""Structured exception hierarchy for OCR backend."""

from __future__ import annotations


class OCRBackendError(Exception):
    """Base exception for OCR backend errors.

    All OCR backend exceptions should inherit from this class.
    """

    def __init__(self, error_code: str, message: str, details: dict | None = None):
        """Initialize OCR backend error.

        Args:
            error_code: Machine-readable error code (e.g., "CHECKPOINT_NOT_FOUND")
            message: Human-readable error message
            details: Optional additional error details
        """
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.details = details or {}


class CheckpointNotFoundError(OCRBackendError):
    """Raised when a requested checkpoint cannot be found."""

    def __init__(self, checkpoint_path: str):
        """Initialize checkpoint not found error.

        Args:
            checkpoint_path: Path to the checkpoint that was not found
        """
        super().__init__(
            error_code="CHECKPOINT_NOT_FOUND",
            message=f"Checkpoint not found: {checkpoint_path}",
            details={"checkpoint_path": checkpoint_path},
        )


class ImageDecodingError(OCRBackendError):
    """Raised when image decoding fails."""

    def __init__(self, reason: str):
        """Initialize image decoding error.

        Args:
            reason: Reason for the decoding failure
        """
        super().__init__(
            error_code="IMAGE_DECODING_ERROR",
            message=f"Failed to decode image: {reason}",
            details={"reason": reason},
        )


class InferenceError(OCRBackendError):
    """Raised when inference execution fails."""

    def __init__(self, reason: str):
        """Initialize inference error.

        Args:
            reason: Reason for the inference failure
        """
        super().__init__(
            error_code="INFERENCE_ERROR",
            message=f"Inference failed: {reason}",
            details={"reason": reason},
        )


class ModelLoadError(OCRBackendError):
    """Raised when model checkpoint loading fails."""

    def __init__(self, checkpoint_path: str, reason: str):
        """Initialize model load error.

        Args:
            checkpoint_path: Path to the checkpoint that failed to load
            reason: Reason for the loading failure
        """
        super().__init__(
            error_code="MODEL_LOAD_ERROR",
            message=f"Failed to load model from {checkpoint_path}: {reason}",
            details={"checkpoint_path": checkpoint_path, "reason": reason},
        )


class ServiceNotInitializedError(OCRBackendError):
    """Raised when a required service is not initialized."""

    def __init__(self, service_name: str):
        """Initialize service not initialized error.

        Args:
            service_name: Name of the service that was not initialized
        """
        super().__init__(
            error_code="SERVICE_NOT_INITIALIZED",
            message=f"Service not initialized: {service_name}",
            details={"service_name": service_name},
        )
