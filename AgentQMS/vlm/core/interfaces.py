"""Abstract Interfaces for VLM Module.

Defines contracts for backend implementations and core components.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from AgentQMS.vlm.core.contracts import AnalysisMode, AnalysisResult, BackendConfig, ProcessedImage


class VLMBackend(ABC):
    """Abstract base class for VLM backend implementations."""

    def __init__(self, config: BackendConfig):
        """Initialize backend with configuration.

        Args:
            config: Backend configuration
        """
        self.config = config

    @abstractmethod
    def analyze_image(
        self,
        image_data: ProcessedImage,
        prompt: str,
        mode: AnalysisMode,
        **kwargs: Any,
    ) -> str:
        """Analyze an image using the VLM backend.

        Args:
            image_data: Preprocessed image data
            prompt: Analysis prompt
            mode: Analysis mode
            **kwargs: Additional backend-specific parameters

        Returns:
            Analysis text from VLM

        Raises:
            BackendError: If analysis fails
        """
        pass

    @abstractmethod
    def supports_batch(self) -> bool:
        """Check if backend supports batch processing.

        Returns:
            True if batch processing is supported
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available and configured.

        Returns:
            True if backend can be used
        """
        pass

    @abstractmethod
    def get_max_resolution(self) -> int:
        """Get maximum supported image resolution.

        Returns:
            Maximum resolution (width or height) in pixels
        """
        pass


class ImagePreprocessor(ABC):
    """Abstract base class for image preprocessing."""

    @abstractmethod
    def preprocess(
        self,
        image_path: Path,
        max_resolution: int,
        target_format: Optional[str] = None,
    ) -> ProcessedImage:
        """Preprocess an image for VLM analysis.

        Args:
            image_path: Path to input image
            max_resolution: Maximum resolution (width or height)
            target_format: Target image format (JPEG, PNG, etc.)

        Returns:
            Processed image data

        Raises:
            PreprocessingError: If preprocessing fails
        """
        pass

    @abstractmethod
    def preprocess_batch(
        self,
        image_paths: List[Path],
        max_resolution: int,
        target_format: Optional[str] = None,
    ) -> List[ProcessedImage]:
        """Preprocess multiple images.

        Args:
            image_paths: List of image paths
            max_resolution: Maximum resolution (width or height)
            target_format: Target image format

        Returns:
            List of processed image data
        """
        pass


class ReportIntegrator(ABC):
    """Abstract base class for report integration."""

    @abstractmethod
    def populate_report(
        self,
        report_path: Path,
        analysis_results: List[AnalysisResult],
        via_annotations: Optional[Path] = None,
    ) -> None:
        """Populate a report with analysis results.

        Args:
            report_path: Path to report file
            analysis_results: List of analysis results to include
            via_annotations: Optional path to VIA annotations

        Raises:
            IntegrationError: If report population fails
        """
        pass

    @abstractmethod
    def supports_report_type(self, report_path: Path) -> bool:
        """Check if report type is supported.

        Args:
            report_path: Path to report file

        Returns:
            True if report type is supported
        """
        pass


class BackendError(Exception):
    """Base exception for backend errors."""

    pass


class PreprocessingError(Exception):
    """Base exception for preprocessing errors."""

    pass


class IntegrationError(Exception):
    """Base exception for integration errors."""

    pass
