"""Pydantic models for preprocessing configuration.

Type-safe configuration models with validation.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class BackgroundRemovalModel(str, Enum):
    """Available background removal models."""

    U2NET = "u2net"
    U2NETP = "u2netp"
    SILUETA = "silueta"


class EnhancementMethod(str, Enum):
    """Enhancement intensity levels."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class BackgroundRemovalConfig(BaseModel):
    """Background removal parameters."""

    enable: bool = False
    model: BackgroundRemovalModel = BackgroundRemovalModel.U2NET
    alpha_matting: bool = True

    # Advanced settings
    foreground_threshold: int = Field(default=240, ge=0, le=255)
    background_threshold: int = Field(default=10, ge=0, le=255)
    erode_size: int = Field(default=10, ge=1, le=50)

    @field_validator("alpha_matting")
    @classmethod
    def validate_alpha_matting(cls, v: bool, info) -> bool:
        """Validate alpha matting requires enable."""
        if v and not info.data.get("enable", False):
            raise ValueError("Alpha matting requires background removal to be enabled")
        return v


class DocumentDetectionConfig(BaseModel):
    """Document detection parameters."""

    enable: bool = True
    min_area_ratio: float = Field(default=0.18, ge=0.01, le=0.95)
    use_adaptive: bool = True
    use_fallback_box: bool = True

    @field_validator("min_area_ratio")
    @classmethod
    def validate_min_area_ratio(cls, v: float, info) -> float:
        """Validate min_area_ratio is positive when enabled."""
        if info.data.get("enable", False) and v <= 0:
            raise ValueError("Min area ratio must be positive when detection is enabled")
        return v


class PerspectiveCorrectionConfig(BaseModel):
    """Perspective correction parameters."""

    enable: bool = True
    use_doctr_geometry: bool = False


class OrientationCorrectionConfig(BaseModel):
    """Orientation correction parameters."""

    enable: bool = False
    angle_threshold: float = Field(default=2.0, ge=0.5, le=10.0)


class NoiseEliminationConfig(BaseModel):
    """Noise elimination parameters."""

    enable: bool = False


class BrightnessAdjustmentConfig(BaseModel):
    """Brightness adjustment parameters."""

    enable: bool = False


class EnhancementConfig(BaseModel):
    """Image enhancement parameters."""

    enable: bool = False
    method: EnhancementMethod = EnhancementMethod.CONSERVATIVE

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: EnhancementMethod, info) -> EnhancementMethod:
        """Validate method can only be set when enabled."""
        if v and not info.data.get("enable", False):
            raise ValueError("Enhancement must be enabled to set enhancement method")
        return v


class PreprocessingConfig(BaseModel):
    """Complete preprocessing configuration with validation.

    This model ensures type safety and validates all preprocessing parameters
    according to business rules defined in the schema.
    """

    background_removal: BackgroundRemovalConfig = Field(default_factory=BackgroundRemovalConfig)
    document_detection: DocumentDetectionConfig = Field(default_factory=DocumentDetectionConfig)
    perspective_correction: PerspectiveCorrectionConfig = Field(default_factory=PerspectiveCorrectionConfig)
    orientation_correction: OrientationCorrectionConfig = Field(default_factory=OrientationCorrectionConfig)
    noise_elimination: NoiseEliminationConfig = Field(default_factory=NoiseEliminationConfig)
    brightness_adjustment: BrightnessAdjustmentConfig = Field(default_factory=BrightnessAdjustmentConfig)
    enhancement: EnhancementConfig = Field(default_factory=EnhancementConfig)

    @classmethod
    def from_dict(cls, config_dict: dict) -> PreprocessingConfig:
        """Create PreprocessingConfig from dictionary.

        Args:
            config_dict: Configuration dictionary (from YAML or session state)

        Returns:
            Validated PreprocessingConfig instance

        Raises:
            ValidationError: If config doesn't match schema
        """
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization.

        Returns:
            Configuration as dictionary
        """
        return self.model_dump()

    def to_yaml_dict(self) -> dict:
        """Convert to YAML-friendly dictionary.

        Returns:
            Configuration with enum values as strings
        """
        result = self.model_dump()

        # Convert enums to strings for YAML serialization
        if "background_removal" in result:
            result["background_removal"]["model"] = result["background_removal"]["model"].value

        if "enhancement" in result:
            result["enhancement"]["method"] = result["enhancement"]["method"].value

        return result
