"""Tests for Pydantic data contracts."""

import pydantic
import pytest
from agent_qms.vlm.core.contracts import (
    AnalysisMode,
    AnalysisRequest,
    BackendConfig,
)


class TestImageData:
    """Tests for ImageData model."""

    def test_valid_image_data(self, tmp_path):
        """Test creating valid ImageData."""
        # Create a dummy image file
        image_file = tmp_path / "test.jpg"
        image_file.write_bytes(b"fake image data")

        # This would normally require actual image dimensions
        # For now, we'll test the structure
        pass  # Placeholder - would need actual image processing


class TestBackendConfig:
    """Tests for BackendConfig model."""

    def test_valid_openrouter_config(self):
        """Test creating valid OpenRouter config."""
        config = BackendConfig(
            backend_type="openrouter",
            api_key="test-key",
            model="qwen/qwen-2-vl-72b-instruct",
        )
        assert config.backend_type == "openrouter"
        assert config.api_key == "test-key"

    def test_invalid_backend_type(self):
        """Test invalid backend type raises error."""
        with pytest.raises(pydantic.ValidationError):
            BackendConfig(backend_type="invalid")


class TestAnalysisRequest:
    """Tests for AnalysisRequest model."""

    def test_valid_request(self, tmp_path):
        """Test creating valid analysis request."""
        image_file = tmp_path / "test.jpg"
        image_file.write_bytes(b"fake image data")

        request = AnalysisRequest(
            mode=AnalysisMode.DEFECT,
            image_paths=[image_file],
        )
        assert request.mode == AnalysisMode.DEFECT
        assert len(request.image_paths) == 1
