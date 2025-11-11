"""Comprehensive unit tests for all Pydantic validation models in ocr.validation.models."""

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from ocr.datasets.schemas import ValidatedPolygonData
from ocr.validation.models import (
    BatchSample,
    CollateOutput,
    DatasetSample,
    LightningStepPrediction,
    ModelOutput,
    PolygonArray,
    TransformOutput,
    ValidatedTensorData,
    validate_predictions,
)


class TestValidatedPolygonData:
    """Test the ValidatedPolygonData model with bounds checking.

    This model addresses BUG-20251110-001: 26.5% data corruption from out-of-bounds coordinates.
    """

    def test_valid_polygon_within_bounds(self):
        """Test that valid polygons within image bounds pass validation."""
        points = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float32)
        polygon = ValidatedPolygonData(
            points=points,
            image_width=100,
            image_height=100
        )
        assert polygon.points.shape == (3, 2)
        assert polygon.image_width == 100
        assert polygon.image_height == 100

    def test_valid_polygon_at_boundaries(self):
        """Test that polygons with coordinates at exact boundaries are valid."""
        points = np.array([[0.0, 0.0], [99.0, 99.0], [50.0, 50.0]], dtype=np.float32)
        polygon = ValidatedPolygonData(
            points=points,
            image_width=100,
            image_height=100
        )
        assert polygon.points.shape == (3, 2)

    def test_invalid_x_coordinate_exceeds_width(self):
        """Test that polygons with x-coordinates exceeding image width raise ValidationError."""
        points = np.array([[10.0, 20.0], [150.0, 40.0], [50.0, 60.0]], dtype=np.float32)
        with pytest.raises(ValidationError) as exc_info:
            ValidatedPolygonData(
                points=points,
                image_width=100,
                image_height=100
            )
        error_msg = str(exc_info.value)
        assert "out-of-bounds x-coordinates" in error_msg
        assert "150" in error_msg or "150.0" in error_msg
        assert "[0, 100)" in error_msg

    def test_invalid_y_coordinate_exceeds_height(self):
        """Test that polygons with y-coordinates exceeding image height raise ValidationError."""
        points = np.array([[10.0, 20.0], [30.0, 150.0], [50.0, 60.0]], dtype=np.float32)
        with pytest.raises(ValidationError) as exc_info:
            ValidatedPolygonData(
                points=points,
                image_width=100,
                image_height=100
            )
        error_msg = str(exc_info.value)
        assert "out-of-bounds y-coordinates" in error_msg
        assert "150" in error_msg or "150.0" in error_msg
        assert "[0, 100)" in error_msg

    def test_invalid_negative_x_coordinate(self):
        """Test that polygons with negative x-coordinates raise ValidationError."""
        points = np.array([[-10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float32)
        with pytest.raises(ValidationError) as exc_info:
            ValidatedPolygonData(
                points=points,
                image_width=100,
                image_height=100
            )
        error_msg = str(exc_info.value)
        assert "out-of-bounds x-coordinates" in error_msg

    def test_invalid_negative_y_coordinate(self):
        """Test that polygons with negative y-coordinates raise ValidationError."""
        points = np.array([[10.0, -20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float32)
        with pytest.raises(ValidationError) as exc_info:
            ValidatedPolygonData(
                points=points,
                image_width=100,
                image_height=100
            )
        error_msg = str(exc_info.value)
        assert "out-of-bounds y-coordinates" in error_msg

    def test_invalid_multiple_out_of_bounds_coordinates(self):
        """Test that polygons with multiple out-of-bounds coordinates raise ValidationError."""
        points = np.array([[-10.0, 20.0], [150.0, 40.0], [50.0, 200.0]], dtype=np.float32)
        with pytest.raises(ValidationError):
            ValidatedPolygonData(
                points=points,
                image_width=100,
                image_height=100
            )

    def test_valid_polygon_with_confidence_and_label(self):
        """Test that ValidatedPolygonData with confidence and label works correctly."""
        points = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float32)
        polygon = ValidatedPolygonData(
            points=points,
            image_width=100,
            image_height=100,
            confidence=0.95,
            label="text"
        )
        assert polygon.confidence == 0.95
        assert polygon.label == "text"

    def test_coordinate_at_exact_width_boundary_is_invalid(self):
        """Test that coordinates at exactly image_width are invalid (exclusive upper bound)."""
        points = np.array([[10.0, 20.0], [100.0, 40.0], [50.0, 60.0]], dtype=np.float32)
        with pytest.raises(ValidationError) as exc_info:
            ValidatedPolygonData(
                points=points,
                image_width=100,
                image_height=100
            )
        error_msg = str(exc_info.value)
        assert "out-of-bounds x-coordinates" in error_msg

    def test_coordinate_at_exact_height_boundary_is_invalid(self):
        """Test that coordinates at exactly image_height are invalid (exclusive upper bound)."""
        points = np.array([[10.0, 20.0], [30.0, 100.0], [50.0, 60.0]], dtype=np.float32)
        with pytest.raises(ValidationError) as exc_info:
            ValidatedPolygonData(
                points=points,
                image_width=100,
                image_height=100
            )
        error_msg = str(exc_info.value)
        assert "out-of-bounds y-coordinates" in error_msg

    def test_error_message_indicates_which_coordinates_are_invalid(self):
        """Test that error messages clearly indicate which coordinates are out of bounds."""
        points = np.array([[10.0, 20.0], [150.0, 40.0], [50.0, 60.0]], dtype=np.float32)
        with pytest.raises(ValidationError) as exc_info:
            ValidatedPolygonData(
                points=points,
                image_width=100,
                image_height=100
            )
        error_msg = str(exc_info.value)
        # Should indicate index 1 is invalid
        assert "indices" in error_msg.lower() or "index" in error_msg.lower()
        assert "values" in error_msg.lower() or "value" in error_msg.lower()


class TestValidatedTensorData:
    """Test the ValidatedTensorData model with tensor validation.

    This model addresses BUG-20251112-001 (Dice loss errors) and BUG-20251112-013 (CUDA errors).
    """

    def test_valid_tensor_basic(self):
        """Test that valid tensors pass validation."""
        tensor = torch.rand(2, 3, 224, 224)
        validated = ValidatedTensorData(tensor=tensor)
        assert validated.tensor.shape == (2, 3, 224, 224)

    def test_valid_tensor_with_shape_validation(self):
        """Test tensor with expected shape validation."""
        tensor = torch.rand(2, 3, 224, 224)
        validated = ValidatedTensorData(
            tensor=tensor,
            expected_shape=(2, 3, 224, 224)
        )
        assert validated.tensor.shape == (2, 3, 224, 224)

    def test_invalid_tensor_shape_mismatch(self):
        """Test that tensor with wrong shape raises ValidationError."""
        tensor = torch.rand(2, 3, 224, 224)
        with pytest.raises(ValidationError) as exc_info:
            ValidatedTensorData(
                tensor=tensor,
                expected_shape=(2, 3, 256, 256)
            )
        error_msg = str(exc_info.value)
        assert "shape mismatch" in error_msg.lower()
        assert "(2, 3, 224, 224)" in error_msg
        assert "(2, 3, 256, 256)" in error_msg

    def test_valid_tensor_with_device_validation(self):
        """Test tensor with expected device validation."""
        tensor = torch.rand(2, 3, 224, 224)
        validated = ValidatedTensorData(
            tensor=tensor,
            expected_device="cpu"
        )
        assert validated.tensor.device.type == "cpu"

    def test_invalid_tensor_device_mismatch(self):
        """Test that tensor with wrong device raises ValidationError."""
        tensor = torch.rand(2, 3, 224, 224)  # CPU tensor
        with pytest.raises(ValidationError) as exc_info:
            ValidatedTensorData(
                tensor=tensor,
                expected_device="cuda"
            )
        error_msg = str(exc_info.value)
        assert "device mismatch" in error_msg.lower()

    def test_valid_tensor_with_dtype_validation(self):
        """Test tensor with expected dtype validation."""
        tensor = torch.rand(2, 3, 224, 224, dtype=torch.float32)
        validated = ValidatedTensorData(
            tensor=tensor,
            expected_dtype=torch.float32
        )
        assert validated.tensor.dtype == torch.float32

    def test_invalid_tensor_dtype_mismatch(self):
        """Test that tensor with wrong dtype raises ValidationError."""
        tensor = torch.rand(2, 3, 224, 224, dtype=torch.float32)
        with pytest.raises(ValidationError) as exc_info:
            ValidatedTensorData(
                tensor=tensor,
                expected_dtype=torch.float64
            )
        error_msg = str(exc_info.value)
        assert "dtype mismatch" in error_msg.lower()

    def test_valid_tensor_with_value_range(self):
        """Test tensor with valid value range."""
        tensor = torch.rand(2, 3, 224, 224)  # Values in [0, 1]
        validated = ValidatedTensorData(
            tensor=tensor,
            value_range=(0.0, 1.0)
        )
        assert validated.tensor.min() >= 0.0
        assert validated.tensor.max() <= 1.0

    def test_invalid_tensor_values_below_range(self):
        """Test that tensor with values below range raises ValidationError."""
        tensor = torch.tensor([-0.5, 0.5, 1.0])
        with pytest.raises(ValidationError) as exc_info:
            ValidatedTensorData(
                tensor=tensor,
                value_range=(0.0, 1.0)
            )
        error_msg = str(exc_info.value)
        assert "out of range" in error_msg.lower()
        assert "[0.0, 1.0]" in error_msg or "[0, 1]" in error_msg

    def test_invalid_tensor_values_above_range(self):
        """Test that tensor with values above range raises ValidationError."""
        tensor = torch.tensor([0.0, 0.5, 1.5])
        with pytest.raises(ValidationError) as exc_info:
            ValidatedTensorData(
                tensor=tensor,
                value_range=(0.0, 1.0)
            )
        error_msg = str(exc_info.value)
        assert "out of range" in error_msg.lower()

    def test_invalid_tensor_contains_nan(self):
        """Test that tensor with NaN values raises ValidationError."""
        tensor = torch.tensor([0.0, float('nan'), 1.0])
        with pytest.raises(ValidationError) as exc_info:
            ValidatedTensorData(
                tensor=tensor,
                allow_nan=False
            )
        error_msg = str(exc_info.value)
        assert "nan" in error_msg.lower()

    def test_valid_tensor_with_nan_allowed(self):
        """Test that tensor with NaN values passes when allowed."""
        tensor = torch.tensor([0.0, float('nan'), 1.0])
        validated = ValidatedTensorData(
            tensor=tensor,
            allow_nan=True
        )
        assert torch.isnan(validated.tensor).any()

    def test_invalid_tensor_contains_inf(self):
        """Test that tensor with infinite values raises ValidationError."""
        tensor = torch.tensor([0.0, float('inf'), 1.0])
        with pytest.raises(ValidationError) as exc_info:
            ValidatedTensorData(
                tensor=tensor,
                allow_inf=False
            )
        error_msg = str(exc_info.value)
        assert "inf" in error_msg.lower()

    def test_valid_tensor_with_inf_allowed(self):
        """Test that tensor with infinite values passes when allowed."""
        tensor = torch.tensor([0.0, float('inf'), 1.0])
        validated = ValidatedTensorData(
            tensor=tensor,
            allow_inf=True
        )
        assert torch.isinf(validated.tensor).any()

    def test_invalid_value_range_min_greater_than_max(self):
        """Test that invalid value_range format raises ValidationError."""
        tensor = torch.rand(2, 3, 224, 224)
        with pytest.raises(ValidationError) as exc_info:
            ValidatedTensorData(
                tensor=tensor,
                value_range=(1.0, 0.0)  # min > max
            )
        error_msg = str(exc_info.value)
        assert "min" in error_msg.lower() and "max" in error_msg.lower()

    def test_valid_tensor_with_all_validations(self):
        """Test tensor with all validation options enabled."""
        tensor = torch.rand(2, 3, 224, 224, dtype=torch.float32)
        validated = ValidatedTensorData(
            tensor=tensor,
            expected_shape=(2, 3, 224, 224),
            expected_device="cpu",
            expected_dtype=torch.float32,
            value_range=(0.0, 1.0),
            allow_inf=False,
            allow_nan=False
        )
        assert validated.tensor.shape == (2, 3, 224, 224)
        assert validated.tensor.device.type == "cpu"
        assert validated.tensor.dtype == torch.float32

    def test_invalid_not_a_tensor(self):
        """Test that non-tensor input raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ValidatedTensorData(tensor=np.array([1, 2, 3]))
        error_msg = str(exc_info.value)
        assert "tensor" in error_msg.lower()


class TestPolygonArray:
    """Test the PolygonArray validation model."""

    def test_valid_polygon_array(self):
        """Test that valid polygon arrays pass validation."""
        # Valid polygon with 4 points (square)
        points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        polygon = PolygonArray(points=points)
        assert polygon.points.shape == (4, 2)
        assert polygon.points.dtype == np.float32

    def test_valid_polygon_with_float64(self):
        """Test that valid polygon arrays with float64 remain as float64."""
        points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        polygon = PolygonArray(points=points)
        assert polygon.points.shape == (3, 2)
        assert polygon.points.dtype == np.float64  # Float64 is preserved in Pydantic v2

    def test_polygon_with_at_least_three_points(self):
        """Test that polygons with at least three points are valid."""
        # Minimum valid polygon (triangle)
        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float32)
        polygon = PolygonArray(points=points)
        assert polygon.points.shape == (3, 2)

    def test_invalid_not_numpy_array(self):
        """Test that non-numpy arrays raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PolygonArray(points=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        assert "Input should be an instance of ndarray" in str(exc_info.value)

    def test_invalid_wrong_shape(self):
        """Test that wrong-shaped arrays raise ValidationError."""
        # Shape (3, 3) instead of (N, 2)
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32)
        with pytest.raises(ValidationError) as exc_info:
            PolygonArray(points=points)
        assert "Polygon must be shaped (N, 2)" in str(exc_info.value)

    def test_invalid_not_2d_array(self):
        """Test that non-2D arrays raise ValidationError."""
        points = np.array([0.0, 1.0, 2.0], dtype=np.float32)  # 1D array
        with pytest.raises(ValidationError) as exc_info:
            PolygonArray(points=points)
        assert "Polygon must be shaped (N, 2)" in str(exc_info.value)

    def test_invalid_less_than_three_points(self):
        """Test that polygons with less than three points raise ValidationError."""
        # Only 2 points
        points = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        with pytest.raises(ValidationError) as exc_info:
            PolygonArray(points=points)
        assert "Polygon requires at least three points" in str(exc_info.value)

    def test_invalid_single_point(self):
        """Test that single-point arrays raise ValidationError."""
        points = np.array([[0.0, 0.0]], dtype=np.float32)
        with pytest.raises(ValidationError) as exc_info:
            PolygonArray(points=points)
        assert "Polygon requires at least three points" in str(exc_info.value)

    def test_invalid_empty_array(self):
        """Test that empty arrays raise ValidationError."""
        points = np.array([], dtype=np.float32).reshape(0, 2)
        with pytest.raises(ValidationError) as exc_info:
            PolygonArray(points=points)
        assert "Polygon requires at least three points" in str(exc_info.value)


class TestDatasetSample:
    """Test the DatasetSample validation model."""

    def test_valid_dataset_sample(self):
        """Test that valid dataset samples pass validation."""
        image = np.random.rand(100, 200, 3).astype(np.float32)
        polygons = [np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)]
        prob_maps = np.random.rand(100, 200).astype(np.float32)
        thresh_maps = np.random.rand(100, 200).astype(np.float32)
        inverse_matrix = np.eye(3, dtype=np.float32)
        shape = (100, 200)

        sample = DatasetSample(
            image=image,
            polygons=polygons,
            prob_maps=prob_maps,
            thresh_maps=thresh_maps,
            image_filename="test.jpg",
            image_path="/path/to/test.jpg",
            inverse_matrix=inverse_matrix,
            shape=shape,
        )
        assert sample.image.shape == (100, 200, 3)
        assert len(sample.polygons) == 1
        assert sample.prob_maps.shape == (100, 200)
        assert sample.thresh_maps.shape == (100, 200)
        assert sample.inverse_matrix.shape == (3, 3)
        assert sample.shape == (100, 200)

    def test_default_polygons_empty_list(self):
        """Test that polygons defaults to an empty list."""
        image = np.random.rand(100, 200, 3).astype(np.float32)
        prob_maps = np.random.rand(100, 200).astype(np.float32)
        thresh_maps = np.random.rand(100, 200).astype(np.float32)
        inverse_matrix = np.eye(3, dtype=np.float32)
        shape = (100, 200)

        sample = DatasetSample(
            image=image,
            prob_maps=prob_maps,
            thresh_maps=thresh_maps,
            image_filename="test.jpg",
            image_path="/path/to/test.jpg",
            inverse_matrix=inverse_matrix,
            shape=shape,
        )
        assert sample.polygons == []

    def test_invalid_image_wrong_shape(self):
        """Test that images with wrong shape raise ValidationError."""
        image = np.random.rand(100, 200, 1).astype(np.float32)  # Wrong channel dimension
        polygons = [np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)]
        prob_maps = np.random.rand(100, 200).astype(np.float32)
        thresh_maps = np.random.rand(100, 200).astype(np.float32)
        inverse_matrix = np.eye(3, dtype=np.float32)
        shape = (100, 200)

        with pytest.raises(ValidationError) as exc_info:
            DatasetSample(
                image=image,
                polygons=polygons,
                prob_maps=prob_maps,
                thresh_maps=thresh_maps,
                image_filename="test.jpg",
                image_path="/path/to/test.jpg",
                inverse_matrix=inverse_matrix,
                shape=shape,
            )
        assert "Image must be shaped (H, W, 3)" in str(exc_info.value)

    def test_invalid_prob_maps_wrong_ndim(self):
        """Test that prob_maps with wrong dimensions raise ValidationError."""
        image = np.random.rand(100, 200, 3).astype(np.float32)
        polygons = [np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)]
        prob_maps = np.random.rand(100, 200, 1).astype(np.float32)  # 3D instead of 2D
        thresh_maps = np.random.rand(100, 200).astype(np.float32)
        inverse_matrix = np.eye(3, dtype=np.float32)
        shape = (100, 200)

        with pytest.raises(ValidationError) as exc_info:
            DatasetSample(
                image=image,
                polygons=polygons,
                prob_maps=prob_maps,
                thresh_maps=thresh_maps,
                image_filename="test.jpg",
                image_path="/path/to/test.jpg",
                inverse_matrix=inverse_matrix,
                shape=shape,
            )
        assert "prob_maps must be 2D" in str(exc_info.value)

    def test_invalid_thresh_maps_wrong_ndim(self):
        """Test that thresh_maps with wrong dimensions raise ValidationError."""
        image = np.random.rand(100, 200, 3).astype(np.float32)
        polygons = [np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)]
        prob_maps = np.random.rand(100, 200).astype(np.float32)
        thresh_maps = np.random.rand(100, 200, 1).astype(np.float32)  # 3D instead of 2D
        inverse_matrix = np.eye(3, dtype=np.float32)
        shape = (100, 200)

        with pytest.raises(ValidationError) as exc_info:
            DatasetSample(
                image=image,
                polygons=polygons,
                prob_maps=prob_maps,
                thresh_maps=thresh_maps,
                image_filename="test.jpg",
                image_path="/path/to/test.jpg",
                inverse_matrix=inverse_matrix,
                shape=shape,
            )
        assert "thresh_maps must be 2D" in str(exc_info.value)

    def test_invalid_mismatched_prob_and_thresh_maps_shapes(self):
        """Test that mismatched prob_maps and thresh_maps shapes raise ValidationError."""
        image = np.random.rand(100, 200, 3).astype(np.float32)
        polygons = [np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)]
        prob_maps = np.random.rand(100, 200).astype(np.float32)
        thresh_maps = np.random.rand(50, 100).astype(np.float32)  # Different shape
        inverse_matrix = np.eye(3, dtype=np.float32)
        shape = (100, 200)

        with pytest.raises(ValidationError) as exc_info:
            DatasetSample(
                image=image,
                polygons=polygons,
                prob_maps=prob_maps,
                thresh_maps=thresh_maps,
                image_filename="test.jpg",
                image_path="/path/to/test.jpg",
                inverse_matrix=inverse_matrix,
                shape=shape,
            )
        assert "Probability and threshold maps must share the same shape" in str(exc_info.value)

    def test_invalid_inverse_matrix_wrong_shape(self):
        """Test that inverse_matrix with wrong shape raises ValidationError."""
        image = np.random.rand(100, 200, 3).astype(np.float32)
        polygons = [np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)]
        prob_maps = np.random.rand(100, 200).astype(np.float32)
        thresh_maps = np.random.rand(100, 200).astype(np.float32)
        inverse_matrix = np.eye(4, dtype=np.float32)  # Wrong shape (4, 4) instead of (3, 3)
        shape = (100, 200)

        with pytest.raises(ValidationError) as exc_info:
            DatasetSample(
                image=image,
                polygons=polygons,
                prob_maps=prob_maps,
                thresh_maps=thresh_maps,
                image_filename="test.jpg",
                image_path="/path/to/test.jpg",
                inverse_matrix=inverse_matrix,
                shape=shape,
            )
        assert "Inverse matrix must be shaped (3, 3)" in str(exc_info.value)

    def test_invalid_shape_wrong_length(self):
        """Test that shape with wrong length raises ValidationError."""
        image = np.random.rand(100, 200, 3).astype(np.float32)
        polygons = [np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)]
        prob_maps = np.random.rand(100, 200).astype(np.float32)
        thresh_maps = np.random.rand(100, 200).astype(np.float32)
        inverse_matrix = np.eye(3, dtype=np.float32)
        shape = (100, 200, 1)  # Wrong length - 3 elements instead of 2

        with pytest.raises(ValidationError) as exc_info:
            DatasetSample(
                image=image,
                polygons=polygons,
                prob_maps=prob_maps,
                thresh_maps=thresh_maps,
                image_filename="test.jpg",
                image_path="/path/to/test.jpg",
                inverse_matrix=inverse_matrix,
                shape=shape,
            )
        assert "Tuple should have at most 2 items after validation" in str(exc_info.value)

    def test_invalid_polygon_in_polygons(self):
        """Test that invalid polygons in the polygons list raise ValidationError."""
        image = np.random.rand(100, 200, 3).astype(np.float32)
        # Invalid polygon with only 2 points
        invalid_polygon = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)
        prob_maps = np.random.rand(100, 200).astype(np.float32)
        thresh_maps = np.random.rand(100, 200).astype(np.float32)
        inverse_matrix = np.eye(3, dtype=np.float32)
        shape = (100, 200)

        with pytest.raises(ValidationError) as exc_info:
            DatasetSample(
                image=image,
                polygons=[invalid_polygon],
                prob_maps=prob_maps,
                thresh_maps=thresh_maps,
                image_filename="test.jpg",
                image_path="/path/to/test.jpg",
                inverse_matrix=inverse_matrix,
                shape=shape,
            )
        assert "Polygon requires at least three points" in str(exc_info.value)


class TestTransformOutput:
    """Test the TransformOutput validation model."""

    def test_valid_transform_output(self):
        """Test that valid transform outputs pass validation."""
        image = torch.rand(3, 100, 200)  # (C, H, W)
        polygons = [np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)]
        prob_maps = torch.rand(1, 100, 200)  # (1, H, W)
        thresh_maps = torch.rand(1, 100, 200)  # (1, H, W)
        inverse_matrix = np.eye(3, dtype=np.float32)

        output = TransformOutput(
            image=image,
            polygons=polygons,
            prob_maps=prob_maps,
            thresh_maps=thresh_maps,
            inverse_matrix=inverse_matrix,
        )
        assert output.image.shape == (3, 100, 200)
        assert len(output.polygons) == 1
        assert output.prob_maps.shape == (1, 100, 200)
        assert output.thresh_maps.shape == (1, 100, 200)
        assert output.inverse_matrix.shape == (3, 3)

    def test_invalid_image_wrong_shape(self):
        """Test that image tensors with wrong shape raise ValidationError."""
        image = torch.rand(1, 100, 200)  # Wrong channel dimension - 1 instead of 3
        polygons = [np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)]
        prob_maps = torch.rand(1, 100, 200)
        thresh_maps = torch.rand(1, 100, 200)
        inverse_matrix = np.eye(3, dtype=np.float32)

        with pytest.raises(ValidationError) as exc_info:
            TransformOutput(
                image=image,
                polygons=polygons,
                prob_maps=prob_maps,
                thresh_maps=thresh_maps,
                inverse_matrix=inverse_matrix,
            )
        assert "Transformed image tensor must be shaped (3, H, W)" in str(exc_info.value)

    def test_invalid_prob_maps_wrong_shape(self):
        """Test that prob_maps tensors with wrong shape raise ValidationError."""
        image = torch.rand(3, 100, 200)
        polygons = [np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)]
        prob_maps = torch.rand(3, 100, 200)  # Wrong channel dimension - 3 instead of 1
        thresh_maps = torch.rand(1, 100, 200)
        inverse_matrix = np.eye(3, dtype=np.float32)

        with pytest.raises(ValidationError) as exc_info:
            TransformOutput(
                image=image,
                polygons=polygons,
                prob_maps=prob_maps,
                thresh_maps=thresh_maps,
                inverse_matrix=inverse_matrix,
            )
        assert "prob_maps tensor must be shaped (1, H, W)" in str(exc_info.value)

    def test_invalid_thresh_maps_wrong_shape(self):
        """Test that thresh_maps tensors with wrong shape raise ValidationError."""
        image = torch.rand(3, 100, 200)
        polygons = [np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)]
        prob_maps = torch.rand(1, 100, 200)
        thresh_maps = torch.rand(2, 100, 200)  # Wrong channel dimension - 2 instead of 1
        inverse_matrix = np.eye(3, dtype=np.float32)

        with pytest.raises(ValidationError) as exc_info:
            TransformOutput(
                image=image,
                polygons=polygons,
                prob_maps=prob_maps,
                thresh_maps=thresh_maps,
                inverse_matrix=inverse_matrix,
            )
        assert "thresh_maps tensor must be shaped (1, H, W)" in str(exc_info.value)

    def test_invalid_inverse_matrix_wrong_shape(self):
        """Test that inverse_matrix with wrong shape raises ValidationError."""
        image = torch.rand(3, 100, 200)
        polygons = [np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)]
        prob_maps = torch.rand(1, 100, 200)
        thresh_maps = torch.rand(1, 100, 200)
        inverse_matrix = np.eye(4, dtype=np.float32)  # Wrong shape (4, 4) instead of (3, 3)

        with pytest.raises(ValidationError) as exc_info:
            TransformOutput(
                image=image,
                polygons=polygons,
                prob_maps=prob_maps,
                thresh_maps=thresh_maps,
                inverse_matrix=inverse_matrix,
            )
        assert "Inverse matrix must be shaped (3, 3)" in str(exc_info.value)

    def test_invalid_polygon_in_polygons(self):
        """Test that invalid polygons in the polygons list raise ValidationError."""
        image = torch.rand(3, 100, 200)
        # Invalid polygon with only 2 points
        invalid_polygon = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)
        prob_maps = torch.rand(1, 100, 200)
        thresh_maps = torch.rand(1, 100, 200)
        inverse_matrix = np.eye(3, dtype=np.float32)

        with pytest.raises(ValidationError) as exc_info:
            TransformOutput(
                image=image,
                polygons=[invalid_polygon],
                prob_maps=prob_maps,
                thresh_maps=thresh_maps,
                inverse_matrix=inverse_matrix,
            )
        assert "Polygon requires at least three points" in str(exc_info.value)


class TestBatchSample:
    """Test the BatchSample validation model."""

    def test_valid_batch_sample(self):
        """Test that valid batch samples pass validation."""
        image = torch.rand(3, 100, 200)
        polygons = [np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)]
        prob_maps = torch.rand(1, 100, 200)
        thresh_maps = torch.rand(1, 100, 200)
        inverse_matrix = np.eye(3, dtype=np.float32)
        shape = (100, 200)

        sample = BatchSample(
            image=image,
            polygons=polygons,
            prob_maps=prob_maps,
            thresh_maps=thresh_maps,
            image_filename="test.jpg",
            image_path="/path/to/test.jpg",
            inverse_matrix=inverse_matrix,
            shape=shape,
        )
        assert sample.image.shape == (3, 100, 200)
        assert len(sample.polygons) == 1
        assert sample.prob_maps.shape == (1, 100, 200)
        assert sample.thresh_maps.shape == (1, 100, 200)
        assert sample.inverse_matrix.shape == (3, 3)
        assert sample.shape == (100, 200)

    def test_invalid_inverse_matrix_wrong_shape(self):
        """Test that inverse_matrix with wrong shape raises ValidationError."""
        image = torch.rand(3, 100, 200)
        polygons = [np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)]
        prob_maps = torch.rand(1, 100, 200)
        thresh_maps = torch.rand(1, 100, 200)
        inverse_matrix = np.eye(4, dtype=np.float32)  # Wrong shape
        shape = (100, 200)

        with pytest.raises(ValidationError) as exc_info:
            BatchSample(
                image=image,
                polygons=polygons,
                prob_maps=prob_maps,
                thresh_maps=thresh_maps,
                image_filename="test.jpg",
                image_path="/path/to/test.jpg",
                inverse_matrix=inverse_matrix,
                shape=shape,
            )
        assert "Inverse matrix must be shaped (3, 3)" in str(exc_info.value)

    def test_invalid_polygon_in_polygons(self):
        """Test that invalid polygons in the polygons list raise ValidationError."""
        image = torch.rand(3, 100, 200)
        # Invalid polygon with only 2 points
        invalid_polygon = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)
        prob_maps = torch.rand(1, 100, 200)
        thresh_maps = torch.rand(1, 100, 200)
        inverse_matrix = np.eye(3, dtype=np.float32)
        shape = (100, 200)

        with pytest.raises(ValidationError) as exc_info:
            BatchSample(
                image=image,
                polygons=[invalid_polygon],
                prob_maps=prob_maps,
                thresh_maps=thresh_maps,
                image_filename="test.jpg",
                image_path="/path/to/test.jpg",
                inverse_matrix=inverse_matrix,
                shape=shape,
            )
        assert "Polygon requires at least three points" in str(exc_info.value)


class TestCollateOutput:
    """Test the CollateOutput validation model."""

    def test_valid_collate_output(self):
        """Test that valid collate outputs pass validation."""
        image_filenames = ["test1.jpg", "test2.jpg"]
        image_paths = ["/path/to/test1.jpg", "/path/to/test2.jpg"]
        inverse_matrices = [np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)]
        shapes = [(100, 200), (150, 250)]
        images = torch.rand(2, 3, 100, 200)  # (B, C, H, W)
        polygons = [
            [np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)],
            [np.array([[5.0, 5.0], [15.0, 5.0], [15.0, 15.0], [5.0, 15.0]], dtype=np.float32)],
        ]
        prob_maps = torch.rand(2, 1, 100, 200)  # (B, 1, H, W)
        thresh_maps = torch.rand(2, 1, 100, 200)  # (B, 1, H, W)

        output = CollateOutput(
            image_filename=image_filenames,
            image_path=image_paths,
            inverse_matrix=inverse_matrices,
            shape=shapes,
            images=images,
            polygons=polygons,
            prob_maps=prob_maps,
            thresh_maps=thresh_maps,
        )
        assert output.image_filename == image_filenames
        assert output.image_path == image_paths
        assert len(output.inverse_matrix) == 2
        assert output.images.shape == (2, 3, 100, 200)
        assert len(output.polygons) == 2
        assert output.prob_maps.shape == (2, 1, 100, 200)
        assert output.thresh_maps.shape == (2, 1, 100, 200)

    def test_valid_collate_output_with_optional_fields(self):
        """Test that valid collate outputs with optional fields pass validation."""
        image_filenames = ["test1.jpg", "test2.jpg"]
        image_paths = ["/path/to/test1.jpg", "/path/to/test2.jpg"]
        inverse_matrices = [np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)]
        shapes = [(100, 200), (150, 250)]
        images = torch.rand(2, 3, 100, 200)
        polygons = [
            [np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)],
            [np.array([[5.0, 5.0], [15.0, 5.0], [15.0, 15.0], [5.0, 15.0]], dtype=np.float32)],
        ]
        prob_maps = torch.rand(2, 1, 100, 200)
        thresh_maps = torch.rand(2, 1, 100, 200)
        orientations = [1, 3]
        raw_sizes = [(800, 600), (1024, 768)]
        canonical_sizes = [(512, 512), (768, 768)]

        output = CollateOutput(
            image_filename=image_filenames,
            image_path=image_paths,
            inverse_matrix=inverse_matrices,
            shape=shapes,
            images=images,
            polygons=polygons,
            prob_maps=prob_maps,
            thresh_maps=thresh_maps,
            orientation=orientations,
            raw_size=raw_sizes,
            canonical_size=canonical_sizes,
        )
        assert output.orientation == orientations
        assert output.raw_size == raw_sizes
        assert output.canonical_size == canonical_sizes

    def test_invalid_image_filename_none(self):
        """Test that None for image_filename raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CollateOutput(
                image_filename=None,
                image_path=["/path/to/test.jpg"],
                inverse_matrix=[np.eye(3)],
                shape=[(100, 200)],
                images=torch.rand(1, 3, 100, 200),
                polygons=[[np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)]],
                prob_maps=torch.rand(1, 1, 100, 200),
                thresh_maps=torch.rand(1, 1, 100, 200),
            )
        assert "Batch metadata sequences must contain at least one entry" in str(exc_info.value)

    def test_invalid_image_filename_empty_list(self):
        """Test that empty image_filename list raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CollateOutput(
                image_filename=[],
                image_path=["/path/to/test.jpg"],
                inverse_matrix=[np.eye(3)],
                shape=[(100, 200)],
                images=torch.rand(1, 3, 100, 200),
                polygons=[[np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)]],
                prob_maps=torch.rand(1, 1, 100, 200),
                thresh_maps=torch.rand(1, 1, 100, 200),
            )
        assert "Batch metadata sequences must contain at least one entry" in str(exc_info.value)

    def test_invalid_images_wrong_shape(self):
        """Test that images tensor with wrong shape raises ValidationError."""
        image_filenames = ["test.jpg"]
        image_paths = ["/path/to/test.jpg"]
        inverse_matrices = [np.eye(3, dtype=np.float32)]
        shapes = [(100, 200)]
        images = torch.rand(1, 1, 100, 200)  # Wrong channel dimension - 1 instead of 3
        polygons = [[np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)]]
        prob_maps = torch.rand(1, 1, 100, 200)
        thresh_maps = torch.rand(1, 1, 100, 200)

        with pytest.raises(ValidationError) as exc_info:
            CollateOutput(
                image_filename=image_filenames,
                image_path=image_paths,
                inverse_matrix=inverse_matrices,
                shape=shapes,
                images=images,
                polygons=polygons,
                prob_maps=prob_maps,
                thresh_maps=thresh_maps,
            )
        assert "images tensor must be shaped (B, 3, H, W)" in str(exc_info.value)

    def test_invalid_images_batch_mismatch(self):
        """Test that images batch dimension mismatch raises ValidationError."""
        image_filenames = ["test1.jpg", "test2.jpg"]  # Batch size 2
        image_paths = ["/path/to/test1.jpg", "/path/to/test2.jpg"]
        inverse_matrices = [np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)]
        shapes = [(100, 200), (150, 250)]
        images = torch.rand(1, 3, 100, 200)  # Batch size 1 instead of 2
        polygons = [
            [np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)],
            [np.array([[5.0, 5.0], [15.0, 5.0], [15.0, 15.0], [5.0, 15.0]], dtype=np.float32)],
        ]
        prob_maps = torch.rand(2, 1, 100, 200)
        thresh_maps = torch.rand(2, 1, 100, 200)

        with pytest.raises(ValidationError) as exc_info:
            CollateOutput(
                image_filename=image_filenames,
                image_path=image_paths,
                inverse_matrix=inverse_matrices,
                shape=shapes,
                images=images,
                polygons=polygons,
                prob_maps=prob_maps,
                thresh_maps=thresh_maps,
            )
        assert "Number of images does not match batch metadata" in str(exc_info.value)

    def test_invalid_prob_maps_wrong_shape(self):
        """Test that prob_maps tensor with wrong shape raises ValidationError."""
        image_filenames = ["test.jpg"]
        image_paths = ["/path/to/test.jpg"]
        inverse_matrices = [np.eye(3, dtype=np.float32)]
        shapes = [(100, 200)]
        images = torch.rand(1, 3, 100, 200)
        polygons = [[np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)]]
        prob_maps = torch.rand(1, 3, 100, 200)  # Wrong channel dimension - 3 instead of 1
        thresh_maps = torch.rand(1, 1, 100, 200)

        with pytest.raises(ValidationError) as exc_info:
            CollateOutput(
                image_filename=image_filenames,
                image_path=image_paths,
                inverse_matrix=inverse_matrices,
                shape=shapes,
                images=images,
                polygons=polygons,
                prob_maps=prob_maps,
                thresh_maps=thresh_maps,
            )
        assert "prob_maps tensor must be shaped (B, 1, H, W)" in str(exc_info.value)

    def test_invalid_prob_maps_batch_mismatch(self):
        """Test that prob_maps batch dimension mismatch raises ValidationError."""
        image_filenames = ["test1.jpg", "test2.jpg"]  # Batch size 2
        image_paths = ["/path/to/test1.jpg", "/path/to/test2.jpg"]
        inverse_matrices = [np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)]
        shapes = [(100, 200), (150, 250)]
        images = torch.rand(2, 3, 100, 200)
        polygons = [
            [np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)],
            [np.array([[5.0, 5.0], [15.0, 5.0], [15.0, 15.0], [5.0, 15.0]], dtype=np.float32)],
        ]
        prob_maps = torch.rand(1, 1, 100, 200)  # Batch size 1 instead of 2
        thresh_maps = torch.rand(2, 1, 100, 200)

        with pytest.raises(ValidationError) as exc_info:
            CollateOutput(
                image_filename=image_filenames,
                image_path=image_paths,
                inverse_matrix=inverse_matrices,
                shape=shapes,
                images=images,
                polygons=polygons,
                prob_maps=prob_maps,
                thresh_maps=thresh_maps,
            )
        assert "prob_maps batch dimension must match batch size" in str(exc_info.value)

    def test_invalid_thresh_maps_wrong_shape(self):
        """Test that thresh_maps tensor with wrong shape raises ValidationError."""
        image_filenames = ["test.jpg"]
        image_paths = ["/path/to/test.jpg"]
        inverse_matrices = [np.eye(3, dtype=np.float32)]
        shapes = [(100, 200)]
        images = torch.rand(1, 3, 100, 200)
        polygons = [[np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)]]
        prob_maps = torch.rand(1, 1, 100, 200)
        thresh_maps = torch.rand(1, 2, 100, 200)  # Wrong channel dimension - 2 instead of 1

        with pytest.raises(ValidationError) as exc_info:
            CollateOutput(
                image_filename=image_filenames,
                image_path=image_paths,
                inverse_matrix=inverse_matrices,
                shape=shapes,
                images=images,
                polygons=polygons,
                prob_maps=prob_maps,
                thresh_maps=thresh_maps,
            )
        assert "thresh_maps tensor must be shaped (B, 1, H, W)" in str(exc_info.value)

    def test_invalid_thresh_maps_batch_mismatch(self):
        """Test that thresh_maps batch dimension mismatch raises ValidationError."""
        image_filenames = ["test1.jpg", "test2.jpg"]  # Batch size 2
        image_paths = ["/path/to/test1.jpg", "/path/to/test2.jpg"]
        inverse_matrices = [np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)]
        shapes = [(100, 200), (150, 250)]
        images = torch.rand(2, 3, 100, 200)
        polygons = [
            [np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)],
            [np.array([[5.0, 5.0], [15.0, 5.0], [15.0, 15.0], [5.0, 15.0]], dtype=np.float32)],
        ]
        prob_maps = torch.rand(2, 1, 100, 200)
        thresh_maps = torch.rand(1, 1, 100, 200)  # Batch size 1 instead of 2

        with pytest.raises(ValidationError) as exc_info:
            CollateOutput(
                image_filename=image_filenames,
                image_path=image_paths,
                inverse_matrix=inverse_matrices,
                shape=shapes,
                images=images,
                polygons=polygons,
                prob_maps=prob_maps,
                thresh_maps=thresh_maps,
            )
        assert "thresh_maps batch dimension must match batch size" in str(exc_info.value)

    def test_invalid_polygons_batch_mismatch(self):
        """Test that polygons list length mismatch raises ValidationError."""
        image_filenames = ["test1.jpg", "test2.jpg"]  # Batch size 2
        image_paths = ["/path/to/test1.jpg", "/path/to/test2.jpg"]
        inverse_matrices = [np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)]
        shapes = [(100, 200), (150, 250)]
        images = torch.rand(2, 3, 100, 200)
        # Only 1 polygon list instead of 2
        polygons = [[np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)]]
        prob_maps = torch.rand(2, 1, 100, 200)
        thresh_maps = torch.rand(2, 1, 100, 200)

        with pytest.raises(ValidationError) as exc_info:
            CollateOutput(
                image_filename=image_filenames,
                image_path=image_paths,
                inverse_matrix=inverse_matrices,
                shape=shapes,
                images=images,
                polygons=polygons,
                prob_maps=prob_maps,
                thresh_maps=thresh_maps,
            )
        assert "Polygons list must align with batch size" in str(exc_info.value)

    def test_invalid_polygons_not_list(self):
        """Test that non-list entries in polygons raise ValidationError."""
        image_filenames = ["test.jpg"]
        image_paths = ["/path/to/test.jpg"]
        inverse_matrices = [np.eye(3, dtype=np.float32)]
        shapes = [(100, 200)]
        images = torch.rand(1, 3, 100, 200)
        # Pass a string instead of a list of arrays
        polygons = "not_a_list"  # Not a list
        prob_maps = torch.rand(1, 1, 100, 200)
        thresh_maps = torch.rand(1, 1, 100, 200)

        with pytest.raises(ValidationError) as exc_info:
            CollateOutput(
                image_filename=image_filenames,
                image_path=image_paths,
                inverse_matrix=inverse_matrices,
                shape=shapes,
                images=images,
                polygons=polygons,
                prob_maps=prob_maps,
                thresh_maps=thresh_maps,
            )
        assert "Input should be a valid" in str(exc_info.value)  # Pydantic v2 error message

    def test_invalid_orientation_batch_mismatch(self):
        """Test that orientation length mismatch raises ValidationError."""
        image_filenames = ["test1.jpg", "test2.jpg"]  # Batch size 2
        image_paths = ["/path/to/test1.jpg", "/path/to/test2.jpg"]
        inverse_matrices = [np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)]
        shapes = [(100, 200), (150, 250)]
        images = torch.rand(2, 3, 100, 200)
        polygons = [
            [np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)],
            [np.array([[5.0, 5.0], [15.0, 5.0], [15.0, 15.0], [5.0, 15.0]], dtype=np.float32)],
        ]
        prob_maps = torch.rand(2, 1, 100, 200)
        thresh_maps = torch.rand(2, 1, 100, 200)
        orientations = [1]  # Only 1 orientation instead of 2

        with pytest.raises(ValidationError) as exc_info:
            CollateOutput(
                image_filename=image_filenames,
                image_path=image_paths,
                inverse_matrix=inverse_matrices,
                shape=shapes,
                images=images,
                polygons=polygons,
                prob_maps=prob_maps,
                thresh_maps=thresh_maps,
                orientation=orientations,
            )
        assert "orientation length must match batch size" in str(exc_info.value)

    def test_invalid_orientation_value(self):
        """Test that invalid orientation values raise ValidationError."""
        image_filenames = ["test.jpg"]
        image_paths = ["/path/to/test.jpg"]
        inverse_matrices = [np.eye(3, dtype=np.float32)]
        shapes = [(100, 200)]
        images = torch.rand(1, 3, 100, 200)
        polygons = [[np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)]]
        prob_maps = torch.rand(1, 1, 100, 200)
        thresh_maps = torch.rand(1, 1, 100, 200)
        orientations = [9]  # Invalid orientation (not in VALID_EXIF_ORIENTATIONS)

        with pytest.raises(ValidationError) as exc_info:
            CollateOutput(
                image_filename=image_filenames,
                image_path=image_paths,
                inverse_matrix=inverse_matrices,
                shape=shapes,
                images=images,
                polygons=polygons,
                prob_maps=prob_maps,
                thresh_maps=thresh_maps,
                orientation=orientations,
            )
        assert "must be one of {0, 1, 2, 3, 4, 5, 6, 7, 8}" in str(exc_info.value)

    def test_invalid_raw_size_batch_mismatch(self):
        """Test that raw_size length mismatch raises ValidationError."""
        image_filenames = ["test1.jpg", "test2.jpg"]  # Batch size 2
        image_paths = ["/path/to/test1.jpg", "/path/to/test2.jpg"]
        inverse_matrices = [np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)]
        shapes = [(100, 200), (150, 250)]
        images = torch.rand(2, 3, 100, 200)
        polygons = [
            [np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)],
            [np.array([[5.0, 5.0], [15.0, 5.0], [15.0, 15.0], [5.0, 15.0]], dtype=np.float32)],
        ]
        prob_maps = torch.rand(2, 1, 100, 200)
        thresh_maps = torch.rand(2, 1, 100, 200)
        raw_sizes = [(800, 600)]  # Only 1 size instead of 2

        with pytest.raises(ValidationError) as exc_info:
            CollateOutput(
                image_filename=image_filenames,
                image_path=image_paths,
                inverse_matrix=inverse_matrices,
                shape=shapes,
                images=images,
                polygons=polygons,
                prob_maps=prob_maps,
                thresh_maps=thresh_maps,
                raw_size=raw_sizes,
            )
        assert "raw_size length must match batch size" in str(exc_info.value)

    def test_invalid_raw_size_negative_dimensions(self):
        """Test that raw_size with negative dimensions raises ValidationError."""
        image_filenames = ["test.jpg"]
        image_paths = ["/path/to/test.jpg"]
        inverse_matrices = [np.eye(3, dtype=np.float32)]
        shapes = [(100, 200)]
        images = torch.rand(1, 3, 100, 200)
        polygons = [[np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)]]
        prob_maps = torch.rand(1, 1, 100, 200)
        thresh_maps = torch.rand(1, 1, 100, 200)
        raw_sizes = [(-100, 600)]  # Negative width

        with pytest.raises(ValidationError) as exc_info:
            CollateOutput(
                image_filename=image_filenames,
                image_path=image_paths,
                inverse_matrix=inverse_matrices,
                shape=shapes,
                images=images,
                polygons=polygons,
                prob_maps=prob_maps,
                thresh_maps=thresh_maps,
                raw_size=raw_sizes,
            )
        assert "raw_size dimensions must be non-negative" in str(exc_info.value)

    def test_invalid_canonical_size_batch_mismatch(self):
        """Test that canonical_size length mismatch raises ValidationError."""
        image_filenames = ["test1.jpg", "test2.jpg"]  # Batch size 2
        image_paths = ["/path/to/test1.jpg", "/path/to/test2.jpg"]
        inverse_matrices = [np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)]
        shapes = [(100, 200), (150, 250)]
        images = torch.rand(2, 3, 100, 200)
        polygons = [
            [np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)],
            [np.array([[5.0, 5.0], [15.0, 5.0], [15.0, 15.0], [5.0, 15.0]], dtype=np.float32)],
        ]
        prob_maps = torch.rand(2, 1, 100, 200)
        thresh_maps = torch.rand(2, 1, 100, 200)
        canonical_sizes = [(512, 512)]  # Only 1 size instead of 2

        with pytest.raises(ValidationError) as exc_info:
            CollateOutput(
                image_filename=image_filenames,
                image_path=image_paths,
                inverse_matrix=inverse_matrices,
                shape=shapes,
                images=images,
                polygons=polygons,
                prob_maps=prob_maps,
                thresh_maps=thresh_maps,
                canonical_size=canonical_sizes,
            )
        assert "canonical_size length must match batch size" in str(exc_info.value)


class TestModelOutput:
    """Test the ModelOutput validation model."""

    def test_valid_model_output(self):
        """Test that valid model outputs pass validation."""
        prob_maps = torch.rand(2, 3, 100, 200)
        thresh_maps = torch.rand(2, 3, 100, 200)
        binary_maps = torch.rand(2, 3, 100, 200)

        output = ModelOutput(
            prob_maps=prob_maps,
            thresh_maps=thresh_maps,
            binary_maps=binary_maps,
        )
        assert output.prob_maps.shape == (2, 3, 100, 200)
        assert output.thresh_maps.shape == (2, 3, 100, 200)
        assert output.binary_maps.shape == (2, 3, 100, 200)

    def test_valid_model_output_with_optional_fields(self):
        """Test that valid model outputs with optional fields pass validation."""
        prob_maps = torch.rand(2, 3, 100, 200)
        thresh_maps = torch.rand(2, 3, 100, 200)
        binary_maps = torch.rand(2, 3, 100, 200)
        loss = torch.tensor(0.5)
        loss_dict = {"loss_cls": torch.tensor(0.3), "loss_reg": torch.tensor(0.2)}

        output = ModelOutput(
            prob_maps=prob_maps,
            thresh_maps=thresh_maps,
            binary_maps=binary_maps,
            loss=loss,
            loss_dict=loss_dict,
        )
        assert output.loss == loss
        assert output.loss_dict == loss_dict

    def test_invalid_thresh_maps_wrong_shape(self):
        """Test that thresh_maps with wrong shape raises ValidationError."""
        prob_maps = torch.rand(2, 3, 100, 200)
        thresh_maps = torch.rand(1, 1, 50, 100)  # Different shape
        binary_maps = torch.rand(2, 3, 100, 200)

        with pytest.raises(ValidationError) as exc_info:
            ModelOutput(
                prob_maps=prob_maps,
                thresh_maps=thresh_maps,
                binary_maps=binary_maps,
            )
        assert "Model output tensors must share the same shape" in str(exc_info.value)

    def test_invalid_binary_maps_wrong_shape(self):
        """Test that binary_maps with wrong shape raises ValidationError."""
        prob_maps = torch.rand(2, 3, 100, 200)
        thresh_maps = torch.rand(2, 3, 100, 200)
        binary_maps = torch.rand(1, 1, 50, 100)  # Different shape

        with pytest.raises(ValidationError) as exc_info:
            ModelOutput(
                prob_maps=prob_maps,
                thresh_maps=thresh_maps,
                binary_maps=binary_maps,
            )
        assert "Model output tensors must share the same shape" in str(exc_info.value)


class TestLightningStepPrediction:
    """Test the LightningStepPrediction validation model."""

    def test_valid_lightning_step_prediction(self):
        """Test that valid lightning step predictions pass validation."""
        boxes = [np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)]
        orientation = 1
        raw_size = (800, 600)
        canonical_size = (512, 512)
        image_path = "/path/to/image.jpg"

        prediction = LightningStepPrediction(
            boxes=boxes,
            orientation=orientation,
            raw_size=raw_size,
            canonical_size=canonical_size,
            image_path=image_path,
        )
        assert len(prediction.boxes) == 1
        assert prediction.orientation == orientation
        assert prediction.raw_size == raw_size
        assert prediction.canonical_size == canonical_size
        assert prediction.image_path == image_path

    def test_valid_lightning_step_prediction_with_defaults(self):
        """Test that valid lightning step predictions with defaults pass validation."""
        boxes = [np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)]
        # Use default values
        prediction = LightningStepPrediction(boxes=boxes)
        assert len(prediction.boxes) == 1
        assert prediction.orientation == 1  # Default value
        assert prediction.raw_size is None
        assert prediction.canonical_size is None
        assert prediction.image_path is None

    def test_invalid_box_with_invalid_polygon(self):
        """Test that invalid boxes (polygons) raise ValidationError."""
        # Invalid box with only 2 points
        invalid_box = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)

        with pytest.raises(ValidationError) as exc_info:
            LightningStepPrediction(boxes=[invalid_box])
        assert "Polygon requires at least three points" in str(exc_info.value)

    def test_invalid_orientation_value(self):
        """Test that invalid orientation values raise ValidationError."""
        boxes = [np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)]

        with pytest.raises(ValidationError) as exc_info:
            LightningStepPrediction(boxes=boxes, orientation=9)  # Invalid orientation
        assert "Orientation must be one of {0, 1, 2, 3, 4, 5, 6, 7, 8}" in str(exc_info.value)

    def test_invalid_raw_size_negative_dimensions(self):
        """Test that raw_size with negative dimensions raises ValidationError."""
        boxes = [np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)]

        with pytest.raises(ValidationError) as exc_info:
            LightningStepPrediction(boxes=boxes, raw_size=(-100, 600))
        assert "raw_size dimensions must be non-negative" in str(exc_info.value)

    def test_invalid_canonical_size_negative_dimensions(self):
        """Test that canonical_size with negative dimensions raises ValidationError."""
        boxes = [np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)]

        with pytest.raises(ValidationError) as exc_info:
            LightningStepPrediction(boxes=boxes, canonical_size=(100, -600))
        assert "canonical_size dimensions must be non-negative" in str(exc_info.value)

    def test_invalid_image_path_empty_string(self):
        """Test that empty image_path raises ValidationError."""
        boxes = [np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)]

        with pytest.raises(ValidationError) as exc_info:
            LightningStepPrediction(boxes=boxes, image_path="")
        assert "Image path, when provided, must be a non-empty string" in str(exc_info.value)


class TestValidatePredictionsFunction:
    """Test the validate_predictions function."""

    def test_valid_predictions(self):
        """Test that valid predictions pass validation."""
        filenames = ["test1.jpg", "test2.jpg"]
        predictions = [
            {
                "boxes": [np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)],
                "orientation": 1,
                "raw_size": (800, 600),
                "canonical_size": (512, 512),
                "image_path": "/path/to/test1.jpg",
            },
            {
                "boxes": [np.array([[5.0, 5.0], [15.0, 5.0], [15.0, 15.0], [5.0, 15.0]], dtype=np.float32)],
                "orientation": 3,
                "raw_size": (1024, 768),
                "canonical_size": (768, 768),
                "image_path": "/path/to/test2.jpg",
            },
        ]

        result = validate_predictions(filenames, predictions)
        assert len(result) == 2
        assert result[0].image_path == "/path/to/test1.jpg"
        assert result[1].image_path == "/path/to/test2.jpg"

    def test_predictions_with_mismatched_lengths(self):
        """Test that mismatched lengths between filenames and predictions raise ValidationError."""
        filenames = ["test1.jpg", "test2.jpg"]  # 2 filenames
        predictions = [  # Only 1 prediction
            {
                "boxes": [np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)],
                "orientation": 1,
            }
        ]

        with pytest.raises(ValidationError) as exc_info:
            validate_predictions(filenames, predictions)
        assert "Number of filenames and predictions must match" in str(exc_info.value)

    def test_predictions_with_invalid_prediction(self):
        """Test that invalid predictions raise ValidationError with specific error location."""
        filenames = ["test1.jpg", "test2.jpg"]
        predictions = [
            {
                "boxes": [np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)],
                "orientation": 1,
            },
            {
                "boxes": [np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)],  # Invalid polygon with 2 points
                "orientation": 1,
            },
        ]

        with pytest.raises(ValidationError) as exc_info:
            validate_predictions(filenames, predictions)
        error_str = str(exc_info.value)
        assert "test2.jpg" in error_str  # Should mention the specific filename
        assert "Polygon requires at least three points" in error_str

    def test_predictions_with_invalid_orientation(self):
        """Test that invalid orientations in predictions raise ValidationError."""
        filenames = ["test.jpg"]
        predictions = [
            {
                "boxes": [np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)],
                "orientation": 9,  # Invalid orientation
            }
        ]

        with pytest.raises(ValidationError) as exc_info:
            validate_predictions(filenames, predictions)
        error_str = str(exc_info.value)
        assert "test.jpg" in error_str  # Should mention the specific filename
        assert "Orientation must be one of {0, 1, 2, 3, 4, 5, 6, 7, 8}" in error_str

    def test_predictions_with_empty_filename_list(self):
        """Test that empty filename and prediction lists pass validation."""
        filenames = []
        predictions = []

        result = validate_predictions(filenames, predictions)
        assert result == []

    def test_predictions_with_none_values_in_optional_fields(self):
        """Test that predictions with None values in optional fields pass validation."""
        filenames = ["test.jpg"]
        predictions = [
            {
                "boxes": [np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)],
                "orientation": 1,
                "raw_size": None,
                "canonical_size": None,
                "image_path": None,
            }
        ]

        result = validate_predictions(filenames, predictions)
        assert len(result) == 1
        assert result[0].raw_size is None
        assert result[0].canonical_size is None
        assert result[0].image_path is None
