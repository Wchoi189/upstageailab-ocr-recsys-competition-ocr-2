"""
Tests for coordinate transformation correctness in geometry_utils.

🚨 CRITICAL TEST SUITE - DO NOT DISABLE OR MODIFY WITHOUT UNDERSTANDING IMPACT

BUG-20251116-001: These tests verify that inverse_matrix computation correctly
handles different padding positions. Failure of these tests indicates coordinate
transformation errors that will cause:
- Negative coordinates
- Out-of-bounds coordinates
- Incorrect evaluation metrics
- Misaligned visualizations

See: docs/bug_reports/BUG-20251116-001_DEBUGGING_HANDOVER.md
"""

from __future__ import annotations

import numpy as np

from ocr.core.utils.geometry_utils import calculate_cropbox, calculate_inverse_transform


class TestCoordinateTransformationCorrectness:
    """Test coordinate transformation correctness for different padding positions."""

    def test_top_left_padding_no_translation(self):
        """Test that top_left padding produces inverse matrix with no translation."""
        original_size = (960, 1280)  # width, height
        target_size = 640

        # Compute crop box for top_left padding
        crop_box = calculate_cropbox(original_size, target_size, position="top_left")
        x, y, w, h = crop_box

        # For top_left, offset should be (0, 0)
        assert x == 0, "Top-left padding should have x=0 offset"
        assert y == 0, "Top-left padding should have y=0 offset"

        # Compute inverse matrix
        inv_matrix = calculate_inverse_transform(
            original_size,
            (target_size, target_size),  # transformed_size (padded)
            crop_box=crop_box,
            padding_position="top_left",
        )

        # Translation components should be (0, 0) for top_left
        assert abs(inv_matrix[0, 2]) < 1e-6, f"Expected translation x=0, got {inv_matrix[0, 2]}"
        assert abs(inv_matrix[1, 2]) < 1e-6, f"Expected translation y=0, got {inv_matrix[1, 2]}"

    def test_center_padding_has_translation(self):
        """Test that center padding produces inverse matrix with translation."""
        original_size = (960, 1280)  # width, height
        target_size = 640

        # Compute crop box for center padding
        crop_box = calculate_cropbox(original_size, target_size, position="center")
        x, y, w, h = crop_box

        # For center, offset should be non-zero (centered)
        assert x > 0 or y > 0, "Center padding should have non-zero offset"

        # Compute inverse matrix
        inv_matrix = calculate_inverse_transform(
            original_size,
            (target_size, target_size),  # transformed_size (padded)
            crop_box=crop_box,
            padding_position="center",
        )

        # Translation components should be non-zero for center
        # The translation should undo the centering offset
        assert abs(inv_matrix[0, 2]) > 1e-6 or abs(inv_matrix[1, 2]) > 1e-6, "Center padding should have non-zero translation"

    def test_coordinate_roundtrip_top_left(self):
        """Test that coordinates can be transformed and back with top_left padding."""
        original_size = (960, 1280)  # width, height
        target_size = 640

        # Compute inverse matrix for top_left
        crop_box = calculate_cropbox(original_size, target_size, position="top_left")
        inv_matrix = calculate_inverse_transform(
            original_size,
            (target_size, target_size),
            crop_box=crop_box,
            padding_position="top_left",
        )

        # Test point in original space
        original_point = np.array([[480.0, 640.0]])  # Center of original image

        # Transform to 640x640 space (forward transform would be: scale down)
        scale = target_size / max(original_size)
        transformed_point = original_point * scale

        # Transform back using inverse matrix
        homogeneous = np.vstack([transformed_point.T, [1]])
        recovered = inv_matrix @ homogeneous
        recovered = (recovered[:2] / recovered[2]).T

        # Should recover original point (within rounding error)
        assert np.allclose(recovered, original_point, atol=1.0), f"Roundtrip failed: original={original_point[0]}, recovered={recovered[0]}"

    def test_coordinate_roundtrip_center(self):
        """Test that coordinates can be transformed and back with center padding."""
        original_size = (960, 1280)  # width, height
        target_size = 640

        # Compute inverse matrix for center
        crop_box = calculate_cropbox(original_size, target_size, position="center")
        inv_matrix = calculate_inverse_transform(
            original_size,
            (target_size, target_size),
            crop_box=crop_box,
            padding_position="center",
        )

        # Test point in original space
        original_point = np.array([[480.0, 640.0]])  # Center of original image

        # Transform to 640x640 space (forward transform: scale down, then center)
        scale = target_size / max(original_size)
        scaled_point = original_point * scale
        # Center padding adds offset
        x, y, w, h = crop_box
        transformed_point = scaled_point + np.array([[x, y]])

        # Transform back using inverse matrix
        homogeneous = np.vstack([transformed_point.T, [1]])
        recovered = inv_matrix @ homogeneous
        recovered = (recovered[:2] / recovered[2]).T

        # Should recover original point (within rounding error)
        assert np.allclose(recovered, original_point, atol=1.0), f"Roundtrip failed: original={original_point[0]}, recovered={recovered[0]}"

    def test_no_negative_coordinates_top_left(self):
        """Test that top_left padding doesn't produce negative coordinates."""
        original_size = (960, 1280)  # width, height
        target_size = 640

        crop_box = calculate_cropbox(original_size, target_size, position="top_left")
        x, y, w, h = crop_box
        inv_matrix = calculate_inverse_transform(
            original_size,
            (target_size, target_size),
            crop_box=crop_box,
            padding_position="top_left",
        )

        # Test points within the actual scaled image area (not padding)
        # For top_left, image starts at (0, 0) and has size (w, h)
        test_points = np.array(
            [
                [0, 0],  # Top-left corner of scaled image
                [w // 2, h // 2],  # Center of scaled image
                [w - 1, h - 1],  # Bottom-right corner of scaled image
            ]
        )

        # Transform back to original space
        homogeneous = np.vstack([test_points.T, np.ones(test_points.shape[0])])
        transformed = inv_matrix @ homogeneous
        transformed = (transformed[:2] / transformed[2]).T

        # All coordinates should be non-negative
        assert np.all(transformed >= 0), f"Negative coordinates detected: {transformed[transformed < 0]}"

        # All coordinates should be within original image bounds
        assert np.all(
            transformed[:, 0] <= original_size[0]
        ), f"X coordinates out of bounds: {transformed[transformed[:, 0] > original_size[0]]}"
        assert np.all(
            transformed[:, 1] <= original_size[1]
        ), f"Y coordinates out of bounds: {transformed[transformed[:, 1] > original_size[1]]}"

    def test_no_negative_coordinates_center(self):
        """Test that center padding doesn't produce negative coordinates."""
        original_size = (960, 1280)  # width, height
        target_size = 640

        crop_box = calculate_cropbox(original_size, target_size, position="center")
        x, y, w, h = crop_box
        inv_matrix = calculate_inverse_transform(
            original_size,
            (target_size, target_size),
            crop_box=crop_box,
            padding_position="center",
        )

        # Test points within the actual scaled image area (not padding)
        # For center, image starts at (x, y) and has size (w, h)
        test_points = np.array(
            [
                [x, y],  # Top-left corner of scaled image
                [x + w // 2, y + h // 2],  # Center of scaled image
                [x + w - 1, y + h - 1],  # Bottom-right corner of scaled image
            ]
        )

        # Transform back to original space
        homogeneous = np.vstack([test_points.T, np.ones(test_points.shape[0])])
        transformed = inv_matrix @ homogeneous
        transformed = (transformed[:2] / transformed[2]).T

        # All coordinates should be non-negative
        assert np.all(transformed >= 0), f"Negative coordinates detected: {transformed[transformed < 0]}"

        # All coordinates should be within original image bounds
        assert np.all(
            transformed[:, 0] <= original_size[0]
        ), f"X coordinates out of bounds: {transformed[transformed[:, 0] > original_size[0]]}"
        assert np.all(
            transformed[:, 1] <= original_size[1]
        ), f"Y coordinates out of bounds: {transformed[transformed[:, 1] > original_size[1]]}"

    def test_padding_position_mismatch_detection(self):
        """Test that different padding positions produce different inverse matrices."""
        original_size = (960, 1280)  # width, height
        target_size = 640

        # Compute for top_left
        crop_box_top_left = calculate_cropbox(original_size, target_size, position="top_left")
        inv_matrix_top_left = calculate_inverse_transform(
            original_size,
            (target_size, target_size),
            crop_box=crop_box_top_left,
            padding_position="top_left",
        )

        # Compute for center
        crop_box_center = calculate_cropbox(original_size, target_size, position="center")
        inv_matrix_center = calculate_inverse_transform(
            original_size,
            (target_size, target_size),
            crop_box=crop_box_center,
            padding_position="center",
        )

        # Matrices should be different (translation components differ)
        assert not np.allclose(
            inv_matrix_top_left, inv_matrix_center
        ), "Top-left and center padding should produce different inverse matrices"

        # Specifically, translation components should differ
        translation_diff = np.abs(inv_matrix_top_left[:2, 2] - inv_matrix_center[:2, 2])
        assert np.any(translation_diff > 1e-6), "Translation components should differ between padding positions"
