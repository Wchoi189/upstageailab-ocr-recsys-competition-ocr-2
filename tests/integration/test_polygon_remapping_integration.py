"""
Integration tests for polygon remapping and validation pipeline.

BUG-20251116-001: Tests the full pipeline to understand why errors persist
even after tolerance increase.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from ocr.datasets.schemas import ValidatedPolygonData
from ocr.utils.orientation import polygons_in_canonical_frame, remap_polygons


class TestPolygonRemappingIntegration:
    """Test the full polygon remapping and validation pipeline."""

    def test_orientation_6_polygon_remapping_scenario(self):
        """
        Test the scenario from investigation: Orientation 6, polygon with y=1281.9.

        This simulates what happens in the dataset:
        1. Load polygon from annotation (already in canonical frame)
        2. Check if it's in canonical frame
        3. If not, remap it
        4. Validate the result
        """
        orientation = 6
        raw_width, raw_height = 1280, 960
        canonical_width, canonical_height = 960, 1280

        # Polygon from investigation: y=1281.9 (1.9 pixels over canonical height)
        # This polygon is already in canonical frame (4-point rectangle)
        polygon_original = np.array([
            [276.1, 1258.7],
            [322.5, 1258.7],
            [322.5, 1281.9],
            [276.1, 1281.9]
        ], dtype=np.float32)

        # Step 1: Check if it's detected as canonical with new tolerance (3.0)
        is_canonical = polygons_in_canonical_frame(
            [polygon_original], raw_width, raw_height, orientation, tolerance=3.0
        )
        assert is_canonical is True, "Should detect as canonical with 3.0 tolerance"

        # Step 2: If not canonical, remap it (but it should be skipped)
        if not is_canonical:
            polygon_remapped = remap_polygons([polygon_original], raw_width, raw_height, orientation)[0]
        else:
            polygon_remapped = polygon_original

        # Step 3: Validate the polygon
        # The validation should pass because coordinates are within tolerance
        try:
            validated = ValidatedPolygonData(
                points=polygon_remapped,
                image_width=canonical_width,
                image_height=canonical_height
            )
            # Should not raise an error
            assert validated.points.shape == polygon_remapped.shape
        except Exception as e:
            pytest.fail(f"Validation should pass but raised: {e}")

    def test_orientation_6_polygon_that_gets_remapped(self):
        """
        Test what happens when a polygon in raw frame gets remapped.

        This simulates the case where polygons_in_canonical_frame returns False,
        so the polygon gets remapped, and then we validate it.
        """
        orientation = 6
        raw_width, raw_height = 1280, 960
        canonical_width, canonical_height = 960, 1280

        # Polygon in raw frame (fits raw dimensions) - 4-point rectangle
        polygon_raw = np.array([
            [100, 200],
            [300, 200],
            [300, 400],
            [100, 400]
        ], dtype=np.float32)

        # Step 1: Check if it's detected as canonical
        is_canonical = polygons_in_canonical_frame(
            [polygon_raw], raw_width, raw_height, orientation, tolerance=3.0
        )
        assert is_canonical is False, "Should NOT detect as canonical (fits raw dimensions)"

        # Step 2: Remap it
        polygon_remapped = remap_polygons([polygon_raw], raw_width, raw_height, orientation)[0]

        # Step 3: Validate the remapped polygon
        try:
            validated = ValidatedPolygonData(
                points=polygon_remapped,
                image_width=canonical_width,
                image_height=canonical_height
            )
            # Should not raise an error
            assert validated.points.shape == polygon_remapped.shape
        except Exception as e:
            pytest.fail(f"Validation should pass but raised: {e}")

    def test_orientation_1_negative_coordinate_scenario(self):
        """
        Test the scenario from logs: Orientation 1, polygon with x=-6.0.

        This polygon is beyond tolerance, so it should be rejected.
        """
        orientation = 1  # No rotation
        raw_width, raw_height = 605, 1280
        canonical_width, canonical_height = 605, 1280

        # Polygon from logs: x=-6.0 (3 pixels beyond -3.0 tolerance) - 4-point rectangle
        polygon = np.array([
            [-6.0, 20.0],
            [308.2, 20.0],
            [308.2, 39.7],
            [-6.0, 39.7]
        ], dtype=np.float32)

        # Step 1: Check if it's detected as canonical
        # For orientation 1, orientation_requires_rotation returns False,
        # so polygons_in_canonical_frame returns False immediately
        is_canonical = polygons_in_canonical_frame(
            [polygon], raw_width, raw_height, orientation, tolerance=3.0
        )
        assert is_canonical is False, "Orientation 1 doesn't require rotation check"

        # Step 2: No remapping needed (orientation 1)
        polygon_processed = polygon

        # Step 3: Validate - should fail because x=-6.0 is beyond tolerance
        with pytest.raises(ValidationError):
            ValidatedPolygonData(
                points=polygon_processed,
                image_width=canonical_width,
                image_height=canonical_height
            )

    def test_orientation_6_double_remapping_prevention(self):
        """
        Test that polygons already in canonical frame are not remapped twice.

        BUG-20251116-001: This is the key fix - preventing double remapping.
        """
        orientation = 6
        raw_width, raw_height = 1280, 960
        canonical_width, canonical_height = 960, 1280

        # Polygon already in canonical frame (slightly over boundary) - 4-point rectangle
        polygon_canonical = np.array([
            [276.1, 1258.7],
            [322.5, 1258.7],
            [322.5, 1281.9],
            [276.1, 1281.9]
        ], dtype=np.float32)

        # With old tolerance (1.5): Would NOT be detected as canonical
        is_canonical_old = polygons_in_canonical_frame(
            [polygon_canonical], raw_width, raw_height, orientation, tolerance=1.5
        )
        assert is_canonical_old is False, "Old tolerance (1.5) should NOT detect"

        # With new tolerance (3.0): SHOULD be detected as canonical
        is_canonical_new = polygons_in_canonical_frame(
            [polygon_canonical], raw_width, raw_height, orientation, tolerance=3.0
        )
        assert is_canonical_new is True, "New tolerance (3.0) should detect"

        # If not detected as canonical, it would get remapped (double rotation)
        if not is_canonical_new:
            polygon_double_remapped = remap_polygons(
                [polygon_canonical], raw_width, raw_height, orientation
            )[0]
            # This would produce wrong coordinates (like negative x values)
            # But with new tolerance, this should not happen
            assert False, "Should not reach here - polygon should be detected as canonical"

    def test_remapping_produces_out_of_bounds(self):
        """
        Test that remapping a polygon already in canonical frame produces wrong coordinates.

        This demonstrates why the tolerance fix is important.
        """
        orientation = 6
        raw_width, raw_height = 1280, 960
        canonical_width, canonical_height = 960, 1280

        # Polygon already in canonical frame - 4-point rectangle
        polygon_canonical = np.array([
            [276.1, 1258.7],
            [322.5, 1258.7],
            [322.5, 1281.9],
            [276.1, 1281.9]
        ], dtype=np.float32)

        # If we incorrectly remap it (double rotation), we get wrong coordinates
        polygon_wrongly_remapped = remap_polygons(
            [polygon_canonical], raw_width, raw_height, orientation
        )[0]

        # The remapped coordinates should be wrong (negative x values)
        # This is what was happening before the fix
        assert np.any(polygon_wrongly_remapped[:, 0] < 0), "Double remapping produces negative x"

        # And validation should fail
        with pytest.raises(ValidationError):
            ValidatedPolygonData(
                points=polygon_wrongly_remapped,
                image_width=canonical_width,
                image_height=canonical_height
            )

    def test_why_errors_persist_analysis(self):
        """
        Analyze why some errors persist even after the tolerance fix.

        The fix prevents double-remapping, but some polygons are legitimately
        out of bounds in the source annotations.
        """
        # Case 1: Polygon beyond tolerance (legitimately invalid)
        # x=-6.0 when tolerance is 3.0 → 3 pixels beyond → correctly rejected
        polygon1 = np.array([
            [-6.0, 20.0],
            [300.0, 20.0],
            [300.0, 40.0],
            [-6.0, 40.0]
        ], dtype=np.float32)
        with pytest.raises(ValidationError):
            ValidatedPolygonData(
                points=polygon1,
                image_width=605,
                image_height=1280
            )

        # Case 2: Polygon way beyond bounds (legitimately invalid)
        # x=1290.0 when width=1280 → 7 pixels beyond → correctly rejected
        polygon2 = np.array([
            [1260.0, 100.0],
            [1290.0, 100.0],
            [1290.0, 200.0],
            [1260.0, 200.0]
        ], dtype=np.float32)
        with pytest.raises(ValidationError):
            ValidatedPolygonData(
                points=polygon2,
                image_width=1280,
                image_height=959
            )

        # Case 3: Polygon within tolerance (should be accepted)
        # x=-2.0 when tolerance is 3.0 → within tolerance → should be clamped and accepted
        polygon3 = np.array([
            [-2.0, 20.0],
            [300.0, 20.0],
            [300.0, 40.0],
            [-2.0, 40.0]
        ], dtype=np.float32)
        try:
            validated3 = ValidatedPolygonData(
                points=polygon3,
                image_width=605,
                image_height=1280
            )
            # Should be clamped to x=0.0
            assert validated3.points[0, 0] == 0.0, "Should clamp -2.0 to 0.0"
        except Exception as e:
            pytest.fail(f"Polygon within tolerance should be accepted: {e}")

