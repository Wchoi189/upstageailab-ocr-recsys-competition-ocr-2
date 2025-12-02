"""VGG Image Annotator (VIA) Integration.

Handles VIA annotation export/import, overlay generation, and workflow integration.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont

from agent_qms.vlm.core.contracts import VIAAnnotation, VIARegion
from agent_qms.vlm.core.interfaces import IntegrationError
from agent_qms.vlm.utils.paths import get_path_resolver


class VIAIntegration:
    """Integration with VGG Image Annotator."""

    def __init__(self, via_dir: Optional[Path] = None):
        """Initialize VIA integration.

        Args:
            via_dir: Directory containing VIA files
        """
        resolver = get_path_resolver()
        self.via_dir = via_dir or resolver.get_via_annotations_path()
        self.via_dir.mkdir(parents=True, exist_ok=True)

    def load_annotations(self, annotation_path: Path) -> VIAAnnotation:
        """Load VIA annotations from JSON file.

        Args:
            annotation_path: Path to VIA annotation JSON file

        Returns:
            Validated VIA annotation

        Raises:
            IntegrationError: If loading fails
        """
        try:
            data = json.loads(annotation_path.read_text())

            # VIA format: {filename: {size, regions, file_attributes}}
            # We need to extract the first (or only) entry
            if isinstance(data, dict):
                # Find the filename entry
                filename = None
                annotation_data = None

                for key, value in data.items():
                    if isinstance(value, dict) and "size" in value:
                        filename = key
                        annotation_data = value
                        break

                if not annotation_data:
                    raise IntegrationError("No valid annotation data found in VIA file")

                regions = []
                for region_data in annotation_data.get("regions", []):
                    if isinstance(region_data, dict):
                        regions.append(
                            VIARegion(
                                shape_attributes=region_data.get("shape_attributes", {}),
                                region_attributes=region_data.get("region_attributes", {}),
                            )
                        )

                return VIAAnnotation(
                    filename=filename or annotation_path.stem,
                    size=annotation_data.get("size", 0),
                    regions=regions,
                    file_attributes=annotation_data.get("file_attributes", {}),
                )
            else:
                raise IntegrationError("Invalid VIA annotation format")

        except Exception as e:
            raise IntegrationError(f"Failed to load VIA annotations: {e}") from e

    def create_overlay(
        self,
        image_path: Path,
        annotation: VIAAnnotation,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Create an image overlay with VIA annotations.

        Args:
            image_path: Path to source image
            annotation: VIA annotation data
            output_path: Optional output path (auto-generated if not provided)

        Returns:
            Path to annotated image

        Raises:
            IntegrationError: If overlay creation fails
        """
        try:
            with Image.open(image_path) as img:
                # Create a copy for drawing
                overlay = img.copy()
                draw = ImageDraw.Draw(overlay)

                # Draw each region
                for region in annotation.regions:
                    shape_attrs = region.shape_attributes
                    region_type = shape_attrs.get("name", "rect")

                    if region_type == "rect":
                        # Bounding box
                        x = shape_attrs.get("x", 0)
                        y = shape_attrs.get("y", 0)
                        width = shape_attrs.get("width", 0)
                        height = shape_attrs.get("height", 0)

                        draw.rectangle(
                            [(x, y), (x + width, y + height)],
                            outline="red",
                            width=2,
                        )

                    elif region_type == "polygon":
                        # Polygon
                        all_points_x = shape_attrs.get("all_points_x", [])
                        all_points_y = shape_attrs.get("all_points_y", [])

                        if len(all_points_x) == len(all_points_y):
                            points = list(zip(all_points_x, all_points_y))
                            draw.polygon(points, outline="red", width=2)

                    elif region_type == "circle":
                        # Circle
                        cx = shape_attrs.get("cx", 0)
                        cy = shape_attrs.get("cy", 0)
                        r = shape_attrs.get("r", 0)

                        draw.ellipse(
                            [(cx - r, cy - r), (cx + r, cy + r)],
                            outline="red",
                            width=2,
                        )

                    # Add label if available
                    label = region.region_attributes.get("label", "")
                    if label and region_type == "rect":
                        x = shape_attrs.get("x", 0)
                        y = shape_attrs.get("y", 0)
                        try:
                            # Try to use a default font
                            font = ImageFont.load_default()
                        except Exception:
                            font = None
                        draw.text((x, y - 15), label, fill="red", font=font)

                # Save overlay
                if output_path is None:
                    output_path = image_path.parent / f"{image_path.stem}_annotated{image_path.suffix}"

                overlay.save(output_path)
                return output_path

        except Exception as e:
            raise IntegrationError(f"Failed to create annotation overlay: {e}") from e

    def export_annotations_for_vlm(
        self,
        annotation: VIAAnnotation,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Export annotations in a format suitable for VLM prompts.

        Args:
            annotation: VIA annotation
            output_path: Optional path to save export

        Returns:
            Dictionary with annotation descriptions for VLM
        """
        descriptions = []

        for i, region in enumerate(annotation.regions, 1):
            shape_attrs = region.shape_attributes
            region_attrs = region.region_attributes

            region_type = shape_attrs.get("name", "unknown")
            label = region_attrs.get("label", f"Region {i}")

            description = {
                "label": label,
                "type": region_type,
                "description": f"{label} ({region_type})",
            }

            if region_type == "rect":
                description["bounds"] = {
                    "x": shape_attrs.get("x", 0),
                    "y": shape_attrs.get("y", 0),
                    "width": shape_attrs.get("width", 0),
                    "height": shape_attrs.get("height", 0),
                }

            descriptions.append(description)

        result = {
            "filename": annotation.filename,
            "annotations": descriptions,
        }

        if output_path:
            output_path.write_text(json.dumps(result, indent=2))

        return result

    def get_via_html_path(self) -> Optional[Path]:
        """Get path to VIA HTML file.

        Returns:
            Path to via.html if it exists, None otherwise
        """
        via_html = self.via_dir.parent / "via.html"
        if via_html.exists():
            return via_html

        # Also check in via subdirectory
        via_html = self.via_dir / "via.html"
        if via_html.exists():
            return via_html

        return None
