"""Results viewer component for inference mode.

Displays OCR inference results with visualization and export options.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import streamlit as st


def render_results_viewer(
    state: Any,
    config: dict[str, Any],
) -> None:
    """Render inference results viewer.

    Args:
        state: App state object with inference results
        config: Mode configuration from YAML
    """
    results_config = config.get("results", {})

    # Check if there are results to display
    if not hasattr(state, "inference_results") or not state.inference_results:
        st.info("👋 Upload an image and run inference to see results here.")
        return

    # Get the most recent result
    latest_result = state.inference_results[-1] if isinstance(state.inference_results, list) else state.inference_results

    # Render results in tabs
    tab1, tab2 = st.tabs(["📊 Results", "👁️ Visualization"])

    with tab1:
        _render_results_text(latest_result, results_config)

    with tab2:
        _render_results_visualization(latest_result, results_config)


def _render_results_text(result: Any, config: dict[str, Any]) -> None:
    """Render text results display.

    Args:
        result: Inference result object
        config: Results configuration
    """
    display_options = config.get("display_options", {})
    show_confidence = display_options.get("show_confidence_scores", True)
    show_processing_time = display_options.get("show_processing_time", True)
    show_image_dims = display_options.get("show_image_dimensions", True)

    st.subheader("Inference Results")

    # Display metadata
    if show_processing_time and hasattr(result, "processing_time"):
        st.metric("Processing Time", f"{result.processing_time:.3f}s")

    if show_image_dims and hasattr(result, "image_shape"):
        st.text(f"Image Dimensions: {result.image_shape}")

    # Display detected text regions
    if hasattr(result, "polygons") and result.polygons:
        st.markdown("### Detected Text Regions")
        st.text(f"Total Regions: {len(result.polygons)}")

        # Show polygons in a table
        polygon_data = []
        for idx, polygon in enumerate(result.polygons):
            row = {"Region": idx + 1}

            # Add coordinates (simplified)
            if isinstance(polygon, list | np.ndarray):
                poly_array = np.array(polygon)
                if poly_array.size >= 8:  # At least 4 points
                    row["Points"] = f"{len(poly_array) // 2}"
                    # Calculate bounding box
                    x_coords = poly_array[::2] if poly_array.ndim == 1 else poly_array[:, 0]
                    y_coords = poly_array[1::2] if poly_array.ndim == 1 else poly_array[:, 1]
                    row["Bounds"] = f"({int(x_coords.min())}, {int(y_coords.min())}) - ({int(x_coords.max())}, {int(y_coords.max())})"

            # Add confidence score if available
            if show_confidence and hasattr(result, "scores") and result.scores:
                if idx < len(result.scores):
                    row["Confidence"] = f"{result.scores[idx]:.3f}"

            polygon_data.append(row)

        if polygon_data:
            st.dataframe(polygon_data, use_container_width=True)

    else:
        st.info("No text regions detected.")

    # Export options
    _render_export_options(result, config)


def _render_results_visualization(result: Any, config: dict[str, Any]) -> None:
    """Render visualization of results on image.

    Args:
        result: Inference result object
        config: Results configuration
    """
    viz_config = config.get("visualization", {})
    show_polygons = viz_config.get("show_polygons", True)
    show_scores = viz_config.get("show_scores", True)
    polygon_color = tuple(viz_config.get("polygon_color", [0, 255, 0]))
    polygon_thickness = viz_config.get("polygon_thickness", 2)

    st.subheader("Visualization")

    # Check if we have image and polygons
    if not hasattr(result, "image") or result.image is None:
        st.warning("No image available for visualization.")
        return

    # Create visualization
    try:
        viz_image = result.image.copy()

        if show_polygons and hasattr(result, "polygons") and result.polygons:
            for idx, polygon in enumerate(result.polygons):
                if isinstance(polygon, list | np.ndarray):
                    poly_array = np.array(polygon, dtype=np.int32)

                    # Reshape if needed
                    if poly_array.ndim == 1 and poly_array.size >= 8:
                        # Flat array of coordinates [x1, y1, x2, y2, ...]
                        poly_array = poly_array.reshape(-1, 2)
                    elif poly_array.ndim == 2 and poly_array.shape[1] == 2:
                        # Already in correct shape
                        pass
                    else:
                        continue

                    # Draw polygon
                    cv2.polylines(viz_image, [poly_array], True, polygon_color, polygon_thickness)

                    # Add confidence score if available
                    if show_scores and hasattr(result, "scores") and result.scores and idx < len(result.scores):
                        # Get top-left corner for text placement
                        x, y = poly_array[0]
                        score_text = f"{result.scores[idx]:.2f}"
                        cv2.putText(
                            viz_image,
                            score_text,
                            (int(x), int(y) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            polygon_color,
                            1,
                            cv2.LINE_AA,
                        )

        # Display visualization
        # Convert BGR to RGB if needed
        if len(viz_image.shape) == 3 and viz_image.shape[2] == 3:
            # Assume BGR format (OpenCV default)
            viz_image = cv2.cvtColor(viz_image, cv2.COLOR_BGR2RGB)

        st.image(viz_image, caption="Inference Results", use_container_width=True)

    except Exception as e:
        st.error(f"Error creating visualization: {e}")


def _render_export_options(result: Any, config: dict[str, Any]) -> None:
    """Render export options for results.

    Args:
        result: Inference result object
        config: Results configuration
    """
    export_config = config.get("export", {})
    formats = export_config.get("formats", [])

    if not formats:
        return

    st.markdown("### Export Results")

    # Create export format selector
    format_options = {fmt.get("label", fmt["value"]): fmt["value"] for fmt in formats}
    selected_format = st.selectbox("Export Format", options=list(format_options.keys()), key="export_format")

    export_format = format_options[selected_format]

    if st.button("💾 Export", key="export_button"):
        try:
            exported_data = _export_results(result, export_format)

            # Create download button
            file_extension = export_format
            file_name = f"inference_results.{file_extension}"

            st.download_button(
                label=f"Download {file_extension.upper()}",
                data=exported_data,
                file_name=file_name,
                mime=_get_mime_type(export_format),
                key="download_results",
            )

        except Exception as e:
            st.error(f"Error exporting results: {e}")


def _export_results(result: Any, format: str) -> str:
    """Export results to specified format.

    Args:
        result: Inference result object
        format: Export format ('json', 'csv', 'txt')

    Returns:
        Exported data as string
    """
    import json

    if format == "json":
        # Export as JSON
        data = {
            "num_regions": len(result.polygons) if hasattr(result, "polygons") else 0,
            "polygons": [poly.tolist() if isinstance(poly, np.ndarray) else poly for poly in result.polygons]
            if hasattr(result, "polygons")
            else [],
            "scores": result.scores.tolist()
            if hasattr(result, "scores") and isinstance(result.scores, np.ndarray)
            else (result.scores if hasattr(result, "scores") else []),
        }
        return json.dumps(data, indent=2)

    elif format == "csv":
        # Export as CSV
        lines = ["region_id,num_points,confidence\n"]
        if hasattr(result, "polygons") and result.polygons:
            for idx, polygon in enumerate(result.polygons):
                poly_array = np.array(polygon)
                num_points = len(poly_array) // 2 if poly_array.ndim == 1 else len(poly_array)
                confidence = result.scores[idx] if hasattr(result, "scores") and idx < len(result.scores) else 0.0
                lines.append(f"{idx},{num_points},{confidence:.4f}\n")
        return "".join(lines)

    elif format == "txt":
        # Export as plain text
        lines = ["Inference Results\n", "=" * 50 + "\n\n"]
        lines.append(f"Total Regions: {len(result.polygons) if hasattr(result, 'polygons') else 0}\n\n")

        if hasattr(result, "polygons") and result.polygons:
            for idx, polygon in enumerate(result.polygons):
                lines.append(f"Region {idx + 1}:\n")
                lines.append(f"  Polygon: {polygon}\n")
                if hasattr(result, "scores") and idx < len(result.scores):
                    lines.append(f"  Confidence: {result.scores[idx]:.4f}\n")
                lines.append("\n")

        return "".join(lines)

    else:
        raise ValueError(f"Unsupported export format: {format}")


def _get_mime_type(format: str) -> str:
    """Get MIME type for export format.

    Args:
        format: Export format

    Returns:
        MIME type string
    """
    mime_types = {
        "json": "application/json",
        "csv": "text/csv",
        "txt": "text/plain",
    }
    return mime_types.get(format, "text/plain")
