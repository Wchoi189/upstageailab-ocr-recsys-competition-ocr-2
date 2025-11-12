"""Results comparison UI component for comparison mode.

This module provides side-by-side comparison visualization of results from
different parameter configurations, including:
- Grid/side-by-side image display
- Parameter labels and metadata
- Visual difference highlighting
- Sortable results
"""

from typing import Any

import streamlit as st


def render_results_comparison(
    results: list[dict[str, Any]],
    comparison_config: dict[str, Any],
    state_key: str = "comparison_results",
) -> None:
    """Render results comparison grid or table.

    Args:
        results: List of result dictionaries with structure:
            {
                "config_label": str,
                "config_params": dict,
                "image": np.ndarray or PIL.Image,
                "metrics": dict,
                "processing_time": float,
            }
        comparison_config: Comparison mode configuration from YAML
        state_key: Session state key for comparison display settings
    """
    if not results:
        st.info("No results to display. Run a comparison first.")
        return

    # Initialize state
    if state_key not in st.session_state:
        st.session_state[state_key] = {
            "layout": "grid",
            "sort_by": "processing_time",
            "ascending": True,
            "show_params": True,
            "show_metrics": True,
        }

    display_state = st.session_state[state_key]
    results_config = comparison_config.get("results_comparison", {})

    # Display controls
    _render_display_controls(display_state, results_config, state_key)

    st.divider()

    # Sort results if enabled
    if results_config.get("sorting", {}).get("enabled", True):
        sorted_results = _sort_results(
            results,
            display_state["sort_by"],
            display_state["ascending"],
        )
    else:
        sorted_results = results

    # Highlight best result if enabled
    best_idx = None
    if results_config.get("display_options", {}).get("highlight_best", True):
        best_idx = _find_best_result(sorted_results, display_state["sort_by"])

    # Render based on layout
    layout_type = display_state["layout"]

    if layout_type == "grid":
        _render_grid_layout(
            sorted_results,
            results_config,
            display_state,
            best_idx,
        )
    elif layout_type == "side_by_side":
        _render_side_by_side_layout(
            sorted_results,
            results_config,
            display_state,
            best_idx,
        )
    elif layout_type == "table":
        _render_table_layout(
            sorted_results,
            results_config,
            display_state,
            best_idx,
        )


def _render_display_controls(
    display_state: dict[str, Any],
    results_config: dict[str, Any],
    state_key: str,
) -> None:
    """Render display control panel."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        layout_options = ["grid", "side_by_side", "table"]
        display_state["layout"] = st.selectbox(
            "Layout",
            options=layout_options,
            index=layout_options.index(display_state["layout"]),
            key=f"{state_key}_layout",
        )

    with col2:
        sort_options = ["processing_time", "config_label", "num_detections"]
        display_state["sort_by"] = st.selectbox(
            "Sort by",
            options=sort_options,
            index=sort_options.index(display_state["sort_by"]) if display_state["sort_by"] in sort_options else 0,
            key=f"{state_key}_sort",
        )

    with col3:
        display_state["ascending"] = st.checkbox(
            "Ascending",
            value=display_state["ascending"],
            key=f"{state_key}_ascending",
        )

    with col4:
        display_state["show_params"] = st.checkbox(
            "Show parameters",
            value=display_state["show_params"],
            key=f"{state_key}_show_params",
        )


def _render_grid_layout(
    results: list[dict[str, Any]],
    results_config: dict[str, Any],
    display_state: dict[str, Any],
    best_idx: int | None,
) -> None:
    """Render results in a grid layout."""
    layout_config = results_config.get("layout", {})
    max_columns = layout_config.get("max_columns", 3)
    columns = layout_config.get("columns", 2)

    # Determine number of columns based on number of results
    if layout_config.get("auto_layout", True):
        num_results = len(results)
        if num_results == 1:
            columns = 1
        elif num_results == 2:
            columns = 2
        else:
            columns = min(columns, min(num_results, max_columns))

    # Create grid
    cols = st.columns(columns)

    for idx, result in enumerate(results):
        col_idx = idx % columns
        with cols[col_idx]:
            _render_result_card(
                result,
                results_config,
                display_state,
                is_best=(idx == best_idx),
                card_idx=idx,
            )


def _render_side_by_side_layout(
    results: list[dict[str, Any]],
    results_config: dict[str, Any],
    display_state: dict[str, Any],
    best_idx: int | None,
) -> None:
    """Render results side by side."""
    num_results = len(results)
    cols = st.columns(num_results)

    for idx, (result, col) in enumerate(zip(results, cols, strict=False)):
        with col:
            _render_result_card(
                result,
                results_config,
                display_state,
                is_best=(idx == best_idx),
                card_idx=idx,
            )


def _render_table_layout(
    results: list[dict[str, Any]],
    results_config: dict[str, Any],
    display_state: dict[str, Any],
    best_idx: int | None,
) -> None:
    """Render results in a table format."""
    # Create table data
    table_data = []

    for idx, result in enumerate(results):
        row = {
            "Configuration": result.get("config_label", f"Config {idx + 1}"),
            "Processing Time (s)": f"{result.get('processing_time', 0.0):.3f}",
        }

        # Add metrics
        metrics = result.get("metrics", {})
        if "num_detections" in metrics:
            row["Detections"] = metrics["num_detections"]
        if "avg_confidence" in metrics:
            row["Avg Confidence"] = f"{metrics['avg_confidence']:.3f}"

        # Mark best result
        if idx == best_idx:
            row[""] = "⭐"
        else:
            row[""] = ""

        table_data.append(row)

    st.dataframe(table_data, use_container_width=True)

    # Show images in expandable sections
    for idx, result in enumerate(results):
        with st.expander(f"View: {result.get('config_label', f'Config {idx + 1}')}"):
            image = result.get("image")
            if image is not None:
                st.image(image, use_container_width=True)

            if display_state["show_params"]:
                st.json(result.get("config_params", {}), expanded=False)


def _render_result_card(
    result: dict[str, Any],
    results_config: dict[str, Any],
    display_state: dict[str, Any],
    is_best: bool,
    card_idx: int,
) -> None:
    """Render a single result card."""
    display_options = results_config.get("display_options", {})

    # Header with label
    config_label = result.get("config_label", f"Config {card_idx + 1}")

    if is_best and display_options.get("highlight_best", True):
        st.markdown(f"### ⭐ {config_label}")
    else:
        st.markdown(f"### {config_label}")

    # Display image
    image = result.get("image")
    if image is not None:
        st.image(
            image,
            use_container_width=True,
            caption=config_label,
        )

    # Display parameters if enabled
    if display_state["show_params"] and display_options.get("show_parameters", True):
        with st.expander("Parameters", expanded=False):
            params = result.get("config_params", {})
            if params:
                st.json(params, expanded=False)
            else:
                st.write("No parameters configured")

    # Display metrics
    if display_state.get("show_metrics", True):
        _render_result_metrics(result, display_options)


def _render_result_metrics(
    result: dict[str, Any],
    display_options: dict[str, Any],
) -> None:
    """Render metrics for a single result."""
    metrics = result.get("metrics", {})
    processing_time = result.get("processing_time", 0.0)

    # Always show processing time
    if display_options.get("show_processing_time", True):
        st.metric("Processing Time", f"{processing_time:.3f}s")

    # Show other metrics
    if "num_detections" in metrics:
        st.metric("Detections", metrics["num_detections"])

    if "avg_confidence" in metrics:
        st.metric("Avg Confidence", f"{metrics['avg_confidence']:.3f}")

    if "preprocessing_time" in metrics:
        st.metric("Preprocessing", f"{metrics['preprocessing_time']:.3f}s")

    if "inference_time" in metrics:
        st.metric("Inference", f"{metrics['inference_time']:.3f}s")


def _sort_results(
    results: list[dict[str, Any]],
    sort_by: str,
    ascending: bool,
) -> list[dict[str, Any]]:
    """Sort results by specified metric."""

    def get_sort_key(result: dict[str, Any]) -> Any:
        if sort_by == "config_label":
            return result.get("config_label", "")
        elif sort_by == "processing_time":
            return result.get("processing_time", float("inf"))
        elif sort_by == "num_detections":
            return result.get("metrics", {}).get("num_detections", 0)
        elif sort_by == "avg_confidence":
            return result.get("metrics", {}).get("avg_confidence", 0.0)
        else:
            return 0

    return sorted(results, key=get_sort_key, reverse=not ascending)


def _find_best_result(
    results: list[dict[str, Any]],
    metric: str,
) -> int | None:
    """Find the best result based on a metric."""
    if not results:
        return None

    # Determine if lower is better
    lower_is_better = metric in ["processing_time", "preprocessing_time", "inference_time"]

    best_idx = 0
    best_value = None

    for idx, result in enumerate(results):
        if metric == "config_label":
            continue  # Can't determine "best" for labels

        # Get metric value
        if metric == "processing_time":
            value = result.get("processing_time", float("inf"))
        elif metric in ["num_detections", "avg_confidence"]:
            value = result.get("metrics", {}).get(metric, 0)
        else:
            value = result.get("metrics", {}).get(metric, 0)

        # Update best
        if best_value is None:
            best_value = value
            best_idx = idx
        elif lower_is_better and value < best_value:
            best_value = value
            best_idx = idx
        elif not lower_is_better and value > best_value:
            best_value = value
            best_idx = idx

    return best_idx


def render_export_controls(
    results: list[dict[str, Any]],
    comparison_config: dict[str, Any],
) -> None:
    """Render export controls for comparison results."""
    if not results:
        return

    st.divider()
    st.subheader("💾 Export Results")

    export_config = comparison_config.get("export", {})
    formats = export_config.get("formats", [])

    col1, col2 = st.columns([2, 1])

    with col1:
        format_options = {f["value"]: f["label"] for f in formats}
        selected_format = st.selectbox(
            "Export format",
            options=list(format_options.keys()),
            format_func=lambda x: format_options[x],
            key="export_format",
        )

    with col2:
        if st.button("Download", key="export_download"):
            # Generate export data based on format
            export_data = _generate_export_data(results, selected_format, export_config)

            if export_data:
                st.download_button(
                    label=f"Download {format_options[selected_format]}",
                    data=export_data,
                    file_name=f"comparison_results.{selected_format}",
                    mime=_get_mime_type(selected_format),
                    key="export_download_btn",
                )


def _generate_export_data(
    results: list[dict[str, Any]],
    format_type: str,
    export_config: dict[str, Any],
) -> str | bytes | None:
    """Generate export data in specified format."""
    import io
    import json

    if format_type == "json":
        # Export as JSON
        export_obj = {
            "results": [
                {
                    "config_label": r.get("config_label", ""),
                    "config_params": r.get("config_params", {}),
                    "metrics": r.get("metrics", {}),
                    "processing_time": r.get("processing_time", 0.0),
                }
                for r in results
            ],
        }
        return json.dumps(export_obj, indent=2)

    elif format_type == "csv":
        # Export as CSV
        import csv

        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(["Configuration", "Processing Time", "Metrics"])

        # Data rows
        for result in results:
            writer.writerow(
                [
                    result.get("config_label", ""),
                    result.get("processing_time", 0.0),
                    json.dumps(result.get("metrics", {})),
                ]
            )

        return output.getvalue()

    elif format_type == "yaml":
        # Export as YAML
        import yaml

        export_obj = {
            "results": [
                {
                    "config_label": r.get("config_label", ""),
                    "config_params": r.get("config_params", {}),
                    "metrics": r.get("metrics", {}),
                    "processing_time": r.get("processing_time", 0.0),
                }
                for r in results
            ],
        }
        return yaml.dump(export_obj, default_flow_style=False)

    return None


def _get_mime_type(format_type: str) -> str:
    """Get MIME type for file format."""
    mime_types = {
        "json": "application/json",
        "csv": "text/csv",
        "yaml": "application/x-yaml",
        "html": "text/html",
    }
    return mime_types.get(format_type, "text/plain")
