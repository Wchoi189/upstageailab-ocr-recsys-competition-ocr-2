"""Metrics display component for comparison mode.

This module provides visualization of performance metrics and analysis:
- Performance metrics charts
- Metric comparison tables
- Statistical analysis
- Parameter impact visualization
"""

from typing import Any

import pandas as pd
import streamlit as st


def render_metrics_display(
    results: list[dict[str, Any]],
    comparison_config: dict[str, Any],
    state_key: str = "comparison_metrics",
) -> None:
    """Render metrics analysis and visualization.

    Args:
        results: List of result dictionaries with metrics
        comparison_config: Comparison mode configuration from YAML
        state_key: Session state key for metrics display settings
    """
    if not results:
        st.info("No results to analyze. Run a comparison first.")
        return

    metrics_config = comparison_config.get("metrics", {})

    # Display performance metrics
    st.subheader("📊 Performance Metrics")
    _render_performance_metrics(results, metrics_config)

    st.divider()

    # Display quality metrics
    st.subheader("🎯 Quality Metrics")
    _render_quality_metrics(results, metrics_config)

    st.divider()

    # Display comparison table
    st.subheader("📈 Metrics Comparison Table")
    _render_metrics_table(results, metrics_config)

    # Display charts if enabled
    charts_config = metrics_config.get("charts", {})
    if charts_config.get("show_comparison_table", True):
        st.divider()
        st.subheader("📊 Metrics Visualization")
        _render_metrics_charts(results, metrics_config)


def _render_performance_metrics(
    results: list[dict[str, Any]],
    metrics_config: dict[str, Any],
) -> None:
    """Render performance metrics overview."""
    perf_metrics = metrics_config.get("performance_metrics", [])

    if not perf_metrics:
        st.write("No performance metrics configured.")
        return

    # Create metrics summary
    cols = st.columns(len(perf_metrics))

    for idx, metric_def in enumerate(perf_metrics):
        metric_id = metric_def["id"]
        metric_label = metric_def["label"]
        metric_unit = metric_def.get("unit", "")
        format_str = metric_def.get("format", ".2f")

        # Calculate statistics
        values = _extract_metric_values(results, metric_id)

        if values:
            avg_value = sum(values) / len(values)
            min_value = min(values)
            max_value = max(values)

            with cols[idx]:
                st.metric(
                    label=metric_label,
                    value=f"{avg_value:{format_str}} {metric_unit}",
                    delta=f"Range: {min_value:{format_str}} - {max_value:{format_str}}",
                )


def _render_quality_metrics(
    results: list[dict[str, Any]],
    metrics_config: dict[str, Any],
) -> None:
    """Render quality metrics overview."""
    quality_metrics = metrics_config.get("quality_metrics", [])

    if not quality_metrics:
        st.write("No quality metrics configured.")
        return

    # Display quality metrics in a compact format
    for metric_def in quality_metrics:
        metric_id = metric_def["id"]
        metric_label = metric_def["label"]

        col1, col2 = st.columns([1, 3])

        with col1:
            st.write(f"**{metric_label}:**")

        with col2:
            # Get unique values
            values = _extract_metric_values(results, metric_id)
            unique_values = list(set(values))

            if len(unique_values) == 1:
                st.write(f"{unique_values[0]}")
            else:
                st.write(f"Varies: {', '.join(map(str, unique_values[:3]))}" + ("..." if len(unique_values) > 3 else ""))


def _render_metrics_table(
    results: list[dict[str, Any]],
    metrics_config: dict[str, Any],
) -> None:
    """Render comprehensive metrics comparison table."""
    # Collect all metrics
    perf_metrics = metrics_config.get("performance_metrics", [])

    # Build table data
    table_data = []

    for result in results:
        row = {
            "Configuration": result.get("config_label", "Unnamed"),
        }

        # Add performance metrics
        for metric_def in perf_metrics:
            metric_id = metric_def["id"]
            metric_label = metric_def["label"]
            format_str = metric_def.get("format", ".2f")
            unit = metric_def.get("unit", "")

            value = _get_metric_value(result, metric_id)
            if value is not None:
                if isinstance(value, int | float):
                    row[metric_label] = f"{value:{format_str}} {unit}".strip()
                else:
                    row[metric_label] = str(value)
            else:
                row[metric_label] = "N/A"

        table_data.append(row)

    # Display as dataframe
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True)


def _render_metrics_charts(
    results: list[dict[str, Any]],
    metrics_config: dict[str, Any],
) -> None:
    """Render metrics visualization charts."""
    perf_metrics = metrics_config.get("performance_metrics", [])
    charts_config = metrics_config.get("charts", {})

    # Filter metrics that should be shown in charts
    chart_metrics = [m for m in perf_metrics if m.get("show_chart", True)]

    if not chart_metrics:
        st.write("No charts configured.")
        return

    # Create tabs for different chart types
    tabs = st.tabs([m["label"] for m in chart_metrics])

    for idx, (metric_def, tab) in enumerate(zip(chart_metrics, tabs, strict=False)):
        with tab:
            _render_single_metric_chart(results, metric_def, charts_config)


def _render_single_metric_chart(
    results: list[dict[str, Any]],
    metric_def: dict[str, Any],
    charts_config: dict[str, Any],
) -> None:
    """Render a chart for a single metric."""
    metric_id = metric_def["id"]
    metric_label = metric_def["label"]

    # Prepare data
    chart_data = []
    for result in results:
        value = _get_metric_value(result, metric_id)
        if value is not None:
            chart_data.append(
                {
                    "Configuration": result.get("config_label", "Unnamed"),
                    metric_label: value,
                }
            )

    if not chart_data:
        st.write("No data available for this metric.")
        return

    df = pd.DataFrame(chart_data)

    # Render chart based on type
    chart_type = charts_config.get("type", "bar")

    if chart_type == "bar":
        st.bar_chart(df.set_index("Configuration"))
    elif chart_type == "line":
        st.line_chart(df.set_index("Configuration"))
    else:
        # Fallback to dataframe
        st.dataframe(df, use_container_width=True)

    # Show statistics
    with st.expander("Statistics", expanded=False):
        values = [item[metric_label] for item in chart_data]
        if values:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Mean", f"{sum(values) / len(values):.3f}")
            with col2:
                st.metric("Min", f"{min(values):.3f}")
            with col3:
                st.metric("Max", f"{max(values):.3f}")
            with col4:
                range_val = max(values) - min(values)
                st.metric("Range", f"{range_val:.3f}")


def _extract_metric_values(
    results: list[dict[str, Any]],
    metric_id: str,
) -> list[Any]:
    """Extract all values for a specific metric from results."""
    values = []

    for result in results:
        value = _get_metric_value(result, metric_id)
        if value is not None:
            values.append(value)

    return values


def _get_metric_value(
    result: dict[str, Any],
    metric_id: str,
) -> Any | None:
    """Get a specific metric value from a result."""
    # Check direct fields
    if metric_id == "processing_time":
        return result.get("processing_time")

    # Check metrics dict
    metrics = result.get("metrics", {})
    if metric_id in metrics:
        return metrics[metric_id]

    # Check nested paths
    if "." in metric_id:
        parts = metric_id.split(".")
        value: Any = result
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None
        return value

    return None


def render_analysis_summary(
    results: list[dict[str, Any]],
    comparison_config: dict[str, Any],
) -> None:
    """Render analysis summary and recommendations."""
    analysis_config = comparison_config.get("analysis", {})

    if not analysis_config.get("auto_analyze", {}).get("enabled", False):
        return

    st.divider()
    st.subheader("🎯 Analysis Summary")

    # Calculate overall scores if recommendations enabled
    if analysis_config.get("recommendations", {}).get("enabled", False):
        recommendations = _calculate_recommendations(results, analysis_config)

        if recommendations:
            st.success(f"**Recommended Configuration:** {recommendations[0]['label']}")

            with st.expander("View Detailed Recommendations", expanded=False):
                for idx, rec in enumerate(recommendations):
                    st.write(f"{idx + 1}. **{rec['label']}** (Score: {rec['score']:.3f})")
                    st.write(f"   - Processing Time: {rec.get('processing_time', 0):.3f}s")
                    if "reasoning" in rec:
                        st.write(f"   - {rec['reasoning']}")
    else:
        st.info("Auto-analysis disabled. Enable in configuration to see recommendations.")


def _calculate_recommendations(
    results: list[dict[str, Any]],
    analysis_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Calculate recommendations based on weighted criteria."""
    recommendations_config: dict[str, Any] = analysis_config.get("recommendations", {})
    criteria = recommendations_config.get("criteria", [])

    if not criteria:
        return []

    # Calculate weighted scores
    scored_results = []

    for result in results:
        total_score = 0.0
        total_weight = 0.0

        for criterion in criteria:
            metric = criterion["metric"]
            weight = criterion.get("weight", 1.0)
            lower_is_better = criterion.get("lower_is_better", False)

            value = _get_metric_value(result, metric)

            if value is not None:
                # Normalize value (simple min-max normalization)
                all_values = _extract_metric_values(results, metric)
                if all_values:
                    min_val = min(all_values)
                    max_val = max(all_values)

                    if max_val > min_val:
                        normalized = (value - min_val) / (max_val - min_val)
                        if lower_is_better:
                            normalized = 1.0 - normalized

                        total_score += normalized * weight
                        total_weight += weight

        # Calculate final score
        final_score = total_score / total_weight if total_weight > 0 else 0.0

        scored_results.append(
            {
                "label": result.get("config_label", "Unnamed"),
                "score": final_score,
                "processing_time": result.get("processing_time", 0.0),
            }
        )

    # Sort by score (descending)
    scored_results.sort(key=lambda x: x["score"], reverse=True)

    return scored_results
