"""Preprocessing stage viewer component.

Provides visualization of preprocessing pipeline stages with side-by-side and step-by-step views.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import streamlit as st

from ...models.app_state import UnifiedAppState
from ..shared.image_display import (
    display_image_grid,
    display_side_by_side,
)

logger = logging.getLogger(__name__)


def render_stage_viewer(
    state: UnifiedAppState,
    mode_config: dict[str, Any],
    processing_results: dict[str, np.ndarray] | None = None,
) -> None:
    """Render preprocessing stage viewer in main area.

    Args:
        state: Application state
        mode_config: Mode configuration from YAML
        processing_results: Dict mapping stage names to processed images
    """
    # Get layout configuration
    layout_config = mode_config.get("layout", {})
    tabs_config = layout_config.get("main_area", {}).get("tabs", [])

    # Create tabs
    tab_labels = [tab["label"] for tab in tabs_config]
    tabs = st.tabs(tab_labels)

    # Render each tab
    for idx, tab_config in enumerate(tabs_config):
        with tabs[idx]:
            tab_id = tab_config["id"]
            tab_description = tab_config.get("description", "")

            if tab_description:
                st.caption(tab_description)

            if tab_id == "side_by_side":
                _render_side_by_side_view(state, mode_config, processing_results)

            elif tab_id == "step_by_step":
                _render_step_by_step_view(state, mode_config, processing_results)

            elif tab_id == "parameters":
                _render_parameters_view(state, mode_config)

            else:
                st.info(f"Tab '{tab_id}' coming soon")


def _render_side_by_side_view(
    state: UnifiedAppState,
    mode_config: dict[str, Any],
    processing_results: dict[str, np.ndarray] | None,
) -> None:
    """Render side-by-side comparison view.

    Args:
        state: Application state
        mode_config: Mode configuration
        processing_results: Processing results by stage
    """
    current_image = state.get_current_image()

    if current_image is None:
        st.info("📤 Upload an image to start preprocessing")
        return

    if processing_results is None or not processing_results:
        st.warning("⚙️ Configure parameters and run preprocessing")
        return

    # Get pipeline stages
    pipeline_stages = mode_config.get("pipeline", {}).get("stages", [])
    enabled_stages = [s for s in pipeline_stages if s.get("enabled", False)]

    if not enabled_stages:
        st.warning("⚠️ No preprocessing stages enabled")
        return

    # Display options
    col1, col2 = st.columns([3, 1])

    with col1:
        # Stage selector for comparison
        stage_options = ["original"] + [s["id"] for s in enabled_stages]
        stage_labels = ["Original"] + [s["label"] for s in enabled_stages]

        left_stage = st.selectbox(
            "Left image",
            range(len(stage_options)),
            index=0,
            format_func=lambda i: stage_labels[i],
            key="left_stage_selector",
        )

        right_stage = st.selectbox(
            "Right image",
            range(len(stage_options)),
            index=len(stage_options) - 1,  # Default to last stage
            format_func=lambda i: stage_labels[i],
            key="right_stage_selector",
        )

    with col2:
        display_mode = st.radio(
            "Display",
            ["Split", "Grid"],
            horizontal=True,
            key="display_mode_selector",
        )

    st.divider()

    # Get selected images
    left_image = current_image if left_stage == 0 else processing_results.get(stage_options[left_stage])
    right_image = current_image if right_stage == 0 else processing_results.get(stage_options[right_stage])

    # Display based on mode
    if display_mode == "Split":
        if left_image is not None and right_image is not None:
            display_side_by_side(
                left_image,
                right_image,
                labels=(stage_labels[left_stage], stage_labels[right_stage]),
            )
        else:
            st.warning("Selected stages not available in results")
    else:
        # Grid view - show all stages
        images = [current_image]
        labels = ["Original"]

        for stage in enabled_stages:
            stage_id = stage["id"]
            if stage_id in processing_results:
                images.append(processing_results[stage_id])
                labels.append(stage["label"])

        display_image_grid(images, labels, mode_config)


def _render_step_by_step_view(
    state: UnifiedAppState,
    mode_config: dict[str, Any],
    processing_results: dict[str, np.ndarray] | None,
) -> None:
    """Render step-by-step execution view.

    Args:
        state: Application state
        mode_config: Mode configuration
        processing_results: Processing results by stage
    """
    current_image = state.get_current_image()

    if current_image is None:
        st.info("📤 Upload an image to start preprocessing")
        return

    # Get pipeline stages
    pipeline_stages = mode_config.get("pipeline", {}).get("stages", [])
    enabled_stages = [s for s in pipeline_stages if s.get("enabled", False)]

    if not enabled_stages:
        st.warning("⚠️ No preprocessing stages enabled")
        return

    # Stage selector
    st.markdown("### 🎯 Select Stage to Execute")

    stage_options = list(range(len(enabled_stages)))
    stage_labels = [s["label"] for s in enabled_stages]

    selected_stage_idx = st.select_slider(
        "Pipeline stage",
        options=stage_options,
        value=state.get_preference("current_step_stage", 0),
        format_func=lambda i: f"Step {i+1}: {stage_labels[i]}",
        key="step_stage_selector",
    )

    # Save preference
    state.set_preference("current_step_stage", selected_stage_idx)

    selected_stage = enabled_stages[selected_stage_idx]

    # Show stage info
    st.markdown(f"**{selected_stage['label']}**")
    st.caption(selected_stage.get("description", ""))

    if "performance_note" in selected_stage:
        st.info(f"ℹ️ {selected_stage['performance_note']}")

    st.divider()

    # Show result if available
    stage_id = selected_stage["id"]

    if processing_results and stage_id in processing_results:
        # Show before/after
        st.markdown("### Results")

        # Get previous stage image (or original)
        if selected_stage_idx == 0:
            before_image = current_image
            before_label = "Original"
        else:
            prev_stage_id = enabled_stages[selected_stage_idx - 1]["id"]
            before_image = processing_results.get(prev_stage_id, current_image)
            before_label = enabled_stages[selected_stage_idx - 1]["label"]

        after_image = processing_results[stage_id]
        after_label = selected_stage["label"]

        # Display comparison
        display_side_by_side(
            before_image,
            after_image,
            labels=(f"Before: {before_label}", f"After: {after_label}"),
        )

        # Show stage metadata if available
        if hasattr(state, "preprocessing_metadata"):
            stage_meta = state.preprocessing_metadata.get(stage_id, {})
            if stage_meta:
                with st.expander("📊 Stage Metadata"):
                    st.json(stage_meta)
    else:
        st.info("⚙️ Run preprocessing to see results for this stage")

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button(
            "⬅️ Previous",
            disabled=selected_stage_idx == 0,
            use_container_width=True,
        ):
            state.set_preference("current_step_stage", selected_stage_idx - 1)
            st.rerun()

    with col3:
        if st.button(
            "Next ➡️",
            disabled=selected_stage_idx == len(enabled_stages) - 1,
            use_container_width=True,
        ):
            state.set_preference("current_step_stage", selected_stage_idx + 1)
            st.rerun()


def _render_parameters_view(
    state: UnifiedAppState,
    mode_config: dict[str, Any],
) -> None:
    """Render detailed parameters view with validation.

    Args:
        state: Application state
        mode_config: Mode configuration
    """
    st.markdown("### 🛠️ Current Configuration")

    if not state.preprocessing_config:
        st.info("No parameters configured yet")
        return

    # Display current config
    with st.expander("📋 Full Configuration", expanded=True):
        st.json(state.preprocessing_config)

    st.divider()

    # Show parameter summary by stage
    st.markdown("### 📊 Parameter Summary")

    pipeline_stages = mode_config.get("pipeline", {}).get("stages", [])

    for stage in pipeline_stages:
        stage["id"]
        stage_config_key = stage["config_key"]

        stage_params = state.preprocessing_config.get(stage_config_key, {})

        if not stage_params:
            continue

        is_enabled = stage_params.get("enable", False)

        # Show stage card
        with st.expander(
            f"{'✅' if is_enabled else '⏸️'} {stage['label']}",
            expanded=is_enabled,
        ):
            if is_enabled:
                # Show parameters
                for param_name, param_value in stage_params.items():
                    if param_name != "enable":
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.text(param_name.replace("_", " ").title())
                        with col2:
                            st.code(str(param_value))
            else:
                st.caption("Stage disabled")

    # Export section
    st.divider()
    st.markdown("### 💾 Export Configuration")

    export_config = mode_config.get("export", {})
    formats = export_config.get("formats", ["yaml", "json"])

    export_format = st.radio(
        "Format",
        formats,
        horizontal=True,
        key="export_format_selector",
    )

    if st.button("📥 Download Configuration", use_container_width=True):
        import json
        from datetime import datetime

        import yaml

        # Prepare export data
        export_data = {
            "preprocessing": state.preprocessing_config,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_images": len(state.uploaded_images),
            },
        }

        # Convert to format
        if export_format == "yaml":
            export_str = yaml.dump(export_data, default_flow_style=False)
            filename = f"preprocessing_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
            mime = "text/yaml"
        else:
            export_str = json.dumps(export_data, indent=2)
            filename = f"preprocessing_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            mime = "application/json"

        st.download_button(
            label="⬇️ Download",
            data=export_str,
            file_name=filename,
            mime=mime,
            use_container_width=True,
        )
