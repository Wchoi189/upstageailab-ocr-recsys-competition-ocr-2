"""Preprocessing parameter panel component.

Provides interactive controls for all preprocessing parameters based on YAML config.
"""

from __future__ import annotations

import logging
from typing import Any

import streamlit as st

from ...models.app_state import UnifiedAppState

logger = logging.getLogger(__name__)


def render_parameter_panel(
    state: UnifiedAppState,
    mode_config: dict[str, Any],
) -> dict[str, Any]:
    """Render preprocessing parameter panel in sidebar.

    Args:
        state: Application state
        mode_config: Mode configuration from YAML

    Returns:
        Current parameter values as dict
    """
    parameters = mode_config.get("parameters", {})
    pipeline_stages = mode_config.get("pipeline", {}).get("stages", [])

    # Initialize current parameters from state or defaults
    if not state.preprocessing_config:
        state.preprocessing_config = _get_default_parameters(parameters)

    current_params = state.preprocessing_config.copy()

    # Show parameter sections based on pipeline stages
    st.sidebar.markdown("### ⚙️ Preprocessing Parameters")

    # Show advanced mode toggle
    show_advanced = st.sidebar.checkbox(
        "Show Advanced Parameters",
        value=state.get_preference("show_advanced_params", False),
        key="show_advanced_toggle",
    )
    state.set_preference("show_advanced_params", show_advanced)

    # Render parameter sections for each enabled stage
    for stage in pipeline_stages:
        stage["id"]
        stage_label = stage["label"]
        stage_config_key = stage["config_key"]

        # Get parameters for this stage
        stage_params = parameters.get(stage_config_key, {})

        if not stage_params:
            continue

        # Render section with expander
        with st.sidebar.expander(f"🎯 {stage_label}", expanded=stage.get("enabled", False)):
            # Check if there's an enable toggle
            enable_param = stage_params.get("enable")

            if enable_param:
                is_enabled = st.checkbox(
                    enable_param.get("label", "Enable"),
                    value=current_params.get(stage_config_key, {}).get(
                        "enable",
                        enable_param.get("default", False),
                    ),
                    help=enable_param.get("help", ""),
                    key=f"{stage_config_key}_enable",
                )

                # Update current params
                if stage_config_key not in current_params:
                    current_params[stage_config_key] = {}
                current_params[stage_config_key]["enable"] = is_enabled

                # Show performance warning if exists
                if "performance_warning" in enable_param and is_enabled:
                    st.warning(enable_param["performance_warning"])

                # Render other parameters if enabled
                if is_enabled:
                    _render_stage_parameters(
                        stage_config_key,
                        stage_params,
                        current_params,
                        show_advanced,
                    )
            else:
                # No enable toggle, just render all parameters
                _render_stage_parameters(
                    stage_config_key,
                    stage_params,
                    current_params,
                    show_advanced,
                )

    # Save to state
    state.preprocessing_config = current_params
    state.to_session()

    return current_params


def _render_stage_parameters(
    stage_key: str,
    stage_params: dict[str, Any],
    current_params: dict[str, Any],
    show_advanced: bool,
) -> None:
    """Render parameters for a specific stage.

    Args:
        stage_key: Stage configuration key
        stage_params: Parameter definitions from YAML
        current_params: Current parameter values
        show_advanced: Whether to show advanced parameters
    """
    # Ensure stage exists in current params
    if stage_key not in current_params:
        current_params[stage_key] = {}

    # Render each parameter
    for param_name, param_def in stage_params.items():
        # Skip 'enable' as it's already rendered
        if param_name == "enable":
            continue

        # Handle nested advanced parameters
        if param_name == "advanced":
            if show_advanced:
                st.markdown("**Advanced Parameters**")
                for adv_param_name, adv_param_def in param_def.items():
                    _render_single_parameter(
                        f"{stage_key}.{adv_param_name}",
                        adv_param_def,
                        current_params[stage_key],
                        adv_param_name,
                    )
            continue

        # Check if should show (based on show_advanced flag)
        if param_def.get("show_advanced", False) and not show_advanced:
            continue

        # Check dependencies
        depends_on = param_def.get("depends_on")
        if depends_on:
            # Check if dependency is satisfied
            if not current_params[stage_key].get(depends_on, False):
                continue

        # Render the parameter
        _render_single_parameter(
            f"{stage_key}.{param_name}",
            param_def,
            current_params[stage_key],
            param_name,
        )


def _render_single_parameter(
    widget_key: str,
    param_def: dict[str, Any],
    param_container: dict[str, Any],
    param_name: str,
) -> None:
    """Render a single parameter widget.

    Args:
        widget_key: Unique key for Streamlit widget
        param_def: Parameter definition from YAML
        param_container: Dict to store parameter value
        param_name: Name of parameter in container
    """
    param_type = param_def.get("type", "bool")
    label = param_def.get("label", param_name)
    help_text = param_def.get("help", "")
    default = param_def.get("default")

    # Get current value
    current_value = param_container.get(param_name, default)

    # Render widget based on type
    if param_type == "bool":
        value = st.checkbox(
            label,
            value=current_value if current_value is not None else default,
            help=help_text,
            key=widget_key,
        )

    elif param_type == "int":
        min_val = param_def.get("min", 0)
        max_val = param_def.get("max", 100)
        step = param_def.get("step", 1)

        value = st.slider(
            label,
            min_value=min_val,
            max_value=max_val,
            value=current_value if current_value is not None else default,
            step=step,
            help=help_text,
            key=widget_key,
        )

    elif param_type == "float":
        min_val = param_def.get("min", 0.0)
        max_val = param_def.get("max", 1.0)
        step = param_def.get("step", 0.01)

        value = st.slider(
            label,
            min_value=min_val,
            max_value=max_val,
            value=float(current_value if current_value is not None else default),
            step=step,
            help=help_text,
            key=widget_key,
        )

    elif param_type == "select":
        options = param_def.get("options", [])

        # Extract option values and labels
        if options and isinstance(options[0], dict):
            option_values = [opt["value"] for opt in options]
            option_labels = [opt["label"] for opt in options]

            # Find current index
            try:
                current_idx = option_values.index(current_value) if current_value else 0
            except ValueError:
                current_idx = 0

            selected_idx = st.selectbox(
                label,
                range(len(options)),
                index=current_idx,
                format_func=lambda i: option_labels[i],
                help=help_text,
                key=widget_key,
            )

            value = option_values[selected_idx]

            # Show performance info if available
            perf_info = options[selected_idx].get("performance")
            if perf_info:
                st.caption(f"⚡ {perf_info}")
        else:
            # Simple list of options
            value = st.selectbox(
                label,
                options,
                index=options.index(current_value) if current_value in options else 0,
                help=help_text,
                key=widget_key,
            )

    elif param_type == "text":
        value = st.text_input(
            label,
            value=current_value if current_value is not None else default,
            help=help_text,
            key=widget_key,
        )

    else:
        st.warning(f"Unknown parameter type: {param_type}")
        return

    # Update parameter container
    param_container[param_name] = value


def _get_default_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Extract default parameter values from config.

    Args:
        parameters: Parameter definitions from YAML

    Returns:
        Dict with default values for all parameters
    """
    defaults = {}

    for stage_key, stage_params in parameters.items():
        defaults[stage_key] = {}

        for param_name, param_def in stage_params.items():
            if param_name == "advanced":
                # Handle nested advanced parameters
                for adv_param_name, adv_param_def in param_def.items():
                    defaults[stage_key][adv_param_name] = adv_param_def.get("default")
            else:
                defaults[stage_key][param_name] = param_def.get("default")

    return defaults


def render_preset_management(
    current_params: dict[str, Any],
    mode_config: dict[str, Any],
) -> None:
    """Render preset save/load controls.

    Args:
        current_params: Current parameter values
        mode_config: Mode configuration from YAML
    """
    st.sidebar.markdown("### 💾 Preset Management")

    mode_config.get("export", {})

    col1, col2 = st.sidebar.columns(2)

    with col1:
        if st.button("📥 Load", use_container_width=True):
            st.info("Preset loading coming soon")

    with col2:
        if st.button("📤 Save", use_container_width=True):
            st.info("Preset saving coming soon")

    # Reset to defaults button
    if st.sidebar.button("🔄 Reset to Defaults", use_container_width=True):
        st.session_state.preprocessing_config = _get_default_parameters(mode_config.get("parameters", {}))
        st.rerun()
