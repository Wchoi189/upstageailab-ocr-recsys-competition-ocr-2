"""Parameter sweep UI component for comparison mode.

This module provides UI controls for defining parameter sweeps, including:
- Manual parameter selection
- Range-based sweeps
- Grid search configurations
- Preset comparisons
"""

from typing import Any

import streamlit as st


def render_parameter_sweep(
    comparison_config: dict[str, Any],
    state_key: str = "comparison_sweep",
) -> dict[str, Any]:
    """Render parameter sweep configuration UI.

    Args:
        comparison_config: Comparison mode configuration from YAML
        state_key: Session state key for storing sweep configuration

    Returns:
        Dictionary containing sweep configuration with structure:
        {
            "comparison_type": str,  # preprocessing, inference, or end_to_end
            "sweep_mode": str,  # manual, range, or grid
            "configurations": list[dict],  # List of parameter configurations
            "max_configs": int,  # Maximum number of configurations
        }
    """
    # Initialize state
    if state_key not in st.session_state:
        st.session_state[state_key] = {
            "comparison_type": "preprocessing",
            "sweep_mode": "manual",
            "configurations": [],
            "max_configs": 5,
        }

    sweep_config = st.session_state[state_key]

    # Comparison type selector
    st.subheader("📊 Comparison Type")
    comparison_types = comparison_config.get("comparison_types", [])

    type_options = {ct["id"]: f"{ct.get('icon', '')} {ct['label']}" for ct in comparison_types}

    selected_type_label = st.selectbox(
        "Select comparison type",
        options=list(type_options.values()),
        index=list(type_options.keys()).index(sweep_config["comparison_type"]) if sweep_config["comparison_type"] in type_options else 0,
        help="Choose what aspect to compare",
        key=f"{state_key}_type",
    )

    # Get comparison type ID from label
    selected_type = next(ct_id for ct_id, label in type_options.items() if label == selected_type_label)

    if selected_type != sweep_config["comparison_type"]:
        sweep_config["comparison_type"] = selected_type
        sweep_config["configurations"] = []  # Reset configurations

    st.divider()

    # Sweep mode selector
    st.subheader("🔄 Sweep Mode")
    sweep_settings = comparison_config.get("parameter_sweep", {})
    mode_options_config = sweep_settings.get("mode_selector", {}).get("options", [])

    mode_options = {opt["value"]: f"{opt['label']}\n{opt.get('description', '')}" for opt in mode_options_config}

    selected_mode = st.radio(
        "How do you want to configure the comparison?",
        options=list(mode_options.keys()),
        format_func=lambda x: mode_options[x].split("\n")[0],
        index=list(mode_options.keys()).index(sweep_config["sweep_mode"]) if sweep_config["sweep_mode"] in mode_options else 0,
        key=f"{state_key}_mode",
    )

    if selected_mode != sweep_config["sweep_mode"]:
        sweep_config["sweep_mode"] = selected_mode
        sweep_config["configurations"] = []  # Reset configurations

    st.divider()

    # Max configurations slider
    max_configs = st.slider(
        "Maximum configurations to compare",
        min_value=sweep_settings.get("max_configurations", {}).get("min", 2),
        max_value=sweep_settings.get("max_configurations", {}).get("max", 10),
        value=sweep_config["max_configs"],
        help=sweep_settings.get("max_configurations", {}).get("help_text", ""),
        key=f"{state_key}_max_configs",
    )
    sweep_config["max_configs"] = max_configs

    st.divider()

    # Render mode-specific configuration UI
    if selected_mode == "manual":
        _render_manual_mode(
            sweep_config,
            comparison_config,
            selected_type,
            state_key,
        )
    elif selected_mode == "range":
        _render_range_mode(
            sweep_config,
            comparison_config,
            selected_type,
            state_key,
        )
    elif selected_mode == "grid":
        _render_grid_mode(
            sweep_config,
            comparison_config,
            selected_type,
            state_key,
        )

    # Show preset selector
    _render_preset_selector(sweep_config, comparison_config, state_key)

    return sweep_config


def _render_manual_mode(
    sweep_config: dict[str, Any],
    comparison_config: dict[str, Any],
    comparison_type: str,
    state_key: str,
) -> None:
    """Render manual parameter selection mode."""
    st.subheader("🎯 Manual Configuration")
    st.write("Add specific configurations to compare:")

    # Display existing configurations
    num_configs = len(sweep_config["configurations"])

    if num_configs == 0:
        st.info("No configurations added yet. Click 'Add Configuration' below.")
    else:
        for idx, config in enumerate(sweep_config["configurations"]):
            with st.expander(f"Configuration {idx + 1}: {config.get('label', 'Unnamed')}", expanded=False):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.json(config.get("params", {}), expanded=False)
                with col2:
                    if st.button("Remove", key=f"{state_key}_remove_{idx}"):
                        sweep_config["configurations"].pop(idx)
                        st.rerun()

    # Add new configuration button
    if num_configs < sweep_config["max_configs"]:
        if st.button("➕ Add Configuration", key=f"{state_key}_add_config"):
            # Add a default configuration
            sweep_config["configurations"].append(
                {
                    "label": f"Config {num_configs + 1}",
                    "params": {},
                }
            )
            st.rerun()
    else:
        st.warning(f"Maximum of {sweep_config['max_configs']} configurations reached.")


def _render_range_mode(
    sweep_config: dict[str, Any],
    comparison_config: dict[str, Any],
    comparison_type: str,
    state_key: str,
) -> None:
    """Render range sweep mode."""
    st.subheader("📏 Range Sweep")
    st.write("Test a range of values for a single parameter:")

    sweep_settings = comparison_config.get("parameter_sweep", {})

    # Get parameter definitions
    if comparison_type == "preprocessing":
        param_defs = sweep_settings.get("preprocessing_parameters", {})
    elif comparison_type == "inference":
        param_defs = sweep_settings.get("inference_parameters", {})
    else:
        param_defs = {}

    if not param_defs:
        st.warning("No parameters available for range sweep in this mode.")
        return

    # Parameter selector
    param_names = list(param_defs.keys())
    selected_param = st.selectbox(
        "Select parameter",
        options=param_names,
        format_func=lambda x: param_defs[x].get("label", x),
        key=f"{state_key}_range_param",
    )

    param_def = param_defs[selected_param]

    # Range configuration based on parameter type
    if param_def.get("type") == "numeric":
        min_val = st.number_input(
            "Minimum value",
            value=param_def.get("min", 0.0),
            min_value=param_def.get("min", 0.0),
            max_value=param_def.get("max", 1.0),
            step=param_def.get("step", 0.1),
            key=f"{state_key}_range_min",
        )

        max_val = st.number_input(
            "Maximum value",
            value=param_def.get("max", 1.0),
            min_value=param_def.get("min", 0.0),
            max_value=param_def.get("max", 1.0),
            step=param_def.get("step", 0.1),
            key=f"{state_key}_range_max",
        )

        num_steps = st.slider(
            "Number of steps",
            min_value=2,
            max_value=sweep_config["max_configs"],
            value=min(5, sweep_config["max_configs"]),
            key=f"{state_key}_range_steps",
        )

        if st.button("Generate Range", key=f"{state_key}_generate_range"):
            import numpy as np

            values = np.linspace(min_val, max_val, num_steps)
            sweep_config["configurations"] = [
                {
                    "label": f"{param_def.get('label', selected_param)} = {val:.3f}",
                    "params": {selected_param: float(val)},
                }
                for val in values
            ]
            st.success(f"Generated {num_steps} configurations")
            st.rerun()

    elif param_def.get("type") == "categorical":
        values = param_def.get("values", [])
        selected_values = st.multiselect(
            "Select values to compare",
            options=values,
            default=param_def.get("default_sweep", values[:2]),
            key=f"{state_key}_range_categorical",
        )

        if st.button("Generate Configurations", key=f"{state_key}_generate_categorical"):
            sweep_config["configurations"] = [
                {
                    "label": f"{param_def.get('label', selected_param)} = {val}",
                    "params": {selected_param: val},
                }
                for val in selected_values
            ]
            st.success(f"Generated {len(selected_values)} configurations")
            st.rerun()


def _render_grid_mode(
    sweep_config: dict[str, Any],
    comparison_config: dict[str, Any],
    comparison_type: str,
    state_key: str,
) -> None:
    """Render grid search mode."""
    st.subheader("🔲 Grid Search")
    st.write("Test all combinations of multiple parameters:")

    sweep_settings = comparison_config.get("parameter_sweep", {})

    # Get parameter definitions
    if comparison_type == "preprocessing":
        param_defs = sweep_settings.get("preprocessing_parameters", {})
    elif comparison_type == "inference":
        param_defs = sweep_settings.get("inference_parameters", {})
    else:
        param_defs = {}

    if not param_defs:
        st.warning("No parameters available for grid search in this mode.")
        return

    st.info("Grid search is not yet implemented. Use manual or range mode instead.")
    # TODO: Implement grid search functionality


def _render_preset_selector(
    sweep_config: dict[str, Any],
    comparison_config: dict[str, Any],
    state_key: str,
) -> None:
    """Render preset configuration selector."""
    presets = comparison_config.get("presets", [])

    if not presets:
        return

    st.divider()
    st.subheader("💾 Quick Presets")

    preset_options = {p["id"]: p["label"] for p in presets}

    selected_preset_id = st.selectbox(
        "Load a preset configuration",
        options=[""] + list(preset_options.keys()),
        format_func=lambda x: "-- Select a preset --" if x == "" else preset_options[x],
        key=f"{state_key}_preset",
    )

    if selected_preset_id:
        preset = next(p for p in presets if p["id"] == selected_preset_id)

        with st.expander("Preset Details", expanded=True):
            st.write(f"**Description:** {preset.get('description', 'N/A')}")
            st.write(f"**Type:** {preset.get('comparison_type', 'N/A')}")

            if "configurations" in preset:
                st.write(f"**Configurations:** {len(preset['configurations'])}")

        if st.button("Load Preset", key=f"{state_key}_load_preset"):
            sweep_config["comparison_type"] = preset.get("comparison_type", "preprocessing")

            if "configurations" in preset:
                sweep_config["configurations"] = preset["configurations"]
            elif "sweep_config" in preset:
                # Handle sweep-based presets
                st.info("Sweep-based presets not yet implemented")

            st.success(f"Loaded preset: {preset['label']}")
            st.rerun()


def _get_available_parameters(
    comparison_config: dict[str, Any],
    comparison_type: str,
) -> list[str]:
    """Get available parameters for a comparison type."""
    comparison_types = comparison_config.get("comparison_types", [])

    for ct in comparison_types:
        if ct["id"] == comparison_type:
            return ct.get("available_parameters", [])

    return []
