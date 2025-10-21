"""Checkpoint selector component for inference mode.

Provides UI for selecting trained model checkpoints with metadata display.
"""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_checkpoint_selector(
    state: Any,
    config: dict[str, Any],
    checkpoints: list[Any],
) -> Any | None:
    """Render checkpoint selector with metadata display.

    Args:
        state: App state object
        config: Mode configuration from YAML
        checkpoints: List of available checkpoints

    Returns:
        Selected checkpoint metadata or None
    """
    if not checkpoints:
        st.warning("⚠️ No checkpoints found. Train a model first!")
        return None

    # Get configuration
    selector_config = config.get("model_selection", {}).get("checkpoint_selector", {})
    label = selector_config.get("label", "Select Checkpoint")
    help_text = selector_config.get("help_text", "")
    show_metadata = selector_config.get("show_metadata", True)

    # Create checkpoint options (display format)
    checkpoint_options = {}
    for ckpt in checkpoints:
        # Format display name based on checkpoint metadata
        display_name = _format_checkpoint_display_name(ckpt)
        checkpoint_options[display_name] = ckpt

    # Checkpoint selector
    st.subheader("🤖 Model Selection")

    selected_display_name = st.selectbox(
        label,
        options=list(checkpoint_options.keys()),
        help=help_text,
        key="checkpoint_selector",
    )

    if selected_display_name is None:
        return None

    selected_checkpoint = checkpoint_options[selected_display_name]

    # Show metadata if enabled
    if show_metadata and selected_checkpoint is not None:
        _render_checkpoint_metadata(selected_checkpoint, config)

    return selected_checkpoint


def _format_checkpoint_display_name(checkpoint: Any) -> str:
    """Format checkpoint display name from metadata.

    Args:
        checkpoint: Checkpoint info object

    Returns:
        Formatted display name
    """
    try:
        # Try to access common checkpoint attributes
        if hasattr(checkpoint, "metadata") and checkpoint.metadata:
            metadata = checkpoint.metadata
            arch = getattr(metadata.model, "architecture", "unknown")
            encoder = getattr(metadata.model, "encoder", "")
            epoch = getattr(metadata.training, "epoch", 0)

            if encoder:
                return f"{arch}-{encoder} (epoch {epoch})"
            return f"{arch} (epoch {epoch})"

        # Fallback to checkpoint path
        if hasattr(checkpoint, "checkpoint_path"):
            path = str(checkpoint.checkpoint_path)
            return path.split("/")[-1].replace(".ckpt", "")

        return "Unknown Checkpoint"

    except Exception:
        return "Unknown Checkpoint"


def _render_checkpoint_metadata(checkpoint: Any, config: dict[str, Any]) -> None:
    """Render checkpoint metadata display.

    Args:
        checkpoint: Checkpoint info object
        config: Mode configuration
    """
    try:
        if not hasattr(checkpoint, "metadata") or not checkpoint.metadata:
            return

        metadata = checkpoint.metadata

        with st.expander("📋 Checkpoint Details", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Model Information**")

                if hasattr(metadata, "model") and metadata.model:
                    model_info = metadata.model
                    if hasattr(model_info, "architecture"):
                        st.text(f"Architecture: {model_info.architecture}")
                    if hasattr(model_info, "encoder"):
                        st.text(f"Encoder: {model_info.encoder}")

                if hasattr(metadata, "training") and metadata.training:
                    training_info = metadata.training
                    if hasattr(training_info, "epoch"):
                        st.text(f"Epoch: {training_info.epoch}")
                    if hasattr(training_info, "global_step"):
                        st.text(f"Global Step: {training_info.global_step}")

            with col2:
                st.markdown("**Performance Metrics**")

                if hasattr(metadata, "metrics") and metadata.metrics:
                    metrics = metadata.metrics
                    if isinstance(metrics, dict):
                        for key, value in metrics.items():
                            if isinstance(value, float):
                                st.text(f"{key}: {value:.4f}")
                            else:
                                st.text(f"{key}: {value}")

                # Show checkpoint path
                if hasattr(checkpoint, "checkpoint_path"):
                    st.markdown("**Checkpoint Path**")
                    st.code(str(checkpoint.checkpoint_path), language="text")

    except Exception as e:
        st.error(f"Error displaying checkpoint metadata: {e}")


def render_mode_selector(config: dict[str, Any]) -> str:
    """Render processing mode selector (single vs batch).

    Args:
        config: Mode configuration from YAML

    Returns:
        Selected mode ('single' or 'batch')
    """
    mode_config = config.get("model_selection", {}).get("mode_selector", {})
    label = mode_config.get("label", "Processing Mode")
    options = mode_config.get("options", [])

    # Get default option
    default_option = next((opt for opt in options if opt.get("default", False)), options[0] if options else {})
    default_value = default_option.get("value", "single")

    # Create display options
    display_options = {f"{opt.get('icon', '')} {opt.get('label', opt['value'])}": opt["value"] for opt in options}

    # Find default index
    default_display = next(
        (display for display, value in display_options.items() if value == default_value),
        list(display_options.keys())[0] if display_options else None,
    )
    default_index = list(display_options.keys()).index(default_display) if default_display else 0

    st.subheader("Processing Mode")
    selected_display = st.radio(
        label,
        options=list(display_options.keys()),
        index=default_index,
        horizontal=True,
        key="inference_processing_mode_selector",
    )

    return display_options[selected_display]


def render_hyperparameters(config: dict[str, Any]) -> dict[str, float]:
    """Render hyperparameter sliders.

    Args:
        config: Mode configuration from YAML

    Returns:
        Dict of hyperparameter values
    """
    hyper_config = config.get("hyperparameters", {})

    if not hyper_config:
        return {}

    st.subheader("⚙️ Hyperparameters")

    hyperparameters = {}

    for param_name, param_config in hyper_config.items():
        if not isinstance(param_config, dict):
            continue

        label = param_config.get("label", param_name)
        min_val = param_config.get("min", 0.0)
        max_val = param_config.get("max", 1.0)
        default_val = param_config.get("default", 0.5)
        step = param_config.get("step", 0.05)
        help_text = param_config.get("help_text", "")

        value = st.slider(
            label,
            min_value=min_val,
            max_value=max_val,
            value=default_val,
            step=step,
            help=help_text,
            key=f"hyperparam_{param_name}",
        )

        hyperparameters[param_name] = value

    return hyperparameters
