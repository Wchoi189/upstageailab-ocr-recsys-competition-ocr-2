"""Unified OCR Development Studio - Main Application.

Single entry point for preprocessing tuning, model inference, and A/B testing.

Architecture:
    - YAML-driven configuration (configs/ui/unified_app.yaml)
    - Mode-based UI (preprocessing, inference, comparison)
    - Service layer for business logic
    - Pydantic models for type safety

Usage:
    streamlit run ui/apps/unified_ocr_app/app.py
"""

from __future__ import annotations

import os

# VERY FIRST LINE - ABSOLUTE FIRST DEBUG POINT
import sys

_debug_f = open("/tmp/streamlit_debug.log", "a")
_debug_f.write("\n" + "=" * 80 + "\n")
_debug_f.write("ABSOLUTE FIRST LINE EXECUTED!\n")
_debug_f.write("File: " + __file__ + "\n")
_debug_f.write("CWD: " + os.getcwd() + "\n")
_debug_f.flush()

import logging
from pathlib import Path
from typing import Any, Literal, cast

# Debug file for troubleshooting - write BEFORE any imports
# Use absolute path to ensure we can find it
debug_file_path = Path("/tmp/streamlit_debug.log")
try:
    with open(debug_file_path, "a") as f:
        f.write(f"DEBUG: Script {__file__} starting...\n")
        f.write(f"DEBUG: CWD = {os.getcwd()}\n")
        f.flush()
except Exception as e:
    # Failsafe - if we can't write, at least print
    print(f"FAILED TO WRITE DEBUG LOG: {e}", file=sys.stderr, flush=True)

import streamlit as st

with open(debug_file_path, "a") as f:
    f.write("DEBUG: streamlit imported\n")
    f.flush()

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

with open(debug_file_path, "a") as f:
    f.write("DEBUG: project root added to path\n")
    f.write("DEBUG: Starting imports...\n")
    f.flush()

debug_file = debug_file_path  # Keep same variable name for rest of code

from ui.apps.unified_ocr_app.models.app_state import UnifiedAppState

with open(debug_file, "a") as f:
    f.write("DEBUG: UnifiedAppState imported\n")
    f.flush()

from ui.apps.unified_ocr_app.services.config_loader import load_mode_config, load_unified_config

with open(debug_file, "a") as f:
    f.write("DEBUG: config_loader imported\n")
    f.flush()

# Import all components and services at module level to avoid lazy import issues
from ui.apps.unified_ocr_app.components.comparison import (
    render_metrics_display,
    render_parameter_sweep,
    render_results_comparison,
)
from ui.apps.unified_ocr_app.components.comparison.metrics_display import render_analysis_summary
from ui.apps.unified_ocr_app.components.comparison.results_comparison import render_export_controls
from ui.apps.unified_ocr_app.components.inference import render_checkpoint_selector, render_results_viewer
from ui.apps.unified_ocr_app.components.inference.checkpoint_selector import render_hyperparameters, render_mode_selector
from ui.apps.unified_ocr_app.components.preprocessing import render_parameter_panel, render_stage_viewer
from ui.apps.unified_ocr_app.components.preprocessing.parameter_panel import render_preset_management
from ui.apps.unified_ocr_app.components.shared import render_image_upload
from ui.apps.unified_ocr_app.services.comparison_service import get_comparison_service
from ui.apps.unified_ocr_app.services.inference_service import InferenceService, load_checkpoints
from ui.apps.unified_ocr_app.services.preprocessing_service import PreprocessingService

with open(debug_file, "a") as f:
    f.write("DEBUG: All components and services imported\n")
    f.flush()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

with open(debug_file, "a") as f:
    f.write("DEBUG: Logger initialized\n")
    f.flush()


def main() -> None:
    """Main application entry point."""
    with open(debug_file, "a") as f:
        f.write("\n>>> MAIN FUNCTION CALLED\n")
        f.flush()

    print("=" * 80, file=sys.stderr, flush=True)
    print(">>> UNIFIED OCR APP STARTED - MAIN FUNCTION CALLED", file=sys.stderr, flush=True)
    print("=" * 80, file=sys.stderr, flush=True)
    logger.info("=" * 80)
    logger.info(">>> UNIFIED OCR APP STARTED")
    logger.info("=" * 80)

    # Load main configuration
    with open(debug_file, "a") as f:
        f.write("Loading unified config...\n")
        f.flush()

    print("Loading unified config...", file=sys.stderr, flush=True)
    try:
        config = load_unified_config("unified_app")
        with open(debug_file, "a") as f:
            f.write(f"Config loaded successfully: {config['app']['title']}\n")
            f.flush()
        print(f"Config loaded successfully: {config['app']['title']}", file=sys.stderr, flush=True)
    except Exception as e:
        with open(debug_file, "a") as f:
            f.write(f"ERROR loading config: {e}\n")
            f.flush()
        print(f"ERROR loading config: {e}", file=sys.stderr, flush=True)
        st.error(f"❌ Failed to load configuration: {e}")
        st.info("Please check that configs/ui/unified_app.yaml exists and is valid.")
        st.stop()

    # Set page config
    print("Setting page config...", file=sys.stderr, flush=True)
    layout: Literal["centered", "wide"] = cast(Literal["centered", "wide"], config["app"].get("layout", "wide"))
    sidebar_state: Literal["auto", "expanded", "collapsed"] = cast(
        Literal["auto", "expanded", "collapsed"], config["app"].get("initial_sidebar_state", "expanded")
    )

    st.set_page_config(
        page_title=config["app"]["title"],
        page_icon=config["app"].get("page_icon", "🔍"),
        layout=layout,
        initial_sidebar_state=sidebar_state,
    )
    print("Page config set successfully", file=sys.stderr, flush=True)

    # Display title
    print("Displaying title...", file=sys.stderr, flush=True)
    st.title(config["app"]["title"])
    if "subtitle" in config["app"]:
        st.markdown(config["app"]["subtitle"])
    print("Title displayed", file=sys.stderr, flush=True)

    # Initialize state
    print("Initializing state...", file=sys.stderr, flush=True)
    state = UnifiedAppState.from_session()
    print(f"State initialized, current_mode={state.current_mode}", file=sys.stderr, flush=True)

    # Render sidebar with mode selector
    with st.sidebar:
        st.header("🎯 Mode Selection")

        # Get available modes
        modes = config["app"]["modes"]
        mode_options = {mode["id"]: f"{mode['icon']} {mode['label']}" for mode in modes}

        # Find default mode
        default_mode = next((mode["id"] for mode in modes if mode.get("default", False)), modes[0]["id"])

        # Get current mode from state or use default
        if not state.current_mode or state.current_mode not in mode_options:
            state.current_mode = default_mode
            state.to_session()

        # Radio buttons for mode selection
        selected_label = st.radio(
            "Select Mode",
            options=list(mode_options.values()),
            index=list(mode_options.keys()).index(state.current_mode),
            key="mode_selector",
            help="Choose between preprocessing tuning, model inference, or comparison",
        )

        # Get mode ID from label
        selected_mode = next(mode_id for mode_id, label in mode_options.items() if label == selected_label)

        # Update state if changed
        if selected_mode != state.current_mode:
            state.set_mode(selected_mode)
            state.to_session()
            st.rerun()

    # Load mode-specific configuration
    try:
        mode_config = load_mode_config(state.current_mode, validate=False)  # Disable strict validation for now
    except Exception as e:
        st.error(f"❌ Failed to load {state.current_mode} configuration: {e}")
        st.info(f"Please check that configs/ui/modes/{state.current_mode}.yaml exists and is valid.")
        st.stop()

    # Display mode description
    if "description" in mode_config:
        st.info(f"ℹ️ {mode_config['description']}")

    # Render mode-specific UI
    if state.current_mode == "preprocessing":
        render_preprocessing_mode(state, config, mode_config)
    elif state.current_mode == "inference":
        render_inference_mode(state, config, mode_config)
    elif state.current_mode == "comparison":
        render_comparison_mode(state, config, mode_config)
    else:
        st.error(f"Unknown mode: {state.current_mode}")

    logger.info("=" * 80)
    logger.info("<<< UNIFIED OCR APP COMPLETED")
    logger.info("=" * 80)


def render_preprocessing_mode(state: UnifiedAppState, app_config: dict, mode_config: dict) -> None:
    """Render preprocessing mode UI.

    Args:
        state: Application state
        app_config: Main app configuration
        mode_config: Preprocessing mode configuration
    """
    # Imports are now at module level
    # Sidebar: Image upload and parameters
    with st.sidebar:
        st.divider()

        # Image upload section
        with st.expander("📤 Image Upload", expanded=True):
            render_image_upload(state, app_config.get("shared", {}))

        # Parameter panel
        current_params = render_parameter_panel(state, mode_config)

        # Preset management
        st.divider()
        render_preset_management(current_params, mode_config)

    # Main area: Check if we have an image
    current_image = state.get_current_image()

    if current_image is None:
        st.info("👈 Upload an image from the sidebar to start preprocessing")
        st.markdown(
            """
            ### 🎨 Preprocessing Studio

            This mode allows you to:
            - Tune preprocessing parameters interactively
            - Visualize each stage of the pipeline
            - Compare before/after results
            - Export configurations as presets

            **Get started by uploading an image!**
            """
        )
        return

    # Process button
    col1, col2, col3 = st.columns([2, 1, 2])

    with col2:
        process_button = st.button(
            "🚀 Run Pipeline",
            use_container_width=True,
            type="primary",
            help="Execute preprocessing with current parameters",
        )

    # Process image if button clicked
    processing_results = None

    if process_button:
        with st.spinner("Processing image through pipeline..."):
            # Create service
            service = PreprocessingService(mode_config)

            # Generate cache key based on image and parameters
            import hashlib
            import json

            param_str = json.dumps(current_params, sort_keys=True)
            image_hash = hashlib.md5(current_image.tobytes()).hexdigest()[:8]
            cache_key = f"{image_hash}_{hashlib.md5(param_str.encode()).hexdigest()[:8]}"

            # Process
            result = service.process_image(current_image, current_params, cache_key)

            # Store in state
            state.preprocessing_results = result.get("stages", {})
            state.preprocessing_metadata = result.get("metadata", {})
            state.to_session()

            processing_results = state.preprocessing_results

            # Show success message
            num_stages = len(processing_results) - 1  # Exclude original
            total_time = result.get("metadata", {}).get("total_time", 0)
            st.success(f"✅ Processed {num_stages} stages in {total_time:.2f}s")
    else:
        # Use cached results if available
        processing_results = state.preprocessing_results

    # Render stage viewer
    render_stage_viewer(state, mode_config, processing_results)


def render_inference_mode(state: UnifiedAppState, app_config: dict, mode_config: dict) -> None:
    """Render inference mode UI.

    Args:
        state: Application state
        app_config: Main app configuration
        mode_config: Inference mode configuration
    """
    # Imports are now at module level
    # Load checkpoints
    checkpoints = load_checkpoints(app_config)

    # Sidebar: Controls
    with st.sidebar:
        st.divider()

        # Processing mode selector
        processing_mode = render_mode_selector(mode_config)

        st.divider()

        # Checkpoint selector
        selected_checkpoint = render_checkpoint_selector(state, mode_config, checkpoints)

        if selected_checkpoint is None:
            st.warning("No checkpoint selected. Please train a model first or check checkpoint directory.")
            return

        st.divider()

        # Hyperparameters
        hyperparameters = render_hyperparameters(mode_config)

        st.divider()

        # Preprocessing options
        st.subheader("🎨 Preprocessing Options")
        enable_preprocessing = st.checkbox(
            "Enable Preprocessing",
            value=False,
            key="enable_preprocessing",
            help="Apply preprocessing pipeline before inference",
        )

        if enable_preprocessing:
            st.info("⚠️ Preprocessing integration coming soon")

        st.divider()

        # Image upload (for single mode)
        if processing_mode == "single":
            with st.expander("📤 Image Upload", expanded=True):
                render_image_upload(state, app_config.get("shared", {}))

    # Main area
    if processing_mode == "single":
        _render_single_image_inference(state, mode_config, selected_checkpoint, hyperparameters)
    elif processing_mode == "batch":
        _render_batch_inference(state, mode_config, selected_checkpoint, hyperparameters)
    else:
        st.error(f"Unknown processing mode: {processing_mode}")


def _render_single_image_inference(state: UnifiedAppState, mode_config: dict, checkpoint: Any, hyperparameters: dict) -> None:
    """Render single image inference UI.

    Args:
        state: Application state
        mode_config: Mode configuration
        checkpoint: Selected checkpoint
        hyperparameters: Inference hyperparameters
    """
    # Imports are now at module level
    # Check if we have an image
    current_image = state.get_current_image()

    if current_image is None:
        st.info("👈 Upload an image from the sidebar to start inference")
        st.markdown(
            """
            ### 🤖 Model Inference Studio

            This mode allows you to:
            - Run OCR inference with trained models
            - Visualize detected text regions
            - Export results in multiple formats
            - Batch process multiple images

            **Get started by uploading an image!**
            """
        )
        return

    # Run inference button
    col1, col2, col3 = st.columns([2, 1, 2])

    with col2:
        run_button = st.button(
            "🚀 Run Inference",
            use_container_width=True,
            type="primary",
            help="Execute model inference on the image",
        )

    # Run inference if button clicked
    if run_button:
        try:
            with st.spinner("Running inference..."):
                # Create service
                service = InferenceService(mode_config)

                # Generate cache key
                import hashlib
                import json

                param_str = json.dumps(hyperparameters, sort_keys=True)
                image_hash = hashlib.md5(current_image.tobytes()).hexdigest()[:8]
                checkpoint_hash = str(checkpoint.checkpoint_path) if hasattr(checkpoint, "checkpoint_path") else "unk"
                cache_key = f"{image_hash}_{hashlib.md5((param_str + checkpoint_hash).encode()).hexdigest()[:8]}"

                # Run inference
                result = service.run_inference(current_image, checkpoint, hyperparameters, cache_key)

                # Store in state
                if not hasattr(state, "inference_results"):
                    state.inference_results = []
                state.inference_results.append(result)
                state.to_session()

                # Show success message
                num_regions = len(result.polygons)
                st.success(f"✅ Detected {num_regions} text regions in {result.processing_time:.2f}s")

        except Exception as e:
            st.error(f"❌ Inference failed: {e}")
            logger.error(f"Inference error: {e}", exc_info=True)
            return

    # Render results
    render_results_viewer(state, mode_config)


def _render_batch_inference(state: UnifiedAppState, mode_config: dict, checkpoint: Any, hyperparameters: dict) -> None:
    """Render batch inference UI.

    Args:
        state: Application state
        mode_config: Mode configuration
        checkpoint: Selected checkpoint
        hyperparameters: Inference hyperparameters
    """
    st.subheader("📁 Batch Processing")

    # Input/output directory configuration
    col1, col2 = st.columns(2)

    with col1:
        input_dir = st.text_input(
            "Input Directory",
            value="",
            placeholder="/path/to/images",
            help="Directory containing images to process",
            key="batch_input_dir",
        )

    with col2:
        output_dir = st.text_input(
            "Output Directory",
            value="",
            placeholder="/path/to/output",
            help="Directory for saving results",
            key="batch_output_dir",
        )

    # File type selection
    batch_config = mode_config.get("upload", {}).get("batch_mode", {})
    default_types = batch_config.get("file_types", {}).get("default", ["jpg", "jpeg", "png"])

    file_types = st.multiselect(
        "File Types",
        options=["jpg", "jpeg", "png", "bmp", "tiff"],
        default=default_types,
        help="Select image file types to process",
        key="batch_file_types",
    )

    # Run batch inference button
    if st.button("🚀 Start Batch Processing", type="primary", use_container_width=True):
        if not input_dir or not output_dir:
            st.error("Please specify both input and output directories")
            return

        try:
            from pathlib import Path

            # Validate directories
            input_path = Path(input_dir)
            if not input_path.exists():
                st.error(f"Input directory does not exist: {input_dir}")
                return

            # Get list of images
            image_paths = []
            for ext in file_types:
                image_paths.extend(list(input_path.glob(f"*.{ext}")))

            if not image_paths:
                st.warning(f"No images found in {input_dir} with extensions: {', '.join(file_types)}")
                return

            st.info(f"Found {len(image_paths)} images to process")

            # Run batch inference
            with st.spinner(f"Processing {len(image_paths)} images..."):
                service = InferenceService(mode_config)
                result = service.run_batch_inference([str(p) for p in image_paths], checkpoint, hyperparameters, output_dir)

                if result.get("status") == "completed":
                    st.success(f"✅ Batch processing completed! Results saved to {output_dir}")
                else:
                    st.error(f"❌ Batch processing failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            st.error(f"❌ Batch processing error: {e}")
            logger.error(f"Batch processing error: {e}", exc_info=True)


def render_comparison_mode(state: UnifiedAppState, app_config: dict, mode_config: dict) -> None:
    """Render comparison mode UI.

    Args:
        state: Application state
        app_config: Main app configuration
        mode_config: Comparison mode configuration
    """
    # Imports are now at module level
    # Sidebar: Configuration and controls
    with st.sidebar:
        st.divider()

        # Image upload
        with st.expander("📤 Image Upload", expanded=True):
            render_image_upload(state, app_config.get("shared", {}))

        st.divider()

        # Parameter sweep configuration
        sweep_config = render_parameter_sweep(mode_config, state_key="comparison_sweep")

    # Main area: Check if we have an image
    current_image = state.get_current_image()

    if current_image is None:
        st.info("👈 Upload an image from the sidebar to start comparison")
        st.markdown(
            """
            ### 📊 Comparison Studio

            This mode allows you to:
            - Compare different preprocessing configurations
            - Test multiple inference hyperparameters
            - Run A/B tests on full pipeline
            - Analyze performance metrics
            - Identify optimal parameters

            **Get started by:**
            1. Upload an image
            2. Configure parameter sweep in the sidebar
            3. Run comparison to analyze results

            **Quick Start Presets:**
            - Preprocessing Quality Comparison
            - Detection Threshold Sweep
            - Model Checkpoint Comparison
            """
        )
        return

    # Check if we have configurations to compare
    configurations = sweep_config.get("configurations", [])

    if not configurations:
        st.warning("⚠️ No configurations defined. Please add at least 2 configurations in the sidebar.")
        st.info("Use the parameter sweep panel in the sidebar to define configurations to compare.")
        return

    # Show summary
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Image Loaded", "✓")
    with col2:
        st.metric("Configurations", len(configurations))
    with col3:
        comparison_type = sweep_config.get("comparison_type", "preprocessing")
        st.metric("Comparison Type", comparison_type.replace("_", " ").title())

    st.divider()

    # Run comparison button
    col_a, col_b, col_c = st.columns([2, 1, 2])

    with col_b:
        run_button = st.button(
            "🚀 Run Comparison",
            use_container_width=True,
            type="primary",
            help=f"Compare {len(configurations)} configurations",
        )

    # Run comparison if button clicked
    comparison_results = None

    if run_button:
        try:
            with st.spinner(f"Running comparison across {len(configurations)} configurations..."):
                # Get comparison service
                service = get_comparison_service()

                # Run comparison based on type
                if comparison_type == "preprocessing":
                    comparison_results = service.run_preprocessing_comparison(
                        current_image,
                        configurations,
                    )
                elif comparison_type == "inference":
                    comparison_results = service.run_inference_comparison(
                        current_image,
                        configurations,
                    )
                elif comparison_type == "end_to_end":
                    comparison_results = service.run_end_to_end_comparison(
                        current_image,
                        configurations,
                    )
                else:
                    st.error(f"Unknown comparison type: {comparison_type}")
                    return

                # Store in state
                state.comparison_results = comparison_results
                state.to_session()

                # Show success message
                num_successful = sum(1 for r in comparison_results if "error" not in r.get("metrics", {}))
                st.success(f"✅ Comparison completed! {num_successful}/{len(configurations)} configurations ran successfully.")

        except Exception as e:
            st.error(f"❌ Comparison failed: {e}")
            logger.error(f"Comparison error: {e}", exc_info=True)
            return

    else:
        # Use cached results if available
        comparison_results = getattr(state, "comparison_results", None)

    # Display results if available
    if comparison_results:
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(
            [
                "📊 Results Comparison",
                "📈 Metrics Analysis",
                "🎯 Parameter Impact",
            ]
        )

        with tab1:
            st.subheader("Results Comparison")
            render_results_comparison(comparison_results, mode_config)

            st.divider()

            # Export controls
            render_export_controls(comparison_results, mode_config)

        with tab2:
            st.subheader("Metrics Analysis")
            render_metrics_display(comparison_results, mode_config)

            st.divider()

            # Analysis summary
            render_analysis_summary(comparison_results, mode_config)

        with tab3:
            st.subheader("Parameter Impact Analysis")
            st.info("📊 Parameter impact visualization coming soon!")

            # Show parameter summary table
            st.markdown("**Configuration Parameters:**")

            param_summary = []
            for result in comparison_results:
                param_summary.append(
                    {
                        "Configuration": result.get("config_label", "Unnamed"),
                        "Parameters": str(result.get("config_params", {}))[:100] + "...",
                        "Processing Time": f"{result.get('processing_time', 0):.3f}s",
                    }
                )

            st.dataframe(param_summary, use_container_width=True)
    else:
        st.info("💡 Click 'Run Comparison' to see results here.")


if __name__ == "__main__":
    main()
