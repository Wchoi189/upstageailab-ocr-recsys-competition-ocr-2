"""Inference Studio - Run OCR models on images.

This page allows users to:
- Select trained model checkpoints
- Run single image or batch inference
- Visualize detected text regions
- Export results in multiple formats
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

# Import PROJECT_ROOT from central path utility (stable, works from any location)
try:
    from ocr.utils.path_utils import PROJECT_ROOT
    project_root = PROJECT_ROOT
except ImportError:
    # Fallback: add project root to path first, then import
    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from ocr.utils.path_utils import PROJECT_ROOT
    project_root = PROJECT_ROOT

# Ensure project root is in sys.path for imports
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st

from ui.apps.unified_ocr_app.components.inference import render_checkpoint_selector, render_results_viewer
from ui.apps.unified_ocr_app.components.inference.checkpoint_selector import render_hyperparameters, render_mode_selector
from ui.apps.unified_ocr_app.components.shared import render_image_upload

# Import only what THIS PAGE needs (lazy loading!)
from ui.apps.unified_ocr_app.services.config_loader import load_mode_config
from ui.apps.unified_ocr_app.services import get_inference_service
from ui.apps.unified_ocr_app.services.inference_service import load_checkpoints

# Import shared utilities
from ui.apps.unified_ocr_app.shared_utils import get_app_config, get_app_state, setup_page

# Setup logging
logger = logging.getLogger(__name__)

# Setup page
setup_page("Inference", "🤖")

# Get shared state and config
state = get_app_state()
app_config = get_app_config()

# Load mode-specific configuration
try:
    mode_config = load_mode_config("inference", validate=False)
except Exception as e:
    st.error(f"❌ Failed to load inference configuration: {e}")
    st.info("Please check that configs/ui/modes/inference.yaml exists and is valid.")
    st.stop()

# Page title and description
st.title("🤖 Inference Studio")
if "description" in mode_config:
    st.info(f"ℹ️ {mode_config['description']}")

# Load checkpoints (cached)
checkpoints = load_checkpoints(app_config)

# === SIDEBAR ===
with st.sidebar:
    st.header("Inference Controls")
    st.divider()

    # Processing mode selector
    processing_mode = render_mode_selector(mode_config)

    st.divider()

    # Checkpoint selector
    selected_checkpoint = render_checkpoint_selector(state, mode_config, checkpoints)

    if selected_checkpoint is None:
        st.warning("No checkpoint selected. Please train a model first or check checkpoint directory.")
        st.stop()

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

# === MAIN AREA ===
if processing_mode == "single":
    # Single Image Inference
    current_image = state.get_current_image()

    if current_image is None:
        st.info("👈 Upload an image from the sidebar to start inference")
        st.markdown("""
            ### 🤖 Model Inference Studio

            This mode allows you to:
            - Run OCR inference with trained models
            - Visualize detected text regions
            - Export results in multiple formats
            - Batch process multiple images

            **Get started by uploading an image!**
        """)
        st.stop()

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
                # Get cached service for better performance
                service = get_inference_service(mode_config)

                # Generate cache key
                param_str = json.dumps(hyperparameters, sort_keys=True)
                image_hash = hashlib.md5(current_image.tobytes()).hexdigest()[:8]
                checkpoint_hash = str(selected_checkpoint.checkpoint_path) if hasattr(selected_checkpoint, "checkpoint_path") else "unk"
                cache_key = f"{image_hash}_{hashlib.md5((param_str + checkpoint_hash).encode()).hexdigest()[:8]}"

                # Run inference
                result = service.run_inference(current_image, selected_checkpoint, hyperparameters, cache_key)

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

    # Render results
    render_results_viewer(state, mode_config)

elif processing_mode == "batch":
    # Batch Inference
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
        else:
            try:
                # Validate directories
                input_path = Path(input_dir)
                if not input_path.exists():
                    st.error(f"Input directory does not exist: {input_dir}")
                else:
                    # Get list of images
                    image_paths = []
                    for ext in file_types:
                        image_paths.extend(list(input_path.glob(f"*.{ext}")))

                    if not image_paths:
                        st.warning(f"No images found in {input_dir} with extensions: {', '.join(file_types)}")
                    else:
                        st.info(f"Found {len(image_paths)} images to process")

                        # Run batch inference
                        with st.spinner(f"Processing {len(image_paths)} images..."):
                            # Get cached service for better performance
                            service = get_inference_service(mode_config)
                            result = service.run_batch_inference(
                                [str(p) for p in image_paths], selected_checkpoint, hyperparameters, output_dir
                            )

                            if result.get("status") == "completed":
                                st.success(f"✅ Batch processing completed! Results saved to {output_dir}")
                            else:
                                st.error(f"❌ Batch processing failed: {result.get('error', 'Unknown error')}")

            except Exception as e:
                st.error(f"❌ Batch processing error: {e}")
                logger.error(f"Batch processing error: {e}", exc_info=True)

else:
    st.error(f"Unknown processing mode: {processing_mode}")
