"""Preprocessing Studio - Interactive parameter tuning and pipeline visualization.

This page allows users to:
- Upload images and tune preprocessing parameters
- Visualize each stage of the pipeline
- Compare before/after results
- Export configurations as presets
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

# Import PROJECT_ROOT from central path utility (stable, works from any location)
from ocr.utils.path_utils import PROJECT_ROOT

try:
    project_root = PROJECT_ROOT
except ImportError:
    # Fallback: add project root to path first, then import
    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    project_root = PROJECT_ROOT

# Ensure project root is in sys.path for imports
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
from ui.apps.unified_ocr_app.components.preprocessing import render_parameter_panel, render_stage_viewer
from ui.apps.unified_ocr_app.components.preprocessing.parameter_panel import render_preset_management
from ui.apps.unified_ocr_app.components.shared import render_image_upload
from ui.apps.unified_ocr_app.services import get_preprocessing_service

# Import only what THIS PAGE needs (lazy loading!)
from ui.apps.unified_ocr_app.services.config_loader import load_mode_config

# Import shared utilities
from ui.apps.unified_ocr_app.shared_utils import get_app_config, get_app_state, setup_page

# Setup logging
logger = logging.getLogger(__name__)

# Setup page
setup_page("Preprocessing", "🎨")

# Get shared state and config
state = get_app_state()
app_config = get_app_config()

# Load mode-specific configuration
try:
    mode_config = load_mode_config("preprocessing", validate=False)
except Exception as e:
    st.error(f"❌ Failed to load preprocessing configuration: {e}")
    st.info("Please check that configs/ui/modes/preprocessing.yaml exists and is valid.")
    st.stop()

# Page title and description
st.title("🎨 Preprocessing Studio")
if "description" in mode_config:
    st.info(f"ℹ️ {mode_config['description']}")

# === SIDEBAR ===
with st.sidebar:
    st.header("Preprocessing Controls")

    # Image upload
    with st.expander("📤 Image Upload", expanded=True):
        render_image_upload(state, app_config.get("shared", {}))

    # Parameter panel
    current_params = render_parameter_panel(state, mode_config)

    # Preset management
    st.divider()
    render_preset_management(current_params, mode_config)

# === MAIN AREA ===
current_image = state.get_current_image()

if current_image is None:
    st.info("👈 Upload an image from the sidebar to start preprocessing")
    st.markdown("""
        ### 🎨 Preprocessing Studio

        This mode allows you to:
        - Tune preprocessing parameters interactively
        - Visualize each stage of the pipeline
        - Compare before/after results
        - Export configurations as presets

        **Get started by uploading an image!**
    """)
    st.stop()

# Process button
col1, col2, col3 = st.columns([2, 1, 2])

with col2:
    process_button = st.button(
        "🚀 Run Pipeline",
        use_container_width=True,
        type="primary",
        help="Execute preprocessing with current parameters",
    )

# Initialize processing results
processing_results = None

# Process image if button clicked
if process_button:
    with st.spinner("Processing image through pipeline..."):
        # Get cached service for better performance
        service = get_preprocessing_service(mode_config)

        # Generate cache key based on image and parameters
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
