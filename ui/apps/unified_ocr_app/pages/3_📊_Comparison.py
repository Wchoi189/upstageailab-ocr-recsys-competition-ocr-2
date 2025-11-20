"""Comparison Studio - A/B test configurations and models.

This page allows users to:
- Compare different preprocessing configurations
- Test multiple inference hyperparameters
- Run end-to-end pipeline comparisons
- Analyze performance metrics
"""

from __future__ import annotations

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

from ui.apps.unified_ocr_app.components.comparison import (
    render_metrics_display,
    render_parameter_sweep,
    render_results_comparison,
)
from ui.apps.unified_ocr_app.components.comparison.metrics_display import render_analysis_summary
from ui.apps.unified_ocr_app.components.comparison.results_comparison import render_export_controls
from ui.apps.unified_ocr_app.components.shared import render_image_upload
from ui.apps.unified_ocr_app.services.comparison_service import get_comparison_service

# Import only what THIS PAGE needs (lazy loading!)
from ui.apps.unified_ocr_app.services.config_loader import load_mode_config

# Import shared utilities
from ui.apps.unified_ocr_app.shared_utils import get_app_config, get_app_state, setup_page

# Setup logging
logger = logging.getLogger(__name__)

# Setup page
setup_page("Comparison", "📊")

# Get shared state and config
state = get_app_state()
app_config = get_app_config()

# Load mode-specific configuration
try:
    mode_config = load_mode_config("comparison", validate=False)
except Exception as e:
    st.error(f"❌ Failed to load comparison configuration: {e}")
    st.info("Please check that configs/ui/modes/comparison.yaml exists and is valid.")
    st.stop()

# Page title and description
st.title("📊 Comparison Studio")
if "description" in mode_config:
    st.info(f"ℹ️ {mode_config['description']}")

# === SIDEBAR ===
with st.sidebar:
    st.header("Comparison Controls")
    st.divider()

    # Image upload
    with st.expander("📤 Image Upload", expanded=True):
        render_image_upload(state, app_config.get("shared", {}))

    st.divider()

    # Parameter sweep configuration
    sweep_config = render_parameter_sweep(mode_config, state_key="comparison_sweep")

# === MAIN AREA ===
current_image = state.get_current_image()

if current_image is None:
    st.info("👈 Upload an image from the sidebar to start comparison")
    st.markdown("""
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
    """)
    st.stop()

# Check if we have configurations to compare
configurations = sweep_config.get("configurations", [])

if not configurations:
    st.warning("⚠️ No configurations defined. Please add at least 2 configurations in the sidebar.")
    st.info("Use the parameter sweep panel in the sidebar to define configurations to compare.")
    st.stop()

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

# Initialize comparison results
comparison_results = None

# Run comparison if button clicked
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
                st.stop()

            # Store in state
            state.comparison_results = comparison_results
            state.to_session()

            # Show success message
            num_successful = sum(1 for r in comparison_results if "error" not in r.get("metrics", {}))
            st.success(f"✅ Comparison completed! {num_successful}/{len(configurations)} configurations ran successfully.")

    except Exception as e:
        st.error(f"❌ Comparison failed: {e}")
        logger.error(f"Comparison error: {e}", exc_info=True)
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
