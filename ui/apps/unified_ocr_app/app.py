"""Unified OCR Development Studio - Home Page.

Welcome screen and configuration for the multi-page app.
Select a mode from the sidebar to begin.

Architecture:
    - Multi-page Streamlit app with automatic page discovery
    - Each mode is a separate page for better performance and maintainability
    - Shared state and configuration across all pages
    - Lazy loading of heavy resources per-page

Usage:
    streamlit run ui/apps/unified_ocr_app/app.py

Pages:
    - 🎨 Preprocessing: Interactive parameter tuning and pipeline visualization
    - 🤖 Inference: Run OCR models on images
    - 📊 Comparison: A/B test configurations and analyze metrics
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Literal, cast

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st

from ui.apps.unified_ocr_app.shared_utils import get_app_config, get_app_state

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)
PLAYGROUND_BETA_URL = os.environ.get("PLAYGROUND_BETA_URL", "").strip()


def main() -> None:
    """Main application entry point."""
    logger.info("=== UNIFIED OCR APP STARTED ===")

    # Load configuration (cached globally)
    config = get_app_config()

    # Setup page configuration
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

    # Initialize state (shared across all pages)
    state = get_app_state()

    # === HOME PAGE UI ===
    st.title(config["app"]["title"])

    if "subtitle" in config["app"]:
        st.markdown(config["app"]["subtitle"])

    st.divider()

    st.info("👈 Select a mode from the sidebar to begin")
    if PLAYGROUND_BETA_URL:
        st.success(
            f"✨ Preview the upcoming playground experience [here]({PLAYGROUND_BETA_URL}). "
            "This beta delivers Albumentations-style previews and faster rembg routing.",
        )

    # Mode descriptions
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            ### 🎨 Preprocessing
            Tune image processing pipelines interactively.

            - 7-stage pipeline visualization
            - Real-time parameter adjustment
            - Preset management
            - Before/after comparison
        """)

    with col2:
        st.markdown("""
            ### 🤖 Inference
            Run OCR models on images.

            - Single or batch processing
            - Multiple model checkpoints
            - Adjustable hyperparameters
            - Export results (JSON, CSV)
        """)

    with col3:
        st.markdown("""
            ### 📊 Comparison
            A/B test configurations and models.

            - Parameter sweep
            - Side-by-side comparison
            - Performance metrics
            - Statistical analysis
        """)

    st.divider()

    # Quick stats
    st.markdown("### 📊 Quick Stats")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.metric("Images Processed (Session)", state.get_image_count(), help="Images uploaded this session")

    with col_b:
        current_mode = state.current_mode if hasattr(state, "current_mode") else "None"
        st.metric("Current Mode", current_mode or "None", help="Last selected mode")

    with col_c:
        preprocessing_count = 0
        if hasattr(state, "preprocessing_results") and state.preprocessing_results:
            preprocessing_count = len(state.preprocessing_results)
        st.metric("Pipeline Runs", preprocessing_count, help="Preprocessing pipeline executions")

    # Footer
    st.divider()
    st.markdown(
        """
        <sub>
        💡 **Tip**: Use the sidebar to navigate between modes. Your session state is preserved across pages.
        </sub>
    """,
        unsafe_allow_html=True,
    )

    logger.info("=== HOME PAGE RENDERED ===")


if __name__ == "__main__":
    main()
