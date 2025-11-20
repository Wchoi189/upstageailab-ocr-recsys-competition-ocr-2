"""Shared utilities for all pages in the Unified OCR App.

This module provides common functionality used across multiple pages:
- Configuration loading and caching
- App state management
- Page setup utilities
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Literal

import streamlit as st

# Import PROJECT_ROOT from central path utility (stable, works from any location)
try:
    from ocr.utils.path_utils import PROJECT_ROOT
    project_root = PROJECT_ROOT
except ImportError:
    # Fallback: add project root to path first, then import
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from ocr.utils.path_utils import PROJECT_ROOT
    project_root = PROJECT_ROOT

# Ensure project root is in sys.path for imports
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ui.apps.unified_ocr_app.models.app_state import UnifiedAppState
from ui.apps.unified_ocr_app.services.config_loader import load_unified_config

logger = logging.getLogger(__name__)


@st.cache_resource
def get_app_config() -> dict[str, Any]:
    """Get cached app configuration.

    Returns:
        App configuration dictionary from unified_app.yaml
    """
    try:
        config = load_unified_config("unified_app")
        logger.info(f"Loaded app config: {config['app']['title']}")
        return config
    except Exception as e:
        logger.error(f"Failed to load app config: {e}")
        st.error(f"❌ Failed to load configuration: {e}")
        st.info("Please check that configs/ui/unified_app.yaml exists and is valid.")
        st.stop()
        return {}  # Never reached, but makes type checker happy


def get_app_state() -> UnifiedAppState:
    """Get or create app state from session.

    Returns:
        UnifiedAppState instance
    """
    return UnifiedAppState.from_session()


def setup_page(
    title: str,
    icon: str,
    layout: Literal["centered", "wide"] = "wide",
    sidebar_state: Literal["auto", "expanded", "collapsed"] = "expanded",
) -> None:
    """Setup common page configuration.

    Args:
        title: Page title
        icon: Page icon (emoji)
        layout: Page layout ("centered" or "wide")
        sidebar_state: Initial sidebar state
    """
    st.set_page_config(
        page_title=f"{title} - OCR Studio",
        page_icon=icon,
        layout=layout,
        initial_sidebar_state=sidebar_state,
    )
