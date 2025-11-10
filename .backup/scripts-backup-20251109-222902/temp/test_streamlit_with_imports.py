"""Test Streamlit with unified app imports."""

import logging
import sys
from pathlib import Path

import streamlit as st

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

logger.info("=" * 80)
logger.info(">>> TEST APP STARTING")
logger.info("=" * 80)

logger.info("Importing UnifiedAppState...")
from ui.apps.unified_ocr_app.models.app_state import UnifiedAppState

logger.info("Importing config_loader...")
from ui.apps.unified_ocr_app.services.config_loader import load_unified_config

logger.info("Loading unified config...")
config = load_unified_config("unified_app")
logger.info(f"Config loaded: {config['app']['title']}")

logger.info("Setting page config...")
st.set_page_config(
    page_title=config["app"]["title"],
    page_icon=config["app"].get("page_icon", "🔍"),
    layout="wide",
    initial_sidebar_state="expanded",
)

logger.info("Displaying title...")
st.title(config["app"]["title"])

logger.info("Initializing state...")
state = UnifiedAppState.from_session()

logger.info("Creating sidebar...")
with st.sidebar:
    st.header("🎯 Mode Selection")
    st.write("Test sidebar content")

logger.info("App loaded successfully!")
st.success("✅ If you see this, the app loaded successfully!")

logger.info("=" * 80)
logger.info("<<< TEST APP COMPLETED")
logger.info("=" * 80)
