"""Minimal test to isolate the hanging issue."""

import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import streamlit as st

# Write debug info
debug_log = Path("/tmp/test_app_debug.log")
with open(debug_log, "w") as f:
    f.write("Step 1: Streamlit imported\n")

st.set_page_config(page_title="Test", layout="wide")

with open(debug_log, "a") as f:
    f.write("Step 2: Page config set\n")

st.title("Test App")

with open(debug_log, "a") as f:
    f.write("Step 3: Title displayed\n")

# Try importing UnifiedAppState
try:
    with open(debug_log, "a") as f:
        f.write("Step 4: About to import UnifiedAppState\n")

    from ui.apps.unified_ocr_app.models.app_state import UnifiedAppState

    with open(debug_log, "a") as f:
        f.write("Step 5: UnifiedAppState imported successfully\n")

    state = UnifiedAppState.from_session()

    with open(debug_log, "a") as f:
        f.write(f"Step 6: State initialized: {state}\n")

    st.success("✅ All imports and state initialization successful!")

except Exception as e:
    with open(debug_log, "a") as f:
        f.write(f"ERROR: {e}\n")
        import traceback

        f.write(traceback.format_exc())

    st.error(f"Error: {e}")

with open(debug_log, "a") as f:
    f.write("Step 7: Script completed\n")

st.write("App loaded successfully!")
