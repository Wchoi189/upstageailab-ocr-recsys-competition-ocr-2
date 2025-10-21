"""Minimal Streamlit test to verify basic functionality."""

import sys
from pathlib import Path

import streamlit as st

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("=" * 80, file=sys.stderr)
print("STREAMLIT APP STARTING", file=sys.stderr)
print("=" * 80, file=sys.stderr)

st.title("Minimal Test App")
st.write("If you see this, Streamlit is working!")

print("STREAMLIT APP LOADED SUCCESSFULLY", file=sys.stderr)
