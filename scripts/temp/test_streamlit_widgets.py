#!/usr/bin/env python3
"""Test Streamlit widgets in isolation to identify which causes the freeze.

Run with: uv run streamlit run test_streamlit_widgets.py

This creates a simple app that tests each suspect widget independently.
"""

from __future__ import annotations

import sys

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Widget Freeze Test", layout="wide")

st.title("🔬 Streamlit Widget Freeze Test")

st.markdown("""
This app tests each widget that might be causing the freeze.
Try each test and see which one causes the freeze.
""")

# Test 1: st.dataframe with realistic data
st.header("Test 1: st.dataframe()")

test1_button = st.button("Run Test 1: st.dataframe with 10 rows", key="test1")

if test1_button:
    print(">>> TEST 1: Creating dataframe...", file=sys.stderr, flush=True)

    # Create a dataframe similar to what results.py creates
    df = pd.DataFrame(
        {
            "Result #": list(range(1, 11)),
            "Predictions": [85, 92, 78, 103, 67, 88, 91, 76, 82, 95],
            "Avg Confidence": [98.5, 97.2, 99.1, 96.8, 98.9, 97.5, 98.2, 99.3, 97.8, 98.1],
            "Checkpoint": ["best_model.ckpt"] * 10,
        }
    )

    print("    Calling st.dataframe()...", file=sys.stderr, flush=True)

    try:
        st.dataframe(df, use_container_width=True)
        print("    st.dataframe() completed successfully!", file=sys.stderr, flush=True)
        st.success("✓ Test 1 passed - st.dataframe() works!")
    except Exception as e:
        print(f"    ERROR in st.dataframe(): {e}", file=sys.stderr, flush=True)
        st.error(f"✗ Test 1 failed: {e}")

st.divider()

# Test 2: Multiple st.expander() widgets
st.header("Test 2: Multiple st.expander()")

test2_button = st.button("Run Test 2: Create 10 expanders", key="test2")

if test2_button:
    print(">>> TEST 2: Creating expanders...", file=sys.stderr, flush=True)

    try:
        for i in range(1, 11):
            print(f"    Creating expander {i}...", file=sys.stderr, flush=True)

            with st.expander(f"Result {i}: 85 predictions | Confidence: 98.5%"):
                st.write(f"This is expander {i}")
                st.write("Some prediction data here")

        print("    All expanders created successfully!", file=sys.stderr, flush=True)
        st.success("✓ Test 2 passed - st.expander() works!")
    except Exception as e:
        print(f"    ERROR in st.expander(): {e}", file=sys.stderr, flush=True)
        st.error(f"✗ Test 2 failed: {e}")

st.divider()

# Test 3: st.image() with dummy data
st.header("Test 3: st.image()")

test3_button = st.button("Run Test 3: Display multiple images", key="test3")

if test3_button:
    print(">>> TEST 3: Creating images...", file=sys.stderr, flush=True)

    try:
        import numpy as np

        # Create a dummy image
        dummy_image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)

        for i in range(1, 6):
            print(f"    Displaying image {i}...", file=sys.stderr, flush=True)

            st.image(
                dummy_image,
                caption=f"Test Image {i}",
                use_column_width=True,
                clamp=True,
                channels="RGB",
            )

        print("    All images displayed successfully!", file=sys.stderr, flush=True)
        st.success("✓ Test 3 passed - st.image() works!")
    except Exception as e:
        print(f"    ERROR in st.image(): {e}", file=sys.stderr, flush=True)
        st.error(f"✗ Test 3 failed: {e}")

st.divider()

# Test 4: Combined test (most realistic)
st.header("Test 4: Combined (dataframe + expanders)")

test4_button = st.button("Run Test 4: Dataframe + Expanders", key="test4")

if test4_button:
    print(">>> TEST 4: Combined test...", file=sys.stderr, flush=True)

    try:
        # First the dataframe
        print("    Creating dataframe...", file=sys.stderr, flush=True)
        df = pd.DataFrame(
            {
                "Result #": list(range(1, 6)),
                "Predictions": [85, 92, 78, 103, 67],
                "Avg Confidence": [98.5, 97.2, 99.1, 96.8, 98.9],
            }
        )
        st.dataframe(df, use_container_width=True)
        print("    Dataframe created", file=sys.stderr, flush=True)

        st.divider()

        # Then the expanders
        print("    Creating expanders...", file=sys.stderr, flush=True)
        for i in range(1, 6):
            print(f"    Creating expander {i}...", file=sys.stderr, flush=True)
            with st.expander(f"Result {i}"):
                st.write(f"Details for result {i}")

        print("    Combined test completed successfully!", file=sys.stderr, flush=True)
        st.success("✓ Test 4 passed - Combined widgets work!")
    except Exception as e:
        print(f"    ERROR in combined test: {e}", file=sys.stderr, flush=True)
        st.error(f"✗ Test 4 failed: {e}")

st.divider()

# Instructions
st.header("📋 Instructions")

st.markdown("""
1. Run each test by clicking its button
2. Watch the terminal for print statements
3. If a test causes freeze:
   - Note which test number
   - Check the last print statement in terminal
   - That tells us which widget is the problem
4. If all tests pass, the issue is NOT the widgets themselves
   - Problem might be the data being passed to them
   - Or interaction with session state

**Expected behavior:**
- Each test should complete and show a green success message
- If it freezes, the terminal will show the last operation before freeze
""")

st.markdown("---")
st.markdown("**Monitoring:** Run this in another terminal: `tail -f /tmp/streamlit_debug.log`")
