"""Minimal results renderer for debugging freeze issues.

This is a stripped-down version of results.py that only displays text,
no complex widgets. Use this to test if the freeze is caused by
st.dataframe() or st.expander() rendering.

To use: In app.py, change:
    from .components import results as results_component
to:
    from .components import results_minimal as results_component
"""

from __future__ import annotations

import sys

import streamlit as st

from ..models.ui_config import UIConfig
from ..state import InferenceState


def render_results(state: InferenceState, config: UIConfig) -> None:
    """Minimal results display - text only, no widgets."""
    print(">>> MINIMAL render_results CALLED", file=sys.stderr, flush=True)

    st.header("📊 Inference Results (Minimal Debug Mode)")

    if not state.inference_results:
        st.info("No results yet. Run inference to see predictions.")
        print("    No results to display", file=sys.stderr, flush=True)
        return

    print(f"    Displaying {len(state.inference_results)} results", file=sys.stderr, flush=True)

    # Simple text display - no dataframe, no expander, no images
    st.markdown(f"**Total results:** {len(state.inference_results)}")

    # Show summary statistics
    total_predictions = sum(r.num_predictions for r in state.inference_results)
    avg_confidence = sum(r.avg_confidence for r in state.inference_results) / len(state.inference_results)

    st.markdown(f"""
    - **Total predictions:** {total_predictions}
    - **Average confidence:** {avg_confidence:.1f}%
    - **Results in session:** {len(state.inference_results)}
    """)

    print("    Summary stats displayed", file=sys.stderr, flush=True)

    # Show last 5 results as simple text
    st.markdown("### Last 5 Results")

    for idx, result in enumerate(state.inference_results[-5:], 1):
        print(f"    Rendering result {idx}", file=sys.stderr, flush=True)

        st.markdown(f"""
        **Result {idx}**
        - Predictions: {result.num_predictions}
        - Confidence: {result.avg_confidence:.1f}%
        - Checkpoint: {result.checkpoint_name if hasattr(result, "checkpoint_name") else "N/A"}
        """)

    print("<<< MINIMAL render_results COMPLETED", file=sys.stderr, flush=True)


def _render_results_table(*args, **kwargs):
    """Placeholder - not used in minimal version."""
    pass


def _render_single_result(*args, **kwargs):
    """Placeholder - not used in minimal version."""
    pass
