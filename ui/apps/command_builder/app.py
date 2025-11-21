from __future__ import annotations

import os

import streamlit as st

from ui.apps.command_builder.components import render_predict_page, render_sidebar, render_test_page, render_training_page
from ui.apps.command_builder.state import CommandBuilderState, CommandType

PLAYGROUND_BETA_URL = os.environ.get("PLAYGROUND_BETA_URL", "").strip()


def run() -> None:
    st.set_page_config(page_title="OCR Command Builder", page_icon="🔍", layout="wide")
    st.title("🔍 OCR Training Command Builder")
    st.caption("Build and execute training, testing, and prediction commands with metadata-aware defaults.")

    if PLAYGROUND_BETA_URL:
        st.info(
            f"🚀 A faster playground is available. [Open beta SPA]({PLAYGROUND_BETA_URL}) for Albumentations-style previews.",
            icon="✨",
        )

    state = CommandBuilderState.from_session()

    # Render sidebar FIRST (fast, no heavy operations)
    command_type = render_sidebar(state)

    # Lazy load services only for the active page (Phase 2 optimization)
    # This reduces initialization overhead for unused pages
    if command_type == CommandType.TRAIN:
        from ui.apps.command_builder.utils import (
            get_command_builder,
            get_config_parser,
            get_recommendation_service,
        )

        command_builder = get_command_builder()
        config_parser = get_config_parser()
        recommendation_service = get_recommendation_service()
        render_training_page(state, command_builder, recommendation_service, config_parser)
    elif command_type == CommandType.TEST:
        from ui.apps.command_builder.utils import get_command_builder

        command_builder = get_command_builder()
        render_test_page(state, command_builder)
    else:  # PREDICT
        from ui.apps.command_builder.utils import get_command_builder

        command_builder = get_command_builder()
        render_predict_page(state, command_builder)

    state.persist()


def main() -> None:
    run()


if __name__ == "__main__":
    main()
