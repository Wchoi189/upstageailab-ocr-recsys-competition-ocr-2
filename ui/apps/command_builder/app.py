from __future__ import annotations

import streamlit as st

from ui.apps.command_builder.components import render_predict_page, render_sidebar, render_test_page, render_training_page
from ui.apps.command_builder.state import CommandBuilderState, CommandType
from ui.apps.command_builder.utils import (
    get_command_builder,
    get_config_parser,
    get_recommendation_service,
)


def run() -> None:
    st.set_page_config(page_title="OCR Command Builder", page_icon="🔍", layout="wide")
    st.title("🔍 OCR Training Command Builder")
    st.caption("Build and execute training, testing, and prediction commands with metadata-aware defaults.")

    state = CommandBuilderState.from_session()

    # Use cached services for better performance
    command_builder = get_command_builder()
    config_parser = get_config_parser()
    recommendation_service = get_recommendation_service()

    command_type = render_sidebar(state)

    if command_type == CommandType.TRAIN:
        render_training_page(state, command_builder, recommendation_service, config_parser)
    elif command_type == CommandType.TEST:
        render_test_page(state, command_builder)
    else:
        render_predict_page(state, command_builder)

    state.persist()


def main() -> None:
    run()


if __name__ == "__main__":
    main()
