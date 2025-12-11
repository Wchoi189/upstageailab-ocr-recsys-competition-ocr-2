from __future__ import annotations

"""Streamlit app orchestration layer.

Refer to ``docs/ai_handbook/02_protocols/11_streamlit_maintenance_protocol.md``
and ``docs/ai_handbook/02_protocols/12_streamlit_refactoring_protocol.md``
before changing layout or control flow. Configuration lives in
``configs/ui/inference.yaml`` with additional metadata in ``ui_meta/`` and
schemas in ``docs/schemas/``—keep those sources authoritative and avoid
guessing widget behaviour.
"""

from typing import Literal, cast

import streamlit as st

from .components import results as results_component
from .components import sidebar as sidebar_component
from .models.batch_request import BatchPredictionRequest
from .models.checkpoint import CheckpointInfo
from .models.ui_events import InferenceRequest
from .services.checkpoint import CatalogOptions, build_lightweight_catalog
from .services.config_loader import load_ui_config
from .services.inference_runner import InferenceService
from .state import InferenceState


@st.cache_data(show_spinner=False)
def _load_catalog(options: CatalogOptions) -> list[CheckpointInfo]:
    return build_lightweight_catalog(options)


def run() -> None:
    import logging
    import sys

    LOGGER = logging.getLogger(__name__)

    # Use print for immediate output (bypasses logging system)
    print("\n" + "=" * 80, file=sys.stderr, flush=True)
    print(">>> APP.RUN() STARTED", file=sys.stderr, flush=True)
    print("=" * 80 + "\n", file=sys.stderr, flush=True)

    LOGGER.info("=" * 80)
    LOGGER.info(">>> APP.RUN() STARTED")
    LOGGER.info("=" * 80)

    LOGGER.info("Loading UI config")
    config = load_ui_config()
    layout: Literal["centered", "wide"] = cast(
        Literal["centered", "wide"],
        config.app.layout if config.app.layout in {"centered", "wide"} else "wide",
    )
    sidebar_state: Literal["auto", "expanded", "collapsed"] = cast(
        Literal["auto", "expanded", "collapsed"],
        config.app.initial_sidebar_state if config.app.initial_sidebar_state in {"auto", "expanded", "collapsed"} else "auto",
    )

    LOGGER.info("Setting page config")
    st.set_page_config(
        page_title=config.app.title,
        page_icon=config.app.page_icon,
        layout=layout,
        initial_sidebar_state=sidebar_state,
    )
    st.title(config.app.title)
    st.markdown(config.app.subtitle)

    LOGGER.info("Loading state and catalog")
    state = InferenceState.from_session()
    options = CatalogOptions.from_paths(config.paths)
    catalog = _load_catalog(options)
    inference_service = InferenceService()

    LOGGER.info("Rendering sidebar")
    request = sidebar_component.render_controls(state, config, catalog)
    LOGGER.info(f"Sidebar rendered, request type: {type(request).__name__ if request else None}")

    if request is not None:
        if isinstance(request, BatchPredictionRequest):
            LOGGER.info(">>> STARTING BATCH PREDICTION")
            # Handle batch prediction request
            inference_service.run_batch_prediction(state, request)
            LOGGER.info("<<< BATCH PREDICTION COMPLETED")
        elif isinstance(request, InferenceRequest):
            LOGGER.info(">>> STARTING SINGLE INFERENCE")
            # Handle single image inference request
            inference_service.run(state, request, state.hyperparams)
            LOGGER.info("<<< SINGLE INFERENCE COMPLETED")

    print(">>> CALLING render_results", file=sys.stderr, flush=True)
    LOGGER.info(">>> CALLING render_results")
    results_component.render_results(state, config)
    print("<<< render_results RETURNED", file=sys.stderr, flush=True)
    LOGGER.info("<<< render_results RETURNED")
    print("=" * 80, file=sys.stderr, flush=True)
    print("<<< APP.RUN() COMPLETED", file=sys.stderr, flush=True)
    print("=" * 80 + "\n", file=sys.stderr, flush=True)
    LOGGER.info("=" * 80)
    LOGGER.info("<<< APP.RUN() COMPLETED")
    LOGGER.info("=" * 80)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
