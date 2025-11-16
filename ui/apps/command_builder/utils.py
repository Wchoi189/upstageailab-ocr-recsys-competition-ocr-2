"""Utilities for Command Builder app.

This module provides cached service factories and other utility functions
to improve performance and maintain clean code organization.
"""

from __future__ import annotations

import streamlit as st

from ui.utils.command import CommandBuilder
from ui.utils.config_parser import ConfigParser
from ui.apps.command_builder.services.recommendations import UseCaseRecommendationService


def rerun_app() -> None:
    """Trigger a Streamlit rerun using the available API surface."""

    rerun_callable = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rerun_callable is None:
        raise RuntimeError("Streamlit rerun API is unavailable in this version.")

    rerun_callable()  # type: ignore[misc]


# ============================================================================
# Cached Service Factories
# ============================================================================


@st.cache_resource(show_spinner=False)
def get_command_builder() -> CommandBuilder:
    """Get cached CommandBuilder instance.

    Returns:
        Cached CommandBuilder instance.
    """
    return CommandBuilder()


@st.cache_resource(show_spinner=False)
def get_config_parser() -> ConfigParser:
    """Get cached ConfigParser instance.

    Returns:
        Cached ConfigParser instance.
    """
    return ConfigParser()


@st.cache_resource(show_spinner=False)
def get_recommendation_service() -> UseCaseRecommendationService:
    """Get cached UseCaseRecommendationService instance.

    Returns:
        Cached UseCaseRecommendationService instance.
    """
    return UseCaseRecommendationService(get_config_parser())


# ============================================================================
# Cached ConfigParser Methods
# ============================================================================


@st.cache_data(ttl=3600, show_spinner=False)
def get_architecture_metadata() -> dict:
    """Get cached architecture metadata.

    Returns:
        Dictionary of architecture metadata.
    """
    return get_config_parser().get_architecture_metadata()


@st.cache_data(ttl=3600, show_spinner=False)
def get_available_models() -> dict[str, list[str]]:
    """Get cached available models.

    Returns:
        Dictionary mapping model types to lists of available models.
    """
    return get_config_parser().get_available_models()


@st.cache_data(ttl=3600, show_spinner=False)
def get_available_architectures() -> list[str]:
    """Get cached available architectures.

    Returns:
        List of available architecture names.
    """
    return get_config_parser().get_available_architectures()


@st.cache_data(ttl=3600, show_spinner=False)
def get_optimizer_metadata() -> dict:
    """Get cached optimizer metadata.

    Returns:
        Dictionary of optimizer metadata.
    """
    return get_config_parser().get_optimizer_metadata()
