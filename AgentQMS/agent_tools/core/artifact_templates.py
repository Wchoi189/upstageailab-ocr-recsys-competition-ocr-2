#!/usr/bin/env python3
"""Shim for artifact templates.

Delegates to `AgentQMS.toolkit.core.artifact_templates` while
exposing a stable `AgentQMS.agent_tools` import path.
"""

from __future__ import annotations

from AgentQMS.toolkit.core.artifact_templates import (
    ArtifactTemplates,
    create_artifact,
    get_available_templates,
    get_template,
)

__all__ = [
    "ArtifactTemplates",
    "create_artifact",
    "get_available_templates",
    "get_template",
]

