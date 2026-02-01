"""AgentQMS – Quality Management Framework for AI Coding Agents.

This package provides:
- `tools` – Canonical implementation layer (core artifacts, context bundling, compliance)
- `standards` – Project standards, conventions, and registry
- `bin` – Command-line interface utilities (aqms, adt)
- `middleware` – MCP server integration and protocol bridging

Usage:
    from AgentQMS.tools.core.artifacts.workflow import ArtifactWorkflow
    from AgentQMS.tools.compliance.validate_artifacts import ArtifactValidator
    from AgentQMS.tools.utils.config.loader import load_config
"""

__version__ = "2.0.0"
