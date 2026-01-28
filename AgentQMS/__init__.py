"""AgentQMS – Quality Management Framework for AI Coding Agents.

This package provides:
- `agent_tools` – Canonical implementation layer (validation, artifact workflows, documentation tools)
- `toolkit` – Legacy compatibility shim (delegates to agent_tools)
- `conventions` – Artifact types, schemas, templates, audit framework
- `knowledge` – Protocols, references, and agent-facing instructions

Usage:
    from AgentQMS.tools.core.artifact_workflow import ArtifactWorkflow
    from AgentQMS.tools.compliance.validate_artifacts import ArtifactValidator
    from AgentQMS.tools.utils.config import load_config
"""

__version__ = "2.0.0"
