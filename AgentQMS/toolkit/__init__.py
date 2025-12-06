"""Agent automation tooling for artifact workflows and compliance.

⚠️  DEPRECATED: As of AgentQMS 0.3.2, the toolkit module is deprecated in favor of
AgentQMS.agent_tools, which provides the canonical implementation surface.

Migration Path:
- Version 0.3.2: Deprecation warnings (this module still works)
- Version 0.4.0: Removal (use AgentQMS.agent_tools instead)

For migration guidance, see:
  → docs/artifacts/design_documents/2025-12-06_design_toolkit-deprecation-roadmap.md
  → .copilot/context/migration-guide.md

Example migration:
  OLD: from AgentQMS.toolkit.core.artifact_templates import ArtifactTemplates
  NEW: from AgentQMS.agent_tools.core.artifact_templates import ArtifactTemplates

All functionality remains available until 0.4.0 release, but you will see deprecation
warnings when importing from this module. Please plan to migrate your code to use
AgentQMS.agent_tools instead.
"""

import warnings

# Emit deprecation warning when toolkit is imported
warnings.warn(
    "AgentQMS.toolkit is deprecated as of 0.3.2 and will be removed in 0.4.0. "
    "Use AgentQMS.agent_tools instead. "
    "See docs/artifacts/design_documents/2025-12-06_design_toolkit-deprecation-roadmap.md for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)
