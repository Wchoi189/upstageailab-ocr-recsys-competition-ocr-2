"""Path resolution helpers for AgentQMS.

Re-exports from system module for backward compatibility.
"""

from .system.paths import (
    ensure_within_project,
    get_agent_interface_dir,
    get_agent_tools_dir,
    get_artifacts_dir,
    get_container_path,
    get_docs_dir,
    get_framework_root,
    get_project_config_dir,
    get_project_conventions_dir,
    get_project_root,
)