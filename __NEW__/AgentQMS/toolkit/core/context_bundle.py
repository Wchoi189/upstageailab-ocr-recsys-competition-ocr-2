#!/usr/bin/env python3
"""
Context Bundle Generator - Legacy Shim

This is a compatibility shim. The canonical implementation is in:
    AgentQMS.agent_tools.core.context_bundle

See AgentQMS/toolkit/README.md for migration guidance.
"""

# Re-export from canonical location
from AgentQMS.agent_tools.core.context_bundle import (
    BUNDLES_DIR,
    PROJECT_ROOT,
    TASK_KEYWORDS,
    analyze_task_type,
    expand_glob_pattern,
    get_context_bundle,
    is_fresh,
    list_available_bundles,
    load_bundle_definition,
    main,
    print_context_bundle,
    validate_bundle_files,
)

__all__ = [
    "BUNDLES_DIR",
    "PROJECT_ROOT",
    "TASK_KEYWORDS",
    "analyze_task_type",
    "expand_glob_pattern",
    "get_context_bundle",
    "is_fresh",
    "list_available_bundles",
    "load_bundle_definition",
    "main",
    "print_context_bundle",
    "validate_bundle_files",
]

if __name__ == "__main__":
    main()
