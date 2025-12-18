from __future__ import annotations

# Re-export from canonical location
from AgentQMS.agent_tools.utilities.get_context import (
    list_bundles,
    load_index,
    lookup_entry,
    main,
    parse_args,
    print_bundle,
)

"""Lookup utility for documentation context bundles - Legacy Shim

This is a compatibility shim. The canonical implementation is in:
    AgentQMS.agent_tools.utilities.get_context

See AgentQMS/toolkit/README.md for migration guidance.
"""

__all__ = [
    "list_bundles",
    "load_index",
    "lookup_entry",
    "main",
    "parse_args",
    "print_bundle",
]

if __name__ == "__main__":
    import sys

    sys.exit(main())
