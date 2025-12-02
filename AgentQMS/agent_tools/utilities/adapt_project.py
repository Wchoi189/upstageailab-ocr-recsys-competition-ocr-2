#!/usr/bin/env python3
"""Canonical wrapper for the project adaptation script.

The underlying implementation currently lives in
`AgentQMS.toolkit.utilities.adapt_project`. This module exposes the same
behavior via the `AgentQMS.agent_tools` namespace for agents and docs.
"""

from __future__ import annotations

from AgentQMS.toolkit.utilities.adapt_project import *  # noqa: F401,F403


if __name__ == "__main__":  # pragma: no cover - thin CLI shim
    from AgentQMS.toolkit.utilities.adapt_project import main

    main()


