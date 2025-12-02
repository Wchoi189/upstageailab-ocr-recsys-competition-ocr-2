#!/usr/bin/env python3
"""Canonical CLI entrypoint for development/debug tracking and experiments.

Implementation currently lives in `AgentQMS.toolkit.utilities.tracking.cli`;
this module provides the stable `AgentQMS.agent_tools` path for agents.
"""

from __future__ import annotations

from AgentQMS.toolkit.utilities.tracking.cli import main


if __name__ == "__main__":  # pragma: no cover - thin CLI shim
    main()


