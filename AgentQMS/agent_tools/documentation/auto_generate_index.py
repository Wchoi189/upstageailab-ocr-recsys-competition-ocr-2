#!/usr/bin/env python3
"""Shim for handbook index generation/validation.

Delegates to `AgentQMS.toolkit.documentation.auto_generate_index` while
exposing a stable `AgentQMS.agent_tools` CLI path.
"""

from __future__ import annotations

from AgentQMS.toolkit.documentation.auto_generate_index import main


if __name__ == "__main__":  # pragma: no cover - thin CLI shim
    main()


