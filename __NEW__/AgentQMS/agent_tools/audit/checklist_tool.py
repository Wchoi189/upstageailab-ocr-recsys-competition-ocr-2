#!/usr/bin/env python3
"""Shim for the audit checklist tool.

Delegates to `AgentQMS.toolkit.audit.checklist_tool` while exposing the
canonical `AgentQMS.agent_tools.audit` CLI path.
"""

from __future__ import annotations

from AgentQMS.toolkit.audit.checklist_tool import *  # noqa: F401,F403


if __name__ == "__main__":  # pragma: no cover - thin CLI shim
    from AgentQMS.toolkit.audit.checklist_tool import main

    main()


