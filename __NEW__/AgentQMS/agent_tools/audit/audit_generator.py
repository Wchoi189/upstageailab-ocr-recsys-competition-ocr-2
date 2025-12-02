#!/usr/bin/env python3
"""Shim for the audit document generator.

Delegates to `AgentQMS.toolkit.audit.audit_generator` while exposing the
canonical `AgentQMS.agent_tools.audit` CLI path.
"""

from __future__ import annotations

from AgentQMS.toolkit.audit.audit_generator import *  # noqa: F401,F403


if __name__ == "__main__":  # pragma: no cover - thin CLI shim
    from AgentQMS.toolkit.audit.audit_generator import main

    main()


