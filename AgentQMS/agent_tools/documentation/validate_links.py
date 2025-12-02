#!/usr/bin/env python3
"""Shim for documentation link validation.

Delegates to `AgentQMS.toolkit.documentation.validate_links` while exposing
the canonical `AgentQMS.agent_tools` CLI path.
"""

from __future__ import annotations

from AgentQMS.toolkit.documentation.validate_links import main


if __name__ == "__main__":  # pragma: no cover - thin CLI shim
    main()


