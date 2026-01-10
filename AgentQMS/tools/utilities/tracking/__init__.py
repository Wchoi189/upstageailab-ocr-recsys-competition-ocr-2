"""
Tracking Subsystem for AgentQMS

Components:
- `db.py`: persistence layer (SQLite) with CRUD helpers
- `cli.py`: command-line interface for tracking operations

Purpose:
- Provide consistent tracking of artifacts (plans, experiments, sessions) across tools.
- Serve as the single source of truth for tracking data and avoid duplication elsewhere.
"""
