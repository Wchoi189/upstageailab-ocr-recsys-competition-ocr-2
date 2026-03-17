m2m_session_handover_v1:
  session:
    session_id: "2026-03-17_0000"
    branch: "003-ocr-data-quality-remediation"
    start_time_utc: "2026-03-17T00:00:00Z"
    end_time_utc: "2026-03-17T00:00:00Z"

  session_summary:
    objective: "Containerize spec-kit planning artifacts under specs/ and establish multi-session handover protocol."
    progress:
      - spec: "Planning/Scaffolding"
        changes:
          - files:
              - "specs/global-agentqms/CONSTITUTION.md"
              - "specs/global-agentqms/ORDER_OF_OPERATIONS.md"
              - "specs/global-agentqms/runs/2026-03-17_0000/spec_index.md"
              - "specs/global-agentqms/runs/2026-03-17_0000/backlog.md"
              - "specs/global-agentqms/runs/2026-03-17_0000/specs/spec_A_dynamic_project_resolution.md"
              - "specs/global-agentqms/runs/2026-03-17_0000/specs/spec_B_cli_entry_point.md"
              - "specs/global-agentqms/runs/2026-03-17_0000/specs/spec_C_init_scaffolding.md"
            evidence:
              - "filesystem_write_check: create+read+delete test file succeeded"
    decisions:
      - "Session handovers policy: Option A (multiple handovers with timestamped filenames) under specs/global-agentqms/session_handovers/."
      - "All artifacts containerized under specs/global-agentqms/ with timestamped run folders under runs/."
    risks_known_issues:
      - "Implementation work (Spec A/B/C) not started yet; only planning artifacts scaffolded."

  artifacts:
    active_run_folder: "specs/global-agentqms/runs/2026-03-17_0000/"
    spec_index: "specs/global-agentqms/runs/2026-03-17_0000/spec_index.md"
    backlog: "specs/global-agentqms/runs/2026-03-17_0000/backlog.md"

  backlog_state:
    in_progress: []
    next_up:
      - id: "A1"
        task: "Identify current project_root detection implementation and call sites (ConfigLoader + server startup)."

  next_session_entry_point:
    instruction: "Start Spec A by locating the current project root detection logic and its callers; update A1 with file paths/line refs and proposed patch points."
    command: "Search for ConfigLoader._detect_project_root and project_root usage in AgentQMS + unified server startup paths."
    expected_observable: "A1 updated with exact file paths and the minimal set of functions/modules that must change for dynamic root resolution."

